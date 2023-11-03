#ifndef KERNELS
#define KERNELS

#include <cuda_runtime.h>
#include <thrust/tuple.h>

#include "utils.cu"

// ------ TYPES AND OPERATORS ------- //

/**
 * Generic Add operator that can be instantiated over
 *  numeric-basic types, such as int32_t, int64_t,
 *  float, double, etc.
 */
template<class T>
class Add {
  public:
    typedef T ElTp;
    static __device__ __host__ inline T identInp()                    { return (T)0;    }
    static __device__ __host__ inline T mapFun(const T& el)           { return el;      }
    static __device__ __host__ inline T identity()                    { return (T)0;    }
    static __device__ __host__ inline T apply(const T t1, const T t2) { return t1 + t2; }

    static __device__ __host__ inline bool equals(const T t1, const T t2) { return (t1 == t2); }
    static __device__ __host__ inline T remVolatile(volatile T& t)    { T res = t; return res; }
};

// -------------------------------------//

//---------- UTILITY KERNEL ----------//
// The kernel naiveMemcpy, copyFromGlb2ShrMem, copyFromShr2GlbMem, scanIncWarp,
// scanIncBlock are taken from the weekly assignment 2

// naive memcpy kernel for benchmark 
__global__ void naiveMemcpy(int* d_out, int* d_inp, const uint32_t N) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid < N) {
        d_out[gid] = d_inp[gid];
    }
}

// coalesced mem copy glb to shr of CHUNK*B elems
template<class T, uint32_t CHUNK>
__device__ inline void
copyFromGlb2ShrMem( const uint32_t glb_offs
                  , const uint32_t N
                  , const T& ne
                  , T* d_inp
                  , volatile T* shmem_inp) {
    #pragma unroll
    for(uint32_t i=0; i<CHUNK; i++) {

        uint32_t loc_ind = i*blockDim.x + threadIdx.x;
        uint32_t glb_ind = glb_offs + loc_ind;
        T elm = ne;
        if(glb_ind < N) { elm = d_inp[glb_ind]; }
        shmem_inp[loc_ind] = elm;
    }
    __syncthreads(); 
}

// coalesced mem copy shr to glb of CHUNK*B elems
template<class T, uint32_t CHUNK>
__device__ inline void
copyFromShr2GlbMem( const uint32_t glb_offs
                  , const uint32_t N
                  , T* d_out
                  , volatile T* shmem_red) {
    #pragma unroll
    for (uint32_t i = 0; i < CHUNK; i++) {

        uint32_t loc_ind = i*blockDim.x + threadIdx.x;
        uint32_t glb_ind = glb_offs + loc_ind;
        if (glb_ind < N) {
            T elm = const_cast<const T&>(shmem_red[loc_ind]);
            d_out[glb_ind] = elm;
        }
    }
    __syncthreads(); 
}

// Scan in warp, used at block-level
template<class OP>
__device__ inline typename OP::ElTp
scanIncWarp( volatile typename OP::ElTp* ptr, const unsigned int idx ) {
    const unsigned int lane = idx & (WARP-1);
    
    #pragma unroll
    for (int d=0; d<lgWARP; ++d) {
        int h = 1 << d;
        if (lane>=h) ptr[idx] = OP::apply(ptr[idx-h], ptr[idx]);
    }
    
    return OP::remVolatile(ptr[idx]);
}

// Scan at block-level
template<class OP>
__device__ inline typename OP::ElTp
scanIncBlock(volatile typename OP::ElTp* ptr, const unsigned int idx) {
    const unsigned int lane   = idx & (WARP-1); // index of thread in warp (0..31)
    const unsigned int warpid = idx >> lgWARP; // warp index in block

    // 1. perform scan at warp level
    typename OP::ElTp res = scanIncWarp<OP>(ptr,idx);
    __syncthreads(); 

    // 2. place the end-of-warp results in
    //   the first warp. This works because
    //   warp size = 32, and 
    //   max block size = 32^2 = 1024
    typename OP::ElTp tmp = OP::remVolatile(ptr[idx]);
    __syncthreads();
    if (lane == (WARP-1)) {
        ptr[warpid] = tmp;
    }
    __syncthreads();

    // 3. scan again the first warp
    if (warpid == 0) scanIncWarp<OP>(ptr, idx);
    __syncthreads();

    // 4. accumulate results from previous step;
    if (warpid > 0) {
        res = OP::apply(ptr[warpid-1], res);
    }
    __syncthreads();

    ptr[idx] = res;
    return res;
}

// ---------------------------------------------- //

//----------- SINGLE PASS SCAN KERNEL----------//
// From the weekly assignment 2 kernel scan3rdKernel, modify it to do the scan
// in a single kernel.
/**
 * `N` is the length of the input array
 * `CHUNK` (the template parameter) is the number of elements to
 *    be processed sequentially by a thread in one go.
 * `d_out` is the result array of length `N`
 * `d_in`  is the input  array of length `N`
 */

template<class OP, uint8_t CHUNK>
__global__ void
spScanKernelDepr ( typename OP::ElTp* d_out
                 , typename OP::ElTp* d_in
                 , volatile typename OP::ElTp* aggregates
                 , volatile typename OP::ElTp* prefixes
                 , volatile uint8_t* flags
                 , volatile uint32_t* dyn_block_id
                 , uint32_t N
) {
    typedef typename OP::ElTp ElTp;

    extern __shared__ ElTp shared_mem[];
    ElTp* shmem_inp = (ElTp*)shared_mem; // CHUNK * BLOCK
    ElTp* shmem_red = (ElTp*)shared_mem; // BLOCK

    __shared__ uint32_t tmp_block_id; // <- is volatile needed here?
    if (threadIdx.x == 0) {
        tmp_block_id = atomicAdd((uint32_t*)dyn_block_id, 1);
    }
    __syncthreads();

    uint32_t thread_id = threadIdx.x;
    uint32_t block_id = tmp_block_id;
    uint32_t block_offset = CHUNK * blockDim.x  * block_id;

    // register memory for storing the scanned elements.
    ElTp chunk[CHUNK];

    // // 1. copy `CHUNK` input elements per thread from global to shared memory
    copyFromGlb2ShrMem<ElTp, CHUNK>(block_offset, N, OP::identInp(), d_in, shmem_inp);

    // // 2. each thread sequentially reduces its `CHUNK` elements, result is stored in `tmp`
    ElTp tmp = OP::identity();
    uint32_t shmem_offset = thread_id * CHUNK;
    #pragma unroll
    for (uint8_t i = 0; i < CHUNK; i++) {
        ElTp elm = shmem_inp[shmem_offset + i];
        tmp = OP::apply(tmp, elm);
        chunk[i] = tmp;
    }
    __syncthreads();

    // 3. Each thread publishes in shared memory the reduced result of its `CHUNK` elements 
    shmem_red[threadIdx.x] = tmp;
    __syncthreads();

    // 4. perform an intra-CUDA-block scan 
    ElTp agg = scanIncBlock<OP>(shmem_red, thread_id);
    __syncthreads();
    if (thread_id == blockDim.x -1) {
        if (block_id > 0) {
            aggregates[block_id] = agg;
            __threadfence(); // <- which thread fence to use?
            flags[block_id] = AGG;
        } else {
            prefixes[block_id] = agg;
            __threadfence(); // <- which thread fence to use?
            flags[block_id] = PRE;
        }
    }
    __syncthreads();

    if (thread_id == blockDim.x - 1) {
        if (block_id > 0) {
            // printf("#TID: %d, #DBID: %d, #SBID %d \n", thread_id, block_id, blockIdx.x);
            while (flags[block_id-1] != PRE) {}
            const int32_t prev_prefix = prefixes[block_id-1];

            prefixes[block_id] = agg + prev_prefix;
            __threadfence();
            flags[block_id] = PRE;

            int32_t acc = prev_prefix;
            for (uint32_t i = block_id * blockDim.x * CHUNK; i < block_id * blockDim.x * CHUNK + blockDim.x * CHUNK; i++) {
                d_out[i] = acc + d_in[i];
                acc = d_out[i];
            }
        } else {
            int32_t acc = 0;
            for (uint32_t i = block_id * blockDim.x * CHUNK; i < block_id * blockDim.x * CHUNK + blockDim.x * CHUNK; i++) {
                d_out[i] = acc + d_in[i];
                acc = d_out[i];
            }
        }
    }
}

// ---------------------------------------------- //

//----------- SINGLE PASS SCAN KERNEL----------//
// From the weekly assignment 2 kernel scan3rdKernel, modify it to do the scan
// in a single kernel.
/**
 * `N` is the length of the input array
 * `CHUNK` (the template parameter) is the number of elements to
 *    be processed sequentially by a thread in one go.
 * `d_out` is the result array of length `N`
 * `d_in`  is the input  array of length `N`
 */

template<class OP, uint8_t CHUNK>
__global__ void
spScanKernel ( typename OP::ElTp* d_out
             , typename OP::ElTp* d_in
             , volatile typename OP::ElTp* aggregates
             , volatile typename OP::ElTp* prefixes
             , volatile uint8_t* flags // <- initialize all elements with INC flag
             , volatile uint32_t* dyn_block_id
             , uint32_t N
) {
    typedef typename OP::ElTp ElTp;

    extern __shared__ ElTp shared_mem[]; // CHUNK * BLOCK

    __shared__ uint32_t tmp_block_id;
    if (threadIdx.x == 0) {
        tmp_block_id = atomicAdd((uint32_t*)dyn_block_id, 1);
    }
    __syncthreads();

    uint32_t thread_id = threadIdx.x;
    uint32_t block_id = tmp_block_id;
    uint32_t block_offset = CHUNK * blockDim.x  * block_id;

    // register memory for storing the scanned elements.
    ElTp chunk[CHUNK];

    // // 1. copy `CHUNK` input elements per thread from global to shared memory
    copyFromGlb2ShrMem<ElTp, CHUNK>(block_offset, N, OP::identInp(), d_in, shared_mem);

    // // 2. each thread sequentially reduces its `CHUNK` elements, result is stored in `tmp`
    ElTp tmp = OP::identity();
    uint32_t shared_mem_offset = thread_id * CHUNK;
    #pragma unroll
    for (uint32_t i = 0; i < CHUNK; i++) {
        ElTp elm = shared_mem[shared_mem_offset + i];
        tmp = OP::apply(tmp, elm);
        chunk[i] = tmp;
    }
    __syncthreads();

    // 3. Each thread publishes in shared memory the reduced result of its `CHUNK` elements 
    shared_mem[thread_id] = tmp;
    __syncthreads();

    // 4. perform an intra-CUDA-block scan 
    ElTp agg = scanIncBlock<OP>(shared_mem, thread_id);
    __syncthreads();
    if (thread_id == blockDim.x -1) {
        if (block_id > 0) {
            aggregates[block_id] = agg;
            __threadfence();
            flags[block_id] = AGG;
        } else {
            prefixes[block_id] = agg;
            __threadfence();
            flags[block_id] = PRE;
        }
    }
    __syncthreads();

    ElTp prev_chunk_prefix = OP::identity();
    if (thread_id > 0) {
        prev_chunk_prefix = shared_mem[thread_id-1];
    }
    __syncthreads();

    __shared__ ElTp prev_block_prefix;
    if (thread_id == blockDim.x - 1) {
        if (block_id > 0) {
            while (flags[block_id-1] != PRE) {}
            prev_block_prefix = prefixes[block_id-1];
            prefixes[block_id] = OP::apply(prev_block_prefix, agg);
            __threadfence();
            flags[block_id] = PRE;
        } else {
            prev_block_prefix = OP::identity();
        }
    }
    __syncthreads();

    ElTp prev_total_prefix = OP::apply(prev_block_prefix, prev_chunk_prefix);
    for (uint32_t i = 0; i < CHUNK; i++) {
        shared_mem[shared_mem_offset + i] = OP::apply(prev_total_prefix, chunk[i]);
    }
    __syncthreads();

    // 5. write back from shared to global memory in coalesced fashion.
    copyFromShr2GlbMem<ElTp, CHUNK>(block_offset, N, d_out, shared_mem);
}

template<class OP, uint8_t CHUNK>
__global__ void
spLookbackScanKernel ( typename OP::ElTp* d_out
                     , typename OP::ElTp* d_in
                     , volatile typename OP::ElTp* aggregates
                     , volatile typename OP::ElTp* prefixes
                     , volatile uint8_t* flags // <- initialize all elements with INC flag
                     , volatile uint32_t* dyn_block_id
                     , uint32_t N
) {
    typedef typename OP::ElTp ElTp;

    extern __shared__ ElTp shared_mem[]; // CHUNK * BLOCK

    __shared__ uint32_t tmp_block_id;
    if (threadIdx.x == 0) {
        tmp_block_id = atomicAdd((uint32_t*)dyn_block_id, 1);
    }
    __syncthreads();

    uint32_t thread_id = threadIdx.x;
    uint32_t block_id = tmp_block_id;
    uint32_t block_offset = CHUNK * blockDim.x  * block_id;

    // register memory for storing the scanned elements.
    ElTp chunk[CHUNK];

    // // 1. copy `CHUNK` input elements per thread from global to shared memory
    copyFromGlb2ShrMem<ElTp, CHUNK>(block_offset, N, OP::identInp(), d_in, shared_mem);

    // // 2. each thread sequentially reduces its `CHUNK` elements, result is stored in `tmp`
    ElTp tmp = OP::identity();
    uint32_t shared_mem_offset = thread_id * CHUNK;
    #pragma unroll
    for (uint32_t i = 0; i < CHUNK; i++) {
        ElTp elm = shared_mem[shared_mem_offset + i];
        tmp = OP::apply(tmp, elm);
        chunk[i] = tmp;
    }
    __syncthreads();

    // 3. Each thread publishes in shared memory the reduced result of its `CHUNK` elements
    shared_mem[thread_id] = tmp;
    __syncthreads();

    // 4. perform an intra-CUDA-block scan
    ElTp agg = scanIncBlock<OP>(shared_mem, thread_id);
    __syncthreads();
    if (thread_id == blockDim.x -1) {
        if (block_id > 0) {
            aggregates[block_id] = agg;
            __threadfence();
            flags[block_id] = AGG;
        } else {
            prefixes[block_id] = agg;
            __threadfence();
            flags[block_id] = PRE;
        }
    }
    __syncthreads();

    ElTp prev_chunk_prefix = OP::identity();
    if (thread_id > 0) {
        prev_chunk_prefix = shared_mem[thread_id-1];
    }
    __syncthreads();

    __shared__ ElTp prev_block_prefix;
    if (thread_id == 0) {
        prev_block_prefix = OP::identity();
        if (block_id > 0) {
            uint32_t prev_block_id = block_id-1;
            while (true) {
                uint8_t flag = flags[prev_block_id];
                if (flag == PRE) {
                    prev_block_prefix = OP::apply(prefixes[prev_block_id], prev_block_prefix);
                    break;
                } else if (flag == AGG) {
                    prev_block_prefix = OP::apply(aggregates[prev_block_id], prev_block_prefix);
                    prev_block_id--;
                }
            }
        }
    }
    __syncthreads();

    if (thread_id == blockDim.x - 1) {
        prefixes[block_id] = OP::apply(prev_block_prefix, agg);
        __threadfence();
        flags[block_id] = PRE;
    }

    ElTp prev_total_prefix = OP::apply(prev_block_prefix, prev_chunk_prefix);
    for (uint32_t i = 0; i < CHUNK; i++) {
        shared_mem[shared_mem_offset + i] = OP::apply(prev_total_prefix, chunk[i]);
    }
    __syncthreads();

    // 5. write back from shared to global memory in coalesced fashion.
    copyFromShr2GlbMem<ElTp, CHUNK>(block_offset, N, d_out, shared_mem);
}

template<class OP, uint8_t CHUNK>
__global__ void
spWarpLookbackScanKernel ( typename OP::ElTp* d_out
                         , typename OP::ElTp* d_in
                         , volatile typename OP::ElTp* aggregates
                         , volatile typename OP::ElTp* prefixes
                         , volatile uint8_t* flags
                         , volatile uint32_t* dyn_block_id
                         , uint32_t N
) {
    typedef typename OP::ElTp ElTp;

    extern __shared__ ElTp shared_mem[]; // CHUNK * BLOCK

    __shared__ uint32_t tmp_block_id;
    if (threadIdx.x == 0) {
        tmp_block_id = atomicAdd((uint32_t*)dyn_block_id, 1);
    }
    __syncthreads();

    uint32_t thread_id = threadIdx.x;
    uint32_t block_id = tmp_block_id;
    uint32_t block_offset = CHUNK * blockDim.x  * block_id;

    // register memory for storing the scanned elements.
    ElTp chunk[CHUNK];

    // // 1. copy `CHUNK` input elements per thread from global to shared memory
    copyFromGlb2ShrMem<ElTp, CHUNK>(block_offset, N, OP::identInp(), d_in, shared_mem);

    // // 2. each thread sequentially reduces its `CHUNK` elements, result is stored in `tmp`
    ElTp tmp = OP::identity();
    uint32_t shared_mem_offset = thread_id * CHUNK;
    #pragma unroll
    for (uint32_t i = 0; i < CHUNK; i++) {
        ElTp elm = shared_mem[shared_mem_offset + i];
        tmp = OP::apply(tmp, elm);
        chunk[i] = tmp;
    }
    __syncthreads();

    // 3. Each thread publishes in shared memory the reduced result of its `CHUNK` elements
    shared_mem[thread_id] = tmp;
    __syncthreads();

    // 4. perform an intra-CUDA-block scan
    ElTp agg = scanIncBlock<OP>(shared_mem, thread_id);
    __syncthreads();
    if (thread_id == blockDim.x -1) {
        if (block_id > 0) {
            aggregates[block_id] = agg;
            __threadfence();
            flags[block_id] = AGG;
        } else {
            prefixes[block_id] = agg;
            __threadfence();
            flags[block_id] = PRE;
        }
    }
    __syncthreads();

    ElTp prev_chunk_prefix = OP::identity();
    if (thread_id > 0) {
        prev_chunk_prefix = shared_mem[thread_id-1];
    }
    __syncthreads();

    __shared__ ElTp prev_block_prefix;
    if (thread_id < 32) {
        if (thread_id == 0) {
            prev_block_prefix = OP::identity();
        }

        if (block_id > 0) {
            __shared__ uint8_t flag_buffer[WARP];
            __shared__ ElTp elem_buffer[WARP];
            __shared__ uint8_t status; // 0 = redo, 1 = continue, 2 = done
            int32_t prev_block_base_id;

            if (thread_id == 0) {
                prev_block_base_id = block_id - WARP;
            }

            while (true) {
                int32_t prev_block_id = prev_block_base_id + thread_id;

                uint8_t flag = flags[prev_block_id];
                if (flag == PRE) {
                    elem_buffer[thread_id] = prefixes[prev_block_id];
                } else if (flag == AGG) {
                    elem_buffer[thread_id] = aggregates[prev_block_id];
                }

                flag_buffer[thread_id] = flag;

                ElTp loop_prefix;
                if (thread_id == 0) {
                    loop_prefix = OP::identity();

                    for (int32_t i = WARP-1; i >= 0; i--) {
                        uint8_t flag = flag_buffer[i];
                        if (flag == PRE) {
                            loop_prefix = OP::apply(elem_buffer[i], loop_prefix);
                            status = 2; // stop WARP loop
                            break;
                        } else if (flag == AGG) {
                            loop_prefix = OP::apply(elem_buffer[i], loop_prefix);
                            status = 1; // continue WARP loop
                        } else {
                            status = 0; // redo current loop
                            break;
                        }
                    }
                }
                __syncwarp(); // sync so all threads would see the updated status

                if (status == 1) { // continue WARP loop
                    if (thread_id == 0) {
                        prev_block_prefix = OP::apply(loop_prefix, prev_block_prefix);
                    }
                    prev_block_base_id -= WARP;
                } else if (status == 2) { // stop WARP loop
                    if (thread_id == 0) {
                        prev_block_prefix = OP::apply(loop_prefix, prev_block_prefix);
                    }
                    break;
                }
            }
        }
    }
    __syncthreads();

    if (thread_id == blockDim.x-1) {
        prefixes[block_id] = OP::apply(prev_block_prefix, agg);
        __threadfence();
        flags[block_id] = PRE;
    }

    ElTp prev_total_prefix = OP::apply(prev_block_prefix, prev_chunk_prefix);
    for (uint32_t i = 0; i < CHUNK; i++) {
        shared_mem[shared_mem_offset + i] = OP::apply(prev_total_prefix, chunk[i]);
    }
    __syncthreads();

    // 5. write back from shared to global memory in coalesced fashion.
    copyFromShr2GlbMem<ElTp, CHUNK>(block_offset, N, d_out, shared_mem);
}

template<class OP, uint8_t CHUNK>
__global__ void
spWarpLookbackScanKernelOpt ( typename OP::ElTp* d_out
                            , typename OP::ElTp* d_in
                            , volatile typename OP::ElTp* aggregates
                            , volatile typename OP::ElTp* prefixes
                            , volatile uint8_t* flags
                            , volatile uint32_t* dyn_block_id
                            , uint32_t N
) {
    typedef typename OP::ElTp ElTp;
    typedef typename thrust::pair<uint8_t, ElTp> FV;

    extern __shared__ ElTp shared_mem[]; // CHUNK * BLOCK

    __shared__ uint32_t tmp_block_id;
    if (threadIdx.x == 0) {
        tmp_block_id = atomicAdd((uint32_t*)dyn_block_id, 1);
    }
    __syncthreads();

    uint32_t thread_id = threadIdx.x;
    uint32_t block_id = tmp_block_id;
    uint32_t block_offset = CHUNK * blockDim.x  * block_id;

    // register memory for storing the scanned elements.
    ElTp chunk[CHUNK];

    // // 1. copy `CHUNK` input elements per thread from global to shared memory
    copyFromGlb2ShrMem<ElTp, CHUNK>(block_offset, N, OP::identInp(), d_in, shared_mem);

    // // 2. each thread sequentially reduces its `CHUNK` elements, result is stored in `tmp`
    ElTp tmp = OP::identity();
    uint32_t shared_mem_offset = thread_id * CHUNK;
    #pragma unroll
    for (uint32_t i = 0; i < CHUNK; i++) {
        ElTp elm = shared_mem[shared_mem_offset + i];
        tmp = OP::apply(tmp, elm);
        chunk[i] = tmp;
    }
    __syncthreads();

    // 3. Each thread publishes in shared memory the reduced result of its `CHUNK` elements
    shared_mem[thread_id] = tmp;
    __syncthreads();

    // 4. perform an intra-CUDA-block scan
    ElTp agg = scanIncBlock<OP>(shared_mem, thread_id);
    __syncthreads();
    if (thread_id == blockDim.x -1) {
        if (block_id > 0) {
            aggregates[block_id] = agg;
            __threadfence();
            flags[block_id] = AGG;
        } else {
            prefixes[block_id] = agg;
            __threadfence();
            flags[block_id] = PRE;
        }
    }
    __syncthreads();

    ElTp prev_chunk_prefix = OP::identity();
    if (thread_id > 0) {
        prev_chunk_prefix = shared_mem[thread_id-1];
    }
    __syncthreads();

    __shared__ ElTp prev_block_prefix;
    if (thread_id < 32) {
        if (thread_id == 0) {
            prev_block_prefix = OP::identity();
        }

        if (block_id > 0) {
            FV *flagged_values = (FV*) shared_mem;
            __shared__ uint8_t continue_lookback;
            __shared__ int32_t prev_block_base_id;

            if (thread_id == 0) {
                prev_block_base_id = block_id - WARP;
                continue_lookback = 1;
            }

            while (true) {
                int32_t prev_block_id = prev_block_base_id + thread_id;

                uint8_t flag = flags[prev_block_id];
                ElTp value;
                if (flag == PRE) {
                    value = prefixes[prev_block_id];
                } else if (flag == AGG) {
                    value = aggregates[prev_block_id];
                }
                flagged_values[thread_id] = thrust::make_pair(flag, value);

                if (thread_id == 0) {
                    int32_t i = WARP-1;
                    while (i >= 0) {
                        FV fv = flagged_values[i];
                        uint8_t flag = thrust::get<0>(fv);
                        ElTp value = thrust::get<1>(fv);

                        if (flag == PRE) { // found the first prefix, time to stop both the inner and the main loops
                            prev_block_prefix = OP::apply(value, prev_block_prefix);
                            continue_lookback = 0;
                            break;
                        } else if (flag == AGG) { // found another agg, continue the inner loop
                            prev_block_prefix = OP::apply(value, prev_block_prefix);
                        } else { // it's empty, stop the inner loop and continue the main loop
                            break;
                        }
                        i--;
                    }
                    i++; // add 1 back (to compensate for "i = WARP-1")
                    prev_block_base_id -= WARP - i;
                }
                __syncwarp(); // sync so all threads would see the updated prev_block_base_id and continue_lookback

                if (continue_lookback == 0) { // stop the main loop
                    break;
                }
            }
        }
    }
    __syncthreads();

    if (thread_id == blockDim.x-1) {
        prefixes[block_id] = OP::apply(prev_block_prefix, agg);
        __threadfence();
        flags[block_id] = PRE;
    }

    ElTp prev_total_prefix = OP::apply(prev_block_prefix, prev_chunk_prefix);
    for (uint32_t i = 0; i < CHUNK; i++) {
        shared_mem[shared_mem_offset + i] = OP::apply(prev_total_prefix, chunk[i]);
    }
    __syncthreads();

    // 5. write back from shared to global memory in coalesced fashion.
    copyFromShr2GlbMem<ElTp, CHUNK>(block_offset, N, d_out, shared_mem);
}
#endif // KERNELS
