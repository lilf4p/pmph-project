#ifndef KERNELS
#define KERNELS

#include <cuda_runtime.h>

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
    // Use only one type
    //typedef T InpElTp;
    //typedef T RedElTp;
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
spScanKernel ( typename OP::ElTp* d_out
             , typename OP::ElTp* d_in
             , volatile typename OP::ElTp* aggregates
             , volatile typename OP::ElTp* prefixes
             , volatile uint8_t* flags // <- initialize all elements with INC flag
             , volatile uint32_t* dyn_block_id
             , size_t N
             , const uint32_t BLOCK
) {
    typedef typename OP::ElTp ElTp;

    __shared__ ElTp shmem_inp[CHUNK * BLOCK];
    __shared__ ElTp shmem_red[BLOCK];

    __shared__ uint32_t tmp_block_id; // <- is volatile needed here?
    if (threadIdx.x == BLOCK - 1) {
        tmp_block_id = atomicAdd((uint32_t*)dyn_block_id, 1);
    }
    __syncthreads();

    uint32_t thread_id = threadIdx.x;
    uint32_t block_id = tmp_block_id;
    uint32_t block_offset = CHUNK * BLOCK * block_id;

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
    if (thread_id == BLOCK-1) {
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

    // 5. write back from shared to global memory in coalesced fashion.
    copyFromShr2GlbMem<ElTp, CHUNK>(block_offset, N, d_out, shmem_inp);
}

#endif // KERNELS
