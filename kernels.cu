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

// coalesced mem copy glb to shr
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

// coalesced mem copy shr to glb 
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

//-----------SCAN KERNEL----------//
// From the weekly assignment 2 kernel scan3rdKernel, modify it to do the scan
// in a single kernel.

/**
 * This kernel assumes that the generic-associative binary operator
 *   `OP` is NOT-necessarily commutative. It implements the third
 *   stage of the scan (parallel prefix sum), which scans within
 *   a block.  (The first stage is a per block reduction with the
 *   `redAssocKernel` kernel, and the second one is the `scan1Block`
 *    kernel that scans the reduced elements of each CUDA block.)
 *
 * `N` is the length of the input array
 * `CHUNK` (the template parameter) is the number of elements to
 *    be processed sequentially by a thread in one go.
 * `d_out` is the result array of length `N`
 * `d_in`  is the input  array of length `N`
 * `d_tmp` is the array holding the per-block scanned results.
 *         it has number-of-CUDA-blocks elements, i.e., element
 *         `d_tmp[i-1]` is the scanned prefix that needs to be
 *         accumulated to each of the scanned elements corresponding
 *         to block `i`.
 * This kernels scans the elements corresponding to the current block
 *   `i`---in number of num_seq_chunks*CHUNK*blockDim.x---and then it
 *   accumulates to each of them the prefix of the previous block `i-1`,
 *   which is stored in `d_tmp[i-1]`.
 */

__device__ uint32_t dyn_block_id = 0; // dyn block id

template<class OP, int CHUNK>
__global__ void
scan3rdKernel ( typename OP::ElTp* d_out
              , typename OP::ElTp* d_in
              , typename OP::ElTp* aggregs
              , typename OP::ElTp* prefs
              , char* flags
              , uint32_t N 
) {
    extern __shared__ char sh_mem[];
    // shared memory for the input elements (types)
    volatile typename OP::ElTp* shmem_inp = (typename OP::ElTp*)sh_mem;

    // shared memory for the reduce-element type; it overlaps with the
    //   `shmem_inp` since they are not going to be used in the same time.
    volatile typename OP::ElTp* shmem_red = (typename OP::ElTp*)sh_mem;

    // number of elements to be processed by each block
    uint32_t num_elems_per_block = CHUNK * blockDim.x;

    // the current block start processing input elements from this offset:
    uint32_t inp_block_offs = num_elems_per_block * blockIdx.x;

    // accumulator updated at each iteration of the "virtualization"
    //   loop so we remember the prefix for the current elements.

    // TODO: change this (d_tmp is undefined)
    // typename OP::RedElTp accum = (blockIdx.x == 0) ? OP::identity() : d_tmp[blockIdx.x-1];
    typename OP::ElTp accum = (blockIdx.x == 0) ? OP::identity() : 0;

    // register memory for storing the scanned elements.
    typename OP::ElTp chunk[CHUNK];

    // 1. copy `CHUNK` input elements per thread from global to shared memory
    //    in coalesced fashion (for global memory)
    copyFromGlb2ShrMem<typename OP::ElTp, CHUNK>
            (inp_block_offs, N, OP::identInp(), d_in, shmem_inp);

    // 2. each thread sequentially scans its `CHUNK` elements
    //    and stores the result in the `chunk` array. The reduced
    //    result is stored in `tmp`.
    typename OP::ElTp tmp = OP::identity();
    uint32_t shmem_offset = threadIdx.x * CHUNK;
    #pragma unroll
    for (uint32_t i = 0; i < CHUNK; i++) {
        typename OP::ElTp elm = shmem_inp[shmem_offset + i];
        typename OP::ElTp red = OP::mapFun(elm);
        tmp = OP::apply(tmp, red);
        chunk[i] = tmp;
    }
    __syncthreads();

    // 3. Each thread publishes in shared memory the reduced result of its
    //    `CHUNK` elements 
    shmem_red[threadIdx.x] = tmp;
    __syncthreads();

    // 4. perform an intra-CUDA-block scan 
    tmp = scanIncBlock<OP>(shmem_red, threadIdx.x);
    __syncthreads();

    // 5. write the scan result back to shared memory
    shmem_red[threadIdx.x] = tmp;
    __syncthreads();

    // 6. the previous element is read from shared memory in `tmp`: 
    //       it is the prefix of the previous threads in the current block.
    tmp   = OP::identity();
    if (threadIdx.x > 0) 
        tmp = OP::remVolatile(shmem_red[threadIdx.x-1]);

    // 7. the prefix of the previous blocks (and iterations) is hold
    //    in `accum` and is accumulated to `tmp`, which now holds the
    //    global prefix for the `CHUNK` elements processed by the current thread.
    tmp   = OP::apply(accum, tmp);

    // 8. `accum` is also updated with the reduced result of the current
    //    iteration, i.e., of the last thread in the block: `shmem_red[blockDim.x-1]`
    accum = OP::apply(accum, shmem_red[blockDim.x-1]);
    __syncthreads();

    // 9. the `tmp` prefix is accumulated to all the `CHUNK` elements
    //      locally processed by the current thread (i.e., the ones
    //      in `chunk` array hold in registers).
    #pragma unroll
    for (uint32_t i = 0; i < CHUNK; i++) {
        shmem_red[threadIdx.x*CHUNK + i] = OP::apply(tmp, chunk[i]);
    }
    __syncthreads();

    // 10. write back from shared to global memory in coalesced fashion.
    // TODO: change this (seq is undefined)
    // copyFromShr2GlbMem<typename OP::RedElTp, CHUNK>(inp_block_offs+seq, N, d_out, shmem_red);
    copyFromShr2GlbMem<typename OP::ElTp, CHUNK>(inp_block_offs+0, N, d_out, shmem_red);
}
#endif // KERNELS