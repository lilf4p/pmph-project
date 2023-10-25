#ifndef SCAN_HOST
#define SCAN_HOST

#include "utils.cu"
#include "kernels.cu"

uint32_t closestMul32(uint32_t x) {
    return ((x + 31) / 32) * 32;
}

void log2UB(uint32_t n, uint32_t* ub, uint32_t* lg) {
    uint32_t r = 0;
    uint32_t m = 1;
    if( n <= 0 ) { printf("Error: log2(0) undefined. Exiting!!!"); exit(1); }
    while(m<n) {
        r = r + 1;
        m = m * 2;
    }
    *ub = m;
    *lg = r;
}

/**
 * Host Wrapper orchestraiting the execution of scan:
 * d_in  is the input array
 * d_out is the result array (result of scan)
 * t_tmp is a temporary array (used to scan in-place across the per-block results)
 * Implementation consist of one phase
 */
template<class OP>                     // element-type and associative operator properties
void scanInc( const uint32_t     B     // desired CUDA block size ( <= 1024, multiple of 32)
            , const size_t       N     // length of the input array
            , typename OP::ElTp* d_out
            , typename OP::ElTp* d_in
            ){

    // const uint32_t CHUNK = ELEMS_PER_THREAD*4 / sizeof(typename OP::ElTp);
    const uint32_t CHUNK = 12;
    const uint32_t num_blocks = (N + B - 1) / B;
    const size_t   shmem_size = B * sizeof(typename OP::ElTp) * CHUNK;

    typename OP::ElTp* aggregates;
    typename OP::ElTp* prefixes;
    uint8_t* flags;
    uint32_t* dyn_block_id;

    cudaMalloc((void**)&aggregates, num_blocks*sizeof( typename OP::ElTp));
    cudaMalloc((void**)&prefixes, num_blocks*sizeof( typename OP::ElTp));
    cudaMalloc((void**)&flags, num_blocks*sizeof( uint8_t ));
    cudaMalloc((void**)&dyn_block_id, sizeof( uint32_t ));

    cudaMemset(prefixes, INC, num_blocks);
    cudaMemset(dyn_block_id, 0, 1);

    spScanKernel<OP, CHUNK, B><<<num_blocks, B>>>(d_out, d_in, aggregates, prefixes, flags, N);
}

#endif
