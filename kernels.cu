#ifndef KERNELS
#define KERNELS

#include <cuda_runtime.h>
#include <thrust/tuple.h>

#include "utils.cu"

//---------- UTILITY KERNEL ----------//
// The kernel naiveMemcpy, copyFromGlb2ShrMem, copyFromShr2GlbMem, scanIncWarp,
// scanIncBlock are taken from the weekly assignment 2.

// Naive memcpy kernel for benchmark.
__global__ void naiveMemcpy(int* d_out, int* d_inp, const uint32_t N) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid < N) {
        d_out[gid] = d_inp[gid];
    }
}

// Coalesced mem copy glb to shr of CHUNK*B elems.
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

// Coalesced mem copy shr to glb of CHUNK*B elems.
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

// Scan in warp, used at block-level.
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

// Scan at block-level.
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

    // 5. saving the result so it could be accessed by other threads in the block
    ptr[idx] = res;
    return res;
}

//----------- SINGLE PASS SCAN KERNELS----------//

// The baseline version (does not do a lookback). It's basically the chained-scan approach.
template<class OP, uint8_t CHUNK>
__global__ void
spScanKernel ( typename OP::ElTp* d_out
             , typename OP::ElTp* d_in
             , volatile typename OP::ElTp* aggregates
             , volatile typename OP::ElTp* prefixes
             , volatile uint8_t* flags
             , volatile uint32_t* dyn_block_id
             , uint32_t N
) {
    typedef typename OP::ElTp ElTp;

    // 1. Declaring shared memory array used for various purposes,
    //      for example, copying elements to and from the global memory.
    extern __shared__ ElTp shared_mem[]; // of size CHUNK * BLOCK

    // 2. Atomically increasing the block identifier counter and reading
    //      the previous value to be visible for all threads.
    __shared__ uint32_t tmp_block_id;
    if (threadIdx.x == 0) {
        tmp_block_id = atomicAdd((uint32_t*)dyn_block_id, 1);
    }
    __syncthreads();

    uint32_t thread_id = threadIdx.x;
    uint32_t block_id = tmp_block_id;
    uint32_t block_offset = CHUNK * blockDim.x  * block_id;

    // register memory for storing the scanned elements per thread.
    ElTp chunk[CHUNK];

    // 3. Copy `CHUNK` input elements per thread from global to shared memory.
    copyFromGlb2ShrMem<ElTp, CHUNK>(block_offset, N, OP::identInp(), d_in, shared_mem);

    // 4. Each thread sequentially reduces its `CHUNK` elements,
    //      the result is stored in `tmp`.
    ElTp tmp = OP::identity();
    uint32_t shared_mem_offset = thread_id * CHUNK;
    #pragma unroll
    for (uint32_t i = 0; i < CHUNK; i++) {
        ElTp elm = shared_mem[shared_mem_offset + i];
        tmp = OP::apply(tmp, elm);
        chunk[i] = tmp;
    }
    __syncthreads();

    // 5. Each thread publishes in shared memory the reduced result
    //       of its `CHUNK` elements.
    shared_mem[thread_id] = tmp;
    __syncthreads();

    // 6. Perform an intra-CUDA-block scan. Since the last thread has
    //      the final block scan result, use it to update the inter-block state
    //      with the progress of the current block:the prefix is ready if
    //      it's the 1st, block, otherwise, only the aggregate has been computed.
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

    // 7. Save the reduction result of the previous chunk
    //      that was computed by another thread.
    ElTp prev_chunk_prefix = OP::identity();
    if (thread_id > 0) {
        prev_chunk_prefix = shared_mem[thread_id-1];
    }
    __syncthreads();

    // 8/9. Wait until the immediatly preceeding block has published
    //      the inter-block inclusive prefix and save it in a variable.
    //      Futhermore, update inter-block state with the progress
    //      of the current block: the prefix is ready.
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

    // 10. Combine the inter-block prefix with the reduction of the previous
    //      chunk. Use it to update each element of the per thread chunk
    //      and write the result to the shared memory array.
    ElTp prev_total_prefix = OP::apply(prev_block_prefix, prev_chunk_prefix);
    for (uint32_t i = 0; i < CHUNK; i++) {
        shared_mem[shared_mem_offset + i] = OP::apply(prev_total_prefix, chunk[i]);
    }
    __syncthreads();

    // 11. Copy `CHUNK` elements per thread of the result back to global memory.
    copyFromShr2GlbMem<ElTp, CHUNK>(block_offset, N, d_out, shared_mem);
}

// The version that performs the lookback step with a single thread.
template<class OP, uint8_t CHUNK>
__global__ void
spLookbackScanKernel ( typename OP::ElTp* d_out
                     , typename OP::ElTp* d_in
                     , volatile typename OP::ElTp* aggregates
                     , volatile typename OP::ElTp* prefixes
                     , volatile uint8_t* flags
                     , volatile uint32_t* dyn_block_id
                     , uint32_t N
) {
    typedef typename OP::ElTp ElTp;

    // 1. Declaring shared memory array used for various purposes,
    //      for example, copying elements to and from the global memory.
    extern __shared__ ElTp shared_mem[]; // of size CHUNK * BLOCK

    // 2. Atomically increasing the block identifier counter and reading
    //      the previous value to be visible for all threads.
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

    // 3. Copy `CHUNK` input elements per thread from global to shared memory.
    copyFromGlb2ShrMem<ElTp, CHUNK>(block_offset, N, OP::identInp(), d_in, shared_mem);

    // 4. Each thread sequentially reduces its `CHUNK` elements,
    //      the result is stored in `tmp`.
    ElTp tmp = OP::identity();
    uint32_t shared_mem_offset = thread_id * CHUNK;
    #pragma unroll
    for (uint32_t i = 0; i < CHUNK; i++) {
        ElTp elm = shared_mem[shared_mem_offset + i];
        tmp = OP::apply(tmp, elm);
        chunk[i] = tmp;
    }
    __syncthreads();

    // 5. Each thread publishes in shared memory the reduced result
    //       of its `CHUNK` elements.
    shared_mem[thread_id] = tmp;
    __syncthreads();

    // 6. Perform an intra-CUDA-block scan. Since the last thread has
    //      the final block scan result, use it to update the inter-block state
    //      with the progress of the current block:the prefix is ready if
    //      it's the 1st, block, otherwise, only the aggregate has been computed.
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

    // 7. Save the reduction result of the previous chunk
    //      that was computed by another thread.
    ElTp prev_chunk_prefix = OP::identity();
    if (thread_id > 0) {
        prev_chunk_prefix = shared_mem[thread_id-1];
    }
    __syncthreads();

    // 8. Iterate through the inter-block state until a predeccesor
    //      has published the inter-block inclusive prefix.
    //      If an aggregate is available use it to update the partial
    //      prefix and move on to the previous block in the array.
    //      In case neither the prefix nor aggregate are present
    //      wait until they are published by the corresponding block.
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

    // 9. Update inter-block state with the progress
    //      of the current block: the prefix is ready.
    if (thread_id == blockDim.x - 1) {
        prefixes[block_id] = OP::apply(prev_block_prefix, agg);
        __threadfence();
        flags[block_id] = PRE;
    }

    // 10. Combine the inter-block prefix with the reduction of the previous
    //      chunk. Use it to update each element of the per thread chunk
    //      and write the result to the shared memory array.
    ElTp prev_total_prefix = OP::apply(prev_block_prefix, prev_chunk_prefix);
    for (uint32_t i = 0; i < CHUNK; i++) {
        shared_mem[shared_mem_offset + i] = OP::apply(prev_total_prefix, chunk[i]);
    }
    __syncthreads();

    // 11. Copy `CHUNK` elements per thread of the result back to global memory.
    copyFromShr2GlbMem<ElTp, CHUNK>(block_offset, N, d_out, shared_mem);
}

// The version that performs the lookback step: reading the inter-block global
// state with a warp of threads but doing lookback with a single thread.
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

    // 1. Declaring shared memory array used for various purposes,
    //      for example, copying elements to and from the global memory.
    extern __shared__ ElTp shared_mem[]; // CHUNK * BLOCK

    // 2. Atomically increasing the block identifier counter and reading
    //      the previous value to be visible for all threads.
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

    // 3. Copy `CHUNK` input elements per thread from global to shared memory.
    copyFromGlb2ShrMem<ElTp, CHUNK>(block_offset, N, OP::identInp(), d_in, shared_mem);

    // 4. Each thread sequentially reduces its `CHUNK` elements,
    //      the result is stored in `tmp`.
    ElTp tmp = OP::identity();
    uint32_t shared_mem_offset = thread_id * CHUNK;
    #pragma unroll
    for (uint32_t i = 0; i < CHUNK; i++) {
        ElTp elm = shared_mem[shared_mem_offset + i];
        tmp = OP::apply(tmp, elm);
        chunk[i] = tmp;
    }
    __syncthreads();

    // 5. Each thread publishes in shared memory the reduced result
    //       of its `CHUNK` elements.
    shared_mem[thread_id] = tmp;
    __syncthreads();

    // 6. Perform an intra-CUDA-block scan. Since the last thread has
    //      the final block scan result, use it to update the inter-block state
    //      with the progress of the current block:the prefix is ready if
    //      it's the 1st, block, otherwise, only the aggregate has been computed.
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

    // 7. Save the reduction result of the previous chunk
    //      that was computed by another thread.
    ElTp prev_chunk_prefix = OP::identity();
    if (thread_id > 0) {
        prev_chunk_prefix = shared_mem[thread_id-1];
    }
    __syncthreads();

    // 8. Iterate through the inter-block state with WARP size step until 
    //      a first prefix is encountered and on the way accumulate the
    //      aggregates. If at least one block is marked as INC
    //      (neither prefix nor aggregate is available), redo the
    //      WARP read until there is not INC.
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

    // 9. Update inter-block state with the progress
    //      of the current block: the prefix is ready.
    if (thread_id == blockDim.x-1) {
        prefixes[block_id] = OP::apply(prev_block_prefix, agg);
        __threadfence();
        flags[block_id] = PRE;
    }

    // 10. Combine the inter-block prefix with the reduction of the previous
    //      chunk. Use it to update each element of the per thread chunk
    //      and write the result to the shared memory array.
    ElTp prev_total_prefix = OP::apply(prev_block_prefix, prev_chunk_prefix);
    for (uint32_t i = 0; i < CHUNK; i++) {
        shared_mem[shared_mem_offset + i] = OP::apply(prev_total_prefix, chunk[i]);
    }
    __syncthreads();

    // 11. Copy `CHUNK` elements per thread of the result back to global memory.
    copyFromShr2GlbMem<ElTp, CHUNK>(block_offset, N, d_out, shared_mem);
}

// The final version that performs the lookback step: reading the inter-block global
// state with a warp of threads but doing lookback with a single thread.
// This kernel employs slight optimizations over the previous one.
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

    // 1. Declaring shared memory array used for various purposes,
    //      for example, copying elements to and from the global memory.
    extern __shared__ ElTp shared_mem[]; // CHUNK * BLOCK

    // 2. Atomically increasing the block identifier counter and reading
    //      the previous value to be visible for all threads.
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

    // 3. Copy `CHUNK` input elements per thread from global to shared memory.
    copyFromGlb2ShrMem<ElTp, CHUNK>(block_offset, N, OP::identInp(), d_in, shared_mem);

    // 4. Each thread sequentially reduces its `CHUNK` elements,
    //      the result is stored in `tmp`.
    ElTp tmp = OP::identity();
    uint32_t shared_mem_offset = thread_id * CHUNK;
    #pragma unroll
    for (uint32_t i = 0; i < CHUNK; i++) {
        ElTp elm = shared_mem[shared_mem_offset + i];
        tmp = OP::apply(tmp, elm);
        chunk[i] = tmp;
    }
    __syncthreads();

    // 5. Each thread publishes in shared memory the reduced result
    //       of its `CHUNK` elements.
    shared_mem[thread_id] = tmp;
    __syncthreads();

    // 6. Perform an intra-CUDA-block scan. Since the last thread has
    //      the final block scan result, use it to update the inter-block state
    //      with the progress of the current block:the prefix is ready if
    //      it's the 1st, block, otherwise, only the aggregate has been computed.
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

    // 7. Save the reduction result of the previous chunk
    //      that was computed by another thread.
    ElTp prev_chunk_prefix = OP::identity();
    if (thread_id > 0) {
        prev_chunk_prefix = shared_mem[thread_id-1];
    }
    __syncthreads();

    // 8. Iterate through the inter-block state with WARP size step until 
    //      a first prefix is encountered and on the way accumulate the
    //      aggregates. If at least one block is marked as INC
    //      (neither prefix nor aggregate is available), update the cursor
    //      and continue with the main loop.
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

    // 9. Update inter-block state with the progress
    //      of the current block: the prefix is ready.
    if (thread_id == blockDim.x-1) {
        prefixes[block_id] = OP::apply(prev_block_prefix, agg);
        __threadfence();
        flags[block_id] = PRE;
    }

    // 10. Combine the inter-block prefix with the reduction of the previous
    //      chunk. Use it to update each element of the per thread chunk
    //      and write the result to the shared memory array.
    ElTp prev_total_prefix = OP::apply(prev_block_prefix, prev_chunk_prefix);
    for (uint32_t i = 0; i < CHUNK; i++) {
        shared_mem[shared_mem_offset + i] = OP::apply(prev_total_prefix, chunk[i]);
    }
    __syncthreads();

    // 11. Copy `CHUNK` elements per thread of the result back to global memory.
    copyFromShr2GlbMem<ElTp, CHUNK>(block_offset, N, d_out, shared_mem);
}
#endif // KERNELS
