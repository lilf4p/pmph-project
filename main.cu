#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "kernels.cu"
#include "utils.cu"

#include <iostream>
#include <fstream>

// Measure a more-realistic optimal bandwidth by a simple, memcpy-like kernel 
int bandwidthMemcpy( const uint32_t B     // desired CUDA block size ( <= 1024, multiple of 32)
                   , const size_t   N     // length of the input array
                   , int* d_in            // device input  of length N
                   , int* d_out           // device result of length N
) {
    // dry run to exercise the d_out allocation!
    const uint32_t num_blocks = (N + B - 1) / B;
    naiveMemcpy<<< num_blocks, B >>>(d_out, d_in, N);

    double gigaBytesPerSec;
    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;

    { // timing the naivememcpy 
        gettimeofday(&t_start, NULL); 

        for(int i=0; i<RUNS_GPU; i++) {
            naiveMemcpy<<< num_blocks, B >>>(d_out, d_in, N);
        }
        cudaDeviceSynchronize();

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / RUNS_GPU;
        gigaBytesPerSec = 2 * N * sizeof(int) * 1.0e-3f / elapsed;
        printf("Naive Memcpy GPU Kernel runs in: %lu microsecs, GB/sec: %.2f\n"
              , elapsed, gigaBytesPerSec);
    }
 
    gpuAssert( cudaPeekAtLastError() );
    return gigaBytesPerSec;
}

// Function that benchmark and validate the single pass scan 
// Return the gigaBytesPerSec of the sps 
template<class OP, uint8_t CHUNK>
int spScanInc( uint32_t B     // desired CUDA block size ( <= 1024, multiple of 32)
                   , uint32_t N     // length of the input array
                   , int* h_in            // host input    of size: N * sizeof(int)
                   , int* d_in            // device input  of size: N * sizeof(int)
                   , int* d_out           // device result of size: N * sizeof(int)
                   , uint32_t kernel_version    // scan kernel version
                   , int validate     
) {

    const size_t mem_size = N * sizeof(int);
    int* h_out = (int*)malloc(mem_size);
    int* h_ref = (int*)malloc(mem_size);
    cudaMemset(d_out, 0, N*sizeof(int));

    // kernel parameters 
    const uint32_t elems_per_block = B * CHUNK;
    const uint32_t num_blocks = (N + elems_per_block - 1) / elems_per_block;
    const uint32_t shared_mem_size = B * sizeof(typename OP::ElTp) * CHUNK;
    //printf("elems_per_block=%d, CHUNK=%d, num_blocks=%d, shmem_size=%d\n", elems_per_block, CHUNK, num_blocks, shared_mem_size);

    if (shared_mem_size>=MAX_SHMEM) {
        printf("Max Shared Memory Limit! Skip this run...\n");
        return -1;
    }

    // mallocs 
    typename OP::ElTp* aggregates;
    typename OP::ElTp* prefixes;
    uint8_t* flags;
    uint32_t* dyn_block_id;

    cudaMalloc((void**)&aggregates, num_blocks*sizeof( typename OP::ElTp));
    cudaMalloc((void**)&prefixes, num_blocks*sizeof( typename OP::ElTp));
    cudaMalloc((void**)&flags, num_blocks*sizeof( uint8_t ));
    cudaMalloc((void**)&dyn_block_id, sizeof( uint32_t ));

    // ------- 10 dry run to exercise allocation ------- //
    for (int i=0; i<10; i++) {
        cudaMemset(flags, INC, num_blocks * sizeof(uint8_t));
        cudaMemset(dyn_block_id, 0, sizeof(uint32_t));
        // choose which version of the kernel to run
        switch (kernel_version)
        {
        case 1:
            spScanKernel<OP, CHUNK><<<num_blocks, B, shared_mem_size>>>(d_out, d_in, aggregates, prefixes, flags, dyn_block_id, N);
            break;
        case 2: 
            spLookbackScanKernel<OP, CHUNK><<<num_blocks, B, shared_mem_size>>>(d_out, d_in, aggregates, prefixes, flags, dyn_block_id, N);
            break;
        case 3: 
            spWarpLookbackScanKernel<OP, CHUNK><<<num_blocks, B, shared_mem_size>>>(d_out, d_in, aggregates, prefixes, flags, dyn_block_id, N);
            break;
        case 4: 
            spWarpLookbackScanKernelOpt<OP, CHUNK><<<num_blocks, B, shared_mem_size>>>(d_out, d_in, aggregates, prefixes, flags, dyn_block_id, N);
            break;
        default:
            printf("Kernel Version must be a value between 0-4\n");
            printf("<kernel-version>:\n"
            "    - 1: Without loopback (spScanKernel)\n"
            "    - 2: Single thread Loopback (spLookbackScanKernel)\n"
            "    - 3: Warp Loopback (spWarpLookbackScanKernel)\n"
            "    - 4: Optimized Warp Loopback (spWarpLookbackScanKernelOpt)\n\n");            
            exit(1);
        }
    }
    // ------------------------------ //

    // ---------- Time the GPU computation---------- //
    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL); 

    for(int i=0; i<RUNS_GPU; i++) {

        // reset before every execution
        cudaMemset(flags, INC, num_blocks * sizeof(uint8_t));
        cudaMemset(dyn_block_id, 0, sizeof(uint32_t));

        // choose which version of the kernel to run
        switch (kernel_version)
        {
        case 1:
            spScanKernel<OP, CHUNK><<<num_blocks, B, shared_mem_size>>>(d_out, d_in, aggregates, prefixes, flags, dyn_block_id, N);
            break;
        case 2: 
            spLookbackScanKernel<OP, CHUNK><<<num_blocks, B, shared_mem_size>>>(d_out, d_in, aggregates, prefixes, flags, dyn_block_id, N);
            break;
        case 3: 
            spWarpLookbackScanKernel<OP, CHUNK><<<num_blocks, B, shared_mem_size>>>(d_out, d_in, aggregates, prefixes, flags, dyn_block_id, N);
            break;
        case 4: 
            spWarpLookbackScanKernelOpt<OP, CHUNK><<<num_blocks, B, shared_mem_size>>>(d_out, d_in, aggregates, prefixes, flags, dyn_block_id, N);
            break;
        default:
            printf("Kernel Version must be a value between 0-4\n");
            printf("<kernel-version>:\n"
            "    - 1: Without loopback (spScanKernel)\n"
            "    - 2: Single thread Loopback (spLookbackScanKernel)\n"
            "    - 3: Warp Loopback (spWarpLookbackScanKernel)\n"
            "    - 4: Optimized Warp Loopback (spWarpLookbackScanKernelOpt)\n\n");            
            exit(1);
        }
    }
    cudaDeviceSynchronize();

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / RUNS_GPU;
    double gigaBytesPerSec = N  * (2*sizeof(int) + sizeof(int)) * 1.0e-3f / elapsed;
    printf("Single Pass Scan GPU Kernel runs in: %lu microsecs, GB/sec: %.2f\n"
          , elapsed, gigaBytesPerSec);

    gpuAssert( cudaPeekAtLastError() );
    //-------------------------------------//

    // ---------- Sequential computation for validation --------------- //
    if (validate) { 
        gettimeofday(&t_start, NULL);
        // printf("INPUT:\n");
        for(int i=0; i<RUNS_CPU; i++) {
            int acc = 0;
            for(uint32_t i=0; i<N; i++) {
                acc += h_in[i];
                h_ref[i] = acc;
                // printf("%d ", h_in[i]);
            }
            // printf("\n\n");
        }
        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / RUNS_CPU;
        double gigaBytesPerSec = N * (sizeof(int) + sizeof(int)) * 1.0e-3f / elapsed;
        printf("Scan CPU Sequential runs in: %lu microsecs, GB/sec: %.2f\n"
              , elapsed, gigaBytesPerSec);
    
        cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost);

        // printf("REF OUTPUT\n");
        // for(uint32_t i = 0; i<N; i++) {
        //     printf("%d ", h_ref[i]);
        // }
        // printf("\n");

        // printf("SPS OUTPUT\n");
        // for(uint32_t i = 0; i<N; i++) {
        //     printf("%d ", h_out[i]);
        // }
        // printf("\n\n");

        for(uint32_t i = 0; i<N; i++) {
            if(h_out[i] != h_ref[i]) {
                printf("!!!INVALID!!!: Single Pass Scan at index %d, dev-val: %d, host-val: %d\n"
                      , i, h_out[i], h_ref[i]);
                exit(1);
            }
        }
        printf("Single Pass Scan: VALID result!\n\n");
        // --------------------------------------- //
    }

    free(h_out);
    free(h_ref);

    // return bandwidth of sps 
    return gigaBytesPerSec;
}

int main (int argc, char * argv[]) {

    if (argc != 5) {
        printf("Usage: %s <benchmark> <array-length> <block-size> <kernel-version>\n", argv[0]);
        printf("<kernel-version>:\n"
        "    - 1: Without loopback (spScanKernel)\n"
        "    - 2: Single thread Loopback (spLookbackScanKernel)\n"
        "    - 3: Warp Loopback (spWarpLookbackScanKernel)\n"
        "    - 4: Optimized Warp Loopback (spWarpLookbackScanKernelOpt)\n\n");
        exit(1);
    }

    initHwd();

    // Configuration from main call 
    uint8_t BENCHMARK = atoi(argv[1]);
    uint32_t N = atoi(argv[2]);
    uint32_t B = atoi(argv[3]);
    uint32_t KERNEL = atoi(argv[4]);
    
    // Constant value 
    const uint8_t CHUNK = CHUNK_VALUE;

    int* h_in;
    int* d_in;
    int* d_out;

    if (BENCHMARK) {

        printf("==== Benchmark ====\n");
        
        // Try different configuration
        uint32_t kernel_versions[] = {1};
        uint32_t n_sizes[] = {1024, 221184, 1000000, 10000000, 100003565}; 
        uint32_t block_sizes[] = {64,128,256,512,1024};
        //const uint32_t chunk_values[] = {1,2,6,10,12,14}; // Do this manually
        
        int num_ker = sizeof(kernel_versions)/sizeof(kernel_versions[0]);
        int num_n = sizeof(n_sizes)/sizeof(n_sizes[0]);
        int num_block = sizeof(block_sizes)/sizeof(block_sizes[0]);
        printf("Total number of configurations: %d\n", (num_ker*num_n*num_block) );

        int count = 0;

        std::ofstream results;
        results.open("benchmarks/bench-sps-nolb-" + std::to_string(CHUNK) + ".csv");
        results << "kernel,input,block,chunk,bandwidth\n";
        //std::ofstream naive_memcpy_res;
        //naive_memcpy_res.open("bench-naiveMemcpy-" + std::to_string(CHUNK) + ".csv");
        //naive_memcpy_res << "input,block,bandwidth\n";
        //std::ofstream cuda_memcpy_res;
        //cuda_memcpy_res.open("bench-cudaMemcpy-" + std::to_string(CHUNK) + ".csv");
        //cuda_memcpy_res << "input,bandwidth\n";

        for (int kernel = 0; kernel < num_ker; kernel++) {
            for (int n = 0; n < num_n; n++) {
                for (int block_size = 0; block_size < num_block; block_size++) {
                    //for (int c = 0; c < arrayLength(chunk_values); c++) {
                        
                        // write config of first run
                        results << kernel_versions[kernel] << "," << n_sizes[n] << "," << block_sizes[block_size] << "," << std::to_string(CHUNK) << ",";
                        //naive_memcpy_res << n_sizes[n] << "," << block_sizes[block_size] << ",";
                        //cuda_memcpy_res << n_sizes[n] << ",";

                        count++;
                        //printf("======== Bench Run %d =======\n", count);
                        //printf("Configuration: KERNEL=%d, N=%d, B=%d, CHUNK=%d\n", kernel_versions[kernel], n_sizes[n], block_sizes[block_size], CHUNK);
                        //if (kernel_versions[kernel] == 4) printf("Latest Version of the SPScan Kernel is running...\n");
                        //else printf("An older version of the SPScan Kernel is running. For the best performance run %s <array-length> <block-size> 3\n", argv[0]);

                        // run with current config 
                        const size_t mem_size = n_sizes[n]*sizeof(int);
                        h_in = (int*) malloc(mem_size);
                        cudaMalloc((void**)&d_in ,   mem_size);
                        cudaMalloc((void**)&d_out,   mem_size);

                        initArray(h_in, n_sizes[n], 13);
        
                        // run the single pass scan without validation 
                        double gigaBytesPerSec = spScanInc<Add<int>, CHUNK>(block_sizes[block_size], n_sizes[n], h_in, d_in, d_out, kernel_versions[kernel], 0);

                        // computing a "realistic/achievable" bandwidth figure
                        //double gbN = bandwidthMemcpy(block_sizes[block_size], n_sizes[n], d_in, d_out);

                        // write results
                        results << gigaBytesPerSec << "\n";
                        //naive_memcpy_res << gbN << "\n";
                        //cuda_memcpy_res << gbC << "\n";

                        //printf("==================\n");

                    //}
                }
            }
        }
        results.close();

    } else {

        // Check parameters
        if (B % 32 != 0) {
            printf("Block size must be a multiple of 32!\n");
            exit(1);
        }
        if (KERNEL > 4) {
            printf("Kernel version must be one between 0-3\n");
            exit(1);
        }

        // Info Current Run
        printf("N=%d, B=%d, Kernel Version=%d\n", N, B, KERNEL);
        if (KERNEL == 4) printf("Latest Version of the SPScan Kernel is running...\n\n");
        else printf("An older version of the SPScan Kernel is running. For the best performance run %s <array-length> <block-size> 3\n\n", argv[0]);

        const size_t mem_size = N*sizeof(int); 
        h_in = (int*) malloc(mem_size);
        cudaMalloc((void**)&d_in ,   mem_size);
        cudaMalloc((void**)&d_out,   mem_size);

        initArray(h_in, N, 13);
        cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);

        // computing a "realistic/achievable" bandwidth figure
        bandwidthMemcpy(B, N, d_in, d_out);
        
        // run the single pass scan with validation
        spScanInc<Add<int>, CHUNK>(B, N, h_in, d_in, d_out, KERNEL, 1);

    }

    // cleanup memory
    free(h_in);
    cudaFree(d_in );
    cudaFree(d_out);

}
