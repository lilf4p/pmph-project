#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "kernels.cu"
#include "utils.cu"

// initialize a random array of size N
void initArray(int32_t* inp_arr, const uint32_t N, const int R) {
    const uint32_t M = 2*R+1;
    for(uint32_t i=0; i<N; i++) {
        inp_arr[i] = (rand() % M) - R;
    }
}

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

    { // timing the GPU implementations
        gettimeofday(&t_start, NULL); 

        for(int i=0; i<RUNS_GPU; i++) {
            naiveMemcpy<<< num_blocks, B >>>(d_out, d_in, N);
        }
        cudaDeviceSynchronize();

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / RUNS_GPU;
        gigaBytesPerSec = 2 * N * sizeof(int) * 1.0e-3f / elapsed;
        printf("Naive Memcpy GPU Kernel runs in: %lu microsecs, GB/sec: %.2f\n\n\n"
              , elapsed, gigaBytesPerSec);
    }
 
    gpuAssert( cudaPeekAtLastError() );
    return 0;
}

int spScanIncAddI32( const uint32_t B     // desired CUDA block size ( <= 1024, multiple of 32)
                   , const size_t   N     // length of the input array
                   , int* h_in            // host input    of size: N * sizeof(int)
                   , int* d_in            // device input  of size: N * sizeof(ElTp)
                   , int* d_out           // device result of size: N * sizeof(int)
) {
    // TODO: add validation + benchmarks
    return 0;
}

int main (int argc, char * argv[]) {
    if (argc != 3) {
        printf("Usage: %s <array-length> <block-size>\n", argv[0]);
        exit(1);
    }

    initHwd();

    const uint32_t N = atoi(argv[1]);
    const uint32_t B = atoi(argv[2]);
    printf("N=%d, B=%d\n", N, B);

    const size_t mem_size = N*sizeof(int);
    int* h_in    = (int*) malloc(mem_size);
    int* d_in;
    int* d_out;
    cudaMalloc((void**)&d_in ,   mem_size);
    cudaMalloc((void**)&d_out,   mem_size);

    initArray(h_in, N, 13);
    cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);

    // computing a "realistic/achievable" bandwidth figure
    bandwidthMemcpy(B, N, d_in, d_out);

    spScanIncAddI32(B, N, h_in, d_in, d_out);
    printf("Single Pass Scan is yet to be implemented...\n");

    // cleanup memory
    free(h_in);
    cudaFree(d_in );
    cudaFree(d_out);
}