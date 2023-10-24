#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <kernel.cu>
#include <utils.cu>

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

