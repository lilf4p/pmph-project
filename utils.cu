#ifndef UTILS
#define UTILS

#include <sys/time.h>
#include <time.h> 

#define DEBUG_INFO  true

// flag values
#define INC 0 // nothing is available
#define AGG 1 // aggregate is available
#define PRE 2 // prefix is availabe

#define lgWARP      5
#define WARP        (1<<lgWARP)

//#define WORKGROUP_SIZE      128
//#define MAX_WORKGROUP_SIZE  1024

#define RUNS_GPU            1
#define RUNS_CPU            1
#define NUM_BLOCKS_SCAN     1024
#define ELEMS_PER_THREAD    12

typedef unsigned int uint32_t;
typedef int           int32_t;

typedef unsigned char uint8_t;

uint32_t MAX_HWDTH;
uint32_t MAX_BLOCK;
uint32_t MAX_SHMEM;

cudaDeviceProp prop;

void initHwd() {
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    cudaGetDeviceProperties(&prop, 0);
    MAX_HWDTH = prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount;
    MAX_BLOCK = prop.maxThreadsPerBlock;
    MAX_SHMEM = prop.sharedMemPerBlock;

    if (DEBUG_INFO) {
        printf("Device name: %s\n", prop.name);
        printf("Number of hardware threads: %d\n", MAX_HWDTH);
        printf("Max block size: %d\n", MAX_BLOCK);
        printf("Shared memory size: %d\n", MAX_SHMEM);
        puts("====");
    }
}

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    unsigned int resolution=1000000;
    long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
    result->tv_sec = diff / resolution;
    result->tv_usec = diff % resolution;
    return (diff<0);
}

int gpuAssert(cudaError_t code) {
  if(code != cudaSuccess) {
    printf("GPU Error: %s\n", cudaGetErrorString(code));
    return -1;
  }
  return 0;
}

void initArray(int32_t* arr, const uint32_t N, const int32_t R) {
    const uint32_t M = 2*R+1;
    for (uint32_t i = 0; i < N; i++) {
        // arr[i] = (rand() % M) - R;
        arr[i] = i;
    }
}

void printArray(const int32_t* arr, const uint32_t N) {
    for (u_int32_t i = 0; i < N; i++) {
        printf("%d, ", arr[i]);
    }
    printf("\n");
}

void seqIncScan(const int32_t* h_in, int32_t* h_out, const uint32_t N) {
    h_out[0] = h_in[0];
    for (u_int32_t i = 1; i < N; i++) {
        h_out[i] = h_out[i-1] + h_in[i];
    }
}

void validate(const int32_t* ref_arr, const int32_t* arr, const uint32_t N) {
    for(uint32_t i = 0; i<N; i++) {
        if(ref_arr[i] != arr[i]) {
            printf("!!!INVALID!!!: at index %d, ref_arr: %d, arr: %d\n", i, ref_arr[i], arr[i]);
            exit(1);
        }
    }
    printf("VALID result!\n\n");
}
#endif // UTILS
