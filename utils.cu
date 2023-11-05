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

#define RUNS_GPU            100
#define RUNS_CPU            10

typedef unsigned int uint32_t;
typedef int           int32_t;

typedef unsigned char uint8_t;

uint32_t MAX_HWDTH;
uint32_t MAX_BLOCK;
uint32_t MAX_SHMEM;

cudaDeviceProp prop;

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

void initHwd() {
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    cudaGetDeviceProperties(&prop, 0);
    MAX_HWDTH = prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount;
    MAX_BLOCK = prop.maxThreadsPerBlock;
    MAX_SHMEM = prop.sharedMemPerBlock;

    if (DEBUG_INFO) {
        printf("==Hwd Info==\n");
        printf("Device name: %s\n", prop.name);
        printf("Number of hardware threads: %d\n", MAX_HWDTH);
        printf("Max block size: %d\n", MAX_BLOCK);
        printf("Shared memory size: %d\n", MAX_SHMEM);
        printf("==========\n");
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
        arr[i] = (rand() % M) - R;
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

#endif // UTILS
