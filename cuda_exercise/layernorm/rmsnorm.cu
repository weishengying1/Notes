#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>

/*
intput: (n, h)
one block for one row
*/

#define THREAD_PER_BLOCK 256
#define WARP_SIZE 32

template<typename T>
__device__ __forceinline__ T warp_reduce_sum(T val){
    #pragma unroll
    for(int mask = WARP_SIZE >> 1; mask >= 1; mask >>=1){
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template<typename T>
__device__ T block_reduce_sum(T val){
    int tid = threadIdx.x;
    int lane_id = tid & (WARP_SIZE - 1);
    int warp_id = tid >> 5;
    constexpr int NUM_WARP = THREAD_PER_BLOCK / WARP_SIZE;

    __shared__ T smem[NUM_WARP];
    val = warp_reduce_sum<T>(val);
    if(lane_id == 0) smem[warp_id] = val;
    __syncthreads();

    val = lane_id < NUM_WARP ? smem[lane_id] : 0;
    val = warp_reduce_sum<T>(val);
    return val;

}
template<typename T>
__global__ void rms_norm_kernel(T* in, T* out, int n, int h) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    in += bid * h;

    float x = 0.f;
    float variance = 0.f;
    for(int i = tid; i < h; i += THREAD_PER_BLOCK){
        x = in[tid];
        variance = x * x;
    }

    variance = block_reduce_sum<float>(variance); // 求平方和

    float s_variance = rsqrtf(variance / h);

    for(int i = tid; i < h; i += THREAD_PER_BLOCK){
        out[bid * h + tid] = (T)(x * s_variance);
    }
}

int main() {
    constexpr int N = 1024;
    constexpr int H = 4096;
    constexpr int threads_per_block = THREAD_PER_BLOCK;

    
    float* d_in, *d_out;
    cudaMalloc((void**)&d_in, N * H * sizeof(float));
    cudaMalloc((void**)&d_out, N * H * sizeof(float));

    dim3 grid_dim(N);
    dim3 block_dim(threads_per_block);
    rms_norm_kernel<float><<<grid_dim, block_dim>>>(d_in, d_out, N, H);
    

    int TEST_TIMES = 100;
    for (int i = 0; i < TEST_TIMES; ++i) {
        rms_norm_kernel<float><<<grid_dim, block_dim>>>(d_in, d_out, N, H);
    }
    float time_elapsed=0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start,0);    //记录当前时间
    for (int i = 0; i < TEST_TIMES; ++i) {
        rms_norm_kernel<float><<<grid_dim, block_dim>>>(d_in, d_out, N, H);
    }
    cudaEventRecord(stop,0);    //记录当前时间
    cudaEventSynchronize(start);    //Waits for an event to complete.
    cudaEventSynchronize(stop);     //Waits for an event to complete.Record之前的任务
    cudaEventElapsedTime(&time_elapsed, start, stop);    //计算时间差
    std::cout << "rmsnorm elasped time = " << time_elapsed/TEST_TIMES << "ms" << std::endl;

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}