#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>

#define THREAD_PER_BLOCK 256

__device__ void warp_reduce(volatile float* cache,int tid) {
    cache[tid] += cache[tid+32];
    cache[tid] += cache[tid+16];
    cache[tid] += cache[tid+8];
    cache[tid] += cache[tid+4];
    cache[tid] += cache[tid+2];
    cache[tid] += cache[tid+1];
}

__global__ void reduce_sum_kernel(float* input, float* res) {
    // load data to smem
    __shared__ float smem[THREAD_PER_BLOCK];
    int idx = threadIdx.x + 2 * blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    smem[tid] = input[idx] + input[idx + blockDim.x];
    __syncthreads();

    for(int i = THREAD_PER_BLOCK / 2; i > 32; i >>= 1) {
        if(tid < i)
            smem[tid] = smem[tid] + smem[tid + i];
        __syncthreads();
    }

    if (tid < 32) {
        warp_reduce(buffer, tid);
    }

    if(tid == 0)
        res[blockIdx.x] = smem[0];
}


bool check(float *out, float *res, int N){
    for(int i=0; i<N; i++){
        if(out[i]!= res[i]) {
            printf("out[%d]=%f, ref[%d]=%f\n", i, out[i], i, res[i]);
            return false;
        }
    }
    return true;
}

int main() {
    constexpr int N = 1024 * 1024;
    constexpr int threads_per_block = THREAD_PER_BLOCK;

    float* arr = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; ++i) {
        arr[i] = 1; // i * (N - 1) / (float) (N * 10);
    }
    
    float* d_arr;
    cudaMalloc((void**)&d_arr, N * sizeof(float));
    cudaMemcpy(d_arr, arr, N * sizeof(float), cudaMemcpyHostToDevice);

    float* out = (float*)malloc(N / threads_per_block * sizeof(float));
    float* out_ref = (float*)malloc(N / threads_per_block * sizeof(float));

    float* d_out;
    cudaMalloc((void**)&d_out, N / threads_per_block * sizeof(float));

    for (int i = 0; i < N / threads_per_block; ++i) {
        float sum = 0.f;
        for (int j = 0; j < threads_per_block; ++j) {
            sum += arr[i * threads_per_block + j];
        }
        out_ref[i] = sum;
    }

    dim3 grid_dim(N / (2 * threads_per_block));
    dim3 block_dim(threads_per_block);

    reduce_sum_kernel<<<grid_dim, block_dim>>>(d_arr, d_out);
    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        printf("Kernel execution failed: %s\n", cudaGetErrorString(syncErr));
        return 1;
    }
    cudaMemcpy(out, d_out, N/threads_per_block * sizeof(float), cudaMemcpyDeviceToHost);

    if (check(out, out_ref, N/threads_per_block)) {
        std::cout << "succeed!" << std::endl;
    } else {
        std::cout << "failed!" << std::endl;
        return 1;
    }

    int TEST_TIMES = 10;
    for (int i = 0; i < 10; ++i) { // warm up
        reduce_sum_kernel<<<grid_dim, block_dim>>>(d_arr, d_out);
    }
    float time_elapsed=0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start,0);    //记录当前时间
    for (int i = 0; i < TEST_TIMES; ++i) {
        reduce_sum_kernel<<<grid_dim, block_dim>>>(d_arr, d_out);
    }
    cudaEventRecord(stop,0);    //记录当前时间
    cudaEventSynchronize(start);    //Waits for an event to complete.
    cudaEventSynchronize(stop);     //Waits for an event to complete.Record之前的任务
    cudaEventElapsedTime(&time_elapsed, start, stop);    //计算时间差
    std::cout << "reduce_sum_kernel elasped time = " << time_elapsed/TEST_TIMES << "ms" << std::endl;

    cudaFree(d_arr);
    cudaFree(d_out);
    free(arr);
    free(out);
    free(out_ref);
    return 0;
}