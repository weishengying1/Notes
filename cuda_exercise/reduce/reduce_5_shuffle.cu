#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>

#define THREAD_PER_BLOCK 256
#define WARP_SIZE 32

template<unsigned int N>
__device__ __forceinline__ float warp_reduce_sum(float sum) {
    //shuffle原语允许一个warp内的线程直接交换寄存器中的数据，而不需要经过共享内存
    // thread16将sum值传递给thread0,此时thread0的值为sum(本身)+sum(thread16)
    //最后warp内所有的sum全都累加到thread0上
    if(N >= 32) sum += __shfl_down_sync(0xffffffff, sum, 16);
    if(N >= 16) sum += __shfl_down_sync(0xffffffff, sum, 8);
    if(N >= 8) sum += __shfl_down_sync(0xffffffff, sum, 4);
    if(N >= 4) sum += __shfl_down_sync(0xffffffff, sum, 2);
    if(N >= 2) sum += __shfl_down_sync(0xffffffff, sum, 1);
    return sum;
}


__global__ void reduce_sum_kernel(float* input, float* res) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    float sum = input[idx];
    __syncthreads();

    // reduce in warp
    sum = warp_reduce_sum<THREAD_PER_BLOCK>(sum); //这里以 warp 为粒度做 reduce sum

    __shared__ float warp_reduce_sum_buffer[THREAD_PER_BLOCK/WARP_SIZE];
    int lane_id = tid & 0x1f;
    int warp_id = tid >> 5;

    if (lane_id == 0) warp_reduce_sum_buffer[warp_id] = sum; 
    __syncthreads();

    // 把不同 warp 的 lane_id=0 的数据（即该 warp 内的 sum）放到一个warp内。tid为前 8 个才有sum
    // 即前 8 个线程从 smem 中读取数据
    sum = tid < THREAD_PER_BLOCK/32 ? warp_reduce_sum_buffer[tid] : 0;
    
    // 第一个 warp 再做一个 warp 内的 reduce
    if(warp_id == 0) sum = warp_reduce_sum<THREAD_PER_BLOCK/32>(sum);

    if(tid == 0) res[blockIdx.x] = sum;
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

    dim3 grid_dim(N / threads_per_block);
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