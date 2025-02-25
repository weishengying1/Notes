#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>

#define THREAD_PER_BLOCK 256
#define WARP_SIZE 32


template<int REDUCE_SIZE>
__device__ __forceinline__ float warp_reduce_sum(float val){
    #pragma unroll
    for(int mask = REDUCE_SIZE / 2; mask >= 1; mask >>= 1){
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val) {
    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    constexpr int NUM_WARP = THREAD_PER_BLOCK / WARP_SIZE;
    __shared__ float smem[NUM_WARP];

    val = warp_reduce_sum<WARP_SIZE>(val);
    if(lane_id == 0) smem[warp_id] = val;
    __syncthreads();

    val = lane_id < NUM_WARP ? smem[lane_id] : 0;
    val = warp_reduce_sum<WARP_SIZE>(val);
    return val;
}

/*
in:(N ,1)
block 在 N 上平铺
*/
__global__ void reduce_sum_kernel(const float* in, float* res, int n){
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * THREAD_PER_BLOCK + tid;

    float val = idx < n ? in[idx] : 0;
    val = block_reduce_sum(val);
    if(tid == 0) atomicAdd(res, val);
}


bool check(float *out, float *res, int N){
    for(int i=0; i<N; i++){
        if(out[i]!= res[i]) {
            // printf("out[%d]=%f, ref[%d]=%f\n", i, out[i], i, res[i]);
            return false;
        }
    }
    return true;
}

int main() {
    constexpr int N = 1 * 1024;
    constexpr int threads_per_block = THREAD_PER_BLOCK;

    float* arr = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; ++i) {
        arr[i] = 1; // i * (N - 1) / (float) (N * 10);
    }
    
    float* d_arr;
    cudaMalloc((void**)&d_arr, N * sizeof(float));
    cudaMemcpy(d_arr, arr, N * sizeof(float), cudaMemcpyHostToDevice);

    float* out = (float*)malloc(1 * sizeof(float));
    float* out_ref = (float*)malloc(1 * sizeof(float));

    float* d_out;
    cudaMalloc((void**)&d_out, 1 * sizeof(float));

    float sum = 0.f;
    for (int i = 0; i < N ; ++i) {
        sum += arr[i];
    }
    out_ref[0] = sum;

    dim3 grid_dim(N / threads_per_block);
    dim3 block_dim(threads_per_block);

    reduce_sum_kernel<<<grid_dim, block_dim>>>(d_arr, d_out, N);
    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        printf("Kernel execution failed: %s\n", cudaGetErrorString(syncErr));
        return 1;
    }
    cudaMemcpy(out, d_out, 1 * sizeof(float), cudaMemcpyDeviceToHost);

    if (check(out, out_ref, 1)) {
        std::cout << "succeed!" << std::endl;
    } else {
        std::cout << "failed!" << std::endl;
        return 1;
    }

    int TEST_TIMES = 10;
    for (int i = 0; i < 10; ++i) { // warm up
        reduce_sum_kernel<<<grid_dim, block_dim>>>(d_arr, d_out, N);
    }
    float time_elapsed=0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start,0);    //记录当前时间
    for (int i = 0; i < TEST_TIMES; ++i) {
        reduce_sum_kernel<<<grid_dim, block_dim>>>(d_arr, d_out, N);
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