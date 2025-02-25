#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>

#define WARP_SIZE 32
#define FLT_MAX ((float)(1e10))

template<typename T>
struct Add{
    __device__ __forceinline__ T operator()(const T& x, const T& y){
        return x + y;
    }
};

template<typename T>
struct Max{
    __device__ __forceinline__ T operator()(const T& x, const T& y){
        return x > y ? x : y;
    }
};

template<typename T, int REDUCE_SIZE, template<typename> class Op>
__device__ __forceinline__ T warp_reduce(T val){
    for(int i = REDUCE_SIZE / 2; i >= 1; i >>= 1){
        val = Op<T>()(val, __shfl_xor_sync(0xffffff, val, i));
    }
    return val;
}

template<typename T, int threads_per_block, template<typename>class Op>
__device__ T block_reduce(T val){
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    constexpr int NUM_WARP = threads_per_block / WARP_SIZE;

    __shared__ float smem[threads_per_block / WARP_SIZE];

    val = warp_reduce<T, WARP_SIZE, Op>(val);
    if (lane_id == 0) smem[warp_id] = val;
    __syncthreads();

    // 将 smem 存储的局部 sum or max值再重新放入每个 warp 中
    val = lane_id < NUM_WARP ? smem[lane_id] : 0;
    val = warp_reduce<T, WARP_SIZE, Op>(val); //每个 warp 内部再做一次 reduce

    return val; //最后 block 中，所有线程都得到了正确的结果
}

// one block for one row
template<int threads_per_block>
__global__ void softmax_kernel(float* input, int row, int col) {
    input = input + blockIdx.x * col;

    int tid = threadIdx.x;
    // cal max
    float max = input[tid];
    for (int i = tid; i < col; i += threads_per_block){
        max = input[i] > max ? input[i] : max;
    } // 问题从 row 缩减为 blocksize

    max = block_reduce<float, threads_per_block, Max>(max);
    // cal sum
    float sum = 0.f;
    for (int i = tid; i < col; i += threads_per_block){
        input[i] = expf(input[i] - max);
        sum += input[i]; 
    } // 问题从 row 缩减为 blocksize

    sum = block_reduce<float, threads_per_block, Add>(sum);

    for(int i = tid; i < col; i += threads_per_block){
        input[i] /= sum;
    }
}

/**
 *  softmax kernel v2
 *  one warp for one row
 *  one block for warp_num row
*/
template<int threads_per_block=128>
__global__ void softmax_kernel_v2(float* input, int row, int col) {
    constexpr int warp_num = threads_per_block / WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int cur_row = blockIdx.x * warp_num + warp_id;
    if (cur_row >= row) return;
    input += cur_row * col;
    int lane_id = threadIdx.x % WARP_SIZE;

    // cal max
    float max = input[lane_id];
    for(int i = lane_id; i < col; i += WARP_SIZE) {
        max = input[i] > max ? input[i] : max;
    } // 问题规模缩减为 WARP_SIZE
    max = warp_reduce<float, WARP_SIZE, Max>(max);

    // cal sum
    float sum = 0.f;
    for(int i = lane_id; i < col; i += WARP_SIZE) {
        input[i] = expf(input[i] - max);
        sum += input[i];
    }
    sum = warp_reduce<float, WARP_SIZE, Add>(sum);

    for(int i = lane_id; i < col; i += WARP_SIZE) {
        input[i] /= sum;
    }
}

void softmax_cpu(float* input, float* output, int row, int col) {
    for (int i = 0; i < row; ++i) {
        float max = -FLT_MAX;
        for (int j = 0; j < col; ++j) {
            max = max > input[i * col + j] ? max : input[i * col + j];
        }
        float sum = 0.f;
        for (int j = 0; j < col; ++j) {
            sum += expf(input[i * col + j] - max);
        }
        for (int j = 0; j < col; ++j) {
            output[i * col + j] = expf(input[i * col + j] - max) / sum;
        }
    }
}

bool check(float *out, float *res, int N){
    for (int i = 0; i < N; ++i) {
        if (fabs(out[i] - res[i]) > 1e-5) {
            printf("error\n");
            return false;
        }
    }
    printf("success\n");
    return true;
}

void print(float* val, int row, int col) {
    printf("======================\n");
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            printf("%.3f ", val[i * col + j]);
        }
        printf("\n");
    }
}

int main() {
    int row = 127;
    int col = 1024;
    constexpr int threads_per_block = 128;
    constexpr int warp_num = threads_per_block / WARP_SIZE;

    float* input = (float*)malloc(row * col * sizeof(float));
    float* output = (float*)malloc(row * col * sizeof(float));
    float* output_ref = (float*)malloc(row * col * sizeof(float));
    float* d_input;
    cudaMalloc((void**)&d_input, row * col * sizeof(float));

    srand(time(NULL));
    for (int i = 0; i < row * col; ++i) {
        input[i] = rand() % 17;
    }
    softmax_cpu(input, output_ref, row, col);
    // one block for one row
    cudaMemcpy(d_input, input, row * col * sizeof(float), cudaMemcpyHostToDevice);
    softmax_kernel<threads_per_block><<<row, threads_per_block>>>(d_input, row, col);
    cudaMemcpy(output, d_input, row * col * sizeof(float), cudaMemcpyDeviceToHost);
    check(output, output_ref, row * col);
    // one warp for one row
    cudaMemcpy(d_input, input, row * col * sizeof(float), cudaMemcpyHostToDevice);
    softmax_kernel_v2<threads_per_block><<<(row + warp_num - 1) / warp_num, threads_per_block>>>(d_input, row, col);
    cudaMemcpy(output, d_input, row * col * sizeof(float), cudaMemcpyDeviceToHost);
    check(output, output_ref, row * col);
    return 0;
}