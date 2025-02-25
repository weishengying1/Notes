
#include <iostream>
#include "cuda_runtime.h"

// input:(N, 1)
#define THREAD_PER_BLOCK 256
#define WARP_SIZE 32

__device__ float warp_prefix_sum(float val){
    int lane = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for(int mask = 1; mask <= WARP_SIZE / 2; mask <<=1){
        float tmp = __shfl_up_sync(0xffffffff, val, mask);
        if(lane >= 1)
            val += tmp;
    }
    return val;
}

__device__ float block_prefix_sum(float val){
    int lane_id = threadIdx.x  % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    constexpr int NUM_WARP = THREAD_PER_BLOCK / WARP_SIZE;
    __shared__ float smem[NUM_WARP];

    val = warp_prefix_sum(val);
    if(lane_id == WARP_SIZE - 1)
        smem[warp_id] = val;
    __syncthreads();

    if(warp_id == 0)
        smem[lane_id] = warp_prefix_sum(smem[lane_id]);
    __syncthreads();

    if(warp_id >= 1) val += smem[warp_id - 1];

}

/*
dim3 gridDim(ceil(N/THREAD_PER_BLOCK))
*/
__global__ void prefix_sum_kernel(const float* in, int N) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * THREAD_PER_BLOCK + tid;

    float val = idx < N ? in[idx] : 0;

    val = block_prefix_sum(val);
}