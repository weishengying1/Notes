/*
input(M, K) : K = head_dim, 一般 <= 1024
layernorm 逻辑：
y = (x - mean(x)) / std(x)
其中标准差 std(x) = sqrt( sum((x - mean(x))^2) / K)
*/

/*
one block for on row
*/
#define WARP_SIZE 32

template<int REDUCE_SIZE>
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for(int mask = REDUCE_SIZE >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template<int THREADS_PER_BLOCK>
__device__ __forceinline__ float block_reduce_sum(float val) {
    constexpr int WARP_NUM = THREADS_PER_BLOCK / WARP_SIZE;
    __shared__ float smem[WARP_NUM];

    val = warp_reduce_sum<WARP_SIZE>(val);

    int tid = threadIdx.x;
    int lane_id = tid % 32;
    int warp_id = tid / 32;
    if(lane_id == 0) smem[warp_id] = val;
    __syncthreads();

    if (tid < WARP_NUM) val = smem[tid];
    __syncthreads();

    val = warp_reduce_sum<WARP_NUM>(val);
    return val;
}

template<int THREADS_PER_BLOCK>
__global__ void layernorm(float* in, int M, int K) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * K + tid;

    float val = in[idx];

    __shared__ s_mean = 0.f;
    __shared__ s_variance = 0.f;

    // 计算 sum
    float sum = block_reduce_sum<THREADS_PER_BLOCK>(val);
    if(tid == 0) s_mean = sum / K;
    __syncthreads();

    // 计算标准差
    float variance = (var - mean)(var - mean)
    variance = block_reduce_sum<THREADS_PER_BLOCK>(variance);
    if(tid == 0) s_variance = rsqrtf(variance / K);
    __syncthreads();

    input[idx] = (val - s_mean) * s_variance;

}