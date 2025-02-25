/*
A(M, N) B(N, 1) --> C(M, 1)
假设 N 是 32 的倍数
one warp for one row
dim3 gridDim(M / ROWS_PER_BLOCK)
*/

#define WARP_SIZE 32
#define THREADS_PER_BLOCK 256

constexpr int WARP_NUM = THREADS_PER_BLOCK / WARP_SIZE;
constexpr int ROWS_PER_BLOCK = WARP_NUM;

template<typename T, int REDUCE_SIZE>
__device__ __forceinline__ T reduce_sum(T val){
    #pragma unroll
    for(int mask = REDUCE_SIZE >> 1; mask >=1; mask >= 1){
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

// one warp for one row
// A(M,N) B(N,1)
__global__ void gemv_k32(const float* a, const float* b, float* c, int M, int N){
    // 计算每个 tid 负责的行
    int lane_id = blockIdx.x % WARP_SIZE;
    int warp_id = blockIdx.x / WARP_SIZE;
    int row = blockIdx.x * ROWS_PER_BLOCK + warp_id;
    if(row >= M) return;

    float sum = 0.f;
    for(int i = lane_id; i < N; i += WARP_SIZE) {
        sum += a[row * N + i] * b[i]
    }
    sum = warp_reduce_sum<float, WARP_SIZE>(sum);
    if(lane_id == 0) c[row] = sum;
}