# softmax
```c++
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

    int tid = threadIdx.x;

    __shared__ float smem[threads_per_block / WARP_SIZE];

    val = warp_reduce<T, WARP_SIZE, Op>(val);

    if (lane_id == 0)
        smem[warp_id] = val;
    __syncthreads();

    // 将 smem 存储的局部 sum or max值放入第一个 warp 中
    val = tid < NUM_WARP ? smem[tid] : val;
    val = warp_reduce<T, NUM_WARP, Op>(val);

    return val; //最后 block 中，只有第一个 warp 中才能得到 max or sum 值
}

// one block for one row
template<int threads_per_block>
__global__ void softmax_kernel(float* input, int row, int col) {
    input = input + blockIdx.x * col;

    int tid = threadIdx.x;
    __shared__ float s_max, s_sum;

    // cal max
    float max = input[tid];
    for (int i = tid; i < col; i += threads_per_block){
        max = input[i] > max ? input[i] : max;
    } // 问题从 row 缩减为 blocksize

    max = block_reduce<float, threads_per_block, Max>(max); // 只有第一个 warp 中才能得到 max 值
    if(tid == 0) s_max = max;
    __syncthreads(); // 因为其他线程都需要读 s_max，所以必须等待线程 0 完成

    // cal sum
    float sum = 0.f;
    for (int i = tid; i < col; i += threads_per_block){
        input[i] = expf(input[i] - s_max);
        sum += input[i]; 
    } // 问题从 row 缩减为 blocksize

    sum = block_reduce<float, threads_per_block, Add>(sum); // 只有第一个 warp 中才能得到 sum 值
    if(tid == 0) s_sum = 1.0f / sum;
    __syncthreads();

    for(int i = tid; i < col; i += threads_per_block){
        input[i] = input[i] * s_sum;
    }
}
```

# gemm
```c++
#define BLOCK_SIZE 32
/*
dim3 gridDim(ceil(N/BLOCK_SIZE), ceil(M/BLOCK_SIZE))
dim3 blockDim(BLOCK_SIZE * BLOCK_SIZE)
*/
__global__ void gemm(const float* a, const float* b, float* c, int M, int N, int K) {
    __shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sB[BLOCK_SIZE][BLOCK_SIZE];

    // 计算每个 block 负责区域的起始地址
    a += blockIdx.y * BLOCK_SIZE * K;
    b += blockIdx.x * BLOCK_SIZE;
    c += blockIdx.y * BLOCK_SIZE * N + blockIdx.x * BLOCK_SIZE;

    int t_row = threadIdx.x / BLOCK_SIZE;
    int t_col = threadIdx.x % BLOCK_SIZE;

    float tmp = 0.f;
    for(int i = 0; i < K; i += BLOCK_SIZE) {
        sA[t_row][t_col] = a[t_row * K + t_col];
        sB[t_row][t_col] = b[t_row * N + t_col];
        __syncthreads();

        for(int j = 0; j < BLOCK_SIZE; j++) {
            tmp += sA[t_row][j] * sB[j][t_col];
        }
        __syncthreads();
        a += BLOCK_SIZE;
        b += BLOCK_SIZE * N;
    } 
    c[t_row * N + t_col] = tmp;
}
```



# reduce sum
/*
input
*/