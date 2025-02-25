# 关于 block reduce
以 reduce sum 为例：
```c++
#define WARP_SIZE 32

template<int REDUCE_SIZE>
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for(int mask = REDUCE_SIZE >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val; // warp 内所有线程都得到了最大值
}

template<int THREAD_PER_BLOCK>
__device__ float block_reduce(float val){
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    constexpr int NUM_WARP = THREAD_PER_BLOCK / WARP_SIZE;

    __shared__ float smem[THREAD_PER_BLOCK / WARP_SIZE];

    val = warp_reduce_sum(val);
    if (lane_id == 0) smem[warp_id] = val;
    __syncthreads();

    // 将 smem 存储的局部 sum or max值再重新放入每个 warp 中
    val = lane_id < NUM_WARP ? smem[lane_id] : 0;
    val = warp_reduce_sum(val); //每个 warp 内部再做一次 reduce

    return val; //最后 block 中，所有线程都得到了正确的结果
}
```

在一些场景中，会将 smem 存储的局部 sum or max 值再重新放入第一个 warp 中，然后第一个 warp 做 reduce 操作，这样 block 中只有第一个 warp 的线程的得到了答案。