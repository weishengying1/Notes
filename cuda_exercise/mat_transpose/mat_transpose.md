
# 朴素实现
矩阵转置的朴素实现非常直观, 思路即使用二维的线程/线程块排布, 让每个线程负责一个矩阵元素的转置. 实现上, 只需要将矩阵的行列索引 x y 进行反转即可.
需要注意的是 grid 和 block 的中维度设置与多维数组中的表示是相反的, 即 grid.x 应该对应 N 维度, grid.y 应该对应 M 维度.

```cpp
/*
input: (M, N)
dim3 gridDim(ceil(M/Tile_Size), ceil(N/Tile_size))
dim3 blockDim(Tile_size, Tile_size)
*/

#define Tile_Size 16
__global__ void transpose_naive(const float* A, float* B, int M, int N) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if(idx < N && row < idy) {
        B[idx * M + idy] = A[idy * N + idx];
    }
}
```

# 合并内存访问
```c++
#define Tile_Size 16
__global__ void transpose_naive(const float* A, float* B, int M, int N) {
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int bx = blockIdx.x, by = blockIdx.y;

    const int x = bx * Tile_Size + tx;
    const int y = by * Tile_Size + ty;

    __shared__ float smem[Tile_Size][Tile_Size];
    if(x < N && y < M) {
        smem[ty][tx] = A[y * N + x]; //(by * Tile_Size + ty) * N + bx * Tile_Size + tx
    }
    __sythreads();

    x = by * Tile_Size + tx;
    y = bx * Tile_Size + ty;
    if(x < M && y < N) {
        B[y * M + x] = smem[tx][ty]
    }
}

#define Tile_Size 16
__global__ void transpose_naive(const float* A, float* B, int M, int N) {
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int bx = blockIdx.x, by = blockIdx.y;

    // 计算每个 block 转置之前负责区域的起始地址
    A += by * Tile_Size * N + bx * Tile_Size; // y 行 x 列

    __shared__ float smem[Tile_Size][Tile_Size];
    if(idx < N && row < idy) {
        smem[ty][tx] = A[ty * N + tx];
    }

    // 计算转置后每个 block 转置之前负责区域的起始地址
    B += bx * Tile_Size * M + by * Tile_Size; // x 行 y 列
    if(idx < N && row < idy) {
        B[ty * M + tx] = smem[tx][ty] //完成转置
    }
}
```