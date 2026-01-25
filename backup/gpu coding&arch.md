# gemm
```c++
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

# define BLOCKSIZE 16

__global__ void gemm_kernel_tile(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ c,
                                  int M, int N, int K){
                                
                                int y = blockIdx.y * blockDim.y + threadIdx.y;
                                int x = blockIdx.x * blockDim.x + threadIdx.x;
                                
                                __shared__ float a_tile[BLOCKSIZE][BLOCKSIZE], b_tile[BLOCKSIZE][BLOCKSIZE];
                                
                                float sum = 0.0f;

                                for(int i=0;i<K/BLOCKSIZE;++i){
                                    // load to sahared memory a
                                    int ax = threadIdx.x + i * BLOCKSIZE;
                                    int ay = y;
                                    if(ax < K && ay < M){
                                        a_tile[threadIdx.y][threadIdx.x] = a[ay * K + ax];   // ⭐️ 防止共享内存坐标越界
                                    }else{
                                        a_tile[threadIdx.y][threadIdx.x] = 0.0f;
                                    }
                                    // load to sahared memory b
                                    int bx = threadIdx.x;
                                    int by = threadIdx.y + i * BLOCKSIZE;
                                    if(bx < N && by < K){
                                        b_tile[threadIdx.y][threadIdx.x] = b[by * N + bx];
                                    }else{
                                        b_tile[threadIdx.y][threadIdx.x] = 0.0f;
                                    }

                                    __syncthreads();  // ⭐️ 数据加载完成之后同步

                                    
                                    for(int i=0;i<BLOCKSIZE;++i){
                                        sum += a_tile[threadIdx.y][i] * b_tile[i][threadIdx.x];
                                    }
                                    __syncthreads();  // ⭐️ 同步，防止提前进入下一轮计算然后累加错误
                                }
                                if(x < N && y < M){
                                    c[y * N + x] = sum;
                                }
                            }
                                    


__global__ void gemm_kernel(const float* __restrict__ a,
                            const float* __restrict__ b,
                            float* __restrict__ c,
                            int M, int N, int K) {
    // Each thread computes one element of C
    int row = blockIdx.y * blockDim.y + threadIdx.y; // y -> row in C (0..M-1)
    int col = blockIdx.x * blockDim.x + threadIdx.x; // x -> col in C (0..N-1)

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            // A[row][k] * B[k][col]
            sum += a[row * K + k] * b[k * N + col];
        }
        c[row * N + col] = sum;
    }
}

int main() {
    // Matrix dimensions: A(MxK) * B(KxN) = C(MxN)
    const int M = 5000;
    const int K = 6000;
    const int N = 4000;

    const size_t size_a = M * K * sizeof(float);
    const size_t size_b = K * N * sizeof(float);
    const size_t size_c = M * N * sizeof(float);

    // Host memory allocation
    float *h_a = (float*)malloc(size_a);
    float *h_b = (float*)malloc(size_b);
    float *h_c = (float*)malloc(size_c);
    float *h_c_ref = (float*)malloc(size_c); // Optional: CPU reference

    // Initialize host matrices
    for (int i = 0; i < M * K; ++i) h_a[i] = 1.0f; // A all 1s
    for (int i = 0; i < K * N; ++i) h_b[i] = 2.0f; // B all 2s
    for (int i = 0; i < M * N; ++i) h_c[i] = 0.0f; // Initialize to 0

    // Device memory allocation
    float *d_a, *d_b, *d_c;
    cudaError_t err;

    err = cudaMalloc(&d_a, size_a);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_a failed: %s\n", cudaGetErrorString(err)); return 1; }

    err = cudaMalloc(&d_b, size_b);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_b failed: %s\n", cudaGetErrorString(err)); return 1; }

    err = cudaMalloc(&d_c, size_c);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_c failed: %s\n", cudaGetErrorString(err)); return 1; }

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, size_c, cudaMemcpyHostToDevice);

    // Kernel launch configuration
    dim3 blockSize(16, 16); // 256 threads per block
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                  (M + blockSize.y - 1) / blockSize.y);

    // Launch kernel
    gemm_kernel_tile<<<gridSize, blockSize>>>(d_a, d_b, d_c, M, N, K);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost);

    printf("First 10 elements of C (row 0, columns 0～9):\n");
    for (int i = 0; i < 10 && i < N; ++i) {
        printf("C[0][%d] = %.2f\n", i, h_c[i]);
    }
    printf("\n");

    // Optional: Verify result (C should be all 2*K = 400.0f)
    bool correct = true;
    float expected = 2.0f * K; // since A=1, B=2, sum over K terms: 1*2*K
    for (int i = 0; i < M * N; ++i) {
        if (abs(h_c[i] - expected) > 1e-5) {
            correct = false;
            break;
        }
    }

    printf("Matrix multiplication result: %s\n", correct ? "PASSED" : "FAILED");
    if (!correct) {
        printf("Example: h_c[0] = %f, expected = %f\n", h_c[0], expected);
    }

    // Cleanup
    free(h_a); free(h_b); free(h_c); free(h_c_ref);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    // no
    return 0;
}

```

# transpose
```c++
# include <stdio.h>
# include <math.h>

#define BLOCK_SIZE 32
#define M 3000
#define N 1000

__managed__ int matrix[N][M];
__managed__ int gpu_result[M][N];
__managed__ int cpu_result[M][N];

__global__ void gpu_matrix_transpose(int in[N][M], int out[M][N])
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if( x < M && y < N)
    {
        out[x][y] = in[y][x];
    }
}

// 创建m行，n列的线程数量【由多个线程块组成的】
__global__ void gpu_shared_matrix_transpose(int in[N][M], int out[M][N])
{

    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int x = threadIdx.x + blockDim.x * blockIdx.x;

    __shared__ int ken[BLOCK_SIZE+1][BLOCK_SIZE+1];//ken[32] warp

    // step1：
    if(x < M && y < N)
    {   
        // step1：读到共享内存
        ken[threadIdx.y][threadIdx.x] = in[y][x];
    }
    __syncthreads();

    // 原则：相邻的线程访问相邻的坐标

    // step2：  块反转，块内坐标不变
    int x1 = threadIdx.x + blockDim.y * blockIdx.y;
    int y1 = threadIdx.y + blockDim.x * blockIdx.x;
    
    if(x1 < N && y1 < M)
    {
    // step3：从共享内存读到输出数据
        out[y1][x1] = ken[threadIdx.x][threadIdx.y];//32 bank
    }

}

void cpu_matrix_transpose(int in[N][M], int out[M][N])
{
    for(int y = 0; y < N; y++)
    {
        for(int x = 0; x < M; x++)
        {
            out[x][y] = in[y][x];
        }
    }
}

int main()
{
    for(int y=0; y<N; y++)
    {
        for(int x=0; x<M; x++)
        {
            matrix[y][x] = rand()%1024;
        }
    }

    cudaEvent_t start, stop_gpu, stop_cpu;
    cudaEventCreate(&start);
    cudaEventCreate(&stop_cpu);
    cudaEventCreate(&stop_gpu);

    cudaEventRecord(start);
    cudaEventSynchronize(start);

    dim3 dimGrid((M + BLOCK_SIZE - 1)/BLOCK_SIZE, (N + BLOCK_SIZE -1)/BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    for(int i = 0; i < 20; i++)
    {
        // gpu_matrix_transpose<<<dimGrid,dimBlock>>>(matrix, gpu_result);
        gpu_shared_matrix_transpose<<<dimGrid,dimBlock>>>(matrix, gpu_result);
        cudaDeviceSynchronize();
    }

    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);

    cpu_matrix_transpose(matrix, cpu_result);

    cudaEventRecord(stop_cpu);
    cudaEventSynchronize(stop_cpu);

    float time_cpu, time_gpu;
    cudaEventElapsedTime(&time_gpu, start, stop_gpu);
    cudaEventElapsedTime(&time_cpu, stop_gpu, stop_cpu);

    bool errors = false;
    for(int y = 0; y<M; y++)
    {
        for (int x = 0; x < N; x++)
        {
            if(fabs(cpu_result[y][x] - gpu_result[y][x]) > (1.0e-10))
            {
                errors = true;
            }
        }
        
    }

    printf("Result: %s\n", errors?"Error":"Pass");
    printf("CPU time: %.2f\nGPU time: %.2f\n", time_cpu, time_gpu/20.0);

    return 0;
}
```