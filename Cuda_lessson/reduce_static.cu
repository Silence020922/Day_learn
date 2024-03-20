#include <stdio.h>
#include "./tools/setDevice.cuh"

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

const int iElemCount = 1e8;                     // 设置元素数量
const unsigned blocksize = 128;
const unsigned gridsize = 10240;
__device__ real static_y[gridsize];

__global__ void reduce_gpu(const real *d_x, real *d_y, const int N){ // 由于每个线程块都会返回一个值，需要存储到数组内
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    extern __shared__ real s_y[];

    real y = 0.0;
    const int stride = blockDim.x*gridDim.x; // 间隔
    for (int n = bid*blockDim.x+tid; n<N;n+=stride){
        y += d_x[n]; 
    }

    s_y[tid] = y; // 将每个线程中寄存器存储的y存放到共享内存中
    __syncthreads();

    for (int offset = blockDim.x >> 1;offset >= 32;offset >>= 1){ // 使用位操作，相当于 offset /= 2
        if (tid < offset){
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads(); // 块内同步
    }

    y = s_y[tid]; //每个线程在寄存器上存储一个y

    thread_block_tile<32> g = tiled_partition<32>(this_thread_block());

    for (int i = g.size()>>1 ;i > 0;i >>= 1){ 
        y += g.shfl_down(y,i);
    }

    if (tid == 0){
        d_y[bid] = y;
    }
}
real reduce(const real *d_x){
    real *d_y;
    ErrorCheck(cudaGetSymbolAddress((void **)&d_y,static_y),__FILE__,__LINE__); // 将d_y 与 静态全局变量联系起来
    const int smem = sizeof(real) * blocksize;

    real h_y[1] = {0};

    reduce_gpu<<<gridsize,blocksize,smem>>>(d_x,d_y,iElemCount);
    reduce_gpu<<<1,1024,sizeof(real)*1024>>>(d_y,d_y,gridsize);

    ErrorCheck(cudaMemcpy(h_y,d_y,sizeof(real),cudaMemcpyDeviceToHost),__FILE__,__LINE__);

    return h_y[0];
}


void initialData(real *addr, int const elemCount)
{
    for (int i = 0; i < elemCount; i++)
    {
        addr[i] = 1.23; 
    }
    return;
}


int main(){

    SetGPU();
    // 1、分配主机内存，并初始化

    size_t stBytesCount = iElemCount * sizeof(real); // 字节数
    
    real *H_x;
    H_x = (real *)malloc(stBytesCount); // 分配动态内存

    if (H_x != NULL)
    {
        memset(H_x, 0, stBytesCount);  // 主机内存初始化为0
    }
    else
    {
        printf("Fail to allocate host memory!\n");
        exit(-1);
    }


    real *D_x;
    ErrorCheck(cudaMalloc((real**)&D_x, stBytesCount), __FILE__, __LINE__);
    if (D_x != NULL){
    ErrorCheck(cudaMemset(D_x, 0, stBytesCount), __FILE__, __LINE__); // 设备内存初始化为0
    }
    else{
        printf("fail to allocate memory\n");
        free(H_x); // 释放先前CPU中制定的内存
        exit(-1);
    }
    // 初始化主机数据并copy到设备内存
    initialData(H_x, iElemCount);
    ErrorCheck(cudaMemcpy(D_x,H_x,stBytesCount,cudaMemcpyHostToDevice),__FILE__,__LINE__);
    // 启动记时函数
    cudaEvent_t start, stop; 
    ErrorCheck(cudaEventCreate(&start), __FILE__, __LINE__);
    ErrorCheck(cudaEventCreate(&stop), __FILE__, __LINE__);
    ErrorCheck(cudaEventRecord(start), __FILE__, __LINE__);
    cudaEventQuery(start);

    real ans = reduce(D_x);
    // 将计算结果传回到主机

    ErrorCheck(cudaEventRecord(stop), __FILE__, __LINE__);
    ErrorCheck(cudaEventSynchronize(stop), __FILE__, __LINE__);
    float elapsed_time;
    ErrorCheck(cudaEventElapsedTime(&elapsed_time, start, stop), __FILE__, __LINE__);
    printf("Time = %g ms.\n", elapsed_time); // 打印时间

    // 释放主机与设备内存
    // 4 打印结果
    printf("Sum: %.6f\t\n",ans);
    // 释放主机内存，结束程序
    free(H_x);
    ErrorCheck(cudaFree(D_x), __FILE__, __LINE__);
    ErrorCheck(cudaDeviceReset(), __FILE__, __LINE__);

    return 0;
}

