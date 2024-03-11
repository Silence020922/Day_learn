#include <stdio.h>
#include "./tools/setDevice.cuh"

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

real reduce_CPU(real *x, const int N){ //CPU中执行规约计算
    real sum = 0.0;
    for (int i = 0;i<N;i++){
        sum += x[i];
    }
    return sum;
}

__global__ void reduce_gpu(real *d_x, real *d_y, const int N){ // 由于每个线程块都会返回一个值，需要存储到数组内
    const int tid = threadIdx.x;
    const int n = blockDim.x*blockIdx.x + threadIdx.x;
    
    __shared__ real s_y[128]; 
    s_y[tid] = (n<N)? d_x[n]:0.0; //对每个线程块的共享内存赋不同的值。
    __syncthreads();

    for (int offset = blockDim.x >> 1;offset > 0;offset >>= 1){ // 使用位操作，相当于 offset /= 2
        if (tid < offset){
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads(); // 块内同步
    }
    if (tid == 0){
        d_y[blockIdx.x] = s_y[0];
    }
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
    const int iElemCount = 1e8;                     // 设置元素数量
    int blocksize = 128;
    int gridsize = (iElemCount-1)/blocksize + 1;
    size_t stBytesCount = iElemCount * sizeof(real); // 字节数
    
    real *H_x, *H_y;
    H_x = (real *)malloc(stBytesCount); // 分配动态内存
    H_y = (real *)malloc(gridsize * sizeof(real));


    if (H_x != NULL && H_y != NULL)
    {
        memset(H_x, 0, stBytesCount);  // 主机内存初始化为0
        memset(H_y,0,gridsize*sizeof(real));
    }
    else
    {
        printf("Fail to allocate host memory!\n");
        exit(-1);
    }
    real *D_x, *D_y;
    ErrorCheck(cudaMalloc((real**)&D_x, stBytesCount), __FILE__, __LINE__);
    ErrorCheck(cudaMalloc((real**)&D_y, gridsize*sizeof(real)), __FILE__, __LINE__);
    if (D_x != NULL && D_y != NULL){
    ErrorCheck(cudaMemset(D_x, 0, stBytesCount), __FILE__, __LINE__); // 设备内存初始化为0
    ErrorCheck(cudaMemset(D_y, 0, gridsize*sizeof(real)), __FILE__, __LINE__);
    }
    else{
        printf("fail to allocate memory\n");
        free(H_x); // 释放先前CPU中制定的内存
        free(H_y);
        exit(-1);
    }

    // 2、初始化主机中数据
    initialData(H_x, iElemCount);
    // 主机复制到设备
    ErrorCheck(cudaMemcpy(D_x,H_x,stBytesCount,cudaMemcpyHostToDevice),__FILE__,__LINE__);
    ErrorCheck(cudaMemcpy(D_y,H_y,gridsize*sizeof(real),cudaMemcpyHostToDevice),__FILE__,__LINE__);
    // 3 调用归约函数并记时
    cudaEvent_t start, stop; 
    ErrorCheck(cudaEventCreate(&start), __FILE__, __LINE__);
    ErrorCheck(cudaEventCreate(&stop), __FILE__, __LINE__);
    ErrorCheck(cudaEventRecord(start), __FILE__, __LINE__);
    cudaEventQuery(start);

    reduce_gpu<<<gridsize,blocksize>>>(D_x,D_y,iElemCount);
    // 将计算结果传回到主机
    ErrorCheck(cudaMemcpy(H_y, D_y, gridsize*sizeof(real), cudaMemcpyDeviceToHost),__FILE__,__LINE__);
    real sum = reduce_CPU(H_y,gridsize);
    
    ErrorCheck(cudaEventRecord(stop), __FILE__, __LINE__);
    ErrorCheck(cudaEventSynchronize(stop), __FILE__, __LINE__);
    float elapsed_time;
    ErrorCheck(cudaEventElapsedTime(&elapsed_time, start, stop), __FILE__, __LINE__);
    printf("Time = %g ms.\n", elapsed_time); // 打印时间

    // 释放主机与设备内存
    // 4 打印结果
    printf("Sum: %.6f\t\n",sum);
    // 释放主机内存，结束程序
    free(H_x);
    free(H_y);
    ErrorCheck(cudaFree(D_x), __FILE__, __LINE__);
    ErrorCheck(cudaFree(D_y), __FILE__, __LINE__);
    ErrorCheck(cudaDeviceReset(), __FILE__, __LINE__);

    return 0;
}

