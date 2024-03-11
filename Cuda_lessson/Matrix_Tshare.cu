#include <stdio.h>
#include "./tools/setDevice.cuh"

// 设计核函数
__global__ void transpose1(const float *A,float *B, const int N){
    const int nx = threadIdx.x + blockDim.x*blockIdx.x;
    const int ny = threadIdx.y + blockDim.y*blockIdx.y;
    __shared__ float S[32][33];
    
    if (nx < N && ny < N){
        S[threadIdx.y][threadIdx.x] = A[ny*N + nx]; // 合并访问，copy操作
    }
    __syncthreads();

    // const int nx2 = threadIdx.x + blockDim.x*blockIdx.x;
    // const int ny2 = threadIdx.y + blockDim.y*blockIdx.y;
    // if (nx2 < N && ny2< N){
    //     B[nx2*N + ny2] = S[ny2][nx2]; // 
    // }
    const int nx2 = threadIdx.y + blockDim.x*blockIdx.x;
    const int ny2 = threadIdx.x + blockDim.y*blockIdx.y;
    if (nx2 < N && ny2< N){
        B[nx2*N + ny2] = S[threadIdx.x][threadIdx.y]; // 调换threadIdx.x 和 threadIdx.y 使得对B的访问也是合并的 
    }
}

void initialData(float *addr, int elemCount)
{
    for (int i = 0; i < elemCount; i++)
    {
        addr[i] = (float)(rand() & 0xFF) / 10.f; // 做逻辑与操作，限制随机数不超过255/10
    }
    return;
}


int main(void){
    SetGPU(); // 设置GPU
    const int N = 128; // 设置矩阵大小
    int iElemCount = N*N;             // 设置元素数量
    size_t stBytesCount = iElemCount * sizeof(float); // 字节数
    // 分配主机内存
    float *fpHost_A, *fpHost_B;
    fpHost_A = (float *)malloc(stBytesCount); // 分配动态内存
    fpHost_B = (float *)malloc(stBytesCount);
    if (fpHost_A != NULL && fpHost_B != NULL)
    {
        memset(fpHost_A, 0, stBytesCount);  // 主机内存初始化为0
        memset(fpHost_B, 0, stBytesCount);
    }
    else
    {
        printf("Fail to allocate host memory!\n");
        exit(-1);
    }

    float *fpDevice_A, *fpDevice_B;
    ErrorCheck(cudaMalloc((float**)&fpDevice_A, stBytesCount), __FILE__, __LINE__);
    ErrorCheck(cudaMalloc((float**)&fpDevice_B, stBytesCount), __FILE__, __LINE__);
    if (fpDevice_A != NULL && fpDevice_B != NULL){
    ErrorCheck(cudaMemset(fpDevice_A, 0, stBytesCount), __FILE__, __LINE__); // 设备内存初始化为0
    ErrorCheck(cudaMemset(fpDevice_B, 0, stBytesCount), __FILE__, __LINE__);
    }
    else{
        printf("fail to allocate memory\n");
        free(fpHost_A); // 释放先前CPU中制定的内存
        free(fpHost_B);
        exit(-1);
    }

     // 初始化主机中数据
    srand(666); // 设置随机种子
    initialData(fpHost_A, iElemCount);
    // 主机复制到设备
    ErrorCheck(cudaMemcpy(fpDevice_A,fpHost_A,stBytesCount,cudaMemcpyHostToDevice),__FILE__,__LINE__);
    ErrorCheck(cudaMemcpy(fpDevice_B,fpHost_B,stBytesCount,cudaMemcpyHostToDevice),__FILE__,__LINE__);
   
    // 调用核函数
    const dim3 block(32,32); // 设置block大小为2维32, 32
    dim3 grid((N-1)/32 + 1,(N-1)/32 + 1); //向上取整，设置grid大小
    // 设置开始事件
    cudaEvent_t start, stop; 
    ErrorCheck(cudaEventCreate(&start), __FILE__, __LINE__);
    ErrorCheck(cudaEventCreate(&stop), __FILE__, __LINE__);
    ErrorCheck(cudaEventRecord(start), __FILE__, __LINE__);
    cudaEventQuery(start);

    transpose1<<<grid, block>>>(fpDevice_A, fpDevice_B,N);  //调用核函数

    ErrorCheck(cudaEventRecord(stop), __FILE__, __LINE__);
    ErrorCheck(cudaEventSynchronize(stop), __FILE__, __LINE__);
    float elapsed_time;
    ErrorCheck(cudaEventElapsedTime(&elapsed_time, start, stop), __FILE__, __LINE__);
    printf("Time = %g ms.\n", elapsed_time); // 打印时间
    // 将计算得到的数据从设备传给主机
    ErrorCheck(cudaMemcpy(fpHost_B, fpDevice_B, stBytesCount, cudaMemcpyDeviceToHost),__FILE__,__LINE__);
    // 释放主机与设备内存
    free(fpHost_A);
    free(fpHost_B);
    ErrorCheck(cudaFree(fpDevice_A), __FILE__, __LINE__);
    ErrorCheck(cudaFree(fpDevice_B), __FILE__, __LINE__);
    ErrorCheck(cudaDeviceReset(), __FILE__, __LINE__);
    return 0;
}