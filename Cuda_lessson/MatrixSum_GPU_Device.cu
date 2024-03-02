#include <stdio.h>
#include "./tools/setDevice.cuh"

// 声明设备函数
__device__ float add(const float a, const float b){
    return a+b;
}
// 设计核函数
__global__ void addFromGPU(float *A,float *B, float *C,const int N){
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int id = tid + bid * blockDim.x; 

    if (id >= N){return ;}
    C[id] = add(A[id] ,B[id]);
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

    int iElemCount = 512;                     // 设置元素数量
    size_t stBytesCount = iElemCount * sizeof(float); // 字节数
    // 分配主机内存
    float *fpHost_A, *fpHost_B, *fpHost_C;
    fpHost_A = (float *)malloc(stBytesCount); // 分配动态内存
    fpHost_B = (float *)malloc(stBytesCount);
    fpHost_C = (float *)malloc(stBytesCount);
    if (fpHost_A != NULL && fpHost_B != NULL && fpHost_C != NULL)
    {
        memset(fpHost_A, 0, stBytesCount);  // 主机内存初始化为0
        memset(fpHost_B, 0, stBytesCount);
        memset(fpHost_C, 0, stBytesCount);
    }
    else
    {
        printf("Fail to allocate host memory!\n");
        exit(-1);
    }

    float *fpDevice_A, *fpDevice_B, *fpDevice_C;
    cudaMalloc((float **)&fpDevice_A,stBytesCount);
    cudaMalloc((float **)&fpDevice_B,stBytesCount);
    cudaMalloc((float **)&fpDevice_C,stBytesCount);
    if (fpDevice_A != NULL && fpDevice_B != NULL && fpDevice_C != NULL){
        cudaMemset(fpDevice_A,0,stBytesCount);
        cudaMemset(fpDevice_B,0,stBytesCount);
        cudaMemset(fpDevice_C,0,stBytesCount);
    }
    else{
        printf("fail to allocate memory\n");
        free(fpHost_A); // 释放先前CPU中制定的内存
        free(fpHost_B);
        free(fpHost_C);
        exit(-1);
    }

     // 初始化主机中数据
    srand(666); // 设置随机种子
    initialData(fpHost_A, iElemCount);
    initialData(fpHost_B, iElemCount);
    // 主机复制到设备
    cudaMemcpy(fpDevice_A,fpHost_A,stBytesCount,cudaMemcpyHostToDevice);
    cudaMemcpy(fpDevice_B,fpHost_B,stBytesCount,cudaMemcpyHostToDevice);
    cudaMemcpy(fpDevice_C,fpHost_C,stBytesCount,cudaMemcpyHostToDevice);
   
    // 调用核函数
    dim3 block(32);
    dim3 grid((iElemCount - 1)/ 32 + 1); // 取上整
    addFromGPU<<<grid, block>>>(fpDevice_A, fpDevice_B, fpDevice_C, iElemCount); 

    // 将计算得到的数据从设备传给主机
    cudaMemcpy(fpHost_C, fpDevice_C, stBytesCount, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; i++)    // 打印
    {
        printf("idx=%2d\tmatrix_A:%.2f\tmatrix_B:%.2f\tresult=%.2f\n", i+1, fpHost_A[i], fpHost_B[i], fpHost_C[i]);
    }

    // 释放主机与设备内存
    free(fpHost_A);
    free(fpHost_B);
    free(fpHost_C);
    cudaFree(fpDevice_A);
    cudaFree(fpDevice_B);
    cudaFree(fpDevice_C);

    cudaDeviceReset();
    return 0;
}