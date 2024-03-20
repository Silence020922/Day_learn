#include <stdio.h> // 头文件
# include "cudaError.cuh"
# include <cooperative_groups.h>
using namespace cooperative_groups;

int SetGPU(){
    int iDeviceCount=0;
    cudaError_t error =ErrorCheck(cudaGetDeviceCount(&iDeviceCount),__FILE__,__LINE__); // 获取GPU数量，成功返回cudaSuccess
    if (error != cudaSuccess || iDeviceCount == 0){
        printf("No CUDA camptable GPU found! \n");
        exit(-1);
    }
    else{
        printf("The count of GPUs is %d. \n",iDeviceCount);
    }

    int iDev = 0;
    error = ErrorCheck(cudaSetDevice(iDev),__FILE__,__LINE__); // host 设置GPU
    if (error != cudaSuccess){
        printf("Fail to set GPU %d for computing. \n",iDev);
        exit(-1);
    }
    else{
        printf("Set GPU successfully. \n");
    }
    return 0;
}