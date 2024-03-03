#include <stdio.h>
#include "./tools/setDevice.cuh"

int main(){
    // 分配主机内存并初始化
    float *fphostA;
    fphostA = (float *)malloc(4);
    memset(fphostA,0,4);

    // 设备内存并初始化
    float *fpDeviceA;
    ErrorCheck(cudaMalloc((float **)&fpDeviceA,4),__FILE__,__LINE__);
    ErrorCheck(cudaMemset(fpDeviceA,0,4),__FILE__,__LINE__);

    // 数据从主机复制到设备
    ErrorCheck(cudaMemcpy(fpDeviceA,fphostA,4,cudaMemcpyDeviceToHost),__FILE__,__LINE__); //但定义传输方向为Device to Host

    // 释放主机和设备内存
    free(fphostA);
    ErrorCheck(cudaFree(fpDeviceA),__FILE__,__LINE__);

    // 重置设备设置
    ErrorCheck(cudaDeviceReset(),__FILE__,__LINE__);

    return 0;
}
