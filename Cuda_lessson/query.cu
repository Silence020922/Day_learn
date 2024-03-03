#include "./tools/setDevice.cuh"
#include <stdio.h>

int main(void)
{
    int device_id = 0;
    ErrorCheck(cudaSetDevice(device_id), __FILE__, __LINE__); // 设置使用的GPU

    cudaDeviceProp prop; // 结构体
    ErrorCheck(cudaGetDeviceProperties(&prop, device_id), __FILE__, __LINE__);

    printf("Device id:                                 %d\n", // 设备ID
        device_id);
    printf("Device name:                               %s\n", // 设备名称
        prop.name);
    printf("Compute capability:                        %d.%d\n", // 计算能力，.major主版本号，.minor次版本号
        prop.major, prop.minor);
    printf("Amount of global memory:                   %g GB\n", // 显存
        prop.totalGlobalMem / (1024.0 * 1024 * 1024));
    printf("Amount of constant memory:                 %g KB\n", // 常量内存
        prop.totalConstMem  / 1024.0);
    printf("Maximum grid size:                         %d %d %d\n", // 最大grid数量
        prop.maxGridSize[0], 
        prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Maximum block size:                        %d %d %d\n", // 最大block数量
        prop.maxThreadsDim[0], prop.maxThreadsDim[1], 
        prop.maxThreadsDim[2]);
    printf("Number of SMs:                             %d\n", //SM(Streaming Multiprocessors) 数量
        prop.multiProcessorCount);
    printf("Maximum amount of shared memory per block: %g KB\n", //每个块共享内存
        prop.sharedMemPerBlock / 1024.0);
    printf("Maximum amount of shared memory per SM:    %g KB\n", //每个SM共享内存
        prop.sharedMemPerMultiprocessor / 1024.0);
    printf("Maximum number of registers per block:     %d K\n", // 寄存器
        prop.regsPerBlock / 1024);
    printf("Maximum number of registers per SM:        %d K\n",
        prop.regsPerMultiprocessor / 1024);
    printf("Maximum number of threads per block:       %d\n", //线程
        prop.maxThreadsPerBlock);
    printf("Maximum number of threads per SM:          %d\n",
        prop.maxThreadsPerMultiProcessor);

    return 0;
}