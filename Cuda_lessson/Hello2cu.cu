#include <stdio.h>

__global__ void hello_world2gpu(){
    printf("Hello world, from GPU.\n");
}

int main(void){
    hello_world2gpu<<<1,1>>>(); // <<<grid_size,block_size>>>配置线程 线程块数量-单个块线程数量
    cudaDeviceSynchronize(); // 同步函数
    return 0;
}

// 内建变量 gridDim.x blockDim.x blockIdx.x threadIdx.x  线程具有唯一标识
// nvcc xxx.cu -o helloworld -arch=compute_XX -code=sm_XX 指定虚拟计算能力和真实计算能力 