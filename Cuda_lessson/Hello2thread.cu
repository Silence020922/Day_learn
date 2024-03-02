#include <stdio.h>   

__global__ void hello_world2gpu(){
    const int block_id = blockIdx.x;
    const int thread_id = threadIdx.x;
    const int id = block_id * blockDim.x + thread_id;
    printf("Hello world, from block %d, thread_id %d, global_id %d.\n",block_id,thread_id,id);
}

int main(void){
    hello_world2gpu<<<2,2>>>(); // <<<grid_size,block_size>>>配置线程 线程块数量-单个块线程数量
    cudaDeviceSynchronize(); // 同步函数
    return 0;
}

// 定义多维网格和线程块 dim3 grid_size( , , ) dim3 block_size( , , )
// 4060 真实架构计算能力8.0