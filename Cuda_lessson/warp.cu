#include <stdio.h>
#include "tools/setDevice.cuh"

const unsigned width = 8;
const unsigned block_size = 16; // 实际创建单个线程块具有16个线程
const unsigned full_mask = 0xffffffff; //32位全1的二进制掩码的16进制表示，使用0x代表16进制数。注意，第n个线程对应第n个二进制位。

void __global__ test_warp_primitives(void);

int main(int argc, char **argv){
    test_warp_primitives<<<1,block_size>>>();
    ErrorCheck(cudaDeviceSynchronize(),__FILE__,__LINE__); //同步
    return 0;   
}

void __global__ test_warp_primitives(void){
    int tid = threadIdx.x;
    int lane_id = tid & (width -1); //逻辑上线程束中线程为0-7,即线程束大小为8

    // 输出线程数量
    if (tid==0) {printf("threadIdx.x:");}
    printf("%2d ",tid);
    if (tid==0){ printf("\n");}
    // 输出线程在线程束内id
    if (tid==0) {printf("lane_id:");}
    printf("%2d ",lane_id);
    if (tid==0){ printf("\n");}

    unsigned mask1 = __ballot_sync(full_mask,tid>0); //ballot_sync(mask,predict), 对0线程，tid>0=0，predict为0的线程被排除，故mask1为排除0后的mask
    unsigned mask2 = __ballot_sync(full_mask,tid == 0); // 仅保留 tid0
    if(tid ==0 ){printf("full_mask = %x\n",full_mask);} //full_mask,掩码全1
    if(tid == 1) printf("mask1 = %x\n",mask1); 
    if(tid == 0) printf("mask2 = %x\n",mask2);

    int result = __all_sync(full_mask,tid); //tid0 predict 值为0 故result=0
    if(tid == 0) printf("all_sync(full_mask) = %d\n",result);

    result = __all_sync(mask1,tid); // ? 有问题
    if (tid==1) printf("all_sync(mask1) = %d\n",result);

    result = __any_sync(full_mask,tid);
    if (tid == 0) printf("any_sync(full_mask) = %d\n",result);

    result = __any_sync(mask2,tid); // ?同样有问题
    if(tid==0) printf("any_sync(mask2) = %d\n",result);

    int value = __shfl_sync(full_mask,tid,2,width); // 线程束洗牌函数作用在各个迷你线程束中。
    if (tid==0) printf("shfl:");
    printf("%2d ",value);
    if (tid==0) printf("\n");


    value = __shfl_up_sync(full_mask,tid,1,width);
    if (tid==0) printf("shfl_up:");
    printf("%2d ",value);
    if (tid==0) printf("\n");


    value = __shfl_down_sync(full_mask,tid,1,width);
    if (tid==0) printf("shfl_down:");
    printf("%2d ",value);
    if (tid==0) printf("\n");


    value = __shfl_xor_sync(full_mask,tid,1,width);
    if (tid==0) printf("shfl_xor:");
    printf("%2d ",value);
    if (tid==0) printf("\n");
}