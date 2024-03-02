#include <stdio.h>

cudaError_t ErrorCheck(cudaError_t error_code, const char* filename, int linenum){
    if (error_code != cudaSuccess){
        printf("Cuda_Error:\ncode:%d, name:%s, description:%s\n file:%s, line:%d\n ",
        error_code,cudaGetErrorName(error_code),cudaGetErrorString(error_code),filename,linenum);
    }
    return error_code;
}
