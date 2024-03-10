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

    void initialData(real *addr, int const elemCount)
    {
        for (int i = 0; i < elemCount; i++)
        {
            addr[i] = 1.23; // 做逻辑与操作，限制随机数不超过255/10
        }
        return;
    }

    int main(){

        SetGPU();

        // 1、分配主机内存，并初始化
        int iElemCount = 1e8;                     // 设置元素数量
        size_t stBytesCount = iElemCount * sizeof(real); // 字节数
        
        real *d_x;
        d_x = (real *)malloc(stBytesCount); // 分配动态内存
        if (d_x != NULL)
        {
            memset(d_x, 0, stBytesCount);  // 主机内存初始化为0
        }
        else
        {
            printf("Fail to allocate host memory!\n");
            exit(-1);
        }
        

        // 2、初始化主机中数据
        initialData(d_x, iElemCount);

        // 3 调用归约函数并记时
        cudaEvent_t start, stop; 
        ErrorCheck(cudaEventCreate(&start), __FILE__, __LINE__);
        ErrorCheck(cudaEventCreate(&stop), __FILE__, __LINE__);
        ErrorCheck(cudaEventRecord(start), __FILE__, __LINE__);
        cudaEventQuery(start);
        real sum = reduce_CPU(d_x,iElemCount);
        ErrorCheck(cudaEventRecord(stop), __FILE__, __LINE__);
        ErrorCheck(cudaEventSynchronize(stop), __FILE__, __LINE__);
        float elapsed_time;
        ErrorCheck(cudaEventElapsedTime(&elapsed_time, start, stop), __FILE__, __LINE__);
        printf("Time = %g ms.\n", elapsed_time); // 打印时间
        // 4 打印结果
        printf("Ans: %.6f\t\n",sum);

        // 释放主机内存，结束程序
        free(d_x);
        ErrorCheck(cudaDeviceReset(), __FILE__, __LINE__);

        return 0;
    }

