#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

class KernelAdd
{
public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                uint32_t inputDataNum, uint32_t tileDataNum,
                                uint32_t tailDataNum, uint32_t tileNum)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        // Set Value
        this->inputDataNum = inputDataNum;
        this->tileDataNum = tileDataNum/BUFFER_NUM;
        this->tailDataNum = tailDataNum;
        this->tileNum = tileNum;

        // Set GM
        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x, this->inputDataNum);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y, this->inputDataNum);
        zGm.SetGlobalBuffer((__gm__ DTYPE_Z *)z, this->inputDataNum);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(DTYPE_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileDataNum * sizeof(DTYPE_Y));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileDataNum * sizeof(DTYPE_Z));
    }

    __aicore__ inline void Process()
    {

        int32_t loopCount = this->tileNum * BUFFER_NUM;
        this->processDataNum = this->tileDataNum;
        for (int32_t i = 0; i < loopCount; i++)
        {   if ( i == loopCount - 2){
	    this->processDataNum = (this->tailDataNum > this->tileDataNum)? this->tileDataNum: this->tailDataNum;
		}
            if (i == loopCount - 1)
            {
                this->processDataNum = (this->tailDataNum > this->tileDataNum) ? this->tailDataNum - this->tileDataNum: 0;
		if (this->processDataNum == 0) break;
            }
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        LocalTensor<DTYPE_Y> yLocal = inQueueY.AllocTensor<DTYPE_Y>();
        DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
        DataCopy(yLocal, yGm[progress * this->tileDataNum], this->processDataNum);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        LocalTensor<DTYPE_Y> yLocal = inQueueY.DeQue<DTYPE_Y>();
        LocalTensor<DTYPE_Z> zLocal = outQueueZ.AllocTensor<DTYPE_Z>();
        Add(zLocal, xLocal, yLocal, this->processDataNum);
        outQueueZ.EnQue<DTYPE_Z>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<DTYPE_Z> zLocal = outQueueZ.DeQue<DTYPE_Z>();
        DataCopy(zGm[progress * this->tileDataNum], zLocal, this->processDataNum);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    // Pipe
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<DTYPE_Y> yGm;
    GlobalTensor<DTYPE_Z> zGm;

    uint32_t inputDataNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t tileNum;
    uint32_t processDataNum;
};


extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KernelAdd op;
    op.Init(x, y, z, tiling_data.inputNum, tiling_data.tileDataNum, tiling_data.tailDataNum, tiling_data.tileNum);
    op.Process();
}
