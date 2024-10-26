#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
class KernelAddBroadCast{
public:
    __aicore__ inline KernelAddBroadCast() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                uint32_t inputDataNum, uint32_t tileDataNum,
                                uint32_t tailDataNum, uint32_t tileNum,int32_t dimn,
                                int32_t* xnarray, int32_t* ynarray, int32_t* znarray,
                                int32_t* xsumnarray, int32_t* ysumnarray,
                                int32_t *zsumnarray)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

        // Set Value
        this->inputDataNum = inputDataNum;
        this->tileDataNum = tileDataNum/BUFFER_NUM;
        this->tailDataNum = tailDataNum;
        this->tileNum = tileNum;
        this->dimn = dimn;
        this->xnarray = xnarray;
        this->ynarray = ynarray;
        this->znarray = znarray;
        this->xsumnarray = xsumnarray;
        this->ysumnarray = ysumnarray;
        this->zsumnarray = zsumnarray;

        // Set GM
        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x, 1);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y, 1);
        zGm.SetGlobalBuffer((__gm__ DTYPE_Z *)z, 1);
    }
    __aicore__ inline void Process(){
    for (int i =0;i<this->zsumnarray[this->dimn];i++){
        int32_t x_start = 0;
        int32_t y_start = 0;
        for (int k =0;k<this->dimn;k++){
            if (xnarray[k] != 1){
                x_start += xsumnarray[k]*((i / zsumnarray[k]) % znarray[k]);
            }
            if (ynarray[k] != 1){
                y_start += ysumnarray[k]*((i / zsumnarray[k]) % znarray[k]);
            } 
        }
        float z = (float)xGm.GetValue(x_start) + (float)yGm.GetValue(y_start);
        zGm.SetValue(i,(DTYPE_Z) z);
    }
    }
private:
    //GM
    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<DTYPE_Y> yGm;
    GlobalTensor<DTYPE_Z> zGm;

    uint32_t inputDataNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t tileNum;
    uint32_t processDataNum;
    int32_t dimn;
    int32_t* xnarray;
    int32_t* ynarray;
    int32_t* znarray;
    int32_t* xsumnarray;
    int32_t* ysumnarray;
    int32_t* zsumnarray;
};

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

    if (TILING_KEY_IS(1)){
    KernelAdd op;
    op.Init(x, y, z, tiling_data.inputNum, tiling_data.tileDataNum, tiling_data.tailDataNum, tiling_data.tileNum);
    op.Process();
    }else if (TILING_KEY_IS(2)){
    KernelAddBroadCast op;
    op.Init(x, y, z, tiling_data.inputNum, tiling_data.tileDataNum, tiling_data.tailDataNum, tiling_data.tileNum, 
    tiling_data.zdimn,tiling_data.x_narray,tiling_data.y_narray,tiling_data.z_narray,
    tiling_data.x_sumnarray,tiling_data.y_sumnarray,tiling_data.z_sumnarray);
    op.Process();
    }
}
