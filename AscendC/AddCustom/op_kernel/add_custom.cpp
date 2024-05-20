#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

class KernelAdd
{
public:
  __aicore__ inline KernelAdd() {}
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                              uint32_t blockLength, uint32_t totalLength,
                              uint32_t tileNum, uint32_t tileLength,
                              uint32_t lasttileLength, uint32_t formerNum,
                              uint32_t formerLength, uint32_t formertileNum,
                              uint32_t formertileLength,
                              uint32_t formerlasttileLength, uint32_t tailNum,
                              uint32_t tailLength, uint32_t tailtileNum,
                              uint32_t tailtileLength, uint32_t taillasttileLength,
                              uint32_t tilingKey, uint32_t y_length)
  {

    ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

    // 1:y为x最后一维 2:y为标量
    this->y_length = y_length;
    uint32_t gen_class = 0; // 泛化情况，0代表x与y维度相同
    if (totalLength > y_length)
    {
      gen_class = 1;
      if (y_length == 1)
      {
        gen_class = 2;
      }
    }
    this->gen_class = gen_class;

    if (tilingKey == 1)
    { // 对齐场景

      this->blockLength = blockLength;
      uint32_t offset = this->blockLength * GetBlockIdx();
      ASSERT(tileNum != 0 && "tile num can not be zero!");
      this->tileNum = tileNum;
      this->lasttileLength = lasttileLength;
      this->tileLength = tileLength / BUFFER_NUM;

      xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + offset,
                          this->blockLength);
      if (this->gen_class == 0)
      {
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + offset,
                            this->blockLength);
      }
      else
      {
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y, y_length);
        this->yScalar = yGm.GetValue(0); // 每一个核保留完整的y
      }
      zGm.SetGlobalBuffer((__gm__ DTYPE_Z *)z + offset,
                          this->blockLength);
    }
    if (tilingKey == 2)
    { // 不均分
      this->formerNum = formerNum;
      this->formerLength = formerLength;
      this->formertileNum = formertileNum;
      this->formertileLength = formertileLength;
      this->formerlasttileLength = formerlasttileLength;

      this->tailNum = tailNum;
      this->tailLength = tailLength;
      this->tailtileNum = tailtileNum;
      this->tailtileLength = tailtileLength;
      this->taillasttileLength = taillasttileLength;

      if (GetBlockIdx() < this->formerNum)
      { // 大块
        this->tileLength = this->formertileLength / BUFFER_NUM;
        this->lasttileLength = this->formerlasttileLength;
        this->tileNum = this->formertileNum;
        uint32_t offset = this->formerLength * GetBlockIdx();
        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + offset, formerLength);
        if (this->gen_class == 0)
        {
          yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + offset, formerLength);
        }
        else
        {
          yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y, y_length);
          this->yScalar = yGm.GetValue(0);
        }
        zGm.SetGlobalBuffer((__gm__ DTYPE_Z *)z + offset, formerLength);
      }
      else
      { // 小块
        this->tileLength = this->tailtileLength / BUFFER_NUM;
        this->lasttileLength = this->taillasttileLength;
        this->tileNum = this->tailtileNum;
        uint32_t offset = this->formerLength * this->formerNum + this->tailLength * (GetBlockIdx() - this->formerNum);
        xGm.SetGlobalBuffer(
            (__gm__ DTYPE_X *)x + offset, this->tailLength);
        if (this->gen_class == 0)
        {
          yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + offset, this->tailLength);
        }
        else
        {
          yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y, y_length);
          this->yScalar = yGm.GetValue(0);
        }
        zGm.SetGlobalBuffer(
            (__gm__ DTYPE_Z *)z + offset, this->tailLength);
      }
    }
    pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
    pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
    pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Z));
  }
  __aicore__ inline void Process()
  {
    int32_t loopCount = this->tileNum * BUFFER_NUM;
    for (int32_t i = 0; i < loopCount; i++)
    {
      CopyIn(i);
      Compute(i);
      CopyOut(i);
    }
  }

private:
  // CopyIn()
  __aicore__ inline void CopyIn(int32_t progress)
  {

    LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
    LocalTensor<DTYPE_Y> yLocal = inQueueY.AllocTensor<DTYPE_Y>();
    if (this->gen_class == 1)
    {
      int32_t X_start;
      if ((progress == (this->tileNum * BUFFER_NUM - 2)) ||
          (progress == (this->tileNum * BUFFER_NUM - 1)))
      {
        // 尾块处理
        DataCopy(
            xLocal,
            xGm[(progress - 2) * (this->tileLength) + this->lasttileLength],
            (this->tileLength));
        X_start = this->blockLength * GetBlockIdx() + (progress - 2) * (this->tileLength) + this->lasttileLength;
      }
      else
      {
        DataCopy(xLocal, xGm[progress * (this->tileLength)],
                 (this->tileLength));
        X_start = this->blockLength * GetBlockIdx() + progress * this->tileLength;
      }
      int32_t gmStart = X_start % this->y_length;
      int32_t localStart = 0;                // y1Local的起始位置
      int32_t copyLength = this->tileLength; // 本次拷贝的长度
      for (int32_t i = 0; i < copyLength; i++)
      {
        yLocal.SetValue(i, yGm.GetValue((gmStart + i) % this->y_length));
      }
    }
    else if (this->gen_class == 2)
    {
      if ((progress == (this->tileNum * BUFFER_NUM - 2)) ||
          (progress == (this->tileNum * BUFFER_NUM - 1)))
      {
        DataCopy(
            xLocal,
            xGm[(progress - 2) * (this->tileLength) + this->lasttileLength], // (progress - 2) * (this->tileLength)
            (this->tileLength));
        DataCopy(yLocal, yGm[0], (this->tileLength));
      }
      else
      {
        DataCopy(xLocal, xGm[progress * (this->tileLength)],
                 (this->tileLength));
        DataCopy(yLocal, yGm[0], (this->tileLength));
      }
    }
    else if (this->gen_class == 0)
    {

      // 尾块处理
      if ((progress == (this->tileNum * BUFFER_NUM - 2)) ||
          (progress == (this->tileNum * BUFFER_NUM - 1)))
      {
        DataCopy(xLocal,
                 xGm[(progress - 2) * (this->tileLength) + this->lasttileLength], // (progress - 2) * (this->tileLength)
                 (this->tileLength));
        DataCopy(yLocal,
                 yGm[((progress - 2) * (this->tileLength) + this->lasttileLength)],
                 (this->tileLength));
      }
      else
      {
        DataCopy(xLocal, xGm[progress * (this->tileLength)],
                 (this->tileLength));
        DataCopy(yLocal, yGm[(progress * this->tileLength)],
                 this->tileLength);
      }
    }
    // 将LocalTesor放入VECIN（代表矢量编程中搬入数据的逻辑存放位置）的Queue中
    inQueueX.EnQue(xLocal);
    inQueueY.EnQue(yLocal);
  } // CopyIn

  __aicore__ inline void Compute(int32_t progress)
  {
    LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
    LocalTensor<DTYPE_Y> yLocal = inQueueY.DeQue<DTYPE_Y>();
    LocalTensor<DTYPE_Z> zLocal = outQueueZ.AllocTensor<DTYPE_Z>();
    if (this->gen_class != 2)
    {
      Add(zLocal, xLocal, yLocal, this->tileLength);
    }
    else
    {
      // DTYPE_Y yScalar = yLocal.GetValue(0);
      Adds(zLocal, xLocal, this->yScalar, this->tileLength);
    }
    outQueueZ.EnQue<DTYPE_Z>(zLocal);
    inQueueX.FreeTensor(xLocal);
    inQueueY.FreeTensor(yLocal);
  }

  __aicore__ inline void CopyOut(int32_t progress)
  {
    LocalTensor<DTYPE_Z> zLocal = outQueueZ.DeQue<DTYPE_Z>();
    if ((progress == (this->tileNum * BUFFER_NUM - 2)) ||
        (progress == (this->tileNum * BUFFER_NUM - 1)))
    {
      // 倒数第2个分块数据的起始地址向前移动（tileLength-lasttileLength)，最后一个分块的起始地址以此为基础进行移动
      DataCopy(
          zGm[(progress - 2) * (this->tileLength) + this->lasttileLength],
          zLocal, (this->tileLength));
    }

    else
    {
      DataCopy(zGm[progress * (this->tileLength)], zLocal,
               (this->tileLength));
    }
    outQueueZ.FreeTensor(zLocal);
  }

private:

  // Pipe内存管理对象
  TPipe pipe;
  TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
  TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;

  GlobalTensor<DTYPE_X> xGm;
  GlobalTensor<DTYPE_Y> yGm;
  GlobalTensor<DTYPE_Z> zGm;
  uint32_t blockLength;
  uint32_t tileNum;
  uint32_t tileLength;
  uint32_t lasttileLength;
  uint32_t formerNum;
  uint32_t formerLength;
  uint32_t formertileNum;
  uint32_t formertileLength;
  uint32_t formerlasttileLength;
  uint32_t tailNum;
  uint32_t gen_class;
  uint32_t tailLength;
  uint32_t tailtileNum;
  uint32_t tailtileLength;
  uint32_t taillasttileLength;
  uint32_t y_length;
  DTYPE_Y yScalar;
};

extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y,
                                                 GM_ADDR z,
                                                 GM_ADDR workspace,
                                                 GM_ADDR tiling)
{
  GET_TILING_DATA(tilingData, tiling);
  // TODO: user kernel impl
  KernelAdd op;
  uint32_t tilingKey = 1;
  if (TILING_KEY_IS(1))
  {
    tilingKey = 1;
  }
  else if (TILING_KEY_IS(2))
  {
    tilingKey = 2;
  }
  else
  {
    tilingKey = 1;
  }
  op.Init(x, y, z, tilingData.blockLength, tilingData.totalLength,
          tilingData.tileNum, tilingData.tileLength,
          tilingData.lasttileLength, tilingData.formerNum,
          tilingData.formerLength, tilingData.formertileNum,
          tilingData.formertileLength, tilingData.formerlasttileLength,
          tilingData.tailNum, tilingData.tailLength, tilingData.tailtileNum,
          tilingData.tailtileLength, tilingData.taillasttileLength,
          tilingKey, tilingData.ylength);
  op.Process();
}

#ifndef __CCE_KT_TEST__
// call of kernel function
void add_custom_do(uint32_t blockDim, void *l2ctrl, void *stream,
                   uint8_t *x, uint8_t *y, uint8_t *z,
                   uint8_t *workspace, uint8_t *tiling)
{
  add_custom<<<blockDim, l2ctrl, stream>>>(x, y, z, workspace, tiling);
}
#endif

