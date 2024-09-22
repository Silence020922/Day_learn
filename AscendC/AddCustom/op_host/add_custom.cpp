#include "add_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling
{

  const uint32_t BLOCK_SIZE = 32;
  static ge::graphStatus TilingFunc(gert::TilingContext *context)
  {

    // infinite value
    TilingData tiling;
    uint32_t sizeofdatatype;
    uint32_t totalLengthAligned;
    uint32_t y_length;
    uint64_t ub_size;
    uint32_t totalLength;

    auto dt = context->GetInputTensor(0)->GetDataType();
    if (dt == 1 || dt == 6)
    { // float16 int16
      sizeofdatatype = 2;
    }
    else if (dt == 0 || dt == 3)
    { // float int32
      sizeofdatatype = 4;
    }

    // 平台信息获取
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto socVersion = ascendcPlatform.GetSocVersion();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    auto aivNum = ascendcPlatform.GetCoreNumAiv();

    // 核内拆分，策略是尽可能的填满ub_size，最后一包单独处理，
    uint32_t ub_block_num_real =
        ((ub_size) / BLOCK_SIZE / sizeofdatatype) / 3; // input-2, output-1, 理论最高值
    uint32_t ub_block_num = 5;                         // 为测试方便，验证代码流程
    uint32_t tile_num;
    if (ub_block_num % 2 != 0)
    {
      ub_block_num = ub_block_num - 1;
    }

    // 平台信息打印
    printf("----------platform info-----------\n");
    printf("ub_block_num_real: %d \n", ub_block_num_real);

    // input datatype size
    totalLength = context->GetInputTensor(0)->GetShapeSize();
    y_length = context->GetInputTensor(1)->GetShapeSize();

    uint32_t ALIGN_NUM = BLOCK_SIZE / sizeofdatatype;

    // totalLength -> totalLengthAligned
    if (totalLength % ALIGN_NUM != 0)
    { // 不对齐，先32位对齐
      totalLengthAligned =
          ((totalLength + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM;
    }
    else
    {
      totalLengthAligned = totalLength;
    }

    // BlockDim, ub_block_num*是允许一个核处理的最大数据量
    if (totalLengthAligned <= ub_block_num * ALIGN_NUM)
    { // shape较小，用单核
      context->SetBlockDim(1);
    }
    else
    {
      if (((totalLengthAligned / ALIGN_NUM) % ub_block_num) ==
          0)
      { // 可以核间均分
        if ((totalLengthAligned / ALIGN_NUM / ub_block_num) <=
            aivNum)
        { // 且计算出均分后的核数小于当前aicore数量，按计算值
          context->SetBlockDim(totalLengthAligned / ALIGN_NUM / ub_block_num);
        }
        else
        {
          // ... 按照aivNum切分
          context->SetBlockDim(aivNum);
        }
      }
      else
      { // 核间不能均分
        if (((totalLengthAligned / ALIGN_NUM / ub_block_num) + 1) <=
            aivNum)
        { // 且计算出均分后的核数小于当前aicore数量，按计算值
          context->SetBlockDim((totalLengthAligned / ALIGN_NUM / ub_block_num) + 1);
        }
        else
        {
          // ... 按照aivNum切分
          context->SetBlockDim(aivNum);
        }
      }
    } // 导致结果是不是直接就是一个核了？

    // infinite value
    auto block_dim = context->GetBlockDim();
    uint32_t blockLength = 0;
    uint32_t tileLength = 0;
    uint32_t lasttileLength = 0;
    uint32_t formertileLength = 0;
    uint32_t formerlasttileLength = 0;

    // tile_num, tileLength
    if ((totalLengthAligned / ALIGN_NUM) % block_dim == 0)
    { // 核间可均分
      blockLength = totalLengthAligned / block_dim;
      tile_num = blockLength / ALIGN_NUM / ub_block_num; // 分块数量
      if ((blockLength / ALIGN_NUM) % ub_block_num == 0 ||
          tile_num == 0)
      { // 满足32字节对齐，可以核内均分
        if (tile_num == 0)
        {
          tile_num = 1;
        }
        tileLength = ub_block_num * ALIGN_NUM;
        lasttileLength = tileLength;
      }
      else
      { // 满足32字节对齐，核内不能均分， 存在尾块
        tile_num = tile_num + 1;
        tileLength = ub_block_num * ALIGN_NUM;
        lasttileLength = blockLength - (tile_num - 1) * tileLength;
      }
      // 调试数据打印
      printf("----------------核间均分--------------\n");
      printf("blockLength: %d \n", blockLength);
      printf("tileNum: %d \n", tile_num);
      printf("ub_block_num %d \n", ub_block_num);
      printf("ALIGN_NUM: %d \n", ALIGN_NUM);
      printf("tileLength: %d \n", tileLength);
      printf("lasttileLength: %d \n", lasttileLength);
      printf("yLength: %d \n", y_length);
      // tiling信息填充
      context->SetTilingKey(1);
      tiling.set_totalLength(totalLength);
      tiling.set_blockLength(blockLength);
      tiling.set_tileNum(tile_num);
      tiling.set_tileLength(tileLength);
      tiling.set_lasttileLength(lasttileLength);
      tiling.set_ylength(y_length);
      tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                          context->GetRawTilingData()->GetCapacity());
      context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
      size_t *currentWorkspace = context->GetWorkspaceSizes(1);
      currentWorkspace[0] = 0;
      return ge::GRAPH_SUCCESS;
    }
    else
    { // 核间不可均分——大小核

      // 计算大小快的数量及数据量
      uint32_t formerNum = (totalLengthAligned / ALIGN_NUM) % block_dim;
      uint32_t tailNum = block_dim - formerNum;
      uint32_t formerLength =
          (((totalLengthAligned + block_dim - 1) / block_dim + ALIGN_NUM - 1) /
           ALIGN_NUM) *
          ALIGN_NUM;
      uint32_t tailLength = (totalLengthAligned / block_dim / ALIGN_NUM) * ALIGN_NUM;

      // former 情况
      uint32_t former_tile_num = formerLength / ALIGN_NUM / ub_block_num;
      if ((formerLength / ALIGN_NUM) % ub_block_num == 0 ||
          former_tile_num == 0)
      { // 核内均分
        if (former_tile_num == 0)
        {
          former_tile_num = 1;
        }
        formertileLength = ub_block_num * ALIGN_NUM;
        formerlasttileLength = formertileLength;
      }
      else
      {
        former_tile_num = former_tile_num + 1;
        formertileLength = ub_block_num * ALIGN_NUM;
        formerlasttileLength =
            (formerLength - (former_tile_num - 1) * formertileLength);
      }

      // tail情况
      uint32_t tail_tile_num = tailLength / ALIGN_NUM / ub_block_num;
      uint32_t tailtileLength;
      uint32_t taillasttileLength;
      if ((tailLength / ALIGN_NUM) % ub_block_num == 0 ||
          tail_tile_num == 0)
      { // 核内可以均分
        if (tail_tile_num == 0)
        {
          tail_tile_num = 1;
        }
        tailtileLength = ub_block_num * ALIGN_NUM;
        taillasttileLength = tailtileLength;
      }
      else
      { // 核内不均分
        tail_tile_num = tail_tile_num + 1;
        tailtileLength = ub_block_num * ALIGN_NUM;
        taillasttileLength = (tailLength - (tail_tile_num - 1) * tailtileLength);
      }
      // 打印调试信息
      printf("---------------核间非均分------------- \n");
      printf("formerNum: %d \n", formerNum);
      printf("tailNum: %d \n", tailNum);
      printf("formerLength: %d \n", formerLength);
      printf("tailLength: %d \n", tailLength);
      printf("formertileNum: %d \n", former_tile_num);
      printf("tailtileNum: %d \n", tail_tile_num);

      // tiling填充
      tiling.set_formerNum(formerNum);
      tiling.set_formerLength(formerLength);
      tiling.set_formertileNum(former_tile_num);
      tiling.set_formertileLength(formertileLength);
      tiling.set_formerlasttileLength(formerlasttileLength);
      tiling.set_tailNum(tailNum);
      tiling.set_tailLength(tailLength);
      tiling.set_tailtileNum(tail_tile_num);
      tiling.set_tailtileLength(tailtileLength);
      tiling.set_taillasttileLength(taillasttileLength);
      tiling.set_ylength(y_length);
      tiling.set_totalLength(totalLength);
      context->SetTilingKey(2);
      tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                          context->GetRawTilingData()->GetCapacity());
      context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
      size_t *currentWorkspace = context->GetWorkspaceSizes(1);
      currentWorkspace[0] = 0;
      return ge::GRAPH_SUCCESS;
    }
  }
} // namespace optiling

namespace ge
{
  static ge::graphStatus InferShape(gert::InferShapeContext *context)
  {
    const gert::Shape *x1_shape = context->GetInputShape(0);
    gert::Shape *y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
  }
} // namespace ge

namespace ops
{
  class AddCustom : public OpDef
  {
  public:
    explicit AddCustom(const char *name) : OpDef(name)
    {
      this->Input("x")
          .ParamType(REQUIRED)
          .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT16, ge::DT_INT32})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
      this->Input("y")
          .ParamType(REQUIRED)
          .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT16, ge::DT_INT32})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
      this->Output("z")
          .ParamType(REQUIRED)
          .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT16, ge::DT_INT32})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

      this->SetInferShape(ge::InferShape);

      this->AICore().SetTiling(optiling::TilingFunc);
      //    this->AICore().AddConfig("ascend910");
      //    this->AICore().AddConfig("ascend310p");
      //    this->AICore().AddConfig("ascend910b");
      this->AICore().AddConfig("ascend310b");
    }
  };

  OP_ADD(AddCustom);
} // namespace ops

