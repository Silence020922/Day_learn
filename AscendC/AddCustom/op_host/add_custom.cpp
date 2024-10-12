#include "tiling/platform/platform_ascendc.h"
#include "add_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
const uint32_t BLOCK_SIZE = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
  AddCustomTilingData tiling;
  // Get ubSize
  uint64_t ubSize;
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize); // Byte

  // Get totalLength
  uint32_t totalNum;
  totalNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize(); // element
  uint32_t sizeoftype = GetSizeByDataType(context->GetInputDesc(0)->GetDataType());
  uint32_t totalLength = totalNum * sizeoftype; // Byte

  // Set ubdataNum
  uint32_t ubdataNum = 4; // 2 input and 1 output
  uint32_t ubBlockNum = (ubSize/BLOCK_SIZE)/(ubdataNum); // avoid overload
  uint32_t ubDataNum = (ubBlockNum * BLOCK_SIZE)/(sizeoftype);// element

  // 32B Aligned
  uint32_t totalLengthAligned = ((totalLength + BLOCK_SIZE -1)/BLOCK_SIZE)*BLOCK_SIZE;
  uint32_t everycoreInputBlockNum = totalLengthAligned/BLOCK_SIZE;
  uint32_t everycoreInputDataNum = everycoreInputBlockNum * BLOCK_SIZE /sizeoftype;
  uint32_t tileNum = everycoreInputBlockNum/ubBlockNum;
  uint32_t tileDataNum = ubDataNum;
  uint32_t finaltileNum = (everycoreInputBlockNum % ubBlockNum == 0)? tileNum : tileNum + 1;

  // Tail
  uint32_t tailDataNum = everycoreInputDataNum - (tileNum * ubDataNum);//element
  tailDataNum = (tailDataNum == 0)? tileDataNum:tailDataNum;
  // Set Block_Dim
  context->SetBlockDim(1);
  
  //printf
  printf("sizeoftype: %d \n",sizeoftype);
  printf("everycoreInputDataNum: %d \n",everycoreInputDataNum);
  printf("tileDataNum: %d\n",tileDataNum);
  printf("tailDataNum: %d \n",tailDataNum);
  printf("tileNum: %d\n",finaltileNum);

  tiling.set_inputNum(everycoreInputDataNum);
  tiling.set_tileDataNum(tileDataNum);
  tiling.set_tailDataNum(tailDataNum);
  tiling.set_tileNum(finaltileNum);

  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}


namespace ops {
class AddCustom : public OpDef {
public:
    explicit AddCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(AddCustom);
}
