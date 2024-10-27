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

  tiling.set_inputNum(everycoreInputDataNum);
  tiling.set_tileDataNum(tileDataNum);
  tiling.set_tailDataNum(tailDataNum);
  tiling.set_tileNum(finaltileNum);

  // broadcast
  uint32_t xSize = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
  uint32_t ySize = context->GetInputShape(1)->GetStorageShape().GetShapeSize();
  uint32_t zSize = context->GetOutputShape(0)->GetStorageShape().GetShapeSize();

  if (zSize != xSize || zSize != ySize){
  context->SetTilingKey(2);
  int32_t x_narray[8], y_narray[8],z_narray[8];
  int32_t x_dimn, y_dimn,z_dimn;
  auto shape_z = context->GetOutputShape(0)->GetOriginShape();
  auto shape_x = context->GetInputTensor(0)->GetOriginShape();
  auto shape_y = context->GetInputTensor(1)->GetOriginShape();
  z_dimn = shape_z.GetDimNum();
  x_dimn = shape_x.GetDimNum();
  y_dimn = shape_y.GetDimNum();
  
  for (int i = 0;i<z_dimn;i++){
  z_narray[z_dimn-i-1] = shape_z.GetDim(i);
  if (i < x_dimn) x_narray[x_dimn -i - 1] = shape_x.GetDim(i);else x_narray[i] = 1;
  if (i< y_dimn) y_narray[y_dimn -i -1] = shape_y.GetDim(i);else y_narray[i] = 1;
  }

  int32_t x_sumnarray[8],y_sumnarray[8],z_sumnarray[8];
  x_sumnarray[0] = 1;
  y_sumnarray[0] = 1;
  z_sumnarray[0] = 1;
  for (int i = 1;i<=z_dimn;i++){
    z_sumnarray[i] = z_sumnarray[i-1]*z_narray[i-1];
    x_sumnarray[i] = x_sumnarray[i-1]*x_narray[i-1];
    y_sumnarray[i] = y_sumnarray[i-1]*y_narray[i-1];
  }

  tiling.set_zdimn(z_dimn);
  tiling.set_x_narray(x_narray);
  tiling.set_y_narray(y_narray);
  tiling.set_z_narray(z_narray);
  tiling.set_x_sumnarray(x_sumnarray);
  tiling.set_y_sumnarray(y_sumnarray);
  tiling.set_z_sumnarray(z_sumnarray);
  }else{
    context->SetTilingKey(1);
    //printf
    printf("sizeoftype: %d \n",sizeoftype);
    printf("everycoreInputDataNum: %d \n",everycoreInputDataNum);
    printf("tileDataNum: %d\n",tileDataNum);
    printf("tailDataNum: %d \n",tailDataNum);
    printf("tileNum: %d\n",finaltileNum);
  }

  context->SetBlockDim(1);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  size_t *currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] = 0;
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
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT16, ge::DT_INT32})
            .Format({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND});
        this->Input("y")
            .ParamType(REQUIRED)
	    .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT16, ge::DT_INT32})
            .Format({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
	    .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT16, ge::DT_INT32})
            .Format({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape);
        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(AddCustom);
}
