
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AddCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, inputNum);
  TILING_DATA_FIELD_DEF(uint32_t, tileDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, tailDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
  TILING_DATA_FIELD_DEF(int32_t, zdimn);
  TILING_DATA_FIELD_DEF_ARR(int32_t, 8, x_narray);
  TILING_DATA_FIELD_DEF_ARR(int32_t, 8, y_narray);
  TILING_DATA_FIELD_DEF_ARR(int32_t, 8, z_narray);
  TILING_DATA_FIELD_DEF_ARR(int32_t, 8, x_sumnarray);
  TILING_DATA_FIELD_DEF_ARR(int32_t, 8, y_sumnarray);
  TILING_DATA_FIELD_DEF_ARR(int32_t, 8, z_sumnarray);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AddCustom, AddCustomTilingData)
}
