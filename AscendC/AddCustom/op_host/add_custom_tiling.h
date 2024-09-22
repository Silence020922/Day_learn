#ifndef ADD_CUSTOM_TILING_H
#define ADD_CUSTOM_TILING_H
#include "register/tilingdata_base.h"

namespace optiling
{
  BEGIN_TILING_DATA_DEF(TilingData);
  TILING_DATA_FIELD_DEF(uint32_t, blockLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
  TILING_DATA_FIELD_DEF(uint32_t, tileLength);
  TILING_DATA_FIELD_DEF(uint32_t, lasttileLength);
  TILING_DATA_FIELD_DEF(uint32_t, formerNum);
  TILING_DATA_FIELD_DEF(uint32_t, formerLength);
  TILING_DATA_FIELD_DEF(uint32_t, formertileNum);
  TILING_DATA_FIELD_DEF(uint32_t, formertileLength);
  TILING_DATA_FIELD_DEF(uint32_t, formerlasttileLength);
  TILING_DATA_FIELD_DEF(uint32_t, tailNum);
  TILING_DATA_FIELD_DEF(uint32_t, tailLength);
  TILING_DATA_FIELD_DEF(uint32_t, tailtileNum);
  TILING_DATA_FIELD_DEF(uint32_t, tailtileLength);
  TILING_DATA_FIELD_DEF(uint32_t, taillasttileLength);
  TILING_DATA_FIELD_DEF(uint32_t, ylength);
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  END_TILING_DATA_DEF;
  REGISTER_TILING_DATA_CLASS(AddCustom, TilingData)
}
#endif // ADD_CUSTOM_TILING_H

