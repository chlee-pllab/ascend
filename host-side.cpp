#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "lib/matmul_tiling.h"  // High-level MatMul Tiling header

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    // 1. Get hardware platform instance (e.g., Ascend 910B / 310P etc.)
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(context->GetPlatformInfo());
    
    // 2. Instantiate the Multi-Core MatMul Tiling object
    matmul_tiling::MultiCoreMatmulTiling tilingApi(*ascendcPlatform);

    // 3. Extract problem shapes from input tensors dynamically
    auto shapeA = context->GetInputShape(0)->GetStorageShape();
    int32_t M = shapeA.GetDim(0);
    int32_t K = shapeA.GetDim(1);
    
    auto shapeB = context->GetInputShape(1)->GetStorageShape();
    int32_t N = shapeB.GetDim(1);

    // 4. Configure Matrix Properties (Data positions, memory formats, data types)
    tilingApi.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tilingApi.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tilingApi.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    
    // 5. Assign Shapes and Hardware allocation boundaries
    tilingApi.SetShape(M, N, K);
    
    int usedCoreNum = 20; // Number of AI Cores allocated for this task
    tilingApi.SetDim(usedCoreNum);

    // 6. Compute Tiling parameters and export into the TCubeTiling data structure
    TCubeTiling tilingData;
    ge::graphStatus ret = tilingApi.GetTiling(tilingData);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    // 7. Append structural payload to context to be sent over to the NPU Kernel
    context->SetBlockDim(usedCoreNum);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
