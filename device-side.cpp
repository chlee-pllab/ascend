#include "kernel_operator.h"
#include "lib/matmul_intf.h" // Provides the high-level Matmul template interface

using namespace AscendC;

// Instantiate the high-level Matmul template class
// Template arguments map: <TypeA, TypeB, TypeC, TypeBias, LayoutPolicies...>
typedef MatmulType<ConfigMode::COMMON, float, float, float, float> MyMatMul;

class KernelMatMul {
public:
    __aicore__ inline KernelMatMul() {}
    
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, TCubeTiling& tiling) {
        // Assign external Global Tensor addresses
        aGm.SetGlobalBuffer((__gm__ float*)a);
        bGm.SetGlobalBuffer((__gm__ float*)b);
        cGm.SetGlobalBuffer((__gm__ float*)c);

        // Initialize the high-level Matmul handler using the automated structural data
        matmulObj.Init(&tiling);
        
        // Pass base matrix pointers to the automated handler
        matmulObj.SetTensorA(aGm);
        matmulObj.SetTensorB(bGm);
    }

    __aicore__ inline void Process() {
        // Execute the automated Tiling pipeline. 
        // Iterate() automatically handles multi-buffer scheduling, slice offsets, 
        // internal data streaming, and automatically writes the final output to cGm.
        matmulObj.Iterate(cGm);
    }

private:
    GlobalTensor<float> aGm;
    GlobalTensor<float> bGm;
    GlobalTensor<float> cGm;
    
    MyMatMul matmulObj; // Embedded high-level runtime instance
};

extern "C" __global__ __aicore__ void matmul_high_level_kernel(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR tilingGmem) {
    // 1. Unpack the structural Tiling parameters transmitted from the Host side
    TCubeTiling tiling;
    auto tilingPtr = (__gm__ uint32_t*)tilingGmem;
    tiling.LoadFromBuffer(tilingPtr);

    // 2. Initialize and kickstart execution
    KernelMatMul op;
    op.Init(a, b, c, tiling);
    op.Process();
}
