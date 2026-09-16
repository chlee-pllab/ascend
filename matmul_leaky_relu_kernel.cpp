#include <torch/extension.h>
#include <torch/script.h>

// Build-time capability probe: only compile the Ascend NPU device-kernel path
// when EVERY header it needs is actually present. Checking NPUStream.h alone
// is not a reliable signal -- this torch_npu build ships NPUStream.h but does
// NOT ship AsycDebug.h, so a check on NPUStream.h alone lets a broken #include
// through. (Note: this can only detect whether the toolchain/headers are
// present at compile time -- it cannot detect whether real NPU hardware is
// physically present, which is only knowable at runtime. The runtime
// PrivateUse1 device check further down is what actually gates on hardware.)
#if defined(USE_NPU) || \
    (__has_include("torch_npu/csrc/core/npu/NPUStream.h") && \
     __has_include("torch_npu/csrc/core/npu/interface/AsycDebug.h") && \
     __has_include("kernel_operator.h"))
    #define HAS_ASCEND_NPU 1
    #include "torch_npu/csrc/core/npu/NPUStream.h"
    #include "torch_npu/csrc/core/npu/interface/AsycDebug.h"
    #include "kernel_operator.h"
    using namespace AscendC;
#else
    #define HAS_ASCEND_NPU 0
#endif

// ============================================================================
// 1. Ascend C Device-Side AI Core Kernel (Compiled only in NPU environments)
// ============================================================================
#if HAS_ASCEND_NPU
class KernelMatMulLeakyReLU {
public:
    __aicore__ inline KernelMatMulLeakyReLU() {}
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, int M, int N, int K, float alpha)
    {
        // Initialize Global Memory buffer addresses
        aGm.SetGlobalBuffer((__gm__ float*)a);
        bGm.SetGlobalBuffer((__gm__ float*)b);
        cGm.SetGlobalBuffer((__gm__ float*)c);
        
        // Configure matrix multiplication (Cube) specifications
        tiling.M = M;
        tiling.N = N;
        tiling.K = K;
        
        // Initialize memory sizes for internal TQue queues
        pipe.InitBuffer(inQueueA, 1, M * K * sizeof(float));
        pipe.InitBuffer(inQueueB, 1, K * N * sizeof(float));
        pipe.InitBuffer(outQueueC, 1, M * N * sizeof(float));
        
        this->alpha = alpha;
        this->totalSize = M * N;
    }

    __aicore__ inline void Process()
    {
        // CopyIn
        Local Tensor<float> aLocal = inQueueA.AllocTensor<float>();
        Local Tensor<float> bLocal = inQueueB.AllocTensor<float>();
        DataCopy(aLocal, aGm, tiling.M * tiling.K);
        DataCopy(bLocal, bGm, tiling.K * tiling.N);
        inQueueA.EnQueue(aLocal);
        inQueueB.EnQueue(bLocal);

        Local Tensor<float> aMat = inQueueA.DeQueue<float>();
        Local Tensor<float> bMat = inQueueB.DeQueue<float>();
        Local Tensor<float> cLocal = outQueueC.AllocTensor<float>();

        MatMul(cLocal, aMat, bMat, tiling.M, tiling.N, tiling.K);
        LeakyReLU(cLocal, cLocal, (float)alpha, totalSize);

        outQueueC.EnQueue(cLocal);
        inQueueA.FreeTensor(aMat);
        inQueueB.FreeTensor(bMat);

        // CopyOut
        Local Tensor<float> cOut = outQueueC.DeQueue<float>();
        DataCopy(cGm, cOut, totalSize);
        outQueueC.FreeTensor(cOut);
    }
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueA;
    TQue<QuePosition::VECIN, 1> inQueueB;
    TQue<QuePosition::VECOUT, 1> outQueueC;
    GlobalTensor<float> aGm;
    GlobalTensor<float> bGm;
    GlobalTensor<float> cGm;
    struct { int M; int N; int K; } tiling;
    float alpha;
    int totalSize;
};

// Device-side kernel entry function
extern "C" __global__ __aicore__ void matmul_leaky_relu_kernel(GM_ADDR a, GM_ADDR b, GM_ADDR c, int M, int N, int K, float alpha) {
    KernelMatMulLeakyReLU op;
    op.Init(a, b, c, M, N, K, alpha);
    op.Process();
}
#endif

// ============================================================================
// 2. Host-Side C++ Entry Point: Automatic NPU / CPU Fallback Selection
// ============================================================================
at::Tensor fused_matmul_leaky_relu(const at::Tensor& tensor_a, const at::Tensor& tensor_b, float alpha) {
    // ---- Case A: Running on Ascend NPU hardware environment ----
    #if HAS_ASCEND_NPU
    if (tensor_a.device().type() == c10::DeviceType::PrivateUse1) {
        // Ensure input tensors are contiguous in memory
        auto a_contiguous = tensor_a.contiguous();
        auto b_contiguous = tensor_b.contiguous();

        int M = a_contiguous.size(0);
        int K = a_contiguous.size(1);
        int N = b_contiguous.size(1);

        // Allocate the output NPU Tensor
        auto options = tensor_a.options();
        at::Tensor tensor_c = at::empty({M, N}, options);

        // Fetch the current active PyTorch NPU Stream
        c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream();

        // Launch the Ascend C hardware kernel (BlockDim set to 1 for simplicity)
        uint32_t blockDim = 1;
        ACL_LAUNCH_KERNEL_ARGS(matmul_leaky_relu_kernel, blockDim, stream.stream(),
                               a_contiguous.data_ptr(), b_contiguous.data_ptr(), tensor_c.data_ptr(), M, N, K, alpha);
        return tensor_c;
    }
    #endif
    // Fallback to PyTorch CPU (No Ascend hardware)
    at::Tensor tensor_c = at::matmul(tensor_a, tensor_b.t());
    at::leaky_relu_(tensor_c, alpha);

    return tensor_c;
}

// ============================================================================
// 3. Register PyTorch Module Interface via pybind11
// ============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_activation_forward", &fused_matmul_leaky_relu,
          "Fused MatMul + LeakyReLU Kernel (Universal Support)",
          pybind11::arg("tensor_a"), pybind11::arg("tensor_b"), pybind11::arg("alpha") = 0.01f);
}
