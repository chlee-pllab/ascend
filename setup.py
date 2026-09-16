import os
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

try:
    from torch_npu.utils.cpp_extension import NpuExtension
    ext_class = NpuExtension
    print("NPU environment detected, building with NpuExtension...")
except ImportError:
    ext_class = CppExtension
    print("No NPU detected, falling back to plain CppExtension...")

setup(
    name='fused_matmul_lib',
    ext_modules=[
        ext_class(
            name='fused_matmul_lib',
            sources=['matmul_leaky_relu_kernel.cpp'],
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
