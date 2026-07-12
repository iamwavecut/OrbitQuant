"""Golden snapshot tests for backend_capabilities().

The EXPECTED literals below were captured from the hand-built implementation
of orbitquant.kernels.dispatch.backend_capabilities prior to its table-driven
refactor. They freeze the full reported capability payload (values and key
order) for a representative grid of availability combinations. Only regenerate
them when the reported capabilities are intentionally changed.
"""

import pytest

import orbitquant.kernels.dispatch as dispatch_module
from orbitquant.kernels.dispatch import backend_capabilities

# Frozen copies of long space-free payload strings (deliberately NOT imported
# from orbitquant.kernels.dispatch, so the snapshot stays independent).
_CPU_STAGES = "activation_norm_rpbh_quant_rescale,packed_weight_matmul,adaln_rtn_packed_matmul"
_MPS_STAGES = "activation_norm_rpbh_quant_rescale,packed_weight_dequant,packed_weight_matmul"
_TRITON_STAGES = (
    "activation_norm_rpbh_quant_rescale,packed_weight_dequant,packed_weight_matmul,"
    "lowbit_pack,lowbit_unpack,weight_rotation_fwht_quant_pack,adaln_rtn_quant_pack,"
    "adaln_rtn_dequant,adaln_rtn_packed_matmul"
)
_CPU_IMPL_NATIVE_ALL = "native_exact_activation+native_exact_packed_matmul+native_packed_adaln_int4"
_CPU_IMPL_MATMUL_ADALN = (
    "native_exact_packed_matmul+torch_activation_reference+native_packed_adaln_int4"
)

# Each case: (availability probes monkeypatched onto dispatch_module,
# `backends` argument passed to backend_capabilities).
CASES: dict[str, tuple[dict[str, bool], dict[str, bool]]] = {
    "all_false": (
        {'_native_cpu_packed_matmul_available': False,
         '_native_cpu_activation_available': False,
         '_native_cpu_adaln_available': False,
         '_mps_metal_available': False,
         '_native_packed_matmul_available': False},
        {'cpu': False,
         'mps': False,
         'triton_cuda': False,
         'triton_rocm': False,
         'triton_xpu': False},
    ),
    "cpu_native_all": (
        {'_native_cpu_packed_matmul_available': True,
         '_native_cpu_activation_available': True,
         '_native_cpu_adaln_available': True,
         '_mps_metal_available': False,
         '_native_packed_matmul_available': False},
        {'cpu': True,
         'mps': False,
         'triton_cuda': False,
         'triton_rocm': False,
         'triton_xpu': False},
    ),
    "cpu_activation_only": (
        {'_native_cpu_packed_matmul_available': False,
         '_native_cpu_activation_available': True,
         '_native_cpu_adaln_available': False,
         '_mps_metal_available': False,
         '_native_packed_matmul_available': False},
        {'cpu': True,
         'mps': False,
         'triton_cuda': False,
         'triton_rocm': False,
         'triton_xpu': False},
    ),
    "cpu_adaln_only": (
        {'_native_cpu_packed_matmul_available': False,
         '_native_cpu_activation_available': False,
         '_native_cpu_adaln_available': True,
         '_mps_metal_available': False,
         '_native_packed_matmul_available': False},
        {'cpu': True,
         'mps': False,
         'triton_cuda': False,
         'triton_rocm': False,
         'triton_xpu': False},
    ),
    "mps_shader_and_native_matmul": (
        {'_native_cpu_packed_matmul_available': False,
         '_native_cpu_activation_available': False,
         '_native_cpu_adaln_available': False,
         '_mps_metal_available': True,
         '_native_packed_matmul_available': True},
        {'cpu': True,
         'mps': True,
         'triton_cuda': False,
         'triton_rocm': False,
         'triton_xpu': False},
    ),
    "mps_native_matmul_only": (
        {'_native_cpu_packed_matmul_available': False,
         '_native_cpu_activation_available': False,
         '_native_cpu_adaln_available': False,
         '_mps_metal_available': False,
         '_native_packed_matmul_available': True},
        {'cpu': True,
         'mps': True,
         'triton_cuda': False,
         'triton_rocm': False,
         'triton_xpu': False},
    ),
    "triton_cuda": (
        {'_native_cpu_packed_matmul_available': False,
         '_native_cpu_activation_available': False,
         '_native_cpu_adaln_available': False,
         '_mps_metal_available': False,
         '_native_packed_matmul_available': False},
        {'cpu': True,
         'mps': False,
         'triton_cuda': True,
         'triton_rocm': False,
         'triton_xpu': False},
    ),
    "triton_rocm": (
        {'_native_cpu_packed_matmul_available': False,
         '_native_cpu_activation_available': False,
         '_native_cpu_adaln_available': False,
         '_mps_metal_available': False,
         '_native_packed_matmul_available': False},
        {'cpu': True,
         'mps': False,
         'triton_cuda': False,
         'triton_rocm': True,
         'triton_xpu': False},
    ),
    "triton_xpu": (
        {'_native_cpu_packed_matmul_available': False,
         '_native_cpu_activation_available': False,
         '_native_cpu_adaln_available': False,
         '_mps_metal_available': False,
         '_native_packed_matmul_available': False},
        {'cpu': True,
         'mps': False,
         'triton_cuda': False,
         'triton_rocm': False,
         'triton_xpu': True},
    ),
    "mixed": (
        {'_native_cpu_packed_matmul_available': True,
         '_native_cpu_activation_available': False,
         '_native_cpu_adaln_available': True,
         '_mps_metal_available': True,
         '_native_packed_matmul_available': False},
        {'cpu': True, 'mps': True, 'triton_cuda': True, 'triton_rocm': False, 'triton_xpu': True},
    ),
}

EXPECTED: dict[str, dict[str, dict[str, object]]] = {
    "all_false": {
        "cpu": {'available': False,
                'claim_status': 'reference_only',
                'optimized': False,
                'full_fusion': False,
                'implemented_stage': _CPU_STAGES,
                'optimized_stage': None,
                'weight_dequant_optimized': False,
                'weight_pack_optimized': False,
                'lowbit_unpack_optimized': False,
                'weight_quant_optimized': False,
                'adaln_quant_optimized': False,
                'adaln_dequant_optimized': False,
                'adaln_packed_matmul_optimized': False,
                'device_types': ['cpu'],
                'implementation': 'torch_reference',
                'package_format': 'torch_reference',
                'hf_kernel_builder_compliant': False,
                'notes': 'Activation quantization uses the reference PyTorch CPU path. The main '
                         'linear matmul uses the reference PyTorch CPU path. AdaLN uses the '
                         'explicit BF16 dequantization path.'},
        "mps": {'available': False,
                'claim_status': 'reference_only',
                'optimized': False,
                'full_fusion': False,
                'implemented_stage': _MPS_STAGES,
                'optimized_stage': None,
                'weight_dequant_optimized': False,
                'weight_pack_optimized': False,
                'lowbit_unpack_optimized': False,
                'weight_quant_optimized': False,
                'adaln_quant_optimized': False,
                'adaln_dequant_optimized': False,
                'adaln_packed_matmul_optimized': False,
                'device_types': ['mps'],
                'implementation': 'torch_reference_mps',
                'package_format': 'torch_reference',
                'upstream_native_mps_op': False,
                'hf_kernel_builder_compliant': False,
                'notes': 'Runs the reference PyTorch path on MPS tensors; native Metal shader '
                         'support and the native packed matmul package are not available in this '
                         'environment.'},
        "triton_cuda": {'available': False,
                        'claim_status': 'unavailable',
                        'optimized': False,
                        'full_fusion': False,
                        'implemented_stage': _TRITON_STAGES,
                        'optimized_stage': None,
                        'weight_dequant_optimized': False,
                        'weight_pack_optimized': False,
                        'lowbit_unpack_optimized': False,
                        'weight_quant_optimized': False,
                        'adaln_quant_optimized': False,
                        'adaln_dequant_optimized': False,
                        'adaln_packed_matmul_optimized': False,
                        'device_types': ['cuda'],
                        'implementation': 'python_triton_orbitquant_pipeline',
                        'package_format': 'python_triton',
                        'hf_kernel_builder_compliant': False,
                        'notes': 'Triton handles runtime activation norm, RPBH/FWHT rotation, '
                                 'codebook lookup/rescale, packed weight dequant, packed weight '
                                 'matmul, offline low-bit pack/unpack, offline weight RPBH/FWHT '
                                 'codebook indexing with direct low-bit packing, and AdaLN INT4 '
                                 'RTN quantize/pack, dequant, and packed matmul. The default '
                                 'auto_fused runtime prefers packed low-bit matmul when a native '
                                 'or Triton packed kernel is available; full-model speedup '
                                 'claims still require separate benchmark artifacts.'},
        "triton_rocm": {'available': False,
                        'claim_status': 'unavailable',
                        'optimized': False,
                        'full_fusion': False,
                        'implemented_stage': _TRITON_STAGES,
                        'optimized_stage': None,
                        'weight_dequant_optimized': False,
                        'weight_pack_optimized': False,
                        'lowbit_unpack_optimized': False,
                        'weight_quant_optimized': False,
                        'adaln_quant_optimized': False,
                        'adaln_dequant_optimized': False,
                        'adaln_packed_matmul_optimized': False,
                        'device_types': ['cuda'],
                        'implementation': 'python_triton_orbitquant_pipeline_rocm_candidate',
                        'package_format': 'python_triton_rocm',
                        'hf_kernel_builder_compliant': False,
                        'notes': 'PyTorch exposes HIP tensors through the cuda device type. The '
                                 'candidate reuses the exact packed Triton pipeline without '
                                 'loading CUDA-native extensions. It remains experimental until '
                                 'correctness, memory, profiler, and performance evidence is '
                                 'recorded on supported AMD hardware.'},
        "triton_xpu": {'available': False,
                       'claim_status': 'unavailable',
                       'optimized': False,
                       'full_fusion': False,
                       'implemented_stage': _TRITON_STAGES,
                       'optimized_stage': None,
                       'weight_dequant_optimized': False,
                       'weight_pack_optimized': False,
                       'lowbit_unpack_optimized': False,
                       'weight_quant_optimized': False,
                       'adaln_quant_optimized': False,
                       'adaln_dequant_optimized': False,
                       'adaln_packed_matmul_optimized': False,
                       'device_types': ['xpu'],
                       'implementation': 'python_triton_orbitquant_pipeline_xpu_candidate',
                       'package_format': 'python_triton_xpu',
                       'hf_kernel_builder_compliant': False,
                       'notes': 'The candidate reuses the exact packed Triton pipeline on '
                                'torch.xpu. It remains explicit-only and experimental until '
                                'correctness, memory, profiler, and performance evidence is '
                                'recorded on supported Intel GPU hardware.'},
    },
    "cpu_native_all": {
        "cpu": {'available': True,
                'claim_status': 'partial_optimized',
                'optimized': True,
                'full_fusion': False,
                'implemented_stage': _CPU_STAGES,
                'optimized_stage': _CPU_STAGES,
                'weight_dequant_optimized': False,
                'weight_pack_optimized': False,
                'lowbit_unpack_optimized': False,
                'weight_quant_optimized': False,
                'adaln_quant_optimized': False,
                'adaln_dequant_optimized': True,
                'adaln_packed_matmul_optimized': True,
                'device_types': ['cpu'],
                'implementation': _CPU_IMPL_NATIVE_ALL,
                'package_format': 'native_kernel_package_torch_stable_abi',
                'hf_kernel_builder_compliant': True,
                'notes': 'The native exact activation pipeline performs FP32 token norm, '
                         'RPBH/FWHT, codebook assignment, and rescale. Native exact packed '
                         'matmul consumes low-bit weights without a full floating-point cache. '
                         'Native packed INT4 AdaLN matmul consumes group-64 weights without a '
                         'full floating-point cache.'},
        "mps": {'available': False,
                'claim_status': 'reference_only',
                'optimized': False,
                'full_fusion': False,
                'implemented_stage': _MPS_STAGES,
                'optimized_stage': None,
                'weight_dequant_optimized': False,
                'weight_pack_optimized': False,
                'lowbit_unpack_optimized': False,
                'weight_quant_optimized': False,
                'adaln_quant_optimized': False,
                'adaln_dequant_optimized': False,
                'adaln_packed_matmul_optimized': False,
                'device_types': ['mps'],
                'implementation': 'torch_reference_mps',
                'package_format': 'torch_reference',
                'upstream_native_mps_op': False,
                'hf_kernel_builder_compliant': False,
                'notes': 'Runs the reference PyTorch path on MPS tensors; native Metal shader '
                         'support and the native packed matmul package are not available in this '
                         'environment.'},
        "triton_cuda": {'available': False,
                        'claim_status': 'unavailable',
                        'optimized': False,
                        'full_fusion': False,
                        'implemented_stage': _TRITON_STAGES,
                        'optimized_stage': None,
                        'weight_dequant_optimized': False,
                        'weight_pack_optimized': False,
                        'lowbit_unpack_optimized': False,
                        'weight_quant_optimized': False,
                        'adaln_quant_optimized': False,
                        'adaln_dequant_optimized': False,
                        'adaln_packed_matmul_optimized': False,
                        'device_types': ['cuda'],
                        'implementation': 'python_triton_orbitquant_pipeline',
                        'package_format': 'python_triton',
                        'hf_kernel_builder_compliant': False,
                        'notes': 'Triton handles runtime activation norm, RPBH/FWHT rotation, '
                                 'codebook lookup/rescale, packed weight dequant, packed weight '
                                 'matmul, offline low-bit pack/unpack, offline weight RPBH/FWHT '
                                 'codebook indexing with direct low-bit packing, and AdaLN INT4 '
                                 'RTN quantize/pack, dequant, and packed matmul. The default '
                                 'auto_fused runtime prefers packed low-bit matmul when a native '
                                 'or Triton packed kernel is available; full-model speedup '
                                 'claims still require separate benchmark artifacts.'},
        "triton_rocm": {'available': False,
                        'claim_status': 'unavailable',
                        'optimized': False,
                        'full_fusion': False,
                        'implemented_stage': _TRITON_STAGES,
                        'optimized_stage': None,
                        'weight_dequant_optimized': False,
                        'weight_pack_optimized': False,
                        'lowbit_unpack_optimized': False,
                        'weight_quant_optimized': False,
                        'adaln_quant_optimized': False,
                        'adaln_dequant_optimized': False,
                        'adaln_packed_matmul_optimized': False,
                        'device_types': ['cuda'],
                        'implementation': 'python_triton_orbitquant_pipeline_rocm_candidate',
                        'package_format': 'python_triton_rocm',
                        'hf_kernel_builder_compliant': False,
                        'notes': 'PyTorch exposes HIP tensors through the cuda device type. The '
                                 'candidate reuses the exact packed Triton pipeline without '
                                 'loading CUDA-native extensions. It remains experimental until '
                                 'correctness, memory, profiler, and performance evidence is '
                                 'recorded on supported AMD hardware.'},
        "triton_xpu": {'available': False,
                       'claim_status': 'unavailable',
                       'optimized': False,
                       'full_fusion': False,
                       'implemented_stage': _TRITON_STAGES,
                       'optimized_stage': None,
                       'weight_dequant_optimized': False,
                       'weight_pack_optimized': False,
                       'lowbit_unpack_optimized': False,
                       'weight_quant_optimized': False,
                       'adaln_quant_optimized': False,
                       'adaln_dequant_optimized': False,
                       'adaln_packed_matmul_optimized': False,
                       'device_types': ['xpu'],
                       'implementation': 'python_triton_orbitquant_pipeline_xpu_candidate',
                       'package_format': 'python_triton_xpu',
                       'hf_kernel_builder_compliant': False,
                       'notes': 'The candidate reuses the exact packed Triton pipeline on '
                                'torch.xpu. It remains explicit-only and experimental until '
                                'correctness, memory, profiler, and performance evidence is '
                                'recorded on supported Intel GPU hardware.'},
    },
    "cpu_activation_only": {
        "cpu": {'available': True,
                'claim_status': 'partial_optimized',
                'optimized': True,
                'full_fusion': False,
                'implemented_stage': _CPU_STAGES,
                'optimized_stage': 'activation_norm_rpbh_quant_rescale',
                'weight_dequant_optimized': False,
                'weight_pack_optimized': False,
                'lowbit_unpack_optimized': False,
                'weight_quant_optimized': False,
                'adaln_quant_optimized': False,
                'adaln_dequant_optimized': False,
                'adaln_packed_matmul_optimized': False,
                'device_types': ['cpu'],
                'implementation': 'native_exact_activation+reference_weight_matmul',
                'package_format': 'native_kernel_package_torch_stable_abi',
                'hf_kernel_builder_compliant': True,
                'notes': 'The native exact activation pipeline performs FP32 token norm, '
                         'RPBH/FWHT, codebook assignment, and rescale. The main linear matmul '
                         'uses the reference PyTorch CPU path. AdaLN uses the explicit BF16 '
                         'dequantization path.'},
        "mps": {'available': False,
                'claim_status': 'reference_only',
                'optimized': False,
                'full_fusion': False,
                'implemented_stage': _MPS_STAGES,
                'optimized_stage': None,
                'weight_dequant_optimized': False,
                'weight_pack_optimized': False,
                'lowbit_unpack_optimized': False,
                'weight_quant_optimized': False,
                'adaln_quant_optimized': False,
                'adaln_dequant_optimized': False,
                'adaln_packed_matmul_optimized': False,
                'device_types': ['mps'],
                'implementation': 'torch_reference_mps',
                'package_format': 'torch_reference',
                'upstream_native_mps_op': False,
                'hf_kernel_builder_compliant': False,
                'notes': 'Runs the reference PyTorch path on MPS tensors; native Metal shader '
                         'support and the native packed matmul package are not available in this '
                         'environment.'},
        "triton_cuda": {'available': False,
                        'claim_status': 'unavailable',
                        'optimized': False,
                        'full_fusion': False,
                        'implemented_stage': _TRITON_STAGES,
                        'optimized_stage': None,
                        'weight_dequant_optimized': False,
                        'weight_pack_optimized': False,
                        'lowbit_unpack_optimized': False,
                        'weight_quant_optimized': False,
                        'adaln_quant_optimized': False,
                        'adaln_dequant_optimized': False,
                        'adaln_packed_matmul_optimized': False,
                        'device_types': ['cuda'],
                        'implementation': 'python_triton_orbitquant_pipeline',
                        'package_format': 'python_triton',
                        'hf_kernel_builder_compliant': False,
                        'notes': 'Triton handles runtime activation norm, RPBH/FWHT rotation, '
                                 'codebook lookup/rescale, packed weight dequant, packed weight '
                                 'matmul, offline low-bit pack/unpack, offline weight RPBH/FWHT '
                                 'codebook indexing with direct low-bit packing, and AdaLN INT4 '
                                 'RTN quantize/pack, dequant, and packed matmul. The default '
                                 'auto_fused runtime prefers packed low-bit matmul when a native '
                                 'or Triton packed kernel is available; full-model speedup '
                                 'claims still require separate benchmark artifacts.'},
        "triton_rocm": {'available': False,
                        'claim_status': 'unavailable',
                        'optimized': False,
                        'full_fusion': False,
                        'implemented_stage': _TRITON_STAGES,
                        'optimized_stage': None,
                        'weight_dequant_optimized': False,
                        'weight_pack_optimized': False,
                        'lowbit_unpack_optimized': False,
                        'weight_quant_optimized': False,
                        'adaln_quant_optimized': False,
                        'adaln_dequant_optimized': False,
                        'adaln_packed_matmul_optimized': False,
                        'device_types': ['cuda'],
                        'implementation': 'python_triton_orbitquant_pipeline_rocm_candidate',
                        'package_format': 'python_triton_rocm',
                        'hf_kernel_builder_compliant': False,
                        'notes': 'PyTorch exposes HIP tensors through the cuda device type. The '
                                 'candidate reuses the exact packed Triton pipeline without '
                                 'loading CUDA-native extensions. It remains experimental until '
                                 'correctness, memory, profiler, and performance evidence is '
                                 'recorded on supported AMD hardware.'},
        "triton_xpu": {'available': False,
                       'claim_status': 'unavailable',
                       'optimized': False,
                       'full_fusion': False,
                       'implemented_stage': _TRITON_STAGES,
                       'optimized_stage': None,
                       'weight_dequant_optimized': False,
                       'weight_pack_optimized': False,
                       'lowbit_unpack_optimized': False,
                       'weight_quant_optimized': False,
                       'adaln_quant_optimized': False,
                       'adaln_dequant_optimized': False,
                       'adaln_packed_matmul_optimized': False,
                       'device_types': ['xpu'],
                       'implementation': 'python_triton_orbitquant_pipeline_xpu_candidate',
                       'package_format': 'python_triton_xpu',
                       'hf_kernel_builder_compliant': False,
                       'notes': 'The candidate reuses the exact packed Triton pipeline on '
                                'torch.xpu. It remains explicit-only and experimental until '
                                'correctness, memory, profiler, and performance evidence is '
                                'recorded on supported Intel GPU hardware.'},
    },
    "cpu_adaln_only": {
        "cpu": {'available': True,
                'claim_status': 'partial_optimized',
                'optimized': True,
                'full_fusion': False,
                'implemented_stage': _CPU_STAGES,
                'optimized_stage': 'adaln_rtn_packed_matmul',
                'weight_dequant_optimized': False,
                'weight_pack_optimized': False,
                'lowbit_unpack_optimized': False,
                'weight_quant_optimized': False,
                'adaln_quant_optimized': False,
                'adaln_dequant_optimized': True,
                'adaln_packed_matmul_optimized': True,
                'device_types': ['cpu'],
                'implementation': 'torch_reference+native_packed_adaln_int4',
                'package_format': 'native_kernel_package_torch_stable_abi',
                'hf_kernel_builder_compliant': True,
                'notes': 'Activation quantization uses the reference PyTorch CPU path. The main '
                         'linear matmul uses the reference PyTorch CPU path. Native packed INT4 '
                         'AdaLN matmul consumes group-64 weights without a full floating-point '
                         'cache.'},
        "mps": {'available': False,
                'claim_status': 'reference_only',
                'optimized': False,
                'full_fusion': False,
                'implemented_stage': _MPS_STAGES,
                'optimized_stage': None,
                'weight_dequant_optimized': False,
                'weight_pack_optimized': False,
                'lowbit_unpack_optimized': False,
                'weight_quant_optimized': False,
                'adaln_quant_optimized': False,
                'adaln_dequant_optimized': False,
                'adaln_packed_matmul_optimized': False,
                'device_types': ['mps'],
                'implementation': 'torch_reference_mps',
                'package_format': 'torch_reference',
                'upstream_native_mps_op': False,
                'hf_kernel_builder_compliant': False,
                'notes': 'Runs the reference PyTorch path on MPS tensors; native Metal shader '
                         'support and the native packed matmul package are not available in this '
                         'environment.'},
        "triton_cuda": {'available': False,
                        'claim_status': 'unavailable',
                        'optimized': False,
                        'full_fusion': False,
                        'implemented_stage': _TRITON_STAGES,
                        'optimized_stage': None,
                        'weight_dequant_optimized': False,
                        'weight_pack_optimized': False,
                        'lowbit_unpack_optimized': False,
                        'weight_quant_optimized': False,
                        'adaln_quant_optimized': False,
                        'adaln_dequant_optimized': False,
                        'adaln_packed_matmul_optimized': False,
                        'device_types': ['cuda'],
                        'implementation': 'python_triton_orbitquant_pipeline',
                        'package_format': 'python_triton',
                        'hf_kernel_builder_compliant': False,
                        'notes': 'Triton handles runtime activation norm, RPBH/FWHT rotation, '
                                 'codebook lookup/rescale, packed weight dequant, packed weight '
                                 'matmul, offline low-bit pack/unpack, offline weight RPBH/FWHT '
                                 'codebook indexing with direct low-bit packing, and AdaLN INT4 '
                                 'RTN quantize/pack, dequant, and packed matmul. The default '
                                 'auto_fused runtime prefers packed low-bit matmul when a native '
                                 'or Triton packed kernel is available; full-model speedup '
                                 'claims still require separate benchmark artifacts.'},
        "triton_rocm": {'available': False,
                        'claim_status': 'unavailable',
                        'optimized': False,
                        'full_fusion': False,
                        'implemented_stage': _TRITON_STAGES,
                        'optimized_stage': None,
                        'weight_dequant_optimized': False,
                        'weight_pack_optimized': False,
                        'lowbit_unpack_optimized': False,
                        'weight_quant_optimized': False,
                        'adaln_quant_optimized': False,
                        'adaln_dequant_optimized': False,
                        'adaln_packed_matmul_optimized': False,
                        'device_types': ['cuda'],
                        'implementation': 'python_triton_orbitquant_pipeline_rocm_candidate',
                        'package_format': 'python_triton_rocm',
                        'hf_kernel_builder_compliant': False,
                        'notes': 'PyTorch exposes HIP tensors through the cuda device type. The '
                                 'candidate reuses the exact packed Triton pipeline without '
                                 'loading CUDA-native extensions. It remains experimental until '
                                 'correctness, memory, profiler, and performance evidence is '
                                 'recorded on supported AMD hardware.'},
        "triton_xpu": {'available': False,
                       'claim_status': 'unavailable',
                       'optimized': False,
                       'full_fusion': False,
                       'implemented_stage': _TRITON_STAGES,
                       'optimized_stage': None,
                       'weight_dequant_optimized': False,
                       'weight_pack_optimized': False,
                       'lowbit_unpack_optimized': False,
                       'weight_quant_optimized': False,
                       'adaln_quant_optimized': False,
                       'adaln_dequant_optimized': False,
                       'adaln_packed_matmul_optimized': False,
                       'device_types': ['xpu'],
                       'implementation': 'python_triton_orbitquant_pipeline_xpu_candidate',
                       'package_format': 'python_triton_xpu',
                       'hf_kernel_builder_compliant': False,
                       'notes': 'The candidate reuses the exact packed Triton pipeline on '
                                'torch.xpu. It remains explicit-only and experimental until '
                                'correctness, memory, profiler, and performance evidence is '
                                'recorded on supported Intel GPU hardware.'},
    },
    "mps_shader_and_native_matmul": {
        "cpu": {'available': True,
                'claim_status': 'reference_only',
                'optimized': False,
                'full_fusion': False,
                'implemented_stage': _CPU_STAGES,
                'optimized_stage': None,
                'weight_dequant_optimized': False,
                'weight_pack_optimized': False,
                'lowbit_unpack_optimized': False,
                'weight_quant_optimized': False,
                'adaln_quant_optimized': False,
                'adaln_dequant_optimized': False,
                'adaln_packed_matmul_optimized': False,
                'device_types': ['cpu'],
                'implementation': 'torch_reference',
                'package_format': 'torch_reference',
                'hf_kernel_builder_compliant': False,
                'notes': 'Activation quantization uses the reference PyTorch CPU path. The main '
                         'linear matmul uses the reference PyTorch CPU path. AdaLN uses the '
                         'explicit BF16 dequantization path.'},
        "mps": {'available': True,
                'claim_status': 'partial_optimized',
                'optimized': True,
                'full_fusion': False,
                'implemented_stage': _MPS_STAGES,
                'optimized_stage': _MPS_STAGES,
                'weight_dequant_optimized': True,
                'weight_pack_optimized': False,
                'lowbit_unpack_optimized': False,
                'weight_quant_optimized': False,
                'adaln_quant_optimized': False,
                'adaln_dequant_optimized': False,
                'adaln_packed_matmul_optimized': False,
                'device_types': ['mps'],
                'implementation': 'torch_mps_compile_shader_fused_activation+native_packed_matmul',
                'package_format': 'torch.mps.compile_shader,native_kernel_package',
                'upstream_native_mps_op': False,
                'hf_kernel_builder_compliant': False,
                'notes': 'Fused Metal shaders handle activation norm, RPBH/FWHT rotation, '
                         'codebook lookup/rescale, and packed weight dequant. The native packed '
                         'matmul package handles packed low-bit matmul.'},
        "triton_cuda": {'available': False,
                        'claim_status': 'unavailable',
                        'optimized': False,
                        'full_fusion': False,
                        'implemented_stage': _TRITON_STAGES,
                        'optimized_stage': None,
                        'weight_dequant_optimized': False,
                        'weight_pack_optimized': False,
                        'lowbit_unpack_optimized': False,
                        'weight_quant_optimized': False,
                        'adaln_quant_optimized': False,
                        'adaln_dequant_optimized': False,
                        'adaln_packed_matmul_optimized': False,
                        'device_types': ['cuda'],
                        'implementation': 'python_triton_orbitquant_pipeline',
                        'package_format': 'python_triton',
                        'hf_kernel_builder_compliant': False,
                        'notes': 'Triton handles runtime activation norm, RPBH/FWHT rotation, '
                                 'codebook lookup/rescale, packed weight dequant, packed weight '
                                 'matmul, offline low-bit pack/unpack, offline weight RPBH/FWHT '
                                 'codebook indexing with direct low-bit packing, and AdaLN INT4 '
                                 'RTN quantize/pack, dequant, and packed matmul. The default '
                                 'auto_fused runtime prefers packed low-bit matmul when a native '
                                 'or Triton packed kernel is available; full-model speedup '
                                 'claims still require separate benchmark artifacts.'},
        "triton_rocm": {'available': False,
                        'claim_status': 'unavailable',
                        'optimized': False,
                        'full_fusion': False,
                        'implemented_stage': _TRITON_STAGES,
                        'optimized_stage': None,
                        'weight_dequant_optimized': False,
                        'weight_pack_optimized': False,
                        'lowbit_unpack_optimized': False,
                        'weight_quant_optimized': False,
                        'adaln_quant_optimized': False,
                        'adaln_dequant_optimized': False,
                        'adaln_packed_matmul_optimized': False,
                        'device_types': ['cuda'],
                        'implementation': 'python_triton_orbitquant_pipeline_rocm_candidate',
                        'package_format': 'python_triton_rocm',
                        'hf_kernel_builder_compliant': False,
                        'notes': 'PyTorch exposes HIP tensors through the cuda device type. The '
                                 'candidate reuses the exact packed Triton pipeline without '
                                 'loading CUDA-native extensions. It remains experimental until '
                                 'correctness, memory, profiler, and performance evidence is '
                                 'recorded on supported AMD hardware.'},
        "triton_xpu": {'available': False,
                       'claim_status': 'unavailable',
                       'optimized': False,
                       'full_fusion': False,
                       'implemented_stage': _TRITON_STAGES,
                       'optimized_stage': None,
                       'weight_dequant_optimized': False,
                       'weight_pack_optimized': False,
                       'lowbit_unpack_optimized': False,
                       'weight_quant_optimized': False,
                       'adaln_quant_optimized': False,
                       'adaln_dequant_optimized': False,
                       'adaln_packed_matmul_optimized': False,
                       'device_types': ['xpu'],
                       'implementation': 'python_triton_orbitquant_pipeline_xpu_candidate',
                       'package_format': 'python_triton_xpu',
                       'hf_kernel_builder_compliant': False,
                       'notes': 'The candidate reuses the exact packed Triton pipeline on '
                                'torch.xpu. It remains explicit-only and experimental until '
                                'correctness, memory, profiler, and performance evidence is '
                                'recorded on supported Intel GPU hardware.'},
    },
    "mps_native_matmul_only": {
        "cpu": {'available': True,
                'claim_status': 'reference_only',
                'optimized': False,
                'full_fusion': False,
                'implemented_stage': _CPU_STAGES,
                'optimized_stage': None,
                'weight_dequant_optimized': False,
                'weight_pack_optimized': False,
                'lowbit_unpack_optimized': False,
                'weight_quant_optimized': False,
                'adaln_quant_optimized': False,
                'adaln_dequant_optimized': False,
                'adaln_packed_matmul_optimized': False,
                'device_types': ['cpu'],
                'implementation': 'torch_reference',
                'package_format': 'torch_reference',
                'hf_kernel_builder_compliant': False,
                'notes': 'Activation quantization uses the reference PyTorch CPU path. The main '
                         'linear matmul uses the reference PyTorch CPU path. AdaLN uses the '
                         'explicit BF16 dequantization path.'},
        "mps": {'available': True,
                'claim_status': 'partial_optimized',
                'optimized': True,
                'full_fusion': False,
                'implemented_stage': _MPS_STAGES,
                'optimized_stage': 'packed_weight_matmul',
                'weight_dequant_optimized': False,
                'weight_pack_optimized': False,
                'lowbit_unpack_optimized': False,
                'weight_quant_optimized': False,
                'adaln_quant_optimized': False,
                'adaln_dequant_optimized': False,
                'adaln_packed_matmul_optimized': False,
                'device_types': ['mps'],
                'implementation': 'native_packed_matmul',
                'package_format': 'native_kernel_package',
                'upstream_native_mps_op': False,
                'hf_kernel_builder_compliant': False,
                'notes': 'The native packed matmul package handles packed low-bit matmul. '
                         'Activation quantization helpers use the reference PyTorch path in this '
                         'environment.'},
        "triton_cuda": {'available': False,
                        'claim_status': 'unavailable',
                        'optimized': False,
                        'full_fusion': False,
                        'implemented_stage': _TRITON_STAGES,
                        'optimized_stage': None,
                        'weight_dequant_optimized': False,
                        'weight_pack_optimized': False,
                        'lowbit_unpack_optimized': False,
                        'weight_quant_optimized': False,
                        'adaln_quant_optimized': False,
                        'adaln_dequant_optimized': False,
                        'adaln_packed_matmul_optimized': False,
                        'device_types': ['cuda'],
                        'implementation': 'python_triton_orbitquant_pipeline',
                        'package_format': 'python_triton',
                        'hf_kernel_builder_compliant': False,
                        'notes': 'Triton handles runtime activation norm, RPBH/FWHT rotation, '
                                 'codebook lookup/rescale, packed weight dequant, packed weight '
                                 'matmul, offline low-bit pack/unpack, offline weight RPBH/FWHT '
                                 'codebook indexing with direct low-bit packing, and AdaLN INT4 '
                                 'RTN quantize/pack, dequant, and packed matmul. The default '
                                 'auto_fused runtime prefers packed low-bit matmul when a native '
                                 'or Triton packed kernel is available; full-model speedup '
                                 'claims still require separate benchmark artifacts.'},
        "triton_rocm": {'available': False,
                        'claim_status': 'unavailable',
                        'optimized': False,
                        'full_fusion': False,
                        'implemented_stage': _TRITON_STAGES,
                        'optimized_stage': None,
                        'weight_dequant_optimized': False,
                        'weight_pack_optimized': False,
                        'lowbit_unpack_optimized': False,
                        'weight_quant_optimized': False,
                        'adaln_quant_optimized': False,
                        'adaln_dequant_optimized': False,
                        'adaln_packed_matmul_optimized': False,
                        'device_types': ['cuda'],
                        'implementation': 'python_triton_orbitquant_pipeline_rocm_candidate',
                        'package_format': 'python_triton_rocm',
                        'hf_kernel_builder_compliant': False,
                        'notes': 'PyTorch exposes HIP tensors through the cuda device type. The '
                                 'candidate reuses the exact packed Triton pipeline without '
                                 'loading CUDA-native extensions. It remains experimental until '
                                 'correctness, memory, profiler, and performance evidence is '
                                 'recorded on supported AMD hardware.'},
        "triton_xpu": {'available': False,
                       'claim_status': 'unavailable',
                       'optimized': False,
                       'full_fusion': False,
                       'implemented_stage': _TRITON_STAGES,
                       'optimized_stage': None,
                       'weight_dequant_optimized': False,
                       'weight_pack_optimized': False,
                       'lowbit_unpack_optimized': False,
                       'weight_quant_optimized': False,
                       'adaln_quant_optimized': False,
                       'adaln_dequant_optimized': False,
                       'adaln_packed_matmul_optimized': False,
                       'device_types': ['xpu'],
                       'implementation': 'python_triton_orbitquant_pipeline_xpu_candidate',
                       'package_format': 'python_triton_xpu',
                       'hf_kernel_builder_compliant': False,
                       'notes': 'The candidate reuses the exact packed Triton pipeline on '
                                'torch.xpu. It remains explicit-only and experimental until '
                                'correctness, memory, profiler, and performance evidence is '
                                'recorded on supported Intel GPU hardware.'},
    },
    "triton_cuda": {
        "cpu": {'available': True,
                'claim_status': 'reference_only',
                'optimized': False,
                'full_fusion': False,
                'implemented_stage': _CPU_STAGES,
                'optimized_stage': None,
                'weight_dequant_optimized': False,
                'weight_pack_optimized': False,
                'lowbit_unpack_optimized': False,
                'weight_quant_optimized': False,
                'adaln_quant_optimized': False,
                'adaln_dequant_optimized': False,
                'adaln_packed_matmul_optimized': False,
                'device_types': ['cpu'],
                'implementation': 'torch_reference',
                'package_format': 'torch_reference',
                'hf_kernel_builder_compliant': False,
                'notes': 'Activation quantization uses the reference PyTorch CPU path. The main '
                         'linear matmul uses the reference PyTorch CPU path. AdaLN uses the '
                         'explicit BF16 dequantization path.'},
        "mps": {'available': False,
                'claim_status': 'reference_only',
                'optimized': False,
                'full_fusion': False,
                'implemented_stage': _MPS_STAGES,
                'optimized_stage': None,
                'weight_dequant_optimized': False,
                'weight_pack_optimized': False,
                'lowbit_unpack_optimized': False,
                'weight_quant_optimized': False,
                'adaln_quant_optimized': False,
                'adaln_dequant_optimized': False,
                'adaln_packed_matmul_optimized': False,
                'device_types': ['mps'],
                'implementation': 'torch_reference_mps',
                'package_format': 'torch_reference',
                'upstream_native_mps_op': False,
                'hf_kernel_builder_compliant': False,
                'notes': 'Runs the reference PyTorch path on MPS tensors; native Metal shader '
                         'support and the native packed matmul package are not available in this '
                         'environment.'},
        "triton_cuda": {'available': True,
                        'claim_status': 'partial_optimized',
                        'optimized': True,
                        'full_fusion': False,
                        'implemented_stage': _TRITON_STAGES,
                        'optimized_stage': _TRITON_STAGES,
                        'weight_dequant_optimized': True,
                        'weight_pack_optimized': True,
                        'lowbit_unpack_optimized': True,
                        'weight_quant_optimized': True,
                        'adaln_quant_optimized': True,
                        'adaln_dequant_optimized': True,
                        'adaln_packed_matmul_optimized': True,
                        'device_types': ['cuda'],
                        'implementation': 'python_triton_orbitquant_pipeline',
                        'package_format': 'python_triton',
                        'hf_kernel_builder_compliant': False,
                        'notes': 'Triton handles runtime activation norm, RPBH/FWHT rotation, '
                                 'codebook lookup/rescale, packed weight dequant, packed weight '
                                 'matmul, offline low-bit pack/unpack, offline weight RPBH/FWHT '
                                 'codebook indexing with direct low-bit packing, and AdaLN INT4 '
                                 'RTN quantize/pack, dequant, and packed matmul. The default '
                                 'auto_fused runtime prefers packed low-bit matmul when a native '
                                 'or Triton packed kernel is available; full-model speedup '
                                 'claims still require separate benchmark artifacts.'},
        "triton_rocm": {'available': False,
                        'claim_status': 'unavailable',
                        'optimized': False,
                        'full_fusion': False,
                        'implemented_stage': _TRITON_STAGES,
                        'optimized_stage': None,
                        'weight_dequant_optimized': False,
                        'weight_pack_optimized': False,
                        'lowbit_unpack_optimized': False,
                        'weight_quant_optimized': False,
                        'adaln_quant_optimized': False,
                        'adaln_dequant_optimized': False,
                        'adaln_packed_matmul_optimized': False,
                        'device_types': ['cuda'],
                        'implementation': 'python_triton_orbitquant_pipeline_rocm_candidate',
                        'package_format': 'python_triton_rocm',
                        'hf_kernel_builder_compliant': False,
                        'notes': 'PyTorch exposes HIP tensors through the cuda device type. The '
                                 'candidate reuses the exact packed Triton pipeline without '
                                 'loading CUDA-native extensions. It remains experimental until '
                                 'correctness, memory, profiler, and performance evidence is '
                                 'recorded on supported AMD hardware.'},
        "triton_xpu": {'available': False,
                       'claim_status': 'unavailable',
                       'optimized': False,
                       'full_fusion': False,
                       'implemented_stage': _TRITON_STAGES,
                       'optimized_stage': None,
                       'weight_dequant_optimized': False,
                       'weight_pack_optimized': False,
                       'lowbit_unpack_optimized': False,
                       'weight_quant_optimized': False,
                       'adaln_quant_optimized': False,
                       'adaln_dequant_optimized': False,
                       'adaln_packed_matmul_optimized': False,
                       'device_types': ['xpu'],
                       'implementation': 'python_triton_orbitquant_pipeline_xpu_candidate',
                       'package_format': 'python_triton_xpu',
                       'hf_kernel_builder_compliant': False,
                       'notes': 'The candidate reuses the exact packed Triton pipeline on '
                                'torch.xpu. It remains explicit-only and experimental until '
                                'correctness, memory, profiler, and performance evidence is '
                                'recorded on supported Intel GPU hardware.'},
    },
    "triton_rocm": {
        "cpu": {'available': True,
                'claim_status': 'reference_only',
                'optimized': False,
                'full_fusion': False,
                'implemented_stage': _CPU_STAGES,
                'optimized_stage': None,
                'weight_dequant_optimized': False,
                'weight_pack_optimized': False,
                'lowbit_unpack_optimized': False,
                'weight_quant_optimized': False,
                'adaln_quant_optimized': False,
                'adaln_dequant_optimized': False,
                'adaln_packed_matmul_optimized': False,
                'device_types': ['cpu'],
                'implementation': 'torch_reference',
                'package_format': 'torch_reference',
                'hf_kernel_builder_compliant': False,
                'notes': 'Activation quantization uses the reference PyTorch CPU path. The main '
                         'linear matmul uses the reference PyTorch CPU path. AdaLN uses the '
                         'explicit BF16 dequantization path.'},
        "mps": {'available': False,
                'claim_status': 'reference_only',
                'optimized': False,
                'full_fusion': False,
                'implemented_stage': _MPS_STAGES,
                'optimized_stage': None,
                'weight_dequant_optimized': False,
                'weight_pack_optimized': False,
                'lowbit_unpack_optimized': False,
                'weight_quant_optimized': False,
                'adaln_quant_optimized': False,
                'adaln_dequant_optimized': False,
                'adaln_packed_matmul_optimized': False,
                'device_types': ['mps'],
                'implementation': 'torch_reference_mps',
                'package_format': 'torch_reference',
                'upstream_native_mps_op': False,
                'hf_kernel_builder_compliant': False,
                'notes': 'Runs the reference PyTorch path on MPS tensors; native Metal shader '
                         'support and the native packed matmul package are not available in this '
                         'environment.'},
        "triton_cuda": {'available': False,
                        'claim_status': 'unavailable',
                        'optimized': False,
                        'full_fusion': False,
                        'implemented_stage': _TRITON_STAGES,
                        'optimized_stage': None,
                        'weight_dequant_optimized': False,
                        'weight_pack_optimized': False,
                        'lowbit_unpack_optimized': False,
                        'weight_quant_optimized': False,
                        'adaln_quant_optimized': False,
                        'adaln_dequant_optimized': False,
                        'adaln_packed_matmul_optimized': False,
                        'device_types': ['cuda'],
                        'implementation': 'python_triton_orbitquant_pipeline',
                        'package_format': 'python_triton',
                        'hf_kernel_builder_compliant': False,
                        'notes': 'Triton handles runtime activation norm, RPBH/FWHT rotation, '
                                 'codebook lookup/rescale, packed weight dequant, packed weight '
                                 'matmul, offline low-bit pack/unpack, offline weight RPBH/FWHT '
                                 'codebook indexing with direct low-bit packing, and AdaLN INT4 '
                                 'RTN quantize/pack, dequant, and packed matmul. The default '
                                 'auto_fused runtime prefers packed low-bit matmul when a native '
                                 'or Triton packed kernel is available; full-model speedup '
                                 'claims still require separate benchmark artifacts.'},
        "triton_rocm": {'available': True,
                        'claim_status': 'experimental_unverified',
                        'optimized': False,
                        'full_fusion': False,
                        'implemented_stage': _TRITON_STAGES,
                        'optimized_stage': None,
                        'weight_dequant_optimized': False,
                        'weight_pack_optimized': False,
                        'lowbit_unpack_optimized': False,
                        'weight_quant_optimized': False,
                        'adaln_quant_optimized': False,
                        'adaln_dequant_optimized': False,
                        'adaln_packed_matmul_optimized': False,
                        'device_types': ['cuda'],
                        'implementation': 'python_triton_orbitquant_pipeline_rocm_candidate',
                        'package_format': 'python_triton_rocm',
                        'hf_kernel_builder_compliant': False,
                        'notes': 'PyTorch exposes HIP tensors through the cuda device type. The '
                                 'candidate reuses the exact packed Triton pipeline without '
                                 'loading CUDA-native extensions. It remains experimental until '
                                 'correctness, memory, profiler, and performance evidence is '
                                 'recorded on supported AMD hardware.'},
        "triton_xpu": {'available': False,
                       'claim_status': 'unavailable',
                       'optimized': False,
                       'full_fusion': False,
                       'implemented_stage': _TRITON_STAGES,
                       'optimized_stage': None,
                       'weight_dequant_optimized': False,
                       'weight_pack_optimized': False,
                       'lowbit_unpack_optimized': False,
                       'weight_quant_optimized': False,
                       'adaln_quant_optimized': False,
                       'adaln_dequant_optimized': False,
                       'adaln_packed_matmul_optimized': False,
                       'device_types': ['xpu'],
                       'implementation': 'python_triton_orbitquant_pipeline_xpu_candidate',
                       'package_format': 'python_triton_xpu',
                       'hf_kernel_builder_compliant': False,
                       'notes': 'The candidate reuses the exact packed Triton pipeline on '
                                'torch.xpu. It remains explicit-only and experimental until '
                                'correctness, memory, profiler, and performance evidence is '
                                'recorded on supported Intel GPU hardware.'},
    },
    "triton_xpu": {
        "cpu": {'available': True,
                'claim_status': 'reference_only',
                'optimized': False,
                'full_fusion': False,
                'implemented_stage': _CPU_STAGES,
                'optimized_stage': None,
                'weight_dequant_optimized': False,
                'weight_pack_optimized': False,
                'lowbit_unpack_optimized': False,
                'weight_quant_optimized': False,
                'adaln_quant_optimized': False,
                'adaln_dequant_optimized': False,
                'adaln_packed_matmul_optimized': False,
                'device_types': ['cpu'],
                'implementation': 'torch_reference',
                'package_format': 'torch_reference',
                'hf_kernel_builder_compliant': False,
                'notes': 'Activation quantization uses the reference PyTorch CPU path. The main '
                         'linear matmul uses the reference PyTorch CPU path. AdaLN uses the '
                         'explicit BF16 dequantization path.'},
        "mps": {'available': False,
                'claim_status': 'reference_only',
                'optimized': False,
                'full_fusion': False,
                'implemented_stage': _MPS_STAGES,
                'optimized_stage': None,
                'weight_dequant_optimized': False,
                'weight_pack_optimized': False,
                'lowbit_unpack_optimized': False,
                'weight_quant_optimized': False,
                'adaln_quant_optimized': False,
                'adaln_dequant_optimized': False,
                'adaln_packed_matmul_optimized': False,
                'device_types': ['mps'],
                'implementation': 'torch_reference_mps',
                'package_format': 'torch_reference',
                'upstream_native_mps_op': False,
                'hf_kernel_builder_compliant': False,
                'notes': 'Runs the reference PyTorch path on MPS tensors; native Metal shader '
                         'support and the native packed matmul package are not available in this '
                         'environment.'},
        "triton_cuda": {'available': False,
                        'claim_status': 'unavailable',
                        'optimized': False,
                        'full_fusion': False,
                        'implemented_stage': _TRITON_STAGES,
                        'optimized_stage': None,
                        'weight_dequant_optimized': False,
                        'weight_pack_optimized': False,
                        'lowbit_unpack_optimized': False,
                        'weight_quant_optimized': False,
                        'adaln_quant_optimized': False,
                        'adaln_dequant_optimized': False,
                        'adaln_packed_matmul_optimized': False,
                        'device_types': ['cuda'],
                        'implementation': 'python_triton_orbitquant_pipeline',
                        'package_format': 'python_triton',
                        'hf_kernel_builder_compliant': False,
                        'notes': 'Triton handles runtime activation norm, RPBH/FWHT rotation, '
                                 'codebook lookup/rescale, packed weight dequant, packed weight '
                                 'matmul, offline low-bit pack/unpack, offline weight RPBH/FWHT '
                                 'codebook indexing with direct low-bit packing, and AdaLN INT4 '
                                 'RTN quantize/pack, dequant, and packed matmul. The default '
                                 'auto_fused runtime prefers packed low-bit matmul when a native '
                                 'or Triton packed kernel is available; full-model speedup '
                                 'claims still require separate benchmark artifacts.'},
        "triton_rocm": {'available': False,
                        'claim_status': 'unavailable',
                        'optimized': False,
                        'full_fusion': False,
                        'implemented_stage': _TRITON_STAGES,
                        'optimized_stage': None,
                        'weight_dequant_optimized': False,
                        'weight_pack_optimized': False,
                        'lowbit_unpack_optimized': False,
                        'weight_quant_optimized': False,
                        'adaln_quant_optimized': False,
                        'adaln_dequant_optimized': False,
                        'adaln_packed_matmul_optimized': False,
                        'device_types': ['cuda'],
                        'implementation': 'python_triton_orbitquant_pipeline_rocm_candidate',
                        'package_format': 'python_triton_rocm',
                        'hf_kernel_builder_compliant': False,
                        'notes': 'PyTorch exposes HIP tensors through the cuda device type. The '
                                 'candidate reuses the exact packed Triton pipeline without '
                                 'loading CUDA-native extensions. It remains experimental until '
                                 'correctness, memory, profiler, and performance evidence is '
                                 'recorded on supported AMD hardware.'},
        "triton_xpu": {'available': True,
                       'claim_status': 'experimental_unverified',
                       'optimized': False,
                       'full_fusion': False,
                       'implemented_stage': _TRITON_STAGES,
                       'optimized_stage': None,
                       'weight_dequant_optimized': False,
                       'weight_pack_optimized': False,
                       'lowbit_unpack_optimized': False,
                       'weight_quant_optimized': False,
                       'adaln_quant_optimized': False,
                       'adaln_dequant_optimized': False,
                       'adaln_packed_matmul_optimized': False,
                       'device_types': ['xpu'],
                       'implementation': 'python_triton_orbitquant_pipeline_xpu_candidate',
                       'package_format': 'python_triton_xpu',
                       'hf_kernel_builder_compliant': False,
                       'notes': 'The candidate reuses the exact packed Triton pipeline on '
                                'torch.xpu. It remains explicit-only and experimental until '
                                'correctness, memory, profiler, and performance evidence is '
                                'recorded on supported Intel GPU hardware.'},
    },
    "mixed": {
        "cpu": {'available': True,
                'claim_status': 'partial_optimized',
                'optimized': True,
                'full_fusion': False,
                'implemented_stage': _CPU_STAGES,
                'optimized_stage': 'packed_weight_matmul,adaln_rtn_packed_matmul',
                'weight_dequant_optimized': False,
                'weight_pack_optimized': False,
                'lowbit_unpack_optimized': False,
                'weight_quant_optimized': False,
                'adaln_quant_optimized': False,
                'adaln_dequant_optimized': True,
                'adaln_packed_matmul_optimized': True,
                'device_types': ['cpu'],
                'implementation': _CPU_IMPL_MATMUL_ADALN,
                'package_format': 'native_kernel_package_torch_stable_abi',
                'hf_kernel_builder_compliant': True,
                'notes': 'Activation quantization uses the reference PyTorch CPU path. Native '
                         'exact packed matmul consumes low-bit weights without a full '
                         'floating-point cache. Native packed INT4 AdaLN matmul consumes '
                         'group-64 weights without a full floating-point cache.'},
        "mps": {'available': True,
                'claim_status': 'partial_optimized',
                'optimized': True,
                'full_fusion': False,
                'implemented_stage': _MPS_STAGES,
                'optimized_stage': 'activation_norm_rpbh_quant_rescale,packed_weight_dequant',
                'weight_dequant_optimized': True,
                'weight_pack_optimized': False,
                'lowbit_unpack_optimized': False,
                'weight_quant_optimized': False,
                'adaln_quant_optimized': False,
                'adaln_dequant_optimized': False,
                'adaln_packed_matmul_optimized': False,
                'device_types': ['mps'],
                'implementation': 'torch_mps_compile_shader_fused_activation',
                'package_format': 'torch.mps.compile_shader',
                'upstream_native_mps_op': False,
                'hf_kernel_builder_compliant': False,
                'notes': 'Fused Metal shaders handle activation norm, RPBH/FWHT rotation, '
                         'codebook lookup/rescale, and packed weight dequant. The native packed '
                         'matmul package is not available in this environment.'},
        "triton_cuda": {'available': True,
                        'claim_status': 'partial_optimized',
                        'optimized': True,
                        'full_fusion': False,
                        'implemented_stage': _TRITON_STAGES,
                        'optimized_stage': _TRITON_STAGES,
                        'weight_dequant_optimized': True,
                        'weight_pack_optimized': True,
                        'lowbit_unpack_optimized': True,
                        'weight_quant_optimized': True,
                        'adaln_quant_optimized': True,
                        'adaln_dequant_optimized': True,
                        'adaln_packed_matmul_optimized': True,
                        'device_types': ['cuda'],
                        'implementation': 'python_triton_orbitquant_pipeline',
                        'package_format': 'python_triton',
                        'hf_kernel_builder_compliant': False,
                        'notes': 'Triton handles runtime activation norm, RPBH/FWHT rotation, '
                                 'codebook lookup/rescale, packed weight dequant, packed weight '
                                 'matmul, offline low-bit pack/unpack, offline weight RPBH/FWHT '
                                 'codebook indexing with direct low-bit packing, and AdaLN INT4 '
                                 'RTN quantize/pack, dequant, and packed matmul. The default '
                                 'auto_fused runtime prefers packed low-bit matmul when a native '
                                 'or Triton packed kernel is available; full-model speedup '
                                 'claims still require separate benchmark artifacts.'},
        "triton_rocm": {'available': False,
                        'claim_status': 'unavailable',
                        'optimized': False,
                        'full_fusion': False,
                        'implemented_stage': _TRITON_STAGES,
                        'optimized_stage': None,
                        'weight_dequant_optimized': False,
                        'weight_pack_optimized': False,
                        'lowbit_unpack_optimized': False,
                        'weight_quant_optimized': False,
                        'adaln_quant_optimized': False,
                        'adaln_dequant_optimized': False,
                        'adaln_packed_matmul_optimized': False,
                        'device_types': ['cuda'],
                        'implementation': 'python_triton_orbitquant_pipeline_rocm_candidate',
                        'package_format': 'python_triton_rocm',
                        'hf_kernel_builder_compliant': False,
                        'notes': 'PyTorch exposes HIP tensors through the cuda device type. The '
                                 'candidate reuses the exact packed Triton pipeline without '
                                 'loading CUDA-native extensions. It remains experimental until '
                                 'correctness, memory, profiler, and performance evidence is '
                                 'recorded on supported AMD hardware.'},
        "triton_xpu": {'available': True,
                       'claim_status': 'experimental_unverified',
                       'optimized': False,
                       'full_fusion': False,
                       'implemented_stage': _TRITON_STAGES,
                       'optimized_stage': None,
                       'weight_dequant_optimized': False,
                       'weight_pack_optimized': False,
                       'lowbit_unpack_optimized': False,
                       'weight_quant_optimized': False,
                       'adaln_quant_optimized': False,
                       'adaln_dequant_optimized': False,
                       'adaln_packed_matmul_optimized': False,
                       'device_types': ['xpu'],
                       'implementation': 'python_triton_orbitquant_pipeline_xpu_candidate',
                       'package_format': 'python_triton_xpu',
                       'hf_kernel_builder_compliant': False,
                       'notes': 'The candidate reuses the exact packed Triton pipeline on '
                                'torch.xpu. It remains explicit-only and experimental until '
                                'correctness, memory, profiler, and performance evidence is '
                                'recorded on supported Intel GPU hardware.'},
    },
}


@pytest.mark.parametrize("case_id", list(CASES))
def test_backend_capabilities_matches_golden_snapshot(
    case_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe_overrides, backend_flags = CASES[case_id]
    for attr, value in probe_overrides.items():
        monkeypatch.setattr(dispatch_module, attr, lambda value=value: value)
    result = backend_capabilities(backends=dict(backend_flags))
    expected = EXPECTED[case_id]
    assert result == expected
    # Dict equality ignores insertion order; freeze the key order as well.
    assert list(result) == list(expected)
    for backend_name, entry in result.items():
        assert list(entry) == list(expected[backend_name])


def test_golden_grid_is_consistent() -> None:
    assert set(CASES) == set(EXPECTED)
