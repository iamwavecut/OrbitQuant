import json

import pytest

import orbitquant.cli.main as cli_main
from orbitquant import __version__
from orbitquant.cli.main import main
from orbitquant.config import OrbitQuantConfig


def test_cli_version_prints_version(capsys):
    assert main(["--version"]) == 0

    output = capsys.readouterr().out
    assert __version__ in output


def test_cli_packed_matmul_runtime_modes_skip_dequant_prewarm():
    assert cli_main._should_prewarm_quantized_weights(None) is True
    assert cli_main._should_prewarm_quantized_weights(OrbitQuantConfig()) is False
    assert (
        cli_main._should_prewarm_quantized_weights(OrbitQuantConfig(runtime_mode="dequant_bf16"))
        is True
    )
    assert (
        cli_main._should_prewarm_quantized_weights(
            OrbitQuantConfig(runtime_mode="triton_packed_matmul")
        )
        is False
    )
    assert (
        cli_main._should_prewarm_quantized_weights(
            OrbitQuantConfig(runtime_mode="native_packed_matmul")
        )
        is False
    )


def test_cli_generation_placement_uses_model_cpu_offload_when_requested():
    class TinyPipeline:
        def __init__(self):
            self.calls = []

        def to(self, device):
            self.calls.append(("to", device))
            return self

        def enable_model_cpu_offload(self, *, device):
            self.calls.append(("offload", device))

    pipeline = TinyPipeline()
    cli_main._place_pipeline_for_generation(
        pipeline,
        device="cuda",
        enable_model_cpu_offload=True,
    )

    assert pipeline.calls == [("offload", "cuda")]


def test_cli_generation_placement_fails_loud_when_offload_is_missing():
    class TinyPipeline:
        def to(self, device):
            return self

    with pytest.raises(RuntimeError, match="enable_model_cpu_offload"):
        cli_main._place_pipeline_for_generation(
            TinyPipeline(),
            device="cuda",
            enable_model_cpu_offload=True,
        )


def test_cli_native_suites_lists_no_range_smoke_settings(capsys):
    assert main(["native-suites"]) == 0

    output = capsys.readouterr().out
    assert "flux2-native" in output
    assert "wan-native" in output
    assert "range" not in output.lower()


def test_cli_kernel_info_reports_backend_capabilities(capsys, monkeypatch):
    monkeypatch.setattr(
        cli_main,
        "backend_capabilities",
        lambda: {
            "cpu": {
                "available": True,
                "claim_status": "reference_only",
                "optimized": False,
                "implemented_stage": None,
                "optimized_stage": None,
                "weight_dequant_optimized": False,
                "weight_pack_optimized": False,
                "weight_quant_optimized": False,
                "adaln_quant_optimized": False,
                "adaln_dequant_optimized": False,
                "hf_kernel_builder_compliant": False,
            },
            "mps": {
                "claim_status": "partial_optimized",
                "implementation": "torch_mps_compile_shader_codebook_rescale+native_packed_matmul",
                "package_format": "torch.mps.compile_shader,native_kernel_package",
                "implemented_stage": (
                    "codebook_lookup_rescale,packed_weight_dequant,packed_weight_matmul"
                ),
                "optimized_stage": (
                    "codebook_lookup_rescale,packed_weight_dequant,packed_weight_matmul"
                ),
                "weight_dequant_optimized": True,
                "weight_pack_optimized": False,
                "weight_quant_optimized": False,
                "adaln_quant_optimized": False,
                "adaln_dequant_optimized": False,
                "full_fusion": False,
                "upstream_native_mps_op": False,
                "hf_kernel_builder_compliant": False,
            },
            "triton_cuda": {
                "claim_status": "partial_optimized",
                "implemented_stage": (
                    "activation_norm_rpbh_quant_rescale,packed_weight_dequant,"
                    "packed_weight_matmul,lowbit_pack,lowbit_unpack,"
                    "weight_rotation_fwht_quant_pack,"
                    "adaln_rtn_quant_pack,adaln_rtn_dequant,"
                    "adaln_rtn_packed_matmul"
                ),
                "optimized_stage": (
                    "activation_norm_rpbh_quant_rescale,packed_weight_dequant,"
                    "packed_weight_matmul,lowbit_pack,lowbit_unpack,"
                    "weight_rotation_fwht_quant_pack,"
                    "adaln_rtn_quant_pack,adaln_rtn_dequant,"
                    "adaln_rtn_packed_matmul"
                ),
                "implementation": "python_triton_orbitquant_pipeline",
                "package_format": "python_triton",
                "weight_dequant_optimized": True,
                "weight_pack_optimized": True,
                "weight_quant_optimized": True,
                "adaln_quant_optimized": True,
                "adaln_dequant_optimized": True,
                "full_fusion": False,
                "hf_kernel_builder_compliant": False,
            },
        },
    )

    assert main(["kernel-info"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["cpu"]["available"] is True
    assert payload["cpu"]["claim_status"] == "reference_only"
    assert payload["cpu"]["optimized"] is False
    assert payload["cpu"]["weight_dequant_optimized"] is False
    assert payload["cpu"]["weight_pack_optimized"] is False
    assert payload["cpu"]["weight_quant_optimized"] is False
    assert payload["cpu"]["adaln_quant_optimized"] is False
    assert payload["cpu"]["adaln_dequant_optimized"] is False
    assert (
        payload["mps"]["implementation"]
        == "torch_mps_compile_shader_codebook_rescale+native_packed_matmul"
    )
    assert payload["mps"]["package_format"] == "torch.mps.compile_shader,native_kernel_package"
    assert payload["mps"]["claim_status"] == "partial_optimized"
    assert (
        payload["mps"]["implemented_stage"]
        == "codebook_lookup_rescale,packed_weight_dequant,packed_weight_matmul"
    )
    assert (
        payload["mps"]["optimized_stage"]
        == "codebook_lookup_rescale,packed_weight_dequant,packed_weight_matmul"
    )
    assert payload["mps"]["weight_dequant_optimized"] is True
    assert payload["mps"]["weight_pack_optimized"] is False
    assert payload["mps"]["weight_quant_optimized"] is False
    assert payload["mps"]["adaln_quant_optimized"] is False
    assert payload["mps"]["adaln_dequant_optimized"] is False
    assert payload["mps"]["full_fusion"] is False
    assert payload["mps"]["upstream_native_mps_op"] is False
    assert payload["mps"]["hf_kernel_builder_compliant"] is False
    expected_triton_stage = (
        "activation_norm_rpbh_quant_rescale,packed_weight_dequant,"
        "packed_weight_matmul,lowbit_pack,lowbit_unpack,"
        "weight_rotation_fwht_quant_pack,"
        "adaln_rtn_quant_pack,adaln_rtn_dequant,adaln_rtn_packed_matmul"
    )
    assert payload["triton_cuda"]["implemented_stage"] == expected_triton_stage
    assert payload["triton_cuda"]["optimized_stage"] == expected_triton_stage
    assert payload["triton_cuda"]["implementation"] == "python_triton_orbitquant_pipeline"
    assert payload["triton_cuda"]["package_format"] == "python_triton"
    assert payload["triton_cuda"]["claim_status"] == "partial_optimized"
    assert payload["triton_cuda"]["weight_dequant_optimized"] is True
    assert payload["triton_cuda"]["weight_pack_optimized"] is True
    assert payload["triton_cuda"]["weight_quant_optimized"] is True
    assert payload["triton_cuda"]["adaln_quant_optimized"] is True
    assert payload["triton_cuda"]["adaln_dequant_optimized"] is True
    assert payload["triton_cuda"]["full_fusion"] is False
    assert payload["triton_cuda"]["hf_kernel_builder_compliant"] is False


def test_cli_kernel_bench_prints_stage_timings(capsys):
    assert (
        main(
            [
                "kernel-bench",
                "--tokens",
                "4",
                "--in-features",
                "16",
                "--out-features",
                "8",
                "--block-size",
                "8",
                "--activation-kernel-backend",
                "cpu",
                "--runtime-mode",
                "debug_no_activation_quant",
                "--device",
                "cpu",
                "--dtype",
                "float32",
                "--warmup",
                "0",
                "--iterations",
                "1",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["device"] == "cpu"
    assert payload["runtime_mode"] == "debug_no_activation_quant"
    assert payload["full_fusion"] is False
    assert payload["prewarm"]["total_modules"] == 1
    assert payload["timings_ms"]["weight_quantize_pack_cold_ms"] >= 0.0
    assert payload["timings_ms"]["weight_quantize_pack_hot_ms"] >= 0.0
    assert payload["timings_ms"]["forward_prewarmed_ms"] >= 0.0
    assert payload["selected_activation_kernel_backend"] == "cpu"
    assert payload["weight_quantization_backend"] == "torch_reference"
    assert payload["quantization_buffers"]["source_weight_device"] == "cpu"
    assert payload["quantization_buffers"]["source_weight_is_cuda"] is False
    assert payload["quantization_buffers"]["packed_weight_indices_device"] == "cpu"


@pytest.mark.parametrize("runtime_mode", ["triton_packed_matmul", "native_packed_matmul"])
def test_cli_kernel_bench_passes_packed_matmul_tile_options(monkeypatch, capsys, runtime_mode):
    seen_kwargs = []

    def fake_benchmark_orbit_linear(**kwargs):
        seen_kwargs.append(kwargs)
        return {"ok": True}

    monkeypatch.setattr(cli_main, "benchmark_orbit_linear", fake_benchmark_orbit_linear)

    assert (
        main(
            [
                "kernel-bench",
                "--runtime-mode",
                runtime_mode,
                "--packed-matmul-block-m",
                "32",
                "--packed-matmul-block-n",
                "64",
                "--packed-matmul-block-k",
                "64",
                "--packed-matmul-num-warps",
                "8",
            ]
        )
        == 0
    )

    assert json.loads(capsys.readouterr().out) == {"ok": True}
    assert seen_kwargs[0]["runtime_mode"] == runtime_mode
    assert seen_kwargs[0]["packed_matmul_block_m"] == 32
    assert seen_kwargs[0]["packed_matmul_block_n"] == 64
    assert seen_kwargs[0]["packed_matmul_block_k"] == 64
    assert seen_kwargs[0]["packed_matmul_num_warps"] == 8


def test_cli_quantize_bench_prints_full_model_staging_timings(capsys):
    assert (
        main(
            [
                "quantize-bench",
                "--layers",
                "1",
                "--in-features",
                "16",
                "--hidden-features",
                "32",
                "--block-size",
                "8",
                "--source-device",
                "cpu",
                "--quantization-device",
                "cpu",
                "--staging-mode",
                "component",
                "--dtype",
                "float32",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["source_device"] == "cpu"
    assert payload["quantization_device"] == "cpu"
    assert payload["staging_mode"] == "component"
    assert payload["synchronize_per_module"] is False
    assert payload["summary"]["quantization_staging_mode"] == "component"
    assert payload["summary"]["synchronize_per_module"] is False
    assert payload["summary"]["source_linear_device_counts"]["cpu"] == 7
    assert payload["summary"]["device_transfer_seconds"] >= 0.0
    assert payload["summary"]["quantized_modules"]
