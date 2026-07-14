import json
import sys
from types import SimpleNamespace

import torch
from PIL import Image

import orbitquant.cli.main as cli_main
from orbitquant.artifacts import save_orbitquant_artifact, validate_orbitquant_artifact
from orbitquant.cli.main import main
from orbitquant.config import OrbitQuantConfig
from orbitquant.eval.prompts import default_prompt_payload, select_prompt_record
from orbitquant.modeling import quantize_linear_modules


def test_cli_generate_pack_dry_run_lists_prompt_seed_jobs(capsys, tmp_path):
    model = torch.nn.Module()
    model.transformer_blocks = torch.nn.ModuleList(
        [torch.nn.ModuleDict({"attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(8, 8)})})]
    )
    config = OrbitQuantConfig(block_size=4, target_policy="generic_dit")
    summary = quantize_linear_modules(model, config)
    save_orbitquant_artifact(
        model,
        tmp_path,
        config=config,
        source_model_id="example/artifact-model",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )

    assert (
        main(
            [
                "generate-pack",
                "--suite",
                "flux2-native",
                "--artifact",
                str(tmp_path),
                "--prompt-limit",
                "2",
                "--seeds",
                "0,1",
                "--device",
                "cpu",
                "--dry-run",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["job_count"] == 4
    assert output["jobs"][0]["prompt_record"]["id"] == "simple-object"
    assert output["jobs"][0]["seed"] == 0
    assert output["jobs"][2]["seed"] == 1
    assert output["output"] == str(tmp_path / "assets")


def test_cli_generate_pack_dry_run_accepts_geneval_smoke_prompt_pack(capsys, tmp_path):
    model = torch.nn.Module()
    model.transformer_blocks = torch.nn.ModuleList(
        [torch.nn.ModuleDict({"attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(8, 8)})})]
    )
    config = OrbitQuantConfig(block_size=4, target_policy="generic_dit")
    summary = quantize_linear_modules(model, config)
    save_orbitquant_artifact(
        model,
        tmp_path,
        config=config,
        source_model_id="example/artifact-model",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )

    assert (
        main(
            [
                "generate-pack",
                "--suite",
                "flux1-schnell-native",
                "--artifact",
                str(tmp_path),
                "--prompt-pack",
                "geneval-smoke",
                "--prompt-limit",
                "2",
                "--seeds",
                "0",
                "--device",
                "cpu",
                "--dry-run",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["prompt_pack"] == "geneval_smoke_v1"
    assert output["job_count"] == 2
    assert output["jobs"][0]["prompt_record"]["id"].startswith("geneval-00000-")
    assert output["jobs"][0]["prompt_record"]["geneval"]["tag"] == "single_object"


def test_cli_generate_pack_dry_run_accepts_geneval_metadata_jsonl(capsys, tmp_path):
    model = torch.nn.Module()
    model.transformer_blocks = torch.nn.ModuleList(
        [torch.nn.ModuleDict({"attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(8, 8)})})]
    )
    config = OrbitQuantConfig(block_size=4, target_policy="generic_dit")
    summary = quantize_linear_modules(model, config)
    save_orbitquant_artifact(
        model,
        tmp_path,
        config=config,
        source_model_id="example/artifact-model",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )
    metadata_jsonl = tmp_path / "evaluation_metadata.jsonl"
    metadata_jsonl.write_text(
        json.dumps(
            {
                "tag": "single_object",
                "include": [{"class": "bench", "count": 1}],
                "prompt": "a photo of a bench",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert (
        main(
            [
                "generate-pack",
                "--suite",
                "z-image-native",
                "--artifact",
                str(tmp_path),
                "--prompt-metadata-jsonl",
                str(metadata_jsonl),
                "--seeds",
                "4",
                "--device",
                "cpu",
                "--dry-run",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["prompt_pack"] == "geneval_metadata_jsonl"
    assert output["job_count"] == 1
    assert output["jobs"][0]["seed"] == 4
    assert output["jobs"][0]["prompt_record"]["geneval"]["include"] == [
        {"class": "bench", "count": 1}
    ]


def test_cli_generate_pack_runs_jobs_once_per_prompt_seed_and_records_artifacts(
    monkeypatch,
    capsys,
    tmp_path,
):
    class TinyPipeline:
        def __init__(self):
            self.transformer = torch.nn.Module()
            self.transformer.transformer_blocks = torch.nn.ModuleList(
                [
                    torch.nn.ModuleDict(
                        {"attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(8, 8)})}
                    )
                ]
            )
            self.calls = []

        def to(self, device):
            self.device = device
            return self

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(images=[Image.new("RGB", (16, 16), "green")])

    source = TinyPipeline()
    config = OrbitQuantConfig(block_size=4, target_policy="generic_dit")
    summary = quantize_linear_modules(source.transformer, config)
    save_orbitquant_artifact(
        source.transformer,
        tmp_path,
        config=config,
        source_model_id="example/artifact-model",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )
    restored = TinyPipeline()

    class FakeDiffusionPipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            assert model_id == "example/artifact-model"
            assert kwargs["torch_dtype"] is torch.float32
            assert kwargs["revision"] == "abc123"
            return restored

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(Flux2KleinPipeline=FakeDiffusionPipeline),
    )

    assert (
        main(
            [
                "generate-pack",
                "--suite",
                "flux2-native",
                "--artifact",
                str(tmp_path),
                "--prompt-id",
                "simple-object",
                "--prompt-id",
                "counting",
                "--seeds",
                "3",
                "--device",
                "cpu",
                "--dtype",
                "float32",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    manifest = json.loads((tmp_path / "orbitquant_manifest.json").read_text())
    metrics_rows = (tmp_path / "benchmark" / "orbitquant.metrics.jsonl").read_text().splitlines()
    assert output["job_count"] == 2
    assert len(restored.calls) == 2
    assert "assets/flux2-native_seed3_W4A4_simple-object.png" in manifest["checksums"]
    assert "assets/flux2-native_seed3_W4A4_counting.png" in manifest["checksums"]
    assert len(metrics_rows) == 2
    assert json.loads(metrics_rows[0])["metadata"]["prompt_record"]["id"] == "simple-object"


def test_cli_generate_pack_with_packed_runtime_artifact_skips_dequant_prewarm(
    monkeypatch,
    capsys,
    tmp_path,
):
    class TinyPipeline:
        def __init__(self):
            self.transformer = torch.nn.Module()
            self.transformer.transformer_blocks = torch.nn.ModuleList(
                [
                    torch.nn.ModuleDict(
                        {"attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(8, 8)})}
                    )
                ]
            )

        def to(self, device):
            return self

        def __call__(self, **kwargs):
            return SimpleNamespace(images=[Image.new("RGB", (16, 16), "green")])

    source = TinyPipeline()
    config = OrbitQuantConfig(
        block_size=4,
        target_policy="generic_dit",
        runtime_mode="triton_packed_matmul",
    )
    summary = quantize_linear_modules(source.transformer, config)
    save_orbitquant_artifact(
        source.transformer,
        tmp_path,
        config=config,
        source_model_id="example/artifact-model",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )

    class FakeDiffusionPipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            assert model_id == "example/artifact-model"
            return TinyPipeline()

    def fail_prewarm(*args, **kwargs):
        raise AssertionError("packed runtime must not materialize dequant prewarm")

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(Flux2KleinPipeline=FakeDiffusionPipeline),
    )
    monkeypatch.setattr(cli_main, "_prewarm_pipeline_component", fail_prewarm)

    assert (
        main(
            [
                "generate-pack",
                "--suite",
                "flux2-native",
                "--artifact",
                str(tmp_path),
                "--prompt-id",
                "simple-object",
                "--seeds",
                "3",
                "--device",
                "cpu",
                "--dtype",
                "float32",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["run_count"] == 1
    assert output["outputs"][0]["output_path"].endswith("flux2-native_seed3_W4A4_simple-object.png")


def test_cli_generate_pack_skip_checksums_refreshes_artifact_once_at_end(
    monkeypatch,
    capsys,
    tmp_path,
):
    class TinyPipeline:
        def __init__(self):
            self.transformer = torch.nn.Module()
            self.transformer.transformer_blocks = torch.nn.ModuleList(
                [
                    torch.nn.ModuleDict(
                        {"attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(8, 8)})}
                    )
                ]
            )
            self.calls = []

        def to(self, device):
            return self

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(images=[Image.new("RGB", (16, 16), "green")])

    source = TinyPipeline()
    config = OrbitQuantConfig(block_size=4, target_policy="generic_dit")
    summary = quantize_linear_modules(source.transformer, config)
    save_orbitquant_artifact(
        source.transformer,
        tmp_path,
        config=config,
        source_model_id="example/artifact-model",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )
    restored = TinyPipeline()

    class FakeDiffusionPipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            assert model_id == "example/artifact-model"
            return restored

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(Flux2KleinPipeline=FakeDiffusionPipeline),
    )

    assert (
        main(
            [
                "generate-pack",
                "--suite",
                "flux2-native",
                "--artifact",
                str(tmp_path),
                "--prompt-id",
                "simple-object",
                "--seeds",
                "3",
                "--resume-existing",
                "--skip-artifact-checksums",
                "--device",
                "cpu",
                "--dtype",
                "float32",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    manifest = json.loads((tmp_path / "orbitquant_manifest.json").read_text())
    validation = validate_orbitquant_artifact(tmp_path)
    image_path = "assets/flux2-native_seed3_W4A4_simple-object.png"
    metadata_path = "assets/flux2-native_seed3_W4A4_simple-object.png.json"
    assert output["job_count"] == 1
    assert output["run_count"] == 1
    assert output["checksum_refresh"]["checksum_count"] == len(manifest["checksums"])
    assert validation["valid"] is True
    assert image_path in manifest["checksums"]
    assert metadata_path in manifest["checksums"]
    assert "benchmark/orbitquant.metrics.jsonl" in manifest["checksums"]
    assert output["artifact_comparisons"] == []
    assert output["outputs"][0]["comparisons"] == []


def test_cli_generate_pack_defers_comparison_creation_until_after_jobs(
    monkeypatch,
    capsys,
    tmp_path,
):
    class TinyPipeline:
        def __init__(self):
            self.transformer = torch.nn.Module()
            self.transformer.transformer_blocks = torch.nn.ModuleList(
                [
                    torch.nn.ModuleDict(
                        {"attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(8, 8)})}
                    )
                ]
            )

        def to(self, device):
            return self

        def __call__(self, **kwargs):
            return SimpleNamespace(images=[Image.new("RGB", (16, 16), "green")])

    source = TinyPipeline()
    config = OrbitQuantConfig(block_size=4, target_policy="generic_dit")
    summary = quantize_linear_modules(source.transformer, config)
    save_orbitquant_artifact(
        source.transformer,
        tmp_path,
        config=config,
        source_model_id="example/artifact-model",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )

    class FakeDiffusionPipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            assert model_id == "example/artifact-model"
            return TinyPipeline()

    comparison_calls = []

    def fake_create_comparisons(artifact_dir, **kwargs):
        comparison_calls.append({"artifact_dir": artifact_dir, "kwargs": kwargs})
        return ["assets/fake-comparison.webp"]

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(Flux2KleinPipeline=FakeDiffusionPipeline),
    )
    monkeypatch.setattr(cli_main, "create_artifact_image_comparisons", fake_create_comparisons)

    assert (
        main(
            [
                "generate-pack",
                "--suite",
                "flux2-native",
                "--artifact",
                str(tmp_path),
                "--prompt-id",
                "simple-object",
                "--prompt-id",
                "counting",
                "--seeds",
                "3",
                "--device",
                "cpu",
                "--dtype",
                "float32",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert len(comparison_calls) == 1
    assert comparison_calls[0]["kwargs"]["comparison_keys"] == {
        ("flux2-native", 3, "simple-object"),
        ("flux2-native", 3, "counting"),
    }
    assert output["run_count"] == 2
    assert output["artifact_comparisons"] == ["assets/fake-comparison.webp"]
    assert [item["comparisons"] for item in output["outputs"]] == [[], []]


def test_cli_generate_pack_prompt_metadata_disables_comparisons_by_default(
    monkeypatch,
    capsys,
    tmp_path,
):
    class TinyPipeline:
        def __init__(self):
            self.transformer = torch.nn.Module()
            self.transformer.transformer_blocks = torch.nn.ModuleList(
                [
                    torch.nn.ModuleDict(
                        {"attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(8, 8)})}
                    )
                ]
            )

        def to(self, device):
            return self

        def __call__(self, **kwargs):
            return SimpleNamespace(images=[Image.new("RGB", (16, 16), "green")])

    source = TinyPipeline()
    config = OrbitQuantConfig(block_size=4, target_policy="generic_dit")
    summary = quantize_linear_modules(source.transformer, config)
    save_orbitquant_artifact(
        source.transformer,
        tmp_path,
        config=config,
        source_model_id="example/artifact-model",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )
    metadata_jsonl = tmp_path / "evaluation_metadata.jsonl"
    metadata_jsonl.write_text(
        json.dumps(
            {
                "tag": "single_object",
                "include": [{"class": "bench", "count": 1}],
                "prompt": "a photo of a bench",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeDiffusionPipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            assert model_id == "example/artifact-model"
            return TinyPipeline()

    def fail_create_comparisons(*args, **kwargs):
        raise AssertionError("GenEval metadata packs must not create comparison sheets by default")

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(Flux2KleinPipeline=FakeDiffusionPipeline),
    )
    monkeypatch.setattr(cli_main, "create_artifact_image_comparisons", fail_create_comparisons)

    assert (
        main(
            [
                "generate-pack",
                "--suite",
                "flux2-native",
                "--artifact",
                str(tmp_path),
                "--prompt-metadata-jsonl",
                str(metadata_jsonl),
                "--seeds",
                "3",
                "--skip-artifact-checksums",
                "--device",
                "cpu",
                "--dtype",
                "float32",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["run_count"] == 1
    assert output["artifact_comparisons"] == []
    assert output["outputs"][0]["comparisons"] == []


def test_cli_generate_pack_resume_existing_skips_completed_outputs(
    monkeypatch,
    capsys,
    tmp_path,
):
    model = torch.nn.Module()
    model.transformer_blocks = torch.nn.ModuleList(
        [torch.nn.ModuleDict({"attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(8, 8)})})]
    )
    config = OrbitQuantConfig(block_size=4, target_policy="generic_dit")
    summary = quantize_linear_modules(model, config)
    save_orbitquant_artifact(
        model,
        tmp_path,
        config=config,
        source_model_id="example/artifact-model",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )
    output_path = tmp_path / "assets" / "flux2-native_seed3_W4A4_simple-object.png"
    Image.new("RGB", (16, 16), "green").save(output_path)
    output_path.with_suffix(".png.json").write_text(
        json.dumps(
            {
                "suite": "flux2-native",
                "model_id": "example/artifact-model",
                "prompt": select_prompt_record(
                    default_prompt_payload("flux2"), prompt_id="simple-object"
                )["prompt"],
                "seed": 3,
                "height": 1024,
                "width": 1024,
                "frames": None,
                "steps": 4,
                "guidance": 1.0,
                "quantization": {
                    "config": {
                        "weight_bits": 4,
                        "activation_bits": 4,
                    }
                },
            }
        )
        + "\n"
    )
    monkeypatch.setattr(
        cli_main,
        "load_pipeline_for_suite",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("completed resume jobs must not load a pipeline")
        ),
    )

    assert (
        main(
            [
                "generate-pack",
                "--suite",
                "flux2-native",
                "--artifact",
                str(tmp_path),
                "--prompt-id",
                "simple-object",
                "--seeds",
                "3",
                "--resume-existing",
                "--device",
                "cpu",
                "--dtype",
                "float32",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["job_count"] == 1
    assert output["run_count"] == 0
    assert output["skipped_count"] == 1
    assert output["skipped_outputs"] == [str(output_path)]


def test_cli_generate_pack_resume_existing_reruns_invalid_metadata(
    monkeypatch,
    capsys,
    tmp_path,
):
    class TinyPipeline:
        def __init__(self):
            self.transformer = torch.nn.Module()
            self.transformer.transformer_blocks = torch.nn.ModuleList(
                [
                    torch.nn.ModuleDict(
                        {"attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(8, 8)})}
                    )
                ]
            )
            self.calls = []

        def to(self, device):
            return self

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(images=[Image.new("RGB", (16, 16), "purple")])

    source = TinyPipeline()
    config = OrbitQuantConfig(block_size=4, target_policy="generic_dit")
    summary = quantize_linear_modules(source.transformer, config)
    save_orbitquant_artifact(
        source.transformer,
        tmp_path,
        config=config,
        source_model_id="example/artifact-model",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )
    output_path = tmp_path / "assets" / "flux2-native_seed3_W4A4_simple-object.png"
    Image.new("RGB", (16, 16), "green").save(output_path)
    output_path.with_suffix(".png.json").write_text('{"status":"complete"}\n')
    restored = TinyPipeline()

    class FakeDiffusionPipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            assert model_id == "example/artifact-model"
            return restored

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(Flux2KleinPipeline=FakeDiffusionPipeline),
    )

    assert (
        main(
            [
                "generate-pack",
                "--suite",
                "flux2-native",
                "--artifact",
                str(tmp_path),
                "--prompt-id",
                "simple-object",
                "--seeds",
                "3",
                "--resume-existing",
                "--device",
                "cpu",
                "--dtype",
                "float32",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["job_count"] == 1
    assert output["run_count"] == 1
    assert output["skipped_count"] == 0
    assert len(restored.calls) == 1
