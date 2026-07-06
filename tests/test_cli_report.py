import json

import torch

from orbitquant.artifacts import record_artifact_metrics, save_orbitquant_artifact
from orbitquant.cli.main import main
from orbitquant.config import OrbitQuantConfig
from orbitquant.modeling import quantize_linear_modules


class TinyCliReportModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList(
            [torch.nn.ModuleDict({"attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(8, 8)})})]
        )


def test_cli_report_writes_native_eval_report(capsys, tmp_path):
    artifact_dir = tmp_path / "flux1-w4a4"
    model = TinyCliReportModel()
    config = OrbitQuantConfig(block_size=4, target_policy="flux")
    summary = quantize_linear_modules(model, config)
    save_orbitquant_artifact(
        model,
        artifact_dir,
        config=config,
        source_model_id="black-forest-labs/FLUX.1-schnell",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )
    record_artifact_metrics(
        artifact_dir,
        split="orbitquant",
        metrics={"geneval_overall": 0.7, "wall_time_seconds": 9.0},
        metadata={"suite": "flux1-schnell-native", "seed": 0, "bit_setting": "W4A4"},
    )

    assert (
        main(
            [
                "report",
                "--artifact",
                str(artifact_dir),
                "--output",
                str(tmp_path / "reports"),
                "--date",
                "20260706",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["report_path"].endswith("orbitquant-native-eval-20260706.md")
    assert output["artifact_count"] == 1
