import subprocess
import sys

from orbitquant.fused.groups import PackedGroup, _segments


def test_segments_merge_runs_of_one_weight_width():
    assert _segments([(256, 2), (128, 2), (128, 3)]) == ((384, 2), (128, 3))
    assert _segments([(128, 4), (128, 4)]) == ((256, 4),)


def test_packed_rows_are_byte_rows_for_one_width_and_a_flat_run_for_mixed_widths():
    assert PackedGroup.packed_shape(((384, 4),), 1024) == (384, 512)
    assert PackedGroup.packed_shape(((384, 2), (128, 3)), 1024) == (384 * 256 + 128 * 384,)


def test_recognizing_fused_checkpoints_does_not_need_the_triton_runtime():
    # The HF quantizer imports the group types for every checkpoint tensor it checks.
    code = (
        "import sys, orbitquant.fused, orbitquant.fused.groups;"
        "print(any(name.startswith('orbitquant.kernels.triton') for name in sys.modules))"
    )
    command = [sys.executable, "-c", code]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False"


def test_host_constants_move_to_the_weight_device_once():
    import torch

    group = torch.nn.Module()
    group.register_buffer("codes_w4", torch.arange(16, dtype=torch.int8), persistent=False)
    first = PackedGroup._on_device(group, "codes_w4", torch.device("meta"))
    assert first.device.type == "meta" and group.codes_w4 is first
    assert PackedGroup._on_device(group, "codes_w4", torch.device("meta")) is first
    assert "codes_w4" in dict(group.named_buffers())
