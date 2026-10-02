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
