import torch

from orbitquant.packing import pack_lowbit, unpack_lowbit


def test_lowbit_pack_unpack_round_trips_supported_widths():
    for bits in (2, 3, 4, 6):
        values = torch.arange(0, 37, dtype=torch.uint8) % (2**bits)

        packed = pack_lowbit(values, bits=bits)
        unpacked = unpack_lowbit(packed, bits=bits, length=values.numel())

        assert packed.dtype == torch.uint8
        assert torch.equal(unpacked, values)


def test_lowbit_pack_unpack_round_trips_large_vectorized_path():
    generator = torch.Generator(device="cpu").manual_seed(0)
    values = torch.randint(0, 8, (100_003,), generator=generator, dtype=torch.uint8)

    packed = pack_lowbit(values, bits=3)
    unpacked = unpack_lowbit(packed, bits=3, length=values.numel())

    assert torch.equal(unpacked, values)


def test_lowbit_unpack_rejects_values_that_do_not_fit_bits():
    values = torch.tensor([0, 1, 2, 4], dtype=torch.uint8)

    try:
        pack_lowbit(values, bits=2)
    except ValueError as exc:
        assert "fit in 2 bits" in str(exc)
    else:
        raise AssertionError("pack_lowbit accepted an out-of-range value")
