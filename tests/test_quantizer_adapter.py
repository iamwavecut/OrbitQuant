from orbitquant.config import OrbitQuantConfig
from orbitquant.quantizer import OrbitQuantizer, register_hf_quantizers


def test_quantizer_adapter_reports_no_calibration_requirement():
    quantizer = OrbitQuantizer(OrbitQuantConfig())

    assert quantizer.requires_parameters_quantization is True
    assert quantizer.requires_calibration is False
    assert quantizer.is_serializable() is True


def test_hf_registration_is_best_effort_without_optional_dependencies():
    result = register_hf_quantizers()

    assert set(result) == {"diffusers", "transformers"}
