from orbitquant.eval.native_settings import NativeSuite, get_native_suite, list_native_suites
from orbitquant.eval.prompts import (
    IMAGE_PROMPTS,
    VIDEO_PROMPTS,
    build_prompt_seed_jobs,
    default_prompt_payload,
    select_prompt_record,
)

__all__ = [
    "IMAGE_PROMPTS",
    "NativeEvalReportResult",
    "VIDEO_PROMPTS",
    "build_external_eval_plan",
    "build_external_eval_script",
    "build_prompt_seed_jobs",
    "NativeSuite",
    "create_image_comparison_sheet",
    "create_video_contact_sheet",
    "default_prompt_payload",
    "get_native_suite",
    "generate_native_eval_report",
    "list_native_suites",
    "select_prompt_record",
]


def __getattr__(name: str):
    if name in {"create_image_comparison_sheet", "create_video_contact_sheet"}:
        from orbitquant.eval.assets import (
            create_image_comparison_sheet,
            create_video_contact_sheet,
        )

        values = {
            "create_image_comparison_sheet": create_image_comparison_sheet,
            "create_video_contact_sheet": create_video_contact_sheet,
        }
        return values[name]
    if name in {"NativeEvalReportResult", "generate_native_eval_report"}:
        from orbitquant.eval.report import NativeEvalReportResult, generate_native_eval_report

        values = {
            "NativeEvalReportResult": NativeEvalReportResult,
            "generate_native_eval_report": generate_native_eval_report,
        }
        return values[name]
    if name in {"build_external_eval_plan", "build_external_eval_script"}:
        from orbitquant.eval.external_plan import (
            build_external_eval_plan,
            build_external_eval_script,
        )

        values = {
            "build_external_eval_plan": build_external_eval_plan,
            "build_external_eval_script": build_external_eval_script,
        }
        return values[name]
    raise AttributeError(f"module 'orbitquant.eval' has no attribute {name!r}")
