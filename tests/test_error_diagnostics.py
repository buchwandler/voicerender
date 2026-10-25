from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from voicerender.error_diagnostics import (
    exception_diagnostic,
    root_cause,
    root_cause_summary,
)
from voicerender.errors import EngineBackendError


class CustomValue:
    pass


def test_exception_diagnostic_preserves_explicit_three_level_chain() -> None:
    root = RuntimeError("provider execution failed")
    middle = ValueError("engine failed")
    outer = EngineBackendError("synthesis failed", engine="fake")
    middle.__cause__ = root
    outer.__cause__ = middle

    diagnostic = exception_diagnostic(outer)

    assert diagnostic["type"] == "EngineBackendError"
    assert diagnostic["details"] == {"engine": "fake"}
    assert diagnostic["cause"]["type"] == "ValueError"
    assert diagnostic["cause"]["cause"]["type"] == "RuntimeError"
    assert root_cause(diagnostic)["message"] == "provider execution failed"
    assert root_cause_summary(diagnostic) == "RuntimeError: provider execution failed"
    json.dumps(diagnostic)


def test_exception_diagnostic_falls_back_to_unsuppressed_context() -> None:
    try:
        raise ValueError("inner")
    except ValueError:
        try:
            raise RuntimeError("outer")
        except RuntimeError as error:
            diagnostic = exception_diagnostic(error)

    assert diagnostic["cause"]["type"] == "ValueError"
    assert diagnostic["cause"]["message"] == "inner"


def test_exception_diagnostic_respects_suppressed_context() -> None:
    try:
        raise ValueError("inner")
    except ValueError:
        try:
            raise RuntimeError("outer") from None
        except RuntimeError as error:
            diagnostic = exception_diagnostic(error)

    assert "cause" not in diagnostic


def test_exception_diagnostic_bounds_depth_and_detects_cycles() -> None:
    first = RuntimeError("first")
    second = ValueError("second")
    first.__cause__ = second
    second.__cause__ = first

    diagnostic = exception_diagnostic(first, max_depth=1)
    assert diagnostic["cause_truncated"] is True

    cyclic = exception_diagnostic(first, max_depth=4)
    assert cyclic["cause"]["cause"]["message"] == "[exception cycle omitted]"


def test_details_are_sanitized_and_json_safe() -> None:
    error = EngineBackendError(
        "x" * 5000,
        details={
            "path": Path("/tmp/model.onnx"),
            "scalar": np.int64(7),
            "unsupported": CustomValue(),
            "speech_text": "secret speech",
            "request_text": "secret request",
            "input_audio": [0.1, 0.2],
            "phoneme_ids": [1, 2],
            "token_sequence": [3, 4],
            "environment_variables": {"API_KEY": "secret"},
            "nested": {"transcript": "secret transcript", "safe": "ok"},
        },
    )

    diagnostic = exception_diagnostic(error)
    details = diagnostic["details"]

    assert len(diagnostic["message"]) <= 4096
    assert details["path"] == "/tmp/model.onnx"
    assert details["scalar"] == 7
    assert details["unsupported"].startswith("[omitted unsupported ")
    assert "speech_text" not in details
    assert "request_text" not in details
    assert "input_audio" not in details
    assert "phoneme_ids" not in details
    assert "token_sequence" not in details
    assert "environment_variables" not in details
    assert details["nested"] == {"safe": "ok"}
    json.dumps(diagnostic)


def test_engine_error_keeps_explicit_python_exception_cause() -> None:
    native = RuntimeError("native failure")
    try:
        raise EngineBackendError("wrapped failure") from native
    except EngineBackendError as error:
        assert error.__cause__ is native
        assert exception_diagnostic(error)["cause"]["message"] == "native failure"


def test_root_cause_summary_omits_duplicate_message() -> None:
    diagnostic = {
        "message": "same",
        "cause": {"type": "RuntimeError", "message": "same"},
    }
    assert root_cause_summary(diagnostic) is None


def test_exception_diagnostic_redacts_signed_url_queries_and_inline_credentials() -> (
    None
):
    error = RuntimeError(
        "download failed at "
        "https://example.test/model?X-Amz-Signature=signed-secret&token=query-secret;\n"
        "HF_TOKEN=env-secret"
    )

    diagnostic = exception_diagnostic(error)
    payload = json.dumps(diagnostic)

    assert "https://example.test/model?[REDACTED]" in payload
    assert "signed-secret" not in payload
    assert "query-secret" not in payload
    assert "env-secret" not in payload
