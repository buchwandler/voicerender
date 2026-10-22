from __future__ import annotations

import json
from pathlib import Path

import pytest

from voicerender import cli
from voicerender.errors import RuntimeControlError
from voicerender.runtime import OnnxRuntimeStatus, RuntimeDistribution


def _status() -> OnnxRuntimeStatus:
    return OnnxRuntimeStatus(
        importable=True,
        module_version="1.20.0",
        module_path=Path("/opt/ort/__init__.py"),
        build_device="CUDA",
        build_info="test build",
        distributions=(
            RuntimeDistribution(
                name="onnxruntime-gpu",
                version="1.20.0",
                location=Path("/opt/ort"),
                owns_import=True,
            ),
        ),
        imported_distribution="onnxruntime-gpu",
        distribution_state="resolved",
        available_providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
        configured_providers=("cuda", "cpu"),
        provider_source="user_config",
        resolved_providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
        import_error=None,
        resolution_error=None,
    )


def test_runtime_status_json_serializes_paths_and_provider_tuples(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "onnx_runtime_status", _status)

    assert cli.main(["runtime", "status", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["module_path"] == "/opt/ort/__init__.py"
    assert payload["distributions"][0]["location"] == "/opt/ort"
    assert payload["available_providers"] == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert payload["configured_providers"] == ["cuda", "cpu"]
    assert payload["resolved_providers"] == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]


def test_runtime_status_human_output_uses_non_active_labels(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "onnx_runtime_status", _status)

    assert cli.main(["runtime", "status"]) == 0

    output = capsys.readouterr().out
    assert "Imported distribution: onnxruntime-gpu" in output
    assert "Distribution state: resolved" in output
    assert "Available providers: CUDAExecutionProvider, CPUExecutionProvider" in output
    assert "Configured providers: cuda, cpu" in output
    assert "Resolved providers: CUDAExecutionProvider, CPUExecutionProvider" in output
    assert "Active providers" not in output


def test_runtime_list_json_and_human_output(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "onnx_runtime_status", _status)

    assert cli.main(["runtime", "list", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "distribution_state": "resolved",
        "distributions": [
            {
                "name": "onnxruntime-gpu",
                "version": "1.20.0",
                "location": "/opt/ort",
                "owns_import": True,
            }
        ],
        "available_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    }

    assert cli.main(["runtime", "list"]) == 0
    output = capsys.readouterr().out
    assert "DISTRIBUTION\tVERSION\tLOCATION\tOWNS_IMPORT" in output
    assert "onnxruntime-gpu\t1.20.0\t/opt/ort\tTrue" in output
    assert "CUDAExecutionProvider" in output


def test_runtime_use_delegates_policy_and_emits_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(cli, "set_onnx_runtime_providers", calls.append)

    assert cli.main(["runtime", "use", "cuda,cpu", "--json"]) == 0

    assert calls == ["cuda,cpu"]
    assert json.loads(capsys.readouterr().out) == {
        "status": "ok",
        "action": "use",
        "providers": "cuda,cpu",
        "applies_to": "future_sessions",
    }


def test_runtime_reset_clears_only_saved_preference_and_emits_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        cli, "clear_onnx_runtime_providers", lambda: calls.append("clear")
    )

    assert cli.main(["runtime", "reset", "--json"]) == 0

    assert calls == ["clear"]
    assert json.loads(capsys.readouterr().out) == {
        "status": "ok",
        "action": "reset",
        "environment_overrides_remain": True,
        "existing_sessions_changed": False,
    }


def test_runtime_cli_errors_have_json_details(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def reject(_providers: str) -> None:
        raise RuntimeControlError(
            "Provider unavailable.",
            code="runtime.control_failed",
            details={"provider_validation_message": "CUDA is unavailable"},
        )

    monkeypatch.setattr(cli, "set_onnx_runtime_providers", reject)

    assert cli.main(["runtime", "use", "cuda", "--json"]) == 1

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "status": "error",
        "error": "Provider unavailable.",
        "error_code": "runtime.control_failed",
        "details": {"provider_validation_message": "CUDA is unavailable"},
    }
