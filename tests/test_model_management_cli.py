from __future__ import annotations

import json

from voicerender import cli
from voicerender.engines.assets import ModelArtifact


def test_models_list_json_uses_stable_model_fields(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "list_models",
        lambda *args, **kwargs: (
            ModelArtifact("kokoro", "v1.0", "Kokoro v1", True, "ready", ("en-US",)),
        ),
    )

    exit_code = cli.main(["models", "list", "--engine", "kokoro", "--json"])

    assert exit_code == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows == [
        {
            "display_name": "Kokoro v1",
            "engine": "kokoro",
            "installed": True,
            "installed_qualities": [],
            "languages": ["en-US"],
            "metadata": {},
            "path": None,
            "qualities": [],
            "revision": None,
            "source": None,
            "status": "ready",
            "target_id": "v1.0",
        }
    ]


def test_models_remove_requires_yes_before_calling_api(monkeypatch, capsys) -> None:
    called = False

    def remove(_request):
        nonlocal called
        called = True
        raise AssertionError("removal API must not be called without --yes")

    monkeypatch.setattr(cli, "remove_model", remove)

    exit_code = cli.main(["models", "remove", "v1.0", "--engine", "kokoro"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "requires --yes" in captured.err
    assert called is False


def test_models_install_json_reports_unsupported_diagnostic(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        cli,
        "install_model",
        lambda request: type(
            "Result",
            (),
            {
                "engine": request.engine,
                "target_id": request.target_id,
                "status": "unsupported",
                "changed": False,
                "diagnostics": (
                    "Engine 'kokoro' does not support model install for target 'v1.0'",
                ),
                "artifact": None,
            },
        )(),
    )

    exit_code = cli.main(["models", "install", "v1.0", "--engine", "kokoro", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["status"] == "unsupported"
    assert payload["diagnostics"] == [
        "Engine 'kokoro' does not support model install for target 'v1.0'"
    ]
