from __future__ import annotations

import json
from types import MappingProxyType

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


def _sopro_model() -> ModelArtifact:
    return ModelArtifact(
        "sopro",
        "model-v1",
        "Sopro model v1",
        True,
        "ready",
        ("en",),
        metadata=MappingProxyType(
            {
                "catalog": MappingProxyType(
                    {
                        "profile": MappingProxyType(
                            {
                                "precision": "uint8",
                                "providers": ("CPUExecutionProvider",),
                            }
                        )
                    }
                )
            }
        ),
    )


def _unexpected_model_serialization(_model: ModelArtifact) -> None:
    raise AssertionError("plain model output must not serialize metadata")


def test_models_list_plain_and_offline_supports_sopro_read_only_metadata(
    monkeypatch, capsys
) -> None:
    offline_values = []

    def list_models(*args, **kwargs):
        offline_values.append(kwargs["offline"])
        return (_sopro_model(),)

    monkeypatch.setattr(cli, "list_models", list_models)
    monkeypatch.setattr(cli, "model_artifact_to_dict", _unexpected_model_serialization)

    for extra_args in ([], ["--offline"]):
        exit_code = cli.main(["models", "list", "--engine", "sopro", *extra_args])
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "sopro\tmodel-v1\tready\ttrue\ten\t-" in captured.out
        assert captured.err == ""

    assert offline_values == [False, True]


def test_models_list_json_supports_sopro_read_only_metadata(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(cli, "list_models", lambda *args, **kwargs: (_sopro_model(),))

    exit_code = cli.main(["models", "list", "--engine", "sopro", "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["engine"] == "sopro"
    assert payload[0]["target_id"] == "model-v1"
    assert payload[0]["metadata"]["catalog"]["profile"] == {
        "precision": "uint8",
        "providers": ["CPUExecutionProvider"],
    }


def test_models_inspect_plain_supports_sopro_read_only_metadata(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(cli, "model_status", lambda *args, **kwargs: _sopro_model())
    monkeypatch.setattr(cli, "model_artifact_to_dict", _unexpected_model_serialization)

    exit_code = cli.main(["models", "inspect", "model-v1", "--engine", "sopro"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert (
        "sopro\tmodel-v1\tready\tinstalled=true\tlanguages=en\tqualities=-"
        in captured.out
    )
    assert captured.err == ""


def test_models_inspect_json_supports_sopro_read_only_metadata(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(cli, "model_status", lambda *args, **kwargs: _sopro_model())

    exit_code = cli.main(
        ["models", "inspect", "model-v1", "--engine", "sopro", "--json"]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["engine"] == "sopro"
    assert payload["target_id"] == "model-v1"
    assert payload["metadata"]["catalog"]["profile"] == {
        "precision": "uint8",
        "providers": ["CPUExecutionProvider"],
    }
