from __future__ import annotations

import json
from types import SimpleNamespace

from voicerender import cli
from voicerender.engines.base import EngineCapabilities, EngineOption
from voicerender.engines.options import engine_options
from voicerender.engines.selection import reject_incompatible_options


def test_engine_options_lists_all_accepted_names_and_optional_metadata(
    monkeypatch,
) -> None:
    adapter = SimpleNamespace(
        capabilities=lambda: EngineCapabilities(
            option_names=frozenset({"speed", "seed"}),
            options=frozenset({"legacy_option"}),
        ),
        option_descriptors={
            "speed": {
                "value_type": "number",
                "minimum": 0.5,
                "maximum": 2.0,
                "description": "Playback speed",
            }
        },
    )
    monkeypatch.setattr(
        "voicerender.engines.options.get_adapter", lambda engine: adapter
    )

    options = engine_options("fake", target_id="demo")

    assert [option.name for option in options] == ["legacy_option", "seed", "speed"]
    assert next(item for item in options if item.name == "speed").value_type == "number"
    assert all(isinstance(option, EngineOption) for option in options)
    assert reject_incompatible_options(
        "fake", {"unknown": 1}, frozenset(item.name for item in options)
    ) == ["unknown"]


def test_engine_options_command_emits_deterministic_json(monkeypatch, capsys) -> None:
    option = EngineOption(name="speed", value_type="number", minimum=0.5, maximum=2.0)
    monkeypatch.setattr(
        cli, "engine_options", lambda engine, *, target_id=None: (option,)
    )

    exit_code = cli.main(["engines", "options", "kokoro", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload == [
        {
            "choices": [],
            "default": None,
            "description": "",
            "identity_affecting": True,
            "maximum": 2.0,
            "minimum": 0.5,
            "name": "speed",
            "required": False,
            "runtime_only": False,
            "value_type": "number",
        }
    ]


def test_engine_options_human_output_avoids_python_repr(monkeypatch, capsys) -> None:
    options = (
        EngineOption(
            name="speed",
            value_type="number",
            default=0.8,
            choices=(0.5, 1.0),
        ),
    )
    monkeypatch.setattr(
        cli, "engine_options", lambda engine, *, target_id=None: options
    )

    exit_code = cli.main(["engines", "options", "kokoro"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "speed\tnumber\tfalse\t0.8\t0.5, 1.0\t" in output
    assert "EngineOption(" not in output
