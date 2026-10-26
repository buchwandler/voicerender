from __future__ import annotations

import json
from types import SimpleNamespace

from voicerender import cli
from voicerender.engines.base import EngineCapabilities, EngineOption
from voicerender.engines.options import engine_options
from voicerender.engines.registry import get_adapter
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


def test_kokoro_options_expose_shorthand_and_structured_capabilities() -> None:
    options = {option.name: option for option in engine_options("kokoro")}

    assert options["short_sentence"].value_type == "string|table"
    assert "randomized-phrase" in options["short_sentence"].choices
    assert options["short_sentence.resolve_mode"].default == "randomized-phrase"
    assert options["short_sentence.min_phoneme_length"].minimum == 0
    assert options["voice_level"].value_type == "string|table"
    assert options["voice_level.gain_db"].value_type == "number"

    adapter = get_adapter("kokoro")
    assert "random_seed" in adapter.capabilities().option_names
    assert options["random_seed"].value_type == "integer"
    assert options["random_seed"].default is None
    assert options["lexicons"].value_type == "string|array[string]"
    assert options["lexicon_data_policy"].choices == ("auto", "installed-only")
    assert options["g2lex_files"].value_type == "array[table]"
    assert "g2lex_files" in adapter.capabilities().option_names


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
