from __future__ import annotations

import json
from dataclasses import replace
from types import MappingProxyType

from voicerender import cli
from voicerender.engines.catalog import SynthesisTarget, TargetVoice
from voicerender.errors import ProfileError
from voicerender.voices_api import (
    VoiceDescriptor,
    inspect_voice,
    list_voices,
    parse_voice_selector,
    voice_descriptor_to_dict,
)


def _target() -> SynthesisTarget:
    return SynthesisTarget(
        engine="kokoro",
        id="v1.0",
        display_name="Kokoro",
        languages=("en-US",),
        runtime_available=True,
        voices=("af_heart",),
        speakers=("2",),
        voice_details=(
            TargetVoice(
                id="af_heart",
                gender="female",
                language="en-US",
                locale="en-US",
                languages=("en-US",),
            ),
        ),
        capabilities=frozenset({"reference_voice"}),
        metadata={"source": "test"},
    )


def _multilingual_target() -> SynthesisTarget:
    english_voices = (
        "AV_ariala",
        "AV_brody",
        "AV_david",
        "AV_elliott",
        "AV_ralph",
        "AV_cori",
        "AV_phil",
        "AV_helen",
    )
    german_voices = (
        "AV_thorsten",
        "AV_bernd",
        "AV_eva_k",
        "AV_karlsson",
        "AV_kerstin",
        "AV_mucky",
        "AV_hokuspokus",
        "AV_julia",
    )
    languages = ("en", "en-us", "en-gb", "de", "de-de", "de-at", "de-ch")
    english_languages = ("en", "en-us", "en-gb")
    german_languages = ("de", "de-de", "de-at", "de-ch")
    voice_details = tuple(
        TargetVoice(
            id=voice,
            language=language,
            locale=language,
            language_label="English" if language == "en" else "German",
            languages=english_languages if language == "en" else german_languages,
        )
        for language, voices in (("en", english_voices), ("de", german_voices))
        for voice in voices
    )
    return SynthesisTarget(
        engine="kokoro",
        id="akinvox-multispeaker-v1",
        display_name="AKinVox English/German Multispeaker",
        languages=languages,
        runtime_available=True,
        voices=english_voices + german_voices,
        voice_details=voice_details,
        default_voice="AV_ariala",
        metadata={"source": "test"},
    )


def test_multilingual_voice_filtering_is_language_scoped_and_normalized(
    monkeypatch,
) -> None:
    target = _multilingual_target()
    monkeypatch.setattr(
        "voicerender.voices_api.discover_targets", lambda *a, **k: (target,)
    )

    english = list_voices(engine="kokoro", target_id=target.id, language="en-US")
    german = list_voices(engine="kokoro", target_id=target.id, language="de-DE")
    german_underscore = list_voices(
        engine="kokoro", target_id=target.id, language="de_AT"
    )

    assert {item.voice for item in english} == set(target.voices[:8])
    assert {item.voice for item in german} == set(target.voices[8:])
    assert {item.voice for item in german_underscore} == set(target.voices[8:])
    assert all(item.languages == ("en", "en-us", "en-gb") for item in english)
    assert all(item.languages == ("de", "de-de", "de-at", "de-ch") for item in german)


def test_voice_selectors_list_inspect_and_round_trip(monkeypatch) -> None:
    target = _target()
    monkeypatch.setattr(
        "voicerender.voices_api.discover_targets", lambda *a, **k: (target,)
    )

    descriptors = list_voices(engine="kokoro", language="en-US")
    selectors = {descriptor.selector for descriptor in descriptors}

    assert selectors == {
        "kokoro:v1.0/af_heart",
        "kokoro:v1.0#2",
        "kokoro:v1.0/af_heart#2",
        "kokoro:v1.0",
    }
    for descriptor in descriptors:
        parsed = parse_voice_selector(descriptor.selector)
        assert parsed.engine == descriptor.engine
        assert parsed.target_id == descriptor.target_id
        assert parsed.voice == descriptor.voice
        assert parsed.speaker == descriptor.speaker
    inspected = inspect_voice("kokoro:v1.0/af_heart", language="en-US")
    assert inspected.kind == "builtin"
    assert inspected.gender == "female"
    assert voice_descriptor_to_dict(inspected)["languages"] == ["en-US"]


def test_voice_descriptor_to_dict_serializes_nested_read_only_metadata() -> None:
    descriptor = replace(
        _target_descriptor(),
        metadata=MappingProxyType(
            {"catalog": MappingProxyType({"providers": ("CPUExecutionProvider",)})}
        ),
    )

    payload = voice_descriptor_to_dict(descriptor)

    assert payload["metadata"] == {"catalog": {"providers": ["CPUExecutionProvider"]}}
    json.dumps(payload)


def test_selector_rejects_noncanonical_delimiters() -> None:
    try:
        parse_voice_selector("kokoro:v1.0/a/b")
    except ProfileError as exc:
        assert "voice delimiter" in str(exc)
    else:
        raise AssertionError("multiple voice delimiters must be rejected")


def test_list_voices_can_filter_target_and_kind(monkeypatch) -> None:
    monkeypatch.setattr(
        "voicerender.voices_api.discover_targets", lambda *a, **k: (_target(),)
    )

    descriptors = list_voices(engine="kokoro", target_id="v1.0")

    assert descriptors
    assert all(item.target_id == "v1.0" for item in descriptors)
    assert {item.kind for item in descriptors} == {
        "builtin",
        "speaker",
        "reference-capable",
    }


def test_voice_cli_list_filter_and_inspect_json(monkeypatch, capsys) -> None:
    descriptor = _target_descriptor()
    monkeypatch.setattr(cli, "list_voices", lambda **kwargs: (descriptor,))
    monkeypatch.setattr(
        cli, "inspect_voice", lambda selector, *, language=None: descriptor
    )

    list_exit = cli.main(
        ["voices", "list", "--engine", "kokoro", "--kind", "builtin", "--json"]
    )
    listed = json.loads(capsys.readouterr().out)
    inspect_exit = cli.main(["voices", "inspect", "kokoro:v1.0/af_heart", "--json"])
    inspected = json.loads(capsys.readouterr().out)

    assert list_exit == inspect_exit == 0
    assert listed == [voice_descriptor_to_dict(descriptor)]
    assert inspected == voice_descriptor_to_dict(descriptor)


def _target_descriptor() -> VoiceDescriptor:
    from voicerender.voices_api import VoiceDescriptor

    return VoiceDescriptor(
        selector="kokoro:v1.0/af_heart",
        engine="kokoro",
        target_id="v1.0",
        voice="af_heart",
        speaker=None,
        kind="builtin",
        languages=("en-US",),
    )
