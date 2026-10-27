from __future__ import annotations

import hashlib
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from voicerender.engines.base import (
    EngineSelection,
    PronunciationSpan,
    SpeechRequest,
    SpeechToken,
)
from voicerender.engines.kittensynth import (
    KittenSynthEngineAdapter,
    KittenSynthEngineSession,
)
from voicerender.errors import EngineBackendError
from voicerender.lexicons import LexiconCatalogEntry


def _named_source(
    *,
    installed: bool = True,
    model_support: bool = True,
    encoding: str = "ipa",
) -> SimpleNamespace:
    return SimpleNamespace(
        selector="public-en",
        language="en",
        asset_id="asset-en-v1",
        backend="lexphon",
        default=True,
        installed=installed,
        model_support=model_support,
        display_name="Public English",
        phoneme_encoding=encoding,
        data_version="2025-01",
        content_sha256="a" * 64,
        logical_sha256="b" * 64,
        compatibility_reason=None if model_support else "unsupported encoding",
    )


def _selection(**options: Any) -> EngineSelection:
    return EngineSelection(
        engine="kitten",
        target_id="nano-0.8-int8",
        language="en-US",
        voice="Jasper",
        options=options,
    )


def test_kitten_advertises_linguistic_support_without_hard_token_requirements() -> None:
    adapter = KittenSynthEngineAdapter()

    capabilities = adapter.capabilities()
    descriptors = adapter.option_descriptors()

    assert capabilities.supports_lexicons
    assert capabilities.supports_linguistic_tokens is True
    assert capabilities.supports_pronunciation_overrides
    assert capabilities.pronunciation_alphabets == frozenset({"ipa"})
    assert {
        "lexicons",
        "g2lex_files",
        "lexicon_data_policy",
        "heteronym_policy",
    } <= set(descriptors)
    assert descriptors["lexicon_data_policy"].choices == ("auto", "installed-only")
    assert descriptors["heteronym_policy"].choices == ("auto", "off")
    fragment = adapter.renderprofile_fragment(_selection())
    assert fragment.token_fields == ()


def test_request_tokens_and_ipa_overrides_are_forwarded_to_measure_and_synthesis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("kittensynth")
    module.LinguisticToken = lambda **kwargs: SimpleNamespace(**kwargs)  # type: ignore[attr-defined]
    module.PronunciationOverride = lambda **kwargs: SimpleNamespace(**kwargs)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kittensynth", module)

    class Voice:
        def __init__(self) -> None:
            self.measure_args: list[tuple[str, dict[str, Any]]] = []
            self.synthesis_args: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

        def measure_prepared(self, text: str, **kwargs: Any) -> Any:
            self.measure_args.append((text, kwargs))
            return SimpleNamespace(
                token_count=3,
                max_tokens=32,
                fits=True,
                model_id="nano-0.8-int8",
                metadata={},
            )

        def synthesize_prepared(self, *args: Any, **kwargs: Any) -> Any:
            self.synthesis_args.append((args, kwargs))
            return SimpleNamespace(
                audio=np.array([0.0, 0.25], dtype=np.float32),
                sample_rate=24000,
                voice="Jasper",
                model_ref="nano-0.8-int8",
                speed=1.0,
                metadata={},
            )

    voice = Voice()
    request = SpeechRequest(
        id="request-1",
        text="read",
        language="en-US",
        tokens=(
            SpeechToken(
                start=0,
                end=4,
                text="read",
                pos="VERB",
                tag="VB",
                lemma="read",
                morph="Tense=Pres",
                language="en",
            ),
        ),
        pronunciation_overrides=(
            PronunciationSpan(
                start=0,
                end=4,
                phonemes="ɹ iː d",
                language="en",
                alphabet="ipa",
            ),
        ),
    )
    session = KittenSynthEngineSession(voice, _selection())

    session.measure(request)
    session.synthesize(request)

    measured_kwargs = voice.measure_args[0][1]
    synthesized_kwargs = voice.synthesis_args[0][1]
    for forwarded in (measured_kwargs, synthesized_kwargs):
        assert forwarded["tokens"][0].start == 0
        assert forwarded["tokens"][0].end == 4
        assert forwarded["tokens"][0].text == "read"
        assert forwarded["tokens"][0].tag == "VB"
        assert forwarded["tokens"][0].lemma == "read"
        assert forwarded["tokens"][0].language == "en"
        assert forwarded["tokens"][0].pos == "VERB"
        assert forwarded["tokens"][0].morph == "Tense=Pres"
        assert forwarded["pronunciation_overrides"][0].phonemes == "ɹ iː d"
        assert forwarded["pronunciation_overrides"][0].alphabet == "ipa"


def test_measurement_error_retains_engine_stage_and_native_provenance() -> None:
    class Voice:
        def measure_prepared(self, text: str, **kwargs: Any) -> Any:
            raise ValueError("linguistic token span [7, 8) cannot be projected")

    request = SpeechRequest(
        id="segment-123",
        text="Hello, bow!",
        language="en-US",
        voice="Jasper",
    )
    session = KittenSynthEngineSession(Voice(), _selection())

    with pytest.raises(EngineBackendError) as captured:
        session.measure(request)

    error = captured.value
    assert error.code == "kitten.engine_error"
    assert error.engine == "kitten"
    assert error.target_id == "nano-0.8-int8"
    assert error.request_id == "segment-123"
    assert error.native_error_type == "ValueError"
    assert error.stage == "measure"
    assert error.source == "kittensynth.measure_prepared"
    assert error.details["adapter_code"] == "kitten.engine_error"
    assert "kitten.engine_error during measure" in str(error)
    assert "request/segment=segment-123" in str(error)
    assert "linguistic token span [7, 8) cannot be projected" in str(error)


def test_punctuation_rich_renderplan_tokens_are_forwarded_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("kittensynth")
    module.LinguisticToken = lambda **kwargs: SimpleNamespace(**kwargs)  # type: ignore[attr-defined]
    module.PronunciationOverride = lambda **kwargs: SimpleNamespace(**kwargs)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kittensynth", module)

    class Voice:
        def __init__(self) -> None:
            self.measure_args: list[dict[str, Any]] = []
            self.synthesis_args: list[dict[str, Any]] = []

        def measure_prepared(self, text: str, **kwargs: Any) -> Any:
            self.measure_args.append(kwargs)
            return SimpleNamespace(
                token_count=5,
                max_tokens=32,
                fits=True,
                model_id="nano-0.8-int8",
                metadata={},
            )

        def synthesize_prepared(self, text: str, **kwargs: Any) -> Any:
            self.synthesis_args.append(kwargs)
            return SimpleNamespace(
                audio=np.array([0.0, 0.25], dtype=np.float32),
                sample_rate=24000,
                voice="Jasper",
                model_ref="nano-0.8-int8",
                speed=1.0,
                metadata={},
            )

    tokens = (
        SpeechToken(0, 5, "Hello", pos="INTJ", tag="UH", lemma="hello", language="en"),
        SpeechToken(5, 6, ",", pos="PUNCT", tag="PUNCT", language="en"),
        SpeechToken(
            7, 10, "bow", pos="NOUN", tag="NN", lemma="bow", morph="Number=Sing"
        ),
        SpeechToken(10, 11, "!", pos="PUNCT", tag="PUNCT", language="en"),
    )
    request = SpeechRequest(
        id="punctuation-rich-request",
        text="Hello, bow!",
        language="en-US",
        tokens=tokens,
    )
    voice = Voice()
    session = KittenSynthEngineSession(voice, _selection())

    session.measure(request)
    session.synthesize(request)

    expected = [
        (0, 5, "Hello", "INTJ", "UH", "hello", None, "en"),
        (5, 6, ",", "PUNCT", "PUNCT", None, None, "en"),
        (7, 10, "bow", "NOUN", "NN", "bow", "Number=Sing", None),
        (10, 11, "!", "PUNCT", "PUNCT", None, None, "en"),
    ]
    for arguments in (voice.measure_args[0], voice.synthesis_args[0]):
        assert [
            (
                token.start,
                token.end,
                token.text,
                token.pos,
                token.tag,
                token.lemma,
                token.morph,
                token.language,
            )
            for token in arguments["tokens"]
        ] == expected


def test_kitten_forwards_partial_tokens_to_measurement_and_synthesis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("kittensynth")
    module.LinguisticToken = lambda **kwargs: SimpleNamespace(**kwargs)  # type: ignore[attr-defined]
    module.PronunciationOverride = lambda **kwargs: SimpleNamespace(**kwargs)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kittensynth", module)

    class Voice:
        def __init__(self) -> None:
            self.measure_calls: list[tuple[str, dict[str, Any]]] = []
            self.synthesis_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

        def measure_prepared(self, text: str, **kwargs: Any) -> Any:
            self.measure_calls.append((text, kwargs))
            return SimpleNamespace(
                token_count=3,
                max_tokens=32,
                fits=True,
                model_id="nano-0.8-int8",
                metadata={},
            )

        def synthesize_prepared(self, *args: Any, **kwargs: Any) -> Any:
            self.synthesis_calls.append((args, kwargs))
            return SimpleNamespace(
                audio=np.array([0.0, 0.25], dtype=np.float32),
                sample_rate=24000,
                voice="Jasper",
                model_ref="nano-0.8-int8",
                speed=1.0,
                metadata={},
            )

    voice = Voice()
    session = KittenSynthEngineSession(voice, _selection())
    partial = SpeechRequest(
        id="partial-token-request",
        text="read",
        language="en-US",
        tokens=(
            SpeechToken(
                start=0,
                end=4,
                text="read",
                lemma="read",
                pos=None,
                tag=None,
                morph=None,
                language="en-US",
            ),
        ),
    )
    session.measure(partial)
    session.synthesize(partial)

    measured_token = voice.measure_calls[0][1]["tokens"][0]
    synthesized_token = voice.synthesis_calls[0][1]["tokens"][0]
    for token in (measured_token, synthesized_token):
        assert token.start == 0
        assert token.end == 4
        assert token.text == "read"
        assert token.lemma == "read"
        assert token.pos is None
        assert token.tag is None
        assert token.morph is None

    no_tokens = replace(partial, id="no-token-request", tokens=())
    session.measure(no_tokens)
    session.synthesize(no_tokens)
    assert "tokens" not in voice.measure_calls[1][1]
    assert "tokens" not in voice.synthesis_calls[1][1]


def test_open_forwards_g2p_configuration_and_forces_installed_only_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("kittensynth")
    module.KittenG2PConfig = lambda **kwargs: SimpleNamespace(**kwargs)  # type: ignore[attr-defined]
    captured: dict[str, Any] = {}

    class Voice:
        @classmethod
        def from_pretrained(cls, target_id: str, **kwargs: Any) -> Voice:
            captured["target_id"] = target_id
            captured.update(kwargs)
            return cls()

        def close(self) -> None:
            captured["closed"] = True

    module.KittenVoice = Voice  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kittensynth", module)
    selection = EngineSelection(
        engine="kitten",
        target_id="nano-0.8-int8",
        language="en-US",
        voice="Jasper",
        offline=True,
        options={"lexicons": ("public-en",), "heteronym_policy": "off"},
    )

    with KittenSynthEngineAdapter().open(selection):
        config = captured["g2p_config"]
        assert config.language == "en-US"
        assert config.lexicons == ("public-en",)
        assert config.lexicon_data_policy == "installed-only"
        assert config.heteronym_policy == "off"
    assert captured["closed"] is True


def test_lexicon_discovery_and_g2lex_inspection_are_bridged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _named_source()
    discovery = SimpleNamespace(entries=(source,))
    calls: list[dict[str, Any]] = []
    module = ModuleType("kittensynth")

    def discover_lexicons(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return discovery

    module.discover_lexicons = discover_lexicons  # type: ignore[attr-defined]
    module.inspect_g2lex_file = lambda path: {  # type: ignore[attr-defined]
        "valid": True,
        "path": path,
        "sha256": "a" * 64,
        "logical_sha256": "b" * 64,
        "language": "en",
        "phoneme_encoding": "ipa",
        "compatible": True,
        "metadata": {"language": "en"},
    }
    monkeypatch.setitem(sys.modules, "kittensynth", module)
    adapter = KittenSynthEngineAdapter()

    entries, result = adapter.discover_lexicons(
        language="en-US", offline=True, refresh=True, preference="catalog"
    )
    inspection = adapter.inspect_g2lex_file("local.g2lex")

    assert isinstance(entries[0], LexiconCatalogEntry)
    assert entries[0].selector == "public-en"
    assert entries[0].content_sha256 == "a" * 64
    assert entries[0].logical_sha256 == "b" * 64
    assert entries[0].compatibility_reason is None
    assert result is discovery
    assert calls == [
        {
            "language": "en-US",
            "offline": True,
            "refresh": True,
            "preference": "catalog",
        }
    ]
    assert inspection["valid"] is True
    assert inspection["metadata"] == {"language": "en"}


def test_local_g2lex_validation_checks_language_encoding_and_content_pins(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    content = b"a G2Lex fixture"
    path = tmp_path / "local.g2lex"
    path.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    inspection = {
        "valid": True,
        "compatible": True,
        "language": "en",
        "phoneme_encoding": "ipa",
        "sha256": digest,
        "logical_sha256": "b" * 64,
    }
    adapter = KittenSynthEngineAdapter()
    monkeypatch.setattr(adapter, "inspect_g2lex_file", lambda _path: inspection)

    valid = adapter._validate_pronunciation_sources(
        _selection(g2lex_files=({"path": str(path), "sha256": digest},))
    )
    changed = adapter._validate_pronunciation_sources(
        _selection(g2lex_files=({"path": str(path), "sha256": "c" * 64},))
    )
    inspection["language"] = "fr"
    inspection["phoneme_encoding"] = "arpabet"
    incompatible = adapter._validate_pronunciation_sources(
        _selection(g2lex_files=(str(path),))
    )

    assert valid == ()
    assert {item.code for item in changed} == {"kitten.g2lex_pin_changed"}
    assert {item.code for item in incompatible} >= {
        "kitten.g2lex_language_incompatible",
        "kitten.g2lex_encoding_incompatible",
    }


def test_named_lexicon_validation_checks_compatibility_and_install_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = KittenSynthEngineAdapter()
    source = _named_source(installed=False)
    entries = (
        LexiconCatalogEntry(
            selector=source.selector,
            engine="kitten",
            language=source.language,
            locale="en",
            asset_id=source.asset_id,
            data_backend=source.backend,
            default=source.default,
            installed=source.installed,
            model_support="known",
            phoneme_encoding=source.phoneme_encoding,
            content_sha256=source.content_sha256,
            logical_sha256=source.logical_sha256,
        ),
    )
    monkeypatch.setattr(
        adapter,
        "discover_lexicons",
        lambda **kwargs: (entries, SimpleNamespace(entries=entries)),
    )

    automatic = adapter._validate_pronunciation_sources(
        _selection(lexicons=("public-en",))
    )
    installed_only = adapter._validate_pronunciation_sources(
        _selection(lexicons=("public-en",), lexicon_data_policy="installed-only")
    )
    offline = adapter._validate_pronunciation_sources(
        EngineSelection(
            engine="kitten",
            target_id="nano-0.8-int8",
            language="en-US",
            options={"lexicons": ("public-en",)},
            offline=True,
        )
    )
    wrong_entries = (replace(entries[0], language="fr"),)
    monkeypatch.setattr(
        adapter,
        "discover_lexicons",
        lambda **kwargs: (wrong_entries, SimpleNamespace(entries=wrong_entries)),
    )
    wrong_language = adapter._validate_pronunciation_sources(
        _selection(lexicons=("public-en",))
    )

    assert automatic == ()
    assert [item.code for item in installed_only] == ["kitten.lexicon_not_installed"]
    assert [item.code for item in offline] == ["kitten.lexicon_not_installed"]
    assert [item.code for item in wrong_language] == [
        "kitten.lexicon_language_incompatible"
    ]


def test_pronunciation_stack_reports_effective_order_and_sources(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = _named_source()
    named_entries = SimpleNamespace(entries=(source,))
    module = ModuleType("kittensynth")
    module.discover_lexicons = lambda **kwargs: named_entries  # type: ignore[attr-defined]
    module.inspect_g2lex_file = lambda path: {  # type: ignore[attr-defined]
        "valid": True,
        "compatible": True,
        "path": path,
        "sha256": "c" * 64,
        "logical_sha256": "d" * 64,
        "language": "en",
        "phoneme_encoding": "ipa",
        "metadata": {},
    }
    monkeypatch.setitem(sys.modules, "kittensynth", module)
    stack = KittenSynthEngineAdapter().pronunciation_stack(
        language="en-US",
        target_id="nano-0.8-int8",
        options={
            "lexicons": ("public-en",),
            "g2lex_files": (str(tmp_path / "local.g2lex"),),
            "heteronym_policy": "auto",
        },
    )

    assert stack["precedence"] == [
        "request-local IPA spans",
        "local G2Lex file 1",
        "public-en",
        "contextual heteronym rules",
        "eSpeak fallback",
    ]
    assert stack["local_g2lex"][0]["status"] == "valid"
    assert stack["named_lexicons"][0]["content_sha256"] == "a" * 64
    assert stack["g2p_fallback"] == "espeak"


def test_synthesis_identity_uses_source_content_not_local_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _named_source()
    discovery = SimpleNamespace(entries=(source,))
    module = ModuleType("kittensynth")
    module.runtime_identity = lambda: {  # type: ignore[attr-defined]
        "runtime": "kitten",
        "g2p_revision": "g2p-revision-1",
        "request_api_version": 2,
    }
    module.discover_lexicons = lambda **kwargs: discovery  # type: ignore[attr-defined]
    module.inspect_g2lex_file = lambda path: {  # type: ignore[attr-defined]
        "valid": True,
        "compatible": True,
        "path": path,
        "sha256": "c" * 64,
        "logical_sha256": "d" * 64,
        "language": "en",
        "phoneme_encoding": "ipa",
        "metadata": {},
    }
    monkeypatch.setitem(sys.modules, "kittensynth", module)
    adapter = KittenSynthEngineAdapter()

    first = adapter.canonical_synthesis_identity(
        _selection(lexicons=("public-en",), g2lex_files=("/one/local.g2lex",))
    )
    same_source_other_path = adapter.canonical_synthesis_identity(
        _selection(lexicons=("public-en",), g2lex_files=("/two/local.g2lex",))
    )
    module.inspect_g2lex_file = lambda path: {  # type: ignore[attr-defined]
        "valid": True,
        "compatible": True,
        "path": path,
        "sha256": "e" * 64,
        "logical_sha256": "f" * 64,
        "language": "en",
        "phoneme_encoding": "ipa",
        "metadata": {},
    }
    changed_source = adapter.canonical_synthesis_identity(
        _selection(lexicons=("public-en",), g2lex_files=("/one/local.g2lex",))
    )

    assert first == same_source_other_path
    assert first != changed_source
    pronunciation = first["pronunciation_identity"]
    assert pronunciation["local_g2lex"] == [
        {"order": 0, "sha256": "c" * 64, "logical_sha256": "d" * 64}
    ]
    assert pronunciation["named_lexicons"][0]["asset_id"] == "asset-en-v1"
    assert pronunciation["frontend_identity"]["kitteng2p_revision"] == "g2p-revision-1"
