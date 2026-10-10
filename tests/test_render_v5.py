from __future__ import annotations

import hashlib
import json
import math
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voicerender import (
    EngineRequest,
    RenderEvent,
    RenderEventCallbackError,
    RenderEventKind,
    open_session,
    render_utterplan,
    resolve_voice,
    select_segments,
    synthesize,
)
from voicerender.engines.api_probe import EngineApiProbe
from voicerender.engines.base import (
    EngineCapabilities,
    EngineSelection,
    RenderedSpeech,
    RequestMeasure,
    SpeechRequest,
    SpeechWordTiming,
)
from voicerender.engines.registry import _ADAPTERS
from voicerender.errors import (
    ArtifactError,
    EngineApiIncompatibleError,
    PlanError,
    UnsupportedSynthesisFeatureError,
)
from voicerender.handoff import validate_manifest_source, verify_render_manifest
from voicerender.inputs.utterplan_v5 import load_utterplan_v5, parse_utterplan_v5
from voicerender.manifest import MANIFEST_FILENAME, read_manifest
from voicerender.models import VoiceBinding, VoiceProfile
from voicerender.render_v5 import render_utterplan_v5


def _hash(value):
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _plan(
    operations,
    *,
    file_hash="a" * 64,
    pause_after=None,
    prosody=False,
    pronunciation=False,
    tokens=False,
    document_semantics=None,
):
    document_semantics = document_semantics or {}
    segments = []
    semantics = []
    for index, operation in enumerate(operations):
        if isinstance(operation, str):
            text = operation
            raw_audio = None
        else:
            text = operation.get("text", "")
            raw_audio = operation.get("audio")
        segment = {"text": text, "language": "en-US"}
        projection = {
            "text": text,
            "language": "en-US",
            "directives": {},
            "tokens": [],
            "markers": [],
        }
        directives = {}
        if raw_audio is not None:
            segment["audio"] = raw_audio
            audio = {
                "src": raw_audio["src"],
                "alt_text": raw_audio.get("alt_text"),
                "clip_begin": raw_audio.get("clip_begin"),
                "clip_end": raw_audio.get("clip_end"),
                "speed": raw_audio.get("speed"),
                "repeat_duration": raw_audio.get("repeat_duration"),
                "repeat_count": raw_audio.get("repeat_count"),
                "sound_level": raw_audio.get("sound_level"),
            }
            if raw_audio.get("description") is not None:
                audio["description"] = raw_audio["description"]
            directives["audio"] = audio
        if prosody and index == 0:
            prosody_value = {"rate": "x-fast"} if prosody is True else dict(prosody)
            segment["prosody"] = prosody_value
            directives["prosody"] = {
                key: prosody_value.get(key) for key in ("rate", "pitch", "volume")
            }
        if pronunciation and index == 0:
            pronunciation_value = {"phonemes": "həˈləʊ", "alphabet": "ipa"}
            segment["pronunciation"] = pronunciation_value
            directives["pronunciation"] = pronunciation_value
        if tokens and index == 0 and text:
            segment["token"] = {"span": [[0, len(text)]]}
            projection["tokens"] = [
                {"start": 0, "end": len(text), "lemma": text.casefold()}
            ]
        if directives:
            projection["directives"] = directives
        if pause_after is not None and index == len(operations) - 1:
            segment["pause_after"] = pause_after
            projection["pause_after"] = pause_after
        segments.append(segment)
        semantics.append(projection)
    flow = [{"hash": _hash({"segments": semantics}), "segment": segments}]
    flow_semantics = [{"segments": semantics}]
    data = {
        "format": "utterplan",
        "schema_version": 5,
        "language": "en-US",
        "unit": "paragraph",
        "hash_schema": "utterplan-flow-v1",
        "flow": flow,
    }
    if document_semantics:
        data["document"] = {"semantics": document_semantics}
    data["plan_id"] = _hash(
        {
            "format": "utterplan",
            "schema_version": 5,
            "language": "en-US",
            "unit": "paragraph",
            "hash_schema": "utterplan-flow-v1",
            "document_semantics": document_semantics,
            "flow": flow_semantics,
        }
    )
    return parse_utterplan_v5(data, file_sha256=file_hash)


class _FakeSession:
    def __init__(self, calls):
        self.calls = calls

    def measure(self, request: SpeechRequest):
        return RequestMeasure(
            fits=True,
            amount=len(request.text),
            maximum=len(request.text),
            unit="unknown",
            source="test-fake",
        )

    def synthesize(self, request: SpeechRequest):
        self.calls.append(request)
        audio = np.zeros(32, dtype=np.float32)
        audio[8:24] = 0.2
        word = request.text.split()[0]
        start = request.text.index(word)
        return RenderedSpeech(
            id=request.id,
            audio=audio,
            sample_rate=16000,
            word_timings=(
                SpeechWordTiming(
                    text=word,
                    char_start=start,
                    char_end=start + len(word),
                    start_sample=8,
                    end_sample=24,
                ),
            ),
        )


class _FakeAdapter:
    id = "manifest-test"
    package_name = "manifest-test"

    def __init__(self, *, pronunciation=False, tokens=False):
        self.calls = []
        self.open_count = 0
        self.supports_pronunciation = pronunciation
        self.supports_tokens = tokens

    def capabilities(self):
        return EngineCapabilities(
            id=self.id,
            supports_live=True,
            supports_timestamps=True,
            supports_pronunciation_overrides=self.supports_pronunciation,
            pronunciation_alphabets=frozenset({"ipa"})
            if self.supports_pronunciation
            else frozenset(),
            supports_linguistic_tokens=self.supports_tokens,
        )

    def version(self):
        return "1.0"

    def probe_api(self):
        return EngineApiProbe(
            engine=self.id,
            package=self.package_name,
            compatible=True,
            status="ready",
            distribution_version="1.0",
            expected_api_version=1,
            api_version=1,
            contract_source="explicit",
        )

    def resolve(self, request):
        return (
            EngineSelection(
                engine=self.id,
                target_id=request.target_id or "model-1",
                language=request.language or "en-US",
                voice=request.voice,
                speaker=request.speaker,
                options=request.options,
            ),
            (),
        )

    def validate_selection(self, selection):
        assert selection.engine == self.id
        return ()

    def canonical_synthesis_identity(self, selection):
        options = dict(selection.options)
        source = options.get("voice_source")
        if isinstance(source, dict):
            options["voice_source"] = {
                key: source[key] for key in ("kind", "sha256") if key in source
            }
        return {
            "engine": self.id,
            "target_id": selection.target_id,
            "voice": selection.voice,
            "speaker": selection.speaker,
            "options": options,
        }

    @contextmanager
    def open(self, selection):
        self.open_count += 1
        yield _FakeSession(self.calls)


def _profile():
    return VoiceProfile(
        name="narrator",
        roles={"narrator": VoiceBinding(engine="manifest-test", target_id="model-1")},
    )


def test_take_eight_renders_exact_selection_and_resumes_verified_segments(
    tmp_path, monkeypatch
):
    adapter = _FakeAdapter()
    monkeypatch.setitem(_ADAPTERS, adapter.id, adapter)
    plan = _plan([f"Sentence {index}." for index in range(10)])
    bundle = tmp_path / "bundle"
    cache = tmp_path / "cache"

    result = render_utterplan_v5(
        plan,
        profile=_profile(),
        output_dir=bundle,
        cache_dir=cache,
        take=8,
    )
    manifest_path = bundle / MANIFEST_FILENAME
    manifest = read_manifest(manifest_path)
    assert result.rendered_count == 8
    assert len(adapter.calls) == 8
    assert manifest["selection"]["take"] == 8
    assert manifest["selection"]["selected_count"] == 8
    assert manifest["complete"] is True
    assert [entry["source"]["segment_index"] for entry in manifest["segments"]] == list(
        range(8)
    )
    first = manifest["segments"][0]
    assert first["edges"]["source_frames"] == 32
    assert first["edges"]["trim_start_frames"] == first["edges"]["trim_end_frames"] == 0
    assert first["edges"]["leading_silence_frames"] is None
    assert first["timing"]["provenance"] == "provided"
    audio_path = bundle / first["audio"]["path"]
    audio, sample_rate = sf.read(audio_path)
    assert sample_rate == 16000
    assert len(audio) == 32
    assert audio[0] == audio[-1] == 0.0

    resumed = render_utterplan_v5(
        plan,
        profile=_profile(),
        output_dir=bundle,
        cache_dir=cache,
        take=8,
    )
    assert resumed.rendered_count == 0
    assert resumed.reused_count == 8
    assert len(adapter.calls) == 8
    assert read_manifest(manifest_path)["complete"] is True


def test_batch_requests_use_shared_capacity_and_rebase_timing(tmp_path, monkeypatch):
    class LimitedSession(_FakeSession):
        def measure(self, request):
            maximum = 7
            return RequestMeasure(
                fits=len(request.text) <= maximum,
                amount=len(request.text),
                maximum=maximum,
                unit="unknown",
                source="test-limit",
            )

    class LimitedAdapter(_FakeAdapter):
        @contextmanager
        def open(self, selection):
            yield LimitedSession(self.calls)

    adapter = LimitedAdapter()
    monkeypatch.setitem(_ADAPTERS, adapter.id, adapter)
    bundle = tmp_path / "capacity-bundle"
    result = render_utterplan_v5(
        _plan(["alpha beta gamma"]),
        profile=_profile(),
        output_dir=bundle,
        cache_dir=tmp_path / "capacity-cache",
    )

    manifest = read_manifest(result.manifest_path)
    entry = manifest["segments"][0]
    assert result.rendered_count == 1
    assert [request.text for request in adapter.calls] == ["alpha ", "beta ", "gamma"]
    assert "".join(request.text for request in adapter.calls) == "alpha beta gamma"
    assert entry["audio"]["frames"] == 96
    assert entry["timing"]["spans"] == [
        {
            "start_frame": 8,
            "end_frame": 24,
            "text": "alpha",
            "text_start": 0,
            "text_end": 5,
        },
        {
            "start_frame": 40,
            "end_frame": 56,
            "text": "beta",
            "text_start": 6,
            "text_end": 10,
        },
        {
            "start_frame": 72,
            "end_frame": 88,
            "text": "gamma",
            "text_start": 11,
            "text_end": 16,
        },
    ]


def test_pause_changes_semantic_identity_but_not_acoustic_cache_and_text_invalidates(
    tmp_path, monkeypatch
):
    adapter = _FakeAdapter()
    monkeypatch.setitem(_ADAPTERS, adapter.id, adapter)
    profile = _profile()
    cache = tmp_path / "shared-cache"
    plain = _plan(["Same speech."], file_hash="a" * 64)
    paused = _plan(["Same speech."], file_hash="b" * 64, pause_after="sentence")
    first = render_utterplan_v5(
        plain,
        profile=profile,
        output_dir=tmp_path / "plain",
        cache_dir=cache,
    )
    second = render_utterplan_v5(
        paused,
        profile=profile,
        output_dir=tmp_path / "paused",
        cache_dir=cache,
    )
    plain_manifest = read_manifest(first.manifest_path)
    paused_manifest = read_manifest(second.manifest_path)
    assert (
        plain_manifest["segments"][0]["source"]["semantic_segment_hash"]
        != paused_manifest["segments"][0]["source"]["semantic_segment_hash"]
    )
    assert (
        plain_manifest["segments"][0]["acoustic_cache_key"]
        == paused_manifest["segments"][0]["acoustic_cache_key"]
    )
    assert second.reused_count == 1
    assert len(adapter.calls) == 1

    changed_text = _plan(["Different speech."], file_hash="c" * 64)
    third = render_utterplan_v5(
        changed_text,
        profile=profile,
        output_dir=tmp_path / "changed",
        cache_dir=cache,
    )
    changed_manifest = read_manifest(third.manifest_path)
    assert (
        changed_manifest["segments"][0]["acoustic_cache_key"]
        != plain_manifest["segments"][0]["acoustic_cache_key"]
    )
    assert third.rendered_count == 1
    assert len(adapter.calls) == 2


def test_corrupt_shared_cache_wav_is_not_reused(tmp_path, monkeypatch):
    adapter = _FakeAdapter()
    monkeypatch.setitem(_ADAPTERS, adapter.id, adapter)
    plan = _plan(["Cache integrity."])
    cache = tmp_path / "cache"
    first = render_utterplan_v5(
        plan, profile=_profile(), output_dir=tmp_path / "first", cache_dir=cache
    )
    manifest = read_manifest(first.manifest_path)
    key = manifest["segments"][0]["acoustic_cache_key"].removeprefix("sha256:")
    (cache / f"{key}.wav").write_bytes(b"corrupt cache artifact")

    second = render_utterplan_v5(
        plan, profile=_profile(), output_dir=tmp_path / "second", cache_dir=cache
    )
    assert second.rendered_count == 1
    assert second.reused_count == 0
    assert len(adapter.calls) == 2


def test_reference_audio_cache_identity_uses_bytes_not_path(tmp_path, monkeypatch):
    adapter = _FakeAdapter()
    monkeypatch.setitem(_ADAPTERS, adapter.id, adapter)
    reference_a = tmp_path / "a.wav"
    reference_b = tmp_path / "b.wav"
    reference_a.write_bytes(b"reference bytes")
    reference_b.write_bytes(b"reference bytes")
    profile_a = VoiceProfile(
        name="a",
        roles={
            "narrator": VoiceBinding(
                engine=adapter.id, target_id="model-1", reference_audio=reference_a
            )
        },
    )
    profile_b = VoiceProfile(
        name="b",
        roles={
            "narrator": VoiceBinding(
                engine=adapter.id, target_id="model-1", reference_audio=reference_b
            )
        },
    )
    plan = _plan(["Reference identity."])
    cache = tmp_path / "shared-cache"
    first = render_utterplan_v5(
        plan, profile=profile_a, output_dir=tmp_path / "a", cache_dir=cache
    )
    same_bytes = render_utterplan_v5(
        plan, profile=profile_b, output_dir=tmp_path / "b", cache_dir=cache
    )
    first_key = read_manifest(first.manifest_path)["segments"][0]["acoustic_cache_key"]
    same_key = read_manifest(same_bytes.manifest_path)["segments"][0][
        "acoustic_cache_key"
    ]
    assert first_key == same_key
    assert same_bytes.reused_count == 1

    reference_b.write_bytes(b"different reference bytes")
    changed = render_utterplan_v5(
        plan, profile=profile_b, output_dir=tmp_path / "changed-ref", cache_dir=cache
    )
    changed_key = read_manifest(changed.manifest_path)["segments"][0][
        "acoustic_cache_key"
    ]
    assert changed_key != first_key
    assert changed.rendered_count == 1
    assert len(adapter.calls) == 2


def test_source_insertion_changes_locators_but_reuses_unchanged_acoustics(
    tmp_path, monkeypatch
):
    adapter = _FakeAdapter()
    monkeypatch.setitem(_ADAPTERS, adapter.id, adapter)
    profile = _profile()
    cache = tmp_path / "cache"
    original = _plan(["Unchanged one.", "Unchanged two."], file_hash="1" * 64)
    inserted = _plan(
        ["New introduction.", "Unchanged one.", "Unchanged two."],
        file_hash="2" * 64,
    )
    first = render_utterplan_v5(
        original, profile=profile, output_dir=tmp_path / "original", cache_dir=cache
    )
    second = render_utterplan_v5(
        inserted, profile=profile, output_dir=tmp_path / "inserted", cache_dir=cache
    )
    first_manifest = read_manifest(first.manifest_path)
    second_manifest = read_manifest(second.manifest_path)
    assert second_manifest["segments"][1]["source"]["locator"] == "f000000.s001"
    assert (
        second_manifest["segments"][1]["acoustic_cache_key"]
        == first_manifest["segments"][0]["acoustic_cache_key"]
    )
    assert second.reused_count == 2
    assert second.rendered_count == 1
    assert len(adapter.calls) == 3


def test_failed_synthesis_leaves_final_failure_in_incomplete_manifest(
    tmp_path, monkeypatch
):
    class FailingSession(_FakeSession):
        def synthesize(self, request):
            raise RuntimeError("engine unavailable")

    class FailingAdapter(_FakeAdapter):
        @contextmanager
        def open(self, binding):
            yield FailingSession(self.calls)

    adapter = FailingAdapter()
    monkeypatch.setitem(_ADAPTERS, adapter.id, adapter)
    bundle = tmp_path / "failed"
    plan = _plan(["Will fail."])
    try:
        render_utterplan_v5(
            plan,
            profile=_profile(),
            output_dir=bundle,
            cache_dir=tmp_path / "cache",
        )
    except RuntimeError as exc:
        assert str(exc) == "engine unavailable"
    else:
        raise AssertionError("expected synthesis failure")

    manifest = read_manifest(bundle / MANIFEST_FILENAME, verify_audio=False)
    assert manifest["complete"] is False
    assert manifest["segments"][0]["status"] == "failed"
    assert manifest["segments"][0]["error"]["type"] == "RuntimeError"

    success_adapter = _FakeAdapter()
    monkeypatch.setitem(_ADAPTERS, success_adapter.id, success_adapter)
    resumed = render_utterplan_v5(
        plan,
        profile=_profile(),
        output_dir=bundle,
        cache_dir=tmp_path / "cache",
    )
    assert resumed.rendered_count == 1
    assert resumed.reused_count == 0
    assert read_manifest(bundle / MANIFEST_FILENAME)["complete"] is True
    assert len(success_adapter.calls) == 1


def test_unsupported_document_semantics_fail_before_manifest_or_synthesis(
    tmp_path, monkeypatch
):
    adapter = _FakeAdapter()
    monkeypatch.setitem(_ADAPTERS, adapter.id, adapter)
    plan = load_utterplan_v5(
        Path(__file__).parent / "fixtures" / "utterplan_v5_contract.toml"
    )
    bundle = tmp_path / "must-not-exist"
    try:
        render_utterplan_v5(
            plan,
            profile=_profile(),
            output_dir=bundle,
            select="f0.s0",
        )
    except PlanError as exc:
        assert exc.code == "plan.unsupported_semantics"
        assert "Unsupported required UtterPlan extensions" in str(exc)
    else:
        raise AssertionError("unsupported semantics must be rejected")
    assert not bundle.exists()
    assert adapter.calls == []
    assert adapter.open_count == 0


def test_unsupported_segment_directive_fails_before_manifest_or_synthesis(
    tmp_path, monkeypatch
):
    adapter = _FakeAdapter()
    monkeypatch.setitem(_ADAPTERS, adapter.id, adapter)
    bundle = tmp_path / "unsupported-directive"
    try:
        render_utterplan_v5(
            _plan(["Unsupported prosody."], prosody=True),
            profile=_profile(),
            output_dir=bundle,
            cache_dir=tmp_path / "cache",
        )
    except UnsupportedSynthesisFeatureError as exc:
        assert exc.code == "synthesis.unsupported_feature"
        assert "prosody.rate" in str(exc)
    else:
        raise AssertionError("unsupported directives must be rejected")
    assert not bundle.exists()
    assert adapter.calls == []
    assert adapter.open_count == 0


def test_supported_document_semantics_are_preserved_in_the_manifest(
    tmp_path, monkeypatch
):
    adapter = _FakeAdapter()
    monkeypatch.setitem(_ADAPTERS, adapter.id, adapter)
    semantics = {
        "voice_bindings": {"narrator": "renderer-owned-choice"},
        "prosody_transitions": {
            "enabled": True,
            "same_voice_only": True,
            "rate": "100ms",
        },
    }
    result = render_utterplan_v5(
        _plan(["Known semantics."], document_semantics=semantics),
        profile=_profile(),
        output_dir=tmp_path / "known-semantics",
    )

    manifest = read_manifest(result.manifest_path)
    assert manifest["document_semantics"] == semantics
    assert adapter.open_count == 1


def test_invalid_document_transition_duration_fails_in_preflight(tmp_path, monkeypatch):
    adapter = _FakeAdapter()
    monkeypatch.setitem(_ADAPTERS, adapter.id, adapter)
    bundle = tmp_path / "invalid-transition"
    semantics = {"prosody_transitions": {"enabled": True, "rate": "fast"}}

    with pytest.raises(PlanError, match="finite non-negative duration"):
        render_utterplan_v5(
            _plan(["Invalid transition."], document_semantics=semantics),
            profile=_profile(),
            output_dir=bundle,
        )

    assert adapter.open_count == 0
    assert adapter.calls == []
    assert not bundle.exists()


def test_role_options_override_backend_defaults_without_replacing_plan_language(
    tmp_path, monkeypatch
):
    adapter = _FakeAdapter()
    monkeypatch.setitem(_ADAPTERS, adapter.id, adapter)
    profile = VoiceProfile(
        name="precedence",
        defaults={"language": "fr-FR"},
        backend_defaults={adapter.id: {"speed": 0.75, "pitch": 1.0}},
        roles={
            "narrator": VoiceBinding(
                engine=adapter.id,
                target_id="model-1",
                options={"speed": 1.25},
            )
        },
    )
    render_utterplan_v5(
        _plan(["Authored language."]),
        profile=profile,
        output_dir=tmp_path / "precedence",
    )

    request = adapter.calls[0]
    assert request.language == "en-US"
    assert request.options == {"speed": 1.25, "pitch": 1.0}


def test_supported_pronunciation_and_tokens_are_bound_to_the_acoustic_cache_key(
    tmp_path, monkeypatch
):
    adapter = _FakeAdapter(pronunciation=True, tokens=True)
    monkeypatch.setitem(_ADAPTERS, adapter.id, adapter)
    cache = tmp_path / "semantic-cache"
    contextual = render_utterplan_v5(
        _plan(["Hello world."], pronunciation=True, tokens=True),
        profile=_profile(),
        output_dir=tmp_path / "contextual",
        cache_dir=cache,
    )
    plain = render_utterplan_v5(
        _plan(["Hello world."]),
        profile=_profile(),
        output_dir=tmp_path / "plain",
        cache_dir=cache,
    )

    contextual_entry = read_manifest(contextual.manifest_path)["segments"][0]
    plain_entry = read_manifest(plain.manifest_path)["segments"][0]
    request = adapter.calls[0]
    assert request.tokens[0].text == "Hello world."
    assert request.pronunciation_overrides[0].phonemes == "həˈləʊ"
    native = {
        item["name"]: item["value"]
        for item in contextual_entry["semantic_lowering"]["native_directives"]
    }
    assert native["linguistic_tokens"][0]["text"] == "Hello world."
    assert native["pronunciation"]["alphabet"] == "ipa"
    assert contextual_entry["acoustic_cache_key"] != plain_entry["acoustic_cache_key"]
    assert adapter.open_count == 2


def test_unsupported_tokens_fail_before_engine_open(tmp_path, monkeypatch):
    adapter = _FakeAdapter()
    monkeypatch.setitem(_ADAPTERS, adapter.id, adapter)
    bundle = tmp_path / "unsupported-tokens"

    with pytest.raises(UnsupportedSynthesisFeatureError) as exc_info:
        render_utterplan_v5(
            _plan(["Token support."], tokens=True),
            profile=_profile(),
            output_dir=bundle,
        )

    assert exc_info.value.code == "synthesis.unsupported_feature"
    assert "tokens" in str(exc_info.value)
    assert adapter.open_count == 0
    assert adapter.calls == []
    assert not bundle.exists()


def test_incompatible_engine_api_fails_before_open_with_stable_details(
    tmp_path, monkeypatch
):
    class IncompatibleAdapter(_FakeAdapter):
        def probe_api(self):
            return EngineApiProbe(
                engine=self.id,
                package=self.package_name,
                compatible=False,
                status="api_incompatible",
                distribution_version="2.0",
                expected_api_version=1,
                error_type="AttributeError",
            )

    adapter = IncompatibleAdapter()
    monkeypatch.setitem(_ADAPTERS, adapter.id, adapter)
    bundle = tmp_path / "incompatible-api"

    with pytest.raises(EngineApiIncompatibleError) as exc_info:
        render_utterplan_v5(
            _plan(["Do not open."]),
            profile=_profile(),
            output_dir=bundle,
        )

    assert exc_info.value.code == "engine.api_incompatible"
    assert exc_info.value.details["probe"]["status"] == "api_incompatible"
    assert adapter.open_count == 0
    assert adapter.calls == []
    assert not bundle.exists()


def test_numeric_prosody_is_versioned_and_does_not_pollute_acoustic_cache(
    tmp_path, monkeypatch
):
    adapter = _FakeAdapter()
    monkeypatch.setitem(_ADAPTERS, adapter.id, adapter)
    cache = tmp_path / "prosody-cache"
    prosodic = render_utterplan_v5(
        _plan(
            ["Numeric prosody."],
            prosody={"rate": "slow", "pitch": "+12%", "volume": "soft"},
        ),
        profile=_profile(),
        output_dir=tmp_path / "prosodic",
        cache_dir=cache,
    )
    plain = render_utterplan_v5(
        _plan(["Numeric prosody."]),
        profile=_profile(),
        output_dir=tmp_path / "plain-prosody",
        cache_dir=cache,
    )

    prosodic_entry = read_manifest(prosodic.manifest_path)["segments"][0]
    plain_entry = read_manifest(plain.manifest_path)["segments"][0]
    lowering = prosodic_entry["semantic_lowering"]
    composition = {
        item["name"]: item["value"] for item in lowering["composition_directives"]
    }
    assert lowering["composition_schema"] == "voicerender.numeric-composition.v1"
    assert composition["tempo"] == {"factor": 0.8}
    assert composition["pitch_shift"]["semitones"] == pytest.approx(
        12 * math.log2(1.12)
    )
    assert composition["gain"] == {"db": -6.0}
    assert (
        prosodic_entry["synthesis_semantics_sha256"]
        != plain_entry["synthesis_semantics_sha256"]
    )
    assert prosodic_entry["acoustic_cache_key"] == plain_entry["acoustic_cache_key"]
    assert plain_entry["status"] == "reused"


def test_model_asset_bytes_are_committed_without_absolute_manifest_paths(
    tmp_path, monkeypatch
):
    adapter = _FakeAdapter()
    monkeypatch.setitem(_ADAPTERS, adapter.id, adapter)
    model_file = tmp_path / "model.bin"
    model_file.write_bytes(b"model-v1")
    profile = VoiceProfile(
        name="local-model",
        roles={"narrator": VoiceBinding(engine=adapter.id, target_id=str(model_file))},
    )
    plan = _plan(["Model identity."])
    cache = tmp_path / "cache"
    first = render_utterplan_v5(
        plan, profile=profile, output_dir=tmp_path / "model-one", cache_dir=cache
    )
    first_entry = read_manifest(first.manifest_path)["segments"][0]
    assert first_entry["resolved_voice"]["model"] == "model.bin"
    assert first_entry["resolved_voice"]["model_asset_sha256"] == (
        "sha256:" + hashlib.sha256(b"model-v1").hexdigest()
    )
    assert str(tmp_path) not in json.dumps(first_entry["resolved_voice"])

    model_file.write_bytes(b"model-v2")
    second = render_utterplan_v5(
        plan, profile=profile, output_dir=tmp_path / "model-two", cache_dir=cache
    )
    second_entry = read_manifest(second.manifest_path)["segments"][0]
    assert second_entry["acoustic_cache_key"] != first_entry["acoustic_cache_key"]
    assert second.rendered_count == 1


def test_render_refuses_symlinked_output_directory_before_synthesis(
    tmp_path, monkeypatch
):
    adapter = _FakeAdapter()
    monkeypatch.setitem(_ADAPTERS, adapter.id, adapter)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (bundle / "segments").symlink_to(outside, target_is_directory=True)

    try:
        render_utterplan_v5(
            _plan(["No path escape."]),
            profile=_profile(),
            output_dir=bundle,
            cache_dir=tmp_path / "cache",
        )
    except ArtifactError as exc:
        assert "symlink" in str(exc)
    else:
        raise AssertionError("output symlink must be rejected")

    assert list(outside.iterdir()) == []
    pending = read_manifest(bundle / MANIFEST_FILENAME, verify_audio=False)
    assert pending["complete"] is False
    assert pending["segments"][0]["status"] == "failed"
    assert pending["segments"][0]["error"]["type"] == "ArtifactError"
    assert adapter.calls == []


def test_mixed_speech_audio_selection_only_synthesizes_speech_and_verifies_handoff(
    tmp_path, monkeypatch
):
    adapter = _FakeAdapter()
    monkeypatch.setitem(_ADAPTERS, adapter.id, adapter)
    plan = _plan(
        [
            "Speech before media.",
            {"text": "", "audio": {"src": "chapter/music.wav", "alt_text": "music"}},
            "Speech after media.",
        ]
    )
    result = render_utterplan_v5(
        plan,
        profile=_profile(),
        output_dir=tmp_path / "mixed",
        cache_dir=tmp_path / "cache",
        select="f0.s2,f0.s1,f0.s0",
    )
    manifest = verify_render_manifest(result.manifest_path)

    assert result.selected_count == 2
    assert len(adapter.calls) == 2
    assert manifest["selection"] == {
        "kind": "locators",
        "order": "source",
        "selected_count": 2,
        "selected_source_count": 3,
        "selected_speech_count": 2,
        "is_full_plan": True,
        "locators": ["f000000.s000", "f000000.s001", "f000000.s002"],
    }
    assert [operation["kind"] for operation in manifest["source_operations"]] == [
        "speech",
        "external_audio",
        "speech",
    ]
    assert manifest["source_operations"][1]["audio_directive"]["src"] == (
        "chapter/music.wav"
    )
    assert [entry["source"]["locator"] for entry in manifest["segments"]] == [
        "f000000.s000",
        "f000000.s002",
    ]
    validate_manifest_source(plan, manifest)


def test_all_media_selection_emits_complete_zero_speech_handoff(tmp_path):
    plan = _plan(
        [
            {"text": "", "audio": {"src": "media/one.wav"}},
            {"text": "", "audio": {"src": "media/two.wav"}},
        ]
    )
    result = render_utterplan_v5(
        plan,
        profile=_profile(),
        output_dir=tmp_path / "media-only",
        cache_dir=tmp_path / "cache",
    )
    manifest = verify_render_manifest(result.manifest_path)

    assert result.rendered_count == result.reused_count == result.failed_count == 0
    assert result.selected_count == 0
    assert manifest["complete"] is True
    assert manifest["selection"]["selected_source_count"] == 2
    assert manifest["selection"]["selected_speech_count"] == 0
    assert manifest["segments"] == []
    assert [item["kind"] for item in manifest["source_operations"]] == [
        "external_audio",
        "external_audio",
    ]
    assert not list((tmp_path / "media-only").glob("segments/*.wav"))
    validate_manifest_source(plan, manifest)


def test_empty_non_audio_operation_fails_before_opening_an_engine(
    tmp_path, monkeypatch
):
    adapter = _FakeAdapter()
    monkeypatch.setitem(_ADAPTERS, adapter.id, adapter)
    plan = _plan([{"text": ""}])

    try:
        render_utterplan_v5(
            plan,
            profile=_profile(),
            output_dir=tmp_path / "unsupported",
            cache_dir=tmp_path / "cache",
        )
    except PlanError as exc:
        assert "Unsupported/non-renderable" in str(exc)
    else:
        raise AssertionError("empty non-audio operations must not be rendered")
    assert adapter.calls == []


def test_public_render_api_events_are_typed_ordered_and_resume_aware(
    tmp_path, monkeypatch
):
    adapter = _FakeAdapter()
    monkeypatch.setitem(_ADAPTERS, adapter.id, adapter)
    plan = _plan(["One event.", "Two events."])
    selection = select_segments(plan, take=1)
    output = tmp_path / "public-api"
    events: list[RenderEvent] = []

    result = render_utterplan(
        plan,
        profile=_profile(),
        output_dir=output,
        selection=selection,
        on_event=events.append,
    )

    assert result.selected_count == 1
    assert [event.kind for event in events] == [
        RenderEventKind.PREFLIGHT_STARTED,
        RenderEventKind.PREFLIGHT_FINISHED,
        RenderEventKind.SEGMENT_STARTED,
        RenderEventKind.ENGINE_OPEN_STARTED,
        RenderEventKind.ENGINE_OPEN_FINISHED,
        RenderEventKind.SEGMENT_RENDERED,
        RenderEventKind.RENDER_COMPLETED,
    ]
    assert all(isinstance(event, RenderEvent) for event in events)
    assert events[-1].rendered_count == 1
    assert events[-1].selected_count == 1
    resolved_voice = read_manifest(result.manifest_path)["segments"][0][
        "resolved_voice"
    ]
    assert (
        resolved_voice["engine_identity"]["schema"] == "voicerender.engine-identity.v1"
    )
    assert resolved_voice["engine_identity"]["sha256"].startswith("sha256:")
    assert resolved_voice["engine_api_version"] == 1
    assert resolved_voice["engine_api_contract"] == "explicit"

    resumed_events: list[RenderEvent] = []
    resumed = render_utterplan(
        plan,
        profile=_profile(),
        output_dir=output,
        selection=selection,
        on_event=resumed_events.append,
    )
    assert resumed.rendered_count == 0
    assert resumed.reused_count == 1
    assert [event.kind for event in resumed_events] == [
        RenderEventKind.PREFLIGHT_STARTED,
        RenderEventKind.PREFLIGHT_FINISHED,
        RenderEventKind.SEGMENT_STARTED,
        RenderEventKind.SEGMENT_REUSED,
        RenderEventKind.RENDER_COMPLETED,
    ]


def test_public_render_event_callback_failure_is_not_a_synthesis_failure(tmp_path):
    def fail_callback(event: RenderEvent) -> None:
        raise RuntimeError(event.kind.value)

    with pytest.raises(RenderEventCallbackError) as exc_info:
        render_utterplan(
            _plan(["Callback failure."]),
            profile=_profile(),
            output_dir=tmp_path / "callback-failure",
            on_event=fail_callback,
        )

    assert exc_info.value.code == "render.event_callback_failed"
    assert not (tmp_path / "callback-failure").exists()


def test_public_live_session_returns_bound_acoustic_identity(monkeypatch):
    adapter = _FakeAdapter()
    monkeypatch.setitem(_ADAPTERS, adapter.id, adapter)
    selection, diagnostics = resolve_voice(
        EngineRequest(engine=adapter.id, target_id="model-1", language="en-US")
    )
    assert diagnostics == ()

    request = SpeechRequest(id="live-request", text="Live request.", language="en-US")
    with open_session(selection) as session:
        result = synthesize(session, request)

    identity = result.metadata["acoustic_identity"]
    assert identity["schema"] == "voicerender.live-acoustic-identity.v1"
    assert identity["engine_identity"]["engine"] == adapter.id
    assert identity["request"]["text"] == request.text
    assert identity["capacity_schema"]
    assert identity["sha256"].startswith("sha256:")
