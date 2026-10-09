from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import soundfile as sf

from voicerender.engines import _ADAPTERS
from voicerender.engines.base import (
    EngineCapabilities,
    SynthesisOutput,
    SynthesisRequest,
)
from voicerender.errors import ArtifactError, PlanError
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


def _plan(texts, *, file_hash="a" * 64, pause_after=None, prosody=False):
    segments = []
    semantics = []
    for index, text in enumerate(texts):
        segment = {"text": text, "language": "en-US"}
        projection = {
            "text": text,
            "language": "en-US",
            "directives": {},
            "tokens": [],
            "markers": [],
        }
        if prosody and index == 0:
            segment["prosody"] = {"rate": "slow"}
            projection["directives"] = {
                "prosody": {"rate": "slow", "pitch": None, "volume": None}
            }
        if pause_after is not None and index == len(texts) - 1:
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
    data["plan_id"] = _hash(
        {
            "format": "utterplan",
            "schema_version": 5,
            "language": "en-US",
            "unit": "paragraph",
            "hash_schema": "utterplan-flow-v1",
            "document_semantics": {},
            "flow": flow_semantics,
        }
    )
    return parse_utterplan_v5(data, file_sha256=file_hash)


class _FakeSession:
    def __init__(self, calls):
        self.calls = calls

    def synthesize_request(self, request: SynthesisRequest):
        self.calls.append(request)
        audio = np.zeros(32, dtype=np.float32)
        audio[8:24] = 0.2
        return SynthesisOutput(
            audio=audio,
            sample_rate=16000,
            word_timings=({"start_frame": 8, "end_frame": 24, "text": "sample"},),
        )

    def synthesize(self, **kwargs):
        raise AssertionError("typed requests should be used")


class _FakeAdapter:
    id = "manifest-test"
    package = "manifest-test"
    capabilities = EngineCapabilities(supports_synthesis_request=True)

    def __init__(self):
        self.calls = []

    def version(self):
        return "1.0"

    def validate(self, binding):
        assert binding.engine == self.id

    @contextmanager
    def open(self, binding):
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
        def synthesize_request(self, request):
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
        assert "Unsupported UtterPlan document semantics" in str(exc)
    else:
        raise AssertionError("unsupported semantics must be rejected")
    assert not bundle.exists()
    assert adapter.calls == []


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
    except PlanError as exc:
        assert "Unsupported UtterPlan directives" in str(exc)
    else:
        raise AssertionError("unsupported directives must be rejected")
    assert not bundle.exists()
    assert adapter.calls == []


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
    assert pending["segments"][0]["status"] == "pending"
    assert adapter.calls == []
