from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from voicerender.engines import _ADAPTERS
from voicerender.engines.base import EngineCapabilities, SynthesisOutput
from voicerender.errors import ArtifactError, PausePolicyError
from voicerender.inputs.utterplan_v5 import parse_utterplan_v5
from voicerender.manifest import MANIFEST_FILENAME, read_manifest, seal_manifest
from voicerender.models import VoiceBinding, VoiceProfile
from voicerender.render_v5 import render_utterplan_v5
from voicerender.storage import atomic_json


def _hash(value):
    content = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _plan(
    texts,
    *,
    pause_before=None,
    pause_after=None,
    directives=None,
    file_hash="a" * 64,
):
    pause_before = pause_before or {}
    pause_after = pause_after or {}
    directives = directives or {}
    segments = []
    semantic_segments = []
    for index, text in enumerate(texts):
        segment = {"text": text, "language": "en-US"}
        projection = {
            "text": text,
            "language": "en-US",
            "directives": {},
            "tokens": [],
            "markers": [],
        }
        for name, values in (
            ("pause_before", pause_before),
            ("pause_after", pause_after),
        ):
            if index in values:
                segment[name] = values[index]
                projection[name] = values[index]
        if index in directives:
            segment.update(directives[index])
            normalized = {}
            for name, value in directives[index].items():
                if name == "prosody":
                    normalized[name] = {
                        key: value.get(key) for key in ("rate", "pitch", "volume")
                    }
                elif name == "emphasis":
                    normalized[name] = {"level": value["level"]}
                else:
                    normalized[name] = value
            projection["directives"] = normalized
        segments.append(segment)
        semantic_segments.append(projection)
    flow = [{"hash": _hash({"segments": semantic_segments}), "segment": segments}]
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
            "flow": [{"segments": semantic_segments}],
        }
    )
    return parse_utterplan_v5(data, file_sha256=file_hash)


class _Session:
    def synthesize_request(self, request):
        return SynthesisOutput(
            audio=np.full(100, 0.1, dtype=np.float32),
            sample_rate=16_000,
            word_timings=(
                {
                    "kind": "word",
                    "start_frame": 10,
                    "end_frame": 40,
                    "text": request.text,
                    "text_start": 0,
                    "text_end": len(request.text),
                },
            ),
        )


class _Adapter:
    id = "audiocompose-test"
    package = "audiocompose-test"

    def __init__(self, composition_semantics=()):
        self.capabilities = EngineCapabilities(
            supports_synthesis_request=True,
            composition_semantics=frozenset(composition_semantics),
        )
        self.calls = []

    def version(self):
        return "1.0"

    def validate(self, binding):
        assert binding.engine == self.id

    @contextmanager
    def open(self, binding):
        yield self

    def synthesize_request(self, request):
        self.calls.append(request)
        return _Session().synthesize_request(request)


@pytest.fixture(autouse=True)
def _require_audiocompose():
    pytest.importorskip("audiocompose")


@pytest.fixture
def rendered_bundle(tmp_path, monkeypatch):
    def render(plan, *, take=None, select=None, composition_semantics=()):
        pytest.importorskip("audiocompose")
        adapter = _Adapter(composition_semantics)
        monkeypatch.setitem(_ADAPTERS, adapter.id, adapter)
        profile = VoiceProfile(
            name="narrator",
            roles={
                "narrator": VoiceBinding(engine=adapter.id, target_id="fixture-model"),
                "dialogue": VoiceBinding(engine=adapter.id, target_id="fixture-model"),
            },
            defaults={"role": "narrator"},
        )
        bundle = tmp_path / f"bundle-{len(list(tmp_path.glob('bundle-*')))}"
        result = render_utterplan_v5(
            plan,
            profile=profile,
            output_dir=bundle,
            cache_dir=tmp_path / "cache",
            take=take,
            select=select,
        )
        return bundle / MANIFEST_FILENAME, adapter, result

    return render


def _policy(**default):
    from voicerender.integrations.audiocompose import PAUSE_SCHEMA

    return {"schema": PAUSE_SCHEMA, "merge": "max", "default": default}


def test_voice_specific_pause_override_uses_the_cue_owner(tmp_path, rendered_bundle):
    from voicerender.integrations.audiocompose import build_audio_job

    plan = _plan(
        ["Narrator", "Dialogue"],
        pause_before={1: "sentence"},
        directives={1: {"voice": {"reference": "dialogue"}}},
    )
    manifest_path, _, _ = rendered_bundle(plan)
    policy = {
        "schema": "voicerender.pause-policy.v1",
        "merge": "max",
        "default": {"sentence": "120ms"},
        "voice": {"dialogue": {"sentence": "480ms"}},
    }
    job = build_audio_job(plan, manifest_path, policy)
    assert job.items[1].seconds == pytest.approx(0.480)


def test_full_and_prefix_jobs_merge_boundary_cues_and_omit_preview_tail(
    tmp_path, rendered_bundle
):
    from audiocompose import AudioClip, Silence

    from voicerender.integrations.audiocompose import build_audio_job

    plan = _plan(
        ["Hello", "there", "again"],
        pause_after={0: "sentence", 1: {"type": "timed", "time": "500ms"}},
        pause_before={1: "parenthetical"},
    )
    policy = _policy(sentence="320ms", parenthetical="110ms")
    manifest_path, adapter, _ = rendered_bundle(plan)

    full = build_audio_job(plan, manifest_path, policy)
    assert [type(item) for item in full.items] == [
        AudioClip,
        Silence,
        AudioClip,
        Silence,
        AudioClip,
    ]
    assert full.items[1].seconds == pytest.approx(0.320)
    assert full.items[3].seconds == pytest.approx(0.500)
    assert full.items[0].spans[0].source_start == 0
    assert full.items[0].spans[0].source_end == len("Hello")
    assert full.items[0].spans[0].sample_start == 10
    assert full.items[0].spans[0].metadata["kind"] == "word"
    assert full.job_id.startswith("sha256:")
    full.validate(verify_sources=True)

    prefix_path, _, _ = rendered_bundle(plan, take=2)
    prefix = build_audio_job(plan, prefix_path, policy)
    assert sum(isinstance(item, AudioClip) for item in prefix.items) == 2
    assert sum(isinstance(item, Silence) for item in prefix.items) == 1
    assert prefix.items[-1].id == "speech:f000000.s001"
    assert len(adapter.calls) == 3


def test_voice_override_and_pause_policy_edits_change_job_identity_not_audio(
    tmp_path, rendered_bundle
):
    from voicerender.integrations.audiocompose import build_audio_job

    plan = _plan(["One", "Two"], pause_after={0: "sentence"})
    manifest_path, adapter, _ = rendered_bundle(plan)
    first = build_audio_job(plan, manifest_path, _policy(sentence="300ms"))
    second = build_audio_job(plan, manifest_path, _policy(sentence="360ms"))
    assert first.job_id != second.job_id
    assert first.items[1].seconds == pytest.approx(0.3)
    assert second.items[1].seconds == pytest.approx(0.36)
    assert len(adapter.calls) == 2


def test_strict_exact_plan_join_and_audio_corruption_fail(tmp_path, rendered_bundle):
    from voicerender.integrations.audiocompose import build_audio_job

    plan = _plan(["Integrity."])
    manifest_path, _, _ = rendered_bundle(plan)
    different_fingerprint = _plan(["Integrity."], file_hash="b" * 64)
    with pytest.raises(ArtifactError, match="exact source-file hash"):
        build_audio_job(different_fingerprint, manifest_path, _policy())

    document = read_manifest(manifest_path)
    wav_path = manifest_path.parent / document["segments"][0]["audio"]["path"]
    wav_path.write_bytes(b"corrupt media")
    with pytest.raises(ArtifactError, match="WAV"):
        build_audio_job(plan, manifest_path, _policy())


def test_discontiguous_selection_and_unknown_pause_fail_closed(
    tmp_path, rendered_bundle
):
    from voicerender.integrations.audiocompose import build_audio_job

    plan = _plan(
        ["First", "Skipped", "Third"],
        pause_after={0: "sentence"},
    )
    selected_manifest, _, _ = rendered_bundle(plan, select="f0.s0,f0.s2")
    with pytest.raises(ArtifactError, match="contiguous"):
        build_audio_job(plan, selected_manifest, _policy())

    contiguous_manifest, _, _ = rendered_bundle(plan, take=2)
    with pytest.raises(PausePolicyError, match="plan_id.*pause_after.*sentence"):
        build_audio_job(plan, contiguous_manifest, _policy())


def test_typed_composition_operation_is_materialized_as_numeric_audio_operation(
    tmp_path, rendered_bundle
):
    from audiocompose import Tempo

    from voicerender.integrations.audiocompose import build_audio_job

    plan = _plan(["Slowly"], directives={0: {"prosody": {"rate": "slow"}}})
    manifest_path, _, _ = rendered_bundle(plan, composition_semantics={"prosody"})
    document = read_manifest(manifest_path)
    document["segments"][0]["semantic_lowering"]["composition_directives"][0][
        "value"
    ] = {"type": "tempo", "factor": 0.8}
    seal_manifest(document)
    atomic_json(manifest_path, document)

    job = build_audio_job(plan, manifest_path, _policy())
    assert len(job.items[0].operations) == 1
    assert isinstance(job.items[0].operations[0], Tempo)
    assert job.items[0].operations[0].factor == pytest.approx(0.8)


def test_audiojob_save_uses_supported_copying_bundle_contract(
    tmp_path, rendered_bundle
):
    from audiocompose import AudioJob

    from voicerender.integrations.audiocompose import build_audio_job

    plan = _plan(["Portable bundle."])
    manifest_path, _, _ = rendered_bundle(plan)
    job = build_audio_job(plan, manifest_path, _policy())
    saved = Path(job.save(tmp_path / "audiojob-bundle"))
    loaded = AudioJob.load(saved)
    assert loaded.job_id == job.job_id or loaded.job_id.startswith("sha256:")
    assert loaded.items[0].source.path != job.items[0].source.path
