from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from voicerender.errors import ArtifactError
from voicerender.manifest import (
    EDGE_POLICY,
    MANIFEST_FILENAME,
    MANIFEST_ID_ALGORITHM,
    MANIFEST_SCHEMA,
    read_manifest,
    seal_manifest,
    validate_manifest_document,
    validate_manifest_source,
)
from voicerender.storage import atomic_json, atomic_wav


def _make_document(tmp_path: Path) -> tuple[Path, dict]:
    bundle = tmp_path / "bundle"
    wav_path = bundle / "segments" / "voice.wav"
    wav_info = atomic_wav(wav_path, np.linspace(-0.3, 0.3, 32), 16000)
    source_file_hash = "a" * 64
    entry = {
        "source": {
            "locator": "f000000.s000",
            "flow_index": 0,
            "segment_index": 0,
            "flow_hash": "sha256:" + "b" * 64,
            "semantic_segment_hash": "sha256:" + "c" * 64,
        },
        "logical_voice": "narrator",
        "language": "en-US",
        "resolved_voice": {
            "engine": "piper",
            "model": "amy",
            "model_asset_sha256": None,
            "voice": None,
            "speaker": None,
            "model_revision": None,
            "reference_audio_sha256": None,
            "effective_binding_sha256": "sha256:" + "d" * 64,
        },
        "synthesis_semantics_sha256": "sha256:" + "e" * 64,
        "acoustic_cache_key": "sha256:" + "f" * 64,
        "semantic_lowering": {
            "native_directives": [],
            "composition_directives": [],
            "unsupported_directives": [],
        },
        "edges": {
            "policy": EDGE_POLICY,
            "source_frames": wav_info["frames"],
            "trim_start_frames": 0,
            "trim_end_frames": 0,
            "leading_silence_frames": None,
            "trailing_silence_frames": None,
        },
        "timing": {
            "provenance": "absent",
            "unit": "frames",
            "sample_rate": wav_info["sample_rate"],
            "spans": [],
        },
        "status": "rendered",
        "audio": {
            "path": "segments/voice.wav",
            "sha256": wav_info["sha256"],
            "sample_rate": wav_info["sample_rate"],
            "channels": wav_info["channels"],
            "frames": wav_info["frames"],
            "encoding": "PCM_16",
        },
        "error": None,
    }
    document = {
        "schema": MANIFEST_SCHEMA,
        "manifest_id_algorithm": MANIFEST_ID_ALGORITHM,
        "source": {
            "kind": "utterplan-v5",
            "plan_id": "sha256:" + "0" * 64,
            "plan_file_sha256": "sha256:" + source_file_hash,
            "schema_version": 5,
        },
        "producer": {
            "name": "voicerender",
            "version": "0.2.0",
            "algorithm": "v5-direct.v1",
        },
        "profile": {
            "name": "test-cast",
            "effective_profile_sha256": "sha256:" + "1" * 64,
        },
        "selection": {
            "kind": "all",
            "order": "source",
            "selected_count": 1,
            "is_full_plan": True,
        },
        "edge_policy": EDGE_POLICY,
        "complete": True,
        "segments": [entry],
    }
    seal_manifest(document)
    path = bundle / MANIFEST_FILENAME
    atomic_json(path, document)
    return path, document


def test_manifest_v2_identity_and_verified_pcm_audio(tmp_path):
    path, document = _make_document(tmp_path)
    assert document["manifest_id"].startswith("sha256:")
    loaded = read_manifest(path)
    assert loaded["manifest_id"] == document["manifest_id"]
    assert loaded["segments"][0]["audio"]["encoding"] == "PCM_16"

    reused = copy.deepcopy(document)
    reused["segments"][0]["status"] = "reused"
    seal_manifest(reused)
    assert reused["manifest_id"] == document["manifest_id"]
    assert validate_manifest_document(reused)["complete"] is True


def test_manifest_identity_rejects_tampering_and_stale_plan_join(tmp_path):
    path, document = _make_document(tmp_path)
    changed = copy.deepcopy(document)
    changed["profile"]["name"] = "other"
    with pytest.raises(ArtifactError, match="identity mismatch"):
        validate_manifest_document(changed)

    plan = SimpleNamespace(plan_id=document["source"]["plan_id"], file_sha256="a" * 64)
    validate_manifest_source(document, plan)
    plan.plan_id = "sha256:" + "3" * 64
    with pytest.raises(ArtifactError, match="plan_id"):
        validate_manifest_source(document, plan)
    plan.plan_id = document["source"]["plan_id"]
    plan.file_sha256 = "2" * 64
    with pytest.raises(ArtifactError, match="exact source-file hash"):
        validate_manifest_source(document, plan)
    assert path.exists()


def test_manifest_checks_selection_completeness_and_timing_spans(tmp_path):
    _, document = _make_document(tmp_path)
    broken_selection = copy.deepcopy(document)
    broken_selection["selection"]["selected_count"] = 2
    seal_manifest(broken_selection)
    with pytest.raises(ArtifactError, match="segments length"):
        validate_manifest_document(broken_selection)

    with_spans = copy.deepcopy(document)
    timing = with_spans["segments"][0]["timing"]
    timing["provenance"] = "provided"
    timing["spans"] = [
        {"start_frame": 1, "end_frame": 8, "text": "hello"},
        {"start_frame": 10, "end_frame": 35, "text": "world"},
    ]
    seal_manifest(with_spans)
    with pytest.raises(ArtifactError, match="timing span exceeds"):
        validate_manifest_document(with_spans)

    timing["spans"] = [
        {"start_frame": 10, "end_frame": 20},
        {"start_frame": 1, "end_frame": 3},
    ]
    seal_manifest(with_spans)
    with pytest.raises(ArtifactError, match="timing spans must be ordered"):
        validate_manifest_document(with_spans)


def test_manifest_rejects_escape_paths_symlink_retargeting_and_corrupt_audio(tmp_path):
    path, document = _make_document(tmp_path)
    for bad_path in ("../outside.wav", "/tmp/outside.wav", "C:\\outside.wav"):
        unsafe = copy.deepcopy(document)
        unsafe["segments"][0]["audio"]["path"] = bad_path
        seal_manifest(unsafe)
        with pytest.raises(
            ArtifactError, match="relative POSIX path|escape the bundle root"
        ):
            validate_manifest_document(unsafe)

    duplicate = copy.deepcopy(document)
    duplicate["selection"]["selected_count"] = 2
    duplicate["segments"].append(copy.deepcopy(duplicate["segments"][0]))
    seal_manifest(duplicate)
    with pytest.raises(ArtifactError, match="duplicate source locator"):
        validate_manifest_document(duplicate)

    duplicate = copy.deepcopy(document)
    duplicate["selection"]["selected_count"] = 2
    second = copy.deepcopy(duplicate["segments"][0])
    second["source"]["locator"] = "f000000.s001"
    second["source"]["segment_index"] = 1
    duplicate["segments"].append(second)
    seal_manifest(duplicate)
    with pytest.raises(ArtifactError, match="duplicate audio path"):
        validate_manifest_document(duplicate)

    bundle = path.parent
    wav_path = bundle / "segments" / "voice.wav"
    outside = tmp_path / "outside.wav"
    atomic_wav(outside, np.ones(32, dtype=np.float32) * 0.2, 16000)
    wav_path.unlink()
    wav_path.symlink_to(outside)
    with pytest.raises(ArtifactError, match="symlink"):
        read_manifest(path)

    wav_path.unlink()
    atomic_wav(wav_path, np.ones(32, dtype=np.float32) * 0.2, 16000)
    with pytest.raises(ArtifactError, match="WAV sha256 mismatch"):
        read_manifest(path)


def test_manifest_schema_resource_is_valid_json():
    schema_path = (
        Path(__file__).parents[1]
        / "voicerender"
        / "schemas"
        / "render-manifest-v2.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["properties"]["schema"]["const"] == MANIFEST_SCHEMA
