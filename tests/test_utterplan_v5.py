from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from voicerender.engines.base import EngineCapabilities, SynthesisRequest
from voicerender.errors import PlanError, ProfileError
from voicerender.inputs.utterplan_v5 import (
    load_utterplan_v5,
    lower_directives,
    parse_utterplan_v5,
    resolve_logical_voice,
    select_segments,
)
from voicerender.models import VoiceBinding, VoiceConfig, VoiceProfile

FIXTURES = Path(__file__).parent / "fixtures"


def _canonical_hash(value):
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _synthetic_plan() -> dict:
    flow = []
    flow_semantics = []
    for flow_index in range(191):
        count = 2 if flow_index < 112 else 1
        segments = [
            {"text": f"Flow {flow_index} segment {segment_index}", "language": "en-US"}
            for segment_index in range(count)
        ]
        semantics = [
            {
                "text": segment["text"],
                "language": segment["language"],
                "directives": {},
                "tokens": [],
                "markers": [],
            }
            for segment in segments
        ]
        flow.append(
            {"hash": _canonical_hash({"segments": semantics}), "segment": segments}
        )
        flow_semantics.append({"segments": semantics})
    data = {
        "format": "utterplan",
        "schema_version": 5,
        "language": "en-US",
        "unit": "paragraph",
        "hash_schema": "utterplan-flow-v1",
        "flow": flow,
    }
    data["plan_id"] = _canonical_hash(
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
    return data


def test_reads_canonical_multi_segment_v5_fixture_and_exact_file_hash():
    path = FIXTURES / "utterplan_v5_contract.toml"
    raw = path.read_bytes()
    plan = load_utterplan_v5(path)

    assert (
        plan.plan_id
        == "sha256:d806e55fabbb3428621850f8e6a6dee76842acfd13c134bb4f68d980afa98efa"
    )
    assert plan.file_sha256 == hashlib.sha256(raw).hexdigest()
    assert len(plan.flows) == 1
    assert [segment.source.locator for segment in plan.segments] == [
        "f000000.s000",
        "f000000.s001",
    ]
    assert plan.segments[0].logical_voice == "narrator"
    assert plan.segments[0].language == "en-GB"
    assert plan.segments[0].pause_after == "sentence"
    assert plan.segments[0].semantic_segment_hash == (
        "sha256:ce1ad652872ef383ffd232b903e5bef5a5645b2db6db927384a4814be304bb0b"
    )


def test_v5_flattening_finds_191_flows_and_303_speech_segments():
    plan = parse_utterplan_v5(_synthetic_plan(), file_sha256="a" * 64)
    assert len(plan.flows) == 191
    assert len(plan.segments) == 303
    assert plan.segments[0].source.locator == "f000000.s000"
    assert plan.segments[2].source.locator == "f000001.s000"


def test_selection_uses_flat_source_order_and_exact_take():
    plan = parse_utterplan_v5(_synthetic_plan(), file_sha256="a" * 64)
    selection = select_segments(plan, select="f0.s1,f0.s0")
    assert [segment.source.segment_index for segment in selection.segments] == [0, 1]
    assert selection.as_dict() == {
        "kind": "locators",
        "order": "source",
        "selected_count": 2,
        "is_full_plan": False,
        "locators": ["f000000.s000", "f000000.s001"],
    }
    preview = select_segments(plan, take=8)
    assert len(preview.segments) == 8
    assert preview.as_dict()["is_full_plan"] is False
    with pytest.raises(PlanError, match="Duplicate"):
        select_segments(plan, select="f0.s0,f000000.s000")


def test_v5_unknown_schema_fields_and_bad_flow_hash_fail_closed():
    data = _synthetic_plan()
    data["schema_version"] = 6
    with pytest.raises(PlanError, match="schema_version=5"):
        parse_utterplan_v5(data, file_sha256="a" * 64)

    data = _synthetic_plan()
    data["flow"][0]["segment"][0]["extra_semantics"] = "must not disappear"
    with pytest.raises(PlanError, match=r"Unknown flow\[0\].segment\[0\] fields"):
        parse_utterplan_v5(data, file_sha256="a" * 64)

    data = _synthetic_plan()
    data["flow"][0]["hash"] = "sha256:" + "f" * 64
    with pytest.raises(PlanError, match="flow hash mismatch"):
        parse_utterplan_v5(data, file_sha256="a" * 64)

    data = _synthetic_plan()
    data["plan_id"] = "sha256:" + "f" * 64
    with pytest.raises(PlanError, match="plan_id mismatch"):
        parse_utterplan_v5(data, file_sha256="a" * 64)


def test_voice_toml_merges_backend_defaults_before_role_options(tmp_path):
    path = tmp_path / "voice.toml"
    path.write_text(
        """schema_version = 1
name = "book cast"
[defaults]
role = "narrator"
language = "en-US"
[backend.piper]
native_speed = 1.25
noise_scale = 0.5
[roles.narrator]
engine = "piper"
model = "en_US-amy-medium"
[roles.narrator.options]
native_speed = 1.75
""",
        encoding="utf-8",
    )
    profile = VoiceConfig.load(path)

    assert isinstance(profile, VoiceProfile)
    assert profile.default_role == "narrator"
    assert profile.default_language == "en-US"
    assert profile.roles["narrator"].target_id == "en_US-amy-medium"
    assert profile.roles["narrator"].options == {
        "native_speed": 1.75,
        "noise_scale": 0.5,
    }
    assert (
        resolve_logical_voice(
            load_utterplan_v5(FIXTURES / "utterplan_v5_contract.toml").segments[0],
            profile,
        )
        == "narrator"
    )


def test_role_without_logical_voice_requires_default_or_sole_role(tmp_path):
    plan = load_utterplan_v5(FIXTURES / "utterplan_v5_contract.toml")
    segment = plan.segments[1]
    two_roles = VoiceProfile(
        name="cast",
        roles={
            "narrator": VoiceBinding(engine="piper", target_id="amy"),
            "guest": VoiceBinding(engine="piper", target_id="bob"),
        },
    )
    with pytest.raises(ProfileError, match="default role"):
        resolve_logical_voice(segment, two_roles)
    one_role = VoiceProfile(
        name="solo",
        roles={"narrator": VoiceBinding(engine="piper", target_id="amy")},
    )
    assert resolve_logical_voice(segment, one_role) == "narrator"


def test_unimplemented_v5_directives_are_not_silently_dropped():
    plan = load_utterplan_v5(FIXTURES / "utterplan_v5_contract.toml")
    with pytest.raises(PlanError, match="Unsupported UtterPlan directives.*prosody"):
        lower_directives(plan.segments[0], EngineCapabilities())
    assert lower_directives(plan.segments[1], EngineCapabilities()) == ({}, {})


def test_typed_synthesis_request_has_explicit_source_semantics():
    request = SynthesisRequest("f000000.s000", "Hello.", "en-GB")
    assert request.request_id == "f000000.s000"
    assert request.directives == {}
