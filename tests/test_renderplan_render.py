from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

import pytest
from renderplan import (
    CapacityConstraint,
    PreflightReport,
    RenderPlan,
    RenderPlanSet,
    RenderPlanSetChapter,
    RenderProfile,
    RoleProfile,
    SourceSpan,
    SpeechControl,
    SpeechItem,
    load_manifest,
    load_preflight,
    save_plan,
    save_profile,
    save_renderplan_set,
)

from voicerender.engines.api_probe import EngineApiProbe
from voicerender.engines.base import (
    EngineCapabilities,
    EngineDiagnostic,
    EngineSelection,
    EngineSession,
    RenderedSpeech,
    RenderProfileFragment,
    RequestMeasure,
    SpeechRequest,
)
from voicerender.engines.registry import _ADAPTERS
from voicerender.engines.selection import EngineRequest
from voicerender.errors import (
    ArtifactError,
    PlanError,
    UnsupportedSynthesisFeatureError,
)
from voicerender.preflight import preflight_render_plan
from voicerender.renderplan_render import (
    RENDER_DIAGNOSTICS_FILENAME,
    render_renderplan,
)
from voicerender.storage import wav_info

_ENGINE_ID = "renderplan-render-test-engine"
_TARGET_ID = "renderplan-render-target-v1"
_BINDING: dict[str, Any] = {
    "renderer": _ENGINE_ID,
    "target_id": _TARGET_ID,
    "voice": "voice-a",
    "options": {"speed": 1.0},
    "renderer_metadata": {"frontend": "fake-v1"},
}


class _Session:
    def __init__(self, adapter: _Adapter) -> None:
        self.adapter = adapter

    def measure(self, request: SpeechRequest) -> RequestMeasure:
        self.adapter.measured_texts.append(request.text)
        value = self.adapter.measurements.get(request.text)
        if isinstance(value, Exception):
            raise value
        if value is not None:
            return value
        return RequestMeasure(
            fits=True,
            amount=len(request.text),
            maximum=100,
            unit="model_tokens",
            source="fake.frontend_measure",
        )

    def synthesize(self, request: SpeechRequest) -> RenderedSpeech:
        self.adapter.synthesized_texts.append(request.text)
        self.adapter.synthesized_requests.append(request)
        return RenderedSpeech(
            id=request.id,
            audio=[0.0, 0.2, -0.2, 0.1],
            sample_rate=24000,
        )


class _Adapter:
    id = _ENGINE_ID
    package_name = _ENGINE_ID

    def __init__(self) -> None:
        self.open_count = 0
        self.close_count = 0
        self.measured_texts: list[str] = []
        self.synthesized_texts: list[str] = []
        self.synthesized_requests: list[SpeechRequest] = []
        self.measurements: dict[str, RequestMeasure | Exception] = {}
        self.open_error: Exception | None = None
        self.session = _Session(self)

    def probe_api(self) -> EngineApiProbe:
        return EngineApiProbe(
            engine=self.id,
            package=self.package_name,
            compatible=True,
            status="ready",
            expected_api_version=1,
            api_version=1,
            contract_source="explicit",
        )

    def version(self) -> str:
        return "1.0"

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(id=self.id)

    def resolve(
        self, request: EngineRequest
    ) -> tuple[EngineSelection, tuple[EngineDiagnostic, ...]]:
        return (
            EngineSelection(
                engine=self.id,
                target_id=request.target_id or _TARGET_ID,
                language=request.language or "en-US",
                voice=request.voice,
                speaker=request.speaker,
                options=request.options,
            ),
            (),
        )

    def validate_selection(
        self, selection: EngineSelection
    ) -> tuple[EngineDiagnostic, ...]:
        if selection.target_id != _TARGET_ID:
            return (
                EngineDiagnostic(
                    code="test.target_unknown",
                    severity="error",
                    message="unknown test target",
                ),
            )
        return ()

    def target_metadata(self, selection: EngineSelection) -> dict[str, Any]:
        return {"source_revision": "fake-r1"}

    def canonical_synthesis_identity(
        self, selection: EngineSelection
    ) -> dict[str, Any]:
        return {
            "target_id": selection.target_id,
            "voice": selection.voice,
            "options": dict(selection.options),
            "revision": "fake-r1",
        }

    def renderprofile_fragment(
        self, selection: EngineSelection
    ) -> RenderProfileFragment:
        return RenderProfileFragment(
            binding={
                "renderer": self.id,
                "target_id": selection.target_id,
                "voice": selection.voice,
                "options": dict(selection.options),
            },
            token_fields=(),
            frontend="fake-v1",
            pronunciation="fake-v1",
            renderer_capacity=CapacityConstraint(
                "model_tokens", maximum=100, guarantee="hard"
            ),
        )

    @contextmanager
    def open(self, selection: EngineSelection) -> Iterator[EngineSession]:
        self.open_count += 1
        try:
            if self.open_error is not None:
                raise self.open_error
            yield self.session
        finally:
            self.close_count += 1


def _install(monkeypatch: pytest.MonkeyPatch) -> _Adapter:
    adapter = _Adapter()
    monkeypatch.setitem(_ADAPTERS, adapter.id, adapter)
    return adapter


def _plan(*texts: str) -> RenderPlan:
    flow = tuple(
        SpeechItem(
            number=number,
            text=text,
            language="en-US",
            role="narrator",
            binding=dict(_BINDING),
            source=(
                SourceSpan(
                    locator=f"f{number:06d}.s000",
                    start=0,
                    end=len(text),
                ),
            ),
            planning_level="sentence",
            preflight_required=True,
        )
        for number, text in enumerate(texts, start=1)
    )
    return RenderPlan(
        source_plan_id="sha256:" + "3" * 64,
        render_profile_id="sha256:" + "4" * 64,
        flow=flow,
    ).with_identity()


def _report(adapter: _Adapter, plan: RenderPlan, tmp_path: Path) -> PreflightReport:
    del adapter
    return preflight_render_plan(plan, tmp_path / "preflight.toml")

def test_render_defers_emphasis_without_changing_wav_or_manifest_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = _install(monkeypatch)
    text = "raw voice"
    plain_plan = _plan(text)
    plain_report = preflight_render_plan(plain_plan, tmp_path / "plain-preflight.toml")
    plain_result = render_renderplan(
        plain_plan,
        preflight=plain_report,
        output_dir=tmp_path / "plain-render",
        cache_dir=tmp_path / "cache",
    )

    item = replace(
        plain_plan.speech[0],
        segment_id=None,
        controls=(SpeechControl(start=0, end=3, name="emphasis", params={}),),
    )
    emphasis_plan = replace(plain_plan, flow=(item,), plan_id=None).with_identity()
    original_plan_id = emphasis_plan.plan_id
    original_segment_id = item.computed_id()
    report = preflight_render_plan(
        emphasis_plan,
        tmp_path / "emphasis-preflight.toml",
        emphasis_policy="defer-to-compose",
    )
    adapter.synthesized_texts.clear()
    adapter.synthesized_requests.clear()

    result = render_renderplan(
        emphasis_plan,
        preflight=report,
        output_dir=tmp_path / "emphasis-render",
        cache_dir=tmp_path / "cache",
        emphasis_policy="defer-to-compose",
    )

    assert not result.failed
    assert result.pending_composition is True
    assert adapter.synthesized_texts == [text]
    assert adapter.synthesized_requests[0].text == text
    assert "renderplan_controls" not in adapter.synthesized_requests[0].options
    assert result.manifest.render_plan_id == original_plan_id
    assert result.manifest.speech[0].segment_id == original_segment_id
    assert result.manifest.speech[0].sha256 == plain_result.manifest.speech[0].sha256
    result.manifest.validate_against(emphasis_plan)
    assert item.controls == (SpeechControl(start=0, end=3, name="emphasis", params={}),)

    diagnostics_path = tmp_path / "emphasis-render" / RENDER_DIAGNOSTICS_FILENAME
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert diagnostics["pending_composition"] is True
    assert diagnostics["emphasis_policy"] == "defer-to-compose"
    assert diagnostics["deferred_controls"] == [
        {
            "segment_number": 1,
            "segment_id": original_segment_id,
            "capability": "audiocompose.emphasis.gain.v1",
            "count": 1,
        }
    ]
    assert "pending composition" in diagnostics["warnings"][0]


def test_partial_render_diagnostics_preserve_plan_wide_pending_emphasis(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install(monkeypatch)
    base_plan = _plan("partial windows")
    emphasized = replace(
        base_plan.speech[0],
        segment_id=None,
        controls=(SpeechControl(start=0, end=3, name="emphasis", params={}),),
    )
    plain = replace(base_plan.speech[0], number=2, segment_id=None, controls=())
    plan = replace(base_plan, flow=(emphasized, plain), plan_id=None).with_identity()
    report = preflight_render_plan(
        plan,
        tmp_path / "preflight.toml",
        emphasis_policy="defer-to-compose",
    )

    second_window = render_renderplan(
        plan,
        preflight=report,
        output_dir=tmp_path / "partial-render",
        cache_dir=tmp_path / "cache",
        speech_numbers={2},
        emphasis_policy="defer-to-compose",
    )
    assert second_window.pending_composition is True
    assert not second_window.manifest.complete

    first_window = render_renderplan(
        plan,
        preflight=report,
        output_dir=tmp_path / "partial-render",
        cache_dir=tmp_path / "cache",
        speech_numbers={1},
        emphasis_policy="defer-to-compose",
    )
    assert first_window.pending_composition is True
    assert first_window.manifest.complete
    diagnostics = json.loads(
        (tmp_path / "partial-render" / RENDER_DIAGNOSTICS_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert diagnostics["pending_composition"] is True
    assert diagnostics["deferred_controls"] == [
        {
            "segment_number": 1,
            "segment_id": emphasized.computed_id(),
            "capability": "audiocompose.emphasis.gain.v1",
            "count": 1,
        }
    ]


def test_render_policy_must_match_deferred_preflight(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install(monkeypatch)
    plan = _plan("policy agreement")
    item = replace(
        plan.speech[0],
        segment_id=None,
        controls=(SpeechControl(start=0, end=6, name="emphasis", params={}),),
    )
    plan = replace(plan, flow=(item,), plan_id=None).with_identity()
    report = preflight_render_plan(
        plan,
        tmp_path / "deferred-preflight.toml",
        emphasis_policy="defer-to-compose",
    )

    with pytest.raises(
        UnsupportedSynthesisFeatureError, match="explicitly select"
    ):
        render_renderplan(
            plan,
            preflight=report,
            output_dir=tmp_path / "mismatched-policy",
        )
    assert not (tmp_path / "mismatched-policy").exists()


def test_render_executes_exact_requests_in_order_and_writes_numbered_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = _install(monkeypatch)
    plan = _plan(" first ", "second ")
    report = _report(adapter, plan, tmp_path)
    adapter.measured_texts.clear()
    adapter.synthesized_texts.clear()
    adapter.open_count = 0
    adapter.close_count = 0

    result = render_renderplan(
        plan,
        preflight=report,
        output_dir=tmp_path / "rendered",
        cache_dir=tmp_path / "cache",
    )

    assert not result.failed
    assert result.manifest.complete
    assert result.rendered_count == 2
    assert result.reused_count == 0
    assert adapter.measured_texts == [" first ", "second "]
    assert adapter.synthesized_texts == [" first ", "second "]
    assert adapter.open_count == 1
    assert adapter.close_count == 1
    assert len(result.execution_fingerprint) == 2
    fingerprint = result.execution_fingerprint[0]
    assert fingerprint["voicerender_version"]
    assert fingerprint["adapter_id"] == _ENGINE_ID
    assert fingerprint["adapter_version"] == "1.0"
    assert fingerprint["api_probe_signature"]
    assert fingerprint["canonical_synthesis_identity"]["target_id"] == _TARGET_ID
    assert fingerprint["model"] == _TARGET_ID
    assert fingerprint["voice"] == "voice-a"
    assert fingerprint["options"] == {"speed": 1.0}
    assert fingerprint["capacity"] == {
        "metric": "model_tokens",
        "measured": len(" first "),
        "maximum": 100,
        "source": "fake.frontend_measure",
    }
    assert fingerprint["fingerprint_id"].startswith("sha256:")
    assert [item.wav for item in result.manifest.speech] == ["000001.wav", "000002.wav"]
    assert [item.segment_id for item in result.manifest.speech] == [
        item.segment_id for item in plan.speech
    ]
    for artifact in result.manifest.speech:
        path = result.manifest_path.parent / artifact.wav
        info = wav_info(path)
        assert artifact.sha256 == f"sha256:{info['sha256']}"
        assert artifact.frames == info["frames"]
        assert artifact.sample_rate == info["sample_rate"] == 24000
        assert artifact.channels == info["channels"] == 1
    loaded = load_manifest(result.manifest_path)
    loaded.validate_against(plan)
    assert loaded.manifest_id == result.manifest.manifest_id


def test_verified_cache_reuse_materializes_numbered_wav_and_corruption_regenerates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = _install(monkeypatch)
    plan = _plan("same request")
    report = _report(adapter, plan, tmp_path)
    adapter.open_count = 0
    cache = tmp_path / "shared-cache"
    render_renderplan(
        plan,
        preflight=report,
        output_dir=tmp_path / "first",
        cache_dir=cache,
    )
    assert adapter.synthesized_texts == ["same request"]

    second = render_renderplan(
        plan,
        preflight=report,
        output_dir=tmp_path / "second",
        cache_dir=cache,
    )
    assert second.manifest.complete
    assert second.rendered_count == 0
    assert second.reused_count == 1
    assert (tmp_path / "second" / "000001.wav").is_file()
    assert adapter.synthesized_texts == ["same request"]
    assert adapter.open_count == 1

    cached_wav = next(cache.glob("*.wav"))
    cached_wav.write_bytes(b"corrupt cache")
    third = render_renderplan(
        plan,
        preflight=report,
        output_dir=tmp_path / "third",
        cache_dir=cache,
    )
    assert third.manifest.complete
    assert third.rendered_count == 1
    assert adapter.synthesized_texts == ["same request", "same request"]
    assert wav_info(tmp_path / "third" / "000001.wav")["frames"] > 0


def test_stale_and_missing_ok_reports_fail_before_synthesis(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = _install(monkeypatch)
    plan = _plan("source plan")
    report = _report(adapter, plan, tmp_path)
    stale_plan = _plan("different plan")

    adapter.measured_texts.clear()
    adapter.open_count = 0
    with pytest.raises(PlanError, match="does not match"):
        render_renderplan(
            stale_plan,
            preflight=report,
            output_dir=tmp_path / "stale-output",
        )
    missing_check = replace(report, checks=(), report_id=None).with_identity()
    with pytest.raises(PlanError, match="does not match"):
        render_renderplan(
            plan,
            preflight=missing_check,
            output_dir=tmp_path / "missing-output",
        )
    assert adapter.synthesized_texts == []
    assert adapter.measured_texts == []
    assert adapter.open_count == 0
    assert not (tmp_path / "stale-output").exists()
    assert not (tmp_path / "missing-output").exists()


def test_immediate_remeasurement_failure_stops_and_writes_valid_partial_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = _install(monkeypatch)
    plan = _plan("fits first", "fails next")
    report = _report(adapter, plan, tmp_path)
    adapter.measured_texts.clear()
    adapter.synthesized_texts.clear()
    adapter.measurements["fails next"] = RequestMeasure(
        fits=False,
        amount=101,
        maximum=100,
        unit="model_tokens",
        source="fake.frontend_measure",
    )
    recheck_path = tmp_path / "fresh-replan.toml"

    result = render_renderplan(
        plan,
        preflight=report,
        output_dir=tmp_path / "partial",
        cache_dir=tmp_path / "partial-cache",
        recheck_report_path=recheck_path,
    )

    assert result.failed
    assert result.error is not None
    assert result.error_code == "preflight.replan"
    assert adapter.measured_texts == ["fits first", "fails next"]
    assert adapter.synthesized_texts == ["fits first"]
    assert result.manifest.complete is False
    assert len(result.manifest.speech) == 1
    assert result.manifest.speech[0].segment_id == plan.speech[0].segment_id
    result.manifest.validate_against(plan)
    assert result.fresh_preflight is not None
    assert result.fresh_preflight.status == "replan"
    assert result.fresh_preflight.replan[0].source == plan.speech[1].source
    assert (
        result.fresh_preflight.replan[0].current_level == plan.speech[1].planning_level
    )
    assert load_preflight(recheck_path).status == "replan"
    assert (tmp_path / "partial" / "000001.wav").is_file()
    assert not (tmp_path / "partial" / "000002.wav").exists()
    assert result.fresh_preflight.render_plan_id == plan.plan_id
    assert len(result.fresh_preflight.checks) == 1
    assert result.fresh_preflight.checks[0].segment_number == 2
    assert result.fresh_preflight.checks[0].segment_id == plan.speech[1].segment_id
    assert plan.speech[1].text == "fails next"
    assert result.execution_fingerprint[-1]["capacity"]["maximum"] == 100


def test_session_open_failure_writes_fresh_reject_for_exact_occurrence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = _install(monkeypatch)
    plan = _plan("engine changed after preflight")
    report = _report(adapter, plan, tmp_path)
    adapter.open_error = RuntimeError("backend session unavailable")
    adapter.synthesized_texts.clear()
    recheck_path = tmp_path / "fresh-reject.toml"

    result = render_renderplan(
        plan,
        preflight=report,
        output_dir=tmp_path / "rejected-render",
        recheck_report_path=recheck_path,
    )

    assert result.failed
    assert result.error_code == "preflight.reject"
    assert adapter.synthesized_texts == []
    assert result.fresh_preflight is not None
    assert result.fresh_preflight.status == "reject"
    assert result.fresh_preflight.render_plan_id == plan.plan_id
    assert len(result.fresh_preflight.checks) == 1
    check = result.fresh_preflight.checks[0]
    assert (check.segment_number, check.segment_id) == (
        plan.speech[0].number,
        plan.speech[0].segment_id,
    )
    assert load_preflight(recheck_path).report_id == result.fresh_preflight.report_id
    assert result.execution_fingerprint[0]["execution_status"] == "session_open_failed"
    assert result.execution_fingerprint[0]["capacity"] is None


def test_non_ok_preflight_rejected_without_output_or_synthesis(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = _install(monkeypatch)
    plan = _plan("do not render")
    report = _report(adapter, plan, tmp_path)
    rejected = replace(report, status="reject", report_id=None).with_identity()

    with pytest.raises(PlanError, match="status must be 'ok'"):
        render_renderplan(
            plan,
            preflight=rejected,
            output_dir=tmp_path / "non-ok",
        )
    assert adapter.synthesized_texts == []
    assert not (tmp_path / "non-ok").exists()


def test_cli_renders_and_inspects_shared_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    adapter = _install(monkeypatch)
    plan = _plan("CLI exact request")
    plan_path = tmp_path / "plan.toml"
    report_path = tmp_path / "preflight.toml"
    output_dir = tmp_path / "rendered"
    save_plan(plan, plan_path)
    preflight_render_plan(plan, report_path)
    adapter.measured_texts.clear()
    adapter.synthesized_texts.clear()

    from voicerender.cli import main

    code = main(
        [
            "render",
            str(plan_path),
            "--preflight",
            str(report_path),
            "--out-dir",
            str(output_dir),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--json",
        ]
    )
    render_payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert render_payload["complete"] is True
    assert render_payload["speech_count"] == 1
    assert len(render_payload["execution_fingerprint"]) == 1
    assert render_payload["execution_fingerprint"][0]["adapter_id"] == _ENGINE_ID
    assert render_payload["error_code"] is None
    assert adapter.synthesized_texts == ["CLI exact request"]

    stale_plan_path = tmp_path / "stale.renderplan.toml"
    save_plan(_plan("different CLI plan"), stale_plan_path)
    code = main(
        [
            "render",
            str(stale_plan_path),
            "--preflight",
            str(report_path),
            "--out-dir",
            str(tmp_path / "stale-cli-output"),
            "--json",
        ]
    )
    stale_payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert stale_payload["error_code"] == "plan.invalid"
    assert "does not match" in stale_payload["error"]
    assert adapter.synthesized_texts == ["CLI exact request"]

    code = main(
        ["inspect-manifest", str(output_dir / "render-manifest.toml"), "--json"]
    )
    inspect_payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert inspect_payload["complete"] is True
    assert inspect_payload["verified"] == 1

    plan_bytes = plan_path.read_bytes()
    code = main(
        [
            "render",
            str(plan_path),
            "--preflight",
            str(report_path),
            "--out-dir",
            str(output_dir),
            "--recheck-report",
            str(plan_path),
            "--force",
        ]
    )
    render_error = capsys.readouterr().err
    assert code == 1
    assert "must not overwrite" in render_error
    assert plan_path.read_bytes() == plan_bytes
    assert adapter.synthesized_texts == ["CLI exact request"]


def test_cli_creates_and_validates_shared_renderprofile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    adapter = _install(monkeypatch)
    profile_path = tmp_path / "voice.renderprofile.toml"

    from voicerender.cli import main

    code = main(
        [
            "profile",
            "create",
            "--engine",
            adapter.id,
            "--target",
            _TARGET_ID,
            "--language",
            "en-US",
            "--role",
            "narrator",
            "--voice",
            "voice-a",
            "--option",
            "speed=1.25",
            "--output",
            str(profile_path),
            "--json",
        ]
    )
    created = json.loads(capsys.readouterr().out)
    assert code == 0
    assert created["path"] == str(profile_path)
    assert profile_path.is_file()

    code = main(["profile", "validate", str(profile_path), "--json"])
    validated = json.loads(capsys.readouterr().out)
    assert code == 0
    assert validated["valid"] is True
    assert validated["roles"] == ["narrator"]

    refreshed_path = tmp_path / "voice.refreshed.renderprofile.toml"
    code = main(
        [
            "profile",
            "refresh",
            str(profile_path),
            "--output",
            str(refreshed_path),
            "--json",
        ]
    )
    refreshed = json.loads(capsys.readouterr().out)
    assert code == 0
    assert refreshed["path"] == str(refreshed_path)
    assert refreshed_path.is_file()

    original = profile_path.read_bytes()
    code = main(
        [
            "profile",
            "refresh",
            str(profile_path),
            "--output",
            str(profile_path),
            "--json",
        ]
    )
    refresh_error = capsys.readouterr().err
    assert code == 1
    assert "must not overwrite" in refresh_error
    assert profile_path.read_bytes() == original


def test_partial_speech_windows_keep_original_numbers_and_merge_manifest_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = _install(monkeypatch)
    plan = _plan(*(f"speech {number}" for number in range(1, 61)))
    report = _report(adapter, plan, tmp_path)
    adapter.synthesized_texts.clear()
    output = tmp_path / "chapter"
    cache = tmp_path / "cache"

    first = render_renderplan(
        plan,
        preflight=report,
        output_dir=output,
        cache_dir=cache,
        speech_numbers=range(1, 21),
    )
    assert not first.failed
    assert first.selection_complete
    assert first.selected_count == 20
    assert first.available_count == 20
    assert not first.manifest.complete
    assert (output / "000001.wav").is_file()
    assert (output / "000020.wav").is_file()
    assert not (output / "000021.wav").exists()

    second = render_renderplan(
        plan,
        preflight=report,
        output_dir=output,
        cache_dir=cache,
        speech_numbers=range(50, 61),
    )
    assert not second.failed
    assert second.selection_complete
    assert second.rendered_count == 11
    assert not second.manifest.complete
    assert [artifact.segment_number for artifact in second.manifest.speech] == [
        *range(1, 21),
        *range(50, 61),
    ]
    assert [f"{number:06d}.wav" for number in range(50, 61)] == [
        artifact.wav
        for artifact in second.manifest.speech
        if artifact.segment_number >= 50
    ]
    assert adapter.synthesized_texts == [
        f"speech {number}" for number in range(1, 21)
    ] + [f"speech {number}" for number in range(50, 61)]


def test_partial_output_resumes_to_full_render_and_reuses_without_engine_startup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = _install(monkeypatch)
    plan = _plan(*(f"speech {number}" for number in range(1, 6)))
    report = _report(adapter, plan, tmp_path)
    output = tmp_path / "chapter"
    cache = tmp_path / "cache"

    partial = render_renderplan(
        plan,
        preflight=report,
        output_dir=output,
        cache_dir=cache,
        speech_numbers=range(1, 3),
    )
    assert not partial.failed
    assert not partial.manifest.complete
    initial_synthesized = list(adapter.synthesized_texts)

    full = render_renderplan(plan, preflight=report, output_dir=output, cache_dir=cache)
    assert not full.failed
    assert full.selection_complete
    assert full.manifest.complete
    assert full.planned_count == full.available_count == 5
    assert full.reused_count == 2
    assert full.rendered_count == 3
    assert len(adapter.synthesized_texts) == len(initial_synthesized) + 3

    open_count = adapter.open_count
    reused = render_renderplan(
        plan, preflight=report, output_dir=output, cache_dir=cache
    )
    assert not reused.failed
    assert reused.manifest.complete
    assert reused.rendered_count == 0
    assert reused.reused_count == 5
    assert adapter.open_count == open_count


def test_corrupt_manifest_output_wav_is_dropped_and_regenerated_from_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = _install(monkeypatch)
    plan = _plan("first", "second")
    report = _report(adapter, plan, tmp_path)
    output = tmp_path / "chapter"
    cache = tmp_path / "cache"
    first = render_renderplan(
        plan, preflight=report, output_dir=output, cache_dir=cache
    )
    assert first.manifest.complete
    synthesized = list(adapter.synthesized_texts)
    (output / "000001.wav").write_bytes(b"corrupt output")

    resumed = render_renderplan(
        plan, preflight=report, output_dir=output, cache_dir=cache
    )
    assert not resumed.failed
    assert resumed.manifest.complete
    assert resumed.rendered_count == 0
    assert resumed.reused_count == 2
    assert adapter.synthesized_texts == synthesized
    loaded = load_manifest(resumed.manifest_path)
    loaded.validate_against(plan)
    assert wav_info(output / "000001.wav")["frames"] > 0


def test_stale_plan_requires_force_and_force_resets_only_owned_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = _install(monkeypatch)
    old_plan = _plan("old plan")
    old_report = _report(adapter, old_plan, tmp_path)
    output = tmp_path / "chapter"
    cache = tmp_path / "cache"
    old = render_renderplan(
        old_plan, preflight=old_report, output_dir=output, cache_dir=cache
    )
    (output / "preflight.toml").write_text("old preflight", encoding="utf-8")
    (output / "keep.txt").write_text("user data", encoding="utf-8")
    (output / "voice-note.wav").write_text("user data", encoding="utf-8")

    new_plan = _plan("new plan")
    new_report = _report(adapter, new_plan, tmp_path)
    with pytest.raises(ArtifactError) as stale:
        render_renderplan(
            new_plan, preflight=new_report, output_dir=output, cache_dir=cache
        )
    assert stale.value.code == "output.stale_plan"
    assert load_manifest(old.manifest_path).render_plan_id == old_plan.plan_id

    forced = render_renderplan(
        new_plan,
        preflight=new_report,
        output_dir=output,
        cache_dir=cache,
        force=True,
    )
    assert not forced.failed
    assert forced.manifest.render_plan_id == new_plan.plan_id
    assert (output / "keep.txt").read_text(encoding="utf-8") == "user data"
    assert (output / "voice-note.wav").read_text(encoding="utf-8") == "user data"
    assert not (output / "preflight.toml").exists()


def test_force_rerenders_only_selected_speech_and_keeps_unselected_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = _install(monkeypatch)
    plan = _plan("first", "second")
    report = _report(adapter, plan, tmp_path)
    output = tmp_path / "chapter"
    cache = tmp_path / "cache"
    render_renderplan(plan, preflight=report, output_dir=output, cache_dir=cache)
    second_bytes = (output / "000002.wav").read_bytes()
    before = list(adapter.synthesized_texts)

    forced = render_renderplan(
        plan,
        preflight=report,
        output_dir=output,
        cache_dir=cache,
        force=True,
        speech_numbers={1},
    )
    assert not forced.failed
    assert forced.manifest.complete
    assert forced.selected_count == 1
    assert forced.rendered_count == 1
    assert forced.reused_count == 0
    assert len(adapter.synthesized_texts) == len(before) + 1
    assert (output / "000002.wav").read_bytes() == second_bytes


@pytest.mark.parametrize("selection", [(), (0,), (-1,), (3,), (True,)])
def test_invalid_speech_selection_fails_before_output_or_synthesis(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    selection: tuple[int, ...],
) -> None:
    adapter = _install(monkeypatch)
    plan = _plan("first", "second")
    report = _report(adapter, plan, tmp_path)
    output = tmp_path / "invalid-selection"

    with pytest.raises(PlanError, match="speech_numbers"):
        render_renderplan(
            plan,
            preflight=report,
            output_dir=output,
            speech_numbers=selection,
        )
    assert adapter.synthesized_texts == []
    assert not output.exists()


def test_render_book_cli_integrates_real_fake_engine_for_cumulative_qc(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    adapter = _install(monkeypatch)
    profile = RenderProfile(
        roles={"narrator": RoleProfile(language="en-US", binding=dict(_BINDING))},
        default_role="narrator",
    ).with_identity()
    profile_id = profile.profile_id or profile.computed_id()
    plan = replace(
        _plan("speech one", "speech two", "speech three"),
        render_profile_id=profile_id,
        plan_id=None,
    ).with_identity()

    workspace = tmp_path / "integration-book"
    fingerprint = profile_id.removeprefix("sha256:")[:12]
    variant = f"narration--{fingerprint}"
    variant_dir = workspace / "renderplan" / variant
    variant_dir.mkdir(parents=True)
    save_profile(profile, variant_dir / "renderprofile.toml")
    plan_name = "chapter-0001.renderplan.toml"
    save_plan(plan, variant_dir / plan_name)
    chapter = RenderPlanSetChapter(
        id="chapter-0001",
        source_number=1,
        source="../../chapter-0001.md",
        utterplan="../../utterplan/chapter-0001.toml",
        source_plan_id=plan.source_plan_id,
        renderplan=plan_name,
        render_plan_id=plan.plan_id or plan.computed_id(),
    )
    renderplan_set = RenderPlanSet(
        variant="narration",
        profile_fingerprint=fingerprint,
        producer_version="test",
        manifest="../../book.json",
        utterplan_dir="../../utterplan",
        profile_id=profile_id,
        snapshot="renderprofile.toml",
        source_profile="../../narration.renderprofile.toml",
        chapters=(chapter,),
    )
    save_renderplan_set(variant_dir / "renderplan-set.toml", renderplan_set)

    from voicerender.cli import main

    output_root = tmp_path / "integration-output"
    cache_dir = tmp_path / "integration-cache"

    def invoke(*selection: str) -> dict[str, Any]:
        args = [
            "render-book",
            str(workspace),
            "--variant",
            variant,
            "--output-root",
            str(output_root),
            "--cache-dir",
            str(cache_dir),
            *selection,
            "--json",
        ]
        assert main(args) == 0
        return json.loads(capsys.readouterr().out)

    first = invoke("--chapters", "1", "--speech", "1")
    assert first["items"][0]["status"] == "partial"
    fingerprint = first["items"][0]["execution_fingerprint"][0]
    assert fingerprint["adapter_id"] == _ENGINE_ID
    assert fingerprint["capacity"]["metric"] == "model_tokens"
    book_report = tomllib.loads(
        (output_root / variant / "voicerender-report.toml").read_text(encoding="utf-8")
    )
    report_fingerprint = json.loads(
        book_report["item"][0]["execution_fingerprint_json"]
    )
    assert report_fingerprint[0]["fingerprint_id"] == fingerprint["fingerprint_id"]
    assert first["items"][0]["manifest_complete"] is False
    assert (output_root / variant / "chapter-0001" / "000001.wav").is_file()
    assert adapter.synthesized_texts == ["speech one"]

    second = invoke("--chapters", "1", "--speech", "3")
    assert second["items"][0]["status"] == "partial"
    assert second["items"][0]["available_speech"] == 2
    assert adapter.synthesized_texts == ["speech one", "speech three"]

    completed = invoke("--chapters", "1")
    assert completed["items"][0]["status"] == "complete"
    assert completed["items"][0]["manifest_complete"] is True
    assert completed["items"][0]["rendered"] == 1
    assert completed["items"][0]["reused"] == 2
    assert adapter.synthesized_texts == ["speech one", "speech three", "speech two"]
    manifest_path = output_root / variant / "chapter-0001" / "render-manifest.toml"
    manifest = load_manifest(manifest_path)
    assert manifest.complete
    assert [item.wav for item in manifest.speech] == [
        "000001.wav",
        "000002.wav",
        "000003.wav",
    ]
    open_count = adapter.open_count

    reused = invoke("--chapters", "1")
    assert reused["items"][0]["status"] == "reused"
    assert adapter.open_count == open_count
