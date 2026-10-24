from __future__ import annotations

import json
import shutil
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
    TokenBundle,
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
    EngineBackendError,
    PlanError,
    RenderEventCallbackError,
    UnsupportedSynthesisFeatureError,
)
from voicerender.events import RenderEvent, RenderEventKind
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
        self.adapter.measured_requests.append(request)
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
        if self.adapter.synthesis_error is not None:
            raise self.adapter.synthesis_error
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
        self.measured_requests: list[SpeechRequest] = []
        self.synthesized_texts: list[str] = []
        self.synthesized_requests: list[SpeechRequest] = []
        self.measurements: dict[str, RequestMeasure | Exception] = {}
        self.open_error: Exception | None = None
        self.synthesis_error: Exception | None = None
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


def _plan(*texts: str, tokens: TokenBundle | None = None) -> RenderPlan:
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
            tokens=tokens,
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


def test_preflight_and_render_share_optional_token_lowering(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = _install(monkeypatch)
    tokens = TokenBundle(
        span=((0, 6),),
        pos=("NOUN",),
        tag=("NN",),
    )
    plan = _plan("speech", tokens=tokens)
    original_plan_id = plan.plan_id

    report = preflight_render_plan(plan, tmp_path / "preflight.toml")

    assert report.status == "ok"
    assert adapter.measured_requests[0].tokens == ()
    assert plan.speech[0].tokens == tokens
    assert plan.plan_id == original_plan_id

    result = render_renderplan(
        plan,
        preflight=report,
        output_dir=tmp_path / "render",
        cache_dir=tmp_path / "cache",
    )

    assert not result.failed
    assert adapter.synthesized_requests[0].tokens == ()
    assert plan.speech[0].tokens == tokens
    assert plan.plan_id == original_plan_id


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

    with pytest.raises(UnsupportedSynthesisFeatureError, match="explicitly select"):
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
    adapter.measurements[" first "] = RequestMeasure(
        fits=True,
        amount=len(" first "),
        maximum=100,
        unit="model_tokens",
        source="fake.frontend_measure",
        details={"capacity_policy_revision": "budget.v1"},
    )
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
        "capacity_policy_revision": "budget.v1",
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


def test_immediate_unknown_fit_stops_before_artifact_and_keeps_prior_partial_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = _install(monkeypatch)
    plan = _plan("fits first", "unknown capacity next")
    report = _report(adapter, plan, tmp_path)
    adapter.measured_texts.clear()
    adapter.synthesized_texts.clear()
    adapter.measurements["unknown capacity next"] = RequestMeasure(
        fits=None,
        amount=80,
        maximum=None,
        unit="model_tokens",
        source="inflectsynth.measure_prepared",
    )
    recheck_path = tmp_path / "fresh-unknown-reject.toml"

    result = render_renderplan(
        plan,
        preflight=report,
        output_dir=tmp_path / "partial-unknown",
        cache_dir=tmp_path / "partial-unknown-cache",
        recheck_report_path=recheck_path,
    )

    assert result.failed
    assert result.error_code == "preflight.reject"
    assert adapter.measured_texts == ["fits first", "unknown capacity next"]
    assert adapter.synthesized_texts == ["fits first"]
    assert result.manifest.complete is False
    assert [artifact.segment_id for artifact in result.manifest.speech] == [
        plan.speech[0].segment_id
    ]
    assert result.fresh_preflight is not None
    assert result.fresh_preflight.status == "reject"
    assert result.fresh_preflight.replan == ()
    check = result.fresh_preflight.checks[0]
    assert check.status == "error"
    assert check.metric == "model_tokens"
    assert check.measured == 80
    assert check.maximum is None
    assert check.source == "inflectsynth.measure_prepared"
    assert "authoritative maximum/fit result" in check.detail
    assert load_preflight(recheck_path).status == "reject"
    assert (tmp_path / "partial-unknown" / "000001.wav").is_file()
    assert not (tmp_path / "partial-unknown" / "000002.wav").exists()


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


def test_partial_output_resumes_after_dead_lock_and_reuses_without_engine_startup(
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

    from voicerender import render_lock as render_lock_module

    committed_wavs = {
        artifact.wav: (output / artifact.wav).read_bytes()
        for artifact in partial.manifest.speech
    }
    lock_path = output / ".voicerender.lock"
    lock_path.write_text("999999", encoding="ascii")
    monkeypatch.setattr(
        render_lock_module, "_owner_is_definitely_dead", lambda _owner: True
    )

    full = render_renderplan(plan, preflight=report, output_dir=output, cache_dir=cache)
    assert not full.failed
    assert full.selection_complete
    assert full.manifest.complete
    assert full.planned_count == full.available_count == 5
    assert full.reused_count == 2
    assert full.rendered_count == 3
    assert not lock_path.exists()
    assert {
        name: (output / name).read_bytes() for name in committed_wavs
    } == committed_wavs
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


def test_batch_variant_resumes_partial_output_after_dead_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from types import SimpleNamespace

    import voicerender.bundle_batch as batch_module
    from voicerender.bundle import BundleRenderResult
    from voicerender.variants import BundleVariant, BundleVariantsResult

    adapter = _install(monkeypatch)
    plan = _plan("batch one", "batch two", "batch three")
    report = _report(adapter, plan, tmp_path)
    output = tmp_path / "batch-chapter"
    cache = tmp_path / "batch-cache"
    partial = render_renderplan(
        plan,
        preflight=report,
        output_dir=output,
        cache_dir=cache,
        speech_numbers={1},
    )
    assert not partial.manifest.complete
    assert partial.available_count == 1
    committed_wav = (output / "000001.wav").read_bytes()

    from voicerender import render_lock as render_lock_module

    lock_path = output / ".voicerender.lock"
    lock_path.write_text("999999", encoding="ascii")
    monkeypatch.setattr(
        render_lock_module, "_owner_is_definitely_dead", lambda _owner: True
    )

    bundle = tmp_path / "batch-story.ssmd-bundle"
    variant_name = "narration--test"

    def listing(state: str, available: int) -> BundleVariantsResult:
        return BundleVariantsResult(
            bundle=bundle,
            kind="book",
            renderplan_root=bundle / "renderplan",
            output_root=bundle / "voicerender",
            variants=(
                BundleVariant(
                    name=variant_name,
                    profile_id="sha256:profile",
                    plan_status="valid",
                    validation_error=None,
                    render_status=state,
                    state=state,
                    unit_kind="chapters",
                    completed_units=0,
                    total_units=1,
                    planned_speech=3,
                    available_speech=available,
                    pending_composition=None,
                    report_path=None,
                ),
            ),
        )

    listings = iter((listing("partial", 1), listing("complete", 3)))
    monkeypatch.setattr(
        batch_module, "list_bundle_variants", lambda *_args, **_kwargs: next(listings)
    )
    render_results = []

    def render_variant(*_args: Any, **_kwargs: Any) -> BundleRenderResult:
        result = render_renderplan(
            plan, preflight=report, output_dir=output, cache_dir=cache
        )
        render_results.append(result)
        return BundleRenderResult(
            kind="book",
            value=SimpleNamespace(exit_code=0, error_code=None, error_message=None),
        )

    monkeypatch.setattr(batch_module, "render_bundle", render_variant)
    batch_result = batch_module.render_bundle_batch(bundle)

    assert batch_result.exit_code == 0
    assert batch_result.summary.attempted == batch_result.summary.completed == 1
    assert batch_result.variants[0].final_state == "complete"
    assert render_results[0].manifest.complete
    assert render_results[0].available_count == 3
    assert render_results[0].reused_count == 1
    assert render_results[0].rendered_count == 2
    assert (output / "000001.wav").read_bytes() == committed_wav
    assert not lock_path.exists()


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


def test_reset_stale_plan_reuses_exact_cache_and_preserves_unowned_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = _install(monkeypatch)
    output = tmp_path / "chapter"
    cache = tmp_path / "shared-cache"
    parent_plan = _plan("parent text to split", "stable exact request")
    parent_report = preflight_render_plan(parent_plan, output / "preflight.toml")
    parent = render_renderplan(
        parent_plan,
        preflight=parent_report,
        output_dir=output,
        cache_dir=cache,
    )
    assert not parent.failed
    cache_before = {path.name for path in cache.iterdir()}
    (output / "keep.txt").write_text("user data", encoding="utf-8")

    child_plan = _plan(
        "split child request one",
        "split child request two",
        "stable exact request",
    )
    child_report = preflight_render_plan(child_plan, tmp_path / "child-preflight.toml")
    child = render_renderplan(
        child_plan,
        preflight=child_report,
        output_dir=output,
        cache_dir=cache,
        reset_stale_output=True,
    )

    assert not child.failed
    assert child.manifest.render_plan_id == child_plan.plan_id
    assert child.rendered_count == 2
    assert child.reused_count == 1
    assert adapter.synthesized_texts.count("stable exact request") == 1
    assert "split child request one" in adapter.synthesized_texts
    assert "split child request two" in adapter.synthesized_texts
    assert load_manifest(child.manifest_path).render_plan_id == child_plan.plan_id
    saved_preflight = load_preflight(output / "preflight.toml")
    assert saved_preflight.render_plan_id == child_plan.plan_id
    assert (output / "keep.txt").read_text(encoding="utf-8") == "user data"
    assert cache_before.issubset({path.name for path in cache.iterdir()})
    assert all((output / f"{number:06d}.wav").is_file() for number in range(1, 4))


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

    workspace = tmp_path / "integration-book.ssmd-bundle"
    workspace.mkdir()
    (workspace / "manifest.json").write_text(
        json.dumps({"format": "ssmdconvert.book", "schema_version": 1}),
        encoding="utf-8",
    )
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
            "bundle",
            "render",
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
        payload = json.loads(capsys.readouterr().out)
        assert payload["format"] == "voicerender-bundle-result"
        assert payload["kind"] == "book"
        return payload["result"]

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


def test_render_book_replans_then_activates_child_with_cache_reuse(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    adapter = _install(monkeypatch)
    from voicerender.book import render_book
    from voicerender.book_contract import load_renderplan_set
    from voicerender.cli import main

    profile = RenderProfile(
        roles={"narrator": RoleProfile(language="en-US", binding=dict(_BINDING))},
        default_role="narrator",
    ).with_identity()
    profile_id = profile.profile_id or profile.computed_id()
    parent_plan = replace(
        _plan("stable exact request", "parent text to split"),
        render_profile_id=profile_id,
        plan_id=None,
    ).with_identity()

    workspace = tmp_path / "replan-book.ssmd-bundle"
    workspace.mkdir()
    (workspace / "manifest.json").write_text(
        json.dumps({"format": "ssmdconvert.book", "schema_version": 1}),
        encoding="utf-8",
    )
    fingerprint = profile_id.removeprefix("sha256:")[:12]
    variant = f"narration--{fingerprint}"
    variant_dir = workspace / "renderplan" / variant
    variant_dir.mkdir(parents=True)
    save_profile(profile, variant_dir / "renderprofile.toml")
    plan_name = "chapter-0001.renderplan.toml"
    plan_path = variant_dir / plan_name
    save_plan(parent_plan, plan_path)
    chapter = RenderPlanSetChapter(
        id="chapter-0001",
        source_number=1,
        source="../../chapter-0001.md",
        utterplan="../../utterplan/chapter-0001.toml",
        source_plan_id=parent_plan.source_plan_id,
        renderplan=plan_name,
        render_plan_id=parent_plan.plan_id or parent_plan.computed_id(),
    )
    renderplan_set_path = variant_dir / "renderplan-set.toml"
    save_renderplan_set(
        renderplan_set_path,
        RenderPlanSet(
            variant="narration",
            profile_fingerprint=fingerprint,
            producer_version="test",
            manifest="../../book.json",
            utterplan_dir="../../utterplan",
            profile_id=profile_id,
            snapshot="renderprofile.toml",
            source_profile="../../narration.renderprofile.toml",
            chapters=(chapter,),
        ),
    )

    output_root = tmp_path / "output"
    cache_dir = tmp_path / "cache"
    baseline = render_book(
        workspace,
        variant=variant,
        speech="1",
        output_root=output_root,
        cache_dir=cache_dir,
    )
    assert baseline.outcomes[0].status == "partial"
    assert adapter.synthesized_texts == ["stable exact request"]
    cache_before = {path.name for path in cache_dir.iterdir()}
    chapter_output = baseline.outcomes[0].output_dir
    (chapter_output / "keep.txt").write_text("user data", encoding="utf-8")
    (chapter_output / "preflight.toml").unlink()
    adapter.measurements["parent text to split"] = RequestMeasure(
        fits=False,
        amount=101,
        maximum=100,
        unit="model_tokens",
        source="fake.frontend_measure",
    )

    blocked = render_book(
        workspace,
        variant=variant,
        output_root=output_root,
        cache_dir=cache_dir,
    )
    assert blocked.outcomes[0].error_code == "preflight.replan"
    assert blocked.outcomes[0].replan_required is True
    assert blocked.outcomes[0].preflight_report == chapter_output / "preflight.toml"
    assert load_preflight(chapter_output / "preflight.toml").status == "replan"
    assert (
        load_manifest(chapter_output / "render-manifest.toml").render_plan_id
        == parent_plan.plan_id
    )
    assert adapter.synthesized_texts == ["stable exact request"]

    split_source = parent_plan.speech[1]
    child_parts = tuple(
        replace(
            split_source,
            number=number,
            text=text,
            source=(replace(split_source.source[0], end=len(text)),),
            segment_id=None,
        )
        for number, text in (
            (1, "split child part one"),
            (2, "split child part two"),
        )
    )
    stable_request = replace(parent_plan.speech[0], number=3)
    child_plan = replace(
        parent_plan,
        flow=(*child_parts, stable_request),
        plan_id=None,
    ).with_identity()
    save_plan(child_plan, plan_path)
    active_set = load_renderplan_set(renderplan_set_path)
    save_renderplan_set(
        renderplan_set_path,
        replace(
            active_set,
            chapters=(
                replace(
                    active_set.chapters[0],
                    source_plan_id=child_plan.source_plan_id,
                    render_plan_id=child_plan.plan_id or child_plan.computed_id(),
                ),
            ),
        ),
    )

    exit_code = main(
        [
            "bundle",
            "render",
            str(workspace),
            "--variant",
            variant,
            "--output-root",
            str(output_root),
            "--cache-dir",
            str(cache_dir),
            "--reset-stale",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["format"] == "voicerender-bundle-result"
    assert payload["kind"] == "book"
    payload = payload["result"]
    item = payload["items"][0]
    assert exit_code == 0
    assert item["status"] == "complete"
    assert item["rendered"] == 2
    assert item["reused"] == 1
    assert item["replan_required"] is False
    assert (
        load_manifest(chapter_output / "render-manifest.toml").render_plan_id
        == child_plan.plan_id
    )
    child_preflight = load_preflight(chapter_output / "preflight.toml")
    assert child_preflight.render_plan_id == child_plan.plan_id
    assert child_preflight.status == "ok"
    assert (chapter_output / "keep.txt").read_text(encoding="utf-8") == "user data"
    assert cache_before.issubset({path.name for path in cache_dir.iterdir()})
    assert adapter.synthesized_texts == [
        "stable exact request",
        "split child part one",
        "split child part two",
    ]


def test_render_events_report_ordered_progress_and_final_timing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = _install(monkeypatch)
    plan = _plan("one", "two")
    report = _report(adapter, plan, tmp_path)
    events: list[RenderEvent] = []

    result = render_renderplan(
        plan,
        preflight=report,
        output_dir=tmp_path / "event-render",
        cache_dir=tmp_path / "event-cache",
        on_event=events.append,
    )

    kinds = [event.kind for event in events]
    assert kinds[0] is RenderEventKind.RENDER_STARTED
    assert kinds[-1] is RenderEventKind.RENDER_COMPLETED
    assert kinds.count(RenderEventKind.SEGMENT_STARTED) == 2
    assert kinds.count(RenderEventKind.SEGMENT_RENDERED) == 2
    assert kinds.count(RenderEventKind.ENGINE_OPEN_STARTED) == 1
    assert kinds.count(RenderEventKind.ENGINE_OPEN_FINISHED) == 1
    terminal = [
        event
        for event in events
        if event.kind
        in {
            RenderEventKind.SEGMENT_RENDERED,
            RenderEventKind.SEGMENT_REUSED,
            RenderEventKind.SEGMENT_FAILED,
        }
    ]
    assert [event.completed_count for event in terminal] == [1, 2]
    assert [event.progress for event in terminal] == [0.5, 1.0]
    assert all(event.total_count == 2 for event in terminal)
    assert all(
        event.elapsed_seconds is not None and event.elapsed_seconds >= 0
        for event in events
    )
    engine_opened = next(
        event for event in events if event.kind is RenderEventKind.ENGINE_OPEN_FINISHED
    )
    assert engine_opened.phase_elapsed_seconds is not None
    assert result.elapsed_seconds >= 0
    assert events[-1].eta_seconds == 0.0


def test_existing_output_reuse_advances_progress_without_opening_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = _install(monkeypatch)
    plan = _plan("reuse me")
    report = _report(adapter, plan, tmp_path)
    output = tmp_path / "existing-output"
    cache = tmp_path / "existing-cache"
    render_renderplan(plan, preflight=report, output_dir=output, cache_dir=cache)
    adapter.open_count = 0
    events: list[RenderEvent] = []

    result = render_renderplan(
        plan,
        preflight=report,
        output_dir=output,
        cache_dir=cache,
        on_event=events.append,
    )

    assert result.reused_count == 1
    assert adapter.open_count == 0
    assert [event.kind for event in events] == [
        RenderEventKind.RENDER_STARTED,
        RenderEventKind.SEGMENT_REUSED,
        RenderEventKind.RENDER_COMPLETED,
    ]
    assert events[1].completed_count == 1
    assert events[1].progress == 1.0

    diagnostics = json.loads(
        (output / RENDER_DIAGNOSTICS_FILENAME).read_text(encoding="utf-8")
    )
    assert diagnostics["session_residency"]["opens"] == 0
    assert diagnostics["session_residency"]["peak_open_sessions"] == 0


def test_cache_reuse_is_reported_after_materialization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = _install(monkeypatch)
    plan = _plan("cache me")
    report = _report(adapter, plan, tmp_path)
    output = tmp_path / "cached-output"
    cache = tmp_path / "shared-cache"
    render_renderplan(plan, preflight=report, output_dir=output, cache_dir=cache)
    shutil.rmtree(output)
    adapter.open_count = 0
    events: list[RenderEvent] = []

    result = render_renderplan(
        plan,
        preflight=report,
        output_dir=output,
        cache_dir=cache,
        on_event=events.append,
    )

    reused_events = [
        event for event in events if event.kind is RenderEventKind.SEGMENT_REUSED
    ]
    assert result.reused_count == 1
    assert adapter.open_count == 0
    assert len(reused_events) == 1
    assert reused_events[0].status == "reused"
    assert (output / plan.speech[0].wav).is_file()


def test_partial_selection_uses_selected_denominator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = _install(monkeypatch)
    plan = _plan("one", "two", "three", "four")
    report = _report(adapter, plan, tmp_path)
    events: list[RenderEvent] = []

    result = render_renderplan(
        plan,
        preflight=report,
        output_dir=tmp_path / "partial-output",
        cache_dir=tmp_path / "partial-cache",
        speech_numbers={2, 4},
        on_event=events.append,
    )

    assert not result.failed
    assert all(event.total_count == 2 for event in events)
    assert [
        event.progress
        for event in events
        if event.kind is RenderEventKind.SEGMENT_RENDERED
    ] == [0.5, 1.0]


def test_callback_failure_propagates_after_commit_and_output_is_resumable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = _install(monkeypatch)
    plan = _plan("committed before callback")
    report = _report(adapter, plan, tmp_path)
    output = tmp_path / "callback-output"
    cache = tmp_path / "callback-cache"

    def fail_after_commit(event: RenderEvent) -> None:
        if event.kind is RenderEventKind.SEGMENT_RENDERED:
            raise RuntimeError("consumer stopped")

    with pytest.raises(RenderEventCallbackError):
        render_renderplan(
            plan,
            preflight=report,
            output_dir=output,
            cache_dir=cache,
            on_event=fail_after_commit,
        )

    adapter.open_count = 0
    resumed_events: list[RenderEvent] = []
    resumed = render_renderplan(
        plan,
        preflight=report,
        output_dir=output,
        cache_dir=cache,
        on_event=resumed_events.append,
    )
    assert not resumed.failed
    assert resumed.reused_count == 1
    assert adapter.open_count == 0
    assert any(event.kind is RenderEventKind.SEGMENT_REUSED for event in resumed_events)


def test_open_failure_emits_typed_terminal_progress(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = _install(monkeypatch)
    plan = _plan("will fail")
    report = _report(adapter, plan, tmp_path)
    adapter.open_error = RuntimeError("engine startup failed")
    events: list[RenderEvent] = []

    result = render_renderplan(
        plan,
        preflight=report,
        output_dir=tmp_path / "failed-output",
        cache_dir=tmp_path / "failed-cache",
        on_event=events.append,
    )

    assert result.failed
    assert result.error_type == "RuntimeError"
    assert result.error_diagnostic["message"] == "engine startup failed"
    assert (
        result.diagnostics_path
        == tmp_path / "failed-output" / RENDER_DIAGNOSTICS_FILENAME
    )
    assert result.diagnostics_path.is_file()
    failed = [event for event in events if event.kind is RenderEventKind.SEGMENT_FAILED]
    assert len(failed) == 1
    assert failed[0].completed_count == 1
    assert failed[0].progress == 1.0
    assert failed[0].error_type == "RuntimeError"
    assert failed[0].render_diagnostics == str(result.diagnostics_path)
    assert events[-1].kind is RenderEventKind.RENDER_COMPLETED
    assert events[-1].status == "failed"
    assert events[-1].render_diagnostics == str(result.diagnostics_path)


def test_estimate_renderplan_reports_verified_reuse_without_side_effects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from voicerender import estimate_renderplan

    adapter = _install(monkeypatch)
    plan = _plan("estimate one", "estimate two")
    output_dir = tmp_path / "render-output"

    initial = estimate_renderplan(plan, output_dir=output_dir)
    assert initial.selected_count == 2
    assert initial.known_output_reuse_count == 0
    assert initial.pending_count == 2
    assert initial.estimated_seconds is None
    assert initial.confidence == "unavailable"
    assert initial.source == "workload-only"
    assert not output_dir.exists()
    assert adapter.open_count == 0 and adapter.synthesized_texts == []

    report = _report(adapter, plan, tmp_path)
    render_renderplan(
        plan,
        preflight=report,
        output_dir=output_dir,
        cache_dir=tmp_path / "cache",
    )
    before_manifest = (output_dir / "render-manifest.toml").read_bytes()
    before_open_count = adapter.open_count
    before_synthesized = list(adapter.synthesized_texts)

    complete = estimate_renderplan(plan, output_dir=output_dir)
    partial = estimate_renderplan(plan, output_dir=output_dir, speech_numbers=(1,))
    forced = estimate_renderplan(plan, output_dir=output_dir, force=True)

    assert (complete.selected_count, complete.known_output_reuse_count) == (2, 2)
    assert (complete.pending_count, complete.estimated_seconds) == (0, None)
    assert (partial.selected_count, partial.known_output_reuse_count) == (1, 1)
    assert (forced.known_output_reuse_count, forced.pending_count) == (0, 2)
    assert adapter.open_count == before_open_count
    assert adapter.synthesized_texts == before_synthesized
    assert (output_dir / "render-manifest.toml").read_bytes() == before_manifest


def test_render_failure_diagnostics_identify_segment_and_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = _install(monkeypatch)
    root_cause_error = RuntimeError("provider execution failed")
    engine_error = ValueError("engine adapter failed")
    engine_error.__cause__ = root_cause_error
    adapter.synthesis_error = EngineBackendError(
        "backend failed", engine=adapter.id, request_id="request-1"
    )
    adapter.synthesis_error.__cause__ = engine_error
    plan = _plan("fails during synthesis")
    report = _report(adapter, plan, tmp_path)
    output_dir = tmp_path / "render-output"

    result = render_renderplan(
        plan,
        preflight=report,
        output_dir=output_dir,
        cache_dir=tmp_path / "cache",
    )

    expected_segment_id = plan.speech[0].segment_id or plan.speech[0].computed_id()
    assert result.failed
    assert result.error_segment_number == 1
    assert result.error_segment_id == expected_segment_id
    assert result.error_engine == adapter.id
    assert result.error_type == "EngineBackendError"
    assert result.error_diagnostic["cause"]["cause"]["type"] == "RuntimeError"
    assert (
        result.error_diagnostic["cause"]["cause"]["message"]
        == "provider execution failed"
    )
    assert result.diagnostics_path == output_dir / RENDER_DIAGNOSTICS_FILENAME
    assert result.diagnostics_path.is_file()
    diagnostics = json.loads(
        (output_dir / RENDER_DIAGNOSTICS_FILENAME).read_text(encoding="utf-8")
    )
    assert diagnostics["error_segment_number"] == 1
    assert diagnostics["schema"] == "voicerender.render-diagnostics.v1"
    assert diagnostics["error_segment_id"] == expected_segment_id
    assert diagnostics["error_engine"] == adapter.id
    assert diagnostics["error_code"] == "synthesis.backend_error"
    assert diagnostics["error_type"] == "EngineBackendError"
    assert (
        diagnostics["error_diagnostic"]["cause"]["cause"]["message"]
        == "provider execution failed"
    )
    json.dumps(diagnostics)


def test_diagnostic_serializer_failure_does_not_mask_render_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import voicerender.renderplan_render as render_module

    adapter = _install(monkeypatch)
    adapter.synthesis_error = EngineBackendError("original backend failure")
    plan = _plan("serializer failure")
    report = _report(adapter, plan, tmp_path)
    monkeypatch.setattr(
        render_module,
        "exception_diagnostic",
        lambda _error: (_ for _ in ()).throw(RuntimeError("serializer failed")),
    )

    result = render_renderplan(
        plan,
        preflight=report,
        output_dir=tmp_path / "serializer-failed-render",
        cache_dir=tmp_path / "serializer-failed-cache",
    )

    assert result.error == "original backend failure"
    assert result.error_code == "synthesis.backend_error"
    assert result.error_diagnostic["message"].startswith(
        "[diagnostic serialization failed;"
    )


def test_default_session_pool_bounds_clones_and_preserves_plan_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from voicerender import session_pool as pool_module

    adapter = _install(monkeypatch)
    voices = ("clone-a", "clone-b", "clone-a", "clone-c", "clone-b", "clone-d")
    plan = _plan(*(f"speech {index}" for index in range(1, len(voices) + 1)))
    plan = replace(
        plan,
        flow=tuple(
            replace(item, binding={**item.binding, "voice": voice}, segment_id=None)
            for item, voice in zip(plan.speech, voices, strict=True)
        ),
        plan_id=None,
    ).with_identity()
    report = _report(adapter, plan, tmp_path)
    active: set[str | None] = set()
    open_close_order: list[tuple[str, str | None]] = []
    peak_open = 0

    @contextmanager
    def open_tracked(selection: EngineSelection) -> Iterator[EngineSession]:
        nonlocal peak_open
        assert not active
        active.add(selection.voice)
        peak_open = max(peak_open, len(active))
        open_close_order.append(("open", selection.voice))
        try:
            yield adapter.session
        finally:
            active.remove(selection.voice)
            open_close_order.append(("close", selection.voice))

    monkeypatch.setattr(pool_module, "open_session", open_tracked)
    events: list[RenderEvent] = []
    output = tmp_path / "bounded-sessions"
    result = render_renderplan(
        plan,
        preflight=report,
        output_dir=output,
        cache_dir=tmp_path / "bounded-cache",
        on_event=events.append,
    )

    assert peak_open == 1
    assert active == set()
    assert adapter.synthesized_texts == [item.text for item in plan.speech]
    assert [artifact.segment_number for artifact in result.manifest.speech] == list(
        range(1, len(voices) + 1)
    )
    assert open_close_order == [
        pair for voice in voices for pair in (("open", voice), ("close", voice))
    ]
    diagnostics = json.loads(
        (output / RENDER_DIAGNOSTICS_FILENAME).read_text(encoding="utf-8")
    )
    assert diagnostics["session_residency"] == {
        "max_open_sessions": 1,
        "peak_open_sessions": 1,
        "opens": len(voices),
        "closes": len(voices),
        "evictions": len(voices) - 1,
        "hits": 0,
    }
    close_events = [
        event for event in events if event.kind is RenderEventKind.ENGINE_CLOSE_STARTED
    ]
    assert [event.status for event in close_events] == [
        *("evicted" for _ in range(len(voices) - 1)),
        "render_complete",
    ]
