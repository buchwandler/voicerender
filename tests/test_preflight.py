from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from renderplan import (
    CapacityConstraint,
    RenderPlan,
    SourceSpan,
    SpeechItem,
    load_preflight,
    save_plan,
)

from voicerender.cli import main
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
from voicerender.errors import EngineUnavailableError, SpeechRequestTooLongError
from voicerender.preflight import (
    preflight_exit_code,
    preflight_render_plan,
    preflight_summary,
)

_ENGINE_ID = "preflight-test-engine"
_TARGET_ID = "preflight-target-v1"
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
        raise AssertionError("preflight must never synthesize")


class _Adapter:
    id = _ENGINE_ID
    package_name = _ENGINE_ID

    def __init__(self) -> None:
        self.open_count = 0
        self.close_count = 0
        self.measured_texts: list[str] = []
        self.synthesized_texts: list[str] = []
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
        if self.open_error is not None:
            raise self.open_error
        try:
            yield self.session
        finally:
            self.close_count += 1


def _install(monkeypatch: pytest.MonkeyPatch) -> _Adapter:
    adapter = _Adapter()
    monkeypatch.setitem(_ADAPTERS, adapter.id, adapter)
    return adapter


def _plan(
    *texts: tuple[str, bool, str],
    binding: dict[str, Any] | None = None,
    source_count: int = 1,
) -> RenderPlan:
    flow: list[SpeechItem] = []
    for number, (text, required, level) in enumerate(texts, start=1):
        source = tuple(
            SourceSpan(
                locator=f"f{number:06d}.s{index:03d}",
                start=0,
                end=len(text),
            )
            for index in range(source_count)
        )
        flow.append(
            SpeechItem(
                number=number,
                text=text,
                language="en-US",
                role="narrator",
                binding=dict(binding or _BINDING),
                source=source,
                planning_level=level,
                preflight_required=required,
            )
        )
    return RenderPlan(
        source_plan_id="sha256:" + "1" * 64,
        render_profile_id="sha256:" + "2" * 64,
        flow=tuple(flow),
    ).with_identity()


def test_preflight_persists_all_fit_report_reuses_session_and_never_synthesizes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = _install(monkeypatch)
    plan = _plan(
        (" first ", True, "sentence"),
        ("second", True, "clause"),
        ("skip", False, "sentence"),
    )
    output = tmp_path / "reports" / "chapter.preflight.toml"

    report = preflight_render_plan(plan, output)

    assert report.status == "ok"
    assert preflight_exit_code(report) == 0
    assert [check.status for check in report.checks] == ["ok", "ok"]
    assert adapter.measured_texts == [" first ", "second"]
    assert adapter.open_count == 1
    assert adapter.close_count == 1
    assert adapter.synthesized_texts == []
    assert output.is_file()
    loaded = load_preflight(output)
    loaded.validate_against(plan)
    assert loaded.report_id == report.report_id
    summary = preflight_summary(report, output)
    assert summary["exit_code"] == 0
    assert summary["checks"] == {"ok": 2, "too_long": 0, "error": 0}


def test_too_long_preflight_preserves_source_and_planning_level(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = _install(monkeypatch)
    plan = _plan(("too long", True, "clause"), source_count=2)
    adapter.measurements["too long"] = RequestMeasure(
        fits=False,
        amount=101,
        maximum=100,
        unit="model_tokens",
        source="fake.frontend_measure",
    )
    output = tmp_path / "replan.toml"

    report = preflight_render_plan(plan, output)

    assert report.status == "replan"
    assert preflight_exit_code(report) == 2
    assert len(report.replan) == 1
    item = plan.speech[0]
    replan = report.replan[0]
    assert replan.source == item.source
    assert replan.current_level == item.planning_level == "clause"
    assert replan.metric == "model_tokens"
    assert replan.measured == 101
    assert replan.maximum == 100
    assert output.is_file()
    assert adapter.synthesized_texts == []


def test_measurement_without_exact_fit_is_reject_not_replan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = _install(monkeypatch)
    plan = _plan(("unknown capacity", True, "sentence"))
    adapter.measurements["unknown capacity"] = RequestMeasure(
        fits=None,
        amount=None,
        maximum=None,
        unit="unknown",
        source="fake.no_measurement",
    )

    report = preflight_render_plan(plan, tmp_path / "reject.toml")

    assert report.status == "reject"
    assert preflight_exit_code(report) == 1
    assert report.checks[0].status == "error"
    assert report.checks[0].source == "fake.no_measurement"
    assert report.replan == ()
    assert adapter.synthesized_texts == []


def test_invalid_binding_is_a_reported_reject_for_each_occurrence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = _install(monkeypatch)
    binding = {**_BINDING, "target_id": "unknown"}
    plan = _plan(("one", True, "sentence"), ("two", True, "sentence"), binding=binding)

    report = preflight_render_plan(plan, tmp_path / "binding-reject.toml")

    assert report.status == "reject"
    assert preflight_exit_code(report) == 1
    assert [check.status for check in report.checks] == ["error", "error"]
    assert len({check.segment_number for check in report.checks}) == 2
    assert adapter.measured_texts == []
    assert adapter.open_count == 0


def test_engine_unavailable_measurement_error_is_reject(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = _install(monkeypatch)
    plan = _plan(("throws capacity", True, "sentence"))
    adapter.measurements["throws capacity"] = EngineUnavailableError(
        "too long",
        amount=101,
        maximum=100,
        unit="model_tokens",
        source="fake.frontend_measure",
    )

    report = preflight_render_plan(plan, tmp_path / "typed-too-long.toml")

    assert report.status == "reject"
    assert report.checks[0].status == "error"
    assert report.replan == ()
    assert adapter.synthesized_texts == []


def test_typed_too_long_measurement_error_creates_replan_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = _install(monkeypatch)
    plan = _plan(("throws capacity", True, "sentence"))
    adapter.measurements["throws capacity"] = SpeechRequestTooLongError(
        "request exceeds model capacity",
        amount=101,
        maximum=100,
        unit="model_tokens",
        source="fake.frontend_measure",
    )

    report = preflight_render_plan(plan, tmp_path / "typed-too-long.toml")

    assert report.status == "replan"
    assert report.checks[0].status == "too_long"
    assert report.replan[0].measured == 101
    assert report.replan[0].maximum == 100
    assert adapter.synthesized_texts == []


def test_preflight_cli_writes_report_and_emits_json_exit_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    adapter = _install(monkeypatch)
    plan = _plan(("too long", True, "sentence"))
    adapter.measurements["too long"] = RequestMeasure(
        fits=False,
        amount=101,
        maximum=100,
        unit="model_tokens",
        source="fake.frontend_measure",
    )
    plan_path = tmp_path / "chapter.renderplan.toml"
    report_path = tmp_path / "chapter.preflight.toml"
    save_plan(plan, plan_path)

    exit_code = main(
        ["preflight", str(plan_path), "--output", str(report_path), "--json"]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["status"] == "replan"
    assert payload["exit_code"] == 2
    assert payload["checks"] == {"ok": 0, "too_long": 1, "error": 0}
    assert payload["output"] == str(report_path)
    assert report_path.is_file()
    assert adapter.synthesized_texts == []


def test_preflight_cli_refuses_to_overwrite_input_plan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _install(monkeypatch)
    plan_path = tmp_path / "input.renderplan.toml"
    save_plan(_plan(("keep me", True, "sentence")), plan_path)
    original = plan_path.read_bytes()

    exit_code = main(
        ["preflight", str(plan_path), "--output", str(plan_path), "--json"]
    )

    assert exit_code == 1
    assert "must not overwrite" in capsys.readouterr().err
    assert plan_path.read_bytes() == original
