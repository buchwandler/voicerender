"""Installed-wheel end-to-end smoke with an in-process fake renderer."""

from __future__ import annotations

import shutil
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import resources
from pathlib import Path

from renderplan import RenderPlan, SourceSpan, SpeechItem, load_manifest

import voicerender
from voicerender import (
    BookRenderError,
    BookRenderRequest,
    BookRenderResult,
    ChapterRenderOutcome,
    render_book,
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
from voicerender.engines.registry import list_engines, register_adapter
from voicerender.engines.selection import EngineRequest
from voicerender.preflight import preflight_render_plan
from voicerender.renderplan_render import render_renderplan
from voicerender.storage import verified_wav

ENGINE_ID = "voicerender-wheel-smoke"
TARGET_ID = "wheel-smoke-model-v1"


class _Session:
    def measure(self, request: SpeechRequest) -> RequestMeasure:
        return RequestMeasure(
            fits=True,
            amount=len(request.text),
            maximum=200,
            unit="model_tokens",
            source="wheel_smoke.fake_measurement",
        )

    def synthesize(self, request: SpeechRequest) -> RenderedSpeech:
        return RenderedSpeech(
            id=request.id,
            audio=[0.0, 0.2, -0.2, 0.1],
            sample_rate=24000,
        )


class _Adapter:
    id = ENGINE_ID
    package_name = "wheel-smoke-fake"

    def __init__(self) -> None:
        self.session = _Session()

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

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(id=self.id)

    def discover(self, **_: object) -> tuple[object, ...]:
        return ()

    def resolve(
        self, request: EngineRequest
    ) -> tuple[EngineSelection, tuple[EngineDiagnostic, ...]]:
        return (
            EngineSelection(
                engine=self.id,
                target_id=request.target_id or TARGET_ID,
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
        if selection.target_id != TARGET_ID:
            return (
                EngineDiagnostic(
                    code="wheel_smoke.target_invalid",
                    severity="error",
                    message="unexpected fake target",
                ),
            )
        return ()

    def version(self) -> str:
        return "1.0"

    def canonical_synthesis_identity(
        self, selection: EngineSelection
    ) -> dict[str, object]:
        return {
            "target_id": selection.target_id,
            "voice": selection.voice,
            "revision": "wheel-smoke-r1",
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
            frontend="wheel-smoke",
            pronunciation="wheel-smoke",
        )

    @contextmanager
    def open(self, selection: EngineSelection) -> Iterator[EngineSession]:
        del selection
        yield self.session


def main() -> None:
    checkout = Path(__file__).resolve().parents[1]
    installed_package = Path(voicerender.__file__).resolve()
    assert not installed_package.is_relative_to(checkout), installed_package
    assert resources.files("voicerender").joinpath("py.typed").is_file()
    expected_engine_ids = {
        "inflect",
        "kitten",
        "kokoro",
        "piper",
        "pocket",
        "sopro",
        "supertonic",
    }
    assert {item["id"] for item in list_engines()} == expected_engine_ids
    engine_sdks = {
        "inflectsynth",
        "kittensynth",
        "pykokoro",
        "pipersynth",
        "pocketsynth",
        "soprosynth",
        "supertonicsynth",
    }
    loaded_engine_sdks = {name.split(".", 1)[0] for name in sys.modules} & engine_sdks
    assert not loaded_engine_sdks, loaded_engine_sdks
    register_adapter(_Adapter())

    text = "Installed wheel executes the shared RenderPlan contract."
    item = SpeechItem(
        number=1,
        text=text,
        language="en-US",
        role="narrator",
        binding={
            "renderer": ENGINE_ID,
            "target_id": TARGET_ID,
            "voice": "smoke-voice",
            "options": {"speed": 1.0},
        },
        source=(SourceSpan(locator="wheel-smoke.source", start=0, end=len(text)),),
        planning_level="sentence",
        preflight_required=True,
    )
    plan = RenderPlan(
        source_plan_id="sha256:" + "1" * 64,
        render_profile_id="sha256:" + "2" * 64,
        flow=(item,),
    ).with_identity()

    work = Path.cwd() / "voicerender-wheel-smoke"
    shutil.rmtree(work, ignore_errors=True)

    assert BookRenderError.__module__ == "voicerender.book_contract"
    assert (
        BookRenderRequest.__module__
        == BookRenderResult.__module__
        == ChapterRenderOutcome.__module__
        == "voicerender.book"
    )
    book_workspace = work / "book-workspace"
    (book_workspace / "renderplan").mkdir(parents=True)
    try:
        render_book(book_workspace, variant="missing")
    except BookRenderError as exc:
        assert exc.code == "variant.not_found"
    else:
        raise AssertionError("render_book accepted a missing variant")
    report = preflight_render_plan(plan, work / "preflight.toml")
    assert report.status == "ok"
    result = render_renderplan(
        plan,
        preflight=report,
        output_dir=work / "rendered",
        cache_dir=work / "cache",
    )
    assert not result.failed, result.error
    manifest = load_manifest(result.manifest_path)
    manifest.validate_against(plan)
    assert manifest.complete and len(manifest.speech) == 1
    artifact = manifest.speech[0]
    assert verified_wav(
        result.manifest_path.parent / artifact.wav,
        {
            "sha256": artifact.sha256.removeprefix("sha256:"),
            "frames": artifact.frames,
            "sample_rate": artifact.sample_rate,
            "channels": artifact.channels,
        },
    )
    assert result.execution_fingerprint[0]["adapter_id"] == ENGINE_ID
    print("installed-wheel RenderPlan execution and manifest smoke passed")


if __name__ == "__main__":
    main()
