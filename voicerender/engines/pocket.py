"""PocketSynth 0.2.x named-voice and local reference-audio backend."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version

from ..errors import EngineError, ProfileError
from ..models import VoiceBinding
from .base import EngineCapabilities, SynthesisOutput, SynthesisRequest

_ALLOWED = frozenset(
    {
        "precision",
        "temperature",
        "lsd_steps",
        "max_frames",
        "frames_after_eos",
        "voice_level",
        "offline",
    }
)


class PocketAdapter:
    id = "pocket"
    package = "pocketsynth"
    capabilities = EngineCapabilities(
        supports_reference_audio=True,
        options=_ALLOWED,
        supports_synthesis_request=True,
    )

    def version(self) -> str | None:
        try:
            return version(self.package)
        except PackageNotFoundError:
            return None

    def validate(self, binding: VoiceBinding) -> None:
        unknown = set(binding.options) - _ALLOWED
        if unknown:
            raise ProfileError(f"Unsupported Pocket options: {sorted(unknown)}")
        if not binding.reference_audio and not binding.voice:
            raise ProfileError("Pocket requires a named voice or reference_audio")
        if binding.speaker is not None:
            raise ProfileError("Pocket does not support speaker selection")
        if binding.reference_audio and not binding.reference_audio.is_file():
            raise ProfileError(f"Reference audio missing: {binding.reference_audio}")
        if "native_speed" in binding.options:
            raise ProfileError("PocketSynth has no supported native speed control")

    @contextmanager
    def open(self, binding: VoiceBinding) -> Iterator[object]:
        self.validate(binding)
        try:
            import pocketsynth
        except ImportError as exc:
            raise EngineError(
                'PocketSynth is not installed; run pip install "voicerender[pocket]"'
            ) from exc
        opts = binding.options
        try:
            manager = pocketsynth.BundleAssetManager(
                offline=bool(opts.get("offline", False))
            )
            bundle = manager.resolve_bundle(
                binding.target_id, precision=opts.get("precision", "int8")
            )
            runtime = pocketsynth.PocketRuntime.from_resolved(
                bundle, offline=bool(opts.get("offline", False))
            )
            generation_args = {
                key: opts[key]
                for key in (
                    "temperature",
                    "lsd_steps",
                    "max_frames",
                    "frames_after_eos",
                )
                if key in opts
            }
            generation = pocketsynth.GenerationConfig(**generation_args)
            voice_level = pocketsynth.VoiceLevelConfig(
                mode=opts.get("voice_level", "off")
            )
            voice = runtime.prepare_voice(binding.reference_audio or binding.voice)
        except Exception as exc:
            if "runtime" in locals():
                runtime.close()
            raise EngineError(
                f"Cannot initialize Pocket target {binding.target_id!r}: {exc}"
            ) from exc

        class Session:
            def synthesize_request(self, request: SynthesisRequest) -> SynthesisOutput:
                if request.directives:
                    raise EngineError(
                        "Pocket does not support per-segment UtterPlan directives"
                    )
                return self.synthesize(
                    request_id=request.request_id,
                    text=request.text,
                    language=request.language,
                )

            def synthesize(
                self, *, request_id: str, text: str, language: str
            ) -> SynthesisOutput:
                try:
                    request = pocketsynth.SynthesisRequest(
                        id=request_id, text=text, language=language
                    )
                    result = runtime.synthesize(
                        request, voice=voice, config=generation, voice_level=voice_level
                    )
                except Exception as exc:
                    raise EngineError(f"Pocket synthesis failed: {exc}") from exc
                return SynthesisOutput(
                    result.audio,
                    result.sample_rate,
                    metadata={"warnings": list(getattr(result, "warnings", ()) or ())},
                )

        try:
            yield Session()
        finally:
            runtime.close()


def discover_voices(
    *, language: str | None = None, offline: bool = False
) -> list[dict[str, object]]:
    """List Pocket bundle IDs and named voice choices without loading a model."""
    try:
        import pocketsynth
    except ImportError as exc:
        raise EngineError(
            'Install optional runtime: pip install "voicerender[pocket]"'
        ) from exc
    try:
        manager = pocketsynth.BundleAssetManager(offline=offline)
        bundles = manager.list_bundles(language=language)
        output = []
        for bundle in bundles:
            bundle_id = getattr(bundle, "id", None) or getattr(
                bundle, "bundle_id", None
            )
            if not bundle_id:
                continue
            names = getattr(bundle, "predefined_voices", None)
            if names is None:
                meta = getattr(bundle, "metadata", {}) or {}
                names = (
                    meta.get("predefined_voice_names", ())
                    if isinstance(meta, dict)
                    else ()
                )
            output.append(
                {
                    "engine": "pocket",
                    "target_id": bundle_id,
                    "voices": list(names or ()),
                }
            )
        return output
    except Exception as exc:
        raise EngineError(f"Pocket voice discovery failed: {exc}") from exc
