"""PiperSynth 0.2.x backend. Mirrors its public request-centric contract."""

from __future__ import annotations

import math
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from collections.abc import Iterator

from ..errors import EngineError, ProfileError
from ..models import VoiceBinding
from .base import EngineCapabilities, SynthesisOutput

_ALLOWED = frozenset(
    {
        "native_speed",
        "length_scale",
        "noise_scale",
        "noise_w_scale",
        "normalize_audio",
        "output_gain",
        "voice_level",
        "offline",
    }
)


class PiperAdapter:
    id = "piper"
    package = "pipersynth"
    capabilities = EngineCapabilities(supports_speaker_selection=True, options=_ALLOWED)

    def version(self) -> str | None:
        try:
            return version(self.package)
        except PackageNotFoundError:
            return None

    def validate(self, binding: VoiceBinding) -> None:
        if binding.reference_audio:
            raise ProfileError("PiperSynth cannot clone a voice from reference audio")
        unknown = set(binding.options) - _ALLOWED
        if unknown:
            raise ProfileError(f"Unsupported Piper options: {sorted(unknown)}")
        if binding.options.get("native_speed") is not None:
            speed = binding.options["native_speed"]
            if (
                isinstance(speed, bool)
                or not isinstance(speed, (float, int))
                or not math.isfinite(speed)
                or speed <= 0
            ):
                raise ProfileError("native_speed must be finite and > 0")
            if "length_scale" in binding.options:
                raise ProfileError("Use native_speed or length_scale, not both")
        if binding.voice and binding.voice != binding.target_id:
            raise ProfileError(
                "Piper voice must match the target_id; use speaker for a multi-speaker target"
            )

    @staticmethod
    def _request_language(requested: str, voice: object) -> str:
        # Piper's eSpeak frontend may declare a more specific active locale
        # than a canonical voiceplan language such as "en-US".
        config = getattr(voice, "config", None)
        active = getattr(config, "espeak_voice", None)
        phoneme_type = getattr(config, "phoneme_type", None)
        if getattr(phoneme_type, "value", phoneme_type) != "espeak" or not active:
            return requested
        a = str(active).lower().replace("_", "-")
        r = requested.lower().replace("_", "-")
        if r == a or r == a.split("-")[0] or (a.startswith(r + "-")):
            return str(active)
        return requested

    @contextmanager
    def open(self, binding: VoiceBinding) -> Iterator[object]:
        self.validate(binding)
        try:
            import pipersynth
        except ImportError as exc:
            raise EngineError(
                'PiperSynth is not installed; run pip install "voicerender[piper]"'
            ) from exc
        opts = binding.options
        try:
            voice = pipersynth.PiperVoice.from_pretrained(
                binding.target_id, offline=bool(opts.get("offline", False))
            )
            length_scale = opts.get("length_scale")
            if "native_speed" in opts:
                length_scale = 1.0 / float(opts["native_speed"])
            config = pipersynth.SynthesisConfig(
                length_scale=length_scale,
                noise_scale=opts.get("noise_scale"),
                noise_w_scale=opts.get("noise_w_scale"),
                normalize_audio=opts.get("normalize_audio", True),
                output_gain=opts.get("output_gain", 1.0),
                voice_level=pipersynth.VoiceLevelConfig(
                    mode=opts.get("voice_level", "off")
                ),
            )
        except Exception as exc:
            raise EngineError(
                f"Cannot initialize Piper target {binding.target_id!r}: {exc}"
            ) from exc

        class Session:
            def synthesize(
                self, *, request_id: str, text: str, language: str
            ) -> SynthesisOutput:
                try:
                    native = pipersynth.SynthesisRequest(
                        id=request_id,
                        text=text,
                        language=PiperAdapter._request_language(language, voice),
                        speaker=binding.speaker,
                    )
                    result = voice.synthesize(native, config=config)
                except Exception as exc:
                    raise EngineError(f"Piper synthesis failed: {exc}") from exc
                return SynthesisOutput(
                    result.audio,
                    result.sample_rate,
                    metadata={"warnings": list(getattr(result, "warnings", ()) or ())},
                )

        try:
            yield Session()
        finally:
            voice.close()


def discover_voices(
    *, language: str | None = None, offline: bool = False
) -> list[dict[str, object]]:
    """List published Piper bundle IDs without synthesizing audio."""
    try:
        import pipersynth
    except ImportError as exc:
        raise EngineError(
            'Install optional runtime: pip install "voicerender[piper]"'
        ) from exc
    try:
        entries = pipersynth.VoiceAssetManager(offline=offline).list_voices(
            language=language
        )
        return [
            {
                "engine": "piper",
                "target_id": item.id,
                "name": getattr(item, "name", item.id),
                "language": getattr(item, "language_code", None),
            }
            for item in entries
        ]
    except Exception as exc:
        raise EngineError(f"Piper voice discovery failed: {exc}") from exc
