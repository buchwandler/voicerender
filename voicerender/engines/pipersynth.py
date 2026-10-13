"""Request-centric adapter for the published PiperSynth voice API."""

from __future__ import annotations

import math
from collections.abc import Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import asdict, is_dataclass
from typing import Any, cast

from ..errors import (
    EmptySpeechTextError,
    EngineBackendError,
    EngineError,
    EngineSynthesisError,
    InvalidEngineLanguageError,
    InvalidEngineModelError,
    InvalidEngineOptionError,
    InvalidEngineSpeakerError,
    InvalidSpeechRequestError,
    SpeechRequestTooLongError,
    UnsupportedSynthesisFeatureError,
)
from .api_probe import SUPPORTED_REQUEST_API_VERSION, EngineApiProbe, probe_public_api
from .base import (
    EngineCapabilities,
    EngineSelection,
    PronunciationSpan,
    RenderedSpeech,
    RenderProfileFragment,
    RequestMeasure,
    SpeechRequest,
    SpeechToken,
    distribution_version,
    validate_rendered_speech,
    voice_level_metadata,
)
from .catalog import CatalogRequest, SynthesisTarget, TargetVoice
from .metadata import (
    language_base,
    normalize_gender,
    normalize_language_key,
    normalize_locale_tag,
)
from .profiles import make_renderprofile_fragment

PIPER_RENDER_OPTIONS = frozenset(
    {
        "length_scale",
        "noise_scale",
        "noise_w_scale",
        "normalize_audio",
        "output_gain",
        "voice_level",
        "cache_dir",
        "providers",
        "provider_options",
        "session_options",
        "force_download",
    }
)


class PiperTargetRequiredError(ValueError):
    diagnostic_code = "piper.voice_required"
    diagnostic_field = "render.target.id"

    def __init__(self, language: str) -> None:
        super().__init__(
            "piper.voice_required: Piper needs a voice bundle target ID. Discover available targets "
            f"with `voicerender voices list --engine piper --lang {language}`, then bind the selected "
            "target to a RenderProfile role."
        )


def _target_from_voice_metadata(
    metadata: Any, engine: str = "piper"
) -> SynthesisTarget:
    speaker_map = dict(getattr(metadata, "speaker_id_map", {}) or {})
    language_code = getattr(metadata, "language_code", None)
    locale = normalize_locale_tag(language_code)
    language_family = getattr(metadata, "language_family", None)
    language = language_base(language_family) or language_base(locale)
    language_label = getattr(metadata, "language_label", None)
    if (
        not isinstance(language_label, str)
        or not language_label.strip()
        or (
            len(language_label.strip()) == 2
            and language_label.strip().isalpha()
            and language_label.strip().isupper()
        )
    ):
        language_label = locale or language or "unknown"
    return SynthesisTarget(
        engine=engine,
        id=metadata.id,
        display_name=getattr(metadata, "name", None) or metadata.id,
        languages=(locale,) if locale else (),
        status="ready",
        runtime_available=True,
        sample_rate=getattr(metadata, "sample_rate", None),
        voices=(metadata.id,),
        voice_details=(
            TargetVoice(
                id=metadata.id,
                gender=normalize_gender(getattr(metadata, "gender", None)),
                language=language,
                locale=locale,
                language_label=language_label,
            ),
        ),
        default_voice=metadata.id,
        speakers=tuple(str(name) for name in speaker_map),
        qualities=((metadata.quality,) if getattr(metadata, "quality", None) else ()),
        aliases=tuple(getattr(metadata, "aliases", ()) or ()),
        capabilities=frozenset({"speakers"}),
        metadata={
            "language": language,
            "locale": locale,
            "language_label": language_label,
            "gender": normalize_gender(getattr(metadata, "gender", None)),
            "language_code": language_code,
            "languages": ((locale,) if locale else ()),
            "language_family": language_family,
            "region": getattr(metadata, "region", None),
            "num_speakers": getattr(metadata, "num_speakers", 0),
            "speaker_id_map": speaker_map,
            "source_revision": getattr(metadata, "source_revision", None),
        },
    )


def _translate_piper_error(
    error: Exception,
    selection: EngineSelection,
    request: SpeechRequest | None = None,
    *,
    opening: bool = False,
) -> EngineError:
    import pipersynth

    error_map = (
        ("SynthesisInputTooLongError", SpeechRequestTooLongError),
        ("EmptyTextError", EmptySpeechTextError),
        ("InvalidLanguageError", InvalidEngineLanguageError),
        ("InvalidSpeakerError", InvalidEngineSpeakerError),
        ("InvalidSynthesisConfigError", InvalidEngineOptionError),
        ("InvalidLinguisticTokensError", InvalidSpeechRequestError),
        ("InvalidPronunciationError", InvalidSpeechRequestError),
        ("InvalidRequestError", InvalidSpeechRequestError),
        ("UnsupportedFeatureError", UnsupportedSynthesisFeatureError),
        ("UnsupportedModelError", InvalidEngineModelError),
        ("VoiceNotFoundError", InvalidEngineModelError),
        ("ModelFileNotFoundError", InvalidEngineModelError),
        ("ConfigFileNotFoundError", InvalidEngineModelError),
        ("ModelInferenceError", EngineBackendError),
        ("VoiceClosedError", EngineBackendError),
    )
    error_type: type[EngineError] = EngineBackendError
    for native_name, mapped_type in error_map:
        native_type = getattr(pipersynth, native_name, None)
        if isinstance(native_type, type) and isinstance(error, native_type):
            error_type = mapped_type
            break
    else:
        if opening and isinstance(error, (TypeError, ValueError)):
            error_type = InvalidEngineOptionError
        elif isinstance(error, (TypeError, ValueError)):
            error_type = InvalidSpeechRequestError

    context: dict[str, Any] = {
        "engine": "piper",
        "engine_version": distribution_version("pipersynth"),
        "target_id": selection.target_id,
        "language": request.language if request is not None else selection.language,
        "voice": request.voice if request is not None else selection.voice,
        "speaker": request.speaker if request is not None else selection.speaker,
        "request_id": request.id if request is not None else None,
        "native_error_type": type(error).__name__,
    }
    if error_type is SpeechRequestTooLongError:
        context.update(
            amount=getattr(error, "phoneme_count", None),
            maximum=getattr(error, "max_phonemes", None),
            unit="phoneme_ids",
            text_length=getattr(
                error, "text_length", len(request.text) if request else None
            ),
            source="pipersynth.strict_error_fallback",
        )
    return error_type(str(error), **context)


def _piper_token(item: SpeechToken, module: Any) -> Any:
    return module.LinguisticToken(
        start=item.start,
        end=item.end,
        text=item.text,
        pos=item.pos,
        tag=item.tag,
        lemma=item.lemma,
        language=item.language,
        morph=item.morph,
    )


def _piper_pronunciation(item: PronunciationSpan, module: Any) -> Any:
    if item.alphabet not in (None, "ipa"):
        raise UnsupportedSynthesisFeatureError(
            f"PiperSynth supports IPA pronunciation overrides, not {item.alphabet!r}",
            engine="piper",
            language=item.language,
        )
    return module.PronunciationOverride(
        start=item.start,
        end=item.end,
        phonemes=item.phonemes,
        language=item.language,
    )


def _piper_request_language(language: str, voice: Any) -> str:
    """Resolve a compatible public locale to Piper's active frontend language."""
    config = getattr(voice, "config", None)
    active_language = getattr(config, "espeak_voice", None)
    phoneme_type = getattr(config, "phoneme_type", None)
    phoneme_type = getattr(phoneme_type, "value", phoneme_type)
    if (
        phoneme_type != "espeak"
        or not isinstance(active_language, str)
        or not active_language
    ):
        return language

    requested = normalize_language_key(language)
    active = normalize_language_key(active_language)
    if requested == active:
        return active_language

    requested_parts = requested.split("-")
    active_parts = active.split("-")
    if requested_parts[0] != active_parts[0]:
        return language

    if len(requested_parts) == 1:
        return active_language

    if (
        len(active_parts) > len(requested_parts)
        and active_parts[: len(requested_parts)] == requested_parts
    ):
        return active_language

    return language


class PiperSynthEngineSession:
    """Adapt one exact VoiceRender request to one open PiperVoice."""

    def __init__(
        self, voice: Any, config: Any, selection: EngineSelection | None = None
    ) -> None:
        self._voice = voice
        self._config = config
        self._selection = selection or EngineSelection(
            engine="piper", target_id="unknown", language="und"
        )

    def measure(self, request: SpeechRequest) -> RequestMeasure:
        return RequestMeasure(
            fits=None,
            amount=None,
            maximum=None,
            unit="phoneme_ids",
            source="pipersynth.strict_error_fallback",
        )

    def synthesize(self, request: SpeechRequest) -> RenderedSpeech:
        import pipersynth

        try:
            native = pipersynth.SynthesisRequest(
                id=request.id,
                text=request.text,
                language=_piper_request_language(request.language, self._voice),
                speaker=request.speaker,
                tokens=tuple(
                    _piper_token(token, pipersynth) for token in request.tokens
                ),
                pronunciation_overrides=tuple(
                    _piper_pronunciation(span, pipersynth)
                    for span in request.pronunciation_overrides
                ),
            )
            result = self._voice.synthesize(native, config=self._config)
        except EngineSynthesisError:
            raise
        except Exception as exc:
            raise _translate_piper_error(exc, self._selection, request) from exc

        metadata = dict(result.metadata)
        frontend_diagnostics = metadata.get("frontend_diagnostics")
        if is_dataclass(frontend_diagnostics) and not isinstance(
            frontend_diagnostics, type
        ):
            metadata["frontend_diagnostics"] = asdict(cast(Any, frontend_diagnostics))
        metadata["voice_level"] = voice_level_metadata(
            metadata.get("voice_level"),
            str(self._selection.options.get("voice_level", "off")),
            identity_keys=("calibration_key", "calibration_identity"),
            revision_keys=("catalog_revision", "calibration_revision"),
        )
        rendered = RenderedSpeech(
            id=result.id,
            audio=result.audio,
            sample_rate=result.sample_rate,
            warnings=tuple(result.warnings),
            word_timings=(),
            metadata=metadata,
        )
        return validate_rendered_speech(
            request,
            rendered,
            engine="piper",
            engine_version=distribution_version("pipersynth"),
            target_id=self._selection.target_id,
        )


class PiperSynthEngineAdapter:
    """PiperSynth implementation of VoiceRender's request-oriented engine contract."""

    id = "piper"
    package_name = "pipersynth"

    def version(self) -> str | None:
        return distribution_version(self.package_name)

    def probe_api(self) -> EngineApiProbe:
        return probe_public_api(
            engine=self.id,
            package=self.package_name,
            required_symbols=(
                "PiperVoice",
                "VoiceAssetManager",
                "SynthesisRequest",
                "SynthesisResult",
                "SynthesisConfig",
                "LinguisticToken",
                "PronunciationOverride",
                "VoiceLevelConfig",
                "SynthesisInputTooLongError",
            ),
            required_methods={
                "PiperVoice": ("from_pretrained", "synthesize", "close"),
                "VoiceAssetManager": ("list_voices", "get_voice_metadata"),
            },
            expected_api_version=SUPPORTED_REQUEST_API_VERSION,
        )

    def compatible_api(self) -> bool:
        return self.probe_api().compatible

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            id=self.id,
            voice_binding_namespace="piper",
            supports_named_voices=True,
            voice_binding_scope="target",
            option_names=PIPER_RENDER_OPTIONS,
            supports_speakers=True,
            supports_linguistic_tokens=True,
            supports_pronunciation_overrides=True,
            pronunciation_alphabets=frozenset({"ipa"}),
            supports_voice_level_calibration=True,
            supports_request_measurement=False,
            supports_qualities=True,
            supports_live=True,
            supports_timestamps=False,
        )

    def renderprofile_fragment(
        self, selection: EngineSelection
    ) -> RenderProfileFragment:
        return make_renderprofile_fragment(
            self,
            selection,
            frontend="pipersynth",
            pronunciation="pipersynth.ipa",
        )

    def discover(self, request: CatalogRequest) -> tuple[SynthesisTarget, ...]:
        try:
            import pipersynth
        except ImportError:
            return ()
        manager = pipersynth.VoiceAssetManager(offline=request.offline)
        voices = manager.list_voices(language=request.language, refresh=request.refresh)
        return tuple(_target_from_voice_metadata(item, self.id) for item in voices)

    def validate_request(self, request: Any) -> None:
        """Validate required selection fields without touching PiperSynth APIs."""
        language = normalize_language_key(getattr(request, "language", None) or "en-us")
        target_id = getattr(request, "target_id", None) or getattr(
            request, "voice", None
        )
        if not target_id:
            raise PiperTargetRequiredError(language)

    def resolve(self, request: Any) -> tuple[EngineSelection, tuple[Any, ...]]:
        language = normalize_language_key(getattr(request, "language", None) or "en-us")
        voice = getattr(request, "voice", None)
        target_id = getattr(request, "target_id", None) or voice
        self.validate_request(request)
        if not isinstance(target_id, str) or not target_id:
            raise PiperTargetRequiredError(language)
        options = dict(getattr(request, "options", {}) or {})
        options.update(dict(getattr(request, "engine_options", {}) or {}))
        speed = options.pop("speed", options.pop("rate", None))
        if speed is not None:
            try:
                speed = float(speed)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "piper.invalid_speed: speed must be finite and > 0"
                ) from exc
            if not math.isfinite(speed) or speed <= 0:
                raise ValueError("piper.invalid_speed: speed must be finite and > 0")
            options.setdefault("length_scale", 1.0 / speed)
        if options.get("lexicons"):
            raise ValueError(
                "piper.option_not_supported: PiperSynth does not support lexicons"
            )
        engine_options = {
            key: value for key, value in options.items() if key in PIPER_RENDER_OPTIONS
        }
        return (
            EngineSelection(
                engine=self.id,
                target_id=target_id,
                language=language,
                voice=voice or target_id,
                speaker=getattr(request, "speaker", None),
                options=engine_options,
                offline=bool(getattr(request, "offline", False)),
                refresh=bool(getattr(request, "refresh", False)),
            ),
            (),
        )

    def validate_selection(self, selection: EngineSelection) -> tuple[Any, ...]:
        from .base import EngineDiagnostic

        try:
            import pipersynth
        except ImportError:
            return (
                EngineDiagnostic(
                    code="piper.engine_unavailable",
                    severity="error",
                    message='PiperSynth is not installed. Install with: pip install "voicerender[piper]"',
                    field="synthesis.engine",
                ),
            )
        manager = pipersynth.VoiceAssetManager(offline=selection.offline)
        try:
            metadata = manager.get_voice_metadata(
                selection.target_id, refresh=selection.refresh
            )
        except (KeyError, LookupError, OSError, ValueError) as exc:
            return (
                EngineDiagnostic(
                    code="piper.voice_not_found",
                    severity="error",
                    message=f"Piper voice bundle {selection.target_id!r} was not found: {exc}",
                    field="render.target.id",
                ),
            )
        requested = normalize_language_key(selection.language)
        actual = normalize_language_key(metadata.language_code)
        if (
            requested != actual
            and requested.split("-", 1)[0] != actual.split("-", 1)[0]
        ):
            return (
                EngineDiagnostic(
                    code="piper.language_mismatch",
                    severity="error",
                    message=f"Piper voice {selection.target_id!r} is {actual}, not {requested}.",
                    field="render.target.language",
                ),
            )
        if selection.speaker is not None:
            speaker_map = dict(metadata.speaker_id_map or {})
            valid = selection.speaker in speaker_map or (
                isinstance(selection.speaker, int)
                and selection.speaker in speaker_map.values()
            )
            if not valid:
                return (
                    EngineDiagnostic(
                        code="piper.speaker_not_found",
                        severity="error",
                        message=f"Speaker {selection.speaker!r} is not available in {selection.target_id!r}.",
                        field="render.target.speaker",
                    ),
                )
        return ()

    def target_metadata(self, selection: EngineSelection) -> Mapping[str, Any]:
        try:
            import pipersynth

            metadata = pipersynth.VoiceAssetManager(
                offline=selection.offline
            ).get_voice_metadata(
                selection.target_id,
                refresh=selection.refresh,
            )
        except (ImportError, KeyError, LookupError, OSError, ValueError):
            return {}
        return {
            **_target_from_voice_metadata(metadata).metadata,
            "quality": metadata.quality,
            "sample_rate": getattr(metadata, "sample_rate", None),
        }

    def canonical_synthesis_identity(
        self, selection: EngineSelection
    ) -> Mapping[str, Any]:
        return {
            "engine": self.id,
            "engine_version": self.version(),
            "target_id": selection.target_id,
            "voice": selection.voice,
            "speaker": selection.speaker,
            "options": dict(selection.options),
            "metadata": dict(selection.metadata),
        }

    def open(
        self, selection: EngineSelection
    ) -> AbstractContextManager[PiperSynthEngineSession]:
        import pipersynth

        options = dict(selection.options)
        try:
            voice = pipersynth.PiperVoice.from_pretrained(
                selection.target_id,
                cache_dir=options.get("cache_dir"),
                offline=selection.offline,
                refresh_catalog=selection.refresh,
                force_download=bool(options.get("force_download", False)),
                providers=options.get("providers"),
                provider_options=options.get("provider_options"),
                session_options=options.get("session_options"),
            )
            config = pipersynth.SynthesisConfig(
                length_scale=options.get("length_scale"),
                noise_scale=options.get("noise_scale"),
                noise_w_scale=options.get("noise_w_scale"),
                normalize_audio=bool(options.get("normalize_audio", True)),
                output_gain=float(options.get("output_gain", 1.0)),
                voice_level=pipersynth.VoiceLevelConfig(
                    mode=options.get("voice_level", "off")
                ),
            )
        except Exception as exc:
            raise _translate_piper_error(exc, selection, opening=True) from exc

        @contextmanager
        def session() -> Any:
            try:
                yield PiperSynthEngineSession(voice, config, selection)
            finally:
                voice.close()

        return session()


__all__ = ["PiperSynthEngineAdapter", "PiperSynthEngineSession"]
