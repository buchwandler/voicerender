"""Stable public exception types."""


class VoiceRenderError(Exception):
    """Base error for VoiceRender."""


class RenderEventCallbackError(VoiceRenderError):
    """A synchronous render-event callback raised an exception."""

    code = "render.event_callback_failed"

    def __init__(self, event_kind: str, original: Exception) -> None:
        self.event_kind = event_kind
        self.native_error_type = type(original).__name__
        super().__init__(
            f"Render event callback failed for {event_kind}: {self.native_error_type}"
        )


class PlanError(VoiceRenderError, ValueError):
    """Invalid speech plan or selection."""


class UnsupportedPlanSemanticsError(PlanError):
    """A valid source plan requires semantics VoiceRender cannot preserve."""

    code = "plan.unsupported_semantics"

    def __init__(
        self, message: str, *, details: dict[str, object] | None = None
    ) -> None:
        super().__init__(message)
        self.details = dict(details or {})


class ProfileError(VoiceRenderError, ValueError):
    """Invalid voice profile or unsupported selection."""


class EngineError(VoiceRenderError, RuntimeError):
    """Engine installation, initialization, or synthesis failure."""

    code = "engine.error"

    amount: int | None
    maximum: int | None
    unit: str | None

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        details: dict[str, object] | None = None,
        engine: str | None = None,
        engine_version: str | None = None,
        target_id: str | None = None,
        language: str | None = None,
        voice: str | None = None,
        speaker: str | int | None = None,
        request_id: str | None = None,
        native_error_type: str | None = None,
        amount: int | None = None,
        maximum: int | None = None,
        unit: str | None = None,
        text_length: int | None = None,
        source: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or type(self).code
        self.details = dict(details or {})
        self.details.update(
            {
                key: value
                for key, value in {
                    "engine": engine,
                    "engine_version": engine_version,
                    "target_id": target_id,
                    "language": language,
                    "voice": voice,
                    "speaker": speaker,
                    "request_id": request_id,
                    "native_error_type": native_error_type,
                    "amount": amount,
                    "maximum": maximum,
                    "unit": unit,
                    "text_length": text_length,
                    "source": source,
                }.items()
                if value is not None
            }
        )
        for key, value in self.details.items():
            setattr(self, key, value)
        self.__cause__ = None


class EngineUnavailableError(EngineError):
    code = "engine.unavailable"


class EngineApiIncompatibleError(EngineError):
    code = "engine.api_incompatible"


class EngineSelectionError(EngineError):
    code = "engine.selection_invalid"


class EngineCapacityError(EngineError):
    code = "engine.capacity_exceeded"


class EngineSynthesisError(EngineError):
    code = "engine.synthesis_failed"


class EmptySpeechTextError(EngineSynthesisError):
    code = "synthesis.empty_text"


class InvalidEngineModelError(EngineSynthesisError):
    code = "synthesis.invalid_model"


class InvalidEngineVoiceError(EngineSynthesisError):
    code = "synthesis.invalid_voice"


class InvalidEngineSpeakerError(EngineSynthesisError):
    code = "synthesis.invalid_speaker"


class InvalidEngineLanguageError(EngineSynthesisError):
    code = "synthesis.invalid_language"


class InvalidEngineOptionError(EngineSynthesisError):
    code = "synthesis.invalid_option"


class InvalidSpeechRequestError(EngineSynthesisError):
    code = "synthesis.invalid_request"


class SpeechRequestTooLongError(EngineCapacityError):
    code = "synthesis.request_too_long"


class UnsupportedSynthesisFeatureError(EngineSynthesisError):
    code = "synthesis.unsupported_feature"


class EngineBackendError(EngineSynthesisError):
    code = "synthesis.backend_error"


class ArtifactError(VoiceRenderError, RuntimeError):
    """Corrupt or invalid WAV or cache artifact."""

    code = "artifact.invalid"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code or type(self).code
