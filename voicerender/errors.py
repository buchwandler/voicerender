"""Stable public exception types."""


class VoiceRenderError(Exception):
    """Base error for VoiceRender."""


class PlanError(VoiceRenderError, ValueError):
    """Invalid speech plan or selection."""


class ProfileError(VoiceRenderError, ValueError):
    """Invalid voice profile or unsupported selection."""


class PausePolicyError(VoiceRenderError, ValueError):
    """Invalid or unresolved semantic pause policy."""


class EngineError(VoiceRenderError, RuntimeError):
    """Engine installation, initialization or synthesis failure."""


class ArtifactError(VoiceRenderError, RuntimeError):
    """Corrupt or invalid WAV or cache artifact."""
