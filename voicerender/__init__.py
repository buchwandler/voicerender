"""VoiceRender: standalone, resumable speech synthesis to reusable WAV segments."""

from importlib.metadata import PackageNotFoundError, version

from .errors import (
    ArtifactError,
    EngineError,
    PlanError,
    ProfileError,
    VoiceRenderError,
)
from .models import RenderResult, SpeechSegment, VoiceBinding, VoicePlan, VoiceProfile
from .render import render, render_plan

try:
    __version__ = version("voicerender")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "__version__",
    "render",
    "render_plan",
    "SpeechSegment",
    "VoiceBinding",
    "VoicePlan",
    "VoiceProfile",
    "RenderResult",
    "VoiceRenderError",
    "PlanError",
    "ProfileError",
    "EngineError",
    "ArtifactError",
]
