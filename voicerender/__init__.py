"""VoiceRender: standalone, resumable speech synthesis to reusable WAV segments."""

from importlib.metadata import PackageNotFoundError, version

from .errors import (
    ArtifactError,
    EngineError,
    PausePolicyError,
    PlanError,
    ProfileError,
    VoiceRenderError,
)
from .models import (
    RenderResult,
    SpeechSegment,
    VoiceBinding,
    VoiceConfig,
    VoicePlan,
    VoiceProfile,
)
from .render import render, render_plan
from .render_v5 import render_utterplan_v5

try:
    __version__ = version("voicerender")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "ArtifactError",
    "EngineError",
    "PausePolicyError",
    "PlanError",
    "ProfileError",
    "RenderResult",
    "SpeechSegment",
    "VoiceBinding",
    "VoiceConfig",
    "VoicePlan",
    "VoiceProfile",
    "VoiceRenderError",
    "__version__",
    "render",
    "render_plan",
    "render_utterplan_v5",
]
