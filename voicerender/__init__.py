"""VoiceRender: request-level speech synthesis with shared RenderPlan contracts."""

from importlib.metadata import PackageNotFoundError, version

from renderplan import (
    PreflightReport,
    RenderManifest,
    RenderPlan,
    RenderProfile,
    SpeechItem,
)

from .engines.api_probe import EngineApiProbe
from .engines.base import (
    EngineAdapter,
    EngineCapabilities,
    EngineDiagnostic,
    EngineSelection,
    EngineSession,
    PronunciationSpan,
    RenderedSpeech,
    RenderProfileFragment,
    RequestMeasure,
    SpeechRequest,
    SpeechToken,
    SpeechWordTiming,
)
from .engines.catalog import CatalogRequest, CatalogResult, SynthesisTarget, TargetVoice
from .engines.registry import (
    discover_targets,
    engine_status,
    list_engines,
    open_session,
    resolve_voice,
    synthesize,
)
from .engines.selection import EngineRequest
from .errors import (
    ArtifactError,
    EngineApiIncompatibleError,
    EngineCapacityError,
    EngineError,
    EngineSelectionError,
    EngineSynthesisError,
    EngineUnavailableError,
    PlanError,
    ProfileError,
    RenderEventCallbackError,
    UnsupportedPlanSemanticsError,
    UnsupportedSynthesisFeatureError,
    VoiceRenderError,
)
from .events import RenderEvent, RenderEventCallback, RenderEventKind
from .models import (
    RenderResult,
    SpeechSegment,
    VoiceBinding,
    VoiceConfig,
    VoicePlan,
    VoiceProfile,
)
from .preflight import preflight_render_plan
from .render import render, render_plan
from .renderplan_render import RenderPlanResult, render_renderplan
from .renderprofile import (
    create_render_profile,
    load_render_profile,
    refresh_render_profile,
    save_render_profile,
    validate_render_profile,
)

try:
    __version__ = version("voicerender")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "ArtifactError",
    "CatalogRequest",
    "CatalogResult",
    "EngineAdapter",
    "EngineApiIncompatibleError",
    "EngineApiProbe",
    "EngineCapabilities",
    "EngineCapacityError",
    "EngineDiagnostic",
    "EngineError",
    "EngineRequest",
    "EngineSelection",
    "EngineSelectionError",
    "EngineSession",
    "EngineSynthesisError",
    "EngineUnavailableError",
    "PlanError",
    "PreflightReport",
    "ProfileError",
    "PronunciationSpan",
    "RenderEvent",
    "RenderEventCallback",
    "RenderEventCallbackError",
    "RenderEventKind",
    "RenderManifest",
    "RenderPlan",
    "RenderPlanResult",
    "RenderProfile",
    "RenderProfileFragment",
    "RenderResult",
    "RenderedSpeech",
    "RequestMeasure",
    "SpeechItem",
    "SpeechRequest",
    "SpeechSegment",
    "SpeechToken",
    "SpeechWordTiming",
    "SynthesisTarget",
    "TargetVoice",
    "UnsupportedPlanSemanticsError",
    "UnsupportedSynthesisFeatureError",
    "VoiceBinding",
    "VoiceConfig",
    "VoicePlan",
    "VoiceProfile",
    "VoiceRenderError",
    "__version__",
    "create_render_profile",
    "discover_targets",
    "engine_status",
    "list_engines",
    "load_render_profile",
    "open_session",
    "preflight_render_plan",
    "refresh_render_profile",
    "render",
    "render_plan",
    "render_renderplan",
    "resolve_voice",
    "save_render_profile",
    "synthesize",
    "validate_render_profile",
]
