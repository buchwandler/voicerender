"""VoiceRender: standalone, resumable speech synthesis to reusable WAV segments."""

from importlib.metadata import PackageNotFoundError, version

from .engines.api_probe import EngineApiProbe
from .engines.base import (
    EngineAdapter,
    EngineCapabilities,
    EngineDiagnostic,
    EngineSelection,
    EngineSession,
    PronunciationSpan,
    RenderedSpeech,
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
from .handoff import (
    load_source_plan,
    read_render_manifest,
    validate_manifest_source,
    verify_render_manifest,
)
from .inputs.utterplan_v5 import RenderSelection, UtterPlanV5, select_segments
from .models import (
    RenderResult,
    SpeechSegment,
    VoiceBinding,
    VoiceConfig,
    VoicePlan,
    VoiceProfile,
)
from .render import render, render_plan
from .render_v5 import render_utterplan, render_utterplan_v5, validate_utterplan

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
    "ProfileError",
    "PronunciationSpan",
    "RenderEvent",
    "RenderEventCallback",
    "RenderEventCallbackError",
    "RenderEventKind",
    "RenderResult",
    "RenderSelection",
    "RenderedSpeech",
    "RequestMeasure",
    "SpeechRequest",
    "SpeechSegment",
    "SpeechToken",
    "SpeechWordTiming",
    "SynthesisTarget",
    "TargetVoice",
    "UnsupportedPlanSemanticsError",
    "UnsupportedSynthesisFeatureError",
    "UtterPlanV5",
    "VoiceBinding",
    "VoiceConfig",
    "VoicePlan",
    "VoiceProfile",
    "VoiceRenderError",
    "__version__",
    "discover_targets",
    "engine_status",
    "list_engines",
    "load_source_plan",
    "open_session",
    "read_render_manifest",
    "render",
    "render_plan",
    "render_utterplan",
    "render_utterplan_v5",
    "resolve_voice",
    "select_segments",
    "synthesize",
    "validate_manifest_source",
    "validate_utterplan",
    "verify_render_manifest",
]
