"""VoiceRender ownership and lifecycle helpers for shared RenderProfiles."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from renderplan import (
    PausePolicy,
    PlanningPolicy,
    RenderProfile,
    RoleProfile,
    RoleRequirements,
    load_profile,
    load_profile_for_reidentification,
    save_profile,
)

from .engines.base import EngineSelection, RenderProfileFragment
from .engines.metadata import language_base
from .engines.profiles import validate_voice_source_binding
from .engines.registry import (
    discover_targets,
    get_adapter,
    normalize_engine_id,
    resolve_voice,
)
from .engines.selection import EngineRequest
from .errors import EngineError, EngineSelectionError, ProfileError
from .voice_sources import (
    VoiceSource,
    managed_reference_context,
    normalize_voice_source,
    validate_voice_source,
)

_DEFAULT_PAUSES = PausePolicy(
    mode="manual",
    durations_ms={"clause": 180, "sentence": 500, "paragraph": 800},
)


def create_render_profile(
    *,
    engine: str,
    target_id: str,
    language: str,
    role: str = "narrator",
    voice: str | None = None,
    speaker: str | int | None = None,
    voice_source: Mapping[str, Any] | None = None,
    transcript: str | None = None,
    options: Mapping[str, Any] | None = None,
    offline: bool = False,
    refresh: bool = False,
) -> RenderProfile:
    """Discover a binding and seed a user-owned shared RenderProfile."""
    if not role.strip():
        raise ProfileError("Profile role name must not be empty")
    if not language.strip():
        raise ProfileError("Profile language must not be empty")
    profile_options = dict(options or {})
    selected_voice_source = voice_source
    if selected_voice_source is None:
        selected_voice_source = profile_options.pop("voice_source", None)
    elif "voice_source" in profile_options:
        raise ProfileError("voice_source was provided both directly and in options")

    normalized_source: VoiceSource | None = None
    if selected_voice_source is not None:
        if voice is not None or speaker is not None:
            raise ProfileError("voice/speaker and voice_source cannot both be set")
        try:
            normalized_source = normalize_voice_source(selected_voice_source)
        except (EngineError, ValueError) as exc:
            raise ProfileError(str(exc)) from exc

    if transcript is not None:
        if not isinstance(transcript, str) or not transcript.strip():
            raise ProfileError("Transcript must be a non-empty string")
        if normalized_source is None or normalized_source.kind not in {
            "reference",
            "managed_reference",
        }:
            raise ProfileError(
                "--transcript requires --reference-wav, --voicefolio-ref, or --voice-source"
            )

    if normalized_source is not None and normalized_source.kind in {
        "reference",
        "managed_reference",
    }:
        diagnostics = validate_voice_source(normalized_source, offline=offline)
        errors = [item for item in diagnostics if item.severity == "error"]
        if errors:
            raise ProfileError(errors[0].message)

        source_metadata = normalized_source.metadata
        raw_source_transcript = source_metadata.get("transcript")
        if (
            not isinstance(raw_source_transcript, str)
            or not raw_source_transcript.strip()
        ):
            raw_source_transcript = (
                normalized_source.origin.get("reference_text")
                if normalized_source.origin is not None
                else None
            )
        source_transcript = (
            raw_source_transcript.strip()
            if isinstance(raw_source_transcript, str) and raw_source_transcript.strip()
            else None
        )
        catalog_transcript: str | None = None
        catalog_language: str | None = None
        if normalized_source.kind == "managed_reference":
            context = managed_reference_context(normalized_source.value)
            catalog_transcript = context.transcript
            catalog_language = context.language
            if (
                catalog_language
                and language_base(catalog_language) not in {"", "und"}
                and language_base(catalog_language) != language_base(language)
            ):
                raise ProfileError(
                    f"VoiceFolio reference {normalized_source.value!r} language "
                    f"{catalog_language!r} is incompatible with profile language {language!r}"
                )
            if (
                catalog_transcript is not None
                and transcript is not None
                and catalog_transcript != transcript
            ):
                raise ProfileError(
                    f"Reference transcript conflict for {normalized_source.value!r}: "
                    "the explicit transcript differs from the transcript pinned in "
                    "the VoiceFolio catalog."
                )
            if (
                catalog_transcript is not None
                and source_transcript is not None
                and catalog_transcript != source_transcript
            ):
                raise ProfileError(
                    f"Reference transcript conflict for {normalized_source.value!r}: "
                    "stored source text differs from the transcript pinned in the "
                    "VoiceFolio catalog."
                )
        if (
            transcript is not None
            and source_transcript is not None
            and transcript != source_transcript
        ):
            ref = normalized_source.value
            raise ProfileError(
                f"Reference transcript conflict for {ref!r}: the explicit transcript "
                "differs from the transcript attached to the voice source."
            )
        effective_transcript = transcript or catalog_transcript or source_transcript

        adapter = get_adapter(engine)
        capabilities = adapter.capabilities()
        bind_reference_voice = getattr(adapter, "bind_reference_voice", None)
        if callable(bind_reference_voice):
            targets = discover_targets(
                engine, language=language, offline=offline, refresh=refresh
            )
            target = next((item for item in targets if item.id == target_id), None)
            if target is None:
                raise ProfileError(
                    f"Engine {engine!r} does not expose target {target_id!r}"
                )
            if (
                not capabilities.supports_reference_voice
                and "reference_voice" not in target.capabilities
            ):
                raise EngineSelectionError(
                    f"Engine {adapter.id!r} target {target_id!r} does not support reference voices",
                    engine=adapter.id,
                    target_id=target_id,
                    code="voice_clone.reference_unsupported",
                )
            try:
                bound_options = bind_reference_voice(
                    target=target,
                    language=language,
                    voice_source=normalized_source.as_mapping(),
                    transcript=effective_transcript,
                    options=profile_options,
                )
            except EngineError as exc:
                if (
                    getattr(exc, "code", None) == "pykokoro.reference_text_required"
                    and normalized_source.kind == "managed_reference"
                    and catalog_transcript is None
                    and effective_transcript is None
                ):
                    raise EngineSelectionError(
                        "AkinVox reference cloning requires the exact reference transcript.\n"
                        f"VoiceFolio ref {normalized_source.value!r} has no transcript metadata.\n"
                        "Pass --transcript TEXT.",
                        engine=adapter.id,
                        target_id=target_id,
                        code="pykokoro.reference_text_required",
                    ) from exc
                raise
            if not isinstance(bound_options, Mapping):
                raise EngineSelectionError(
                    f"Engine {adapter.id!r} returned invalid reference voice options",
                    engine=adapter.id,
                    target_id=target_id,
                    code="voice_clone.invalid_binding",
                )
            profile_options = dict(bound_options)
        elif not capabilities.supports_reference_voice:
            targets = discover_targets(
                engine, language=language, offline=offline, refresh=refresh
            )
            target = next((item for item in targets if item.id == target_id), None)
            if target is None:
                raise ProfileError(
                    f"Engine {engine!r} does not expose target {target_id!r}"
                )
            if "reference_voice" not in target.capabilities:
                raise EngineSelectionError(
                    f"Engine {adapter.id!r} target {target_id!r} does not support reference voices",
                    engine=adapter.id,
                    target_id=target_id,
                    code="voice_clone.reference_unsupported",
                )
        else:
            accepted_options = set(capabilities.option_names) | set(
                capabilities.options
            )
            transcript_option = next(
                (
                    name
                    for name in ("reference_text", "transcript")
                    if name in accepted_options
                ),
                None,
            )
            if effective_transcript is not None and transcript_option is not None:
                existing = profile_options.get(transcript_option)
                if existing is not None and existing != effective_transcript:
                    raise ProfileError(
                        f"Conflicting transcript supplied in options.{transcript_option} "
                        "and reference binding"
                    )
                profile_options[transcript_option] = effective_transcript
        profile_options["voice_source"] = normalized_source.as_mapping()
    elif normalized_source is not None:
        profile_options["voice_source"] = normalized_source.as_mapping()

    request = EngineRequest(
        engine=engine,
        target_id=target_id,
        language=language,
        voice=voice,
        speaker=speaker,
        options=profile_options,
        offline=offline,
        refresh=refresh,
    )
    _, fragment = _resolve_fragment(request)
    role_profile = RoleProfile(
        language=language,
        binding=dict(fragment.binding),
        requirements=RoleRequirements(
            token_fields=fragment.token_fields,
            frontend=fragment.frontend,
            pronunciation=fragment.pronunciation,
        ),
        planner_capacity=fragment.planner_capacity,
        renderer_capacity=fragment.renderer_capacity,
    )
    return RenderProfile(
        roles={role: role_profile},
        default_role=role,
        planning=PlanningPolicy(
            split="sentence", capacity_fallback=("clause", "token")
        ),
        pauses=_DEFAULT_PAUSES,
    ).with_identity()


def load_render_profile(profile: str | Path | RenderProfile) -> RenderProfile:
    """Load a shared RenderProfile, preserving RenderPlan's strict codec checks."""
    if isinstance(profile, RenderProfile):
        return profile
    try:
        return load_profile(profile)
    except Exception as exc:
        raise ProfileError(f"Cannot load RenderProfile {profile}: {exc}") from exc



def reidentify_render_profile(profile: str | Path | RenderProfile) -> RenderProfile:
    """Compute the canonical identity without engine or metadata validation."""
    try:
        loaded = (
            load_profile_for_reidentification(profile)
            if not isinstance(profile, RenderProfile)
            else profile
        )
        return replace(loaded, profile_id=None).with_identity()
    except Exception as exc:
        raise ProfileError(f"Cannot re-identify RenderProfile {profile}: {exc}") from exc

def save_render_profile(profile: RenderProfile, path: str | Path) -> Path:
    """Persist a shared RenderProfile using the RenderPlan codec."""
    try:
        saved = save_profile(profile, path)
        if not isinstance(saved, Path):
            raise TypeError("RenderProfile codec did not return a Path")
        return saved
    except Exception as exc:
        raise ProfileError(f"Cannot write RenderProfile {path}: {exc}") from exc


def validate_render_profile(
    profile: str | Path | RenderProfile,
) -> RenderProfile:
    """Check each profile binding against the installed engine adapter/API."""
    loaded = load_render_profile(profile)
    for role_name, role in loaded.roles.items():
        binding = role.binding
        validate_voice_source_binding(binding)
        renderer = _required_string(
            binding.get("renderer"), f"roles.{role_name}.binding.renderer"
        )
        target_id = _required_string(
            binding.get("target_id"), f"roles.{role_name}.binding.target_id"
        )
        options = binding.get("options", {})
        if not isinstance(options, Mapping):
            raise ProfileError(f"Role {role_name!r} binding options must be a table")
        model_revision = binding.get("model_revision")
        if model_revision is not None and (
            not isinstance(model_revision, str) or not model_revision
        ):
            raise ProfileError(
                f"Role {role_name!r} model_revision must be a non-empty string"
            )
        selection, fragment = _resolve_fragment(
            EngineRequest(
                engine=renderer,
                target_id=target_id,
                language=role.language,
                voice=_optional_string(binding.get("voice"), "voice"),
                speaker=binding.get("speaker"),
                options=dict(options),
            )
        )
        if selection.target_id != target_id:
            raise ProfileError(
                f"Role {role_name!r} resolves to unexpected target {selection.target_id!r}"
            )
        expected_requirements = RoleRequirements(
            token_fields=fragment.token_fields,
            frontend=fragment.frontend,
            pronunciation=fragment.pronunciation,
        )
        if (
            role.requirements != expected_requirements
            or role.renderer_capacity != fragment.renderer_capacity
            or role.planner_capacity != fragment.planner_capacity
        ):
            raise ProfileError(
                f"Role {role_name!r} engine-derived requirements/capacity are stale; "
                "write and review a refreshed proposal"
            )
        proposed_metadata = fragment.binding.get("renderer_metadata", {})
        current_metadata = binding.get("renderer_metadata", {})
        if not isinstance(proposed_metadata, Mapping):
            proposed_metadata = {}
        if not isinstance(current_metadata, Mapping):
            current_metadata = {}
        if dict(current_metadata) != dict(proposed_metadata):
            raise ProfileError(
                f"Role {role_name!r} target metadata is stale; "
                "write and review a refreshed proposal"
            )
    return loaded


def refresh_render_profile(
    profile: str | Path | RenderProfile,
) -> RenderProfile:
    """Return a refreshed proposal without changing user-owned profile policy."""
    original = load_render_profile(profile)
    refreshed_roles: dict[str, RoleProfile] = {}
    for role_name, role in original.roles.items():
        binding = dict(role.binding)
        validate_voice_source_binding(binding)
        renderer = _required_string(
            binding.get("renderer"), f"roles.{role_name}.renderer"
        )
        target_id = _required_string(
            binding.get("target_id"), f"roles.{role_name}.target_id"
        )
        options = binding.get("options", {})
        if not isinstance(options, Mapping):
            raise ProfileError(f"Role {role_name!r} binding options must be a table")
        _, fragment = _resolve_fragment(
            EngineRequest(
                engine=renderer,
                target_id=target_id,
                language=role.language,
                voice=_optional_string(binding.get("voice"), "voice"),
                speaker=binding.get("speaker"),
                options=dict(options),
                refresh=True,
            )
        )
        new_binding = dict(binding)
        new_metadata = fragment.binding.get("renderer_metadata")
        if isinstance(new_metadata, Mapping):
            new_binding["renderer_metadata"] = dict(new_metadata)
        if binding.get("model_revision") is None and fragment.binding.get(
            "model_revision"
        ):
            new_binding["model_revision"] = fragment.binding["model_revision"]
        refreshed_roles[role_name] = RoleProfile(
            language=role.language,
            binding=new_binding,
            requirements=RoleRequirements(
                token_fields=fragment.token_fields,
                frontend=fragment.frontend,
                pronunciation=fragment.pronunciation,
            ),
            planner_capacity=fragment.planner_capacity,
            renderer_capacity=fragment.renderer_capacity,
        )
    return replace(original, roles=refreshed_roles, profile_id=None).with_identity()


def _resolve_fragment(
    request: EngineRequest,
) -> tuple[EngineSelection, RenderProfileFragment]:
    if not request.engine:
        raise ProfileError("A renderer engine is required")
    adapter = get_adapter(request.engine)
    probe = adapter.probe_api()
    if not probe.compatible:
        raise ProfileError(f"Engine {adapter.id!r} API is incompatible: {probe.status}")
    selection, diagnostics = resolve_voice(request)
    errors = [
        item for item in diagnostics if getattr(item, "severity", None) == "error"
    ]
    if errors:
        details = "; ".join(f"{item.code}: {item.message}" for item in errors)
        raise ProfileError(f"Engine binding is invalid: {details}")
    if normalize_engine_id(selection.engine) != normalize_engine_id(adapter.id):
        raise ProfileError("Engine adapter resolved an inconsistent renderer binding")
    selection = _with_target_metadata(adapter, selection)
    try:
        fragment = adapter.renderprofile_fragment(selection)
    except (ProfileError, ValueError) as exc:
        raise ProfileError(
            f"Cannot create engine RenderProfile fragment: {exc}"
        ) from exc
    if not isinstance(fragment, RenderProfileFragment):
        raise ProfileError(
            f"Engine {adapter.id!r} returned an invalid RenderProfile fragment"
        )
    validate_voice_source_binding(fragment.binding)
    if fragment.binding.get("renderer") != adapter.id:
        raise ProfileError("RenderProfile fragment renderer does not match its adapter")
    if fragment.binding.get("target_id") != selection.target_id:
        raise ProfileError(
            "RenderProfile fragment target_id does not match its selection"
        )
    return selection, fragment


def _with_target_metadata(adapter: Any, selection: EngineSelection) -> EngineSelection:
    metadata_reader = getattr(adapter, "target_metadata", None)
    if not callable(metadata_reader):
        return selection
    try:
        target_metadata = metadata_reader(selection)
    except Exception as exc:
        raise ProfileError(f"Cannot discover target metadata: {exc}") from exc
    if not isinstance(target_metadata, Mapping):
        return selection
    return replace(
        selection,
        metadata={**dict(selection.metadata), **dict(target_metadata)},
    )


def _required_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProfileError(f"{field_name} must be a non-empty string")
    return value


def _optional_string(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _required_string(value, field_name)


__all__ = [
    "create_render_profile",
    "load_render_profile",
    "refresh_render_profile",
    "save_render_profile",
    "validate_render_profile",
]
