"""Exact RenderPlan speech binding, token, and control conversion."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Any

from renderplan import SpeechControl, SpeechItem, TokenBundle

from .engines.base import (
    EngineSelection,
    PronunciationSpan,
    RenderProfileFragment,
    RequestMeasure,
    SpeechRequest,
    SpeechToken,
)
from .engines.profiles import validate_voice_source_binding
from .engines.registry import get_adapter, normalize_engine_id, resolve_voice
from .engines.selection import EngineRequest
from .errors import (
    EngineApiIncompatibleError,
    EngineSelectionError,
    EngineUnavailableError,
    PlanError,
    ProfileError,
    SpeechRequestTooLongError,
    UnsupportedSynthesisFeatureError,
)


class PreflightDisposition(str, Enum):
    OK = "ok"
    TOO_LONG = "too_long"
    REJECT = "reject"


@dataclass(frozen=True, slots=True)
class MeasurementOutcome:
    disposition: PreflightDisposition
    measurement: RequestMeasure | None = None
    error: str | None = None


EMPHASIS_COMPOSITION_CAPABILITY = "audiocompose.emphasis.gain.v1"


class EmphasisPolicy(str, Enum):
    """Policy for emphasis controls not natively interpreted by VoiceRender."""

    REJECT = "reject"
    DEFER_TO_COMPOSE = "defer-to-compose"


@dataclass(frozen=True, slots=True)
class SpeechRequestConversion:
    """Native speech request plus a trace of controls deferred downstream."""

    request: SpeechRequest
    deferred_capability: str | None = None
    deferred_control_count: int = 0


def engine_selection_from_binding(
    binding: Mapping[str, Any], *, language: str
) -> EngineSelection:
    """Resolve and validate a self-contained RenderPlan renderer binding."""
    if not isinstance(binding, Mapping):
        raise ProfileError("RenderPlan speech binding must be a table")
    validate_voice_source_binding(binding)
    allowed = {
        "renderer",
        "target_id",
        "voice",
        "speaker",
        "options",
        "model_revision",
        "renderer_metadata",
    }
    unknown = set(binding).difference(allowed)
    if unknown:
        raise ProfileError(
            "RenderPlan binding contains unsupported field(s): "
            + ", ".join(sorted(str(item) for item in unknown))
        )
    renderer = _required_string(binding.get("renderer"), "binding.renderer")
    target_id = _required_string(binding.get("target_id"), "binding.target_id")
    language = _required_string(language, "speech.language")
    voice = _optional_string(binding.get("voice"), "binding.voice")
    speaker = binding.get("speaker")
    if speaker is not None and (
        isinstance(speaker, bool) or not isinstance(speaker, (str, int))
    ):
        raise ProfileError("RenderPlan binding speaker must be a name or integer")
    if isinstance(speaker, str) and not speaker.strip():
        raise ProfileError("RenderPlan binding speaker must not be empty")
    options = binding.get("options", {})
    if not isinstance(options, Mapping):
        raise ProfileError("RenderPlan binding options must be a table")
    renderer_metadata = binding.get("renderer_metadata", {})
    if not isinstance(renderer_metadata, Mapping):
        raise ProfileError("RenderPlan binding renderer_metadata must be a table")
    model_revision = binding.get("model_revision")
    if model_revision is not None and (
        not isinstance(model_revision, str) or not model_revision
    ):
        raise ProfileError(
            "RenderPlan binding model_revision must be a non-empty string"
        )

    adapter = get_adapter(renderer)
    probe = adapter.probe_api()
    if not probe.compatible:
        details: dict[str, object] = {"probe": asdict(probe)}
        if probe.status == "package_missing":
            raise EngineUnavailableError(
                f"Engine {adapter.id!r} runtime package is unavailable",
                engine=adapter.id,
                native_error_type=probe.error_type,
                details=details,
            )
        raise EngineApiIncompatibleError(
            f"Engine {adapter.id!r} public synthesis API is incompatible",
            engine=adapter.id,
            engine_version=probe.distribution_version,
            native_error_type=probe.error_type,
            details=details,
        )

    request = EngineRequest(
        engine=renderer,
        target_id=target_id,
        language=language,
        voice=voice,
        speaker=speaker,
        options=dict(options),
    )
    selection, diagnostics = resolve_voice(request)
    errors = [
        item for item in diagnostics if getattr(item, "severity", None) == "error"
    ]
    if errors:
        detail = "; ".join(f"{item.code}: {item.message}" for item in errors)
        raise EngineSelectionError(
            f"Invalid RenderPlan renderer binding: {detail}",
            engine=adapter.id,
            target_id=target_id,
            details={"diagnostics": [item.as_dict() for item in errors]},
        )
    if normalize_engine_id(selection.engine) != normalize_engine_id(adapter.id):
        raise ProfileError("Engine adapter resolved an inconsistent renderer binding")
    if selection.target_id != target_id:
        raise ProfileError(
            f"Renderer {adapter.id!r} resolved unexpected target {selection.target_id!r}"
        )

    metadata = dict(selection.metadata)
    metadata_reader = getattr(adapter, "target_metadata", None)
    if callable(metadata_reader):
        try:
            discovered = metadata_reader(selection)
        except Exception as exc:
            raise ProfileError(
                f"Cannot inspect renderer target metadata: {exc}"
            ) from exc
        if isinstance(discovered, Mapping):
            metadata.update(discovered)
    # Prefer current discovery values; the copied facts are a self-contained fallback.
    for key, value in renderer_metadata.items():
        metadata.setdefault(str(key), value)
    metadata["renderer_metadata"] = dict(renderer_metadata)
    if model_revision is not None:
        metadata["model_revision"] = model_revision
    return replace(selection, metadata=metadata)


def speech_request_conversion_from_renderplan(
    item: SpeechItem,
    *,
    emphasis_policy: EmphasisPolicy | str = EmphasisPolicy.REJECT,
) -> SpeechRequestConversion:
    """Convert one SpeechItem into a request and its deferred-control trace."""
    if not isinstance(item, SpeechItem):
        raise PlanError("RenderPlan item must be a SpeechItem")
    try:
        policy = EmphasisPolicy(emphasis_policy)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "emphasis_policy must be 'reject' or 'defer-to-compose'"
        ) from exc
    selection = engine_selection_from_binding(item.binding, language=item.language)
    adapter = get_adapter(selection.engine)
    fragment_method = getattr(adapter, "renderprofile_fragment", None)
    if not callable(fragment_method):
        raise ProfileError(
            f"Engine {adapter.id!r} does not provide a RenderProfile fragment contract"
        )
    fragment = fragment_method(selection)
    if not isinstance(fragment, RenderProfileFragment):
        raise ProfileError(
            f"Engine {adapter.id!r} returned an invalid RenderProfile fragment"
        )
    tokens = _speech_tokens(item, fragment.token_fields)
    overrides, control_options, deferred_count = _lower_controls(
        item, adapter, selection, emphasis_policy=policy
    )
    options = dict(selection.options)
    if control_options:
        options["renderplan_controls"] = control_options
    request = SpeechRequest(
        id=item.segment_id or item.computed_id(),
        text=item.text,
        language=item.language,
        voice=selection.voice,
        speaker=selection.speaker,
        pronunciation_overrides=overrides,
        tokens=tokens,
        options=options,
    )
    return SpeechRequestConversion(
        request=request,
        deferred_capability=(
            EMPHASIS_COMPOSITION_CAPABILITY if deferred_count else None
        ),
        deferred_control_count=deferred_count,
    )


def speech_request_from_renderplan(
    item: SpeechItem,
    *,
    emphasis_policy: EmphasisPolicy | str = EmphasisPolicy.REJECT,
) -> SpeechRequest:
    """Convert one SpeechItem into the exact request used by measure and synthesize."""
    return speech_request_conversion_from_renderplan(
        item, emphasis_policy=emphasis_policy
    ).request


def _is_exact_amount(value: RequestMeasure) -> bool:
    return _nonnegative_int(value.amount) is not None


def classify_measurement(
    value: RequestMeasure | Exception,
) -> MeasurementOutcome:
    """Map adapter measurements and typed exceptions into stable preflight outcomes."""
    if isinstance(value, RequestMeasure):
        valid = (
            (value.fits is None or isinstance(value.fits, bool))
            and value.unit in {"model_tokens", "phoneme_ids", "unknown"}
            and (value.amount is None or value.unit != "unknown")
            and (isinstance(value.source, str) and bool(value.source.strip()))
            and (value.amount is None or _nonnegative_int(value.amount) is not None)
            and (value.maximum is None or _positive_int(value.maximum) is not None)
            and isinstance(value.details, Mapping)
        )
        if not valid:
            return MeasurementOutcome(
                PreflightDisposition.REJECT,
                measurement=value,
                error="Engine returned an invalid capacity measurement",
            )
        if value.source == "inflectsynth.capacity_metadata_mismatch":
            discovered = value.details.get("discovered_max_input_tokens")
            runtime = value.details.get("runtime_maximum")
            return MeasurementOutcome(
                PreflightDisposition.REJECT,
                measurement=value,
                error=(
                    "InflectSynth capacity metadata mismatch: "
                    f"discovered max_input_tokens={discovered}, "
                    f"runtime maximum={runtime}"
                ),
            )
        if value.source.endswith(".invalid_measurement"):
            return MeasurementOutcome(
                PreflightDisposition.REJECT,
                measurement=value,
                error="Engine returned an invalid capacity measurement",
            )
        if (
            value.amount is not None
            and value.maximum is not None
            and (
                (value.fits is True and value.amount > value.maximum)
                or (value.fits is False and value.amount <= value.maximum)
            )
        ):
            return MeasurementOutcome(
                PreflightDisposition.REJECT,
                measurement=value,
                error="Engine capacity measurement contradicts its fit result",
            )
        if not _is_exact_amount(value):
            return MeasurementOutcome(
                PreflightDisposition.REJECT,
                measurement=value,
                error="Engine did not provide an exact request measurement",
            )
        if value.fits is True:
            return MeasurementOutcome(PreflightDisposition.OK, measurement=value)
        if value.fits is False:
            return MeasurementOutcome(PreflightDisposition.TOO_LONG, measurement=value)
        if value.maximum is None:
            error = (
                "Engine measured the exact request size but did not provide an "
                "authoritative maximum/fit result"
            )
        else:
            error = "Engine measured the exact request size but did not provide an authoritative fit result"
        return MeasurementOutcome(
            PreflightDisposition.REJECT,
            measurement=value,
            error=error,
        )
    if isinstance(value, SpeechRequestTooLongError):
        raw_unit = getattr(value, "unit", None)
        unit = (
            raw_unit
            if isinstance(raw_unit, str)
            and raw_unit in {"model_tokens", "phoneme_ids", "unknown"}
            else "unknown"
        )
        details = getattr(value, "details", {})
        raw_source = getattr(value, "source", None)
        source = (
            raw_source
            if isinstance(raw_source, str) and raw_source.strip()
            else "engine.request_too_long"
        )
        raw_amount = getattr(value, "amount", None)
        raw_maximum = getattr(value, "maximum", None)
        amount = _nonnegative_int(raw_amount)
        maximum = _positive_int(raw_maximum)
        invalid_fields = []
        if raw_amount is not None and amount is None:
            invalid_fields.append("amount")
        if raw_maximum is not None and maximum is None:
            invalid_fields.append("maximum")
        normalized_details = dict(details) if isinstance(details, Mapping) else {}
        if invalid_fields:
            normalized_details["invalid_fields"] = invalid_fields
        measurement = RequestMeasure(
            fits=None if invalid_fields else False,
            amount=amount,
            maximum=maximum,
            unit=unit,  # type: ignore[arg-type]
            source=(f"{source}.invalid_measurement" if invalid_fields else source),
            details=normalized_details,
        )
        if invalid_fields:
            return MeasurementOutcome(
                PreflightDisposition.REJECT,
                measurement=measurement,
                error="Engine returned an invalid capacity measurement",
            )
        return MeasurementOutcome(
            PreflightDisposition.TOO_LONG, measurement=measurement
        )
    return MeasurementOutcome(
        PreflightDisposition.REJECT,
        error=str(value) or "Engine measurement failed",
    )


def _speech_tokens(
    item: SpeechItem, required_fields: tuple[str, ...]
) -> tuple[SpeechToken, ...]:
    bundle = item.tokens
    if bundle is None:
        if required_fields:
            raise ProfileError(
                f"Speech item {item.number} is missing required token fields: "
                + ", ".join(required_fields)
            )
        return ()
    if not isinstance(bundle, TokenBundle):
        raise PlanError(f"Speech item {item.number} has an invalid token bundle")
    spans = bundle.span
    if not isinstance(spans, tuple):
        raise PlanError("Token spans must be a tuple")
    count = len(spans)
    previous_end = -1
    for index, span in enumerate(spans):
        if (
            not isinstance(span, tuple)
            or len(span) != 2
            or any(
                isinstance(offset, bool) or not isinstance(offset, int)
                for offset in span
            )
        ):
            raise PlanError(f"Token span {index} must contain two integer offsets")
        start, end = span
        if start < 0 or start >= end or end > len(item.text) or start < previous_end:
            raise PlanError(f"Token span {index} is invalid for the exact speech text")
        previous_end = end

    dense = {"pos": bundle.pos, "tag": bundle.tag}
    for field_name, values in dense.items():
        if values and (
            not isinstance(values, tuple)
            or len(values) != count
            or any(not isinstance(value, str) for value in values)
        ):
            raise PlanError(
                f"Token field {field_name!r} must be dense and match token spans"
            )
    sparse = {"lemma": bundle.lemma, "morph": bundle.morph}
    for field_name, pairs in sparse.items():
        if not isinstance(pairs, tuple):
            raise PlanError(
                f"Token field {field_name!r} must be sparse token-index/value pairs"
            )
        previous_index = -1
        for pair in pairs:
            if (
                not isinstance(pair, tuple)
                or len(pair) != 2
                or isinstance(pair[0], bool)
                or not isinstance(pair[0], int)
                or pair[0] <= previous_index
                or pair[0] < 0
                or pair[0] >= count
                or not isinstance(pair[1], str)
                or not pair[1]
            ):
                raise PlanError(f"Token field {field_name!r} has invalid sparse values")
            previous_index = pair[0]

    complete = {"span": count > 0}
    for field_name, values in dense.items():
        complete[field_name] = (
            count > 0
            and len(values) == count
            and all(isinstance(value, str) and bool(value.strip()) for value in values)
        )
    for field_name, pairs in sparse.items():
        complete[field_name] = count > 0 and tuple(
            index for index, _ in pairs
        ) == tuple(range(count))
    missing = [
        field_name
        for field_name in required_fields
        if not complete.get(field_name, False)
    ]
    if missing:
        raise ProfileError(
            f"Speech item {item.number} has incomplete required token fields: "
            + ", ".join(missing)
        )

    lemma = dict(bundle.lemma)
    morph = dict(bundle.morph)
    result: list[SpeechToken] = []
    for index, (start, end) in enumerate(spans):
        result.append(
            SpeechToken(
                start=start,
                end=end,
                text=item.text[start:end],
                pos=bundle.pos[index] if bundle.pos else None,
                tag=bundle.tag[index] if bundle.tag else None,
                lemma=lemma.get(index),
                morph=morph.get(index),
                language=item.language,
            )
        )
    return tuple(result)


def _lower_controls(
    item: SpeechItem,
    adapter: Any,
    selection: EngineSelection,
    *,
    emphasis_policy: EmphasisPolicy,
) -> tuple[tuple[PronunciationSpan, ...], tuple[Mapping[str, Any], ...], int]:
    capabilities = adapter.capabilities()
    overrides: list[PronunciationSpan] = []
    lowered: list[Mapping[str, Any]] = []
    deferred_emphasis_count = 0
    for control in item.controls:
        _validate_control_range(item, control)
        if control.name == "pronunciation":
            if not capabilities.supports_pronunciation_overrides:
                raise UnsupportedSynthesisFeatureError(
                    f"Engine {adapter.id!r} does not support pronunciation controls"
                )
            params = control.params
            if not isinstance(params, Mapping):
                raise UnsupportedSynthesisFeatureError(
                    "Pronunciation control parameters must be a table"
                )
            unknown = set(params).difference({"phonemes", "alphabet", "language"})
            if unknown:
                raise UnsupportedSynthesisFeatureError(
                    "Unsupported pronunciation control parameter(s): "
                    + ", ".join(sorted(str(name) for name in unknown))
                )
            phonemes = params.get("phonemes")
            alphabet = params.get("alphabet", "ipa")
            language = params.get("language", item.language)
            if not isinstance(phonemes, str) or not phonemes:
                raise UnsupportedSynthesisFeatureError(
                    "Pronunciation control requires non-empty phonemes"
                )
            if (
                not isinstance(alphabet, str)
                or alphabet not in capabilities.pronunciation_alphabets
            ):
                raise UnsupportedSynthesisFeatureError(
                    f"Engine {adapter.id!r} does not support pronunciation alphabet {alphabet!r}"
                )
            if not isinstance(language, str) or not language:
                raise UnsupportedSynthesisFeatureError(
                    "Pronunciation control language must be a non-empty string"
                )
            overrides.append(
                PronunciationSpan(
                    start=control.start,
                    end=control.end,
                    phonemes=phonemes,
                    language=language,
                    alphabet=alphabet,
                )
            )
            continue

        if control.name == "emphasis":
            if emphasis_policy is not EmphasisPolicy.DEFER_TO_COMPOSE:
                raise UnsupportedSynthesisFeatureError(
                    "Emphasis requires an explicitly declared downstream handler "
                    f"({EMPHASIS_COMPOSITION_CAPABILITY}); select "
                    "emphasis_policy='defer-to-compose' to defer it"
                )
            _validate_emphasis_params(control.params)
            deferred_emphasis_count += 1
            continue

        if control.name != "prosody" and not control.name.startswith("extension:"):
            raise UnsupportedSynthesisFeatureError(
                f"Unsupported RenderPlan speech control {control.name!r}"
            )
        if control.name not in capabilities.native_semantics:
            raise UnsupportedSynthesisFeatureError(
                f"Engine {adapter.id!r} does not declare support for {control.name!r}"
            )
        lowerer = getattr(adapter, "lower_renderplan_control", None)
        if not callable(lowerer):
            raise UnsupportedSynthesisFeatureError(
                f"Engine {adapter.id!r} declares {control.name!r} but has no exact control lowering"
            )
        lowered_options = lowerer(control, selection)
        if (
            not isinstance(lowered_options, Mapping)
            or type(lowered_options.get("start")) is not int
            or lowered_options.get("start") != control.start
            or type(lowered_options.get("end")) is not int
            or lowered_options.get("end") != control.end
            or not isinstance(lowered_options.get("options"), Mapping)
        ):
            raise UnsupportedSynthesisFeatureError(
                f"Engine {adapter.id!r} did not preserve the exact range for {control.name!r}"
            )
        lowered.append(
            {
                "name": control.name,
                "start": control.start,
                "end": control.end,
                "options": dict(lowered_options["options"]),
            }
        )
    return tuple(overrides), tuple(lowered), deferred_emphasis_count


def _validate_control_range(item: SpeechItem, control: SpeechControl) -> None:
    if (
        type(control.start) is not int
        or type(control.end) is not int
        or control.start < 0
        or control.start >= control.end
        or control.end > len(item.text)
    ):
        raise UnsupportedSynthesisFeatureError(
            f"RenderPlan speech control {control.name!r} has an invalid range "
            f"[{control.start}, {control.end}) for text length {len(item.text)}"
        )


def _validate_emphasis_params(params: Any) -> None:
    if not isinstance(params, Mapping):
        raise UnsupportedSynthesisFeatureError(
            "Emphasis control parameters must be a table"
        )
    unknown = set(params).difference({"level"})
    if unknown:
        raise UnsupportedSynthesisFeatureError(
            "Unsupported emphasis control parameter(s): "
            + ", ".join(sorted(str(name) for name in unknown))
        )
    if "level" in params and (
        not isinstance(params["level"], str)
        or params["level"] not in {"reduced", "moderate", "strong"}
    ):
        raise UnsupportedSynthesisFeatureError(
            "Emphasis level must be one of: reduced, moderate, strong"
        )


def _required_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProfileError(f"{field_name} must be a non-empty string")
    return value


def _optional_string(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _required_string(value, field_name)


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


__all__ = [
    "EMPHASIS_COMPOSITION_CAPABILITY",
    "EmphasisPolicy",
    "MeasurementOutcome",
    "PreflightDisposition",
    "SpeechRequestConversion",
    "classify_measurement",
    "engine_selection_from_binding",
    "speech_request_conversion_from_renderplan",
    "speech_request_from_renderplan",
]
