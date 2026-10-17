"""Resumable, content-addressed speech rendering; no Readio dependency."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterable, Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from time import monotonic
from typing import Any

from .engines import get_engine
from .errors import ArtifactError, PlanError, ProfileError, RenderEventCallbackError
from .events import RenderEvent, RenderEventCallback, RenderEventKind, emit_render_event
from .models import (
    SCHEMA_MANIFEST,
    RenderResult,
    SpeechSegment,
    VoiceBinding,
    VoicePlan,
    VoiceProfile,
)
from .progress import ProgressTracker
from .storage import atomic_json, atomic_wav, digest, file_hash, verified_wav

RENDER_ALGORITHM = "voicerender.batch.v1"


def _identity(
    binding: VoiceBinding,
    *,
    engine_version: str | None,
    algorithm: str = RENDER_ALGORITHM,
) -> dict[str, Any]:
    model_asset_hash: str | None = None
    model_path = Path(binding.target_id).expanduser()
    if model_path.is_file():
        model_asset_hash = file_hash(model_path)
    reference_hash: str | None = None
    if binding.reference_audio is not None:
        if not binding.reference_audio.is_file():
            raise ProfileError(f"Reference audio is missing: {binding.reference_audio}")
        reference_hash = file_hash(binding.reference_audio)
    return {
        "engine": binding.engine,
        "engine_version": engine_version,
        "target_id": binding.target_id,
        "model_asset_sha256": model_asset_hash,
        "model_revision": binding.model_revision,
        "voice": binding.voice,
        "speaker": binding.speaker,
        "reference_sha256": reference_hash,
        "options": {
            key: value for key, value in binding.options.items() if key != "offline"
        },
        "algorithm": algorithm,
    }


def _safe_filename(segment: SpeechSegment, index: int) -> str:
    # No unsanitized logical ID is ever treated as an output path.
    return f"{index + 1:06d}-{digest([segment.scope_id, segment.id])[:12]}.wav"


def _select(
    segments: tuple[SpeechSegment, ...],
    *,
    take: int | None,
    select: Iterable[str] | None,
) -> list[SpeechSegment]:
    if take is not None and (
        isinstance(take, bool) or not isinstance(take, int) or take < 1
    ):
        raise PlanError("--take must be a positive integer")
    if take is not None and select is not None:
        raise PlanError("Use take or select, not both")
    if select is None:
        return list(segments[:take])
    wanted = list(select)
    if not wanted or len(wanted) != len(set(wanted)):
        raise PlanError("select must contain distinct segment IDs")
    matched: set[str] = set()
    result: list[SpeechSegment] = []
    for item in segments:
        qualified = f"{item.scope_id}:{item.id}"
        if qualified in wanted or item.id in wanted:
            key = qualified if qualified in wanted else item.id
            matched.add(key)
            result.append(item)
    missing = set(wanted) - matched
    if missing:
        raise PlanError(f"Unknown selected segment(s): {sorted(missing)}")
    for name in wanted:
        if ":" not in name and sum(segment.id == name for segment in segments) > 1:
            raise PlanError(f"Ambiguous segment ID {name!r}; use scope_id:segment_id")
    return result


@contextmanager
def _exclusive_run(output_dir: Path) -> Iterator[None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    lock = output_dir / ".voicerender.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ArtifactError(f"Render directory is locked: {output_dir}") from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        lock.unlink(missing_ok=True)


def _read_cache(cache_dir: Path, key: str) -> tuple[Path, dict[str, Any]] | None:
    wav = cache_dir / f"{key}.wav"
    meta = cache_dir / f"{key}.json"
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("key") != key:
        return None
    info = data.get("wav")
    if not isinstance(info, dict) or not verified_wav(wav, info):
        return None
    return wav, data


def _copy_verified(source: Path, target: Path, info: dict[str, Any]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name("." + target.name + ".tmp-" + str(os.getpid()))
    try:
        shutil.copyfile(source, tmp)
        if not verified_wav(tmp, info):
            raise ArtifactError(f"WAV copy failed integrity validation: {source}")
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def render(
    segments: Iterable[SpeechSegment],
    *,
    profile: VoiceProfile,
    output_dir: str | Path,
    cache_dir: str | Path | None = None,
    take: int | None = None,
    select: Iterable[str] | None = None,
    resume: bool = True,
    force: bool = False,
    source_plan_id: str | None = None,
    fail_fast: bool = True,
    on_event: RenderEventCallback | None = None,
) -> RenderResult:
    """Render selected logical speech segments into a persisted WAV bundle.

    A valid audio file is committed immediately after each successful synthesis.
    Identical effective synthesis requests share a WAV cache across profiles.
    """
    full = tuple(segments)
    if not full:
        raise PlanError("At least one speech segment is required")
    if any(not isinstance(seg, SpeechSegment) for seg in full):
        raise PlanError("segments must contain SpeechSegment values")
    ids = [(seg.scope_id, seg.id) for seg in full]
    if len(set(ids)) != len(ids):
        raise PlanError("Duplicate logical segment IDs")
    chosen = _select(full, take=take, select=select)
    if not chosen:
        raise PlanError("No speech segments selected")
    out = Path(output_dir).expanduser().resolve()
    cache = (
        Path(cache_dir).expanduser().resolve()
        if cache_dir
        else out.parent / ".voicerender-cache"
    )
    # Avoid using an output directory as the shared cache itself.
    if cache == out or cache.is_relative_to(out) or out.is_relative_to(cache):
        raise ArtifactError("cache_dir must be outside output_dir")

    identity_by_role: dict[str, dict[str, Any]] = {}
    for seg in chosen:
        binding = profile.roles.get(seg.role)
        if binding is None:
            raise ProfileError(
                f"No voice binding for role {seg.role!r} (segment {seg.id!r})"
            )
        if seg.role not in identity_by_role:
            adapter = get_engine(binding.engine)
            adapter.validate(binding)
            identity_by_role[seg.role] = _identity(
                binding, engine_version=adapter.version()
            )

    profile_id = digest(
        {role: identity_by_role[role] for role in sorted(identity_by_role)}
    )
    entries: list[dict[str, Any]] = []
    for index, seg in enumerate(chosen):
        acoustic = identity_by_role[seg.role]
        key = digest({"acoustic": acoustic, "text": seg.text, "language": seg.language})
        entries.append(
            {
                "scope_id": seg.scope_id,
                "segment_id": seg.id,
                "role": seg.role,
                "language": seg.language,
                "synthesis_key": key,
                "engine": acoustic["engine"],
                "target_id": acoustic["target_id"],
                "voice": acoustic["voice"],
                "model_revision": acoustic["model_revision"],
                "reference_sha256": acoustic["reference_sha256"],
                "path": f"segments/{_safe_filename(seg, index)}",
                "status": "pending",
                "wav": None,
                "timings": [],
                "timing_source": "absent",
                "error": None,
            }
        )
    document: dict[str, Any] = {
        "schema": SCHEMA_MANIFEST,
        "source_plan_id": source_plan_id,
        "profile_name": profile.name,
        "effective_profile_id": profile_id,
        "selected_count": len(chosen),
        "complete": False,
        "segments": entries,
    }
    manifest_path = out / "manifest.json"
    rendered = reused = failed = 0
    progress_tracker = ProgressTracker(len(chosen))

    def emit_progress(
        kind: RenderEventKind,
        *,
        index: int | None = None,
        segment: SpeechSegment | None = None,
        engine: str | None = None,
        status: str | None = None,
        error: Exception | None = None,
        synthesis_seconds: float | None = None,
    ) -> None:
        snapshot = progress_tracker.snapshot()
        error_code = getattr(error, "code", None) if error is not None else None
        emit_render_event(
            on_event,
            RenderEvent(
                kind=kind,
                plan_id=source_plan_id,
                locator=(
                    f"{segment.scope_id}:{segment.id}" if segment is not None else None
                ),
                role=segment.role if segment is not None else None,
                engine=engine,
                status=status,
                error_type=type(error).__name__ if error is not None else None,
                error_code=error_code if isinstance(error_code, str) else None,
                selected_count=len(chosen),
                scope="render",
                phase="render",
                segment_number=index + 1 if index is not None else None,
                segment_id=segment.id if segment is not None else None,
                completed_count=snapshot.completed_count,
                total_count=snapshot.total_count,
                progress=snapshot.progress,
                rendered_count=snapshot.rendered_count,
                reused_count=snapshot.reused_count,
                failed_count=snapshot.failed_count,
                phase_completed_count=snapshot.completed_count,
                phase_total_count=snapshot.total_count,
                phase_progress=snapshot.progress,
                elapsed_seconds=snapshot.elapsed_seconds,
                synthesis_seconds=synthesis_seconds,
                eta_seconds=snapshot.eta_seconds,
                eta_confidence=snapshot.eta_confidence,
                progress_snapshot=snapshot,
            ),
        )

    emit_progress(RenderEventKind.RENDER_STARTED)

    with _exclusive_run(out):
        prior: dict[tuple[str | None, str | None, str | None], Any] = {}
        if resume and not force and manifest_path.is_file():
            try:
                previous = json.loads(manifest_path.read_text(encoding="utf-8"))
                if previous.get("schema") == SCHEMA_MANIFEST:
                    prior = {
                        (
                            item.get("scope_id"),
                            item.get("segment_id"),
                            item.get("synthesis_key"),
                        ): item
                        for item in previous.get("segments", [])
                        if isinstance(item, dict)
                    }
            except (OSError, ValueError, TypeError):
                pass
        atomic_json(manifest_path, document)
        cache.mkdir(parents=True, exist_ok=True)
        with ExitStack() as opened:
            sessions: dict[str, Any] = {}
            for index, seg in enumerate(chosen):
                entry = entries[index]
                key = entry["synthesis_key"]
                target = out / entry["path"]
                engine_name = str(entry["engine"])
                emit_progress(
                    RenderEventKind.SEGMENT_STARTED,
                    index=index,
                    segment=seg,
                    engine=engine_name,
                    status="started",
                )
                synthesis_seconds: float | None = None
                try:
                    old = prior.get((seg.scope_id, seg.id, key)) if prior else None
                    if (
                        not force
                        and old
                        and isinstance(old.get("wav"), dict)
                        and verified_wav(target, old["wav"])
                    ):
                        entry.update(
                            status="reused",
                            wav=old["wav"],
                            timings=old.get("timings", []),
                            timing_source=old.get("timing_source", "absent"),
                        )
                        reused += 1
                    else:
                        cached = None if force else _read_cache(cache, key)
                        if cached:
                            source, sidecar = cached
                            _copy_verified(source, target, sidecar["wav"])
                            entry.update(
                                status="reused",
                                wav=sidecar["wav"],
                                timings=sidecar.get("timings", []),
                                timing_source=sidecar.get("timing_source", "absent"),
                            )
                            reused += 1
                        else:
                            binding = profile.roles[seg.role]
                            role_key = digest(identity_by_role[seg.role])
                            if role_key not in sessions:
                                emit_progress(
                                    RenderEventKind.ENGINE_OPEN_STARTED,
                                    index=index,
                                    segment=seg,
                                    engine=binding.engine,
                                    status="opening",
                                )
                                try:
                                    session = opened.enter_context(
                                        get_engine(binding.engine).open(binding)
                                    )
                                except RenderEventCallbackError:
                                    raise
                                except Exception as exc:
                                    emit_progress(
                                        RenderEventKind.ENGINE_OPEN_FINISHED,
                                        index=index,
                                        segment=seg,
                                        engine=binding.engine,
                                        status="failed",
                                        error=exc,
                                    )
                                    raise
                                sessions[role_key] = session
                                emit_progress(
                                    RenderEventKind.ENGINE_OPEN_FINISHED,
                                    index=index,
                                    segment=seg,
                                    engine=binding.engine,
                                    status="opened",
                                )
                            synthesis_started = monotonic()
                            result = sessions[role_key].synthesize(
                                request_id=seg.id, text=seg.text, language=seg.language
                            )
                            synthesis_seconds = max(
                                0.0, monotonic() - synthesis_started
                            )
                            info = atomic_wav(
                                cache / f"{key}.wav",
                                result.audio,
                                int(result.sample_rate),
                            )
                            timings = [dict(item) for item in result.word_timings]
                            sidecar = {
                                "key": key,
                                "wav": info,
                                "timings": timings,
                                "timing_source": "provided" if timings else "absent",
                            }
                            atomic_json(cache / f"{key}.json", sidecar)
                            _copy_verified(cache / f"{key}.wav", target, info)
                            entry.update(
                                status="rendered",
                                wav=info,
                                timings=timings,
                                timing_source=sidecar["timing_source"],
                            )
                            rendered += 1
                except RenderEventCallbackError:
                    raise
                except Exception as exc:
                    entry.update(
                        status="failed",
                        error={"type": type(exc).__name__, "message": str(exc)},
                    )
                    failed += 1
                    atomic_json(manifest_path, document)
                    progress_tracker.complete("failed")
                    emit_progress(
                        RenderEventKind.SEGMENT_FAILED,
                        index=index,
                        segment=seg,
                        engine=engine_name,
                        status="failed",
                        error=exc,
                    )
                    if fail_fast:
                        raise
                else:
                    atomic_json(manifest_path, document)
                    if entry["status"] == "rendered":
                        progress_tracker.complete(
                            "rendered", synthesis_seconds=synthesis_seconds
                        )
                        emit_progress(
                            RenderEventKind.SEGMENT_RENDERED,
                            index=index,
                            segment=seg,
                            engine=engine_name,
                            status="rendered",
                            synthesis_seconds=synthesis_seconds,
                        )
                    else:
                        progress_tracker.complete("reused")
                        emit_progress(
                            RenderEventKind.SEGMENT_REUSED,
                            index=index,
                            segment=seg,
                            engine=engine_name,
                            status="reused",
                        )
        document["complete"] = failed == 0
        atomic_json(manifest_path, document)
        emit_progress(
            RenderEventKind.RENDER_COMPLETED,
            status="failed" if failed else "completed",
        )
    return RenderResult(
        manifest_path=manifest_path,
        rendered_count=rendered,
        reused_count=reused,
        failed_count=failed,
        selected_count=len(chosen),
    )


def render_plan(
    plan: VoicePlan | str | Path,
    *,
    profile: VoiceProfile | str | Path,
    output_dir: str | Path,
    cache_dir: str | Path | None = None,
    take: int | None = None,
    select: Iterable[str] | None = None,
    resume: bool = True,
    force: bool = False,
    fail_fast: bool = True,
    on_event: RenderEventCallback | None = None,
) -> RenderResult:
    """Load the neutral voiceplan TOML contract and execute its speech segments."""
    parsed = VoicePlan.load(plan) if isinstance(plan, (str, Path)) else plan
    loaded_profile = (
        VoiceProfile.load(profile) if isinstance(profile, (str, Path)) else profile
    )
    return render(
        parsed.segments,
        profile=loaded_profile,
        output_dir=output_dir,
        cache_dir=cache_dir,
        take=take,
        select=select,
        resume=resume,
        force=force,
        source_plan_id=parsed.source_plan_id,
        fail_fast=fail_fast,
        on_event=on_event,
    )
