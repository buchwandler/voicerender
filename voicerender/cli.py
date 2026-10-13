"""Command-line interface for shared RenderPlan workflows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from renderplan import load_manifest, load_plan

from . import __version__
from .book import BookRenderError, BookRenderResult, render_book
from .engines.registry import (
    discover_targets,
    engine_status,
    get_adapter,
    list_engines,
)
from .errors import VoiceRenderError
from .preflight import preflight_render_plan
from .renderplan_render import RenderPlanResult, render_renderplan
from .renderprofile import (
    create_render_profile,
    load_render_profile,
    refresh_render_profile,
    save_render_profile,
    validate_render_profile,
)
from .storage import wav_info


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voicerender",
        description="Preflight and render shared RenderPlan speech bundles.",
    )
    parser.add_argument(
        "--version", action="version", version=f"voicerender {__version__}"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    engines = commands.add_parser("engines", help="List or inspect synthesis engines")
    engine_commands = engines.add_subparsers(dest="engines_command", required=True)
    engine_list = engine_commands.add_parser("list", help="List registered engines")
    engine_list.add_argument("--json", action="store_true")
    engine_inspect = engine_commands.add_parser("inspect", help="Inspect an engine")
    engine_inspect.add_argument("engine")
    engine_inspect.add_argument("--json", action="store_true")

    voices = commands.add_parser("voices", help="Discover engine targets")
    voice_commands = voices.add_subparsers(dest="voices_command", required=True)
    voice_list = voice_commands.add_parser("list", help="List targets for an engine")
    voice_list.add_argument("--engine", required=True)
    voice_list.add_argument("--lang")
    voice_list.add_argument("--offline", action="store_true")
    voice_list.add_argument("--json", action="store_true")

    profiles = commands.add_parser("profile", help="Create and validate RenderProfiles")
    profile_commands = profiles.add_subparsers(dest="profile_command", required=True)
    profile_create = profile_commands.add_parser(
        "create", help="Create a RenderProfile"
    )
    profile_create.add_argument("--engine", required=True)
    profile_create.add_argument("--target", required=True)
    profile_create.add_argument("--language", required=True)
    profile_create.add_argument("--role", default="narrator")
    profile_create.add_argument("--voice")
    profile_create.add_argument("--speaker")
    profile_create.add_argument(
        "--option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Engine option; VALUE is parsed as JSON when possible (repeatable)",
    )
    profile_create.add_argument("--offline", action="store_true")
    profile_create.add_argument("--refresh", action="store_true")
    profile_create.add_argument("-o", "--output", required=True, type=Path)
    profile_create.add_argument("--json", action="store_true")

    profile_validate = profile_commands.add_parser(
        "validate", help="Validate a RenderProfile"
    )
    profile_validate.add_argument("profile", type=Path)
    profile_validate.add_argument("--json", action="store_true")

    profile_refresh = profile_commands.add_parser(
        "refresh", help="Write a refreshed RenderProfile proposal"
    )
    profile_refresh.add_argument("profile", type=Path)
    profile_refresh.add_argument("-o", "--output", required=True, type=Path)
    profile_refresh.add_argument("--json", action="store_true")

    preflight = commands.add_parser(
        "preflight", help="Measure a RenderPlan without synthesis"
    )
    preflight.add_argument("plan", type=Path)
    preflight.add_argument("-o", "--output", required=True, type=Path)
    preflight.add_argument("--json", action="store_true")

    render = commands.add_parser(
        "render", help="Render a plan using a matching OK report"
    )
    render.add_argument("plan", type=Path)
    render.add_argument("--preflight", required=True, type=Path)
    render.add_argument("--out-dir", required=True, type=Path)
    render.add_argument("--cache-dir", type=Path)
    render.add_argument("--recheck-report", type=Path)
    render.add_argument("--force", action="store_true")
    render.add_argument("--json", action="store_true")

    book_render = commands.add_parser(
        "render-book", help="Render selected chapters from a RenderPlan variant"
    )
    book_render.add_argument(
        "book", type=Path, help="book workspace or its chapters/ directory"
    )
    book_render.add_argument(
        "--variant", required=True, help="exact RenderPlan variant directory name"
    )
    book_render.add_argument("--chapters", default="all", metavar="SELECTOR")
    book_render.add_argument("--speech", default="all", metavar="SELECTOR")
    book_render.add_argument("--renderplan-root", type=Path)
    book_render.add_argument("--output-root", type=Path)
    book_render.add_argument("--cache-dir", type=Path)
    book_render.add_argument("--force", action="store_true")
    book_render.add_argument("--fail-fast", action="store_true")
    book_render.add_argument("--json", action="store_true")

    inspect = commands.add_parser(
        "inspect-manifest", help="Verify a render manifest and its WAVs"
    )
    inspect.add_argument("manifest", type=Path)
    inspect.add_argument("--json", action="store_true")

    return parser


def _json_options(entries: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for entry in entries:
        key, separator, value = entry.partition("=")
        if not separator or not key.strip():
            raise ValueError(f"invalid --option {entry!r}; expected KEY=VALUE")
        key = key.strip()
        try:
            result[key] = json.loads(value)
        except json.JSONDecodeError:
            result[key] = value
    return result


def _target_payload(target: Any) -> dict[str, Any]:
    return {
        "engine": target.engine,
        "target_id": target.id,
        "name": target.display_name,
        "languages": list(target.languages),
        "voices": list(target.voices),
        "speakers": list(target.speakers),
        "status": target.status,
        "metadata": dict(target.metadata),
    }


def _preflight_payload(report: Any) -> dict[str, Any]:
    counts = {
        status: sum(check.status == status for check in report.checks)
        for status in ("ok", "too_long", "error")
    }
    return {
        "render_plan_id": report.render_plan_id,
        "render_profile_id": report.render_profile_id,
        "status": report.status,
        "checked": len(report.checks),
        "ok": counts["ok"],
        "replan": len(report.replan),
        "reject": counts["error"],
        "checks": counts,
        "exit_code": 0
        if report.status == "ok"
        else 2
        if report.status == "replan"
        else 1,
        "output": None,
        "error_code": (
            "preflight.replan"
            if report.status == "replan"
            else "preflight.reject"
            if report.status == "reject"
            else None
        ),
        "error": (
            "Renderer capability requires replanning."
            if report.status == "replan"
            else "Renderer preflight rejected this plan."
            if report.status == "reject"
            else None
        ),
    }


def _render_payload(result: RenderPlanResult) -> dict[str, Any]:
    return {
        "render_plan_id": result.manifest.render_plan_id,
        "manifest_id": result.manifest.manifest_id,
        "complete": result.manifest.complete,
        "speech_count": len(result.manifest.speech),
        "rendered": result.rendered_count,
        "reused": result.reused_count,
        "failed": result.failed,
        "manifest": str(result.manifest_path),
        "error_code": result.error_code,
        "execution_fingerprint": list(result.execution_fingerprint),
        "fresh_preflight": (
            _preflight_payload(result.fresh_preflight)
            if result.fresh_preflight is not None
            else None
        ),
        "error": result.error,
    }


def _print_book_result(result: BookRenderResult) -> None:
    request = result.request
    print("VoiceRender book render")
    print(f"  workspace:   {request.workspace}")
    print(f"  variant:     {request.variant_name}")
    print(f"  renderplans: {request.renderplan_variant_dir}")
    print(f"  output:      {request.output_variant_dir}")
    print(f"  chapters:    {request.chapter_selector}")
    print(f"  selected:    {result.requested}")
    for outcome in result.outcomes:
        print(f"[{outcome.index}/{result.requested}] {outcome.id}")
        if outcome.status == "failed":
            print("       FAILED")
            print(
                f"       {outcome.error_code or 'render.failed'}: {outcome.error_message or 'render failed'}"
            )
        elif outcome.status == "skipped":
            print("       skipped after an earlier chapter failure")
        else:
            if request.speech_selector.strip().casefold() != "all":
                print(
                    f"       speech: {outcome.selected_speech} of "
                    f"{outcome.planned_speech} selected"
                )
            print(f"       rendered: {outcome.rendered} new, {outcome.reused} reused")
            state = "complete" if outcome.manifest_complete else "partial"
            print(
                f"       chapter state: {state} "
                f"({outcome.available_speech}/{outcome.planned_speech} available)"
            )
            if outcome.manifest is not None:
                print(f"       manifest: {outcome.manifest}")
    print("VoiceRender summary")
    print(f"  requested: {result.requested}")
    print(f"  complete:  {result.full_chapters}")
    print(f"  partial:   {result.partial_chapters}")
    print(f"  failed:    {result.failed}")
    print(f"  skipped:   {result.skipped}")


def _inspect_manifest(path: Path) -> tuple[dict[str, Any], bool]:
    manifest = load_manifest(path)
    artifact_rows: list[dict[str, Any]] = []
    valid = True
    for artifact in manifest.speech:
        wav_path = path.parent / artifact.wav
        try:
            info = wav_info(wav_path)
            checks = {
                "sha256": artifact.sha256 == f"sha256:{info['sha256']}",
                "frames": artifact.frames == info["frames"],
                "sample_rate": artifact.sample_rate == info["sample_rate"],
                "channels": artifact.channels == info["channels"],
            }
            error = (
                None
                if all(checks.values())
                else "WAV metadata or digest does not match manifest"
            )
        except (OSError, VoiceRenderError, ValueError) as exc:
            checks = {
                "sha256": False,
                "frames": False,
                "sample_rate": False,
                "channels": False,
            }
            error = str(exc)
        artifact_valid = all(checks.values())
        valid = valid and artifact_valid
        artifact_rows.append(
            {
                "segment_id": artifact.segment_id,
                "number": artifact.segment_number,
                "wav": artifact.wav,
                "valid": artifact_valid,
                "checks": checks,
                "error": error,
            }
        )
    return (
        {
            "render_plan_id": manifest.render_plan_id,
            "manifest_id": manifest.manifest_id,
            "complete": manifest.complete,
            "speech_count": len(manifest.speech),
            "verified": sum(row["valid"] for row in artifact_rows),
            "artifacts": artifact_rows,
            "path": str(path),
        },
        valid,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "engines":
            if args.engines_command == "list":
                engines = list_engines()
                if args.json:
                    print(json.dumps(engines, ensure_ascii=False, indent=2))
                else:
                    for engine_info in engines:
                        state = (
                            "installed" if engine_info["installed"] else "not installed"
                        )
                        version = engine_info["installed_version"] or ""
                        print(f"{engine_info['id']}\t{state}\t{version}")
                return 0
            if args.engines_command == "inspect":
                adapter = get_adapter(args.engine)
                engine_payload = engine_status(adapter.id)
                print(
                    json.dumps(engine_payload, ensure_ascii=False, indent=2)
                    if args.json
                    else engine_payload
                )
                return 0

        if args.command == "voices":
            adapter = get_adapter(args.engine)
            targets = discover_targets(
                adapter.id, language=args.lang, offline=args.offline
            )
            target_payloads = [_target_payload(target) for target in targets]
            if args.json:
                print(json.dumps(target_payloads, ensure_ascii=False, indent=2))
            else:
                for target in target_payloads:
                    print(
                        f"{target['target_id']}\t{', '.join(target['languages']) or '-'}\t"
                        f"{target['name']}\t{', '.join(target['voices'])}"
                    )
            return 0

        if args.command == "profile":
            if args.profile_command == "create":
                profile = create_render_profile(
                    engine=args.engine,
                    target_id=args.target,
                    language=args.language,
                    role=args.role,
                    voice=args.voice,
                    speaker=args.speaker,
                    options=_json_options(args.option),
                    offline=args.offline,
                    refresh=args.refresh,
                )
                output = save_render_profile(profile, args.output)
                payload = {"profile_id": profile.profile_id, "path": str(output)}
            elif args.profile_command == "validate":
                profile = validate_render_profile(load_render_profile(args.profile))
                payload = {
                    "profile_id": profile.profile_id,
                    "roles": sorted(profile.roles),
                    "path": str(args.profile),
                    "valid": True,
                }
            else:
                if args.output.resolve() == args.profile.resolve():
                    raise ValueError(
                        "profile refresh output must not overwrite its source"
                    )
                profile = refresh_render_profile(load_render_profile(args.profile))
                output = save_render_profile(profile, args.output)
                payload = {"profile_id": profile.profile_id, "path": str(output)}
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            else:
                print(f"RenderProfile {payload['profile_id']} -> {payload.get('path')}")
            return 0

        if args.command == "preflight":
            if args.output.resolve() == args.plan.resolve():
                raise ValueError(
                    "preflight report path must not overwrite its RenderPlan"
                )
            report = preflight_render_plan(load_plan(args.plan), args.output)
            payload = _preflight_payload(report)
            payload["output"] = str(args.output)
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            else:
                print(
                    f"Preflight {report.status}: {payload['ok']} ok, "
                    f"{payload['replan']} replan, {payload['reject']} reject; "
                    f"{payload['checked']} speech items"
                )
            return 0 if report.status == "ok" else 2 if report.status == "replan" else 1

        if args.command == "render":
            if args.recheck_report is not None and args.recheck_report.resolve() in {
                args.plan.resolve(),
                args.preflight.resolve(),
            }:
                raise ValueError(
                    "fresh preflight report must not overwrite the RenderPlan or supplied report"
                )
            result = render_renderplan(
                args.plan,
                preflight=args.preflight,
                output_dir=args.out_dir,
                cache_dir=args.cache_dir,
                recheck_report_path=args.recheck_report,
                force=args.force,
            )
            payload = _render_payload(result)
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            elif result.failed:
                print(
                    f"Render incomplete: {result.error_code or 'render.failed'}: {result.error}; "
                    f"manifest: {result.manifest_path}"
                )
            else:
                print(
                    f"Rendered {payload['speech_count']} speech items "
                    f"({result.rendered_count} new, {result.reused_count} reused); "
                    f"manifest: {result.manifest_path}"
                )
            if result.failed:
                fresh = result.fresh_preflight
                return 2 if fresh is not None and fresh.status == "replan" else 1
            return 0

        if args.command == "render-book":
            try:
                book_result = render_book(
                    args.book,
                    variant=args.variant,
                    chapters=args.chapters,
                    speech=args.speech,
                    renderplan_root=args.renderplan_root,
                    output_root=args.output_root,
                    cache_dir=args.cache_dir,
                    force=args.force,
                    fail_fast=args.fail_fast,
                )
            except (BookRenderError, OSError, ValueError, TypeError) as exc:
                error_code = getattr(exc, "code", "workspace.invalid")
                payload = {
                    "error": str(exc),
                    "code": error_code,
                    "exit_code": 2,
                }
                if args.json:
                    print(json.dumps(payload, ensure_ascii=False, indent=2))
                else:
                    print(f"error: {exc}", file=sys.stderr)
                return 2
            if args.json:
                print(json.dumps(book_result.to_dict(), ensure_ascii=False, indent=2))
            else:
                _print_book_result(book_result)
            return book_result.exit_code

        if args.command == "inspect-manifest":
            payload, valid = _inspect_manifest(args.manifest)
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            else:
                state = "complete" if payload["complete"] else "partial"
                print(
                    f"Manifest {state}: {payload['verified']}/{payload['speech_count']} "
                    f"WAVs verified; {payload['path']}"
                )
                for artifact in payload["artifacts"]:
                    if not artifact["valid"]:
                        print(f"  {artifact['wav']}: {artifact['error']}")
            return 0 if valid else 1

    except (VoiceRenderError, OSError, ValueError, TypeError) as exc:
        if getattr(args, "json", False) and args.command == "render":
            error_code = getattr(exc, "code", None)
            payload = {
                "error": str(exc),
                "error_code": error_code
                if isinstance(error_code, str)
                else "input.invalid",
                "exit_code": 1,
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1
    parser.error("unsupported command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
