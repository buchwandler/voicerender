"""Command-line interface for independent VoiceRender usage."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from . import __version__
from .engines import (
    EngineRequest,
    discover_targets,
    engine_status,
    list_engines,
    resolve_voice,
)
from .errors import VoiceRenderError
from .manifest import MANIFEST_SCHEMA as MANIFEST_SCHEMA_V2
from .manifest import read_manifest
from .models import SCHEMA_MANIFEST, VoiceProfile
from .render import render_plan
from .storage import verified_wav


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voicerender",
        description="Render VoicePlan or UtterPlan v5 to resumable WAV bundles",
    )
    parser.add_argument(
        "--version", action="version", version=f"voicerender {__version__}"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    render = commands.add_parser(
        "render", help="Synthesize selected logical speech segments"
    )
    render.add_argument("plan", type=Path)
    voice_input = render.add_mutually_exclusive_group(required=True)
    voice_input.add_argument("--profile", "--voices", dest="profile", type=Path)
    voice_input.add_argument(
        "--voice", type=Path, help="voice.toml for direct UtterPlan v5 rendering"
    )
    render.add_argument("--out-dir", required=True, type=Path)
    render.add_argument("--cache-dir", type=Path)
    select = render.add_mutually_exclusive_group()
    select.add_argument("--take", type=int)
    select.add_argument(
        "--select", metavar="ID[,ID...]", help="Legacy IDs or v5 source locators"
    )
    render.add_argument(
        "--force", action="store_true", help="Regenerate selected audio, bypass cache"
    )
    render.add_argument("--continue-on-error", action="store_true")
    render.add_argument("--json", action="store_true")
    engines = commands.add_parser("engines", help="Inspect installed synthesis engines")
    engines.add_argument("action", choices=["list", "inspect"])
    engines.add_argument("engine", nargs="?", help="Engine ID (required for inspect)")
    engines.add_argument("--json", action="store_true")
    voices = commands.add_parser(
        "voices", help="Discover backend voice and model choices"
    )
    voices.add_argument("action", choices=["list"])
    voices.add_argument("--engine", required=True)
    voices.add_argument("--lang")
    voices.add_argument("--offline", action="store_true")
    voices.add_argument("--refresh", action="store_true")
    voices.add_argument("--json", action="store_true")
    profiles = commands.add_parser("profiles", help="Validate a voice profile")
    profiles.add_argument("action", choices=["validate"])
    profiles.add_argument("path", type=Path)
    profiles.add_argument("--json", action="store_true")
    inspect = commands.add_parser(
        "inspect", help="Inspect a persisted manifest and verify WAVs"
    )
    inspect.add_argument("manifest", type=Path)
    inspect.add_argument(
        "--verify",
        action="store_true",
        help="Verify WAV bytes (verification is also the default)",
    )
    inspect.add_argument("--json", action="store_true")
    validate = commands.add_parser(
        "validate", help="Run no-inference UtterPlan v5 and profile preflight"
    )
    validate.add_argument("plan", type=Path)
    validate.add_argument(
        "--voice", "--profile", dest="voice", required=True, type=Path
    )
    validate.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate":
            from .render_v5 import validate_utterplan

            payload = validate_utterplan(args.plan, profile=args.voice)
            if args.json:
                print(json.dumps(payload, indent=2))
            else:
                engine_names = ", ".join(item["id"] for item in payload["engines"])
                print(
                    f"Valid UtterPlan v5 {payload['plan_id']}: "
                    f"{payload['selection']['selected_speech_count']} speech operations; "
                    f"engines: {engine_names or 'none'}"
                )
            return 0
        if args.command == "render":
            selected = args.select.split(",") if args.select is not None else None
            if args.voice is not None:
                from .render_v5 import render_utterplan_v5

                result = render_utterplan_v5(
                    args.plan,
                    profile=args.voice,
                    output_dir=args.out_dir,
                    cache_dir=args.cache_dir,
                    take=args.take,
                    select=selected,
                    force=args.force,
                    fail_fast=not args.continue_on_error,
                )
            else:
                result = render_plan(
                    args.plan,
                    profile=args.profile,
                    output_dir=args.out_dir,
                    cache_dir=args.cache_dir,
                    take=args.take,
                    select=selected,
                    force=args.force,
                    fail_fast=not args.continue_on_error,
                )
            payload = result.as_dict()
            if args.json:
                print(json.dumps(payload, indent=2))
            else:
                print(
                    f"Rendered {payload['rendered']}, reused {payload['reused']}, "
                    f"failed {payload['failed']} of {payload['selected']} → {result.manifest_path}"
                )
            return 1 if result.failed_count else 0
        if args.command == "engines":
            if args.action == "list":
                if args.engine is not None:
                    raise VoiceRenderError("engines list does not accept an engine ID")
                engines = list_engines()
                if args.json:
                    print(json.dumps(engines, indent=2))
                else:
                    for engine in engines:
                        print(
                            f"{engine['id']}: {engine['installed_version'] or 'not installed'} "
                            f"(package: {engine['package']})"
                        )
                return 0
            if args.engine is None:
                raise VoiceRenderError("engines inspect requires an engine ID")
            status = engine_status(args.engine)
            if args.json:
                print(json.dumps(status, indent=2, default=str))
            else:
                label = "compatible" if status["compatible"] else status["status"]
                print(f"{status['engine']}: {label}")
                version = status["distribution_version"] or "(not installed)"
                print(f"  package: {status['package']} {version}")
                if status["missing_symbols"]:
                    print(f"  missing symbols: {', '.join(status['missing_symbols'])}")
                if status["missing_methods"]:
                    print(f"  missing methods: {', '.join(status['missing_methods'])}")
            return 0 if status["compatible"] else 1
        if args.command == "voices":
            found = discover_targets(
                args.engine,
                language=args.lang,
                offline=args.offline,
                refresh=args.refresh,
            )
            targets_payload = [
                {
                    "engine": target.engine,
                    "id": target.id,
                    "display_name": target.display_name,
                    "languages": target.languages,
                    "status": target.status,
                    "runtime_available": target.runtime_available,
                    "sample_rate": target.sample_rate,
                    "voices": target.voices,
                    "speakers": target.speakers,
                    "voice_details": [
                        {
                            "id": voice.id,
                            "gender": voice.gender,
                            "language": voice.language,
                            "locale": voice.locale,
                            "language_label": voice.language_label,
                            "languages": voice.languages,
                        }
                        for voice in target.voice_details
                    ],
                    "default_voice": target.default_voice,
                    "qualities": target.qualities,
                    "aliases": target.aliases,
                    "capabilities": sorted(target.capabilities),
                    "metadata": dict(target.metadata),
                }
                for target in found
            ]
            if args.json:
                print(json.dumps(targets_payload, indent=2, default=str))
            else:
                for target in found:
                    names = ", ".join(target.voices)
                    print(
                        f"{target.engine}:{target.id}"
                        + (f" ({names})" if names else "")
                        + f" [{target.status}]"
                    )
            return 0
        if args.command == "profiles":
            profile = VoiceProfile.load(args.path)
            for role, binding in profile.roles.items():
                options = dict(binding.options)
                if binding.reference_audio is not None:
                    options["voice_source"] = {
                        "kind": "reference",
                        "value": str(binding.reference_audio),
                        "sha256": hashlib.sha256(
                            binding.reference_audio.read_bytes()
                        ).hexdigest(),
                    }
                _, diagnostics = resolve_voice(
                    EngineRequest(
                        engine=binding.engine,
                        target_id=binding.target_id,
                        language=profile.default_language or "en-us",
                        voice=binding.voice,
                        speaker=binding.speaker,
                        options=options,
                    )
                )
                probe = engine_status(binding.engine)
                if not probe["compatible"]:
                    raise VoiceRenderError(
                        f"Role {role!r} has an incompatible engine API: {probe['status']}"
                    )
                errors = [item for item in diagnostics if item.severity == "error"]
                if errors:
                    details = "; ".join(
                        f"{item.code}: {item.message}" for item in errors
                    )
                    raise VoiceRenderError(f"Role {role!r} is invalid: {details}")
            payload = {
                "valid": True,
                "name": profile.name,
                "roles": list(profile.roles),
            }
            print(
                json.dumps(payload, indent=2)
                if args.json
                else f"Valid profile {profile.name}: {len(profile.roles)} roles"
            )
            return 0
        if args.command == "inspect":
            input_path = args.manifest.expanduser().absolute()
            raw_document = json.loads(input_path.read_text(encoding="utf-8"))
            if not isinstance(raw_document, dict):
                raise VoiceRenderError("Manifest root must be a JSON object")
            if raw_document.get("schema") == MANIFEST_SCHEMA_V2:
                # Manifest v2 verification is mandatory for inspection; --verify
                # makes that behavior explicit in scripts and documentation.
                document = read_manifest(input_path, verify_audio=True)
                findings = [
                    {
                        "locator": entry["source"]["locator"],
                        "status": entry["status"],
                        "wav_valid": entry["status"] in {"rendered", "reused"},
                    }
                    for entry in document["segments"]
                ]
                payload = {
                    "schema": MANIFEST_SCHEMA_V2,
                    "manifest_id": document["manifest_id"],
                    "complete": document["complete"],
                    "valid_wavs": sum(item["wav_valid"] for item in findings),
                    "selected": len(findings),
                    "segments": findings,
                }
            else:
                manifest_path = input_path.resolve()
                document = raw_document
                if document.get("schema") != SCHEMA_MANIFEST:
                    raise VoiceRenderError(
                        f"Unsupported manifest schema: {document.get('schema')}"
                    )
                findings = []
                for entry in document.get("segments", []):
                    wav = entry.get("wav")
                    stored = Path(entry["path"])
                    if stored.is_absolute() or ".." in stored.parts:
                        valid = False
                    else:
                        valid = bool(
                            isinstance(wav, dict)
                            and verified_wav(manifest_path.parent / stored, wav)
                        )
                    findings.append(
                        {
                            "scope_id": entry["scope_id"],
                            "segment_id": entry["segment_id"],
                            "status": entry["status"],
                            "wav_valid": valid,
                        }
                    )
                payload = {
                    "schema": SCHEMA_MANIFEST,
                    "complete": document.get("complete"),
                    "valid_wavs": sum(item["wav_valid"] for item in findings),
                    "selected": len(findings),
                    "segments": findings,
                }
            if args.json:
                print(json.dumps(payload, indent=2))
            else:
                print(
                    f"{payload['valid_wavs']}/{payload['selected']} WAVs verified; "
                    f"complete: {payload['complete']}"
                )
            return (
                0
                if payload["complete"] and payload["valid_wavs"] == payload["selected"]
                else 1
            )
    except (VoiceRenderError, OSError, ValueError, KeyError) as exc:
        print(f"voicerender: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    sys.exit(main())
