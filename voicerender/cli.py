"""Command-line interface for independent VoiceRender usage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .engines import get_engine, list_engines
from .errors import VoiceRenderError
from .models import SCHEMA_MANIFEST, VoicePlan, VoiceProfile
from .render import render_plan
from .storage import verified_wav


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voicerender", description="Render voice plans to resumable WAV bundles"
    )
    parser.add_argument(
        "--version", action="version", version=f"voicerender {__version__}"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    render = commands.add_parser(
        "render", help="Synthesize selected logical speech segments"
    )
    render.add_argument("plan", type=Path)
    render.add_argument(
        "--profile", "--voices", dest="profile", required=True, type=Path
    )
    render.add_argument("--out-dir", required=True, type=Path)
    render.add_argument("--cache-dir", type=Path)
    select = render.add_mutually_exclusive_group()
    select.add_argument("--take", type=int)
    select.add_argument(
        "--select", metavar="ID[,ID...]", help="Speech IDs, optionally scope:ID"
    )
    render.add_argument(
        "--force", action="store_true", help="Regenerate selected audio, bypass cache"
    )
    render.add_argument("--continue-on-error", action="store_true")
    render.add_argument("--json", action="store_true")
    engines = commands.add_parser("engines", help="Inspect installed synthesis engines")
    engines.add_argument("action", choices=["list"])
    engines.add_argument("--json", action="store_true")
    voices = commands.add_parser(
        "voices", help="Discover backend voice and model choices"
    )
    voices.add_argument("action", choices=["list"])
    voices.add_argument("--engine", required=True, choices=["piper", "pocket"])
    voices.add_argument("--lang")
    voices.add_argument("--offline", action="store_true")
    voices.add_argument("--json", action="store_true")
    profiles = commands.add_parser("profiles", help="Validate a voice profile")
    profiles.add_argument("action", choices=["validate"])
    profiles.add_argument("path", type=Path)
    profiles.add_argument("--json", action="store_true")
    inspect = commands.add_parser(
        "inspect", help="Inspect a persisted manifest and verify WAVs"
    )
    inspect.add_argument("manifest", type=Path)
    inspect.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "render":
            selected = args.select.split(",") if args.select is not None else None
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
            engines = list_engines()
            if args.json:
                print(json.dumps(engines, indent=2))
            else:
                for engine in engines:
                    print(
                        f"{engine['id']}: {engine['installed_version'] or 'not installed'} "
                        f"(reference audio: {'yes' if engine['reference_audio'] else 'no'})"
                    )
            return 0
        if args.command == "voices":
            if args.engine == "piper":
                from .engines.piper import discover_voices
            else:
                from .engines.pocket import discover_voices
            found = discover_voices(language=args.lang, offline=args.offline)
            if args.json:
                print(json.dumps(found, indent=2))
            else:
                for item in found:
                    names = ", ".join(item.get("voices", ()))
                    print(
                        f"{item['engine']}:{item['target_id']}"
                        + (f" ({names})" if names else "")
                    )
            return 0
        if args.command == "profiles":
            profile = VoiceProfile.load(args.path)
            for binding in profile.roles.values():
                get_engine(binding.engine).validate(binding)
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
            manifest_path = args.manifest.resolve()
            document = json.loads(manifest_path.read_text(encoding="utf-8"))
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
                    f"{payload['valid_wavs']}/{payload['selected']} WAVs verified; complete: {payload['complete']}"
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
