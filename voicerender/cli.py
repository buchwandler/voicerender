"""Command-line interface for shared RenderPlan workflows."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TextIO, cast

import tomlkit
from renderplan import RoleProfile, load_manifest, load_plan

from . import __version__
from .bundle import BundleRenderResult, render_bundle
from .bundle_batch import (
    BundleBatchRenderResult,
    render_bundle_batch,
)
from .bundle_inspect import (
    BundleInspectResult,
    _inspection_binding,
    inspect_bundle,
)
from .bundle_roles import BundleRolesError, BundleRolesResult, inspect_bundle_roles
from .engines.options import engine_option_to_dict, engine_options
from .engines.qwen3synth import QWEN3_QVOICE_FORMAT
from .engines.registry import (
    engine_status,
    get_adapter,
    list_engines,
)
from .errors import (
    ArtifactError,
    EngineApiIncompatibleError,
    EngineBackendError,
    EngineError,
    EngineUnavailableError,
    InvalidEngineOptionError,
    InvalidEngineVoiceError,
    ProfileError,
    UnsupportedSynthesisFeatureError,
    VoiceRenderError,
)
from .events import RenderEvent, RenderEventKind
from .models_api import (
    ModelInstallRequest,
    ModelInstallResult,
    ModelRemoveRequest,
    ModelRemoveResult,
    ModelVerifyResult,
    install_model,
    list_models,
    model_artifact_to_dict,
    model_result_to_dict,
    model_status,
    remove_model,
    verify_model,
)
from .preflight import (
    preflight_check_details,
    preflight_failure_groups,
    preflight_failure_message,
    preflight_render_plan,
)
from .profile_roles import (
    list_profile_roles,
    remove_profile_role,
    set_default_profile_role,
    set_profile_role,
)
from .renderplan_render import (
    RENDER_DIAGNOSTICS_FILENAME,
    RenderPlanResult,
    render_renderplan,
)
from .renderprofile import (
    create_render_profile,
    load_render_profile,
    refresh_render_profile,
    reidentify_render_profile,
    save_render_profile,
    validate_render_profile,
)
from .role_bindings import role_voice_selector
from .runtime import (
    OnnxRuntimeStatus,
    clear_onnx_runtime_providers,
    onnx_runtime_status,
    set_onnx_runtime_providers,
)
from .storage import install_file_no_replace, wav_info
from .variants import BundleVariantsResult, list_bundle_variants
from .voice_clone import VoiceCloneRequest, clone_voice_profile
from .voice_design import Qwen3VoiceDesignRecipe, design_qwen3_reference
from .voice_source_codec import load_voice_source, save_voice_source
from .voice_sources import (
    create_voice_source,
    local_prepared_voice_source,
    local_reference_source,
    managed_reference_source,
    normalize_voice_source,
    resolve_reference_audio,
    validate_voice_source,
    voice_source_identity,
)
from .voices_api import (
    inspect_voice,
    list_voices,
    voice_descriptor_to_dict,
)


def _add_progress_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show live progress on stderr (default: auto for interactive human output)",
    )


def _add_profile_role_binding_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("profile", type=Path)
    parser.add_argument("role")
    parser.add_argument("--engine", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--language", required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--voice")
    source.add_argument(
        "--voicefolio-ref",
        help="Pinned VoiceFolio reference; catalog transcript metadata is used when available",
    )
    source.add_argument("--reference-wav", type=Path)
    source.add_argument("--prepared-voice", type=Path)
    source.add_argument("--voice-source", type=Path)
    parser.add_argument(
        "--transcript",
        help="Exact reference transcript; fallback when VoiceFolio has no catalog transcript",
    )
    parser.add_argument("--speaker")
    parser.add_argument(
        "--short-sentence",
        choices=("off", "wrap", "phrase", "randomized-phrase"),
        help="Explicitly configure Kokoro short-sentence processing",
    )
    parser.add_argument(
        "--option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Engine option; VALUE is parsed as JSON when possible (repeatable)",
    )
    _add_pronunciation_flags(parser)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--make-default", action="store_true")
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("-o", "--output", type=Path)
    destination.add_argument("--in-place", action="store_true")
    parser.add_argument("--json", action="store_true")


def _add_pronunciation_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--lexicon",
        action="append",
        default=[],
        metavar="NAME",
        help="Select a named pronunciation lexicon in precedence order (repeatable)",
    )
    parser.add_argument(
        "--g2lex",
        action="append",
        default=[],
        type=Path,
        metavar="PATH",
        help="Use a local G2Lex file before named lexicons (repeatable)",
    )
    parser.add_argument(
        "--lexicon-data-policy",
        choices=("auto", "installed-only"),
        help="Provisioning policy for named Lexphon lexicons",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voicerender",
        description="Preflight and render shared RenderPlan speech bundles.",
    )
    parser.add_argument(
        "--version", action="version", version=f"voicerender {__version__}"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    runtime = commands.add_parser(
        "runtime", help="Inspect and configure ONNX Runtime providers"
    )
    runtime_commands = runtime.add_subparsers(dest="runtime_command", required=True)
    for runtime_action in ("status", "list", "reset"):
        runtime_subcommand = runtime_commands.add_parser(
            runtime_action,
            help={
                "status": "Show ONNX Runtime distribution and provider policy status",
                "list": "List detected distributions and available providers",
                "reset": "Clear the saved ONNXVoice provider preference",
            }[runtime_action],
        )
        runtime_subcommand.add_argument("--json", action="store_true")
    runtime_use = runtime_commands.add_parser(
        "use", help="Save an ordered provider preference for future sessions"
    )
    runtime_use.add_argument(
        "providers", help="Provider policy, e.g. cpu, cuda, auto, or cuda,cpu"
    )
    runtime_use.add_argument("--json", action="store_true")
    engines = commands.add_parser("engines", help="List or inspect synthesis engines")
    engine_commands = engines.add_subparsers(dest="engines_command", required=True)
    engine_list = engine_commands.add_parser("list", help="List registered engines")
    engine_list.add_argument("--json", action="store_true")
    engine_inspect = engine_commands.add_parser("inspect", help="Inspect an engine")
    engine_inspect.add_argument("engine")
    engine_inspect.add_argument("--json", action="store_true")

    engine_options = engine_commands.add_parser(
        "options", help="List normalized engine option descriptors"
    )
    engine_options.add_argument("engine")
    engine_options.add_argument("--target")
    engine_options.add_argument("--json", action="store_true")

    models = commands.add_parser("models", help="Manage engine model artifacts")
    model_commands = models.add_subparsers(dest="models_command", required=True)
    model_list = model_commands.add_parser("list", help="List engine models")
    model_list.add_argument("--engine", required=True)
    model_list.add_argument("--lang")
    model_filter = model_list.add_mutually_exclusive_group()
    model_filter.add_argument("--installed", action="store_true")
    model_filter.add_argument("--available", action="store_true")
    model_list.add_argument("--offline", action="store_true")
    model_list.add_argument("--refresh", action="store_true")
    model_list.add_argument("--json", action="store_true")
    model_inspect = model_commands.add_parser("inspect", help="Inspect an engine model")
    model_inspect.add_argument("target")
    model_inspect.add_argument("--engine", required=True)
    model_inspect.add_argument("--quality")
    model_inspect.add_argument("--json", action="store_true")
    model_install = model_commands.add_parser("install", help="Install an engine model")
    model_install.add_argument("target")
    model_install.add_argument("--engine", required=True)
    model_install.add_argument("--quality")
    model_install.add_argument("--source")
    model_install.add_argument("--cache-dir", type=Path)
    model_install.add_argument("--force", action="store_true")
    model_install.add_argument("--json", action="store_true")
    model_remove = model_commands.add_parser("remove", help="Remove an engine model")
    model_remove.add_argument("target")
    model_remove.add_argument("--engine", required=True)
    model_remove.add_argument("--quality")
    model_remove.add_argument("--cache-dir", type=Path)
    model_remove.add_argument("--yes", action="store_true")
    model_remove.add_argument("--json", action="store_true")
    model_verify = model_commands.add_parser("verify", help="Verify an installed model")
    model_verify.add_argument("target")
    model_verify.add_argument("--engine", required=True)
    model_verify.add_argument("--quality")
    model_verify.add_argument("--cache-dir", type=Path)
    model_verify.add_argument("--json", action="store_true")

    voices = commands.add_parser("voices", help="Discover engine targets")
    voice_commands = voices.add_subparsers(dest="voices_command", required=True)
    voice_list = voice_commands.add_parser("list", help="List targets for an engine")
    voice_list.add_argument("--engine")
    voice_list.add_argument("--lang")
    voice_list.add_argument("--offline", action="store_true")
    voice_list.add_argument("--json", action="store_true")
    voice_list.add_argument("--target")
    voice_list.add_argument(
        "--kind",
        choices=("builtin", "speaker", "managed-reference", "reference-capable"),
    )
    voice_list.add_argument("--refresh", action="store_true")
    voice_inspect = voice_commands.add_parser(
        "inspect", help="Inspect a canonical voice selector"
    )
    voice_inspect.add_argument("selector")
    voice_inspect.add_argument("--lang")
    voice_inspect.add_argument("--json", action="store_true")

    lexicons = commands.add_parser(
        "lexicons", help="Discover and inspect pronunciation sources"
    )
    lexicon_commands = lexicons.add_subparsers(dest="lexicons_command", required=True)
    lexicon_list = lexicon_commands.add_parser(
        "list", help="List named pronunciation lexicons"
    )
    lexicon_inspect = lexicon_commands.add_parser(
        "inspect", help="Inspect a named pronunciation lexicon"
    )
    lexicon_inspect.add_argument("selector")
    for lexicon_command in (lexicon_list, lexicon_inspect):
        lexicon_command.add_argument("--engine", default="kokoro")
        lexicon_command.add_argument("--lang")
        lexicon_command.add_argument(
            "--target", help="Limit discovery to an engine model/target"
        )
        lexicon_command.add_argument("--offline", action="store_true")
        lexicon_command.add_argument("--refresh", action="store_true")
        lexicon_command.add_argument(
            "--preference",
            choices=("auto", "github", "huggingface", "upstream"),
            default="auto",
        )
        lexicon_command.add_argument("--json", action="store_true")
    lexicon_file = lexicon_commands.add_parser(
        "inspect-file",
        help="Inspect a local G2Lex manifest without loading its entries",
    )
    lexicon_file.add_argument("path", type=Path)
    lexicon_file.add_argument("--engine", default="kokoro")
    lexicon_file.add_argument("--json", action="store_true")
    profiles = commands.add_parser(
        "profile", help="Create, bind, and inspect RenderProfiles"
    )
    profile_commands = profiles.add_subparsers(dest="profile_command", required=True)
    profile_create = profile_commands.add_parser(
        "create",
        help="Create a RenderProfile with its first role binding",
        description=(
            "Create the first/default role binding of a complete RenderProfile cast; "
            "use `profile bind` to add more roles. VoiceFolio catalog transcripts "
            "are used automatically; pass --transcript when metadata is absent."
        ),
    )
    profile_create.add_argument("--engine", required=True)
    profile_create.add_argument("--target", required=True)
    profile_create.add_argument("--language", required=True)
    profile_create.add_argument("--role", default="narrator")
    voice_source_group = profile_create.add_mutually_exclusive_group()
    voice_source_group.add_argument("--voice")
    voice_source_group.add_argument(
        "--voicefolio-ref",
        help="Pinned VoiceFolio reference; catalog transcript metadata is used when available",
    )
    voice_source_group.add_argument("--reference-wav", type=Path)
    voice_source_group.add_argument("--prepared-voice", type=Path)
    voice_source_group.add_argument("--voice-source", type=Path)
    profile_create.add_argument(
        "--transcript",
        help="Exact reference transcript; fallback when VoiceFolio has no catalog transcript",
    )
    profile_create.add_argument("--speaker")
    profile_create.add_argument(
        "--short-sentence",
        choices=("off", "wrap", "phrase", "randomized-phrase"),
        help="Explicitly configure Kokoro short-sentence processing",
    )
    profile_create.add_argument(
        "--option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Engine option; VALUE is parsed as JSON when possible (repeatable)",
    )
    _add_pronunciation_flags(profile_create)
    profile_create.add_argument("--offline", action="store_true")
    profile_create.add_argument("--refresh", action="store_true")
    profile_create.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="output profile path (normalizes to the .renderprofile.toml suffix)",
    )
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

    profile_reidentify = profile_commands.add_parser(
        "reidentify", help="Repair a stale or missing RenderProfile content ID"
    )
    profile_reidentify.add_argument("profile", type=Path)
    reidentify_destination = profile_reidentify.add_mutually_exclusive_group(
        required=True
    )
    reidentify_destination.add_argument("-o", "--output", type=Path)
    reidentify_destination.add_argument("--in-place", action="store_true")
    profile_reidentify.add_argument("--json", action="store_true")

    profile_inspect = profile_commands.add_parser(
        "inspect", help="Inspect RenderProfile roles and policy"
    )
    profile_inspect.add_argument("profile", type=Path)
    profile_inspect.add_argument("--json", action="store_true")

    profile_show = profile_commands.add_parser(
        "show", help="Show a RenderProfile cast and its policy"
    )
    profile_show.add_argument("profile", type=Path)
    profile_show.add_argument("--json", action="store_true")

    profile_roles = profile_commands.add_parser(
        "role", help="Inspect or mutate RenderProfile roles"
    )
    role_commands = profile_roles.add_subparsers(
        dest="profile_role_command", required=True
    )
    role_list = role_commands.add_parser("list", help="List configured roles")
    role_list.add_argument("profile", type=Path)
    role_list.add_argument("--json", action="store_true")

    role_set = role_commands.add_parser("set", help="Add or replace a role binding")
    _add_profile_role_binding_arguments(role_set)

    profile_bind = profile_commands.add_parser(
        "bind", help="Add or replace a role binding in a RenderProfile"
    )
    _add_profile_role_binding_arguments(profile_bind)

    role_remove = role_commands.add_parser("remove", help="Remove a configured role")
    role_remove.add_argument("profile", type=Path)
    role_remove.add_argument("role")
    role_remove.add_argument("--new-default-role")
    role_remove_output = role_remove.add_mutually_exclusive_group(required=True)
    role_remove_output.add_argument("-o", "--output", type=Path)
    role_remove_output.add_argument("--in-place", action="store_true")
    role_remove.add_argument("--json", action="store_true")

    profile_default_role = profile_commands.add_parser(
        "default-role", help="Select the default profile role"
    )
    profile_default_role.add_argument("profile", type=Path)
    profile_default_role.add_argument("role")
    default_role_output = profile_default_role.add_mutually_exclusive_group(
        required=True
    )
    default_role_output.add_argument("-o", "--output", type=Path)
    default_role_output.add_argument("--in-place", action="store_true")
    profile_default_role.add_argument("--json", action="store_true")
    voice = commands.add_parser("voice", help="Generate or inspect voice assets")
    voice_commands = voice.add_subparsers(dest="voice_command", required=True)

    voice_design = voice_commands.add_parser(
        "design", help="Synthesize a persistent Qwen VoiceDesign reference WAV"
    )
    voice_design.add_argument(
        "--engine", choices=("qwen3", "qwen3synth"), default="qwen3"
    )
    voice_design.add_argument("--prompt", required=True)
    voice_design.add_argument("--text", required=True, help="Fixed reference utterance")
    voice_design.add_argument("--language", default="en")
    voice_design.add_argument("--quality", choices=("int4", "fp32"), default="int4")
    voice_design.add_argument("--provider", default="cpu")
    voice_design.add_argument("--seed", type=int)
    voice_design.add_argument("--temperature", type=float, default=0.9)
    voice_design.add_argument("--top-k", type=int, default=50)
    voice_design.add_argument("--repetition-penalty", type=float, default=1.05)
    voice_design.add_argument("--max-new-tokens", type=int, default=128)
    voice_design.add_argument("--cache-dir", type=Path)
    voice_design.add_argument("--catalog", type=Path)
    voice_design.add_argument("--output", required=True, type=Path)
    voice_design.add_argument("--source-output", type=Path)
    voice_design.add_argument("--force", action="store_true")
    voice_design.add_argument("--json", action="store_true")

    prepare_qwen = voice_commands.add_parser(
        "prepare-qwen", help="Prepare a pinned reference WAV as a Qwen .qvoice profile"
    )
    prepare_source = prepare_qwen.add_mutually_exclusive_group(required=True)
    prepare_source.add_argument("--reference", type=Path)
    prepare_source.add_argument("--voice-source", type=Path)
    prepare_qwen.add_argument("--transcript")
    prepare_qwen.add_argument("--language")
    prepare_qwen.add_argument("--target", default="base-0.6b")
    prepare_qwen.add_argument("--quality", choices=("int4", "fp32"), default="int4")
    prepare_qwen.add_argument("--provider", default="cpu")
    prepare_qwen.add_argument("--cache-dir", type=Path)
    prepare_qwen.add_argument("--catalog", type=Path)
    prepare_qwen.add_argument("--output", required=True, type=Path)
    prepare_qwen.add_argument("--force", action="store_true")
    prepare_qwen.add_argument("--json", action="store_true")

    source_command = voice_commands.add_parser(
        "source", help="Create or inspect reusable voice sources"
    )
    source_subcommands = source_command.add_subparsers(
        dest="voice_source_command", required=True
    )
    source_create = source_subcommands.add_parser(
        "create", help="Create a pinned voice-source sidecar"
    )
    source_input = source_create.add_mutually_exclusive_group(required=True)
    source_input.add_argument("--reference-wav", type=Path)
    source_input.add_argument("--voicefolio-ref")
    source_create.add_argument("--transcript")
    source_create.add_argument("--language")
    source_create.add_argument("--name")
    source_create.add_argument(
        "--metadata", action="append", default=[], metavar="KEY=VALUE"
    )
    source_create.add_argument("-o", "--output", required=True, type=Path)
    source_create.add_argument("--force", action="store_true")
    source_create.add_argument("--json", action="store_true")
    for action in ("inspect", "verify"):
        source_action = source_subcommands.add_parser(
            action, help=f"{action.title()} a voice-source sidecar"
        )
        source_action.add_argument("path", type=Path)
        source_action.add_argument("--json", action="store_true")

    voice_clone = voice_commands.add_parser(
        "clone", help="Bind a reference source to an engine profile"
    )
    voice_clone.add_argument("--engine", required=True)
    voice_clone.add_argument("--target", required=True)
    voice_clone.add_argument("--language", required=True)
    voice_clone.add_argument("--role", default="narrator")
    clone_source = voice_clone.add_mutually_exclusive_group(required=True)
    clone_source.add_argument("--reference-wav", type=Path)
    clone_source.add_argument("--voice-source", type=Path)
    clone_source.add_argument(
        "--voicefolio-ref",
        help="Pinned VoiceFolio reference; catalog transcript metadata is used when available",
    )
    voice_clone.add_argument(
        "--transcript",
        help="Exact reference transcript; fallback when VoiceFolio has no catalog transcript",
    )
    voice_clone.add_argument("--speaker")
    voice_clone.add_argument(
        "--option", action="append", default=[], metavar="KEY=VALUE"
    )
    voice_clone.add_argument("--offline", action="store_true")
    voice_clone.add_argument("--refresh", action="store_true")
    voice_clone.add_argument("-o", "--output", required=True, type=Path)
    voice_clone.add_argument("--force", action="store_true")
    voice_clone.add_argument("--json", action="store_true")

    voice_inspect = voice_commands.add_parser(
        "inspect", help="Inspect a voice-source sidecar or Qwen .qvoice profile"
    )
    voice_inspect.add_argument("path", type=Path)
    voice_inspect.add_argument("--json", action="store_true")

    preflight = commands.add_parser(
        "preflight", help="Measure a RenderPlan without synthesis"
    )
    preflight.add_argument("plan", type=Path)
    preflight.add_argument("-o", "--output", required=True, type=Path)
    preflight.add_argument("--json", action="store_true")
    _add_progress_option(preflight)

    preflight.add_argument(
        "--emphasis-policy",
        choices=("reject", "defer-to-compose"),
        default="reject",
        help="Reject emphasis by default, or defer it to a declared AudioCompose handler",
    )
    render = commands.add_parser(
        "render", help="Render a plan using a matching OK report"
    )
    render.add_argument("plan", type=Path)
    render.add_argument("--preflight", required=True, type=Path)
    render.add_argument("--out-dir", required=True, type=Path)
    render.add_argument("--cache-dir", type=Path)
    render.add_argument("--recheck-report", type=Path)
    render.add_argument("--force", action="store_true")
    render.add_argument(
        "--reset-stale",
        action="store_true",
        help="Reset output owned by a different RenderPlan while reusing cache entries",
    )
    render.add_argument(
        "--max-open-sessions",
        type=int,
        default=1,
        metavar="N",
        help="Maximum number of open engine sessions (default: 1)",
    )
    render.add_argument("--json", action="store_true")
    _add_progress_option(render)
    render.add_argument(
        "--emphasis-policy",
        choices=("reject", "defer-to-compose"),
        default="reject",
        help="Reject emphasis by default, or defer it to a declared AudioCompose handler",
    )

    bundle = commands.add_parser("bundle", help="Operate on an SSMD bundle")
    bundle_commands = bundle.add_subparsers(dest="bundle_command", required=True)
    bundle_render = bundle_commands.add_parser(
        "render",
        help="Render a book or document SSMD bundle",
        description=(
            "Render a book or document SSMD bundle from a path resolving to a "
            "canonical name.ssmd-bundle directory."
        ),
    )
    bundle_render.add_argument(
        "bundle",
        type=Path,
        help="path resolving to a canonical NAME.ssmd-bundle directory",
    )
    bundle_render.add_argument(
        "--variant",
        default=None,
        help=(
            "exact variant to render alone; omit for state-aware batch rendering "
            "of incomplete valid variants (batch --force also includes complete variants)"
        ),
    )
    bundle_render.add_argument("--chapters", default="all", metavar="SELECTOR")
    bundle_render.add_argument("--speech", default="all", metavar="SELECTOR")
    bundle_render.add_argument("--renderplan-root", type=Path)
    bundle_render.add_argument("--output-root", type=Path)
    bundle_render.add_argument("--cache-dir", type=Path)
    bundle_render.add_argument(
        "--force",
        action="store_true",
        help=(
            "batch: include completed valid variants; single variant: bypass verified "
            "output reuse/cache"
        ),
    )
    bundle_render.add_argument(
        "--reset-stale",
        action="store_true",
        help="Reset stale VoiceRender output while reusing cache entries",
    )
    bundle_render.add_argument(
        "--max-open-sessions",
        type=int,
        default=1,
        metavar="N",
        help="Maximum number of open engine sessions (default: 1)",
    )
    bundle_render.add_argument(
        "--emphasis-policy",
        choices=("reject", "defer-to-compose"),
        default="reject",
        help="Reject emphasis by default, or defer it to a declared AudioCompose handler",
    )
    bundle_render.add_argument("--fail-fast", action="store_true")
    bundle_render.add_argument("--json", action="store_true")
    _add_progress_option(bundle_render)

    bundle_variants = bundle_commands.add_parser(
        "variants",
        help="Discover existing RenderPlan variants and their render state",
        description=(
            "List direct RenderPlan variant directories for an SSMD bundle. "
            "The positional path must resolve to the bundle root; "
            "--renderplan-root only overrides where variant directories are read from."
        ),
    )
    bundle_variants.add_argument(
        "bundle",
        type=Path,
        help="path resolving to a canonical NAME.ssmd-bundle directory",
    )
    bundle_variants.add_argument("--renderplan-root", type=Path)
    bundle_variants.add_argument("--output-root", type=Path)
    bundle_variants.add_argument("--json", action="store_true")
    bundle_roles = bundle_commands.add_parser(
        "roles",
        help="List source-required roles and optionally compare a cast",
        description=(
            "List logical speech roles from the UtterPlan inventory and optionally "
            "compare them with an authored RenderProfile or exact RenderPlan variant."
        ),
    )
    bundle_roles.add_argument(
        "bundle",
        type=Path,
        help="path resolving to a canonical NAME.ssmd-bundle directory",
    )
    role_selection = bundle_roles.add_mutually_exclusive_group()
    role_selection.add_argument(
        "--profile", type=Path, help="authored mutable RenderProfile"
    )
    role_selection.add_argument(
        "--variant", help="exact variant's immutable RenderProfile snapshot"
    )
    bundle_roles.add_argument("--utterplan-root", type=Path)
    bundle_roles.add_argument("--renderplan-root", type=Path)
    bundle_roles.add_argument(
        "--check", action="store_true", help="exit 1 unless the selected cast is ready"
    )
    bundle_roles.add_argument("--json", action="store_true")

    bundle_inspect = bundle_commands.add_parser(
        "inspect",
        help="Inspect bundle roles, cast readiness, variants, and render state",
        description=(
            "Compose the UtterPlan source-role inventory and optional cast inspection "
            "with the existing RenderPlan variant and VoiceRender render-state listing."
        ),
    )
    bundle_inspect.add_argument(
        "bundle",
        type=Path,
        help="path resolving to a canonical NAME.ssmd-bundle directory",
    )
    inspect_selection = bundle_inspect.add_mutually_exclusive_group()
    inspect_selection.add_argument(
        "--profile", type=Path, help="authored mutable RenderProfile"
    )
    inspect_selection.add_argument(
        "--variant", help="exact variant's immutable RenderProfile snapshot"
    )
    bundle_inspect.add_argument(
        "--compare-profile",
        type=Path,
        help="read-only authored RenderProfile to compare with --variant",
    )
    bundle_inspect.add_argument("--utterplan-root", type=Path)
    bundle_inspect.add_argument("--renderplan-root", type=Path)
    bundle_inspect.add_argument("--output-root", type=Path)
    bundle_inspect.add_argument(
        "--check", action="store_true", help="exit 1 unless the selected cast is ready"
    )
    bundle_inspect.add_argument("--json", action="store_true")

    inspect = commands.add_parser(
        "inspect-manifest", help="Verify a render manifest and its WAVs"
    )
    inspect.add_argument("manifest", type=Path)
    inspect.add_argument("--json", action="store_true")

    return parser


_PROGRESS_TRANSITIONS = frozenset(
    {
        RenderEventKind.PREFLIGHT_STARTED,
        RenderEventKind.PREFLIGHT_FINISHED,
        RenderEventKind.RENDER_STARTED,
        RenderEventKind.RENDER_COMPLETED,
        RenderEventKind.BOOK_STARTED,
        RenderEventKind.BOOK_COMPLETED,
        RenderEventKind.CHAPTER_STARTED,
        RenderEventKind.CHAPTER_FINISHED,
        RenderEventKind.CHAPTER_REUSED,
        RenderEventKind.CHAPTER_SKIPPED,
        RenderEventKind.ENGINE_OPEN_STARTED,
        RenderEventKind.ENGINE_OPEN_FINISHED,
        RenderEventKind.SEGMENT_FAILED,
    }
)


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"
    whole_seconds = max(0, int(seconds))
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


class _ProgressPrinter:
    """Render typed progress events to stderr without touching stdout."""

    def __init__(
        self,
        stream: TextIO,
        *,
        is_tty: bool,
        clock: Callable[[], float] = time.monotonic,
        min_progress_delta: float = 0.05,
        min_interval_seconds: float = 30.0,
    ) -> None:
        self.stream = stream
        self.is_tty = is_tty
        self._clock = clock
        self._min_progress_delta = min_progress_delta
        self._min_interval_seconds = min_interval_seconds
        self._last_output_at: float | None = None
        self._last_output_progress: float | None = None
        self._last_phase: str | None = None
        self._last_chapter: tuple[str | None, int | None] | None = None
        self._line_open = False
        self._context: str | None = None

    def set_context(self, context: str | None) -> None:
        """Reset progress throttling and identify the next batch variant."""
        self.finish()
        self._context = context
        self._last_output_at = None
        self._last_output_progress = None
        self._last_phase = None
        self._last_chapter = None
        if context:
            self.stream.write(f"[{context}] starting\n")
            self.stream.flush()

    def __call__(self, event: RenderEvent) -> None:
        now = self._clock()
        phase = event.phase or event.scope
        chapter = (event.chapter_id, event.chapter_index)
        phase_changed = self._last_phase is not None and phase != self._last_phase
        chapter_changed = (
            chapter[1] is not None
            and self._last_chapter is not None
            and chapter != self._last_chapter
        )
        self._last_phase = phase or self._last_phase
        if chapter[1] is not None:
            self._last_chapter = chapter

        progress = (
            event.phase_progress
            if phase == "preflight" and event.phase_progress is not None
            else event.progress
        )
        if self.is_tty:
            self.stream.write("\r" + self._format_event(event, progress))
            self.stream.flush()
            self._line_open = True
            return

        advanced = (
            progress is not None
            and self._last_output_progress is not None
            and progress - self._last_output_progress >= self._min_progress_delta
        )
        timed_out = (
            self._last_output_at is not None
            and now - self._last_output_at >= self._min_interval_seconds
        )
        if not (
            self._last_output_at is None
            or event.kind in _PROGRESS_TRANSITIONS
            or phase_changed
            or chapter_changed
            or advanced
            or timed_out
        ):
            return
        self.stream.write(self._format_event(event, progress, non_tty=True) + "\n")
        self.stream.flush()
        self._last_output_at = now
        if progress is not None:
            self._last_output_progress = progress

    def _format_event(
        self,
        event: RenderEvent,
        progress: float | None,
        *,
        non_tty: bool = False,
    ) -> str:
        is_preflight = event.phase == "preflight" or event.scope == "preflight"
        percent = "--.-%" if progress is None else f"{progress * 100:.1f}%"
        completed = (
            event.phase_completed_count
            if is_preflight and event.phase_completed_count is not None
            else event.completed_count
        )
        total = (
            event.phase_total_count
            if is_preflight and event.phase_total_count is not None
            else event.total_count
        )
        count = f"{completed if completed is not None else '?'}"
        count += f"/{total if total is not None else '?'}"
        elapsed = (
            event.phase_elapsed_seconds
            if is_preflight and event.phase_elapsed_seconds is not None
            else event.elapsed_seconds
        )
        chapter = (
            f"chapter {event.chapter_index}/{event.chapter_count}"
            if event.chapter_index is not None and event.chapter_count is not None
            else None
        )
        if is_preflight:
            prefix = "preflight "
            details = [
                f"{prefix}[{progress * 100:5.1f}%] {count} speech"
                if progress is not None
                else f"{prefix}[--.-%] {count} speech"
            ]
        else:
            details = [
                f"[{progress * 100:5.1f}%] {count} speech"
                if progress is not None
                else f"[--.-%] {count} speech"
            ]
            if event.rendered_count is not None or event.reused_count is not None:
                details.append(
                    f"{event.rendered_count or 0} rendered, {event.reused_count or 0} reused"
                )
        status = self._event_status(event)
        if status is not None:
            details.append(status)
        details.append(f"elapsed {_format_duration(elapsed)}")
        if not is_preflight:
            details.append(f"ETA {_format_duration(event.eta_seconds)}")
        if chapter is not None:
            details.append(chapter)
        if non_tty:
            label = "preflight progress" if is_preflight else "progress"
            fields = [label, percent, count]
            if event.rendered_count is not None:
                fields.append(f"rendered={event.rendered_count}")
            if event.reused_count is not None:
                fields.append(f"reused={event.reused_count}")
            fields.append(f"elapsed={_format_duration(elapsed)}")
            if not is_preflight:
                fields.append(f"eta={_format_duration(event.eta_seconds)}")
            if status is not None:
                fields.append(status.replace(" ", "_"))
            if chapter is not None:
                fields.append(chapter.replace(" ", "_"))
            return (f"[{self._context}] " if self._context else "") + " ".join(fields)
        return (f"[{self._context}] " if self._context else "") + " | ".join(details)

    @staticmethod
    def _event_status(event: RenderEvent) -> str | None:
        if event.kind is RenderEventKind.ENGINE_OPEN_STARTED:
            return f"opening {event.engine or 'engine'}"
        if event.kind is RenderEventKind.ENGINE_OPEN_FINISHED:
            return f"opened {event.engine or 'engine'}"
        if event.kind is RenderEventKind.SEGMENT_STARTED and event.engine:
            return f"synthesizing {event.engine}"
        if event.kind is RenderEventKind.SEGMENT_FAILED:
            failure = f"failed {event.error_code or event.status or 'speech item'}"
            summary = _diagnostic_summary_text(event)
            return f"{failure}; {summary}" if summary else failure
        return None

    def finish(self) -> None:
        if self.is_tty and self._line_open:
            self.stream.write("\n")
            self.stream.flush()
            self._line_open = False


def _progress_callback(
    args: Any,
    *,
    stream: TextIO | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> _ProgressPrinter | None:
    output = sys.stderr if stream is None else stream
    is_tty = output.isatty()
    enabled = args.progress if args.progress is not None else is_tty and not args.json
    if not enabled:
        return None
    return _ProgressPrinter(output, is_tty=is_tty, clock=clock)


def _json_options(entries: list[str], *, label: str = "--option") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for entry in entries:
        key, separator, value = entry.partition("=")
        if not separator or not key.strip():
            raise ValueError(f"invalid {label} {entry!r}; expected KEY=VALUE")
        key = key.strip()
        try:
            result[key] = json.loads(value)
        except json.JSONDecodeError:
            result[key] = value
    return result


def _inspection_to_dict(inspection: Any) -> dict[str, Any]:
    to_dict = getattr(inspection, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
    elif isinstance(inspection, Mapping):
        value = dict(inspection)
    else:
        raise TypeError("Engine returned an invalid G2Lex inspection result")
    if not isinstance(value, Mapping):
        raise TypeError("Engine returned an invalid G2Lex inspection result")
    return dict(value)


def _pin_g2lex_files(adapter: Any, files: Any, language: str) -> list[dict[str, str]]:
    if isinstance(files, (str, Path, Mapping)):
        entries = (files,)
    else:
        try:
            entries = tuple(files)
        except TypeError as exc:
            raise ValueError(
                "g2lex_files must be a sequence of file paths or tables"
            ) from exc
    pinned: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    requested_language = language.strip().casefold().replace("_", "-")
    inspector = getattr(adapter, "inspect_g2lex_file", None)
    if not callable(inspector):
        raise TypeError(
            f"Engine {adapter.id!r} does not support local G2Lex inspection"
        )
    for entry in entries:
        if isinstance(entry, Mapping):
            raw_path = entry.get("path")
            expected_sha256 = entry.get("sha256")
            expected_logical_sha256 = entry.get("logical_sha256")
        else:
            raw_path = entry
            expected_sha256 = None
            expected_logical_sha256 = None
        if not isinstance(raw_path, (str, Path)) or not str(raw_path).strip():
            raise ValueError("Each G2Lex file reference requires a non-empty path")
        path = Path(raw_path).expanduser().resolve(strict=True)
        path_key = str(path)
        if path_key in seen_paths:
            raise ValueError(f"Duplicate G2Lex file path: {path}")
        seen_paths.add(path_key)
        details = _inspection_to_dict(inspector(str(path)))
        if details.get("valid") is False:
            raise ValueError(
                str(details.get("reason") or f"G2Lex file {path} is invalid")
            )
        if details.get("compatible") is False:
            raise ValueError(
                str(details.get("reason") or f"G2Lex file {path} is incompatible")
            )
        metadata = details.get("metadata")
        if not isinstance(metadata, Mapping):
            raise TypeError(f"G2Lex inspection for {path} has no metadata table")
        actual_sha256 = details.get("sha256")
        actual_logical_sha256 = details.get("logical_sha256")
        if not isinstance(actual_sha256, str) or not actual_sha256:
            raise ValueError(f"G2Lex inspection for {path} has no content SHA-256")
        if expected_sha256 is not None and _hash_key(expected_sha256) != _hash_key(
            actual_sha256
        ):
            raise ValueError(f"G2Lex file changed since it was pinned: {path}")
        if expected_logical_sha256 is not None and _hash_key(
            expected_logical_sha256
        ) != _hash_key(actual_logical_sha256 or ""):
            raise ValueError(
                f"G2Lex logical content changed since it was pinned: {path}"
            )
        file_language = metadata.get("language") or metadata.get("locale")
        if not isinstance(file_language, str) or not file_language.strip():
            raise ValueError(f"G2Lex file {path} has no language metadata")
        normalized_file_language = file_language.strip().casefold().replace("_", "-")
        if normalized_file_language != requested_language:
            raise ValueError(
                f"G2Lex file {path} is for {normalized_file_language!r}, "
                f"not profile language {requested_language!r}"
            )
        encoding = metadata.get("phoneme_encoding")
        if not isinstance(encoding, str) or not encoding.strip():
            raise ValueError(f"G2Lex file {path} has no phoneme_encoding metadata")
        ref = {"path": str(path), "sha256": actual_sha256}
        if actual_logical_sha256 is not None:
            ref["logical_sha256"] = str(actual_logical_sha256)
        ref["language"] = normalized_file_language
        ref["phoneme_encoding"] = encoding.strip().casefold()
        pinned.append(ref)
    return pinned


def _hash_key(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().casefold().removeprefix("sha256:")


def _profile_pronunciation_options(
    engine: str,
    language: str,
    options: Mapping[str, Any],
    *,
    lexicon_names: list[str],
    g2lex_paths: list[Path],
    lexicon_data_policy: str | None,
) -> dict[str, Any]:
    result = dict(options)
    if lexicon_names and "lexicons" in result:
        raise ValueError("--lexicon conflicts with generic --option lexicons=...")
    if g2lex_paths and "g2lex_files" in result:
        raise ValueError("--g2lex conflicts with generic --option g2lex_files=...")
    if lexicon_data_policy is not None and "lexicon_data_policy" in result:
        raise ValueError(
            "--lexicon-data-policy conflicts with generic --option lexicon_data_policy=..."
        )
    normalized_names = [name.strip() for name in lexicon_names]
    if any(not name for name in normalized_names):
        raise ValueError("--lexicon values must not be empty")
    seen_names: set[str] = set()
    for name in normalized_names:
        key = name.casefold()
        if key in seen_names:
            raise ValueError(f"Duplicate --lexicon selector: {name}")
        seen_names.add(key)
    if normalized_names:
        result["lexicons"] = normalized_names
    if lexicon_data_policy is not None:
        result["lexicon_data_policy"] = lexicon_data_policy

    raw_files: Any = g2lex_paths if g2lex_paths else result.get("g2lex_files")
    if raw_files is not None:
        adapter = get_adapter(engine)
        if not adapter.capabilities().supports_lexicons:
            raise ValueError(
                f"Engine {engine!r} does not support pronunciation lexicons"
            )
        result["g2lex_files"] = _pin_g2lex_files(adapter, raw_files, language)
    elif normalized_names or lexicon_data_policy is not None:
        adapter = get_adapter(engine)
        if not adapter.capabilities().supports_lexicons:
            raise ValueError(
                f"Engine {engine!r} does not support pronunciation lexicons"
            )
    return result


def _profile_short_sentence_option(
    engine: str,
    options: Mapping[str, Any],
    mode: str | None,
) -> dict[str, Any]:
    """Merge the Kokoro convenience mode into its canonical engine option."""
    result = dict(options)
    if mode is None:
        return result
    if engine.strip().casefold() != "kokoro":
        raise InvalidEngineOptionError(
            "--short-sentence is only supported for Kokoro profiles",
            code="kokoro.short_sentence_engine_invalid",
            engine=engine,
        )
    if "short_sentence" in result:
        raise InvalidEngineOptionError(
            "--short-sentence conflicts with --option short_sentence=...",
            code="kokoro.short_sentence_option_conflict",
            engine=engine,
        )
    result["short_sentence"] = mode
    return result


def _pronunciation_stack_for_binding(
    binding: Mapping[str, Any], language: str
) -> dict[str, Any] | None:
    renderer = binding.get("renderer")
    if not isinstance(renderer, str):
        return None
    try:
        adapter = get_adapter(renderer)
    except (ImportError, ProfileError):
        return None
    reader = getattr(adapter, "pronunciation_stack", None)
    if not callable(reader):
        return None
    options = binding.get("options", {})
    if not isinstance(options, Mapping):
        return {
            "engine": renderer,
            "language": language,
            "error": "invalid options table",
        }
    try:
        return dict(
            reader(
                language=language,
                target_id=str(binding.get("target_id", "unknown")),
                options=options,
            )
        )
    except (
        AttributeError,
        EngineError,
        ImportError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        return {"engine": renderer, "language": language, "error": str(exc)}


def _pronunciation_stacks_for_plan(plan: Any) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for item in getattr(plan, "speech", ()) or ():
        stack = _pronunciation_stack_for_binding(item.binding, item.language)
        if stack is None:
            continue
        key = json.dumps(
            {"binding": item.binding, "language": item.language},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        record = grouped.setdefault(
            key,
            {
                "language": item.language,
                "stack": stack,
                "segments": [],
            },
        )
        record["segments"].append(
            {
                "number": item.number,
                "segment_id": item.segment_id or item.computed_id(),
            }
        )
    return list(grouped.values())


def _print_pronunciation_stacks(records: list[dict[str, Any]]) -> None:
    for record in records:
        stack = record.get("stack", {})
        if not isinstance(stack, Mapping):
            continue
        segment_numbers = ", ".join(
            str(item.get("number")) for item in record.get("segments", ())
        )
        precedence = stack.get("precedence", ())
        label = " > ".join(str(item) for item in precedence) or "unresolved"
        print(
            f"Pronunciation stack (segments {segment_numbers or 'unknown'}, "
            f"{stack.get('engine', 'unknown')} {record.get('language', 'unknown')}): {label}"
        )
        for source in stack.get("local_g2lex", ()):
            if isinstance(source, Mapping):
                print(
                    f"  G2Lex: {source.get('path', 'unknown')} "
                    f"[{source.get('status', 'unknown')}] "
                    f"sha256={source.get('sha256') or source.get('current_sha256') or 'unpinned'}"
                )
        if stack.get("error"):
            print(f"  Pronunciation inspection error: {stack['error']}")


def _profile_role_items(profile: Any) -> list[dict[str, Any]]:
    return [
        {
            "role": role_name,
            "default": role_name == profile.default_role,
            "language": role.language,
            "binding": _inspection_binding(role.binding),
            "pronunciation_stack": _pronunciation_stack_for_binding(
                role.binding, role.language
            ),
        }
        for role_name, role in sorted(list_profile_roles(profile).items())
    ]


def _kokoro_short_sentence_lines(options: Any) -> tuple[str, ...]:
    if not isinstance(options, Mapping) or "short_sentence" not in options:
        return ("short sentence: off (implicit)",)

    setting = options["short_sentence"]
    if setting == "off" or setting is None:
        return ("short sentence: off (explicit)",)
    if isinstance(setting, Mapping):
        if setting.get("enabled", True) is False:
            return ("short sentence: off (explicit)",)
        mode = setting.get("resolve_mode", "randomized-phrase")
        threshold = setting.get("min_phoneme_length", 30)
    elif isinstance(setting, str):
        mode = setting
        threshold = 30
    else:
        return ("short sentence: configured (invalid)",)

    lines = [f"short sentence: {mode}"]
    if isinstance(threshold, int) and not isinstance(threshold, bool):
        lines.append(f"threshold: <{threshold} phonemes")
    lines.append("enabled: yes")
    return tuple(lines)


def _print_profile_role_items(
    roles: list[dict[str, Any]], *, show_engine_details: bool = False
) -> None:
    print(
        "  ROLE       LANGUAGE  ENGINE   TARGET             VOICE/SPEAKER/SOURCE / ENROLLMENT"
    )
    for item in roles:
        marker = "*" if item["default"] else " "
        binding = item["binding"]
        engine = _display_cli_value(binding.get("renderer", "unknown"))
        target = _display_cli_value(binding.get("target_id", "unknown"))
        language = _display_cli_value(item["language"])
        selector = role_voice_selector(
            RoleProfile(language=item["language"], binding=binding)
        )
        options = binding.get("options", {})
        if isinstance(options, Mapping):
            source = options.get("voice_source")
            if (
                isinstance(source, Mapping)
                and source.get("kind") == "managed_reference"
            ):
                ref = source.get("ref")
                if isinstance(ref, str) and ref:
                    selector += f":{ref}"
            enrollment = options.get("voice_enrollment")
            if isinstance(enrollment, Mapping) and enrollment.get("engine"):
                selector += f" enrollment:{enrollment['engine']}"
        print(
            f"{marker} {item['role']:<10} {language:<8} {engine:<7} "
            f"{target:<18} {selector}"
        )
        if show_engine_details and binding.get("renderer") == "kokoro":
            print("    Kokoro")
            for line in _kokoro_short_sentence_lines(options):
                print(f"      {line}")
        pronunciation_stack = item.get("pronunciation_stack")
        if isinstance(pronunciation_stack, Mapping):
            precedence = pronunciation_stack.get("precedence", ())
            print(
                "    Pronunciation stack: "
                + (" > ".join(str(value) for value in precedence) or "unresolved")
            )
            for source in pronunciation_stack.get("local_g2lex", ()):
                if isinstance(source, Mapping):
                    print(
                        f"      G2Lex: {source.get('path', 'unknown')} "
                        f"[{source.get('status', 'unknown')}] "
                        f"sha256={source.get('sha256') or source.get('current_sha256') or 'unpinned'}"
                    )
            if pronunciation_stack.get("error"):
                print(f"      Inspection error: {pronunciation_stack['error']}")


def _display_cli_value(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _profile_role_voice_source(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.voicefolio_ref:
        return managed_reference_source(args.voicefolio_ref)
    if args.reference_wav is not None:
        return local_reference_source(args.reference_wav)
    if args.prepared_voice is not None:
        if get_adapter(args.engine).id != "qwen3":
            raise InvalidEngineOptionError(
                "--prepared-voice is only supported for Qwen3 profiles",
                code="qwen3.prepared_voice_engine_invalid",
            )
        source, _ = _prepared_voice_source(args.prepared_voice)
        return source
    if args.voice_source is not None:
        return dict(load_voice_source(args.voice_source))
    return None


def normalize_renderprofile_output_path(path: Path) -> Path:
    """Return the CLI profile-create destination with its canonical suffix."""
    suffix = ".renderprofile.toml"
    name = path.name
    if name.casefold().endswith(suffix):
        return path
    if name.casefold().endswith(".toml"):
        name = name[: -len(".toml")]
    return path.with_name(f"{name}{suffix}")


def _profile_mutation_destination(args: argparse.Namespace) -> Path:
    if args.in_place:
        return cast(Path, args.profile)
    if args.output.resolve() == args.profile.resolve():
        raise ValueError("profile output must not overwrite its source; use --in-place")
    return cast(Path, args.output)


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


def _preflight_payload(
    report: Any, *, pronunciation_stacks: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    counts = {
        status: sum(check.status == status for check in report.checks)
        for status in ("ok", "too_long", "error")
    }
    payload = {
        "render_plan_id": report.render_plan_id,
        "render_profile_id": report.render_profile_id,
        "status": report.status,
        "checked": len(report.checks),
        "ok": counts["ok"],
        "replan": len(report.replan),
        "reject": counts["error"],
        "checks": counts,
        "failure_groups": list(preflight_failure_groups(report)),
        "check_details": preflight_check_details(report),
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
        "error": (preflight_failure_message(report) if report.status != "ok" else None),
    }
    if pronunciation_stacks is not None:
        payload["pronunciation_stacks"] = pronunciation_stacks
    return payload


def _sopro_unknown_fit_summary(report: Any) -> tuple[str, ...]:
    detail = (
        "Engine measured the exact request size but did not provide an "
        "authoritative maximum/fit result"
    )
    checks = tuple(
        check
        for check in report.checks
        if check.source == "soprosynth.measure_request"
        and check.status == "error"
        and check.metric == "model_tokens"
        and check.measured is not None
        and check.maximum is None
        and check.detail == detail
    )
    if len(checks) < 2:
        return ()
    count = len(checks)
    return (
        f"  sopro measurement: {count}/{count} requests have exact model-token counts",
        "  fit authority: missing; maximum: not reported",
        "  source: soprosynth.measure_request",
        "  action: update SoproSynth or verify its request-measurement contract",
    )


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
        "diagnostics": str(result.manifest_path.parent / RENDER_DIAGNOSTICS_FILENAME),
        "pending_composition": result.pending_composition,
        "error_code": result.error_code,
        "error_segment_number": result.error_segment_number,
        "error_segment_id": result.error_segment_id,
        "error_engine": result.error_engine,
        "execution_fingerprint": list(result.execution_fingerprint),
        "fresh_preflight": (
            _preflight_payload(result.fresh_preflight)
            if result.fresh_preflight is not None
            else None
        ),
        "error": result.error,
    }


def _print_bundle_variants(result: BundleVariantsResult) -> None:
    print("VoiceRender bundle variants")
    print(f"  bundle:       {result.bundle}")
    print(f"  kind:         {result.kind}")
    print(f"  renderplans:  {result.renderplan_root}")
    print(f"  output:       {result.output_root}")
    print(f"  variants:     {len(result.variants)}")
    if not result.variants:
        print("No RenderPlan variants found.")
        return

    print()
    print(
        "VARIANT                          PLAN      RENDER       STATE        PROGRESS          COMPOSITION"
    )
    for variant in result.variants:
        progress = (
            f"{variant.completed_units}/{variant.total_units} {variant.unit_kind}"
            if variant.total_units
            else "n/a"
        )
        composition = (
            "pending"
            if variant.pending_composition is True
            else "none"
            if variant.pending_composition is False
            else "n/a"
        )
        print(
            f"{variant.name:<32} {variant.plan_status:<9} "
            f"{variant.render_status:<12} {variant.state:<12} "
            f"{progress:<17} {composition}"
        )
        if variant.validation_error:
            print(f"  Invalid variant: {variant.name}")
            print(f"    error: {variant.validation_error}")
        if variant.error_code or variant.error_message:
            error = ": ".join(
                value for value in (variant.error_code, variant.error_message) if value
            )
            print(f"  Last render error: {error}")
            _print_diagnostic_summary(variant, prefix="    ")
        if variant.output_error:
            print(f"  Output metadata warning: {variant.output_error}")
        if variant.report_path:
            print(f"  report: {variant.report_path}")


def _print_bundle_role_cast(
    result: BundleRolesResult, *, heading: str = "Cast"
) -> None:
    if result.profile_source is None:
        return
    print(heading)
    print(f"  profile:          {result.profile_path}")
    print(f"  profile-id:       {result.profile_id}")
    if result.profile_source == "variant":
        print(f"  variant snapshot: {result.variant}")
    print(f"  default role:     {result.profile_default_role}")
    assert result.coverage is not None and result.distinctness is not None
    print(
        f"  coverage:         {'complete' if result.coverage.complete else 'incomplete'}"
    )
    print(f"  distinct voices:  {'yes' if result.distinctness.distinct else 'no'}")
    print(f"  readiness:        {'ready' if result.ready else 'not ready'}")
    print()
    print(
        "  ROLE       OCCUR  DEFAULT  BOUND  LANGUAGE  ENGINE   TARGET             VOICE/SPEAKER/SOURCE  UNIQUE"
    )
    for role in result.roles:
        bound = "yes" if role.bound else "no"
        default = "*" if role.required_by_default else ""
        unique = (
            "-"
            if role.bound is not True or role.binding_valid is not True
            else "no"
            if role.duplicate_roles
            else "yes"
        )
        print(
            f"  {role.role:<10} {role.occurrences:<6} {default:<8} {bound:<6} "
            f"{role.language or '-':<9} {role.renderer or '-':<8} "
            f"{role.target_id or '-':<18} {role.selector or '-':<21} {unique}"
        )
        if role.binding_error:
            print(f"    invalid binding: {role.binding_error}")
    assert result.coverage is not None
    if result.coverage.missing:
        print(f"  missing roles: {', '.join(result.coverage.missing)}")
    if result.coverage.unused:
        print(f"  unused profile roles: {', '.join(result.coverage.unused)}")
    assert result.distinctness is not None
    if result.distinctness.duplicate_groups:
        print("  Duplicate voices")
        for group in result.distinctness.duplicate_groups:
            identity = group.identity
            selector = (
                identity.voice
                if identity.voice is not None
                else f"speaker:{identity.speaker}"
                if identity.speaker is not None
                else "source"
                if identity.source is not None
                else "target-only"
            )
            print(f"    {', '.join(group.roles)}")
            print(
                f"      engine: {identity.renderer}; target: {identity.target_id}; voice: {selector}"
            )


def _print_bundle_roles(result: BundleRolesResult) -> None:
    print("VoiceRender bundle roles")
    print(f"  bundle:        {result.bundle}")
    print(f"  kind:          {result.kind}")
    print(f"  inventory:     {result.inventory_path}")
    print(f"  required:      {len(result.inventory.roles)} explicit roles")
    if result.inventory.uses_default_role:
        if result.profile_source is None:
            default_status = (
                f"required by {result.inventory.implicit_occurrences} speech segments; "
                "resolves after a profile is selected"
            )
        else:
            default_status = (
                f"required by {result.inventory.implicit_occurrences} speech segments; "
                f"resolves to {result.profile_default_role}"
            )
    else:
        default_status = "not required"
    print(f"  default-role:  {default_status}")
    if result.profile_source is not None:
        _print_bundle_role_cast(result)
        return
    print()
    print("ROLE")
    for role in result.inventory.roles:
        print(role.name)


def _print_bundle_inspect(result: BundleInspectResult) -> None:
    roles = result.roles
    print("VoiceRender bundle inspect")
    print(f"  bundle:           {result.bundle}")
    print(f"  kind:             {result.kind}")
    print(f"  role inventory:   fresh ({roles.inventory_path})")
    role_names = ", ".join(role.name for role in roles.inventory.roles) or "(none)"
    print(f"  explicit roles:   {role_names}")
    default_use = (
        f"yes ({roles.inventory.implicit_occurrences} speech segments)"
        if roles.inventory.uses_default_role
        else "no"
    )
    print(f"  default-role use: {default_use}")
    if roles.profile_source is not None:
        print()
        _print_bundle_role_cast(roles)
    print()
    print("Variants")
    if not result.variants.variants:
        print("  No RenderPlan variants found.")
        return
    print("  VARIANT                          PLAN      RENDER         SPEECH")
    for variant in result.variants.variants:
        speech = f"{variant.available_speech}/{variant.planned_speech}"
        print(
            f"  {variant.name:<32} {variant.plan_status:<9} "
            f"{variant.render_status:<14} {speech}"
        )
        if variant.validation_error:
            print(f"    invalid: {variant.validation_error}")
        if variant.error_code or variant.error_message:
            error = ": ".join(
                item for item in (variant.error_code, variant.error_message) if item
            )
            print(f"    render error: {error}")
            _print_diagnostic_summary(variant, prefix="    ")
        if variant.output_error:
            print(f"    output warning: {variant.output_error}")


def _print_profile_comparison(comparison: Any) -> None:
    print()
    print("Profile comparison")
    print(f"  status:             {comparison.status}")
    print(f"  authored profile:   {comparison.authored_profile_path}")
    print(f"  authored profile-id: {comparison.authored_profile_id}")
    print(f"  computed profile-id: {comparison.authored_computed_profile_id}")
    print(f"  authored ID valid:   {comparison.authored_profile_id_valid}")
    print(f"  variant snapshot:    {comparison.variant_profile_path}")
    print(f"  variant profile-id:  {comparison.variant_profile_id}")
    print(f"  identity match:      {comparison.identity_match}")
    if comparison.error_code or comparison.error_message:
        error = ": ".join(
            item for item in (comparison.error_code, comparison.error_message) if item
        )
        print(f"  comparison error:    {error}")
    elif comparison.differences:
        print("  differences:")
        for difference in comparison.differences:
            print(
                f"    {difference.role}.{difference.field}: "
                f"{difference.authored_value!r} -> {difference.variant_value!r}"
            )
    if comparison.recommended_action:
        print(f"  recommended action:  {comparison.recommended_action}")


def _diagnostic_summary_text(value: Any) -> str | None:
    parts: list[str] = []
    error_type = getattr(value, "error_type", None)
    error_stage = getattr(value, "error_stage", None)
    cause_type = getattr(value, "error_root_cause_type", None)
    cause_message = getattr(value, "error_root_cause_message", None)
    if isinstance(error_type, str) and error_type:
        parts.append(f"type {error_type}")
    if isinstance(error_stage, str) and error_stage:
        parts.append(f"stage {error_stage}")
    cause = ": ".join(
        item for item in (cause_type, cause_message) if isinstance(item, str) and item
    )
    if cause:
        parts.append(f"root cause {cause}")
    diagnostics = getattr(value, "render_diagnostics", None)
    if diagnostics is None:
        diagnostics = getattr(value, "diagnostics_path", None)
    if diagnostics is not None:
        parts.append(f"sidecar {diagnostics}")
    return "; ".join(parts) if parts else None


def _print_diagnostic_summary(value: Any, *, prefix: str) -> None:
    summary = _diagnostic_summary_text(value)
    if summary:
        print(f"{prefix}diagnostics: {summary}")


def _print_bundle_result(result: BundleRenderResult) -> None:
    value: Any = result.value
    if result.kind == "book":
        request = value.request
        selected = value.requested
        output = request.output_variant_dir
    else:
        request = value.request
        selected = 1
        output = request.output_dir

    print("VoiceRender bundle render")
    print(f"  kind:      {result.kind}")
    print(f"  workspace: {request.workspace}")
    print(f"  variant:   {request.variant_name}")
    print(f"  selected:  {selected}")
    print(f"  output:    {output}")

    if result.kind == "book":
        print(f"  renderplans: {request.renderplan_variant_dir}")
        print(f"  chapters:    {request.chapter_selector}")
        for outcome in value.outcomes:
            print(f"[{outcome.index}/{value.requested}] {outcome.id}")
            if outcome.status == "failed":
                print("       FAILED")
                print(
                    f"       {outcome.error_code or 'render.failed'}: "
                    f"{outcome.error_message or 'render failed'}"
                )
                _print_diagnostic_summary(outcome, prefix="       ")
                if outcome.preflight_report is not None:
                    label = (
                        "replan report"
                        if outcome.replan_required
                        else "preflight report"
                    )
                    print(f"       {label}: {outcome.preflight_report}")
            elif outcome.status == "skipped":
                print("       skipped after an earlier chapter failure")
            else:
                if request.speech_selector.strip().casefold() != "all":
                    print(
                        f"       speech: {outcome.selected_speech} of "
                        f"{outcome.planned_speech} selected"
                    )
                print(
                    f"       rendered: {outcome.rendered} new, {outcome.reused} reused"
                )
                state = "complete" if outcome.manifest_complete else "partial"
                print(
                    f"       chapter state: {state} "
                    f"({outcome.available_speech}/{outcome.planned_speech} available)"
                )
                if outcome.manifest is not None:
                    print(f"       manifest: {outcome.manifest}")
            if outcome.pending_composition:
                print(
                    "       WARNING: emphasis remains pending composition; see "
                    f"{outcome.output_dir / RENDER_DIAGNOSTICS_FILENAME}"
                )
        rendered = value.written
        reused = value.reused
        failed = value.failed
        status = "failed" if failed else "completed"
    else:
        print(f"  speech:    {request.speech_selector}")
        print(f"  status:    {value.status}")
        print(f"  rendered:  {value.rendered}")
        print(f"  reused:    {value.reused}")
        print(f"  report:    {value.report_path}")
        if value.preflight_report:
            print(f"  preflight: {value.preflight_report}")
        if value.error_code:
            print(f"FAILED: {value.error_code}: {value.error_message or ''}")
            _print_diagnostic_summary(value, prefix="  ")
        if value.pending_composition:
            print(
                "WARNING: emphasis remains pending composition; see "
                f"{output / RENDER_DIAGNOSTICS_FILENAME}"
            )
        rendered = value.rendered
        reused = value.reused
        failed = int(value.status == "failed")
        status = value.status

    print("VoiceRender summary")
    print(f"  status:    {status}")
    print(f"  selected:  {selected}")
    print(f"  rendered:  {rendered}")
    print(f"  reused:    {reused}")
    print(f"  failed:    {failed}")


def _print_bundle_batch_result(result: BundleBatchRenderResult) -> None:
    summary = result.summary
    print("VoiceRender bundle render")
    print("  mode:        batch")
    print(f"  bundle:      {result.bundle}")
    print(f"  kind:        {result.kind}")
    print(f"  renderplans: {result.renderplan_root}")
    print(f"  output:      {result.output_root}")
    print(f"  discovered:  {summary.discovered}")
    print(f"  scheduled:   {summary.scheduled}")
    print(f"  attempted:   {summary.attempted}")
    print(f"  skipped:     {summary.skipped_complete} complete")

    if summary.discovered == 0:
        print("No variants found; nothing to do.")
    elif result.no_work and summary.skipped_complete == summary.discovered:
        print("All valid variants are already complete; nothing to do.")

    attempted_index = 0
    for variant in result.variants:
        if variant.attempt_status is not None:
            attempted_index += 1
            position = f"[{attempted_index}/{summary.attempted}]"
        else:
            position = "[--]"
        print(f"\n{position} {variant.name} ({variant.initial_state})")
        print(f"  status: {variant.status}")
        if variant.attempt_status is not None:
            print(f"  attempt: {variant.attempt_status}")
        if variant.result is not None:
            value = variant.result.value
            rendered = getattr(value, "written", getattr(value, "rendered", 0))
            reused = getattr(value, "reused", 0)
            print(f"  rendered: {rendered} new, {reused} reused")
        error_message = variant.error_message or variant.validation_error
        if variant.error_code or error_message:
            print(
                "  error: "
                + ": ".join(
                    text for text in (variant.error_code, error_message) if text
                )
            )
            _print_diagnostic_summary(variant, prefix="  ")
        if variant.output_error:
            print(f"  output warning: {variant.output_error}")
        if variant.pending_composition:
            print("  WARNING: audio composition is pending")

    if result.error_message:
        print(
            f"Batch inspection error: {result.error_code or 'bundle.invalid'}: {result.error_message}"
        )
        _print_diagnostic_summary(result, prefix="  ")
    print("\nVoiceRender batch summary")
    print(f"  completed:       {summary.completed}")
    print(f"  partial:         {summary.partial}")
    print(f"  failed:          {summary.failed}")
    print(f"  skipped:         {summary.skipped_complete}")
    print(f"  invalid:         {summary.invalid}")
    print(f"  not attempted:   {summary.not_attempted}")


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


def _qwen3_module() -> Any:
    probe = get_adapter("qwen3").probe_api()
    if not probe.compatible:
        if probe.status == "package_missing":
            raise EngineUnavailableError(
                "Qwen3Synth is not installed; install the qwen3 extra to use this command.",
                engine="qwen3",
                engine_version=probe.module_version,
                native_error_type=probe.error_type,
                code="qwen3.package_missing",
            )
        raise EngineApiIncompatibleError(
            f"Qwen3Synth public API is incompatible: {probe.error_message or probe.status}",
            engine="qwen3",
            engine_version=probe.module_version,
            native_error_type=probe.error_type,
            code="qwen3.api_incompatible",
            details={
                "missing_symbols": probe.missing_symbols,
                "missing_methods": probe.missing_methods,
            },
        )
    return importlib.import_module("qwen3synth")


def _qwen_runtime_error(
    exc: Exception, *, operation: str, target_id: str, engine_version: str | None
) -> EngineError:
    if isinstance(exc, EngineError):
        return exc
    native_name = type(exc).__name__
    if native_name == "ModelNotInstalledError":
        error_type: type[EngineError] = EngineUnavailableError
        code = "qwen3.model_not_installed"
    elif native_name == "CatalogError":
        error_type = EngineUnavailableError
        code = "qwen3.catalog_unavailable"
    elif native_name == "UnsupportedModelCapabilityError":
        error_type = UnsupportedSynthesisFeatureError
        code = "qwen3.capability_unsupported"
    elif native_name in {
        "IncompatibleVoiceProfileError",
        "InvalidReferenceAudioError",
        "InvalidReferenceTranscriptError",
    }:
        error_type = InvalidEngineVoiceError
        code = "qwen3.profile_incompatible"
    else:
        error_type = EngineBackendError
        code = "qwen3.backend_failed"
    return error_type(
        f"Qwen3Synth {operation} failed: {exc}",
        engine="qwen3",
        engine_version=engine_version,
        target_id=target_id,
        native_error_type=native_name,
        code=code,
    )


def _inspect_qwen_voice_profile(path: Path) -> Any:
    module = _qwen3_module()
    try:
        return module.inspect_voice_profile(path)
    except Exception as exc:
        if isinstance(exc, EngineError):
            raise
        raise InvalidEngineVoiceError(
            f"Qwen prepared voice is invalid: {exc}",
            engine="qwen3",
            native_error_type=type(exc).__name__,
            code="qwen3.prepared_voice_invalid",
        ) from exc


def _prepared_voice_source(path: Path) -> tuple[dict[str, Any], Any]:
    info = _inspect_qwen_voice_profile(path)
    source = local_prepared_voice_source(
        path,
        format=QWEN3_QVOICE_FORMAT,
        metadata={
            "model_id": info.model_id,
            "model_revision": info.model_revision,
            "layout": info.layout,
        },
    )
    return source, info


def _cli_output_path(path: Path, *, suffix: str, force: bool, label: str) -> Path:
    target = path.expanduser().absolute()
    if target.suffix.casefold() != suffix:
        raise ArtifactError(f"{label} must use the {suffix} extension")
    if target.exists() and target.is_dir():
        raise ArtifactError(f"{label} is a directory: {target}")
    if (target.exists() or target.is_symlink()) and not force:
        raise ArtifactError(
            f"{label} already exists; pass --force to replace it",
            code="qwen3.voice_design_output_exists",
        )
    return target


def _prepare_qwen_voice(args: argparse.Namespace) -> dict[str, Any]:
    source_value = (
        load_voice_source(args.voice_source)
        if args.voice_source is not None
        else local_reference_source(args.reference)
    )
    source = normalize_voice_source(source_value, allow_named=False)
    if source.kind not in {"reference", "managed_reference"}:
        raise InvalidEngineVoiceError(
            "Qwen prepare-qwen requires a reference audio source",
            engine="qwen3",
            target_id=args.target,
            code="qwen3.reference_required",
        )
    origin = source.origin or {}
    transcript = args.transcript
    if transcript is None:
        transcript = origin.get("reference_text")
    if not isinstance(transcript, str) or not transcript.strip():
        raise InvalidEngineOptionError(
            "Qwen prepare-qwen requires --transcript or a Qwen VoiceDesign source transcript",
            engine="qwen3",
            target_id=args.target,
            code="qwen3.reference_text_required",
        )
    language = args.language or origin.get("language") or "en"
    if not isinstance(language, str) or not language.strip():
        raise InvalidEngineOptionError(
            "Qwen prepare-qwen requires a non-empty language",
            engine="qwen3",
            target_id=args.target,
            code="qwen3.language_required",
        )

    output = _cli_output_path(
        args.output,
        suffix=".qvoice",
        force=args.force,
        label="Qwen prepared voice output",
    )
    resolved = resolve_reference_audio(source, offline=True)
    source_id_payload = {
        "voice": dict(voice_source_identity(source)),
        "reference_text_sha256": hashlib.sha256(transcript.encode("utf-8")).hexdigest(),
    }
    source_id = hashlib.sha256(
        json.dumps(source_id_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    module = _qwen3_module()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".voicerender-prepare-qwen-", suffix=".qvoice", dir=output.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    runtime: Any | None = None
    try:
        try:
            runtime = module.Qwen3Runtime.from_installed(
                args.target,
                quality=args.quality,
                provider=args.provider,
                catalog=args.catalog,
                cache_dir=args.cache_dir,
            )
        except Exception as exc:
            raise _qwen_runtime_error(
                exc,
                operation="runtime opening",
                target_id=args.target,
                engine_version=getattr(module, "__version__", None),
            ) from exc
        try:
            reference = module.ReferenceVoice(
                audio=resolved.path,
                transcript=transcript,
                language=language,
                source_id=source_id,
            )
            prepared = runtime.prepare_voice(reference)
            prepared.save(temporary_path)
        except Exception as exc:
            if isinstance(exc, EngineError):
                raise
            raise _qwen_runtime_error(
                exc,
                operation="voice preparation",
                target_id=args.target,
                engine_version=getattr(module, "__version__", None),
            ) from exc
        finally:
            runtime.close()
        if args.force:
            os.replace(temporary_path, output)
        else:
            try:
                install_file_no_replace(temporary_path, output)
            except FileExistsError as exc:
                raise ArtifactError(
                    f"Qwen prepared voice output appeared during generation: {output}",
                    code="qwen3.voice_design_output_exists",
                ) from exc
            temporary_path.unlink()
    finally:
        temporary_path.unlink(missing_ok=True)

    prepared_source = local_prepared_voice_source(
        output,
        format=QWEN3_QVOICE_FORMAT,
        metadata={
            "model_id": prepared.model_id,
            "model_revision": prepared.model_revision,
            "layout": prepared.layout,
        },
    )
    return {
        "output": str(output),
        "format": QWEN3_QVOICE_FORMAT,
        "sha256": prepared_source["sha256"],
        "source": prepared_source,
        "model_id": prepared.model_id,
        "model_revision": prepared.model_revision,
        "transcript": transcript,
        "language": language,
    }


def _voice_source_command(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    if args.voice_source_command == "create":
        output = args.output.expanduser().absolute()
        if (output.exists() or output.is_symlink()) and not args.force:
            raise ArtifactError(
                f"Voice-source output already exists; pass --force to replace it: {output}",
                code="voice_source.output_exists",
            )
        if (
            args.reference_wav is not None
            and output.resolve() == args.reference_wav.expanduser().resolve()
        ):
            raise ValueError("Voice-source output must not overwrite the reference WAV")
        source = create_voice_source(
            reference_wav=args.reference_wav,
            voicefolio_ref=args.voicefolio_ref,
            transcript=args.transcript,
            language=args.language,
            name=args.name,
            metadata=_json_options(args.metadata, label="--metadata"),
        )
        saved = save_voice_source(source.as_mapping(), output)
        return {"path": str(saved), "voice_source": source.as_mapping()}, 0

    if args.voice_source_command == "inspect":
        loaded_source = load_voice_source(args.path)
        return {
            "path": str(args.path),
            "kind": loaded_source["kind"],
            "voice_source": dict(loaded_source),
            "metadata": dict(loaded_source.get("metadata", {})),
        }, 0

    try:
        validated_source = load_voice_source(args.path)
        diagnostics = validate_voice_source(validated_source)
    except (VoiceRenderError, OSError, ValueError) as exc:
        diagnostic = {
            "code": getattr(exc, "code", "voice_source.invalid"),
            "severity": "error",
            "message": str(exc),
            "details": dict(getattr(exc, "details", {})),
        }
        return {"path": str(args.path), "valid": False, "diagnostics": [diagnostic]}, 1
    values = [diagnostic.as_dict() for diagnostic in diagnostics]
    valid = not any(item["severity"] == "error" for item in values)
    return {
        "path": str(args.path),
        "valid": valid,
        "diagnostics": values,
    }, 0 if valid else 1


def _voice_clone_command(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.expanduser().absolute()
    if (output.exists() or output.is_symlink()) and not args.force:
        raise ArtifactError(
            f"RenderProfile output already exists; pass --force to replace it: {output}",
            code="voice_clone.output_exists",
        )
    source_path = args.voice_source or args.reference_wav
    if (
        source_path is not None
        and output.resolve() == source_path.expanduser().resolve()
    ):
        raise ValueError(
            "RenderProfile output must not overwrite its voice-source input"
        )
    speaker: str | int | None = args.speaker
    if isinstance(speaker, str):
        try:
            numeric_speaker = int(speaker)
        except ValueError:
            pass
        else:
            if str(numeric_speaker) == speaker:
                speaker = numeric_speaker
    profile = clone_voice_profile(
        VoiceCloneRequest(
            engine=args.engine,
            target_id=args.target,
            language=args.language,
            reference_wav=args.reference_wav,
            voice_source=args.voice_source,
            voicefolio_ref=args.voicefolio_ref,
            transcript=args.transcript,
            speaker=speaker,
            options=_json_options(args.option),
            offline=args.offline,
            refresh=args.refresh,
        ),
        role=args.role,
    )
    saved = save_render_profile(profile, output)
    return {"profile_id": profile.profile_id, "path": str(saved), "role": args.role}


def _runtime_path(path: Path | None) -> str | None:
    """Render paths with stable separators across CLI platforms."""
    return path.as_posix() if path is not None else None


def _runtime_status_to_dict(status: OnnxRuntimeStatus) -> dict[str, Any]:
    return {
        "importable": status.importable,
        "module_version": status.module_version,
        "module_path": _runtime_path(status.module_path),
        "build_device": status.build_device,
        "build_info": status.build_info,
        "distributions": [
            {
                "name": item.name,
                "version": item.version,
                "location": _runtime_path(item.location),
                "owns_import": item.owns_import,
            }
            for item in status.distributions
        ],
        "imported_distribution": status.imported_distribution,
        "distribution_state": status.distribution_state,
        "available_providers": list(status.available_providers),
        "configured_providers": list(status.configured_providers),
        "provider_source": status.provider_source,
        "resolved_providers": list(status.resolved_providers),
        "import_error": status.import_error,
        "resolution_error": status.resolution_error,
    }


def _print_runtime_status(status: OnnxRuntimeStatus) -> None:
    rows = (
        ("Imported distribution", status.imported_distribution),
        ("Distribution state", status.distribution_state),
        ("Module version", status.module_version),
        ("Module path", _runtime_path(status.module_path)),
        ("Build device", status.build_device),
        ("Build info", status.build_info),
        ("Available providers", ", ".join(status.available_providers)),
        ("Configured providers", ", ".join(status.configured_providers)),
        ("Provider source", status.provider_source),
        ("Resolved providers", ", ".join(status.resolved_providers)),
        ("Import error", status.import_error),
        ("Resolution error", status.resolution_error),
    )
    for label, value in rows:
        print(f"{label}: {value if value is not None and value != '' else '-'}")


def _print_runtime_list(status: OnnxRuntimeStatus) -> None:
    print("DISTRIBUTION\tVERSION\tLOCATION\tOWNS_IMPORT")
    for item in status.distributions:
        print(
            f"{item.name}\t{item.version or '-'}\t"
            f"{_runtime_path(item.location) or '-'}\t{item.owns_import}"
        )
    if not status.distributions:
        print("-\t-\t-\t-")
    print("Available providers:")
    for provider in status.available_providers:
        print(provider)
    if not status.available_providers:
        print("-")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload: dict[str, Any]
    try:
        if args.command == "runtime":
            if args.runtime_command == "status":
                status = onnx_runtime_status()
                if args.json:
                    print(
                        json.dumps(
                            _runtime_status_to_dict(status),
                            ensure_ascii=False,
                            indent=2,
                            sort_keys=True,
                        )
                    )
                else:
                    _print_runtime_status(status)
                return 0
            if args.runtime_command == "list":
                status = onnx_runtime_status()
                payload = {
                    "distribution_state": status.distribution_state,
                    "distributions": _runtime_status_to_dict(status)["distributions"],
                    "available_providers": list(status.available_providers),
                }
                if args.json:
                    print(
                        json.dumps(
                            payload, ensure_ascii=False, indent=2, sort_keys=True
                        )
                    )
                else:
                    _print_runtime_list(status)
                return 0
            if args.runtime_command == "use":
                set_onnx_runtime_providers(args.providers)
                payload = {
                    "status": "ok",
                    "action": "use",
                    "providers": args.providers,
                    "applies_to": "future_sessions",
                }
                if args.json:
                    print(
                        json.dumps(
                            payload, ensure_ascii=False, indent=2, sort_keys=True
                        )
                    )
                else:
                    print(
                        f"Saved ONNX Runtime provider preference {args.providers!r} "
                        "for future compatible sessions."
                    )
                return 0
            clear_onnx_runtime_providers()
            payload = {
                "status": "ok",
                "action": "reset",
                "environment_overrides_remain": True,
                "existing_sessions_changed": False,
            }
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
            else:
                print(
                    "Cleared saved ONNXVoice provider preference; environment overrides remain in force."
                )
            return 0
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
            if args.engines_command == "options":
                options = engine_options(args.engine, target_id=args.target)
                rows = [engine_option_to_dict(option) for option in options]
                if args.json:
                    print(
                        json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True)
                    )
                else:
                    print("NAME\tTYPE\tREQUIRED\tDEFAULT\tCHOICES\tDESCRIPTION")
                    for option in options:
                        default = (
                            "-"
                            if option.default is None
                            else _display_cli_value(option.default)
                        )
                        choices = (
                            ", ".join(
                                _display_cli_value(choice) for choice in option.choices
                            )
                            or "-"
                        )
                        print(
                            f"{option.name}\t{option.value_type}\t"
                            f"{str(option.required).lower()}\t{default}\t"
                            f"{choices}\t{option.description}"
                        )
                return 0

        if args.command == "lexicons":
            adapter = get_adapter(args.engine)
            if args.lexicons_command == "inspect-file":
                inspector = getattr(adapter, "inspect_g2lex_file", None)
                if not callable(inspector):
                    raise ValueError(
                        f"Engine {adapter.id!r} does not support local G2Lex inspection"
                    )
                payload = _inspection_to_dict(inspector(str(args.path)))
                if "valid" not in payload:
                    payload["valid"] = True
                if args.json:
                    print(
                        json.dumps(
                            payload, ensure_ascii=False, indent=2, sort_keys=True
                        )
                    )
                else:
                    metadata = payload.get("metadata", {})
                    print(f"G2Lex file: {payload.get('path', args.path)}")
                    print(f"  sha256:         {payload.get('sha256', '-')}")
                    print(f"  logical sha256: {payload.get('logical_sha256') or '-'}")
                    if isinstance(metadata, Mapping):
                        for key in sorted(metadata):
                            print(f"  {key}: {metadata[key]}")
                return 0

            discover = getattr(adapter, "discover_lexicons", None)
            if not callable(discover):
                raise ValueError(
                    f"Engine {adapter.id!r} does not support named lexicon discovery"
                )
            entries, _ = discover(
                language=args.lang,
                model=args.target,
                offline=args.offline,
                refresh=args.refresh,
                preference=args.preference,
            )
            rows = [_inspection_to_dict(entry) for entry in entries]
            if args.lexicons_command == "inspect":
                entry = next(
                    (item for item in entries if item.selector == args.selector), None
                )
                if entry is None:
                    available = ", ".join(item.selector for item in entries) or "none"
                    raise ValueError(
                        f"Lexicon selector {args.selector!r} is not available for {args.lang or 'all languages'}. "
                        f"Available selectors: {available}"
                    )
                payload = _inspection_to_dict(entry)
                if args.json:
                    print(
                        json.dumps(
                            payload, ensure_ascii=False, indent=2, sort_keys=True
                        )
                    )
                else:
                    for key, value in payload.items():
                        print(f"{key}: {_display_cli_value(value)}")
                return 0
            if args.json:
                print(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True))
            else:
                print(
                    "SELECTOR\tLANGUAGE\tDEFAULT\tINSTALLED\tENCODING\tBACKEND\tASSET ID\tMODEL SUPPORT"
                )
                for item in entries:
                    installed = (
                        "unknown"
                        if item.installed is None
                        else str(item.installed).lower()
                    )
                    print(
                        f"{item.selector}\t{item.locale}\t{str(item.default).lower()}\t{installed}\t"
                        f"{item.phoneme_encoding or '-'}\t{item.data_backend or '-'}\t"
                        f"{item.asset_id or '-'}\t{item.model_support}"
                    )
            return 0

        if args.command == "models":
            if args.models_command == "list":
                models = list_models(
                    args.engine,
                    language=args.lang,
                    installed=True
                    if args.installed
                    else False
                    if args.available
                    else None,
                    offline=args.offline,
                    refresh=args.refresh,
                )
                if args.json:
                    rows = [model_artifact_to_dict(model) for model in models]
                    print(
                        json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True)
                    )
                else:
                    print("ENGINE\tTARGET\tSTATUS\tINSTALLED\tLANGUAGES\tQUALITIES")
                    for model in models:
                        print(
                            f"{model.engine}\t{model.target_id}\t{model.status}\t"
                            f"{str(model.installed).lower()}\t{', '.join(model.languages) or '-'}\t"
                            f"{', '.join(model.qualities) or '-'}"
                        )
                return 0
            if args.models_command == "inspect":
                artifact = model_status(args.engine, args.target, quality=args.quality)
                if args.json:
                    payload = model_artifact_to_dict(artifact)
                    print(
                        json.dumps(
                            payload, ensure_ascii=False, indent=2, sort_keys=True
                        )
                    )
                else:
                    print(
                        f"{artifact.engine}\t{artifact.target_id}\t{artifact.status}\t"
                        f"installed={str(artifact.installed).lower()}\t"
                        f"languages={','.join(artifact.languages) or '-'}\t"
                        f"qualities={','.join(artifact.qualities) or '-'}"
                    )
                return 0
            model_result: ModelInstallResult | ModelRemoveResult | ModelVerifyResult
            if args.models_command == "install":
                model_result = install_model(
                    ModelInstallRequest(
                        args.engine,
                        args.target,
                        args.quality,
                        args.source,
                        args.cache_dir,
                        args.force,
                    )
                )
            elif args.models_command == "remove":
                if not args.yes:
                    raise ValueError("model removal requires --yes")
                model_result = remove_model(
                    ModelRemoveRequest(
                        args.engine, args.target, args.quality, args.cache_dir
                    )
                )
            else:
                model_result = verify_model(
                    args.engine,
                    args.target,
                    quality=args.quality,
                    cache_dir=args.cache_dir,
                )
            payload = model_result_to_dict(model_result)
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
            else:
                operation = args.models_command
                outcome = model_result.status
                if hasattr(model_result, "changed"):
                    outcome += f" changed={str(model_result.changed).lower()}"
                elif isinstance(model_result, ModelVerifyResult):
                    outcome += f" verified={str(model_result.verified).lower()}"
                print(
                    f"Model {operation} {model_result.engine}:{model_result.target_id}: {outcome}"
                )
                for diagnostic in model_result.diagnostics:
                    print(f"diagnostic: {diagnostic}", file=sys.stderr)
            if model_result.status == "unsupported":
                return 2
            if model_result.status in {"failed", "error"}:
                return 1
            if (
                isinstance(model_result, ModelVerifyResult)
                and not model_result.verified
            ):
                return 1
            return 0

        if args.command == "voices":
            if args.voices_command == "inspect":
                descriptor = inspect_voice(args.selector, language=args.lang)
                payload = voice_descriptor_to_dict(descriptor)
                if args.json:
                    print(
                        json.dumps(
                            payload, ensure_ascii=False, indent=2, sort_keys=True
                        )
                    )
                else:
                    print(
                        f"{descriptor.selector}\t{descriptor.kind}\t"
                        f"{','.join(descriptor.languages) or '-'}"
                    )
                return 0
            descriptors = list_voices(
                engine=args.engine,
                target_id=args.target,
                language=args.lang,
                offline=args.offline,
                refresh=args.refresh,
            )
            if args.kind is not None:
                descriptors = tuple(
                    item for item in descriptors if item.kind == args.kind
                )
            rows = [voice_descriptor_to_dict(item) for item in descriptors]
            if args.json:
                print(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True))
            else:
                print("SELECTOR\tKIND\tLANGUAGES\tGENDER\tLOCALE\tINSTALLED")
                for item in descriptors:
                    print(
                        f"{item.selector}\t{item.kind}\t{','.join(item.languages) or '-'}\t"
                        f"{item.gender}\t{item.locale}\t{item.installed}"
                    )
            return 0

        if args.command == "voice":
            if args.voice_command == "source":
                payload, exit_code = _voice_source_command(args)
                if args.json:
                    print(
                        json.dumps(
                            payload, ensure_ascii=False, indent=2, sort_keys=True
                        )
                    )
                elif args.voice_source_command == "create":
                    source = payload["voice_source"]
                    digest = source.get("sha256")
                    print(
                        f"Voice source {source['kind']} -> {payload['path']} sha256={digest or '-'}"
                    )
                elif args.voice_source_command == "inspect":
                    print(f"Voice source {payload['kind']} {payload['path']}")
                else:
                    print(
                        f"Voice source {payload['path']}: {'valid' if payload['valid'] else 'invalid'}"
                    )
                    for item in payload["diagnostics"]:
                        if item["severity"] == "error":
                            print(f"diagnostic: {item['message']}", file=sys.stderr)
                return exit_code
            if args.voice_command == "clone":
                payload = _voice_clone_command(args)
                if args.json:
                    print(
                        json.dumps(
                            payload, ensure_ascii=False, indent=2, sort_keys=True
                        )
                    )
                else:
                    print(f"RenderProfile {payload['profile_id']} -> {payload['path']}")
                return 0
            if args.voice_command == "design":
                recipe = Qwen3VoiceDesignRecipe(
                    prompt=args.prompt,
                    reference_text=args.text,
                    language=args.language,
                    quality=args.quality,
                    provider=args.provider,
                    seed=args.seed,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                    max_new_tokens=args.max_new_tokens,
                )
                designed = design_qwen3_reference(
                    recipe,
                    args.output,
                    cache_dir=args.cache_dir,
                    catalog=args.catalog,
                    source_output=args.source_output,
                    force=args.force,
                )
                payload = {
                    "output": str(designed.path),
                    "sha256": designed.sha256,
                    "source": dict(designed.source),
                    "model_id": designed.model_id,
                    "runtime_identity": dict(designed.runtime_identity),
                }
            elif args.voice_command == "prepare-qwen":
                payload = _prepare_qwen_voice(args)
            elif args.path.suffix.casefold() == ".qvoice":
                info = _inspect_qwen_voice_profile(args.path)
                payload = {
                    "kind": "prepared",
                    "format": QWEN3_QVOICE_FORMAT,
                    "path": str(args.path),
                    "sha256": info.profile_sha256,
                    "model_id": info.model_id,
                    "model_revision": info.model_revision,
                    "layout": info.layout,
                    "language": info.language,
                    "reference_samples": info.reference_samples,
                }
            else:
                source = load_voice_source(args.path)
                payload = {
                    "kind": source["kind"],
                    "path": str(args.path),
                    "voice_source": dict(source),
                }
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            elif args.voice_command == "design":
                print(
                    f"Designed reference {payload['output']} sha256={payload['sha256']}"
                )
            elif args.voice_command == "prepare-qwen":
                print(
                    f"Prepared Qwen voice {payload['output']} sha256={payload['sha256']}"
                )
            else:
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0

        if args.command == "profile":
            if args.profile_command == "create":
                output_path = normalize_renderprofile_output_path(args.output)
                profile_options = _profile_pronunciation_options(
                    args.engine,
                    args.language,
                    _json_options(args.option),
                    lexicon_names=args.lexicon,
                    g2lex_paths=args.g2lex,
                    lexicon_data_policy=args.lexicon_data_policy,
                )
                profile_options = _profile_short_sentence_option(
                    args.engine, profile_options, args.short_sentence
                )
                input_paths = (
                    args.reference_wav,
                    args.prepared_voice,
                    args.voice_source,
                )
                if any(
                    path is not None and output_path.resolve() == path.resolve()
                    for path in input_paths
                ):
                    raise ValueError(
                        "profile output must not overwrite a voice-source input"
                    )
                voice_source = None
                if args.voicefolio_ref:
                    voice_source = managed_reference_source(args.voicefolio_ref)
                elif args.reference_wav is not None:
                    voice_source = local_reference_source(args.reference_wav)
                elif args.prepared_voice is not None:
                    if get_adapter(args.engine).id != "qwen3":
                        raise InvalidEngineOptionError(
                            "--prepared-voice is only supported for Qwen3 profiles",
                            code="qwen3.prepared_voice_engine_invalid",
                        )
                    voice_source, _ = _prepared_voice_source(args.prepared_voice)
                elif args.voice_source is not None:
                    voice_source = dict(load_voice_source(args.voice_source))
                profile = create_render_profile(
                    engine=args.engine,
                    target_id=args.target,
                    language=args.language,
                    role=args.role,
                    voice=args.voice,
                    voice_source=voice_source,
                    speaker=args.speaker,
                    options=profile_options,
                    transcript=args.transcript,
                    offline=args.offline,
                    refresh=args.refresh,
                )
                output = save_render_profile(profile, output_path)
                payload = {"profile_id": profile.profile_id, "path": str(output)}
            elif args.profile_command == "reidentify":
                if (
                    args.output is not None
                    and args.output.resolve() == args.profile.resolve()
                ):
                    raise ValueError(
                        "profile output must not overwrite its source; use --in-place"
                    )

                document = tomlkit.parse(args.profile.read_text(encoding="utf-8"))
                previous_id = document.get("profile_id")
                profile = reidentify_render_profile(args.profile)
                document["profile_id"] = profile.profile_id
                destination = args.profile if args.in_place else args.output
                assert destination is not None
                destination.parent.mkdir(parents=True, exist_ok=True)
                fd, temporary_name = tempfile.mkstemp(
                    prefix=f".{destination.name}.",
                    suffix=".tmp",
                    dir=destination.parent,
                )
                try:
                    with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
                        stream.write(tomlkit.dumps(document))
                        stream.flush()
                        os.fsync(stream.fileno())
                    if args.in_place:
                        os.chmod(temporary_name, args.profile.stat().st_mode)
                    os.replace(temporary_name, destination)
                finally:
                    if os.path.exists(temporary_name):
                        os.unlink(temporary_name)
                payload = {
                    "path": str(destination),
                    "old_profile_id": str(previous_id)
                    if previous_id is not None
                    else None,
                    "profile_id": profile.profile_id,
                    "changed": previous_id != profile.profile_id,
                }
            elif args.profile_command == "validate":
                profile = validate_render_profile(load_render_profile(args.profile))
                payload = {
                    "profile_id": profile.profile_id,
                    "roles": sorted(profile.roles),
                    "path": str(args.profile),
                    "valid": True,
                }
            elif args.profile_command == "refresh":
                if args.output.resolve() == args.profile.resolve():
                    raise ValueError(
                        "profile refresh output must not overwrite its source"
                    )
                profile = refresh_render_profile(load_render_profile(args.profile))
                output = save_render_profile(profile, args.output)
                payload = {"profile_id": profile.profile_id, "path": str(output)}
            elif args.profile_command in {"inspect", "show"}:
                profile = load_render_profile(args.profile)
                payload = {
                    "profile_id": profile.profile_id or profile.computed_id(),
                    "default_role": profile.default_role,
                    "roles": _profile_role_items(profile),
                    "planning": profile.planning.to_dict(),
                    "pauses": profile.pauses.to_dict(),
                }
            elif args.profile_command in {"role", "bind"}:
                if (
                    args.profile_command == "role"
                    and args.profile_role_command == "list"
                ):
                    profile = load_render_profile(args.profile)
                    payload = {
                        "profile_id": profile.profile_id or profile.computed_id(),
                        "default_role": profile.default_role,
                        "roles": _profile_role_items(profile),
                    }
                else:
                    destination = _profile_mutation_destination(args)
                    if (
                        args.profile_command == "bind"
                        or args.profile_role_command == "set"
                    ):
                        source_paths = (
                            args.reference_wav,
                            args.prepared_voice,
                            args.voice_source,
                        )
                        if any(
                            path is not None and destination.resolve() == path.resolve()
                            for path in source_paths
                        ):
                            raise ValueError(
                                "profile output must not overwrite a voice-source input"
                            )
                        voice_source = _profile_role_voice_source(args)
                        binding_options = _profile_pronunciation_options(
                            args.engine,
                            args.language,
                            _json_options(args.option),
                            lexicon_names=args.lexicon,
                            g2lex_paths=args.g2lex,
                            lexicon_data_policy=args.lexicon_data_policy,
                        )
                        binding_options = _profile_short_sentence_option(
                            args.engine, binding_options, args.short_sentence
                        )
                        profile = set_profile_role(
                            args.profile,
                            args.role,
                            engine=args.engine,
                            target_id=args.target,
                            language=args.language,
                            voice=args.voice,
                            speaker=args.speaker,
                            voice_source=voice_source,
                            options=binding_options,
                            transcript=args.transcript,
                            offline=args.offline,
                            refresh=args.refresh,
                            make_default=args.make_default,
                        )
                    else:
                        profile = remove_profile_role(
                            args.profile,
                            args.role,
                            new_default_role=args.new_default_role,
                        )
                    output = save_render_profile(profile, destination)
                    payload = {
                        "profile_id": profile.profile_id,
                        "default_role": profile.default_role,
                        "roles": _profile_role_items(profile),
                        "path": str(output),
                    }
            else:
                destination = _profile_mutation_destination(args)
                profile = set_default_profile_role(args.profile, args.role)
                output = save_render_profile(profile, destination)
                payload = {
                    "profile_id": profile.profile_id,
                    "default_role": profile.default_role,
                    "path": str(output),
                }
            if args.json:
                print(
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                )
            elif args.profile_command in {"inspect", "show", "role"} and (
                args.profile_command in {"inspect", "show"}
                or args.profile_role_command == "list"
            ):
                print(
                    f"RenderProfile {payload['profile_id']} "
                    f"(default role: {payload['default_role']})"
                )
                _print_profile_role_items(
                    payload["roles"],
                    show_engine_details=args.profile_command in {"inspect", "show"},
                )
                if args.profile_command in {"inspect", "show"}:
                    planning = payload["planning"]
                    fallbacks = ", ".join(planning["capacity_fallback"]) or "none"
                    print(
                        f"Planning: split={planning['split']}; capacity fallback={fallbacks}"
                    )
                    pauses = payload["pauses"]
                    pause_mode = pauses.get("mode", "none")
                    print("Pauses")
                    print(f"  mode: {pause_mode}")
                    if pause_mode == "manual":
                        for boundary in ("clause", "parenthetical", "sentence", "paragraph"):
                            if boundary in pauses:
                                print(f"  {boundary}: {pauses[boundary]}")
                        print("  speech artifact tails: trimmed")
                        print(
                            "  composition semantics: configured pause is inserted "
                            "after normalized speech"
                        )
                    else:
                        print("  speech artifact tails: trimmed")
                        print("  composition semantics: no policy pause is inserted")
            elif args.profile_command == "reidentify":
                print(
                    "RenderProfile reidentified\n"
                    f"  previous profile-id: {payload['old_profile_id']}\n"
                    f"  current profile-id:  {payload['profile_id']}\n"
                    f"  path:                {payload['path']}"
                )
            elif "path" in payload:
                print(f"RenderProfile {payload['profile_id']} -> {payload['path']}")
            else:
                print(f"RenderProfile {payload['profile_id']} is valid")
            return 0
        if args.command == "preflight":
            if args.output.resolve() == args.plan.resolve():
                raise ValueError(
                    "preflight report path must not overwrite its RenderPlan"
                )
            progress = _progress_callback(args)
            try:
                plan = load_plan(args.plan)
                report = preflight_render_plan(
                    plan,
                    args.output,
                    emphasis_policy=args.emphasis_policy,
                    on_event=progress,
                )
            finally:
                if progress is not None:
                    progress.finish()
            payload = _preflight_payload(
                report,
                pronunciation_stacks=_pronunciation_stacks_for_plan(plan),
            )
            payload["output"] = str(args.output)
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            else:
                print(
                    f"Preflight {report.status}: {payload['ok']} ok, "
                    f"{payload['replan']} replan, {payload['reject']} reject; "
                    f"{payload['checked']} speech items"
                )
                _print_pronunciation_stacks(payload.get("pronunciation_stacks", []))
                for line in _sopro_unknown_fit_summary(report):
                    print(line)
            return 0 if report.status == "ok" else 2 if report.status == "replan" else 1

        if args.command == "render":
            render_result: RenderPlanResult
            if args.recheck_report is not None and args.recheck_report.resolve() in {
                args.plan.resolve(),
                args.preflight.resolve(),
            }:
                raise ValueError(
                    "fresh preflight report must not overwrite the RenderPlan or supplied report"
                )
            progress = _progress_callback(args)
            try:
                render_result = render_renderplan(
                    args.plan,
                    preflight=args.preflight,
                    output_dir=args.out_dir,
                    cache_dir=args.cache_dir,
                    recheck_report_path=args.recheck_report,
                    force=args.force,
                    reset_stale_output=args.reset_stale,
                    emphasis_policy=args.emphasis_policy,
                    max_open_sessions=args.max_open_sessions,
                    on_event=progress,
                )
            finally:
                if progress is not None:
                    progress.finish()
            payload = _render_payload(render_result)
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            elif render_result.failed:
                print(
                    f"Render incomplete: {render_result.error_code or 'render.failed'}: {render_result.error}; "
                    f"manifest: {render_result.manifest_path}"
                )
            else:
                print(
                    f"Rendered {payload['speech_count']} speech items "
                    f"({render_result.rendered_count} new, {render_result.reused_count} reused); "
                    f"manifest: {render_result.manifest_path}"
                )
            if render_result.pending_composition:
                print(
                    "WARNING: emphasis is pending composition; see "
                    f"{render_result.manifest_path.parent / RENDER_DIAGNOSTICS_FILENAME}"
                )
            if render_result.failed:
                fresh = render_result.fresh_preflight
                return 2 if fresh is not None and fresh.status == "replan" else 1
            return 0

        if args.command == "bundle" and args.bundle_command == "render":
            progress = _progress_callback(args)
            try:
                try:
                    if args.variant is None:
                        batch_result = render_bundle_batch(
                            args.bundle,
                            chapters=args.chapters,
                            speech=args.speech,
                            renderplan_root=args.renderplan_root,
                            output_root=args.output_root,
                            cache_dir=args.cache_dir,
                            force=args.force,
                            reset_stale=args.reset_stale,
                            emphasis_policy=args.emphasis_policy,
                            fail_fast=args.fail_fast,
                            max_open_sessions=args.max_open_sessions,
                            on_progress=progress,
                        )
                    else:
                        bundle_result = render_bundle(
                            args.bundle,
                            variant=args.variant,
                            chapters=args.chapters,
                            speech=args.speech,
                            renderplan_root=args.renderplan_root,
                            output_root=args.output_root,
                            cache_dir=args.cache_dir,
                            force=args.force,
                            reset_stale=args.reset_stale,
                            emphasis_policy=args.emphasis_policy,
                            fail_fast=args.fail_fast,
                            max_open_sessions=args.max_open_sessions,
                            on_progress=progress,
                        )
                finally:
                    if progress is not None:
                        progress.finish()
            except (VoiceRenderError, OSError, ValueError, TypeError) as exc:
                if args.variant is None:
                    error_code = getattr(exc, "code", None)
                    payload = {
                        "format": "voicerender-bundle-batch-result",
                        "version": 1,
                        "kind": None,
                        "bundle": str(args.bundle),
                        "renderplan_root": (
                            str(args.renderplan_root)
                            if args.renderplan_root is not None
                            else None
                        ),
                        "output_root": (
                            str(args.output_root)
                            if args.output_root is not None
                            else None
                        ),
                        "force": args.force,
                        "summary": {
                            "discovered": 0,
                            "scheduled": 0,
                            "attempted": 0,
                            "completed": 0,
                            "partial": 0,
                            "failed": 0,
                            "skipped_complete": 0,
                            "invalid": 0,
                            "not_attempted": 0,
                        },
                        "variants": [],
                        "error_code": error_code
                        if isinstance(error_code, str)
                        else "bundle.invalid",
                        "error_message": str(exc),
                        "exit_code": 2,
                    }
                else:
                    payload = {
                        "format": "voicerender-bundle-result",
                        "version": 1,
                        "error": str(exc),
                        "code": getattr(exc, "code", "bundle.invalid"),
                        "exit_code": 2,
                    }
                if args.json:
                    print(json.dumps(payload, ensure_ascii=False, indent=2))
                else:
                    print(f"error: {exc}", file=sys.stderr)
                return 2

            if args.variant is None:
                if args.json:
                    print(
                        json.dumps(batch_result.to_dict(), ensure_ascii=False, indent=2)
                    )
                else:
                    _print_bundle_batch_result(batch_result)
                return batch_result.exit_code
            if args.json:
                print(json.dumps(bundle_result.to_dict(), ensure_ascii=False, indent=2))
            else:
                _print_bundle_result(bundle_result)
            return bundle_result.exit_code

        if args.command == "bundle" and args.bundle_command == "roles":
            if args.check and args.profile is None and args.variant is None:
                raise BundleRolesError(
                    "--check requires --profile or --variant",
                    code="bundle.roles_invalid",
                )
            result = inspect_bundle_roles(
                args.bundle,
                profile=args.profile,
                variant=args.variant,
                utterplan_root=args.utterplan_root,
                renderplan_root=args.renderplan_root,
            )
            if args.json:
                print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
            else:
                _print_bundle_roles(result)
            return 1 if args.check and result.ready is not True else 0

        if args.command == "bundle" and args.bundle_command == "inspect":
            if args.check and args.profile is None and args.variant is None:
                raise BundleRolesError(
                    "--check requires --profile or --variant",
                    code="bundle.roles_invalid",
                )
            inspect_result = inspect_bundle(
                args.bundle,
                profile=args.profile,
                variant=args.variant,
                compare_profile=args.compare_profile,
                utterplan_root=args.utterplan_root,
                renderplan_root=args.renderplan_root,
                output_root=args.output_root,
            )
            if args.json:
                print(
                    json.dumps(inspect_result.to_dict(), ensure_ascii=False, indent=2)
                )
            else:
                _print_bundle_inspect(inspect_result)
                if inspect_result.profile_comparison is not None:
                    _print_profile_comparison(inspect_result.profile_comparison)
            return 1 if args.check and inspect_result.roles.ready is not True else 0

        if args.command == "bundle" and args.bundle_command == "variants":
            listing = list_bundle_variants(
                args.bundle,
                renderplan_root=args.renderplan_root,
                output_root=args.output_root,
            )
            if args.json:
                print(json.dumps(listing.to_dict(), ensure_ascii=False, indent=2))
            else:
                _print_bundle_variants(listing)
            return 0

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
        if args.command == "bundle" and args.bundle_command in {"roles", "inspect"}:
            error_code = getattr(exc, "code", None)
            if not isinstance(error_code, str):
                error_code = "bundle.invalid"
            if args.json:
                format_name = (
                    "voicerender-bundle-roles"
                    if args.bundle_command == "roles"
                    else "voicerender-bundle-inspect"
                )
                print(
                    json.dumps(
                        {
                            "format": format_name,
                            "version": 1,
                            "error": str(exc),
                            "error_code": error_code,
                            "exit_code": 2,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            else:
                print(f"error: {exc}", file=sys.stderr)
            return 2
        if (
            getattr(args, "json", False)
            and args.command == "bundle"
            and args.bundle_command == "variants"
        ):
            error_code = getattr(exc, "code", None)
            payload = {
                "format": "voicerender-bundle-variants",
                "version": 1,
                "error": str(exc),
                "error_code": error_code
                if isinstance(error_code, str)
                else "bundle.invalid",
                "exit_code": 2,
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 2
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
        elif getattr(args, "json", False) and (
            args.command in {"engines", "lexicons", "models", "voices"}
            or args.command == "runtime"
            or (
                args.command == "profile"
                and args.profile_command
                in {"create", "inspect", "show", "role", "bind", "default-role"}
            )
            or (args.command == "voice" and args.voice_command in {"source", "clone"})
        ):
            payload = {
                "status": "error",
                "error": str(exc),
                "error_code": getattr(exc, "code", "input.invalid"),
                "details": dict(getattr(exc, "details", {})),
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1
    parser.error("unsupported command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
