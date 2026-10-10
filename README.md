# VoiceRender

**Standalone, resumable multi-engine speech synthesis and reference-voice rendering for Python.**

VoiceRender is a standalone UtterPlan v5 speech producer (and retains the legacy `voiceplan.toml` API). It writes resumable mono WAV bundles with a verifiable render manifest for downstream consumers. VoiceRender does **not** resolve external media, compose, master, or encode audio. The flat package lives in `voicerender/` (no `src/` directory).

> **Status:** Legacy VoicePlan rendering remains supported. UtterPlan v5 accepts only the documented subset of source semantics and rejects unsupported directives before synthesis. Source-addressed manifests preserve media selections, semantic lowering, engine identity, timing, and verified speech artifacts; downstream composition remains outside VoiceRender. Engine SDKs and model assets are installed separately.

## Install

Extract the ZIP, open a shell in the extracted project directory and run:

```bash
python -m pip install -e ".[piper]"
# or with PocketSynth reference-voice support:
python -m pip install -e ".[pocket]"
```

Available request-centric engine extras are `kokoro`, `piper`, `pocket`, `kitten`, `supertonic`, and `inflect`; install only the SDKs you need.

You can also install a built wheel, or use `python -m voicerender` while working from the source tree with its runtime dependencies installed.

### Dynamic versioning

Versioning uses **`setuptools-scm`** (`[project].dynamic = ["version"]`) and Git tags, with the generated Python version module at `voicerender/_version.py` during builds. There is no hard-coded project version and no `src/` layout. Tag releases with `v0.1.0`, `v0.2.0`, etc. Building from an untagged source archive without Git metadata uses the non-release fallback `0+unknown`.

```bash
git init
git add .
git commit -m "Initial VoiceRender MVP"
git tag v0.1.0
python -m pip install build
python -m build
voicerender --version
```

Setuptools-scm's VCS-based version changes when tags/commits change; build from the tagged repository to get the intended release version. For reproducible source distributions, use the build-produced sdist rather than re-zipping a checkout.

## Quick start

The examples directory includes a minimal UtterPlan v5 input and voice configuration, plus the legacy VoicePlan/profile examples.

### Direct UtterPlan v5

Install the matching engine extra and make its selected model available before rendering; for example, `python -m pip install -e ".[piper]"`.
Validate without inference, render, and verify the resulting handoff:

```bash
voicerender validate examples/plans/chapter.utterplan.toml \
  --voice examples/voices/voice.toml
voicerender render examples/plans/chapter.utterplan.toml \
  --voice examples/voices/voice.toml \
  --out-dir renders/voice1
voicerender inspect renders/voice1/render-manifest.json --verify --json
```

`--take N` selects the first N source operations; `--select f000000.s000` selects by source locator (`--select` and `--take` are mutually exclusive). The v5 renderer resumes verified outputs automatically; `--force` bypasses reuse. `--profile` / `--voices` remain supported for the legacy VoicePlan path.

Consumers can load the source and verify its relationship to the complete artifact manifest through the narrow handoff API:

```python
from voicerender import (
    load_source_plan,
    validate_manifest_source,
    verify_render_manifest,
)

plan = load_source_plan("examples/plans/chapter.utterplan.toml")
manifest = verify_render_manifest("renders/voice1/render-manifest.json")
validate_manifest_source(plan, manifest)
```

The manifest and verified WAV files are the standalone handoff. External-media resolution, pause policy, composition, mastering, and export belong to downstream tools.

### Legacy VoicePlan

```bash
voicerender engines list
voicerender voices list --engine piper --lang en-US
voicerender profiles validate examples/voices/piper.toml
voicerender render examples/plans/chapter.voiceplan.toml \
  --profile examples/voices/piper.toml \
  --take 2 \
  --out-dir renders/legacy
voicerender inspect renders/legacy/manifest.json --json
```

Use a separate `--out-dir` per voice profile. The legacy renderer shares a content-addressed cache by default at `OUT_DIR/../.voicerender-cache`; use `--cache-dir` to share a cache across directory structures.

## Python API

```python
from voicerender import (
    SpeechSegment,
    VoiceProfile,
    render,
    render_plan,
    render_utterplan_v5,
)

# From a persisted voiceplan:
result = render_plan(
    "examples/plans/chapter.voiceplan.toml",
    profile="examples/voices/piper.toml",
    output_dir="renders/voice1",
    take=2,
)
print(result.manifest_path)

# Directly from UtterPlan v5 with a role-routed voice.toml:
v5_result = render_utterplan_v5(
    "examples/plans/chapter.utterplan.toml",
    profile="examples/voices/voice.toml",
    output_dir="renders/utterplan-v5",
)

# Without UtterPlan or any file-based plan:
segments = [SpeechSegment(id="hello", text="Hello there!", role="narrator")]
profile = VoiceProfile.load("examples/voices/piper.toml")
result = render(segments, profile=profile, output_dir="renders/hello")
```

For third-party engines, implement the small engine protocol in `voicerender.engines.base` and register it using `voicerender.engines.register_engine(adapter)`. See `tests/test_renderer.py` for a fully executable mock adapter. The core imports no synthesis backend until one is opened.

## UtterPlan v5 voice configuration

Direct v5 rendering uses the `voice.toml` form (`schema_version = 1`), with optional defaults and one or more logical role bindings. See `examples/voices/voice.toml`. A role can bind an `engine` and `model` (or `target_id`), and optionally define native engine options, a language-specific voice, speaker, reference audio, or a cache-only `model_revision`. Relative reference-audio paths resolve from this file.

```toml
schema_version = 1
name = "narration"
[defaults]
role = "narrator"
[roles.narrator]
engine = "piper"
model = "en_US-amy-medium"
```

## Legacy VoiceProfile schema

```toml
schema = "voicerender.voice-profile.v1"
name = "narration"

[roles.narrator]
engine = "piper"
target_id = "en_US-amy-medium"

[roles.narrator.options]
native_speed = 1.1  # maps to Piper length_scale = 1 / native_speed
```

### Reference-voice cloning (PocketSynth)

```toml
schema = "voicerender.voice-profile.v1"
name = "reference-narrator"

[roles.narrator]
engine = "pocket"
target_id = "YOUR_INSTALLED_BUNDLE_ID"
reference_audio = "../recordings/my-consented-speaker.wav"

[roles.narrator.options]
precision = "int8"
temperature = 0.6
```

Paths to reference audio resolve relative to the profile file and are checked before rendering. The **SHA-256 of reference-audio bytes** participates in the acoustic cache key, not the local path, so two identical recordings can share synthesized output. Reference audio is not copied into the render bundle. Use reference recordings you have the right and consent to use; do not misrepresent cloned audio as the speaker's own recording.

`model_revision` is an optional per-role declaration used **only for cache identity**. It does not download or pin the corresponding model revision. Exact model-asset resolution and revision verification need backend-specific follow-up work; changing the configured string changes cache identity but does not change the loaded model.

VoiceRender profile options describe engine-native controls such as Piper `native_speed`. Versioned numeric composition directives are recorded for downstream consumers; VoiceRender does not modify rendered speech audio. PocketSynth does not support Piper's native speed control.

## Persisted contracts

### Input: UtterPlan v5

VoiceRender consumes a UtterPlan v5 TOML export directly; it flattens flow units and segments in source order and addresses each segment as `f{flow_index:06d}.s{segment_index:03d}`. The sample `examples/plans/chapter.utterplan.toml` demonstrates the minimal supported shape. See [`docs/utterplan-v5-integration.md`](docs/utterplan-v5-integration.md) for supported semantics, strict rejection rules, voice routing, selection, timing, and composition boundaries.

Legacy consumers can continue using the custom `utterplan.voiceplan.v1` input and `render_plan()` API. This is a VoiceRender compatibility format, not a replacement for UtterPlan v5.

### Output: `voicerender.render-manifest.v2`

```text
renders/voice1/
  render-manifest.json
  segments/
    <source-addressed-render>.wav
```

Manifest v2 preserves source plan and segment identity separately from versioned VoiceRender semantic identities, records selection, engine/API identity, acoustic cache keys, timing provenance and the `preserved-unmeasured-v1` edge policy. Every completed WAV is checked against its relative safe path, SHA-256 and recorded WAV metadata before reuse or handoff. The manifest is updated atomically as segments move through `pending|rendered|reused|failed`; `voicerender inspect` verifies v2 media by default (and accepts explicit `--verify`).

### Legacy output: `voicerender.render-manifest.v1`

The legacy VoicePlan renderer continues writing v1 manifests with ordered `scope_id`/`segment_id`, engine and voice identity, WAV path and digest, timing, and per-segment status. `inspect` continues to verify v1 bundles.

## Operational boundaries and known limitations

- **Model execution:** test adapters exercise public backend contracts, but validate real Piper/Pocket synthesis with installed engines and model assets in the deployment environment.
- **Fail-closed v5 subset:** unsupported source directives and document semantics are rejected rather than discarded. Capacity fitting uses engine measurements, deterministic legal splitting, and timing rebasing.
- **Synthesis variability:** stochastic backends may vary bit-for-bit; verified outputs are reused for matching acoustic inputs. Stable backend options/seeds are engine-specific.
- **Composition and export:** VoiceRender stores speech WAVs and verified source provenance only. External media resolution, pause policy, composition, mastering, playback, and encoding are downstream responsibilities. Cloning is supported only by engines that explicitly accept reference audio.
- **Concurrency:** an exclusive lock prevents overlapping runs in one output folder. Shared cache writes are atomic, but multi-process computation de-duplication and stale-lock recovery are not implemented.
- **Cache invalidation:** legacy cache identity uses engine, target, profile options, text, language, revision, and reference-audio bytes. V5 keys use effective acoustic inputs and exclude source position and semantic pauses. Set a new `model_revision` or clear the cache when model changes are not reflected in engine identity.

## Tests

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m voicerender --help
python -m voicerender engines list
```

Tests use synthetic arrays and fake engine modules; they do not require downloaded model weights. Validate real Piper/Pocket synthesis separately before production use.

## License

Apache-2.0 (see `LICENSE`). The initial adapters follow the public request-level contracts already used by Readio; VoiceRender is a separate package and has no Readio dependency.
