# VoiceRender

**Standalone, resumable multi-engine speech synthesis and reference-voice rendering for Python.**

VoiceRender turns an ordered, speech-only `voiceplan.toml` plus a TOML voice profile into a portable bundle of mono WAV segments and a JSON manifest. It does **not** depend on Readio, parse SSMD, compose pauses, master audio or encode MP3/M4B. The flat package lives in `voicerender/` (no `src/` directory).

> **Status: MVP source release.** PiperSynth and PocketSynth are optional backend adapters based on the released APIs exercised by Readio. Only stubbed backend-contract tests and mocked audio unit tests have run for this ZIP; real synthesizer/model execution is **not yet verified** here. Full UtterPlan linguistic-token/prosody exports and capacity-aware request splitting are deliberately not implemented. Unsupported segment fields fail validation rather than being silently dropped.

## Install

Extract the ZIP, open a shell in the extracted project directory and run:

```bash
python -m pip install -e ".[piper]"
# or with PocketSynth reference-voice support:
python -m pip install -e ".[pocket]"
```

You can also install a built wheel, or use `python -m voicerender` while working from the source tree with its runtime dependencies installed.

### Dynamic versioning

Versioning uses **`setuptools-scm`** (`[project].dynamic = ["version"]`) and Git tags, with the generated Python version module at `voicerender/_version.py` during builds. There is no hard-coded project version and no `src/` layout. Tag releases with `v0.1.0`, `v0.2.0`, etc. Building from a ZIP without `.git` uses the explicitly declared bootstrap fallback `0.1.0`.

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

The archive includes `examples/plans/chapter.voiceplan.toml`, `examples/voices/piper.toml`, and a Pocket reference-voice example.

```bash
voicerender engines list
voicerender voices list --engine piper --lang en-US
voicerender profiles validate examples/voices/piper.toml
voicerender render examples/plans/chapter.voiceplan.toml \
  --profile examples/voices/piper.toml \
  --take 2 \
  --out-dir renders/voice1
voicerender inspect renders/voice1/manifest.json --json
```

The **primary command is always `voicerender render`**. `--take N` counts logical speech segments, not backend-internal chunks or silence events. Selection by ID is available using `--select chapter-0001:seg-000002` (comma-separate multiple IDs). `--force` bypasses cached audio. The renderer resumes automatically when called again with the same output directory. Use `--json` on `render` for machine-readable summaries.

For ten profiles, render each with its own `--out-dir`. VoiceRender shares a content-addressed cache by default at `OUT_DIR/../.voicerender-cache`; `--cache-dir` can be supplied explicitly to share a cache across directory structures.

## Python API

```python
from voicerender import SpeechSegment, VoiceProfile, render, render_plan

# From a persisted voiceplan:
result = render_plan(
    "examples/plans/chapter.voiceplan.toml",
    profile="examples/voices/piper.toml",
    output_dir="renders/voice1",
    take=2,
)
print(result.manifest_path)

# Without UtterPlan or any file-based plan:
segments = [SpeechSegment(id="hello", text="Hello there!", role="narrator")]
profile = VoiceProfile.load("examples/voices/piper.toml")
result = render(segments, profile=profile, output_dir="renders/hello")
```

For third-party engines, implement the small engine protocol in `voicerender.engines.base` and register it using `voicerender.engines.register_engine(adapter)`. See `tests/test_renderer.py` for a fully executable mock adapter. The core imports no synthesis backend until one is opened.

## Profile schema

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

The MVP separates `native_speed` (engine-native acoustic control) from AudioCompose tempo. Piper supports native speed; PocketSynth does not.

## Persisted contracts

### Input: `utterplan.voiceplan.v1`

```toml
schema = "utterplan.voiceplan.v1"
source_plan_id = "chapter-001"

[[segments]]
id = "seg-001"
scope_id = "chapter-001"
role = "narrator"
language = "en-US"
text = "Once upon a time."
```

UtterPlan's proposed exported voiceplan is **not yet released**; this MVP implements the minimal proposed contract illustrated in the migration roadmap. Only `id`, `scope_id`, `role`, `language`, `text` are accepted for each segment. Native pronunciation overrides, linguistic tokens, alignment hints, whole-request phonemes and advanced multi-language lowering require a versioned follow-up contract and adapter.

### Output: `voicerender.render-manifest.v1`

```text
renders/voice1/
  manifest.json
  segments/
    000001-<hashed-id>.wav
    000002-<hashed-id>.wav
```

The manifest has ordered segment records with stable `scope_id`/`segment_id`, engine and voice identity, synthesis key, relative WAV path, SHA-256, frames, sample rate, channels, timing provenance and `pending|rendered|reused|failed` state. It persists after each segment. WAV files are mono 16-bit PCM. Render failures are recorded in `manifest.json`; `--continue-on-error` attempts remaining segments and returns a nonzero exit code if any failed. `voicerender inspect` verifies WAV hashes and structure.

VoiceRender does not synthesize silence for pauses: `composeplan.toml` and AudioCompose own speech ordering, inter-segment silence, gain/tempo/pitch, fades and mastering. AudioCompose's future manifest adapter is **not part of this MVP**.

## Operational boundaries and known limitations

- **Not yet real-model tested:** test adapters with installed `pipersynth` / `pocketsynth` and actual model downloads in your deployment environment. `tests/test_backends.py` stubs their documented public APIs.
- **Speech-only v1:** advanced UtterPlan attributes are rejected rather than discarded; long requests are submitted once and may be rejected by backend capacity limits. No engine-level auto-splitting yet.
- **No guarantees of bit-for-bit deterministic synthesis:** stochastic backends may vary; first verified WAV output is reused for a given acoustic key. Pass stable backend options/seed when available (not standardized here).
- **WAV only:** no playback, audio composition, FFmpeg/export, model fine-tuning, speaker-embedding training or managed voice-prompt catalog yet. Cloning is limited to Pocket reference-audio inference.
- **Concurrency:** exclusive lock within one output folder prevents accidental overlapping runs. Shared cache writes are atomic, but multi-process computation de-duplication and stale-lock recovery are not implemented yet.
- **Cache invalidation:** includes engine distribution version, target ID, profile-native options, configured model revision, spoken text, language and reference SHA-256. For model updates not reflected in a pinned revision or engine version, clear the shared cache or set a new `model_revision` explicitly.

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
