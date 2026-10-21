# VoiceRender

VoiceRender executes shared [RenderPlan](https://pypi.org/project/renderplan/) files through lazy, request-level synthesis engine adapters. It provides RenderProfile lifecycle tools, no-synthesis preflight reports, exact one-item/one-WAV rendering, content-addressed reuse, and verifiable shared manifests. Audio composition, external-media resolution, mastering, and export are outside VoiceRender.

The package lives in `voicerender/`; engine SDKs and model assets are installed separately.

## Install

Core installation:

```bash
python -m pip install voicerender
```

Install only the engine SDK you need:

```bash
python -m pip install "voicerender[piper]"
python -m pip install "voicerender[sopro]"
```

For development from a checkout:

```bash
python -m pip install -e ".[dev]"
```

Available engine extras are `kokoro`, `piper`, `pocket`, `kitten`, `sopro`, `supertonic`, `inflect`, and `qwen3`. The core package can list engines and import without installing their SDKs. Tests use deterministic fake adapters and do not download models.

### Kitten request measurement

KittenSynth owns Kitten text tokenization. VoiceRender delegates to `KittenVoice.measure_prepared()` and does not re-tokenize Kitten text. Kitten provides exact `model_tokens` measurement for preflight; a hard maximum is only available when the engine/model declares one. An unknown maximum means capacity is unknown, not infinite. The `.[kitten]` extra requires KittenSynth 0.1.4 or newer, which provides this measurement API.

### Piper request measurement

PiperSynth owns the authoritative phoneme-capacity decision, and the `.[piper]` extra requires PiperSynth `>=0.2.5,<0.3`. VoiceRender maps its exact measurement as follows:

| Piper measurement                           | VoiceRender result |
| ------------------------------------------- | ------------------ |
| `fits=True`, finite maximum                 | `ok`               |
| `fits=True`, `maximum=None`, variable graph | `ok`               |
| `fits=False`, finite hard maximum           | `replan`           |
| `fits=None`                                 | `reject`           |

`maximum=None` is not itself an error. `fits=None` is the fail-closed state: VoiceRender rejects preflight without attempting synthesis. Native PiperSynth measurement metadata, including capacity kind/source when provided, is retained in `RequestMeasure.details.native_metadata`; VoiceRender does not infer fit from that metadata.

## CLI workflow

A planning tool produces a shared RenderPlan whose `render_profile_id` and role bindings match the selected RenderProfile. Create the first/default role of a complete cast through adapter discovery when needed; see the [RenderProfile lifecycle](docs/renderprofile.md):

```bash
voicerender engines list
voicerender voices list --engine piper --lang en-US --offline --json
voicerender profile create \
  --engine piper --target en_US-amy-medium --language en-US \
  --offline --output narration.renderprofile.toml
voicerender profile validate narration.renderprofile.toml
```

Parsing a RenderProfile is engine-independent. `voicerender profile validate` checks the selected adapter/API, so it requires that engine's SDK; it does not perform model inference.

The created `profile_id` must be used by the planner when it emits a RenderPlan. For a complete serialized example, the checked-in plan and profile below already share an identity:

```bash
voicerender profile validate examples/voices/narration.renderprofile.toml
voicerender preflight examples/plans/chapter.renderplan.toml \
  -o chapter.preflight.toml --json
voicerender render examples/plans/chapter.renderplan.toml \
  --preflight chapter.preflight.toml \
  --out-dir renders/narration \
  --cache-dir .voicerender-cache
voicerender inspect-manifest renders/narration/render-manifest.toml --json
```

Install the selected engine extra and make its model assets available before preflight/render; VoiceRender does not download models implicitly. Preflight reports `ok`, `replan`, or `reject` for each required speech occurrence. An `ok` report approves that exact RenderPlan under the observed renderer capability; rendering still remeasures each exact request immediately before synthesis. If fresh capacity no longer fits, VoiceRender stops before synthesis for that occurrence and returns a new same-plan/same-occurrence `replan` report; a session-open failure returns a fresh `reject` report. The CLI writes that report to `--recheck-report` when supplied. VoiceRender never creates a child plan, changes text, or adapts chunking. Every RenderPlan speech item maps to exactly one public `SpeechRequest` and one numbered WAV, in plan order; VoiceRender does not split, merge, normalize, or reorder requests. Incremental manifests remain valid when rendering fails partway through. Render results expose an execution fingerprint (VoiceRender/adapter versions, API-probe signature, synthesis identity, voice/options, and measured capacity) separately from `RenderPlan.plan_id`.

### Deferred emphasis

VoiceRender does not apply emphasis during speech synthesis and does not advertise `emphasis` as a native Piper feature. By default, emphasis is rejected. An orchestration pipeline that declares the `audiocompose.emphasis.gain.v1` handler must select `--emphasis-policy defer-to-compose` for **both** preflight and render (or for `bundle render`):

```bash
voicerender preflight chapter.renderplan.toml \
  -o chapter.preflight.toml --emphasis-policy defer-to-compose
voicerender render chapter.renderplan.toml \
  --preflight chapter.preflight.toml --out-dir renders/narration \
  --emphasis-policy defer-to-compose
```

The accepted emphasis parameters are `{}` or one of `{ "level": "reduced" }`, `{ "level": "moderate" }`, or `{ "level": "strong" }`; unknown keys, types, and invalid ranges are rejected. Other unsupported controls remain fail-closed. VoiceRender sends the original text to the engine without an emphasis option, preserves the original controls and identities in the RenderPlan, and leaves control interpretation—including overlapping emphasis—to AudioCompose.

Deferred controls are explicitly pending composition, not fulfilled by the standalone WAV. Their count and capability appear in the preflight check detail and in the render output's `render-diagnostics.json` sidecar, which warns that any produced WAVs contain unchanged speech. Human render output warns when composition is pending, and render JSON plus `bundle render` reports expose `pending_composition`. The shared RenderManifest schema is unchanged; AudioCompose must use the original RenderPlan as the source of truth when applying the declared effect.
The checked-in `examples/voices/narration.renderprofile.toml` and `examples/plans/chapter.renderplan.toml` form an identity-matched example pair. The profile records a discovery snapshot; refresh and regenerate the plan if adapter metadata changes. See `examples/README.md` for the complete command sequence.

Profiles are engine-refreshed proposals: refresh metadata without overwriting user policy, then review the proposal before using it. For example:

```bash
voicerender profile refresh narration.renderprofile.toml \
  -o narration.refreshed.renderprofile.toml
```

## Engine, voice, and profile control surface

VoiceRender exposes engine-facing discovery and lifecycle operations through its public Python API and CLI. Discovery does not open synthesis sessions; model installation/removal is delegated only to an engine's supported public asset API. Unsupported asset operations report the engine and operation rather than falling back to private modules or backend CLIs.

```bash
# Inspect normalized options and available model/voice descriptors.
voicerender engines options kokoro --target v1.0 --json
voicerender models list --engine kokoro --offline --json
voicerender voices list --engine kokoro --lang en-US --json
voicerender voices inspect 'kokoro:v1.0/af_heart' --json

# Asset support is engine-specific; supported removals require explicit confirmation.
voicerender models inspect TARGET --engine ENGINE --json
voicerender models verify TARGET --engine ENGINE --json
voicerender models remove TARGET --engine ENGINE --yes --json

# Pin a reference without copying or modifying its WAV; inspect and verify it later.
voicerender voice source create \
  --reference-wav ./speaker.wav --transcript "Exact spoken words." \
  --language en-US -o speaker.voice-source.toml
voicerender voice source inspect speaker.voice-source.toml --json
voicerender voice source verify speaker.voice-source.toml --json

# Bind a pinned source to a reference-capable target; this does not synthesize.
voicerender voice clone --engine ENGINE --target TARGET \
  --language en-US --voice-source speaker.voice-source.toml \
  --transcript "Exact spoken words." -o clone.renderprofile.toml --json
```

RenderProfile is the complete cast for one render variant. Each logical role is bound independently, so one profile can mix engines and targets while sharing planning and pause policy. Create its initial/default role, bind the rest, and inspect the whole cast:

```bash
voicerender profile create \
  --role narrator --engine kokoro --target v1.0 --voice af_heart \
  --language en-US --offline -o printer-cast.renderprofile.toml
voicerender profile bind printer-cast.renderprofile.toml host \
  --engine kitten --target mini-0.8 --voice Jasper \
  --language en-US --offline --in-place
voicerender profile bind printer-cast.renderprofile.toml guest \
  --engine piper --target en_US-amy-medium \
  --language en-US --offline --in-place
voicerender profile show printer-cast.renderprofile.toml
voicerender profile validate printer-cast.renderprofile.toml
```

The human `profile show` view displays every role's engine, target, language, and voice/speaker/source. `--json` preserves the complete canonical binding dictionaries. `profile inspect` remains available; `profile role set` is the structured role-management equivalent. Role replacement revalidates the selected engine/target, refreshes derived requirements and capacity, preserves other roles and shared policy, and computes a new `profile_id`.

## Reference voices

Kokoro, PocketSynth, SoproSynth, and ChatterboxSynth accept pinned `voice_source` bindings. A local `kind="reference"` source stores an absolute path and computed SHA-256: it is usable in a local profile but is machine-specific, and changed or missing bytes fail validation instead of being silently repinned. A Voicefolio `kind="managed_reference"` source adds resolver, source revision, repository, and path pins so another machine can resolve the same asset.

Create a local Inno voice or a portable AkinVox voice with the CLI. AkinVox requires the exact spoken transcript; Inno does not use a transcript and may accept an `fmax` option:

```bash
voicerender profile create \
  --engine kokoro --target v1.0 --language en-US \
  --reference-wav ./speaker.wav \
  --option 'voice_enrollment={"engine":"auto","fmax":8000}' \
  --output local-kokoro.renderprofile.toml

voicerender profile create \
  --engine kokoro --target en-akinvox-cloning-v1 --language en-US \
  --voicefolio-ref kyutai-tts-voices:alba-mackenna/a-moment-by \
  --option 'voice_enrollment={"engine":"akinvox","reference_text":"Exact words spoken in the clip."}' \
  --output managed-kokoro.renderprofile.toml
```

`engine="auto"` selects Inno for a static target advertising the `inno-v0.2` enroller, or AkinVox for a reference-only target. AkinVox is English-only. Install `.[kokoro]` for Kokoro or `.[sopro]` for SoproSynth and make model assets available using the engine's supported process; VoiceRender does not download models implicitly.

Preflight validates local content pins and managed catalog metadata, and measures text without fetching managed audio, enrolling a voice, or conditioning a reference. Rendering resolves the verified reference and enrolls it lazily; the enrolled voice is reused within that synthesis session. Canonical identity retains content/provenance pins and the exact AkinVox transcript digest, but excludes local/cache paths and raw transcript text. See the [RenderProfile lifecycle](docs/renderprofile.md) and [examples](examples/README.md) for details.

### Kokoro quality and voice-level options

Inspect Kokoro's accepted settings and their structured field descriptions with `voicerender engines options kokoro --json`. Existing string shorthand remains supported: `short_sentence` accepts `off`, `wrap`, `phrase`, or `randomized-phrase`; `voice_level` accepts `off` or `calibrated`. The structured forms expose the PyKokoro controls without adding VoiceRender-side audio processing:

```bash
voicerender profile create \
  --engine kokoro --target v1.0 --voice af_heart --language en-US \
  --option 'short_sentence={"enabled":true,"resolve_mode":"phrase","min_phoneme_length":28}' \
  --option 'voice_level={"mode":"calibrated"}' \
  --output kokoro.renderprofile.toml

# Explicit PyKokoro gain; also available when using an enrolled voice.
voicerender profile bind kokoro.renderprofile.toml narrator \
  --engine kokoro --target v1.0 --voice af_heart --language en-US \
  --option 'voice_level={"mode":"off","gain_db":-1.5}' --in-place
```

`short_sentence` tables accept `enabled`, `resolve_mode` (`wrap`, `phrase`, or `randomized-phrase`), `min_phoneme_length` (default 30), `phoneme_pretext` (default `—`), and `phrase_fallback_tries` (default 1). Omitting the option leaves this PyKokoro quality fix unset; the `off` shorthand disables it. `voice_level` tables accept `mode` (`off` or `calibrated`) and finite numeric `gain_db`; an explicit gain takes precedence over the static calibration catalog. Automatic calibration is catalog-based for known model/quality/voice combinations and is unavailable for custom Inno/AkinVox enrolled voices, where an explicit gain may be used instead. These settings are PyKokoro's deterministic voice-level controls, not VoiceRender loudness normalization or mastering. VoiceRender separately adapts finite PCM overshoots at its artifact boundary; calibration does not guarantee a waveform will never need that adaptation.

The enrollment paths remain distinct: **Inno** is a reference-audio voice tuner, does not require a transcript, and may accept `fmax`. **AkinVox** is transcript-conditioned cloning: it requires the exact non-empty spoken transcript, is English-only, and currently requires synthesis speed `1.0`. `voicerender voice clone --engine kokoro --target en-akinvox-cloning-v1 --reference-wav speaker.wav --transcript "Exact words spoken." --language en-US -o clone.renderprofile.toml` binds the generic transcript into AkinVox's enrollment options. Conflicting transcript sources and transcripts for Inno are rejected.

### ChatterboxSynth

ChatterboxSynth is an optional English-only engine. Install the integration with `python -m pip install "voicerender[chatterbox]"`; VoiceRender discovers supported bundles (currently Nano v2 and Turbo) through ChatterboxSynth and opens only profiles already installed with the ChatterboxSynth/OnnxVoice tooling. Rendering and preflight never download or install a model.

Every profile requires a pinned reference voice. Both a local WAV and a managed Voicefolio reference are supported; named voices and speaker IDs are not. For example:

```bash
voicerender profile create \
  --engine chatterbox --target nano-v2 --language en-US \
  --reference-wav ./speaker.wav \
  --output chatterbox.renderprofile.toml

voicerender profile create \
  --engine chatterbox --target turbo --language en-US \
  --voicefolio-ref <voicefolio-ref> \
  --output chatterbox-turbo.renderprofile.toml
```

Preflight performs exact Chatterbox tokenizer measurement without resolving or preparing the reference. Current bundles publish no finite text-input token maximum, so VoiceRender does not invent a capacity limit or split text; `max_new_tokens` controls generated output, not input size. Rendering resolves and verifies the pinned reference lazily, prepares it once per session, and uses the common artifact and manifest path.

### Qwen3Synth Base and VoiceDesign

Qwen3Synth is optional. Install its extra and the model bundles you intend to use. VoiceRender opens installed models only, and does not download them during profile creation, preflight, or rendering:

```bash
python -m pip install "voicerender[qwen3]"
onnxvoice install qwen3:voice-design-1.7b --quality int4
onnxvoice install qwen3:base-0.6b --quality int4
```

Generate one fixed reference utterance explicitly. The optional sidecar records the exact recipe and runtime/model provenance while pinning the resulting WAV bytes:

```bash
voicerender voice design \
  --prompt "A warm, calm narrator with clear diction." \
  --text "This fixed sentence will be spoken by the designed voice." \
  --language en \
  --output narrator.wav \
  --source-output narrator.voice-source.toml

voicerender voice inspect narrator.voice-source.toml --json
```

The materialized WAV hash is the reference voice identity. The prompt, seed, model revision, and runtime identity are regeneration provenance. Protect the WAV and sidecar as sensitive voice data. The sidecar includes the prompt and exact reference text.

To prepare a Qwen Base `.qvoice`, reuse the exact transcript from a verified Qwen VoiceDesign sidecar, or provide `--transcript` for an ordinary WAV. VoiceRender never guesses a transcript:

```bash
voicerender voice prepare-qwen \
  --voice-source narrator.voice-source.toml \
  --output narrator.qvoice
voicerender voice inspect narrator.qvoice --json

voicerender profile create \
  --engine qwen3 --target base-0.6b --language en-US \
  --prepared-voice narrator.qvoice \
  --output qwen-narrator.renderprofile.toml
```

You can use a pinned WAV directly with Qwen Base by passing `--reference-wav` and an exact `--option 'reference_text=...'`, or pass a Qwen-origin source sidecar with `--voice-source`. Prepared `.qvoice` profiles use their complete file hash as identity. Qwen Base requires the Qwen extra; profile creation and validation inspect metadata but do not synthesize.

For a weaker machine, copy the materialized WAV and create a new local profile for an installed reference-capable engine such as PocketSynth:

```bash
python -m pip install "voicerender[pocket]"
voicerender voices list --engine pocket --offline
voicerender profile create \
  --engine pocket --target <pocket-target> --language en-US \
  --reference-wav ./narrator.wav \
  --output pocket-narrator.renderprofile.toml
```

The new profile pins the copied WAV on that machine. Do not reuse the creator's local profile or source sidecar unchanged, because their absolute WAV path is machine-specific. PocketSynth consumes the pinned WAV and does not need or import Qwen3Synth; Qwen provenance is not an execution dependency. See the [RenderProfile lifecycle](docs/renderprofile.md) for source and portability details.

## Unified SSMD bundle rendering

The canonical editable SSMD workspace is `<name>.ssmd-bundle/`. Its `manifest.json` declares whether it contains a book or one document; VoiceRender reads that authoritative format and dispatches to the matching existing RenderPlan renderer. Book and document rendering use the same command and require the bundle root, never an inner file or subdirectory.

The coordinated pipeline uses the `bundle` command group at every stage:

```bash
ssmdconvert bundle convert story.ssmd.md
utterplan bundle compile story.ssmd-bundle
utterplan bundle render story.ssmd-bundle --profile narration.renderprofile.toml
voicerender bundle render story.ssmd-bundle --variant narration--<profile-fingerprint>
```

For a book, render selected source chapters from the ordered RenderPlan set. The variant is the exact profile-fingerprinted directory name:

```bash
voicerender bundle render Novel.ssmd-bundle \
  --variant narration--<profile-id-prefix> \
  --chapters 5-17 --cache-dir .voicerender-cache --json
```

Omit `--chapters` to select all chapters. A speech QC window selects one source chapter and a one-based speech range; verified WAVs accumulate into the same chapter manifest. `--fail-fast` skips remaining chapters after a failure. `--renderplan-root`, `--output-root`, and `--cache-dir` override their roots. `--reset-stale` replaces stale VoiceRender-owned output while keeping cache entries available for verified reuse; `--force` synthesizes every selected request and bypasses cache.

A document is one render item and accepts only `--chapters all`:

```bash
voicerender bundle render story.ssmd-bundle \
  --variant printer-cast--<profile-fingerprint> \
  --speech 1-20 --reset-stale
```

For either kind, VoiceRender stores output below `<bundle>/voicerender/<variant>/`. The existing internal book and document layouts remain kind-specific, including their reports and manifest locations. Generic `voicerender preflight` and `voicerender render` remain available for individual RenderPlans.

## Python API

```python
from renderplan import load_plan
from voicerender import preflight_render_plan, render_renderplan

plan = load_plan("chapter.renderplan.toml")
report = preflight_render_plan(plan, "chapter.preflight.toml")
if report.status == "ok":
    result = render_renderplan(
        plan,
        preflight=report,
        output_dir="renders/narration",
        cache_dir=".voicerender-cache",
    )
    print(result.manifest_path)
```

### Live progress, ETA, and workload estimates

`render`, `bundle render`, and `preflight` accept `--progress` and `--no-progress`. By default progress is enabled only for interactive stderr and non-JSON output. `--progress` forces it (including with `--json`), but progress always stays on stderr; JSON stdout remains a single unchanged result. Forced redirected output uses throttled newline records without ANSI escapes or carriage returns. Use `--no-progress` to disable it.

Python callers can subscribe to typed events during rendering and preflight. Events carry progress counts, phase/chapter context, elapsed time, and an ETA when synthesis timing evidence is available:

````python
from voicerender import RenderEvent, RenderEventKind, render_bundle


def on_event(event: RenderEvent) -> None:
    if event.kind is RenderEventKind.SEGMENT_RENDERED:
        print(event.progress, event.eta_seconds, event.eta_confidence)


result = render_bundle(
    "Novel.ssmd-bundle", variant="narration--<profile-id>", on_progress=on_event
)

The keyword-only `on_event` callback is also available on `preflight_render_plan()`, `render_renderplan()`, and the legacy `render()`/`render_plan()` APIs. Preflight events report required-occurrence work, but its capacity measurements are not synthesis timing and do not produce an ETA.

`estimate_renderplan()` and `estimate_render_book()` provide side-effect-free workload and verified existing-output reuse counts. Without a calibrated timing model, they return `estimated_seconds=None`, `confidence="unavailable"`, and `source="workload-only"`. Live ETA becomes available only after successful synthesis duration samples exist; it is approximate and can remain unknown for work whose engine has no timing evidence.

```python
from voicerender import estimate_renderplan

estimate = estimate_renderplan(plan, output_dir="renders/narration")
print(
    estimate.selected_count, estimate.known_output_reuse_count, estimate.pending_count
)
````

Preflight capacity values describe whether a request fits the engine's limits; they are not wall-clock predictions and are never converted into duration estimates.

For an emphasis-bearing plan, the Python API uses the same safe default (`reject`). A pipeline that declares the composition handler must pass the opt-in policy to both stages:

```python
report = preflight_render_plan(
    plan, "chapter.preflight.toml", emphasis_policy="defer-to-compose"
)
result = render_renderplan(
    plan,
    preflight=report,
    output_dir="renders/narration",
    emphasis_policy="defer-to-compose",
)
```

Third-party engines can implement the request-level adapter protocol in `voicerender.engines.base` and register with `voicerender.engines.register_adapter`. Engine implementations and model assets remain lazily loaded.

The older VoicePlan-oriented Python API remains available for compatibility, but the primary CLI workflow uses shared RenderPlan and RenderProfile contracts.

## Boundaries

- **No implicit model downloads:** install SDKs and assets through each engine's supported process.
- **Exact requests:** capacity overflow is reported for replanning; the RenderPlan renderer does not create adaptive child requests.
- **Verified reuse:** cached WAVs are checked by digest and recorded WAV metadata before reuse.
- **No audio post-processing:** VoiceRender writes speech WAV artifacts and a shared manifest only.
- **Portable managed voices:** local `kind="reference"` paths are content-pinned and usable but machine-specific; pinned Voicefolio managed references are the portable choice.

## Tests

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
ruff check .
mypy --strict voicerender
```

## License

Apache-2.0 (see `LICENSE`). VoiceRender has no Readio dependency.
