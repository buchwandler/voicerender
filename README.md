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

### Sopro request measurement

SoproSynth owns request tokenization and internal text segmentation. VoiceRender forwards its exact measurement and authoritative `fits` result; a finite `max_input_tokens` is optional. `fits=True` with `maximum=None` is accepted, `fits=False` requests replanning, and `fits=None` remains fail-closed in the engine-neutral preflight classifier.

The `.[sopro]` extra allows SoproSynth `>=0.1.0,<0.2`. For the known legacy releases 0.1.0 and 0.1.1 only, VoiceRender may infer `fits=True` from a valid exact token count when the maximum and producer fit are absent and the API-v1 contract confirms `supports_request_measurement=True` and `caller_owns_text_boundaries=False`. Other versions must provide fit authority themselves; unknown fit continues to reject preflight. The bridge does not estimate token limits from `max_segment_chars` or convert characters to tokens. Repeated identical unknown-fit failures also receive a concise CLI summary while every per-request check remains in the report.

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

`profile create` normalizes its destination to the canonical `.renderprofile.toml` suffix. For example:

```bash
voicerender profile create \
  --engine kokoro --target v1.0 --voice af_heart --language en-us \
  -o profile1
# Writes profile1.renderprofile.toml
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

### Engine failure diagnostics

A successful preflight does not guarantee that model execution will succeed: preflight checks declared engine capabilities and measures the request, but does not perform full inference or reference-voice enrollment. Session opening, immediate remeasurement, synthesis, and artifact writes can still fail during rendering.

On a render-time exception, `render-diagnostics.json` keeps the existing `voicerender.render-diagnostics.v1` schema identifier and adds `error_type` plus an `error_diagnostic` chain. The chain records the VoiceRender error, available engine diagnostic, and Python cause/context without parsing human error strings. Document/book reports and JSON/API results retain compact `error_stage`, `error_root_cause_type`, `error_root_cause_message`, and `render_diagnostics` fields; the JSON sidecar is the canonical full diagnostic.

The serializer is bounded and JSON-safe: it limits cause depth, message and detail sizes, handles cycles, and sanitizes nested values. It never persists traceback locals or arbitrary exception attributes, and omits sensitive speech text, transcripts, phonemes, audio, token sequences, and environment data. If diagnostic serialization fails, the original render error remains authoritative. Normal CLI output prints the stable error code/message plus stage, one root-cause summary, and sidecar path—never a traceback or nested JSON.

Kokoro uses PyKokoro's public `error_diagnostic` helper when the installed version exposes it. Older supported versions (including 0.10.5) continue with Python exception chaining and the available metadata; the optional dependency floor remains unchanged until a diagnostics-capable release is confirmed.
The checked-in `examples/voices/narration.renderprofile.toml` and `examples/plans/chapter.renderplan.toml` form an identity-matched example pair. The profile records a discovery snapshot; refresh and regenerate the plan if adapter metadata changes. See `examples/README.md` for the complete command sequence.

Profiles are engine-refreshed proposals: refresh metadata without overwriting user policy, then review the proposal before using it. For example:

```bash
voicerender profile refresh narration.renderprofile.toml \
  -o narration.refreshed.renderprofile.toml
```


For deliberate manual edits that make a stored profile ID stale, use the explicit recovery command; normal profile loading remains strict, and this does not refresh engine metadata:

```bash
voicerender profile reidentify story-cast6.renderprofile.toml --in-place
# Then inspect/validate and regenerate the RenderPlan variant for the new identity.
```

Use `profile refresh` for engine-derived metadata and capacities. Re-identification changes only the identity field and never rewrites existing RenderPlans or variant directories. See the [RenderProfile lifecycle](docs/renderprofile.md#repairing-a-stale-profile-identity).

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

## ONNX Runtime provider policy

VoiceRender exposes ONNXVoice's default ONNX Runtime provider policy without managing installed runtime distributions. An ONNX Runtime distribution (for example, `onnxruntime` or `onnxruntime-gpu`) is a Python package; an execution provider (for example, `CPUExecutionProvider` or `CUDAExecutionProvider`) is an execution backend reported by that runtime build. `voicerender runtime status` inspects the imported distribution, build, available providers, and configured/resolved policy. Resolved providers describe policy resolution, not the providers active in an existing session or proof that a session successfully used them.

The saved policy is a global default for compatible ONNXVoice-backed or provider-aware sessions. It does not imply that every VoiceRender engine uses ONNX Runtime. An explicit engine `providers` option remains an override and is passed through unchanged. Changing or resetting policy affects future sessions only; already initialized sessions are unchanged. `runtime use` does not install packages, switch distributions, or alter the Python environment.

ONNXVoice resolves defaults in this order: explicit request, `ONNXVOICE_PROVIDERS`, `ONNXVOICE_PROVIDER`, saved ONNXVoice configuration, then the CPU default. Environment variables therefore override a saved preference. `voicerender runtime reset` removes only the saved preference; environment overrides remain in force.

```bash
# Inspect installed runtime and provider policy.
voicerender runtime status
voicerender runtime list --json

# Set a saved default for future compatible sessions.
voicerender runtime use cpu
voicerender runtime use cuda
voicerender runtime use auto
voicerender runtime use cuda,cpu  # ordered provider preference

# Remove only the saved ONNXVoice preference.
voicerender runtime reset
```

## Reference voices

Kokoro, PocketSynth, SoproSynth, and ChatterboxSynth accept pinned `voice_source` bindings. A local `kind="reference"` source stores an absolute path and computed SHA-256: it is usable in a local profile but is machine-specific, and changed or missing bytes fail validation instead of being silently repinned. A VoiceFolio `kind="managed_reference"` source adds resolver, source revision, repository, and path pins so another machine can resolve the same asset.

Create a local Inno voice or use a portable VoiceFolio reference for AkinVox. A local `kind="reference"` pins an absolute path and SHA-256, so it is machine-specific. A managed `kind="managed_reference"` pins the VoiceFolio ref, source digest, source format, revision, repository, and path without storing a cache path.

### Primary multi-role VoiceFolio workflow

When the VoiceFolio catalog entry contains its transcript, the profile commands bind it automatically. No local reference WAV or Kokoro-specific `voice_enrollment` JSON is needed:

```bash
voicerender profile create \
  --role narrator --engine kokoro \
  --target en-akinvox-cloning-v1 --language en-US \
  --voicefolio-ref fine-read-it-voices:S5339 \
  -o story-cast.renderprofile.toml

voicerender profile bind story-cast.renderprofile.toml analyst \
  --engine kokoro --target en-akinvox-cloning-v1 --language en-US \
  --voicefolio-ref fine-read-it-voices:S5393 --in-place
```

If an entry has no catalog transcript, supply the exact spoken words through the generic fallback:

```bash
voicerender profile create \
  --role narrator --engine kokoro \
  --target en-akinvox-cloning-v1 --language en-US \
  --voicefolio-ref other:voice --transcript "Exact words spoken in the reference." \
  -o narrator.renderprofile.toml
```

AkinVox requires a non-empty exact transcript, supports English, and requires synthesis speed `1.0`. VoiceRender rejects conflicting catalog, `--transcript`, or advanced enrollment transcripts without printing their text. Catalog lookup and profile create/bind/validate are metadata-only: VoiceFolio resolves and verifies the audio lazily at render time. Inno remains available for static targets advertising its enroller; it does not require a transcript and may accept `fmax`. Install `.[kokoro]` for Kokoro or `.[sopro]` for SoproSynth and make model assets available using the engine's supported process; VoiceRender does not download models implicitly.

Advanced/manual engine options remain supported when needed, for example `--option 'voice_enrollment={"engine":"auto","fmax":8000}'` for Inno or an explicit Kokoro enrollment configuration. Prefer the generic VoiceFolio source and `--transcript` options for ordinary cloning.

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
voicerender bundle variants story.ssmd-bundle
# Render all incomplete valid variants; complete variants are skipped.
voicerender bundle render story.ssmd-bundle
# Include completed valid variants too.
voicerender bundle render story.ssmd-bundle --force
# Choose the exact existing variant basename from its state/progress listing:
voicerender bundle render story.ssmd-bundle --variant narration--<profile-fingerprint>
```

Bundle path arguments are resolved before their canonical root name is checked, so aliases such as `.` and `./` work when they resolve exactly to a `<name>.ssmd-bundle/` root:

```bash
cd story.ssmd-bundle
voicerender bundle render .
voicerender bundle variants .
```

The path must resolve to the bundle root itself; VoiceRender does not search parent directories, and a `renderplan/` directory, source file, chapter directory, or other inner path is not a bundle root. The bundle root identifies the workspace and bundle kind. `--renderplan-root` only relocates the directory containing RenderPlan variant directories; it does not replace the bundle argument. Each variant directory is a direct child of that RenderPlan root.

`bundle variants` is read-only discovery: it validates existing book or document plans and reports their render state without compiling variants or synthesizing speech. UtterPlan remains responsible for producing RenderPlan variants.

## Inspecting bundle roles and cast readiness

`utterplan bundle compile` writes a bounded `utterplan/role-inventory.toml` sidecar. `voicerender bundle roles` reads that artifact without importing UtterPlan, parsing SSMD, compiling plans, or creating output. It verifies that the inventory still matches the bundle manifest and source bytes before reporting the explicit roles and any use of the profile's default role.

```bash
# Discover source roles before a profile or render variant exists.
voicerender bundle roles story.ssmd-bundle

# Check an authored cast while editing it.
voicerender bundle roles story.ssmd-bundle \
  --profile story-cast.renderprofile.toml --check

# Inspect the exact immutable profile snapshot in an existing variant.
voicerender bundle roles story.ssmd-bundle \
  --variant narration--<profile-fingerprint> --check

# Combine cast status with existing variant/render-state discovery.
voicerender bundle inspect story.ssmd-bundle \
  --profile story-cast.renderprofile.toml --output-root renders --json
```

`--profile` and `--variant` are mutually exclusive. `--profile` inspects the authored, mutable RenderProfile; `--variant` loads the validated snapshot stored in that exact existing variant. `bundle inspect` adds the existing variant and render-state listing, but does not synthesize or create the output root. Neither command writes to the bundle. A missing or stale inventory is not displayed as current—run `utterplan bundle compile` again.

Readiness is structural, not an engine or render check. Explicit inventory roles are required, and the selected profile's `default_role` is also required only when the inventory reports implicit/default-role speech. Every required role must have a valid canonical binding, and required roles must resolve to distinct canonical voice identities; language, delivery options, and unused extra roles do not make voices distinct. No engine SDK/session, model availability, preflight, or completed render is required. With `--check`, exit status is `0` for a ready cast, `1` for an incomplete or duplicate cast, and `2` for invalid input or inventory/profile/variant contracts. Without `--check`, incomplete findings remain ordinary inspection data. JSON mode emits only the result object.

For a book, render selected source chapters from the ordered RenderPlan set. The variant is the exact profile-fingerprinted directory name:

```bash
voicerender bundle render Novel.ssmd-bundle \
  --variant narration--<profile-id-prefix> \
  --chapters 5-17 --cache-dir .voicerender-cache --json
```

Omit `--chapters` to select all chapters. A speech QC window selects one source chapter and a one-based speech range; verified WAVs accumulate into the same chapter manifest. Partial chapter or speech selections can leave the whole variant partial, so a later default batch run may select it again. In batch mode, omitting `--variant` renders every valid variant not yet complete and skips completed variants; invalid variants are reported and make the batch exit nonzero. If no variants are found, or all valid variants are complete, the batch is a successful no-op.

`--chapters`, `--speech`, `--renderplan-root`, `--output-root`, and `--cache-dir` are applied to every selected variant. `--fail-fast` preserves chapter-level fail-fast inside the active book variant and stops scheduling later variants after a batch failure. Batch `--force` includes every valid variant, even completed ones. With an explicit `--variant`, `--force` keeps its existing meaning: bypass verified-output reuse/cache for the selected work. It does not delete the output directory; `--reset-stale` remains the explicit stale-output operation.

A document is one render item and accepts only `--chapters all`:

```bash
voicerender bundle render story.ssmd-bundle \
  --variant printer-cast--<profile-fingerprint> \
  --speech 1-20 --reset-stale
```

For either kind, VoiceRender stores output below `<bundle>/voicerender/<variant>/`. The existing internal book and document layouts remain kind-specific, including their reports and manifest locations. Generic `voicerender preflight` and `voicerender render` remain available for individual RenderPlans.

### Bounded sessions and crash recovery

Rendering and preflight keep engine-session residency bounded. The default `max_open_sessions=1` closes an evicted session before opening another, reducing peak memory even when a plan uses many voices. Because VoiceRender preserves RenderPlan request order, returning to an evicted voice may reopen its session and add startup time. A larger limit can reduce reopen overhead for repeated bindings at the cost of keeping more engine/model memory resident.

Use the positive `--max-open-sessions N` option with direct or bundle rendering; it defaults to `1` and a batch uses the same limit for each selected variant. The Python `render_renderplan()`, `preflight_render_plan()`, and high-level render APIs accept the corresponding `max_open_sessions=N` keyword.

```bash
voicerender render chapter.renderplan.toml --preflight chapter.preflight.toml \
  --out-dir renders/narration --max-open-sessions 2
voicerender bundle render story.ssmd-bundle --max-open-sessions 2
```

If a renderer dies abruptly, its process lock is recovered automatically only when VoiceRender can prove that the same-host owner is dead (including supported legacy PID-only locks). Live, foreign-host, unknown, or malformed locks fail closed. After safe recovery, verified partial-manifest WAVs and reusable cache entries are retained so rendering continues with missing work; `--force` and output-directory deletion are not required.

`--reset-stale` is independent: it resets VoiceRender output belonging to a different RenderPlan identity. It is not a process-lock recovery option and never authorizes stealing a live lock.

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
