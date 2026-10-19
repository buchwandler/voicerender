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

A planning tool produces a shared RenderPlan whose `render_profile_id` and role bindings match the selected RenderProfile. Create a profile through adapter discovery when needed; see the [RenderProfile lifecycle](docs/renderprofile.md):

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

VoiceRender does not apply emphasis during speech synthesis and does not advertise `emphasis` as a native Piper feature. By default, emphasis is rejected. An orchestration pipeline that declares the `audiocompose.emphasis.gain.v1` handler must select `--emphasis-policy defer-to-compose` for **both** preflight and render (or for `render-book`):

```bash
voicerender preflight chapter.renderplan.toml \
  -o chapter.preflight.toml --emphasis-policy defer-to-compose
voicerender render chapter.renderplan.toml \
  --preflight chapter.preflight.toml --out-dir renders/narration \
  --emphasis-policy defer-to-compose
```

The accepted emphasis parameters are `{}` or one of `{ "level": "reduced" }`, `{ "level": "moderate" }`, or `{ "level": "strong" }`; unknown keys, types, and invalid ranges are rejected. Other unsupported controls remain fail-closed. VoiceRender sends the original text to the engine without an emphasis option, preserves the original controls and identities in the RenderPlan, and leaves control interpretation—including overlapping emphasis—to AudioCompose.

Deferred controls are explicitly pending composition, not fulfilled by the standalone WAV. Their count and capability appear in the preflight check detail and in the render output's `render-diagnostics.json` sidecar, which warns that any produced WAVs contain unchanged speech. Human render output warns when composition is pending, and render JSON plus `render-book` reports expose `pending_composition`. The shared RenderManifest schema is unchanged; AudioCompose must use the original RenderPlan as the source of truth when applying the declared effect.
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

RenderProfiles can hold multiple independent role bindings. Role replacement revalidates the selected engine/target, refreshes derived requirements and capacity, preserves the other roles and user-authored planning/pause policy, and computes a new `profile_id`.

```bash
voicerender profile inspect narration.renderprofile.toml --json
voicerender profile role list narration.renderprofile.toml --json
voicerender profile role set narration.renderprofile.toml guest \
  --engine piper --target en_US-amy-medium --language en-US \
  -o narration.with-guest.renderprofile.toml --json
voicerender profile default-role narration.with-guest.renderprofile.toml guest \
  --in-place --json
voicerender profile role remove narration.with-guest.renderprofile.toml narrator \
  --new-default-role guest --in-place --json
```

Role mutations require exactly one explicit `--output PATH` or `--in-place`; the input profile is otherwise left untouched. Removing the final role is rejected, and removing the current default requires `--new-default-role`. The package root exports the model-management, option/voice discovery, source/clone, and multi-role profile APIs for automation. See [RenderProfile lifecycle](docs/renderprofile.md) for API examples and policy details.

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

## SSMDBook RenderPlan collections from UtterPlan

For a book, [UtterPlan](https://pypi.org/project/utterplan/) can project already-saved
chapter plans into one RenderPlan variant per RenderProfile. It does not call
VoiceRender or preflight during projection. The collection index
`renderplan-set.toml` and invocation report `render-report.toml` are UtterPlan-owned
workspace metadata, not RenderPlan artifacts. VoiceRender can consume the ordered index through `render-book`; the direct single-plan workflow remains available.

```bash
utterplan render-book "Novel.ssmdbook" --chapters 5-17 \
  --profile narration.renderprofile.toml

voicerender preflight \
  "Novel.ssmdbook/renderplan/narration--<profile-id-prefix>/chapter-0005.renderplan.toml" \
  -o chapter-0005.preflight.toml
voicerender render \
  "Novel.ssmdbook/renderplan/narration--<profile-id-prefix>/chapter-0005.renderplan.toml" \
  --preflight chapter-0005.preflight.toml \
  --out-dir renders/chapter-0005
```

Repeat the ordinary preflight/render flow for each selected plan. Different profile
identities produce distinct UtterPlan variant directories, while each individual
RenderPlan retains the same shared contract and exact request semantics.

### Render the variant with VoiceRender

Render the chapters from the ordered set directly. The variant value is the exact directory name created by UtterPlan, including its profile fingerprint:

```bash
voicerender render-book "Novel.ssmdbook" \
  --variant "narration--<profile-id-prefix>" \
  --chapters 5-17 \
  --cache-dir .voicerender-cache \
  --json
```

Omit `--chapters` to select every chapter. For speech QC, select exactly one source chapter and a one-based range of RenderPlan speech numbers; whole-chapter preflight still runs, but only the selected speech is synthesized:

```bash
voicerender render-book "Novel.ssmdbook" \
  --variant "narration--<profile-id-prefix>" \
  --chapters 5 --speech 50-60

# A later window adds verified WAVs to the same chapter manifest.
voicerender render-book "Novel.ssmdbook" \
  --variant "narration--<profile-id-prefix>" \
  --chapters 5 --speech 61-70

# Omit --speech to finish the chapter, reusing valid artifacts from both windows.
voicerender render-book "Novel.ssmdbook" \
  --variant "narration--<profile-id-prefix>" --chapters 5
```

A successful speech window exits 0 even though its chapter manifest is intentionally incomplete; subsequent selections accumulate into that manifest. Chapter failures do not stop later chapters by default; use `--fail-fast` to skip the remainder. `--renderplan-root`, `--output-root`, `--cache-dir`, and `--json` support alternate roots and automation. `--reset-stale` lets a rerun accept output whose manifest belongs to a different RenderPlan: it removes only VoiceRender-owned chapter artifacts and keeps the external cache available for verified exact-request reuse. `--force` is stronger: it synthesizes every selected request and bypasses cache.

When a chapter fails with `preflight.replan`, VoiceRender stops before synthesizing requests under that plan and exposes the persisted report path. The chapter outcome includes `preflight_report` and `replan_required`; runtime immediate rechecks point to `preflight-recheck.toml` (also available as the compatibility field `recheck_report`). The JSON output and `voicerender-report.toml` include these fields, and human output prints the report path. Use the report with your planning tool to activate a child RenderPlan, then rerun VoiceRender with `--reset-stale`:

```bash
voicerender render-book "Novel.ssmdbook" \
  --variant "narration--<profile-id-prefix>" \
  --chapters 5 --reset-stale
```

The reset does not delete `.voicerender-cache`; unchanged exact renderer requests can be restored from verified cache entries, while new child requests are synthesized normally. `render`/`render-book` and the `render_renderplan()`/`render_book()` APIs accept `reset_stale_output=True`. VoiceRender reports the replan handoff but does not invoke the planner or generate child plans.

By default, output is kept separate from the RenderPlan inputs:

```text
Novel.ssmdbook/
├── renderplan/<variant>/                 # input plans and renderplan-set.toml
└── voicerender/<variant>/
    ├── voicerender-report.toml            # atomic book-level outcomes
    ├── .voicerender-cache/
    └── chapter-0005/
        ├── preflight.toml                 # whole-chapter preflight
        ├── preflight-recheck.toml           # runtime immediate capacity recheck, if any
        ├── render-manifest.toml            # cumulative verified artifacts
        ├── render-diagnostics.json          # deferred controls, if any, remain pending composition
        ├── 000001.wav
        └── ...
```

The chapter directories use the set's chapter IDs, while `--chapters` selects by source chapter number. The exact variant name is retained under both roots. Existing manual `preflight`/`render` commands above remain available for individual plans.

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

`render`, `render-book`, and `preflight` accept `--progress` and `--no-progress`. By default progress is enabled only for interactive stderr and non-JSON output. `--progress` forces it (including with `--json`), but progress always stays on stderr; JSON stdout remains a single unchanged result. Forced redirected output uses throttled newline records without ANSI escapes or carriage returns. Use `--no-progress` to disable it.

Python callers can subscribe to typed events during rendering and preflight. Events carry progress counts, phase/chapter context, elapsed time, and an ETA when synthesis timing evidence is available:

```python
from voicerender import RenderEvent, RenderEventKind, render_book


def on_event(event: RenderEvent) -> None:
    if event.kind is RenderEventKind.SEGMENT_RENDERED:
        print(event.progress, event.eta_seconds, event.eta_confidence)


result = render_book(
    "Novel.ssmdbook", variant="narration--<profile-id>", on_event=on_event
)
```

The keyword-only `on_event` callback is also available on `preflight_render_plan()`, `render_renderplan()`, and the legacy `render()`/`render_plan()` APIs. Preflight events report required-occurrence work, but its capacity measurements are not synthesis timing and do not produce an ETA.

`estimate_renderplan()` and `estimate_render_book()` provide side-effect-free workload and verified existing-output reuse counts. Without a calibrated timing model, they return `estimated_seconds=None`, `confidence="unavailable"`, and `source="workload-only"`. Live ETA becomes available only after successful synthesis duration samples exist; it is approximate and can remain unknown for work whose engine has no timing evidence.

```python
from voicerender import estimate_renderplan

estimate = estimate_renderplan(plan, output_dir="renders/narration")
print(
    estimate.selected_count, estimate.known_output_reuse_count, estimate.pending_count
)
```

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

## SSMDDoc document workspaces

For one editable SSMDDoc, compile and project explicitly with UtterPlan, then render the exact fingerprinted variant with VoiceRender:

```bash
utterplan compile-document story.ssmdoc
utterplan render-document story.ssmdoc --profile narration.renderprofile.toml
voicerender render-document story.ssmdoc --variant narration--<profile-fingerprint>
```

The document output is stored under `.readio/voicerender/<variant>/document-0001/`. VoiceRender consumes only the selected RenderPlan and does not parse UtterPlan files, invoke either planner/converter, or detect source freshness. After editing `document.ssmd.md`, re-run `compile-document` and UtterPlan `render-document`; use VoiceRender `--reset-stale` to replace stale owned audio while preserving the sibling cache.

## Tests

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
ruff check .
mypy --strict voicerender
```

## License

Apache-2.0 (see `LICENSE`). VoiceRender has no Readio dependency.
