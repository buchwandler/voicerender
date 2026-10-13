# VoiceRender

VoiceRender executes shared [RenderPlan](https://pypi.org/project/renderplan/) files through lazy, request-level synthesis engine adapters. It provides RenderProfile lifecycle tools, no-synthesis preflight reports, exact one-item/one-WAV rendering, content-addressed reuse, and verifiable shared manifests. Audio composition, external-media resolution, mastering, and export are outside VoiceRender.

The package lives in `voicerender/`; engine SDKs and model assets are installed separately.

## Install

```bash
python -m pip install -e ".[dev]"
# Install only the engine SDKs you need, e.g.:
python -m pip install -e ".[piper]"
```

Available engine extras are `kokoro`, `piper`, `pocket`, `kitten`, `supertonic`, and `inflect`. The core package can list engines and import without installing their SDKs. Tests use deterministic fake adapters and do not download models.

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

The checked-in `examples/voices/narration.renderprofile.toml` and `examples/plans/chapter.renderplan.toml` form an identity-matched example pair. The profile records a discovery snapshot; refresh and regenerate the plan if adapter metadata changes. See `examples/README.md` for the complete command sequence.

Profiles are engine-refreshed proposals: refresh metadata without overwriting user policy, then review the proposal before using it. For example:

```bash
voicerender profile refresh narration.renderprofile.toml \
  -o narration.refreshed.renderprofile.toml
```

The shared RenderProfile workflow rejects reference-audio bindings because this execution path does not define a portable reference-audio contract.

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

A successful speech window exits 0 even though its chapter manifest is intentionally incomplete; subsequent selections accumulate into that manifest. Chapter failures do not stop later chapters by default; use `--fail-fast` to skip the remainder. `--renderplan-root`, `--output-root`, `--cache-dir`, `--force`, and `--json` support alternate roots, deliberate reset, and automation.

By default, output is kept separate from the RenderPlan inputs:

```text
Novel.ssmdbook/
├── renderplan/<variant>/                 # input plans and renderplan-set.toml
└── voicerender/<variant>/
    ├── voicerender-report.toml            # atomic book-level outcomes
    ├── .voicerender-cache/
    └── chapter-0005/
        ├── preflight.toml                 # whole-chapter preflight
        ├── render-manifest.toml            # cumulative verified artifacts
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

Third-party engines can implement the request-level adapter protocol in `voicerender.engines.base` and register with `voicerender.engines.register_adapter`. Engine implementations and model assets remain lazily loaded.

The older VoicePlan-oriented Python API remains available for compatibility, but the primary CLI workflow uses shared RenderPlan and RenderProfile contracts.

## Boundaries

- **No implicit model downloads:** install SDKs and assets through each engine's supported process.
- **Exact requests:** capacity overflow is reported for replanning; the RenderPlan renderer does not create adaptive child requests.
- **Verified reuse:** cached WAVs are checked by digest and recorded WAV metadata before reuse.
- **No audio post-processing:** VoiceRender writes speech WAV artifacts and a shared manifest only.
- **Reference audio:** unsupported in the shared RenderProfile execution path for this MVP.

## Tests

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
ruff check .
mypy --strict voicerender
```

## License

Apache-2.0 (see `LICENSE`). VoiceRender has no Readio dependency.
