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

A planning tool produces a shared RenderPlan whose `render_profile_id` and role bindings match the selected RenderProfile. Create a profile through adapter discovery when needed:

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

Install the selected engine extra and make its model assets available before preflight/render; VoiceRender does not download models implicitly. Preflight reports `ok`, `replan`, or `reject` for each required speech occurrence. It measures requests but never synthesizes. Rendering requires a matching all-`ok` report and remeasures each exact request immediately before synthesis. Every RenderPlan speech item maps to exactly one public `SpeechRequest` and one numbered WAV, in plan order; VoiceRender does not split, merge, normalize, or reorder requests. Incremental manifests remain valid when rendering fails partway through.

The checked-in `examples/voices/narration.renderprofile.toml` and `examples/plans/chapter.renderplan.toml` form an identity-matched example pair. The profile records a discovery snapshot; refresh and regenerate the plan if adapter metadata changes. See `examples/README.md` for the complete command sequence.

Profiles are engine-refreshed proposals: refresh metadata without overwriting user policy, then review the proposal before using it. For example:

```bash
voicerender profile refresh narration.renderprofile.toml \
  -o narration.refreshed.renderprofile.toml
```

The shared RenderProfile workflow rejects reference-audio bindings because this execution path does not define a portable reference-audio contract.

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
