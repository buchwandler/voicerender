# VoiceRender documentation

VoiceRender consumes shared RenderPlan and RenderProfile contracts. See `README.md` in the repository root for installation, CLI examples, and the Python API.

```{toctree}
:maxdepth: 1

renderprofile
changelog
```

## Engine and voice control

VoiceRender owns its public engine-option and voice discovery, optional model-asset management, pinned voice-source lifecycle, reference-voice clone workflow, and multi-role RenderProfile mutations. These APIs are available from the `voicerender` package root and corresponding CLI commands; unsupported asset operations are reported rather than delegated to private backend code or backend CLI subprocesses. Discovery and clone/profile preparation do not synthesize audio. See the repository README for command examples and [RenderProfile lifecycle](renderprofile.md#inspecting-and-mutating-profile-roles) for role APIs and write safety.

## Workflow

1. Create or update a RenderProfile and review engine-derived binding metadata (see the [RenderProfile lifecycle](renderprofile.md)).
2. Produce RenderPlans that reference the profile identity: use UtterPlan
   `render-plan` for one saved plan or `render-book` for selected SSMDBook
   chapters; both emit the same shared RenderPlan contract.
3. Run `voicerender preflight PLAN -o REPORT`; this measures requests and never synthesizes.
4. Render only when the report is `ok`: `voicerender render PLAN --preflight REPORT --out-dir OUTPUT`.
5. Verify the bundle with `voicerender inspect-manifest OUTPUT/render-manifest.toml`.

The RenderPlan renderer preserves request order and identity: one speech item becomes one exact request and one numbered WAV. It does not adaptively split, merge, normalize, or reorder requests. An incomplete run leaves a valid partial manifest.

An `ok` preflight approves the exact plan under capability observed at preflight time; VoiceRender remeasures before each synthesis. Capacity drift stops that occurrence without synthesis and returns a fresh plan-matched `replan` report; session-open failure returns `reject`. Pass `--recheck-report PATH` to `voicerender render` to persist either report. VoiceRender never edits speech text or emits a child RenderPlan. Operational render results include a separate execution fingerprint with adapter/API, synthesis, voice/options, and observed-capacity facts; these facts do not affect `RenderPlan.plan_id`.

The repository includes a Piper RenderProfile snapshot at `examples/voices/narration.renderprofile.toml` and a matching RenderPlan at `examples/plans/chapter.renderplan.toml`. See `examples/README.md`; refresh engine-derived metadata and regenerate the plan if the snapshot becomes stale.

For book-level projection, UtterPlan owns the `renderplan-set.toml` collection index
and its per-invocation report as workspace metadata. VoiceRender's `render-book`
command consumes the ordered set and renders its individual `.renderplan.toml` files;
the direct single-plan preflight/render commands remain available.
See [UtterPlan](https://pypi.org/project/utterplan/) and the shared
[RenderPlan](https://pypi.org/project/renderplan/) package.

## Book-level rendering

After UtterPlan has written `renderplan-set.toml`, render the exact variant named in the set directory. By default, VoiceRender writes under `<book>/voicerender/<variant>/`, leaving the input plans under `<book>/renderplan/<variant>/`:

```bash
voicerender render-book "Novel.ssmdbook" \
  --variant "narration--<profile-id-prefix>" \
  --chapters 5-17 --json
```

Omit `--chapters` to select all indexed chapters. For a speech QC window, select one source chapter and one-based speech numbers. Whole-chapter preflight is still performed; successful selected WAVs remain in the chapter manifest and can be accumulated by later invocations:

```bash
voicerender render-book "Novel.ssmdbook" \
  --variant "narration--<profile-id-prefix>" \
  --chapters 5 --speech 50-60

voicerender render-book "Novel.ssmdbook" \
  --variant "narration--<profile-id-prefix>" \
  --chapters 5 --speech 61-70

# Resume the chapter to full completion using its verified partial artifacts.
voicerender render-book "Novel.ssmdbook" \
  --variant "narration--<profile-id-prefix>" --chapters 5
```

A speech window can succeed (exit 0) while the whole-chapter manifest remains partial. The atomic `voicerender-report.toml` records each selected chapter, including execution fingerprints and the path to any fresh recheck report; each chapter directory contains its `preflight.toml`, optional `preflight-recheck.toml`, cumulative `render-manifest.toml`, and numbered WAVs. A stale persisted `ok` is invalidated only by a fresh report for the same plan and failed speech occurrence. The variant directory name is preserved exactly. `--renderplan-root`, `--output-root`, and `--cache-dir` override their respective roots; failures continue by default, or use `--fail-fast` to skip remaining chapters.

## Engine integration

Engine adapters implement the request-level protocol in `voicerender.engines.base`. Engine SDKs are imported lazily and installed through optional extras. The deterministic test suite uses fake adapters and does not download models.

Pinned local and Voicefolio-managed reference sources are supported for PocketSynth, SoproSynth, and Kokoro enrollment. Local path bindings are machine-specific; managed references are portable. Preflight validates source metadata and measures requests without fetching managed audio or enrolling voices. Composition, external-media resolution, mastering, and export belong to downstream tools.
