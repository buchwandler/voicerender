# VoiceRender guide

VoiceRender consumes shared RenderPlan and RenderProfile contracts. See `README.md` in the repository root for installation, CLI examples, and the Python API.

```{toctree}
:maxdepth: 1

renderprofile
changelog
```

## Engine and voice control

VoiceRender owns engine-option and voice discovery, optional model-asset management, pinned voice-source lifecycle, reference-voice cloning, and multi-role RenderProfile mutations. These APIs are available from the `voicerender` package root and corresponding CLI commands; unsupported asset operations are reported rather than delegated to private backend code or backend CLI subprocesses. Discovery and profile preparation do not synthesize audio. See [RenderProfile lifecycle](renderprofile.md) for role APIs and write safety.

## ONNX Runtime provider policy

See the [ONNX Runtime provider policy guide in the README](https://github.com/buchwandler/voicerender/blob/main/README.md#onnx-runtime-provider-policy) for `runtime status`, `list`, `use`, and `reset`, including provider precedence and CPU/CUDA/auto examples. The policy is an ONNXVoice-managed default for compatible future sessions; it does not install or select runtime distributions, imply that every engine is ONNX-backed, or override explicit engine provider options. Environment overrides take precedence over the saved preference, and reset clears only that saved preference.

## Workflow

1. Create a complete-cast RenderProfile with per-role engine bindings.
2. Produce RenderPlans that reference the profile identity.
3. Run `voicerender preflight PLAN -o REPORT`; this measures requests and never synthesizes.
4. Render only when the report is `ok`: `voicerender render PLAN --preflight REPORT --out-dir OUTPUT`.
5. Verify the result with `voicerender inspect-manifest OUTPUT/render-manifest.toml`.

For the coordinated SSMD workflow, the packages share one command-group name and one canonical editable workspace suffix:

```bash
ssmdconvert bundle convert story.ssmd.md
utterplan bundle compile story.ssmd-bundle
utterplan bundle render story.ssmd-bundle --profile narration.renderprofile.toml
voicerender bundle render story.ssmd-bundle --variant narration--<profile-fingerprint>
```

`manifest.json` decides whether the root holds a book or a single document. VoiceRender detects the kind automatically; the same command takes a `.ssmd-bundle/` root for either. It does not accept an inner source file or a chapter directory. Documents are one render item and require `--chapters all`; books support chapter and speech selectors. The canonical packed transport artifact is `<name>.ssmd-bundle.zip`; VoiceRender's render pipeline operates on the editable directory.

For books, UtterPlan owns the `renderplan-set.toml` collection index and its per-invocation report. VoiceRender consumes the ordered set, while direct single-plan preflight and render remain available. A speech window can succeed while the whole-book manifest remains partial; later selections accumulate verified WAVs. `--fail-fast` skips remaining chapters after a failure. `--renderplan-root`, `--output-root`, and `--cache-dir` override their roots. `--reset-stale` replaces stale VoiceRender-owned output while keeping cache entries available for verified reuse; `--force` bypasses cache.

VoiceRender keeps its output below `<bundle>/voicerender/<variant>/`. Existing internal layouts may differ by kind, but they are selected from the manifest, not from a command name. The generic RenderPlan renderer preserves request order and identity: one speech item becomes one exact request and one numbered WAV. It does not adaptively split, merge, normalize, or reorder requests. An incomplete run leaves a valid partial manifest.

An `ok` preflight approves the exact plan under capability observed at preflight time; VoiceRender remeasures before each synthesis. Capacity drift stops that occurrence without synthesis and returns a fresh plan-matched `replan` report; session-open failure returns `reject`. Pass `--recheck-report PATH` to `voicerender render` to persist either report. VoiceRender never edits speech text or emits a child RenderPlan. Operational render results include a separate execution fingerprint with adapter/API, synthesis, voice/options, and observed-capacity facts; these facts do not affect `RenderPlan.plan_id`.

The repository includes a Piper RenderProfile snapshot at `examples/voices/narration.renderprofile.toml` and a matching RenderPlan at `examples/plans/chapter.renderplan.toml`. See `examples/README.md`; refresh engine-derived metadata and regenerate the plan if the snapshot becomes stale.

## Engine integration

Engine adapters implement the request-level protocol in `voicerender.engines.base`. Engine SDKs are imported lazily and installed through optional extras. The deterministic test suite uses fake adapters and does not download models.

Pinned local and Voicefolio-managed reference sources are supported for PocketSynth, SoproSynth, and Kokoro enrollment. Local path bindings are machine-specific; managed references are portable. Preflight validates source metadata and measures requests without fetching managed audio or enrolling voices. Composition, external-media resolution, mastering, and export belong to downstream tools.

## Progress, ETA, and estimates

The `render`, `bundle render`, and `preflight` commands support `--progress` and `--no-progress`. Auto mode writes progress only to interactive stderr and is silent in JSON mode. Forced progress remains on stderr, including with `--json`; redirected output is throttled newline text without terminal control sequences.

Python render and preflight APIs accept typed progress callbacks. Bundle rendering reports book chapter context when applicable; ETA uses successful synthesis timing evidence only and may be unavailable at startup or for engines without samples. `estimate_renderplan()` and `estimate_render_book()` expose selected work and verified output reuse without synthesizing or creating output directories. Estimated seconds remain unknown unless a calibrated timing model is available.

Capacity measurements describe request fit, not synthesis latency. They are never used as a wall-clock prediction. See [Live progress and ETA in the README](https://github.com/buchwandler/voicerender/blob/main/README.md#live-progress-eta-and-workload-estimates) for usage examples.
