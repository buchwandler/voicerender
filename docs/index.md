# VoiceRender documentation

VoiceRender consumes shared RenderPlan and RenderProfile contracts. See `README.md` in the repository root for installation, CLI examples, and the Python API.

```{toctree}
:maxdepth: 1

renderprofile
changelog
```

## Workflow

1. Create or update a RenderProfile and review engine-derived binding metadata (see the [RenderProfile lifecycle](renderprofile.md)).
2. Produce RenderPlans that reference the profile identity: use UtterPlan
   `render-plan` for one saved plan or `render-book` for selected SSMDBook
   chapters; both emit the same shared RenderPlan contract.
3. Run `voicerender preflight PLAN -o REPORT`; this measures requests and never synthesizes.
4. Render only when the report is `ok`: `voicerender render PLAN --preflight REPORT --out-dir OUTPUT`.
5. Verify the bundle with `voicerender inspect-manifest OUTPUT/render-manifest.toml`.

The RenderPlan renderer preserves request order and identity: one speech item becomes one exact request and one numbered WAV. It does not adaptively split, merge, normalize, or reorder requests. An incomplete run leaves a valid partial manifest.

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

A speech window can succeed (exit 0) while the whole-chapter manifest remains partial. The atomic `voicerender-report.toml` records each selected chapter; each chapter directory contains its `preflight.toml`, cumulative `render-manifest.toml`, and numbered WAVs. The variant directory name is preserved exactly. `--renderplan-root`, `--output-root`, and `--cache-dir` override their respective roots; failures continue by default, or use `--fail-fast` to skip remaining chapters.

## Engine integration

Engine adapters implement the request-level protocol in `voicerender.engines.base`. Engine SDKs are imported lazily and installed through optional extras. The deterministic test suite uses fake adapters and does not download models.

The shared RenderProfile MVP rejects reference-audio bindings. Composition, external-media resolution, mastering, and export belong to downstream tools.
