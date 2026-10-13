# VoiceRender documentation

VoiceRender consumes shared RenderPlan and RenderProfile contracts. See `README.md` in the repository root for installation, CLI examples, and the Python API.

```{toctree}
:maxdepth: 1

changelog
```

## Workflow

1. Create or update a RenderProfile and review engine-derived binding metadata.
2. Produce a RenderPlan that references the profile identity.
3. Run `voicerender preflight PLAN -o REPORT`; this measures requests and never synthesizes.
4. Render only when the report is `ok`: `voicerender render PLAN --preflight REPORT --out-dir OUTPUT`.
5. Verify the bundle with `voicerender inspect-manifest OUTPUT/render-manifest.toml`.

The RenderPlan renderer preserves request order and identity: one speech item becomes one exact request and one numbered WAV. It does not adaptively split, merge, normalize, or reorder requests. An incomplete run leaves a valid partial manifest.

The repository includes a Piper RenderProfile snapshot at `examples/voices/narration.renderprofile.toml` and a matching RenderPlan at `examples/plans/chapter.renderplan.toml`. See `examples/README.md`; refresh engine-derived metadata and regenerate the plan if the snapshot becomes stale.

## Engine integration

Engine adapters implement the request-level protocol in `voicerender.engines.base`. Engine SDKs are imported lazily and installed through optional extras. The deterministic test suite uses fake adapters and does not download models.

The shared RenderProfile MVP rejects reference-audio bindings. Composition, external-media resolution, mastering, and export belong to downstream tools.
