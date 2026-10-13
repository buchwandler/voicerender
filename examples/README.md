# VoiceRender examples

## Shared RenderPlan workflow

- `voices/narration.renderprofile.toml` is a RenderProfile snapshot created from the Piper adapter's discovered `en_US-amy-medium` target.
- `plans/chapter.renderplan.toml` references that exact profile identity and contains one required speech item.

The pair can be parsed and validated without inference. To measure and render it, install the Piper extra and make the selected model available locally:

```bash
voicerender profile validate examples/voices/narration.renderprofile.toml
voicerender preflight examples/plans/chapter.renderplan.toml -o chapter.preflight.toml
voicerender render examples/plans/chapter.renderplan.toml \
  --preflight chapter.preflight.toml --out-dir renders/example
voicerender inspect-manifest renders/example/render-manifest.toml --json
```

The profile is a discovery snapshot. If engine-derived metadata is stale, create a reviewed refresh proposal and have the planning tool emit a new plan referencing that proposal's profile ID. Preflight requires no synthesis, but real measurement and rendering may require local engine assets.

## Compatibility API

`plans/chapter.voiceplan.toml` and `voices/piper.toml` remain examples for the older VoicePlan-oriented Python API. The primary CLI consumes shared RenderPlan files.
