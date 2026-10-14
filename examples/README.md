# VoiceRender examples

## Shared RenderPlan workflow

- `voices/narration.renderprofile.toml` is a RenderProfile snapshot created from the Piper adapter's discovered `en_US-amy-medium` target.
- `plans/chapter.renderplan.toml` references that exact profile identity and contains one required speech item.

`voices/pocket-reference.toml` demonstrates a portable Voicefolio-managed WAV reference with resolver, SHA, revision, repository, and source-path pins. Replace its Pocket target placeholder and create a profile from installed bundle metadata. Local `kind="reference"` sources are also supported when created through `--reference-wav`; they pin an absolute path and digest, so the profile is machine-specific.
The pair can be parsed and validated without inference. To measure and render it, install the Piper extra and make the selected model available locally:

```bash
voicerender profile validate examples/voices/narration.renderprofile.toml
voicerender preflight examples/plans/chapter.renderplan.toml -o chapter.preflight.toml
voicerender render examples/plans/chapter.renderplan.toml \
  --preflight chapter.preflight.toml --out-dir renders/example
voicerender inspect-manifest renders/example/render-manifest.toml --json
```

The profile is a discovery snapshot. If engine-derived metadata is stale, create a reviewed refresh proposal and have the planning tool emit a new plan referencing that proposal's profile ID. Preflight requires no synthesis, but real measurement and rendering may require local engine assets.

## Reference-voice profile examples

Discover the available Sopro bundles and install the extra before creating a profile. A local or Voicefolio-managed source can be selected; managed audio is resolved only when synthesis begins:

```bash
python -m pip install -e '.[sopro]'
voicerender voices list --engine sopro --lang en-US
voicerender profile create --engine sopro --target BUNDLE_ID --language en-US \
  --voicefolio-ref kyutai-tts-voices:alba-mackenna/a-moment-by \
  --output sopro.renderprofile.toml
```

Kokoro supports Inno on static targets that advertise `inno-v0.2`, and AkinVox on a reference-only target. Inno does not need a transcript; AkinVox requires the exact reference transcript:

```bash
voicerender profile create --engine kokoro --target STATIC_TARGET_WITH_INNO \
  --language en-US --reference-wav ./speaker.wav \
  --option 'voice_enrollment={"engine":"auto","fmax":8000}' \
  --output local-inno.renderprofile.toml

voicerender profile create --engine kokoro --target en-akinvox-cloning-v1 \
  --language en-US --voicefolio-ref kyutai-tts-voices:alba-mackenna/a-moment-by \
  --option 'voice_enrollment={"engine":"akinvox","reference_text":"Exact spoken words."}' \
  --output managed-akinvox.renderprofile.toml
```

Preflight checks source pins and measures requests without fetching managed WAVs or enrolling voices. The render session resolves and enrolls a reference lazily, then reuses the enrolled voice for that session. Local references are path-specific; use Voicefolio pins when profiles must travel between machines.

## Compatibility API

`plans/chapter.voiceplan.toml` and `voices/piper.toml` remain examples for the older VoicePlan-oriented Python API. The primary CLI consumes shared RenderPlan files.
