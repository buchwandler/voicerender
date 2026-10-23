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

Discover the available Sopro bundles and install the extra before creating a profile. VoiceFolio-managed audio is resolved only when synthesis begins:

```bash
python -m pip install -e '.[sopro]'
voicerender voices list --engine sopro --lang en-US
voicerender profile create --engine sopro --target BUNDLE_ID --language en-US \
  --voicefolio-ref kyutai-tts-voices:alba-mackenna/a-moment-by \
  --output sopro.renderprofile.toml
```

### Kokoro multi-role VoiceFolio AkinVox

When a VoiceFolio catalog entry contains its transcript, create and bind cast roles without a local reference WAV or engine-specific enrollment JSON:

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

For a reference without catalog transcript metadata, pass the exact spoken words with the generic `--transcript` fallback:

```bash
voicerender profile create \
  --role narrator --engine kokoro \
  --target en-akinvox-cloning-v1 --language en-US \
  --voicefolio-ref other:voice --transcript "Exact spoken words." \
  -o narrator.renderprofile.toml
```

AkinVox requires a non-empty exact transcript, supports English only, and requires speed `1.0`. Conflicting transcript sources are rejected. Profile create, bind, and validate read catalog metadata but do not fetch audio; VoiceFolio resolves/verifies the pinned source and AkinVox enrolls lazily during rendering. The profile keeps portable source pins, not cache or materialized WAV paths. Existing advanced `voice_enrollment` options remain available when required.

Inno remains a separate option for static targets that advertise its enroller. It needs no transcript and may accept `fmax`:

```bash
voicerender profile create --engine kokoro --target STATIC_TARGET_WITH_INNO \
  --language en-US --reference-wav ./speaker.wav \
  --option 'voice_enrollment={"engine":"auto","fmax":8000}' \
  --output local-inno.renderprofile.toml
```

## Compatibility API

`plans/chapter.voiceplan.toml` and `voices/piper.toml` remain examples for the older VoicePlan-oriented Python API. The primary CLI consumes shared RenderPlan files.
