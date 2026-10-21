# RenderProfile lifecycle

A RenderProfile is the portable, user-owned renderer configuration shared by
[RenderPlan](https://pypi.org/project/renderplan/) producers and VoiceRender. It
represents the complete cast for one render variant: each logical source role has
its own engine/target/voice binding, while planning and pause policy are shared.
Roles in a single profile may use different engines. Create the first/default role
with VoiceRender so its binding is seeded from adapter metadata, then review and
validate it before planning:

```bash
voicerender engines list
voicerender voices list --engine piper --lang en-US --offline --json
voicerender profile create \
  --role narrator --engine piper --target en_US-amy-medium --language en-US \
  --offline --output narration.renderprofile.toml
voicerender profile validate narration.renderprofile.toml
```

A profile has a canonical `profile_id`; RenderPlans carry that identity and copy
the resolved role bindings needed to execute without reopening the profile.
Refreshing engine metadata writes a proposal rather than overwriting user policy:

```bash
voicerender profile refresh narration.renderprofile.toml \
  -o narration.refreshed.renderprofile.toml
voicerender profile validate narration.refreshed.renderprofile.toml
```

Review refreshed bindings and policy before using the proposal. If identity-bearing
profile semantics change, regenerate the associated RenderPlan using the refreshed
profile. Keep the original profile with its plans for provenance and inspection.

## Inspecting and mutating profile roles

One RenderProfile is the complete cast for a render variant. Its logical roles may use different engines and targets while sharing planning and pause policy. `profile show` is the preferred human view; JSON retains each full canonical binding:

```bash
voicerender profile create \
  --role narrator --engine kokoro --target v1.0 --voice af_heart \
  --language en-US --offline -o printer-cast.renderprofile.toml
voicerender profile bind printer-cast.renderprofile.toml host \
  --engine kitten --target mini-0.8 --voice Jasper \
  --language en-US --offline --in-place
voicerender profile bind printer-cast.renderprofile.toml guest \
  --engine piper --target en_US-amy-medium \
  --language en-US --offline --in-place
voicerender profile show printer-cast.renderprofile.toml
voicerender profile show printer-cast.renderprofile.toml --json
voicerender profile role list printer-cast.renderprofile.toml --json
```

`profile create` makes the initial/default role. `profile bind` adds or replaces a binding and supports voice, Voicefolio, reference WAV, prepared voice, voice-source, speaker, and engine-option selectors. The human view shows each role's engine, target, language, and actual voice/speaker/source identity without dumping renderer metadata.

Use VoiceRender's public role helpers when building or editing profiles programmatically:

```python
from voicerender import (
    list_profile_roles,
    load_render_profile,
    remove_profile_role,
    set_default_profile_role,
    set_profile_role,
)

profile = load_render_profile("narration.renderprofile.toml")
roles = list_profile_roles(profile)
updated = set_profile_role(
    profile,
    "guest",
    engine="piper",
    target_id="en_US-amy-medium",
    language="en-US",
    options={"length_scale": 1.0},
)
updated = set_default_profile_role(updated, "guest")
# If removing the current default, choose a remaining role explicitly.
updated = remove_profile_role(updated, "narrator", new_default_role="guest")
```

`set_profile_role` resolves the engine binding and regenerates its derived requirements and capacity. Existing role bindings and the profile's planning/pause policy are preserved; each successful mutation recomputes `profile_id`. Invalid selections/options, a missing role, removal of the last role, or removal of the default without a replacement are rejected. The helpers return a new profile and never write the input file.

For CLI mutations, `profile bind` is the first-class way to add or replace a cast member; it and the structured `profile role set` form require exactly one of `--output PATH` and `--in-place`. Use `--output` to review a proposal before replacing the original:

```bash
voicerender profile bind printer-cast.renderprofile.toml host \
  --engine kitten --target mini-0.8 --language en-US --voice Jasper \
  --in-place
voicerender profile role set printer-cast.renderprofile.toml guest \
  --engine piper --target en_US-amy-medium --language en-US \
  -o printer-cast.proposed.renderprofile.toml
voicerender profile default-role printer-cast.proposed.renderprofile.toml guest \
  --in-place
voicerender profile role remove printer-cast.proposed.renderprofile.toml narrator \
  --new-default-role guest --in-place
```

The RenderPlan schema is unchanged: plans continue to carry the same profile identity and resolved per-role bindings.

## UtterPlan bundle projection

[UtterPlan](https://pypi.org/project/utterplan/) can project saved book chapter plans into one RenderPlan variant per RenderProfile. It does not call VoiceRender or preflight during projection. The collection index `renderplan-set.toml` and invocation report `render-report.toml` are UtterPlan-owned workspace metadata, not RenderPlan artifacts.

```bash
utterplan bundle compile Novel.ssmd-bundle --chapters 5-17
utterplan bundle render Novel.ssmd-bundle --profile narration.renderprofile.toml
```

The profile fingerprint selects a distinct variant directory containing a canonical `renderprofile.toml` snapshot, one ordinary `.renderplan.toml` per successful chapter, and the ordered set index. Different profile identities can coexist while sharing semantic plans. UtterPlan emits RenderPlans; it does not inspect engine availability, preflight, synthesize, or require VoiceRender to be installed.

## Render an SSMD bundle

VoiceRender detects whether the root contains a book or one document from `manifest.json`. Both kinds use the same command and require the canonical editable `.ssmd-bundle/` root:

```bash
voicerender bundle render Novel.ssmd-bundle \
  --variant narration--<profile-id-prefix> \
  --chapters 5-17 --json
```

Omit `--chapters` to select all chapters. For a speech QC window, select one source chapter and one-based speech numbers. Successful WAVs accumulate in the same chapter manifest:

```bash
voicerender bundle render Novel.ssmd-bundle \
  --variant narration--<profile-id-prefix> --chapters 5 --speech 50-60
voicerender bundle render Novel.ssmd-bundle \
  --variant narration--<profile-id-prefix> --chapters 5 --speech 61-70
voicerender bundle render Novel.ssmd-bundle \
  --variant narration--<profile-id-prefix> --chapters 5
```

A document is one render item and accepts only `--chapters all`:

```bash
voicerender bundle render story.ssmd-bundle \
  --variant printer-cast--<profile-fingerprint>
```

VoiceRender writes below `<bundle>/voicerender/<variant>/`, preserving the existing kind-specific output layout. `--renderplan-root`, `--output-root`, and `--cache-dir` override their respective roots. Failures continue by default for books; `--fail-fast` skips remaining chapters. `--reset-stale` replaces stale owned output while preserving verified cache reuse. `--force` synthesizes every selected request and bypasses cache.

VoiceRender consumes each RenderPlan independently as well. The bundle collection directory does not change the single-plan execution contract; the set index is not passed to generic preflight or render:

```bash
voicerender preflight \
  "Novel.ssmd-bundle/renderplan/narration--<profile-id-prefix>/chapter-0005.renderplan.toml" \
  -o chapter-0005.preflight.toml
voicerender render \
  "Novel.ssmd-bundle/renderplan/narration--<profile-id-prefix>/chapter-0005.renderplan.toml" \
  --preflight chapter-0005.preflight.toml --out-dir renders/chapter-0005
```

Parsing a RenderProfile is engine-independent. `voicerender profile validate` checks the selected adapter/API, so it requires the selected engine SDK, but performs no model inference. Install the selected engine extra and make its model assets available before preflight/render. See [UtterPlan](https://pypi.org/project/utterplan/) and the [RenderPlan package contract](https://pypi.org/project/renderplan/) for the producer/consumer boundary.

## Reference-voice sources

A voice source is stored under the role binding's `options.voice_source`. Local `kind = "reference"` bindings include an absolute, normalized WAV path and its computed SHA-256. They can be used in a local RenderProfile but are machine-specific; do not share a local profile unless that path and file are intentionally available on the other machine. Validation checks the digest and rejects changed or missing files rather than updating the pin. Use `kind = "managed_reference"` with `resolver = "voicefolio"` for portable references; persist the Voicefolio reference, content digest, source revision, repository, and source path pins.

The CLI creates local sources with `--reference-wav PATH` and managed sources with `--voicefolio-ref REF`. For example:

```bash
voicerender profile create \
  --engine kokoro --target en-akinvox-cloning-v1 --language en-US \
  --voicefolio-ref kyutai-tts-voices:alba-mackenna/a-moment-by \
  --option 'voice_enrollment={"engine":"akinvox","reference_text":"Exact words spoken in the clip."}' \
  --output narrator.renderprofile.toml
```

Kokoro selects enrollment based on the target's discovered capabilities. `engine = "auto"` chooses Inno for a static target advertising the `inno-v0.2` enroller and AkinVox for a reference-only target. Inno accepts optional `fmax` and no transcript. AkinVox requires a non-empty exact `reference_text` and currently supports English only. A profile may also set `voice_enrollment` explicitly to `inno` or `akinvox`; unsupported target/engine combinations fail validation.

Preflight verifies local hashes and Voicefolio catalog metadata, then measures the request without fetching managed audio or enrolling/conditioning a voice. At synthesis time, the engine resolves the verified audio and lazily enrolls it; the enrolled voice is reused within that session. Canonical synthesis identity includes content/provenance pins, the selected enrollment engine/options, and the exact AkinVox transcript hash, but excludes local/cache paths and raw transcript text.

## Kokoro quality and voice-level options

`voicerender engines options kokoro --json` lists the adapter's accepted options and structured fields. The compatible shorthand forms remain `short_sentence = "off" | "wrap" | "phrase" | "randomized-phrase"` and `voice_level = "off" | "calibrated"`. These options may also be provided as tables:

```toml
[roles.narrator.binding.options.short_sentence]
enabled = true
resolve_mode = "randomized-phrase"
min_phoneme_length = 30
phoneme_pretext = "—"
phrase_fallback_tries = 1

[roles.narrator.binding.options.voice_level]
mode = "calibrated"
# Optional finite numeric PyKokoro gain; explicit gain takes precedence.
# gain_db = -1.5
```

`short_sentence` supports `enabled`, `resolve_mode` (`wrap`, `phrase`, or `randomized-phrase`), `min_phoneme_length`, `phoneme_pretext`, and `phrase_fallback_tries`; the defaults are shown above. An omitted setting leaves PyKokoro's short-sentence fix unset, while the `off` shorthand disables it. `voice_level` accepts `mode = "off" | "calibrated"` and optional finite numeric `gain_db`. Calibrated mode uses PyKokoro's static per-model/quality/voice catalog; it is not runtime loudness normalization. Automatically matched catalog calibration is not available for custom enrolled voices, though explicit gain is supported. VoiceRender does not add loudness DSP; its separate PCM boundary adapter clamps finite full-scale overshoots and records diagnostics.

Kokoro enrollment modes have different semantics. **Inno** uses reference audio to tune a voice, requires no transcript, and may accept `fmax`. **AkinVox** performs transcript-conditioned cloning: it requires the exact non-empty reference transcript, currently supports English only, and requires synthesis speed `1.0`. The generic `voicerender voice clone --engine kokoro ... --transcript ...` path binds that transcript into AkinVox's enrollment options. Conflicting transcript sources are rejected, and Inno does not accept a transcript. Reference resolution and enrollment remain lazy until rendering; preflight only checks pins/metadata and measures the text request.

## Qwen3Synth Base and VoiceDesign

Qwen3Synth is an optional renderer. Install `voicerender[qwen3]` and install model bundles explicitly with ONNXVoice before using them:

```bash
python -m pip install "voicerender[qwen3]"
onnxvoice install qwen3:voice-design-1.7b --quality int4
onnxvoice install qwen3:base-0.6b --quality int4
```

VoiceDesign is a separate, explicit asset-generation operation. It creates one reference WAV from a fixed utterance and may write a source sidecar with the exact prompt, transcript, model/runtime identity, and generation settings:

```bash
voicerender voice design \
  --prompt "A warm, calm narrator with clear diction." \
  --text "This fixed sentence will be spoken by the designed voice." \
  --language en \
  --output narrator.wav \
  --source-output narrator.voice-source.toml
voicerender voice inspect narrator.voice-source.toml --json
```

The WAV SHA-256 defines the ordinary reference identity. Prompt and seed are provenance, not identity. Output is not overwritten unless `--force` is explicit. Protect both files as sensitive voice data. The sidecar contains the prompt and exact spoken text.

Qwen Base accepts a pinned reference WAV with its exact transcript, or a prepared `.qvoice` profile. Prepare a Qwen-origin reference by reusing the transcript in its sidecar; arbitrary WAVs require an explicit transcript:

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

Alternatively, create a profile with `--reference-wav PATH` and `--option 'reference_text=Exact words spoken.'`, or supply a verified source sidecar with `--voice-source PATH`. `.qvoice` uses the `qwen3.qvoice.v2` prepared format and its complete-file SHA-256 defines prepared-voice identity. Only the Qwen adapter accepts this format initially. Profile creation and validation inspect pins and metadata, they do not generate or prepare voices. Qwen operations use installed assets only; install them with `onnxvoice install` rather than expecting implicit acquisition.

### Copying a designed WAV to PocketSynth

On a weaker machine, install PocketSynth and create a fresh local profile from the copied WAV:

```bash
python -m pip install "voicerender[pocket]"
voicerender voices list --engine pocket --offline
voicerender profile create \
  --engine pocket --target <pocket-target> --language en-US \
  --reference-wav ./narrator.wav \
  --output pocket-narrator.renderprofile.toml
```

The new profile computes a local content pin for the copied bytes. Do not reuse the creator's local RenderProfile or sidecar unchanged, since its absolute path points to the creator's machine. PocketSynth consumes the pinned WAV without importing or requiring Qwen3Synth. Qwen prompt and runtime provenance are not execution dependencies. If retaining that provenance matters, keep the protected source sidecar with the original asset records; the profile remains machine-local.
