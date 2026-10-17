# RenderProfile lifecycle

A RenderProfile is the portable, user-owned renderer configuration shared by
[RenderPlan](https://pypi.org/project/renderplan/) producers and VoiceRender. It
captures engine-backed role bindings, renderer requirements, and planning policy;
it is data, not a loaded engine or synthesis session. Create profiles with
VoiceRender so bindings are seeded from installed adapter metadata, then review and
validate them before planning:

```bash
voicerender engines list
voicerender voices list --engine piper --lang en-US --offline --json
voicerender profile create \
  --engine piper --target en_US-amy-medium --language en-US \
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

A RenderProfile can bind several logical roles (for example, `narrator` and `guest`). Inspect the profile or list its role items without loading an engine:

```bash
voicerender profile inspect narration.renderprofile.toml --json
voicerender profile role list narration.renderprofile.toml --json
```

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

For the CLI, each mutation requires exactly one of `--output PATH` and `--in-place`. Use `--output` to review a proposal before replacing the original; `--in-place` is an explicit request to write over the selected profile:

```bash
voicerender profile role set narration.renderprofile.toml guest \
  --engine piper --target en_US-amy-medium --language en-US \
  -o narration.with-guest.renderprofile.toml
voicerender profile default-role narration.with-guest.renderprofile.toml guest \
  --in-place
voicerender profile role remove narration.with-guest.renderprofile.toml narrator \
  --new-default-role guest --in-place
```

The RenderPlan schema is unchanged: plans continue to carry the same profile identity and resolved per-role bindings.

## UtterPlan book variants

[UtterPlan](https://pypi.org/project/utterplan/) can compile an SSMDBook into one
renderer-neutral semantic plan per chapter, then use `render-book` to project
selected saved plans for one profile identity:

```bash
utterplan compile-book "Novel.ssmdbook" --chapters 5-17
utterplan render-book "Novel.ssmdbook" --chapters 5-17 \
  --profile narration.renderprofile.toml
```

The command writes to a profile-specific variant directory such as
`Novel.ssmdbook/renderplan/narration--<profile-id-prefix>/`. The directory contains
a canonical `renderprofile.toml` snapshot, one ordinary `.renderplan.toml` per
successful chapter, an UtterPlan-owned `renderplan-set.toml` collection index, and
an invocation-level `render-report.toml`. The set index and report are workspace
metadata—not RenderPlan contracts, RenderProfiles, or inputs to VoiceRender.

A different canonical profile identity selects a distinct default variant, so
multiple voices or policies can coexist while sharing the same `utterplan/`
semantic plans. Repeated invocations for the same profile incrementally update the
set index. The snapshot preserves the profile used by that collection, but a later
`utterplan render-book` invocation still takes a profile path with `--profile`.
UtterPlan loads the shared profile and emits RenderPlans; it does not call
VoiceRender, inspect engine availability, preflight, synthesize, or require
VoiceRender to be installed.

## Preflight and render each plan

VoiceRender consumes each chapter's ordinary RenderPlan independently. The
collection directory does not change the single-plan execution contract; the
`renderplan-set.toml` is not passed to these commands:

```bash
voicerender preflight \
  "Novel.ssmdbook/renderplan/narration--<profile-id-prefix>/chapter-0005.renderplan.toml" \
  -o chapter-0005.preflight.toml
voicerender render \
  "Novel.ssmdbook/renderplan/narration--<profile-id-prefix>/chapter-0005.renderplan.toml" \
  --preflight chapter-0005.preflight.toml \
  --out-dir renders/chapter-0005
```

Parsing a RenderProfile is engine-independent. `voicerender profile validate` checks the selected adapter/API, so it requires the selected engine SDK, but performs no model inference.

Install the selected engine extra and make its model assets available before
preflight/render. VoiceRender preflight measures requests but never synthesizes;
rendering requires a matching all-`ok` report. See the
[UtterPlan SSMDBook workflow](https://pypi.org/project/utterplan/) and the
[RenderPlan package contract](https://pypi.org/project/renderplan/) for the
producer/consumer boundary.

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
