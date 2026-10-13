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

Install the selected engine extra and make its model assets available before
preflight/render. VoiceRender preflight measures requests but never synthesizes;
rendering requires a matching all-`ok` report. See the
[UtterPlan SSMDBook workflow](https://pypi.org/project/utterplan/) and the
[RenderPlan package contract](https://pypi.org/project/renderplan/) for the
producer/consumer boundary.
