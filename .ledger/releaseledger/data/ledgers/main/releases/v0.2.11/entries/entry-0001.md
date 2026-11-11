---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: v0.2.11
kind: changed
summary:
  Changed VoiceRender to preserve full synthesis waveforms and delegate boundary
  conditioning and pauses to AudioCompose
status: accepted
audience: null
scopes: []
source_refs:
  - git:805445a6d246ce5d06a43eb2b2d099d13fed0ac9
paths:
  - README.md
  - docs/renderprofile.md
  - pyproject.toml
  - voicerender/audio_artifacts.py
  - voicerender/audio_boundaries.py
  - voicerender/bundle_render_inspect.py
  - voicerender/cli.py
  - voicerender/render.py
  - voicerender/renderplan_render.py
issues: []
prs: []
sources:
  - git:805445a6d246ce5d06a43eb2b2d099d13fed0ac9
contributors:
  - "@holgern"
breaking: false
internal: false
order: 1
---
