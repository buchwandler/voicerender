---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: 0.2.1
kind: added
summary:
  Added optional Qwen3Synth support for VoiceDesign references, prepared voice
  profiles, and Qwen Base rendering
status: accepted
audience: null
scopes: []
source_refs:
  - git:3c82349993e1adc0073141215848ce0d3c10817c
paths:
  - README.md
  - docs/renderprofile.md
  - pyproject.toml
  - tests/test_cli.py
  - tests/test_engine_registry.py
  - tests/test_pocketsynth_managed.py
  - tests/test_qwen3_voice_design.py
  - tests/test_qwen3synth_engine.py
  - tests/test_renderprofile.py
  - tests/test_soprosynth_engine.py
  - tests/test_voice_source_codec.py
  - tests/wheel_smoke.py
  - voicerender/__init__.py
  - voicerender/cli.py
  - voicerender/engines/profiles.py
  - voicerender/engines/qwen3synth.py
  - voicerender/engines/registry.py
  - voicerender/voice_design.py
  - voicerender/voice_source_codec.py
  - voicerender/voice_sources.py
issues: []
prs: []
sources:
  - git:3c82349993e1adc0073141215848ce0d3c10817c
contributors:
  - "@holgern"
breaking: false
internal: false
order: 1
---
