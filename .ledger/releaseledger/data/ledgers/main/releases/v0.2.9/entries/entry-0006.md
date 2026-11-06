---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0006
release_version: v0.2.9
kind: internal
summary: Improved PhononSynth validation diagnostics and multilingual voice test coverage
status: accepted
audience: null
scopes: []
source_refs:
  - git:0b951e4c9690afd7c7392046dd3a63794fdd9990
paths:
  - tests/test_kokoro_reference.py
  - tests/test_phononsynth_engine.py
  - tests/test_preflight.py
  - tests/test_profile_roles_cli.py
  - tests/test_voice_descriptors.py
  - voicerender/engines/phononsynth.py
  - voicerender/voices_api.py
issues: []
prs: []
sources:
  - git:0b951e4c9690afd7c7392046dd3a63794fdd9990
contributors:
  - "@holgern"
breaking: false
internal: true
order: 6
---
