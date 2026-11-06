---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0002
release_version: v0.2.9
kind: added
summary: Added optional PhononSynth support with model, voice, and prepared-text synthesis
status: accepted
audience: null
scopes: []
source_refs:
  - git:7c5933312eb77b2d38fe3051086ca85f4e3f363a
paths:
  - .github/workflows/tests.yml
  - README.md
  - pyproject.toml
  - tests/test_engine_registry.py
  - tests/test_phononsynth_engine.py
  - tests/test_renderplan_input.py
  - tests/wheel_smoke.py
  - voicerender/engines/phononsynth.py
  - voicerender/engines/registry.py
issues: []
prs: []
sources:
  - git:7c5933312eb77b2d38fe3051086ca85f4e3f363a
contributors:
  - "@holgern"
breaking: false
internal: false
order: 2
---
