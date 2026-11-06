---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0005
release_version: v0.2.9
kind: changed
summary:
  Changed PhononSynth preflight to honor authoritative fit and reject incompatible
  APIs
status: accepted
audience: null
scopes: []
source_refs:
  - git:05e2511c2e77b9b4b20e1220397332b6bef08029
paths:
  - .github/workflows/tests.yml
  - README.md
  - pyproject.toml
  - tests/test_engine_registry.py
  - tests/test_phononsynth_engine.py
  - tests/test_preflight.py
  - tests/test_renderplan_input.py
  - voicerender/engines/phononsynth.py
issues: []
prs: []
sources:
  - git:05e2511c2e77b9b4b20e1220397332b6bef08029
contributors:
  - "@holgern"
breaking: false
internal: false
order: 5
---
