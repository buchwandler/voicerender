---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: 0.2.3
kind: changed
summary:
  Changed preflight handling so unavailable managed KittenSynth dependencies
  reject without scheduling a replan
status: accepted
audience: null
scopes: []
source_refs:
  - git:67e4d8ec44e04f96d7041afa8c87c762bec87b6f
paths:
  - pyproject.toml
  - tests/test_kittensynth_engine.py
  - tests/test_preflight.py
  - voicerender/engines/kittensynth.py
issues: []
prs: []
sources:
  - git:67e4d8ec44e04f96d7041afa8c87c762bec87b6f
contributors:
  - "@holgern"
breaking: false
internal: false
order: 1
---
