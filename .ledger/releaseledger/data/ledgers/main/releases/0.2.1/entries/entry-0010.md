---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0010
release_version: 0.2.1
kind: fixed
summary: Fixed Sopro preflight capacity handling for authoritative and legacy measurements
status: accepted
audience: null
scopes: []
source_refs:
  - tl:task-0041
paths:
  - README.md
  - .github/workflows/tests.yml
  - tests/test_soprosynth_engine.py
  - tests/test_sopro_release_compatibility.py
  - voicerender/engines/api_probe.py
  - voicerender/engines/soprosynth.py
  - voicerender/cli.py
issues: []
prs: []
sources: []
contributors: []
breaking: false
internal: false
order: 10
---

VoiceRender validates Sopro's exact model-token measurements, honors producer fit results without requiring a finite maximum, and enables a narrow compatibility bridge for SoproSynth 0.1.0/0.1.1 only when API v1 confirms engine-owned text boundaries. Other unknown-fit measurements remain rejected; character limits are never converted into token limits.
