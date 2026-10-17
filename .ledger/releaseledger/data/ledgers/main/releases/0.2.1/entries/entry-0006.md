---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0006
release_version: 0.2.1
kind: added
summary: Added typed progress, evidence-based ETA, and workload estimates
status: accepted
audience: null
scopes: []
source_refs: []
paths:
  - voicerender/events.py
  - voicerender/progress.py
  - voicerender/cli.py
  - voicerender/estimates.py
  - README.md
  - docs/index.md
issues: []
prs: []
sources: []
contributors: []
breaking: false
internal: false
order: 6
---

Render and preflight callbacks expose item and chapter progress with ETA only after successful synthesis timing samples. CLI progress stays on stderr, while pre-run estimates report verified output reuse and unknown duration without a calibrated timing model.
