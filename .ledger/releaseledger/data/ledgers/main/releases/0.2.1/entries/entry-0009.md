---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0009
release_version: 0.2.1
kind: added
summary: Added read-only bundle role and cast-readiness inspection
status: accepted
audience: null
scopes: []
source_refs:
  - tl:task-0032
paths:
  - README.md
  - docs/index.md
  - docs/renderprofile.md
  - voicerender/bundle_roles.py
  - voicerender/bundle_inspect.py
  - voicerender/cli.py
issues: []
prs: []
sources: []
contributors: []
breaking: false
internal: false
order: 9
---

Introduced bundle roles for fresh source-role discovery and profile/variant cast checks, plus bundle inspect for integrated variant and render-state reporting.
