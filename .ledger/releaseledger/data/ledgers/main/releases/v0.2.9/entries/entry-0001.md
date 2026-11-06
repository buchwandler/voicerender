---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: v0.2.9
kind: fixed
summary:
  Fixed book reports to enforce the shared 2 MiB limit and omit persisted execution
  fingerprints
status: accepted
audience: null
scopes: []
source_refs:
  - git:16913d40e23b25987b17274d02248d8456360c7e
paths:
  - tests/test_render_book.py
  - tests/test_renderplan_render.py
  - voicerender/book.py
  - voicerender/report_contract.py
  - voicerender/variants.py
issues: []
prs: []
sources:
  - git:16913d40e23b25987b17274d02248d8456360c7e
contributors:
  - "@holgern"
breaking: false
internal: false
order: 1
---
