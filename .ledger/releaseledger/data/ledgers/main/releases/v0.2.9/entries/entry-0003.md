---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0003
release_version: v0.2.9
kind: added
summary: Added read-only bundle render inspection and in-memory RenderPlan APIs
status: accepted
audience: null
scopes: []
source_refs:
  - git:d72b77973b675971c38158a3e51fe14011eb2004
paths:
  - tests/test_bundle_inspect.py
  - tests/test_render_book.py
  - tests/test_renderplan_render.py
  - voicerender/__init__.py
  - voicerender/book.py
  - voicerender/bundle_render_inspect.py
  - voicerender/preflight.py
  - voicerender/renderplan_render.py
issues: []
prs: []
sources:
  - git:d72b77973b675971c38158a3e51fe14011eb2004
contributors:
  - "@holgern"
breaking: false
internal: false
order: 3
---
