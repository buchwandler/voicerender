---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: 0.2.5
kind: changed
summary:
  Changed separate book/document rendering to manifest-driven SSMD bundle rendering
  with a unified bundle API
status: accepted
audience: null
scopes: []
source_refs:
  - git:0ab40750b3ede2e90e62b643c33aedc8687bf1b0
paths:
  - README.md
  - docs/index.md
  - docs/renderprofile.md
  - tests/test_bundle_contract.py
  - tests/test_bundle_render.py
  - tests/test_cli.py
  - tests/test_control_surface_exports.py
  - tests/test_document_contract.py
  - tests/test_render_book.py
  - tests/test_renderplan_render.py
  - tests/wheel_smoke.py
  - voicerender/__init__.py
  - voicerender/book.py
  - voicerender/book_contract.py
  - voicerender/book_layout.py
  - voicerender/bundle.py
  - voicerender/bundle_contract.py
  - voicerender/cli.py
  - voicerender/document.py
  - voicerender/document_contract.py
  - voicerender/errors.py
issues: []
prs: []
sources:
  - git:0ab40750b3ede2e90e62b643c33aedc8687bf1b0
contributors:
  - "@holgern"
breaking: true
internal: false
order: 1
---
