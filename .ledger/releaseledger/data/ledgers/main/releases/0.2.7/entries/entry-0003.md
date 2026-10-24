---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0003
release_version: 0.2.7
kind: added
summary:
  Added bounded structured diagnostics for render-time failures across direct,
  document, book, and bundle results
status: accepted
audience: null
scopes: []
source_refs:
  - git:9fa235e094ecf7f9fff064a7ef87d6e9ca40d092
  - git:4b52e7148b630c2ca31efcae25776f408319ba49
paths:
  - README.md
  - docs/index.md
  - tests/test_bundle_batch_render.py
  - tests/test_cli.py
  - tests/test_document_contract.py
  - tests/test_error_diagnostics.py
  - tests/test_kokoro_reference.py
  - tests/test_render_book.py
  - tests/test_renderplan_render.py
  - tests/test_variants.py
  - voicerender/book.py
  - voicerender/bundle_batch.py
  - voicerender/cli.py
  - voicerender/document.py
  - voicerender/engines/kokoro.py
  - voicerender/error_diagnostics.py
  - voicerender/errors.py
  - voicerender/events.py
  - voicerender/renderplan_render.py
  - voicerender/variants.py
issues: []
prs: []
sources:
  - git:9fa235e094ecf7f9fff064a7ef87d6e9ca40d092
  - git:4b52e7148b630c2ca31efcae25776f408319ba49
contributors:
  - "@holgern"
breaking: false
internal: false
order: 3
---
