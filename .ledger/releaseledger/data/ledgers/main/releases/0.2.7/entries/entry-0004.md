---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0004
release_version: 0.2.7
kind: added
summary:
  Added bounded engine-session residency and crash-safe cross-platform render-lock
  recovery with resumable partial outputs
status: accepted
audience: null
scopes: []
source_refs:
  - git:a8df40403864a1f5b42f14dc850e8a6f5cca2e9c
  - git:8f7b701fffe3d5faf6c45911f7e87e566c6cd479
  - git:009bc2f036b00d64b1720b5d58ebfe2ea35397dc
  - git:ed53140775d00d439a5f2a303ddf20a6aaa8d7cc
  - git:b089c79c13aebf0f508c7aa7dc75a720c5085f54
  - git:f74ce0082a24f0ebb8d6b45ec6ecb18c6efc4725
  - git:88a2bdebbd0d119f4679a21465b7e6ac7e39e205
  - git:1132f5f0a56d293cf82f874f08b88b9c45601298
paths:
  - .github/workflows/tests.yml
  - README.md
  - docs/index.md
  - tests/test_bundle_batch_render.py
  - tests/test_bundle_render.py
  - tests/test_cli.py
  - tests/test_document_contract.py
  - tests/test_render_book.py
  - tests/test_render_lock.py
  - tests/test_renderplan_render.py
  - tests/test_session_pool.py
  - voicerender/book.py
  - voicerender/bundle.py
  - voicerender/bundle_batch.py
  - voicerender/cli.py
  - voicerender/document.py
  - voicerender/errors.py
  - voicerender/events.py
  - voicerender/preflight.py
  - voicerender/render.py
  - voicerender/render_lock.py
  - voicerender/renderplan_render.py
  - voicerender/session_pool.py
  - voicerender/storage.py
issues: []
prs: []
sources:
  - git:a8df40403864a1f5b42f14dc850e8a6f5cca2e9c
  - git:8f7b701fffe3d5faf6c45911f7e87e566c6cd479
  - git:009bc2f036b00d64b1720b5d58ebfe2ea35397dc
  - git:ed53140775d00d439a5f2a303ddf20a6aaa8d7cc
  - git:b089c79c13aebf0f508c7aa7dc75a720c5085f54
  - git:f74ce0082a24f0ebb8d6b45ec6ecb18c6efc4725
  - git:88a2bdebbd0d119f4679a21465b7e6ac7e39e205
  - git:1132f5f0a56d293cf82f874f08b88b9c45601298
contributors:
  - "@holgern"
breaking: false
internal: false
order: 4
---
