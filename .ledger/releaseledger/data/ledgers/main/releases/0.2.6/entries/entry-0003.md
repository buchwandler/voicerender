---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0003
release_version: 0.2.6
kind: fixed
summary:
  Fixed voice-level option normalization and preserved Qwen3 seed options only
  when explicitly set
status: accepted
audience: null
scopes: []
source_refs:
  - git:5d929c9e092cd142238cc05a91b237dd00a65cd8
paths:
  - tests/test_cli.py
  - tests/test_kokoro_reference.py
  - tests/test_qwen3synth_engine.py
  - voicerender/engines/kokoro.py
  - voicerender/engines/qwen3synth.py
issues: []
prs: []
sources:
  - git:5d929c9e092cd142238cc05a91b237dd00a65cd8
contributors:
  - "@holgern"
breaking: false
internal: false
order: 3
---
