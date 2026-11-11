---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: v0.2.10
kind: fixed
summary:
  Fixed regional voice discovery to include voices with compatible base-language
  metadata
status: accepted
audience: null
scopes: []
source_refs:
  - git:4828dadb8361678818718635c4765d64eb185a27
paths:
  - tests/test_voice_descriptors.py
  - voicerender/voices_api.py
issues: []
prs: []
sources:
  - git:4828dadb8361678818718635c4765d64eb185a27
contributors:
  - "@holgern"
breaking: false
internal: false
order: 1
---
