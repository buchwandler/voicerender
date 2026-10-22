---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0004
release_version: 0.2.6
kind: fixed
summary:
  Fixed bundle-root validation for paths that resolve to canonical .ssmd-bundle
  directories
status: accepted
audience: null
scopes: []
source_refs:
  - git:4ab4750c673eadbd57ca91a42099fa87f13797cb
paths:
  - README.md
  - tests/test_bundle_contract.py
  - tests/test_cli.py
  - tests/test_document_contract.py
  - tests/test_render_book.py
  - voicerender/book_layout.py
  - voicerender/bundle_contract.py
  - voicerender/cli.py
  - voicerender/document_contract.py
issues: []
prs: []
sources:
  - git:4ab4750c673eadbd57ca91a42099fa87f13797cb
contributors:
  - "@holgern"
breaking: false
internal: false
order: 4
---
