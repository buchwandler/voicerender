---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0002
release_version: v0.1.0
kind: added
summary:
  Added CLI rendering with Piper and Pocket backends plus persistent render
  storage
status: accepted
audience: null
scopes: []
source_refs:
  - git:fb7faafcdaaac6c934f5f6bff02b35612d1817b2
paths:
  - .codecrate.toml
  - .gitignore
  - .ledger/ledger.toml
  - .ledger/releaseledger/.ledger-project.toml
  - .ledger/releaseledger/config.toml
  - .ledger/releaseledger/data/.ledger-project.toml
  - .ledger/taskledger/.ledger-project.toml
  - .ledger/taskledger/config.toml
  - .pre-commit-config.yaml
  - README.md
  - docs/Makefile
  - docs/conf.py
  - docs/make.bat
  - docs/make.py
  - docs/requirements.txt
  - examples/plans/chapter.voiceplan.toml
  - examples/voices/piper.toml
  - examples/voices/pocket-reference.toml
  - pyproject.toml
  - tests/test_backends.py
  - tests/test_renderer.py
  - voicerender/__init__.py
  - voicerender/__main__.py
  - voicerender/cli.py
  - voicerender/engines/__init__.py
  - voicerender/engines/base.py
  - voicerender/engines/piper.py
  - voicerender/engines/pocket.py
  - voicerender/errors.py
  - voicerender/models.py
  - voicerender/py.typed
  - voicerender/render.py
  - voicerender/storage.py
issues: []
prs: []
sources:
  - git:fb7faafcdaaac6c934f5f6bff02b35612d1817b2
contributors:
  - "@holgern"
breaking: false
internal: false
order: 2
---
