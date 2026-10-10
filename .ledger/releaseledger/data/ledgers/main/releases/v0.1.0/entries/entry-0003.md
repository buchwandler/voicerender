---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0003
release_version: v0.1.0
kind: added
summary:
  Added Utterplan v5 parsing, AudioCompose integration, and render-manifest
  validation
status: accepted
audience: null
scopes: []
source_refs:
  - git:264eb9b941d2b2086e092aff11ebebae47ed293c
paths:
  - MANIFEST.in
  - README.md
  - docs/utterplan-v5-integration.md
  - examples/pauses/audiobook.toml
  - examples/plans/chapter.utterplan.toml
  - examples/voices/voice.toml
  - pyproject.toml
  - tests/fixtures/utterplan_v5_contract.toml
  - tests/test_audiocompose_integration.py
  - tests/test_backends.py
  - tests/test_cli_v5.py
  - tests/test_manifest.py
  - tests/test_render_v5.py
  - tests/test_utterplan_v5.py
  - voicerender/__init__.py
  - voicerender/cli.py
  - voicerender/engines/base.py
  - voicerender/engines/piper.py
  - voicerender/engines/pocket.py
  - voicerender/errors.py
  - voicerender/inputs/__init__.py
  - voicerender/inputs/utterplan_v5.py
  - voicerender/integrations/__init__.py
  - voicerender/integrations/audiocompose.py
  - voicerender/manifest.py
  - voicerender/models.py
  - voicerender/render.py
  - voicerender/render_v5.py
  - voicerender/schemas/render-manifest-v2.schema.json
issues: []
prs: []
sources:
  - git:264eb9b941d2b2086e092aff11ebebae47ed293c
contributors:
  - "@holgern"
breaking: false
internal: false
order: 3
---
