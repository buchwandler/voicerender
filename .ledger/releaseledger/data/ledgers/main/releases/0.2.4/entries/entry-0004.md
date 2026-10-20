---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0004
release_version: 0.2.4
kind: changed
summary:
  Improved Kokoro controls, reference cloning, PCM safety, and render failure
  diagnostics
status: accepted
audience: null
scopes: []
source_refs:
  - git:3d486d5fddcc8612430c321760e9ebd3d84607fc
paths:
  - README.md
  - docs/renderprofile.md
  - tests/test_engine_options.py
  - tests/test_kittensynth_engine.py
  - tests/test_kokoro_reference.py
  - tests/test_renderplan_render.py
  - tests/test_voice_clone.py
  - voicerender/book.py
  - voicerender/cli.py
  - voicerender/document.py
  - voicerender/engines/base.py
  - voicerender/engines/kittensynth.py
  - voicerender/engines/kokoro.py
  - voicerender/renderplan_render.py
  - voicerender/voice_clone.py
issues: []
prs: []
sources:
  - git:3d486d5fddcc8612430c321760e9ebd3d84607fc
contributors:
  - "@holgern"
breaking: false
internal: false
order: 4
---
