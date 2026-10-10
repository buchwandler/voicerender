---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0005
release_version: v0.1.0
kind: added
summary:
  Added engine discovery, capacity probing, synthesis backends, and v5 handoff
  improvements
status: accepted
audience: null
scopes: []
source_refs:
  - git:c4a2872bd2830ef7eb348688b2194af169bbe6aa
paths:
  - .github/workflows/tests.yml
  - README.md
  - docs/utterplan-v5-integration.md
  - examples/pauses/audiobook.toml
  - pyproject.toml
  - tests/test_audiocompose_integration.py
  - tests/test_capacity.py
  - tests/test_cli_v5.py
  - tests/test_engine_registry.py
  - tests/test_render_v5.py
  - tests/test_renderer.py
  - tests/test_utterplan_v5.py
  - voicerender/__init__.py
  - voicerender/capacity.py
  - voicerender/cli.py
  - voicerender/engines/__init__.py
  - voicerender/engines/api_probe.py
  - voicerender/engines/base.py
  - voicerender/engines/catalog.py
  - voicerender/engines/inflectsynth.py
  - voicerender/engines/kittensynth.py
  - voicerender/engines/kokoro.py
  - voicerender/engines/metadata.py
  - voicerender/engines/pipersynth.py
  - voicerender/engines/pocketsynth.py
  - voicerender/engines/registry.py
  - voicerender/engines/selection.py
  - voicerender/engines/supertonicsynth.py
  - voicerender/errors.py
  - voicerender/events.py
  - voicerender/handoff.py
  - voicerender/inputs/utterplan_v5.py
  - voicerender/integrations/__init__.py
  - voicerender/integrations/audiocompose.py
  - voicerender/lexicons.py
  - voicerender/manifest.py
  - voicerender/models.py
  - voicerender/render.py
  - voicerender/render_v5.py
  - voicerender/schemas/render-manifest-v2.schema.json
issues: []
prs: []
sources:
  - git:c4a2872bd2830ef7eb348688b2194af169bbe6aa
contributors:
  - "@holgern"
breaking: false
internal: false
order: 5
---
