# UtterPlan v5 and VoiceRender handoff

VoiceRender is a standalone speech producer. It consumes UtterPlan v5 and a VoiceRender voice profile, writes source-addressed speech WAVs, and publishes a verified `voicerender.render-manifest.v2`. It does not resolve external media or perform timeline composition, pause insertion, mastering, or export.

## Source operations and selection

VoiceRender flattens `flow` and each flow unit's `segment` array in source order. Flow and segment indices are zero-based; a source locator is `f{flow_index:06d}.s{segment_index:03d}`.

Before engine access, each selected operation is classified:

- An `audio` directive is an external-media operation. It remains in the source-operation manifest, is never sent to a TTS engine, and has no fabricated WAV.
- A nonblank text operation without external audio is speech and is eligible for synthesis.
- An empty, non-audio operation is unsupported and fails closed.

Selections preserve source order even if locator arguments were supplied in another order. Media-only selections are valid and produce a complete zero-speech manifest. The manifest records selected source operations separately from selected speech operations.

`segment.language` takes precedence over the plan's document language. Profile defaults and engine options do not replace language authored on a segment. Logical roles resolve from an explicit segment role, an explicit profile default, or the sole configured profile role; ambiguous multi-role plans fail preflight.

## Semantic preflight and lowering

VoiceRender validates plan structure and hashes, profile roles, engine API compatibility, target selection, engine option support, and document semantics before opening a synthesis session. Unsupported semantics are errors, not warnings or silent no-ops.

- Pronunciation overrides are passed only when the selected engine advertises compatible pronunciation support and alphabet.
- Linguistic tokens are passed only when the selected engine advertises token support.
- Numeric prosody maps rate to a tempo factor, pitch to semitones, and volume to dB under `voicerender.numeric-composition.v1`. These directives are recorded for downstream processing and do not modify the synthesized WAV or acoustic cache key.
- Unsupported emphasis, `say_as`, substitutions, unknown extensions, and unrecognized document requirements fail before inference.

UtterPlan `document.semantics` may contain supported voice bindings, validated prosody transition durations, and `requires.extensions`. Unknown semantics and required extensions fail closed. The manifest preserves accepted document semantics.

## Capacity and acoustic identity

Batch and in-memory synthesis use the same request/session/capacity spine. Engines may measure requests; oversized requests are split only at legal boundaries, remeasured, merged, and have timing rebased. Pronunciation and token spans protect their semantic ranges. Split requests and timing provenance are retained in `RenderedSpeech.metadata`.

The acoustic cache identity commits to the effective engine and API identity, resolved target/model and revision, reference-audio bytes, request text/language/options/tokens/pronunciation, capacity schema, and output encoding. Composition-only prosody is represented separately so it does not force redundant speech synthesis. Source location and semantic pauses are not acoustic inputs.

## Render manifest v2

A v2 manifest is written as `render-manifest.json` beside a `segments/` directory. It binds the exact UtterPlan file and plan identity, source selection and operation provenance, profile and engine identity, semantic lowering, acoustic cache keys, and verified speech artifacts. Completed speech entries record a safe bundle-relative WAV path, byte SHA-256, sample rate, channel count, frame count, encoding, timing provenance, and preserved edge policy.

The manifest is updated as segments move through `pending`, `rendered`, `reused`, or `failed`. Resume always verifies artifact bytes and WAV shape; status alone is not proof. Absolute and traversal paths, symlink escapes, malformed audio, and mismatched hashes are rejected. Zero-speech manifests contain no WAV entries and are valid when complete.

Use the narrow public handoff API rather than private VoiceRender modules:

```python
from voicerender import (
    load_source_plan,
    validate_manifest_source,
    verify_render_manifest,
)

plan = load_source_plan("chapter.utterplan.toml")
manifest = verify_render_manifest("renders/chapter/render-manifest.json")
validate_manifest_source(plan, manifest)
```

## Standalone workflow

The CLI can validate without synthesis, render, and verify the handoff:

```bash
voicerender validate chapter.utterplan.toml --voice voice.toml
voicerender render chapter.utterplan.toml --voice voice.toml --out-dir renders/chapter
voicerender inspect renders/chapter/render-manifest.json --verify
```

Engine and voice discovery and profile checks use the same registry:

```bash
voicerender engines list
voicerender engines inspect piper
voicerender voices list --engine piper --lang en-US
voicerender profiles validate voice.toml
```

The Python batch API is `render_utterplan(...)`; `render_utterplan_v5(...)` remains a transition alias. Synchronous typed events are available through `on_event`; the library itself does not print progress. The public `SpeechRequest`, `open_session`, and `synthesize` APIs provide in-memory synthesis through the same registry and capacity fitting.

Downstream consumers own external-media resolution, pause policy, timeline construction, mixing, mastering, and export. They should verify the render manifest and join it to the exact source plan before using any speech artifact.
