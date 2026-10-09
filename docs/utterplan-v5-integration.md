# UtterPlan v5 / VoiceRender interoperability contract

This document freezes the VoiceRender consumer-side contracts for direct UtterPlan v5 rendering and the optional AudioCompose materializer. It does not change UtterPlan or AudioCompose. The initial implementation is deliberately strict: the source plan is an input contract, not a format VoiceRender may silently normalize or partially interpret.

## Source addressing and selection

- Flatten `flow` in array order and each unit's `segment` array in order. Both indices are zero-based; `segment_index` is local to its flow unit.
- A source address is `(plan_id, flow_index, segment_index)`. Its printable locator is `f{flow_index:06d}.s{segment_index:03d}` (for example, `f000042.s001`). The locator is an address, not a persisted UtterPlan identifier.
- `segment.language` overrides root `language`. A missing logical `voice` is resolved only by an explicit profile default or by the sole configured role. Multiple roles without a default are an error.
- `--take N` selects the first N flattened speech segments. An explicit locator selection is emitted in source order, irrespective of selector argument order. Composition accepts a full selection or a contiguous prefix/range; discontiguous selections fail rather than implying a physical boundary that was not authored.
- The compact v5 prototype referenced by the implementation brief was not present in this checkout or the adjacent UtterPlan fixture set. Structural coverage uses a synthetic 191-unit/303-segment vector; the exact external prototype remains a deployment smoke fixture when supplied.

## Three non-interchangeable identities

1. **Source address:** plan ID plus flow and segment indices; identifies an occurrence in one source plan.
2. **Semantic segment digest:** `voicerender-segment-semantics.v1`, computed from a normalized consumer projection and not presented as an UtterPlan-owned hash. The projection has `text`, resolved `language`, `directives`, expanded local `tokens`, `markers`, optional `pause_before`/`pause_after`, and optional `heading`. It includes all renderer-relevant segment semantics and excludes array position, profile, output policy, and engine configuration.
   - `directives` is the mapping of v5 directive fields (`voice`, `pronunciation`, `prosody`, `emphasis`, `say_as`, `substitution`, `audio`, `extensions`) after the v5 typed-default expansion (for example, absent prosody attributes are `null`, and pronunciation alphabet defaults to `ipa`). Unknown v5 segment keys fail closed.
   - Each token is represented as `{start, end}` plus its present `pos`, `tag`, `lemma`, and `morph` values. Missing lemma uses the UtterPlan v5 wire decoder's `text[start:end].casefold()` default. Offsets are Python Unicode code points.
   - The digest input is `{"schema":"voicerender-segment-semantics.v1","segment":<projection>}`. Canonical JSON is UTF-8 encoded with `ensure_ascii=False`, lexicographically sorted object keys, separators `(',', ':')`, and non-finite numbers forbidden. Do not apply Unicode normalization. The digest is lowercase SHA-256 prefixed with `sha256:`.
   - This digest does not replace the validated `flow.hash` or plan ID. UtterPlan currently uses `utterplan-flow-v1` for a flow hash and its own `flow_plan_id` for plan identity; VoiceRender preserves those values and does not reinterpret one as another.
3. **Acoustic cache key:** SHA-256 of the versioned effective lowered synthesis request, concrete resolved engine/model/voice/speaker/options, installed engine version, model identity when verifiable, reference-audio byte hash, and edge/output encoding algorithm. It excludes source address, plan/flow hashes, semantic pauses, and composition-only operations. A path change to identical reference bytes does not change the key.

The exact input file byte hash (`sha256:` + SHA-256 of the original TOML bytes) is also recorded and must match for materialization. Therefore harmless TOML reformatting invalidates an existing manifest join. The reader never rewrites the input before hashing.

## Directive handling

Backend capabilities are explicit. Each directive must be classified during preflight as:

- **native:** supported by the selected backend, included in the effective synthesis request/cache key, and recorded as native in the manifest;
- **composition:** supported only by an explicit, typed mapping to a numeric AudioCompose operation, included in a versioned composition-lowering record but not in the speech cache key;
- **unsupported:** rejected before the manifest is written or synthesis begins.

Successful entries have no unsupported directives. Logical voice routing is not itself a concrete engine voice. Effects must never be applied both natively and at composition time. An implementation may support a conservative subset; it may not silently discard v5 semantics.

## Manifest identity and media contract

- New v5 output uses `voicerender.render-manifest.v2`; v1 keeps its existing meaning and is not AudioCompose-ready by implication.
- `manifest_id` is `sha256:` plus the digest of canonical JSON for the complete final manifest with `manifest_id` omitted. There are no timestamps. For identity only, transient per-entry `status` and `error` are omitted; source/selection/profile/semantic-lowering fields and verified audio metadata remain covered. Absolute paths are forbidden; bundle-relative POSIX paths are retained and covered.
- Each selected segment has exactly one entry, in source order. Selection metadata is sufficient to reconstruct that exact selected list. `complete` is true exactly when every selected entry is `rendered` or `reused` and its audio validates; an unselected plan tail is irrelevant.
- Progress is manifest-first: pending entries are atomically written before synthesis, and the same manifest is atomically refreshed after each final success/failure. Successful entries carry actual WAV byte SHA-256, sample rate, mono channel count, frame count, and `PCM_16` encoding. Resume requires re-verification of the target bytes; status alone is never proof.
- Reject absolute/traversal paths, symlink escapes/retargeting, duplicate media paths, missing files, wrong hashes, non-mono files, and rate/frame mismatches.

## Edges and timing

- Initial edge contract: `preserved-unmeasured-v1`. Speech WAV rendering performs no edge trimming or semantic-pause insertion. Record zero trim counts and `leading_silence_frames`/`trailing_silence_frames` as `null` (unknown); never claim the edge padding is measured or normalized. Pause-policy seconds mean additional AudioJob silence, not a guaranteed total acoustic gap.
- Timing uses final clip-local frame coordinates. The text range uses Unicode code-point offsets. Spans must be ordered and bounded by the final frame count. Only actual engine timing is `provided`; `estimated` requires an estimator/method record; otherwise use `absent` with no spans.

## Pause policy and AudioCompose boundary

- `pause.toml` is read only by the optional producer integration. Durations are finite nonnegative seconds; symbolic classes must have an explicit mapping; authored `{type = "timed", time = "500ms"}` resolves to exactly `0.500` seconds.
- Resolve `pause_after` using the left segment's logical voice and `pause_before` using the right segment's logical voice. For a physical boundary, merge the two resolved cues with `max`; create at most one `Silence` and none for a zero duration. Do not append an automatic trailing pause to the final selected clip.
- The optional integration validates the entire plan/manifest/media/pause join before creating any `AudioClip`. It then uses only public AudioCompose types: `AudioClip`, `AudioFileSource`, `AudioSpan`, numeric operations, `Silence`, `OutputPolicy`, and `AudioJob`. Current `AudioJob.save()` copies input audio; VoiceRender will not invent a reference-only JSON format.
- Core VoiceRender imports no AudioCompose module. The integration imports AudioCompose lazily when a builder call is made. AudioCompose DSP, UtterPlan compilation, Readio orchestration, and AudioExport remain outside this checkout.

## Contract evidence

The adjacent UtterPlan project defines canonical JSON as sorted compact UTF-8 JSON with Unicode unescaped and NaN disallowed. Its current v5 `flow.hash` hashes the local executable semantics of all segments in the flow. VoiceRender keeps that contract separate from its per-segment consumer digest and exact file fingerprint. The installed AudioCompose public API accepts verified file sources and already supports the required clip, span, numeric operation, silence, output policy, and AudioJob models; this integration does not require AudioCompose core changes.

## Command-line and Python use

Direct v5 rendering selects the role-routed `voice.toml` interface; `--profile` and `--voices` remain legacy VoicePlan aliases:

```bash
voicerender render examples/plans/chapter.utterplan.toml \
  --voice examples/voices/voice.toml \
  --out-dir renders/chapter
voicerender inspect renders/chapter/manifest.json --verify --json
```

Selection accepts either `--take N` or comma-separated source locators such as `--select f000000.s000,f000001.s000`. Render manifests use `voicerender.render-manifest.v2`; `inspect` verifies their WAV bytes and metadata by default.

Install the optional integration extra to materialize the verified render as an AudioCompose bundle:

```bash
python -m pip install -e ".[audiocompose]"
voicerender audiojob examples/plans/chapter.utterplan.toml \
  --renders renders/chapter/manifest.json \
  --pauses examples/pauses/audiobook.toml \
  --output renders/chapter-audiojob
```

The pause policy maps each symbolic class to a finite nonnegative duration. The materializer uses public AudioCompose models and `AudioJob.save()` so source WAVs are copied into the bundle. The callable API is `voicerender.render_utterplan_v5(...)`; `build_audio_job(plan, manifest, pause_policy)` is available from `voicerender.integrations.audiocompose` and imports AudioCompose only when called. Core rendering and manifest inspection remain usable without the optional package.
