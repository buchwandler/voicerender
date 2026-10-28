from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voicerender import (
    PlanError,
    ProfileError,
    SpeechSegment,
    VoiceBinding,
    VoicePlan,
    VoiceProfile,
    render,
    render_plan,
)
from voicerender.engines import _legacy_adapters
from voicerender.engines.base import (
    EngineCapabilities,
    SynthesisOutput,
)


class FakeEngine:
    id = "testengine"
    package = "fake-backend"
    capabilities = EngineCapabilities(supports_reference_audio=True)

    def __init__(self):
        self.calls = []
        self.fail_once_for = None

        self.audio = np.full(160, 0.1, dtype=np.float32)
        self.word_timings = ()

    def version(self):
        return "1.0"

    def validate(self, binding):
        if binding.target_id == "bad":
            raise ProfileError("target not available")

    @contextmanager
    def open(self, binding):
        owner = self

        class Session:
            def synthesize(self, *, request_id, text, language):
                owner.calls.append((request_id, text, language, binding.target_id))
                if request_id == owner.fail_once_for:
                    owner.fail_once_for = None
                    raise RuntimeError("injected interruption")
                return SynthesisOutput(
                    audio=owner.audio,
                    sample_rate=16000,
                    word_timings=owner.word_timings,
                )

        yield Session()


@pytest.fixture
def mock_engine(monkeypatch):
    adapter = FakeEngine()
    monkeypatch.setitem(_legacy_adapters, adapter.id, adapter)
    return adapter


def segments():
    return [
        SpeechSegment(id="s1", scope_id="chapter", role="narrator", text="First."),
        SpeechSegment(id="s2", scope_id="chapter", role="guest", text="Second."),
        SpeechSegment(id="s3", scope_id="chapter", role="narrator", text="Third."),
    ]


def profile(*, narrator="narrator-model", guest="guest-model", name="cast"):
    return VoiceProfile(
        name=name,
        roles={
            "narrator": VoiceBinding(engine="testengine", target_id=narrator),
            "guest": VoiceBinding(engine="testengine", target_id=guest),
        },
    )


def read_manifest(directory):
    return json.loads((directory / "manifest.json").read_text())


def test_render_reuse_and_cross_profile_cache(tmp_path, mock_engine):
    out = tmp_path / "voice1"
    result = render(segments(), profile=profile(), output_dir=out, take=2)
    assert (result.rendered_count, result.reused_count, result.selected_count) == (
        2,
        0,
        2,
    )
    manifest = read_manifest(out)
    assert manifest["complete"] and len(manifest["segments"]) == 2
    assert manifest["source_plan_id"] is None
    for item in manifest["segments"]:
        assert item["wav"]["frames"] == 160
        assert item["wav"]["sample_rate"] == 16000
        with sf.SoundFile(out / item["path"]) as file:
            assert file.frames == 160
    second = render(segments(), profile=profile(), output_dir=out, take=2)
    assert (second.rendered_count, second.reused_count) == (0, 2)
    other = render(
        segments(),
        profile=profile(name="different-label"),
        output_dir=tmp_path / "voice2",
        take=2,
    )
    assert (other.rendered_count, other.reused_count) == (0, 2)
    assert len(mock_engine.calls) == 2


def test_only_changed_role_rerenders(tmp_path, mock_engine):
    out = tmp_path / "voice1"
    render(segments(), profile=profile(), output_dir=out)
    changed = render(
        segments(), profile=profile(guest="different-guest"), output_dir=out
    )
    assert (changed.rendered_count, changed.reused_count) == (1, 2)
    assert len(mock_engine.calls) == 4


def test_interruption_is_resumable(tmp_path, mock_engine):
    out = tmp_path / "render"
    mock_engine.fail_once_for = "s2"
    with pytest.raises(RuntimeError, match="injected interruption"):
        render(segments(), profile=profile(), output_dir=out)
    interrupted = read_manifest(out)
    assert [item["status"] for item in interrupted["segments"]] == [
        "rendered",
        "failed",
        "pending",
    ]
    assert not interrupted["complete"]
    resumed = render(segments(), profile=profile(), output_dir=out)
    assert (resumed.rendered_count, resumed.reused_count) == (2, 1)
    assert read_manifest(out)["complete"]
    assert len(mock_engine.calls) == 4
    assert not (out / ".voicerender.lock").exists()


def test_corrupt_cache_is_not_reused(tmp_path, mock_engine):
    a = tmp_path / "a"
    render(segments(), profile=profile(), output_dir=a, take=1)
    cache = tmp_path / ".voicerender-cache"
    key = read_manifest(a)["segments"][0]["synthesis_key"]
    (cache / f"{key}.wav").write_bytes(b"bad")
    second = render(segments(), profile=profile(), output_dir=tmp_path / "b", take=1)
    assert second.rendered_count == 1 and second.reused_count == 0


def test_legacy_render_trims_tail_clamps_timings_and_invalidates_old_cache(
    tmp_path, mock_engine
):
    from voicerender.render import _identity

    mock_engine.audio = np.concatenate(
        (np.full(16_000, 0.1, dtype=np.float32), np.zeros(6_400, dtype=np.float32))
    )
    mock_engine.word_timings = (
        {
            "text": "First.",
            "char_start": 0,
            "char_end": 6,
            "start_sample": 100,
            "end_sample": 17_000,
        },
    )
    cache = tmp_path / "legacy-cache"
    output = tmp_path / "legacy-boundary"
    speech = segments()[:1]
    legacy_profile = profile()

    result = render(
        speech,
        profile=legacy_profile,
        output_dir=output,
        cache_dir=cache,
    )
    assert result.rendered_count == 1
    entry = read_manifest(output)["segments"][0]
    assert entry["wav"]["frames"] == 16_000
    assert entry["audio_boundary"]["removed_frames"] == 6_400
    assert entry["timings"][0]["start_sample"] == 100
    assert entry["timings"][0]["end_sample"] == 16_000

    identity = _identity(legacy_profile.roles["narrator"], engine_version="1.0")
    assert identity["algorithm"] == "voicerender.batch.v2"
    assert identity["audio_boundary_policy"] == "voicerender.trailing-silence.v1"
    cache_meta = next(cache.glob("*.json"))
    cache_data = json.loads(cache_meta.read_text(encoding="utf-8"))
    assert cache_data["schema"] == "voicerender.batch-cache.v2"
    assert cache_data["audio_boundary_policy"] == "voicerender.trailing-silence.v1"

    reused = render(
        speech,
        profile=legacy_profile,
        output_dir=tmp_path / "legacy-cache-reuse",
        cache_dir=cache,
    )
    assert reused.reused_count == 1
    assert (
        read_manifest(tmp_path / "legacy-cache-reuse")["segments"][0]["audio_boundary"][
            "output_frames"
        ]
        == 16_000
    )
    assert len(mock_engine.calls) == 1

    cache_data["schema"] = "voicerender.batch-cache.v1"
    cache_meta.write_text(json.dumps(cache_data), encoding="utf-8")
    regenerated = render(
        speech,
        profile=legacy_profile,
        output_dir=tmp_path / "legacy-old-cache",
        cache_dir=cache,
    )
    assert regenerated.rendered_count == 1
    assert len(mock_engine.calls) == 2


def test_force_disables_cache(tmp_path, mock_engine):
    out = tmp_path / "o"
    render(segments(), profile=profile(), output_dir=out, take=1)
    result = render(segments(), profile=profile(), output_dir=out, take=1, force=True)
    assert result.rendered_count == 1 and result.reused_count == 0


def test_reference_audio_affects_key_not_path(tmp_path, mock_engine):
    reference1 = tmp_path / "ref1.wav"
    reference2 = tmp_path / "ref2.wav"
    reference1.write_bytes(b"speaker A")
    reference2.write_bytes(b"speaker A")

    def make(reference):
        return VoiceProfile(
            name="clone",
            roles={
                "narrator": VoiceBinding(
                    engine="testengine",
                    target_id="clone-model",
                    reference_audio=reference,
                )
            },
        )

    speech = [segments()[0]]
    first = render(speech, profile=make(reference1), output_dir=tmp_path / "a")
    second = render(speech, profile=make(reference2), output_dir=tmp_path / "b")
    assert first.rendered_count == 1 and second.reused_count == 1
    reference2.write_bytes(b"speaker B")
    third = render(speech, profile=make(reference2), output_dir=tmp_path / "c")
    assert third.rendered_count == 1
    assert len(mock_engine.calls) == 2


def test_selection_validation(tmp_path, mock_engine):
    with pytest.raises(PlanError, match="Use take or select"):
        render(
            segments(),
            profile=profile(),
            output_dir=tmp_path / "x",
            take=2,
            select=["s2"],
        )
    result = render(
        segments(), profile=profile(), output_dir=tmp_path / "y", select=["chapter:s3"]
    )
    assert result.selected_count == 1 and mock_engine.calls[-1][0] == "s3"
    with pytest.raises(PlanError, match="Ambiguous"):
        render(
            [segments()[0], SpeechSegment(id="s1", scope_id="other", text="X.")],
            profile=profile(),
            output_dir=tmp_path / "z",
            select=["s1"],
        )


def test_legacy_voice_profile_load(tmp_path):
    path = tmp_path / "p.toml"
    path.write_text(
        'schema = "voicerender.voice-profile.v1"\nname = "p"\n[roles.narrator]\nengine = "piper"\ntarget_id = "amy"\n'
    )
    loaded = VoiceProfile.load(path)
    assert loaded.roles["narrator"].target_id == "amy"

    path.write_text(
        'schema = "voicerender.voice-profile.v1"\nname = "p"\n[roles.narrator]\nengine = "piper"\ntarget_id = "amy"\nreference_audio = "reference.wav"\n'
    )
    assert VoiceProfile.load(path).roles["narrator"].reference_audio == (
        tmp_path / "reference.wav"
    )


def test_plan_load_and_legacy_render(tmp_path, mock_engine):
    plan_path = tmp_path / "chapter.toml"
    plan_path.write_text(
        'schema = "utterplan.voiceplan.v1"\nsource_plan_id = "plan123"\n[[segments]]\nid = "1"\nscope_id = "c"\nrole = "narrator"\ntext = "Hello"\n'
    )
    plan = VoicePlan.load(plan_path)
    out = tmp_path / "o"
    events = []
    render_plan(plan, profile=profile(), output_dir=out, on_event=events.append)
    assert read_manifest(out)["source_plan_id"] == "plan123"
    from voicerender.events import RenderEventKind

    assert events[0].kind is RenderEventKind.RENDER_STARTED
    assert events[-1].kind is RenderEventKind.RENDER_COMPLETED
    assert any(event.kind is RenderEventKind.SEGMENT_RENDERED for event in events)
    text = plan_path.read_text() + 'engine_options = "unsupported"\n'
    plan_path.write_text(text)
    with pytest.raises(PlanError, match="Unsupported speech metadata"):
        VoicePlan.load(plan_path)


def test_bad_output_path_segment_id_never_escapes(tmp_path, mock_engine):
    malicious = SpeechSegment(id="../../../escape", text="Safe", role="narrator")
    out = tmp_path / "out"
    render([malicious], profile=profile(), output_dir=out)
    item = read_manifest(out)["segments"][0]
    assert ".." not in Path(item["path"]).parts
    assert (out / item["path"]).is_file()


def test_missing_role_rejected_before_manifest(tmp_path, mock_engine):
    out = tmp_path / "o"
    with pytest.raises(ProfileError, match="No voice binding"):
        render(
            segments(),
            profile=VoiceProfile(
                name="single", roles={"narrator": profile().roles["narrator"]}
            ),
            output_dir=out,
        )
    assert not (out / "manifest.json").exists()


def test_legacy_render_progress_counts_rendered_then_reused_items(
    tmp_path: Path, mock_engine: FakeEngine
) -> None:
    from voicerender.events import RenderEventKind

    output_dir = tmp_path / "legacy-progress"
    events = []
    result = render(
        segments(), profile=profile(), output_dir=output_dir, on_event=events.append
    )

    assert result.rendered_count == 3
    rendered = [
        event for event in events if event.kind is RenderEventKind.SEGMENT_RENDERED
    ]
    assert [event.completed_count for event in rendered] == [1, 2, 3]
    assert rendered[0].eta_seconds is not None
    assert events[0].kind is RenderEventKind.RENDER_STARTED
    assert events[-1].kind is RenderEventKind.RENDER_COMPLETED

    reused_events = []
    reused = render(
        segments(),
        profile=profile(),
        output_dir=output_dir,
        on_event=reused_events.append,
    )
    assert (reused.rendered_count, reused.reused_count) == (0, 3)
    assert (
        sum(event.kind is RenderEventKind.SEGMENT_REUSED for event in reused_events)
        == 3
    )
    assert reused_events[-1].progress == 1.0
    assert len(mock_engine.calls) == 3


def test_legacy_render_callback_failure_is_not_recorded_as_synthesis_failure(
    tmp_path: Path, mock_engine: FakeEngine
) -> None:
    from voicerender import RenderEventCallbackError
    from voicerender.events import RenderEventKind

    output_dir = tmp_path / "callback-failure"

    def callback(event) -> None:
        if event.kind is RenderEventKind.SEGMENT_RENDERED:
            raise RuntimeError("consumer failed")

    with pytest.raises(RenderEventCallbackError):
        render(
            segments(),
            profile=profile(),
            output_dir=output_dir,
            fail_fast=False,
            on_event=callback,
        )
    statuses = [item["status"] for item in read_manifest(output_dir)["segments"]]
    assert statuses == ["rendered", "pending", "pending"]
    assert len(mock_engine.calls) == 1
