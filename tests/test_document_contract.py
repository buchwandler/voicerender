import json
import shutil
from pathlib import Path

import pytest
from renderplan import codec

from voicerender.document_contract import (
    DocumentWorkspaceError,
    load_document_workspace,
)
from voicerender.estimates import estimate_render_document


def _workspace(root: Path) -> Path:
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "format": "ssmdconvert.document",
                "schema_version": 1,
                "ssmd_version": "0.9",
                "document": {"id": "document-0001", "path": "document.ssmd.md"},
            }
        ),
        encoding="utf-8",
    )
    (root / "document.ssmd.md").write_text("Hello.", encoding="utf-8")
    return root


def test_load_ssmdoc_workspace_root_and_canonical_document(tmp_path: Path) -> None:
    root = _workspace(tmp_path / "story.ssmdoc")
    index = load_document_workspace(root)
    assert index.root == root.resolve()
    assert index.document_id == "document-0001"
    assert load_document_workspace(index.document_path) == index


def test_rejects_wrong_format_and_traversal(tmp_path: Path) -> None:
    root = _workspace(tmp_path / "story.ssmdoc")
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["document"]["path"] = "../outside.md"
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(DocumentWorkspaceError):
        load_document_workspace(root)


def test_rejects_zip_input(tmp_path: Path) -> None:
    with pytest.raises(DocumentWorkspaceError):
        load_document_workspace(tmp_path / "story.ssmdoc.zip")


def test_document_estimate_resolves_exact_variant_without_creating_outputs(
    tmp_path: Path,
) -> None:
    root = _workspace(tmp_path / "story.ssmdoc")
    profile = Path(__file__).parents[1] / "examples/voices/narration.renderprofile.toml"
    profile_obj = codec.load_profile(profile)
    fingerprint = (profile_obj.profile_id or profile_obj.computed_id()).removeprefix(
        "sha256:"
    )[:12]
    variant = f"narration--{fingerprint}"
    variant_dir = root / "renderplan" / variant
    variant_dir.mkdir(parents=True)
    shutil.copy(profile, variant_dir / "renderprofile.toml")
    shutil.copy(
        Path(__file__).parents[1] / "examples/plans/chapter.renderplan.toml",
        variant_dir / "document.renderplan.toml",
    )
    from voicerender.document import _resolve_document_request

    request = _resolve_document_request(root, variant=variant)
    assert request.renderplan_variant_dir == (root / "renderplan" / variant).resolve()
    assert (
        request.renderplan_path
        == (root / "renderplan" / variant / "document.renderplan.toml").resolve()
    )
    assert request.output_variant_dir == (root / "voicerender" / variant).resolve()
    assert (
        request.output_dir
        == (root / "voicerender" / variant / "document-0001").resolve()
    )
    assert (
        request.cache_dir
        == (root / "voicerender" / variant / ".voicerender-cache").resolve()
    )
    estimate = estimate_render_document(root, variant=variant, speech="1")
    assert estimate.selected_count == 1
    assert not (root / "voicerender").exists()


def test_document_request_preserves_explicit_root_overrides(tmp_path: Path) -> None:
    from voicerender.document import _resolve_document_request

    root = _workspace(tmp_path / "story.ssmdoc")
    profile = Path(__file__).parents[1] / "examples/voices/narration.renderprofile.toml"
    profile_obj = codec.load_profile(profile)
    fingerprint = (profile_obj.profile_id or profile_obj.computed_id()).removeprefix(
        "sha256:"
    )[:12]
    variant = f"narration--{fingerprint}"
    renderplan_root = tmp_path / "custom-renderplan"
    variant_dir = renderplan_root / variant
    variant_dir.mkdir(parents=True)
    shutil.copy(profile, variant_dir / "renderprofile.toml")
    shutil.copy(
        Path(__file__).parents[1] / "examples/plans/chapter.renderplan.toml",
        variant_dir / "document.renderplan.toml",
    )
    output_root = tmp_path / "custom-voicerender"
    cache_dir = output_root / variant / ".custom-cache"

    request = _resolve_document_request(
        root,
        variant=variant,
        renderplan_root=renderplan_root,
        output_root=output_root,
        cache_dir=cache_dir,
    )

    assert request.renderplan_variant_dir == variant_dir.resolve()
    assert (
        request.renderplan_path == (variant_dir / "document.renderplan.toml").resolve()
    )
    assert request.output_variant_dir == (output_root / variant).resolve()
    assert request.output_dir == (output_root / variant / "document-0001").resolve()
    assert request.cache_dir == cache_dir.resolve()


def test_missing_document_variant_raises_public_coded_error(tmp_path: Path) -> None:
    from voicerender import DocumentRenderError
    from voicerender.document import _resolve_document_request

    root = _workspace(tmp_path / "story.ssmdoc")
    with pytest.raises(DocumentRenderError) as caught:
        _resolve_document_request(root, variant="not-created--000000000000")
    assert caught.value.code == "variant.not_found"


def test_document_render_uses_preflight_and_existing_renderplan_executor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    import voicerender.document as document_module

    root = _workspace(tmp_path / "story.ssmdoc")
    profile_path = (
        Path(__file__).parents[1] / "examples/voices/narration.renderprofile.toml"
    )
    profile_obj = codec.load_profile(profile_path)
    fingerprint = (profile_obj.profile_id or profile_obj.computed_id()).removeprefix(
        "sha256:"
    )[:12]
    variant = f"narration--{fingerprint}"
    variant_dir = root / "renderplan" / variant
    variant_dir.mkdir(parents=True)
    shutil.copy(profile_path, variant_dir / "renderprofile.toml")
    shutil.copy(
        Path(__file__).parents[1] / "examples/plans/chapter.renderplan.toml",
        variant_dir / "document.renderplan.toml",
    )
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        document_module,
        "_get_preflight",
        lambda *args, **kwargs: SimpleNamespace(status="ok"),
    )

    def fake_render(plan: object, **kwargs: object) -> object:
        calls.append(kwargs)
        return SimpleNamespace(
            manifest=SimpleNamespace(complete=False),
            manifest_path=Path(kwargs["output_dir"]) / "render-manifest.toml",
            failed=False,
            planned_count=3,
            selected_count=1,
            available_count=0,
            rendered_count=1,
            reused_count=0,
            fresh_preflight=None,
            error_code=None,
            error=None,
            pending_composition=False,
        )

    monkeypatch.setattr(document_module, "render_renderplan", fake_render)
    result = document_module.render_document(
        root,
        variant=variant,
        speech="1",
        force=True,
        reset_stale_output=True,
        emphasis_policy="defer-to-compose",
    )
    assert result.status == "partial"
    assert result.selected_speech == 1
    assert result.report_path.is_file()
    assert calls[0]["speech_numbers"] == frozenset({1})
    assert calls[0]["force"] is True
    assert calls[0]["reset_stale_output"] is True


def test_document_preflight_reject_stops_before_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    import voicerender.document as document_module

    root = _workspace(tmp_path / "story.ssmdoc")
    profile_path = (
        Path(__file__).parents[1] / "examples/voices/narration.renderprofile.toml"
    )
    profile_obj = codec.load_profile(profile_path)
    fingerprint = (profile_obj.profile_id or profile_obj.computed_id()).removeprefix(
        "sha256:"
    )[:12]
    variant = f"narration--{fingerprint}"
    variant_dir = root / "renderplan" / variant
    variant_dir.mkdir(parents=True)
    shutil.copy(profile_path, variant_dir / "renderprofile.toml")
    shutil.copy(
        Path(__file__).parents[1] / "examples/plans/chapter.renderplan.toml",
        variant_dir / "document.renderplan.toml",
    )
    monkeypatch.setattr(
        document_module,
        "_get_preflight",
        lambda *args, **kwargs: SimpleNamespace(status="replan"),
    )
    monkeypatch.setattr(
        document_module,
        "render_renderplan",
        lambda *args, **kwargs: pytest.fail("must not synthesize after replan"),
    )
    result = document_module.render_document(root, variant=variant)
    assert result.error_code == "preflight.replan"
    assert result.replan_required is True
    assert result.preflight_report is not None
    assert result.report_path.is_file()


def test_document_render_reports_reused_complete_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    import voicerender.document as document_module

    root = _workspace(tmp_path / "story.ssmdoc")
    profile_path = (
        Path(__file__).parents[1] / "examples/voices/narration.renderprofile.toml"
    )
    profile_obj = codec.load_profile(profile_path)
    fingerprint = (profile_obj.profile_id or profile_obj.computed_id()).removeprefix(
        "sha256:"
    )[:12]
    variant = f"narration--{fingerprint}"
    variant_dir = root / "renderplan" / variant
    variant_dir.mkdir(parents=True)
    shutil.copy(profile_path, variant_dir / "renderprofile.toml")
    shutil.copy(
        Path(__file__).parents[1] / "examples/plans/chapter.renderplan.toml",
        variant_dir / "document.renderplan.toml",
    )
    monkeypatch.setattr(
        document_module,
        "_get_preflight",
        lambda *args, **kwargs: SimpleNamespace(status="ok"),
    )
    monkeypatch.setattr(
        document_module,
        "render_renderplan",
        lambda plan, **kwargs: SimpleNamespace(
            manifest=SimpleNamespace(complete=True),
            manifest_path=Path(kwargs["output_dir"]) / "render-manifest.toml",
            failed=False,
            planned_count=1,
            selected_count=1,
            available_count=1,
            rendered_count=0,
            reused_count=1,
            fresh_preflight=None,
            error_code=None,
            error=None,
            pending_composition=False,
        ),
    )
    result = document_module.render_document(root, variant=variant)
    assert result.status == "reused"
    assert result.manifest_complete is True
    assert result.reused == 1
