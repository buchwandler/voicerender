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


@pytest.mark.parametrize(
    ("working_directory", "path"),
    [("root", "."), ("root", "./"), ("child", "..")],
)
def test_document_workspace_and_render_apis_accept_root_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    working_directory: str,
    path: str,
) -> None:
    from voicerender.document import render_document
    from voicerender.errors import DocumentRenderError

    root = _workspace(tmp_path / "story.ssmd-bundle")
    child = root / "child"
    child.mkdir()
    monkeypatch.chdir(root if working_directory == "root" else child)

    index = load_document_workspace(path)
    assert index.root == root.resolve()
    for render_api in (render_document, estimate_render_document):
        with pytest.raises(DocumentRenderError) as caught:
            render_api(path, variant="missing")
        assert caught.value.code == "variant.not_found"


def test_document_workspace_rejects_renderplan_root_and_direct_symlink(
    tmp_path: Path,
) -> None:
    root = _workspace(tmp_path / "story.ssmd-bundle")
    renderplan_root = root / "renderplan"
    renderplan_root.mkdir()
    with pytest.raises(DocumentWorkspaceError, match="canonical .ssmd-bundle"):
        load_document_workspace(renderplan_root)

    link = tmp_path / "link.ssmd-bundle"
    try:
        link.symlink_to(root, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")
    with pytest.raises(DocumentWorkspaceError, match="root must not be a symlink"):
        load_document_workspace(link)


def test_load_document_bundle_root_and_rejects_file_shortcut(
    tmp_path: Path,
) -> None:
    root = _workspace(tmp_path / "story.ssmd-bundle")
    index = load_document_workspace(root)
    assert index.root == root.resolve()
    assert index.document_id == "document-0001"
    with pytest.raises(DocumentWorkspaceError, match="canonical .ssmd-bundle"):
        load_document_workspace(index.document_path)


def test_rejects_wrong_format_and_traversal(tmp_path: Path) -> None:
    root = _workspace(tmp_path / "story.ssmd-bundle")
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["document"]["path"] = "../outside.md"
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(DocumentWorkspaceError):
        load_document_workspace(root)


def test_rejects_zip_input(tmp_path: Path) -> None:
    with pytest.raises(DocumentWorkspaceError):
        load_document_workspace(tmp_path / "story.ssmd-bundle.zip")


def test_document_estimate_resolves_exact_variant_without_creating_outputs(
    tmp_path: Path,
) -> None:
    root = _workspace(tmp_path / "story.ssmd-bundle")
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

    root = _workspace(tmp_path / "story.ssmd-bundle")
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

    root = _workspace(tmp_path / "story.ssmd-bundle")
    with pytest.raises(DocumentRenderError) as caught:
        _resolve_document_request(root, variant="not-created--000000000000")
    assert caught.value.code == "variant.not_found"


def test_document_render_uses_preflight_and_existing_renderplan_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from types import SimpleNamespace

    import voicerender.document as document_module

    root = _workspace(tmp_path / "story.ssmd-bundle")
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

    from voicerender.cli import main

    exit_code = main(
        [
            "bundle",
            "render",
            str(root),
            "--variant",
            variant,
            "--speech",
            "1",
            "--json",
            "--no-progress",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["format"] == "voicerender-bundle-result"
    assert payload["kind"] == "document"
    assert payload["result"]["status"] == "partial"


def test_document_preflight_reject_stops_before_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    import voicerender.document as document_module

    root = _workspace(tmp_path / "story.ssmd-bundle")
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

    root = _workspace(tmp_path / "story.ssmd-bundle")
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


def test_document_report_preserves_error_diagnostic_summary_and_sidecar_path(
    tmp_path: Path,
) -> None:
    import voicerender.document as document_module

    root = _workspace(tmp_path / "diagnostics.ssmd-bundle")
    profile_path = (
        Path(__file__).parents[1] / "examples/voices/narration.renderprofile.toml"
    )
    profile = codec.load_profile(profile_path)
    fingerprint = (profile.profile_id or profile.computed_id()).removeprefix("sha256:")[
        :12
    ]
    variant = f"narration--{fingerprint}"
    variant_dir = root / "renderplan" / variant
    variant_dir.mkdir(parents=True)
    shutil.copy(profile_path, variant_dir / "renderprofile.toml")
    shutil.copy(
        Path(__file__).parents[1] / "examples/plans/chapter.renderplan.toml",
        variant_dir / "document.renderplan.toml",
    )
    request = document_module._resolve_document_request(root, variant=variant)
    diagnostics_path = request.output_dir / "render-diagnostics.json"
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text("{}", encoding="utf-8")
    result = document_module.DocumentRenderResult(
        request=request,
        report_path=request.output_variant_dir / "voicerender-report.toml",
        preflight_report=None,
        recheck_report=None,
        manifest=None,
        status="failed",
        planned_speech=1,
        selected_speech=1,
        available_speech=0,
        rendered=0,
        reused=0,
        manifest_complete=False,
        error_code="synthesis.backend_error",
        error_message="backend failed",
        render_diagnostics=diagnostics_path,
        error_type="EngineBackendError",
        error_stage="inference",
        error_root_cause_type="RuntimeError",
        error_root_cause_message="provider execution failed",
    )

    document_module._write_report(result)

    payload = result.to_dict()
    report = result.report_path.read_text(encoding="utf-8")
    assert payload["render_diagnostics"] == str(diagnostics_path)
    assert payload["error_type"] == "EngineBackendError"
    assert payload["error_stage"] == "inference"
    assert payload["error_root_cause_type"] == "RuntimeError"
    assert payload["error_root_cause_message"] == "provider execution failed"
    assert f'render_diagnostics = "{diagnostics_path}"' in report
    assert 'error_stage = "inference"' in report
