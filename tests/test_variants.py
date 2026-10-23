from __future__ import annotations

import json
import shutil
from pathlib import Path

from renderplan import (
    Producer,
    RenderManifest,
    RenderPlan,
    RenderPlanSet,
    RenderPlanSetChapter,
    RenderProfile,
    RoleProfile,
    SourceSpan,
    SpeechArtifact,
    SpeechItem,
    save_manifest,
    save_plan,
    save_profile,
    save_renderplan_set,
)

from voicerender.variants import list_bundle_variants


def _profile(label: str = "fixture") -> RenderProfile:
    return RenderProfile(
        roles={"narrator": RoleProfile(language="en-US", binding={"renderer": label})},
        default_role="narrator",
    ).with_identity()


def _plan(
    profile: RenderProfile, chapter: int = 1, speech_count: int = 2
) -> RenderPlan:
    profile_id = profile.profile_id or profile.computed_id()
    speech = tuple(
        SpeechItem(
            number=number,
            text=f"chapter {chapter} speech {number}",
            language="en-US",
            role="narrator",
            binding={"renderer": "fixture"},
            source=(
                SourceSpan(
                    locator=f"chapter-{chapter:04d}.md",
                    start=number,
                    end=number + 1,
                ),
            ),
        )
        for number in range(1, speech_count + 1)
    )
    return RenderPlan(
        source_plan_id=f"sha256:{chapter:064d}",
        render_profile_id=profile_id,
        flow=speech,
    ).with_identity()


def _write_book_bundle(
    root: Path,
    *,
    chapters: tuple[tuple[int, int], ...] = ((1, 2),),
    label: str = "fixture",
) -> tuple[Path, str, tuple[RenderPlan, ...]]:
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps({"format": "ssmdconvert.book", "schema_version": 1}),
        encoding="utf-8",
    )
    profile = _profile(label)
    profile_id = profile.profile_id or profile.computed_id()
    fingerprint = profile_id.removeprefix("sha256:")[:12]
    variant = f"narration--{fingerprint}"
    variant_dir = root / "renderplan" / variant
    variant_dir.mkdir(parents=True)
    save_profile(profile, variant_dir / "renderprofile.toml")
    entries: list[RenderPlanSetChapter] = []
    plans: list[RenderPlan] = []
    for chapter, count in chapters:
        chapter_id = f"chapter-{chapter:04d}"
        plan = _plan(profile, chapter, count)
        plan_name = f"{chapter_id}.renderplan.toml"
        save_plan(plan, variant_dir / plan_name)
        plans.append(plan)
        entries.append(
            RenderPlanSetChapter(
                id=chapter_id,
                source_number=chapter,
                source=f"../../chapters/{chapter_id}.md",
                utterplan=f"../../utterplan/{chapter_id}.toml",
                source_plan_id=plan.source_plan_id,
                renderplan=plan_name,
                render_plan_id=plan.plan_id or plan.computed_id(),
            )
        )
    renderplan_set = RenderPlanSet(
        variant="narration",
        profile_fingerprint=fingerprint,
        producer_version="0.1.0",
        manifest="../../book.json",
        utterplan_dir="../../utterplan",
        profile_id=profile_id,
        snapshot="renderprofile.toml",
        source_profile="../../narration.renderprofile.toml",
        chapters=tuple(entries),
    )
    save_renderplan_set(variant_dir / "renderplan-set.toml", renderplan_set)
    return root, variant, tuple(plans)


def _write_document_bundle(
    root: Path, *, label: str = "fixture"
) -> tuple[Path, str, RenderPlan]:
    root.mkdir(parents=True)
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
    (root / "document.ssmd.md").write_text("A test document.", encoding="utf-8")
    profile = _profile(label)
    profile_id = profile.profile_id or profile.computed_id()
    variant = f"narration--{profile_id.removeprefix('sha256:')[:12]}"
    variant_dir = root / "renderplan" / variant
    variant_dir.mkdir(parents=True)
    save_profile(profile, variant_dir / "renderprofile.toml")
    plan = _plan(profile)
    save_plan(plan, variant_dir / "document.renderplan.toml")
    return root, variant, plan


def _write_manifest(
    plan: RenderPlan, path: Path, *, count: int, complete: bool
) -> None:
    artifacts = tuple(
        SpeechArtifact(
            segment_number=item.number,
            segment_id=item.segment_id or item.computed_id(),
            wav=item.wav,
            sha256=f"sha256:{'0' * 64}",
            frames=1,
            sample_rate=24000,
            channels=1,
        )
        for item in plan.speech[:count]
    )
    manifest = RenderManifest(
        render_plan_id=plan.plan_id or plan.computed_id(),
        speech=artifacts,
        complete=complete,
        producer=Producer(name="voicerender", version="test"),
    ).with_identity()
    save_manifest(manifest, path)


def test_empty_and_missing_roots_are_read_only(tmp_path: Path) -> None:
    root = tmp_path / "empty.ssmd-bundle"
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps({"format": "ssmdconvert.book", "schema_version": 1}),
        encoding="utf-8",
    )
    listing = list_bundle_variants(root)
    assert listing.variants == ()
    assert not (root / "renderplan").exists()
    assert not (root / "voicerender").exists()

    missing = tmp_path / "missing.ssmd-bundle"
    missing.mkdir()
    (missing / "manifest.json").write_text(
        json.dumps({"format": "ssmdconvert.book", "schema_version": 1}),
        encoding="utf-8",
    )
    assert list_bundle_variants(missing).variants == ()
    assert not (missing / "renderplan").exists()


def test_valid_book_variant_is_discovered_and_plan_counts_are_reported(
    tmp_path: Path,
) -> None:
    root, variant, plans = _write_book_bundle(
        tmp_path / "book.ssmd-bundle", chapters=((7, 2), (5, 1))
    )
    result = list_bundle_variants(root)
    found = result.variants[0]
    assert result.kind == "book"
    assert found.name == variant
    assert found.plan_status == "valid"
    assert found.render_status == "not_rendered"
    assert found.state == "not_rendered"
    assert found.total_chapters == 2
    assert found.planned_speech == sum(len(plan.speech) for plan in plans) == 3
    assert found.profile_id is not None


def test_valid_document_variant_and_complete_manifest_are_discovered(
    tmp_path: Path,
) -> None:
    root, variant, plan = _write_document_bundle(tmp_path / "doc.ssmd-bundle")
    output = root / "voicerender" / variant / "document-0001"
    _write_manifest(plan, output / "render-manifest.toml", count=2, complete=True)

    found = list_bundle_variants(root).variants[0]
    assert found.plan_status == "valid"
    assert found.render_status == "complete"
    assert found.state == "complete"
    assert found.completed_units == found.total_units == 1
    assert found.unit_kind == "documents"
    assert found.planned_speech == found.available_speech == 2


def test_book_manifest_progress_can_be_partial_or_complete(tmp_path: Path) -> None:
    root, variant, plans = _write_book_bundle(
        tmp_path / "book.ssmd-bundle", chapters=((1, 2), (2, 1))
    )
    output_variant = root / "voicerender" / variant
    _write_manifest(
        plans[0],
        output_variant / "chapter-0001" / "render-manifest.toml",
        count=2,
        complete=True,
    )
    partial = list_bundle_variants(root).variants[0]
    assert partial.render_status == "partial"
    assert partial.completed_chapters == 1
    assert partial.total_chapters == 2
    assert partial.planned_speech == 3
    assert partial.available_speech == 2

    _write_manifest(
        plans[1],
        output_variant / "chapter-0002" / "render-manifest.toml",
        count=1,
        complete=True,
    )
    complete = list_bundle_variants(root).variants[0]
    assert complete.render_status == "complete"
    assert complete.completed_chapters == complete.total_chapters == 2
    assert complete.available_speech == 3


def test_document_failed_report_and_pending_composition_are_preserved(
    tmp_path: Path,
) -> None:
    root, variant, _plan_obj = _write_document_bundle(tmp_path / "doc.ssmd-bundle")
    report = root / "voicerender" / variant / "voicerender-report.toml"
    report.parent.mkdir(parents=True)
    report.write_text(
        'format = "voicerender-bundle-document-report"\n'
        'version = 1\nstatus = "failed"\nplanned_speech = 2\n'
        "available_speech = 1\npending_composition = true\n"
        'error_code = "render.failed"\nerror_message = "backend unavailable"\n'
        'error_type = "EngineBackendError"\nerror_stage = "inference"\n'
        'error_root_cause_type = "RuntimeError"\n'
        'error_root_cause_message = "provider failed"\n'
        'render_diagnostics = "document-0001/render-diagnostics.json"\n',
        encoding="utf-8",
    )
    found = list_bundle_variants(root).variants[0]
    assert found.render_status == found.state == "failed"
    assert found.error_code == "render.failed"
    assert found.error_message == "backend unavailable"
    assert found.pending_composition is True
    assert found.report_path == report
    assert found.available_speech == 1
    assert found.error_type == "EngineBackendError"
    assert found.error_stage == "inference"
    assert found.error_root_cause_type == "RuntimeError"
    assert found.error_root_cause_message == "provider failed"
    assert found.render_diagnostics == (
        report.parent / "document-0001" / "render-diagnostics.json"
    )


def test_book_failed_report_is_surfaced_with_existing_manifest(tmp_path: Path) -> None:
    root, variant, plans = _write_book_bundle(tmp_path / "book.ssmd-bundle")
    output_variant = root / "voicerender" / variant
    _write_manifest(
        plans[0],
        output_variant / "chapter-0001" / "render-manifest.toml",
        count=1,
        complete=False,
    )
    report = output_variant / "voicerender-report.toml"
    report.write_text(
        'format = "voicerender-bundle-book-report"\nversion = 1\n'
        "failed = 1\npartial_chapters = 0\nskipped = 0\n"
        'pending_composition = false\n[[item]]\nstatus = "failed"\n'
        'error_code = "preflight.reject"\nerror_message = "capacity"\n'
        'error_type = "EngineBackendError"\nerror_stage = "inference"\n'
        'error_root_cause_type = "RuntimeError"\n'
        'error_root_cause_message = "provider failed"\n'
        'render_diagnostics = "chapter-0001/render-diagnostics.json"\n',
        encoding="utf-8",
    )
    found = list_bundle_variants(root).variants[0]
    assert found.render_status == "failed"
    assert found.error_code == "preflight.reject"
    assert found.error_message == "capacity"
    assert found.available_speech == 1
    assert found.error_type == "EngineBackendError"
    assert found.error_stage == "inference"
    assert found.error_root_cause_type == "RuntimeError"
    assert found.error_root_cause_message == "provider failed"
    assert found.render_diagnostics == (
        report.parent / "chapter-0001" / "render-diagnostics.json"
    )


def test_boolean_report_version_is_malformed(tmp_path: Path) -> None:
    root, variant, _plan = _write_document_bundle(tmp_path / "doc.ssmd-bundle")
    report = root / "voicerender" / variant / "voicerender-report.toml"
    report.parent.mkdir(parents=True)
    report.write_text(
        'format = "voicerender-bundle-document-report"\n'
        'version = true\nstatus = "complete"\navailable_speech = 1\n',
        encoding="utf-8",
    )

    found = list_bundle_variants(root).variants[0]
    assert found.plan_status == "valid"
    assert found.render_status == "unknown"
    assert "unsupported" in (found.output_error or "")


def test_book_report_with_non_table_item_is_malformed(tmp_path: Path) -> None:
    root, variant, _plans = _write_book_bundle(tmp_path / "book.ssmd-bundle")
    report = root / "voicerender" / variant / "voicerender-report.toml"
    report.parent.mkdir(parents=True)
    report.write_text(
        'format = "voicerender-bundle-book-report"\nversion = 1\n'
        "failed = 1\npartial_chapters = 0\nskipped = 0\n"
        'pending_composition = false\nerror_code = "render.failed"\n'
        'error_message = "failed"\nitem = [1]\n',
        encoding="utf-8",
    )

    found = list_bundle_variants(root).variants[0]
    assert found.plan_status == "valid"
    assert found.render_status == "unknown"
    assert "item must be a table" in (found.output_error or "")


def test_invalid_and_fingerprint_mismatch_candidates_do_not_hide_valid_one(
    tmp_path: Path,
) -> None:
    root, variant, _plans = _write_book_bundle(tmp_path / "book.ssmd-bundle")
    renderplan_root = root / "renderplan"
    (renderplan_root / "aaa-broken").mkdir()
    shutil.copytree(renderplan_root / variant, renderplan_root / "zzz-wrong-name")
    result = list_bundle_variants(root)
    assert [item.name for item in result.variants] == [
        "aaa-broken",
        variant,
        "zzz-wrong-name",
    ]
    assert [item.plan_status for item in result.variants] == [
        "invalid",
        "valid",
        "invalid",
    ]
    assert "renderplan-set.toml" in (result.variants[0].validation_error or "")
    assert "does not match" in (result.variants[2].validation_error or "")


def test_malformed_report_is_conservative_and_does_not_abort_siblings(
    tmp_path: Path,
) -> None:
    root, variant, _existing_plan = _write_document_bundle(tmp_path / "doc.ssmd-bundle")
    variant_root = root / "renderplan"
    second_profile = _profile("sibling")
    second_id = second_profile.profile_id or second_profile.computed_id()
    second_variant = f"narration--{second_id.removeprefix('sha256:')[:12]}"
    second_dir = variant_root / second_variant
    second_dir.mkdir()
    save_profile(second_profile, second_dir / "renderprofile.toml")
    save_plan(_plan(second_profile), second_dir / "document.renderplan.toml")
    output = root / "voicerender" / variant / "voicerender-report.toml"
    output.parent.mkdir(parents=True)
    output.write_text("not = [valid TOML", encoding="utf-8")
    found = list_bundle_variants(root).variants
    assert len(found) == 2
    malformed = next(item for item in found if item.name == variant)
    assert malformed.plan_status == "valid"
    assert malformed.render_status == "unknown"
    assert malformed.output_error is not None
    assert (
        next(item for item in found if item.name == second_variant).plan_status
        == "valid"
    )


def test_custom_roots_and_symlink_escape_are_handled_safely(tmp_path: Path) -> None:
    root, variant, _plan = _write_document_bundle(tmp_path / "doc.ssmd-bundle")
    custom_renderplans = tmp_path / "custom-renderplans"
    custom_renderplans.mkdir()
    shutil.copytree(root / "renderplan" / variant, custom_renderplans / variant)
    outside = tmp_path / "outside-variant"
    outside.mkdir()
    (outside / "renderprofile.toml").write_text("external", encoding="utf-8")
    (custom_renderplans / "escape").symlink_to(outside, target_is_directory=True)
    custom_output = tmp_path / "custom-output"
    result = list_bundle_variants(
        root, renderplan_root=custom_renderplans, output_root=custom_output
    )
    assert result.renderplan_root == custom_renderplans.resolve()
    assert result.output_root == custom_output.resolve()
    assert [item.name for item in result.variants] == ["escape", variant]
    escaped = next(item for item in result.variants if item.name == "escape")
    assert escaped.plan_status == "invalid"
    assert "symlink" in (escaped.validation_error or "")
    assert not custom_output.exists()


def test_json_shape_and_read_only_listing(tmp_path: Path) -> None:
    root, _variant, _plan = _write_document_bundle(tmp_path / "doc.ssmd-bundle")
    before = {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
    result = list_bundle_variants(root)
    payload = result.to_dict()
    after = {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
    assert payload["format"] == "voicerender-bundle-variants"
    assert payload["version"] == 1
    assert payload["count"] == 1
    assert before == after
    assert not (root / "voicerender").exists()


def test_malformed_manifest_reports_unknown_for_that_variant(tmp_path: Path) -> None:
    root, variant, _plan = _write_document_bundle(tmp_path / "doc.ssmd-bundle")
    manifest = root / "voicerender" / variant / "document-0001" / "render-manifest.toml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("not a valid manifest", encoding="utf-8")
    found = list_bundle_variants(root).variants[0]
    assert found.plan_status == "valid"
    assert found.render_status == "unknown"
    assert found.output_error is not None


def test_incomplete_document_manifest_reports_partial_progress(tmp_path: Path) -> None:
    root, variant, plan = _write_document_bundle(tmp_path / "doc.ssmd-bundle")
    manifest = root / "voicerender" / variant / "document-0001" / "render-manifest.toml"
    _write_manifest(plan, manifest, count=1, complete=False)
    found = list_bundle_variants(root).variants[0]
    assert found.render_status == "partial"
    assert found.completed_units == 0
    assert found.total_units == 1
    assert found.available_speech == 1


def test_output_variant_symlink_escape_is_not_followed(tmp_path: Path) -> None:
    root, variant, plan = _write_document_bundle(tmp_path / "doc.ssmd-bundle")
    outside = tmp_path / "outside-output"
    _write_manifest(
        plan,
        outside / "document-0001" / "render-manifest.toml",
        count=2,
        complete=True,
    )
    report = outside / "voicerender-report.toml"
    report.write_text(
        'format = "voicerender-bundle-document-report"\n'
        'version = 1\nstatus = "complete"\nplanned_speech = 2\n'
        "available_speech = 2\n",
        encoding="utf-8",
    )
    output_root = root / "voicerender"
    output_root.mkdir()
    (output_root / variant).symlink_to(outside, target_is_directory=True)

    found = list_bundle_variants(root).variants[0]
    assert found.render_status == "unknown"
    assert found.output_error is not None
    assert "symlink" in found.output_error
    assert found.report_path is None
