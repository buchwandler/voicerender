from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from renderplan import RenderProfile, RoleProfile

from voicerender import cli
from voicerender.lexicons import LexiconCatalogEntry
from voicerender.renderprofile import load_render_profile


def _entry() -> LexiconCatalogEntry:
    return LexiconCatalogEntry(
        selector="lexhint",
        engine="kokoro",
        language="en",
        locale="en-US",
        asset_id="en-us:lexhint",
        data_backend="lexphon",
        default=False,
        installed=True,
        models=("v1.0",),
        model_support="known",
        display_name="LexHint",
        phoneme_encoding="ipa",
        data_version="2026.1",
    )


class _Adapter:
    id = "kokoro"

    def __init__(self) -> None:
        self.discovery_args = None
        self.inspected_path = None
        self.inspection = {
            "path": "/unused.g2lex",
            "sha256": "sha256:" + "a" * 64,
            "logical_sha256": "sha256:" + "b" * 64,
            "metadata": {
                "language": "en-us",
                "phoneme_encoding": "ipa",
                "id": "local-test",
            },
        }

    def capabilities(self):
        return SimpleNamespace(supports_lexicons=True)

    def discover_lexicons(self, **kwargs):
        self.discovery_args = kwargs
        return (_entry(),), SimpleNamespace()

    def inspect_g2lex_file(self, path):
        self.inspected_path = path
        return self.inspection


def test_lexicon_list_and_inspect_json_preserve_catalog_fields(
    monkeypatch, capsys
) -> None:
    adapter = _Adapter()
    monkeypatch.setattr(cli, "get_adapter", lambda engine: adapter)

    assert (
        cli.main(
            [
                "lexicons",
                "list",
                "--engine",
                "kokoro",
                "--lang",
                "en-US",
                "--target",
                "v1.0",
                "--json",
            ]
        )
        == 0
    )
    rows = json.loads(capsys.readouterr().out)
    assert rows == [_entry().to_dict()]
    assert adapter.discovery_args == {
        "language": "en-US",
        "model": "v1.0",
        "offline": False,
        "refresh": False,
        "preference": "auto",
    }

    assert (
        cli.main(["lexicons", "inspect", "lexhint", "--lang", "en-US", "--json"]) == 0
    )
    assert json.loads(capsys.readouterr().out) == _entry().to_dict()


def test_inspect_file_uses_engine_bridge_and_emits_stable_json(
    monkeypatch, capsys, tmp_path
):
    adapter = _Adapter()
    monkeypatch.setattr(cli, "get_adapter", lambda engine: adapter)
    source = tmp_path / "local.g2lex"
    source.write_bytes(b"fixture")

    assert cli.main(["lexicons", "inspect-file", str(source), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True
    assert payload["metadata"] == adapter.inspection["metadata"]
    assert adapter.inspected_path == str(source)


def test_profile_create_pins_and_persists_ordered_pronunciation_sources(
    monkeypatch, capsys, tmp_path
):
    adapter = _Adapter()
    monkeypatch.setattr(cli, "get_adapter", lambda engine: adapter)
    source = tmp_path / "custom.g2lex"
    source.write_bytes(b"fixture")
    output = tmp_path / "profile"
    captured = {}

    def fake_create_render_profile(**kwargs):
        captured.update(kwargs)
        role = RoleProfile(
            language=kwargs["language"],
            binding={
                "renderer": kwargs["engine"],
                "target_id": kwargs["target_id"],
                "options": dict(kwargs["options"] or {}),
            },
        )
        return RenderProfile(
            roles={kwargs["role"]: role}, default_role=kwargs["role"]
        ).with_identity()

    monkeypatch.setattr(cli, "create_render_profile", fake_create_render_profile)
    assert (
        cli.main(
            [
                "profile",
                "create",
                "--engine",
                "kokoro",
                "--target",
                "v1.0",
                "--language",
                "en-US",
                "--voice",
                "af_heart",
                "--lexicon",
                "lexhint",
                "--lexicon",
                "gold",
                "--g2lex",
                str(source),
                "--lexicon-data-policy",
                "installed-only",
                "--output",
                str(output),
                "--json",
            ]
        )
        == 0
    )
    capsys.readouterr()
    options = captured["options"]
    assert options["lexicons"] == ["lexhint", "gold"]
    assert options["lexicon_data_policy"] == "installed-only"
    assert options["g2lex_files"] == [
        {
            "path": str(source.resolve()),
            "sha256": "sha256:" + "a" * 64,
            "logical_sha256": "sha256:" + "b" * 64,
            "language": "en-us",
            "phoneme_encoding": "ipa",
        }
    ]
    saved = load_render_profile(output.with_name("profile.renderprofile.toml"))
    saved_options = saved.roles["narrator"].binding["options"]
    assert saved_options == options
    assert "None" not in (output.with_name("profile.renderprofile.toml").read_text())


def test_profile_pronunciation_flags_reject_duplicates_and_generic_conflicts(
    monkeypatch, capsys, tmp_path
):
    monkeypatch.setattr(
        cli,
        "create_render_profile",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("profile should not be created")
        ),
    )
    base = [
        "profile",
        "create",
        "--engine",
        "kokoro",
        "--target",
        "v1.0",
        "--language",
        "en-US",
        "--voice",
        "af_heart",
        "--output",
        str(tmp_path / "failed"),
        "--json",
    ]
    for extra in (
        ["--lexicon", "gold", "--lexicon", "gold"],
        ["--lexicon", "gold", "--option", 'lexicons=["lexhint"]'],
        ["--lexicon-data-policy", "auto", "--option", 'lexicon_data_policy="auto"'],
    ):
        assert cli.main([*base, *extra]) == 1
        assert json.loads(capsys.readouterr().out)["status"] == "error"

    adapter = _Adapter()
    monkeypatch.setattr(cli, "get_adapter", lambda engine: adapter)
    source = tmp_path / "duplicate.g2lex"
    source.write_bytes(b"fixture")
    assert cli.main([*base, "--g2lex", str(source), "--g2lex", str(source)]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "error"


def test_generic_lexicons_option_remains_backward_compatible(
    monkeypatch, capsys, tmp_path
):
    captured = {}

    def fake_create_render_profile(**kwargs):
        captured.update(kwargs)
        role = RoleProfile(
            language=kwargs["language"],
            binding={"renderer": kwargs["engine"], "target_id": kwargs["target_id"]},
        )
        return RenderProfile(
            roles={kwargs["role"]: role}, default_role=kwargs["role"]
        ).with_identity()

    monkeypatch.setattr(cli, "create_render_profile", fake_create_render_profile)
    assert (
        cli.main(
            [
                "profile",
                "create",
                "--engine",
                "kokoro",
                "--target",
                "v1.0",
                "--language",
                "en-US",
                "--voice",
                "af_heart",
                "--option",
                'lexicons=["gold"]',
                "--output",
                str(tmp_path / "generic"),
                "--json",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert captured["options"]["lexicons"] == ["gold"]


def test_bind_and_role_set_parsers_expose_dedicated_flags(tmp_path: Path) -> None:
    source = str(tmp_path / "profile.renderprofile.toml")
    for command in ("bind", "role"):
        args = ["profile", command]
        if command == "role":
            args.append("set")
        args.extend(
            [
                source,
                "guest",
                "--engine",
                "kokoro",
                "--target",
                "v1.0",
                "--language",
                "en-US",
                "--lexicon",
                "gold",
                "--g2lex",
                "local.g2lex",
                "--lexicon-data-policy",
                "auto",
                "-o",
                str(tmp_path / "out.toml"),
            ]
        )
        parsed = cli.build_parser().parse_args(args)
        assert parsed.lexicon == ["gold"]
        assert parsed.g2lex == [Path("local.g2lex")]
        assert parsed.lexicon_data_policy == "auto"
