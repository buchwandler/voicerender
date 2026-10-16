from pathlib import Path


def test_pep561_marker_exists() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "voicerender" / "py.typed").is_file()
