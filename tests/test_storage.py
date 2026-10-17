from __future__ import annotations

import numpy as np
import pytest

from voicerender.errors import ArtifactError
from voicerender.storage import atomic_wav


def test_no_replace_file_install_falls_back_without_os_link(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from voicerender import storage

    source = tmp_path / "temporary.bin"
    source.write_bytes(b"voice asset")
    destination = tmp_path / "asset.bin"
    monkeypatch.setattr(storage.os, "link", None, raising=False)

    storage.install_file_no_replace(source, destination)

    assert destination.read_bytes() == b"voice asset"
    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(b"do not overwrite")
    with pytest.raises(FileExistsError):
        storage.install_file_no_replace(replacement, destination)
    assert destination.read_bytes() == b"voice asset"


def test_atomic_wav_remains_strict_for_out_of_range_audio(tmp_path) -> None:
    path = tmp_path / "out-of-range.wav"

    with pytest.raises(ArtifactError, match=r"peak=1\.010000"):
        atomic_wav(path, np.array([0.0, 1.01], dtype=np.float32), 24000)

    assert not path.exists()
