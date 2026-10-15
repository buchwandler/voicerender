from __future__ import annotations

import numpy as np
import pytest

from voicerender.errors import ArtifactError
from voicerender.storage import atomic_wav


def test_atomic_wav_remains_strict_for_out_of_range_audio(tmp_path) -> None:
    path = tmp_path / "out-of-range.wav"

    with pytest.raises(ArtifactError, match=r"peak=1\.010000"):
        atomic_wav(path, np.array([0.0, 1.01], dtype=np.float32), 24000)

    assert not path.exists()
