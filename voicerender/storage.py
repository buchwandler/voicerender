"""Canonical hashes, atomic artifact persistence and verified WAV operations."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .errors import ArtifactError


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(
                value,
                stream,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def atomic_wav(path: Path, audio: Any, sample_rate: int) -> dict[str, Any]:
    if (
        isinstance(sample_rate, bool)
        or not isinstance(sample_rate, int)
        or sample_rate <= 0
    ):
        raise ArtifactError("Engine returned invalid sample rate")
    signal = np.asarray(audio, dtype=np.float32)
    if signal.ndim != 1 or not signal.size or not np.all(np.isfinite(signal)):
        raise ArtifactError("Engine must return nonempty, finite, mono audio")
    peak = float(np.max(np.abs(signal)))
    if peak > float(np.float32(1.00001)):
        raise ArtifactError(
            f"Engine returned audio outside [-1, 1] (peak={peak:.6f})"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".tmp-", suffix=".wav", dir=path.parent)
    os.close(fd)
    try:
        sf.write(name, signal, sample_rate, format="WAV", subtype="PCM_16")
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)
    return wav_info(path)


def wav_info(path: Path) -> dict[str, Any]:
    try:
        with sf.SoundFile(path) as wav:
            if (
                wav.format != "WAV"
                or wav.channels != 1
                or wav.frames <= 0
                or wav.samplerate <= 0
            ):
                raise ArtifactError(f"Invalid mono WAV artifact: {path}")
            return {
                "sha256": file_hash(path),
                "sample_rate": wav.samplerate,
                "channels": wav.channels,
                "frames": wav.frames,
                "duration_seconds": wav.frames / wav.samplerate,
            }
    except (RuntimeError, OSError) as exc:
        raise ArtifactError(f"Corrupt or unavailable WAV {path}: {exc}") from exc


def verified_wav(path: Path, expected: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        actual = wav_info(path)
    except ArtifactError:
        return False
    return all(
        actual.get(key) == expected.get(key)
        for key in ("sha256", "frames", "sample_rate", "channels")
    )
