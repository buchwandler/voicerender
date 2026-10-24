from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

from voicerender.jsonutil import json_value


class _Mode(Enum):
    READY = "ready"


@dataclass(frozen=True)
class _TransportRecord:
    metadata: Any
    mode: _Mode


def test_json_value_walks_dataclass_fields_and_nested_read_only_mappings() -> None:
    source = _TransportRecord(
        metadata=MappingProxyType(
            {
                "catalog": MappingProxyType(
                    {
                        "profile": MappingProxyType(
                            {
                                "providers": ("CPUExecutionProvider",),
                                "cache": Path("/tmp/model-cache"),
                            }
                        )
                    }
                )
            }
        ),
        mode=_Mode.READY,
    )

    payload = json_value(source, label="model metadata")

    assert payload == {
        "metadata": {
            "catalog": {
                "profile": {
                    "providers": ["CPUExecutionProvider"],
                    "cache": "/tmp/model-cache",
                }
            }
        },
        "mode": "ready",
    }
    assert payload["metadata"] is not source.metadata
    assert json.loads(json.dumps(payload)) == payload


def test_json_value_preserves_numpy_tolist_transport_support() -> None:
    assert json_value(
        {"values": np.asarray([1, 2], dtype=np.int64), "scalar": np.float32(1.5)},
        label="capacity metadata",
    ) == {"values": [1, 2], "scalar": pytest.approx(1.5)}


def test_json_value_reports_nested_path_for_unsupported_values() -> None:
    with pytest.raises(
        TypeError,
        match=r"model metadata at catalog\.profile\.session contains unsupported JSON value object",
    ):
        json_value(
            MappingProxyType({"catalog": {"profile": {"session": object()}}}),
            label="model metadata",
        )


def test_json_value_rejects_non_finite_floats_with_nested_path() -> None:
    with pytest.raises(
        ValueError,
        match=r"capacity metadata at measure\.details\.value contains non-finite float",
    ):
        json_value(
            {"measure": {"details": {"value": float("nan")}}},
            label="capacity metadata",
        )
