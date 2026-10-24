from __future__ import annotations

import json
from collections.abc import Mapping

import pytest

from voicerender.engines.soprosynth import _target_from_bundle
from voicerender.models_api import _from_target, model_artifact_to_dict


def test_released_discovered_bundle_metadata_crosses_model_json_boundary() -> None:
    soprosynth = pytest.importorskip("soprosynth")
    bundle = soprosynth.DiscoveredBundle(
        id="model-v1",
        ref="sopro:model-v1",
        display_name="Sopro model v1",
        languages=("en",),
        metadata={
            "catalog": {
                "profile": {
                    "precision": "uint8",
                    "providers": ("CPUExecutionProvider",),
                }
            }
        },
    )

    target = _target_from_bundle(bundle)
    artifact = _from_target(target)
    payload = model_artifact_to_dict(artifact)
    encoded = json.dumps(payload, allow_nan=False)

    assert isinstance(target.metadata["catalog"], Mapping)
    assert payload["engine"] == "sopro"
    assert payload["target_id"] == "model-v1"
    assert payload["metadata"]["catalog"]["profile"] == {
        "precision": "uint8",
        "providers": ["CPUExecutionProvider"],
    }
    assert isinstance(payload["metadata"]["catalog"], dict)
    assert isinstance(payload["metadata"]["catalog"]["profile"]["providers"], list)
    assert json.loads(encoded) == payload
