from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def load_module():
    module_path = (
        Path(__file__).resolve().parent.parent / "scripts" / "serve_fastapi.py"
    )
    spec = importlib.util.spec_from_file_location("serve_fastapi", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_prompt_tokens_splits_dot_prompt() -> None:
    module = load_module()
    assert module.parse_prompt_tokens("chair.table.car") == [
        "chair",
        "table",
        "car",
    ]


def test_parse_prompt_tokens_rejects_empty_prompt() -> None:
    module = load_module()
    with pytest.raises(Exception):
        module.parse_prompt_tokens("...")


def test_parse_intrinsics_accepts_3x3_json() -> None:
    module = load_module()
    default_k = np.eye(3, dtype=np.float32)
    intrinsics = module.parse_intrinsics(
        "[[1,0,2],[0,3,4],[0,0,1]]", default_k
    )
    np.testing.assert_allclose(
        intrinsics,
        np.array([[1, 0, 2], [0, 3, 4], [0, 0, 1]], dtype=np.float32),
    )


def test_parse_intrinsics_falls_back_to_default() -> None:
    module = load_module()
    default_k = np.eye(3, dtype=np.float32)
    intrinsics = module.parse_intrinsics(None, default_k)
    np.testing.assert_allclose(intrinsics, default_k)


def test_build_cuboids_formats_json_ready_result() -> None:
    module = load_module()
    cuboids = module.build_cuboids(
        boxes=np.array([[10, 20, 30, 40]], dtype=np.float32),
        boxes3d=np.array(
            [[1, 2, 3, 4, 5, 6, 0.1, 0.2, 0.3, 0.4]],
            dtype=np.float32,
        ),
        scores=np.array([0.9], dtype=np.float32),
        class_ids=np.array([1], dtype=np.int64),
        class_id_mapping={0: "chair", 1: "table"},
    )
    assert cuboids == [
        {
            "score": pytest.approx(0.9),
            "class_id": 1,
            "label": "table",
            "bbox_2d_xyxy": [10.0, 20.0, 30.0, 40.0],
            "center_cam": [1.0, 2.0, 3.0],
            "dimensions_wlh": [4.0, 5.0, 6.0],
            "dimensions_whl": [4.0, 6.0, 5.0],
            "rotation_quat": [0.1, 0.2, 0.3, 0.4],
            "depth": 3.0,
        }
    ]
