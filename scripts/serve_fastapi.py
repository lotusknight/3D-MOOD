"""FastAPI service for 3D-MOOD with sequential B=1 inference."""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import sys
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

CKPT_DEFAULT = (
    "https://huggingface.co/RoyYang0714/3D-MOOD/resolve/main/"
    "gdino3d_swin-b_120e_omni3d_834c97.pt"
)


@dataclass
class RuntimeSettings:
    """Runtime settings for service startup defaults."""

    ckpt: str = CKPT_DEFAULT
    score_threshold: float = 0.1
    max_per_image: int = 100
    nms: bool = True
    class_agnostic_nms: bool = True
    nms_iou_threshold: float = 0.5
    resize_h: int = 800
    resize_w: int = 1333
    host: str = "0.0.0.0"
    port: int = 8000


def _env_bool(key: str, default: bool) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_settings_from_env() -> RuntimeSettings:
    """Build runtime settings from environment variables."""
    return RuntimeSettings(
        ckpt=os.getenv("MOOD_CKPT", CKPT_DEFAULT),
        score_threshold=float(os.getenv("MOOD_SCORE_THRESHOLD", "0.1")),
        max_per_image=int(os.getenv("MOOD_MAX_PER_IMAGE", "100")),
        nms=_env_bool("MOOD_NMS", True),
        class_agnostic_nms=_env_bool("MOOD_CLASS_AGNOSTIC_NMS", True),
        nms_iou_threshold=float(os.getenv("MOOD_NMS_IOU_THRESHOLD", "0.5")),
        resize_h=int(os.getenv("MOOD_RESIZE_H", "800")),
        resize_w=int(os.getenv("MOOD_RESIZE_W", "1333")),
        host=os.getenv("MOOD_HOST", "0.0.0.0"),
        port=int(os.getenv("MOOD_PORT", "8000")),
    )


SETTINGS = load_settings_from_env()


@lru_cache(maxsize=1)
def _runtime_deps() -> dict[str, Any]:
    """Import heavy runtime dependencies lazily.

    This keeps lightweight parser tests importable even when the full GPU
    runtime is not installed in the current Python environment.
    """
    import torch
    from vis4d.common.ckpt import load_model_checkpoint
    from vis4d.data.const import AxisMode
    from vis4d.data.transforms.base import compose
    from vis4d.data.transforms.normalize import NormalizeImages
    from vis4d.data.transforms.resize import ResizeImages, ResizeIntrinsics
    from vis4d.data.transforms.to_tensor import ToTensor
    from vis4d.op.box.box3d import boxes3d_to_corners
    from vis4d.op.geometry.projection import project_points
    from vis4d.vis.image.functional import imshow_bboxes3d

    from opendet3d.data.transforms.pad import (
        CenterPadImages,
        CenterPadIntrinsics,
    )
    from opendet3d.data.transforms.resize import GenResizeParameters

    from demo import get_3d_mood_swin_base

    return {
        "torch": torch,
        "load_model_checkpoint": load_model_checkpoint,
        "AxisMode": AxisMode,
        "compose": compose,
        "NormalizeImages": NormalizeImages,
        "ResizeImages": ResizeImages,
        "ResizeIntrinsics": ResizeIntrinsics,
        "ToTensor": ToTensor,
        "boxes3d_to_corners": boxes3d_to_corners,
        "project_points": project_points,
        "imshow_bboxes3d": imshow_bboxes3d,
        "CenterPadImages": CenterPadImages,
        "CenterPadIntrinsics": CenterPadIntrinsics,
        "GenResizeParameters": GenResizeParameters,
        "get_3d_mood_swin_base": get_3d_mood_swin_base,
    }


def default_intrinsics(h: int, w: int) -> np.ndarray:
    """Construct default intrinsics from image size."""
    focal = 0.72 * float(max(h, w))
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    return np.array(
        [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


@lru_cache(maxsize=8)
def build_preprocess(resize_h: int, resize_w: int):
    """Cached preprocess pipeline for each resize pair."""
    deps = _runtime_deps()
    resize_shape = (resize_h, resize_w)
    return deps["compose"](
        transforms=[
            deps["GenResizeParameters"](shape=resize_shape),
            deps["ResizeImages"](),
            deps["ResizeIntrinsics"](),
            deps["NormalizeImages"](),
            deps["CenterPadImages"](
                stride=1, shape=resize_shape, update_input_hw=True
            ),
            deps["CenterPadIntrinsics"](),
        ]
    )


def parse_prompt_tokens(prompt: str) -> list[str]:
    """Split dot-separated prompt string."""
    tokens = [token.strip() for token in prompt.split(".") if token.strip()]
    if not tokens:
        raise HTTPException(
            status_code=400,
            detail=(
                "Prompt is empty. Use dot-separated tokens, "
                "for example: chair.table.car"
            ),
        )
    return tokens


def parse_intrinsics(
    raw_json: str | None,
    default_k: np.ndarray,
) -> np.ndarray:
    """Parse optional 3x3 intrinsics matrix from JSON."""
    if raw_json is None or not raw_json.strip():
        return default_k
    try:
        matrix = np.asarray(json.loads(raw_json), dtype=np.float32)
    except Exception as exc:  # pragma: no cover - specific message checked
        raise HTTPException(
            status_code=400,
            detail=f"Invalid intrinsics_json: {exc}",
        ) from exc
    if matrix.shape != (3, 3):
        raise HTTPException(
            status_code=400,
            detail=f"intrinsics_json must be a 3x3 matrix, got {matrix.shape}",
        )
    return matrix


def prepare_one(
    pil_image: Image.Image,
    intrinsics: np.ndarray,
    input_texts: list[str],
    preprocess,
) -> tuple[dict[str, Any], dict[int, str]]:
    """Prepare one image exactly like demo.py and batch_benchmark.py."""
    deps = _runtime_deps()
    pil_image = pil_image.convert("RGB")
    images = np.array(pil_image).astype(np.float32)[None, ...]
    h, w = images.shape[1], images.shape[2]
    data_dict = {
        "images": images,
        "original_images": images,
        "input_hw": (h, w),
        "original_hw": (h, w),
        "intrinsics": intrinsics,
        "original_intrinsics": intrinsics,
    }
    data = preprocess([data_dict])[0]
    data = deps["ToTensor"]()([data])[0]
    class_id_mapping = {i: token for i, token in enumerate(input_texts)}
    return data, class_id_mapping


def _to_jsonable(value: Any) -> Any:
    if value.__class__.__module__.startswith("torch"):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _rounded_list(values: np.ndarray, ndigits: int = 6) -> list[float]:
    return [round(float(v), ndigits) for v in values.tolist()]


def _rounded_point(point: np.ndarray, ndigits: int = 1) -> dict[str, float]:
    return {
        "x": round(float(point[0]), ndigits),
        "y": round(float(point[1]), ndigits),
    }


def _sort_face_corners(face_points: np.ndarray) -> dict[str, dict[str, float]]:
    """Map four projected face points to tl/tr/br/bl."""
    order = np.argsort(face_points[:, 1], kind="stable")
    top = face_points[order[:2]]
    bottom = face_points[order[2:]]
    top = top[np.argsort(top[:, 0], kind="stable")]
    bottom = bottom[np.argsort(bottom[:, 0], kind="stable")]
    return {
        "tl": _rounded_point(top[0]),
        "tr": _rounded_point(top[1]),
        "br": _rounded_point(bottom[1]),
        "bl": _rounded_point(bottom[0]),
    }


def _project_box_faces(
    boxes3d: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project 3D cuboids to 2D faces and their depths."""
    deps = _runtime_deps()
    torch = deps["torch"]
    boxes3d_tensor = torch.as_tensor(boxes3d, dtype=torch.float32)
    intrinsics_tensor = torch.as_tensor(intrinsics, dtype=torch.float32)
    corners_3d = deps["boxes3d_to_corners"](
        boxes3d_tensor,
        deps["AxisMode"].OPENCV,
    )
    corners_2d = deps["project_points"](
        corners_3d.reshape(-1, 3),
        intrinsics_tensor,
    ).reshape(-1, 8, 2)
    face_points = corners_2d.reshape(-1, 2, 4, 2).detach().cpu().numpy()
    face_depths = (
        corners_3d[..., 2].reshape(-1, 2, 4).mean(dim=2).detach().cpu().numpy()
    )
    return face_points, face_depths


def build_cuboids(
    boxes3d: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    class_id_mapping: dict[int, str],
    intrinsics: np.ndarray,
) -> list[dict[str, Any]]:
    """Build JSON-ready cuboid records for one image."""
    if scores.size == 0:
        return []

    order = np.argsort(-scores, kind="stable")
    boxes3d = boxes3d[order]
    scores = scores[order]
    class_ids = class_ids[order]
    face_points, face_depths = _project_box_faces(boxes3d, intrinsics)
    cuboids: list[dict[str, Any]] = []
    for idx, (faces_2d, faces_z, score, class_id) in enumerate(
        zip(face_points, face_depths, scores, class_ids),
        start=1,
    ):
        class_id_int = int(class_id)
        front_face_idx = int(np.argmin(faces_z))
        back_face_idx = 1 - front_face_idx
        label = class_id_mapping.get(class_id_int, str(class_id_int))
        cuboids.append(
            {
                "id": uuid.uuid4().hex[:11],
                "direction": "front",
                "front": _sort_face_corners(faces_2d[front_face_idx]),
                "back": _sort_face_corners(faces_2d[back_face_idx]),
                "label": label,
                "order": idx,
                "score": round(float(score), 6),
                "original_cuboid_label": label,
            }
        )
    return cuboids


def _set_roi2det3d_params(
    model,
    score_threshold: float,
    max_per_image: int,
    nms: bool,
    class_agnostic_nms: bool,
    nms_iou_threshold: float,
) -> None:
    """Apply per-request post-process settings on the shared model."""
    roi2det3d = getattr(model, "roi2det3d", None)
    if roi2det3d is None:
        return
    roi2det3d.score_threshold = float(score_threshold)
    roi2det3d.max_per_img = int(max_per_image)
    roi2det3d.nms = bool(nms)
    roi2det3d.class_agnostic_nms = bool(class_agnostic_nms)
    roi2det3d.iou_threshold = float(nms_iou_threshold)


def _render_vis_base64(
    data: dict[str, Any],
    boxes3d,
    scores,
    class_ids,
    class_id_mapping: dict[int, str],
) -> str:
    """Render prediction PNG and return a base64 string."""
    imshow_bboxes3d = _runtime_deps()["imshow_bboxes3d"]
    with tempfile.NamedTemporaryFile(
        suffix=".png", prefix="mood_pred_", delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        imshow_bboxes3d(
            image=data["original_images"].cpu(),
            boxes3d=[box.cpu() for box in boxes3d],
            intrinsics=data["original_intrinsics"].cpu().numpy(),
            scores=[score.cpu() for score in scores],
            class_ids=[cid.cpu() for cid in class_ids],
            class_id_mapping=class_id_mapping,
            file_path=str(tmp_path),
            n_colors=len(class_id_mapping),
        )
        return base64.b64encode(tmp_path.read_bytes()).decode("ascii")
    finally:
        tmp_path.unlink(missing_ok=True)


class ModelRuntime:
    """Singleton model runtime protected by a request lock."""

    def __init__(self) -> None:
        self.device = "cpu"
        self.model = None
        self.model_load_seconds = 0.0
        self.lock = asyncio.Lock()

    def load(self) -> None:
        if self.model is not None:
            return
        deps = _runtime_deps()
        torch = deps["torch"]
        t0 = time.perf_counter()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model = deps["get_3d_mood_swin_base"](
            max_per_image=SETTINGS.max_per_image,
            score_thres=SETTINGS.score_threshold,
            nms=SETTINGS.nms,
            class_agnostic_nms=SETTINGS.class_agnostic_nms,
            iou_thres=SETTINGS.nms_iou_threshold,
        ).to(self.device)
        deps["load_model_checkpoint"](
            model,
            weights=SETTINGS.ckpt,
            rev_keys=[(r"^model\.", ""), (r"^module\.", "")],
        )
        model.eval()
        if self.device == "cuda":
            torch.cuda.synchronize()
        self.model = model
        self.model_load_seconds = time.perf_counter() - t0


RUNTIME = ModelRuntime()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    RUNTIME.load()
    yield


app = FastAPI(
    title="3D-MOOD FastAPI",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, Any]:
    loaded = RUNTIME.model is not None
    return {
        "status": "ok" if loaded else "loading",
        "device": RUNTIME.device,
        "model_loaded": loaded,
        "model_load_seconds": RUNTIME.model_load_seconds,
        "defaults": {
            "score_threshold": SETTINGS.score_threshold,
            "max_per_image": SETTINGS.max_per_image,
            "nms": SETTINGS.nms,
            "class_agnostic_nms": SETTINGS.class_agnostic_nms,
            "nms_iou_threshold": SETTINGS.nms_iou_threshold,
            "resize_h": SETTINGS.resize_h,
            "resize_w": SETTINGS.resize_w,
        },
    }


@app.post("/predict")
async def predict(
    files: list[UploadFile] = File(...),
    prompt: str = Form(...),
    intrinsics_json: str | None = Form(default=None),
    return_vis: bool = Form(default=False),
    score_threshold: float | None = Form(default=None),
    max_per_image: int | None = Form(default=None),
    nms: bool | None = Form(default=None),
    class_agnostic_nms: bool | None = Form(default=None),
    nms_iou_threshold: float | None = Form(default=None),
    resize_h: int | None = Form(default=None),
    resize_w: int | None = Form(default=None),
) -> dict[str, Any]:
    """Run B=1 inference sequentially for one or more uploaded images."""
    torch = _runtime_deps()["torch"]
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    resize_h = SETTINGS.resize_h if resize_h is None else int(resize_h)
    resize_w = SETTINGS.resize_w if resize_w is None else int(resize_w)
    score_threshold = (
        SETTINGS.score_threshold
        if score_threshold is None
        else float(score_threshold)
    )
    max_per_image = (
        SETTINGS.max_per_image
        if max_per_image is None
        else int(max_per_image)
    )
    nms = SETTINGS.nms if nms is None else bool(nms)
    class_agnostic_nms = (
        SETTINGS.class_agnostic_nms
        if class_agnostic_nms is None
        else bool(class_agnostic_nms)
    )
    nms_iou_threshold = (
        SETTINGS.nms_iou_threshold
        if nms_iou_threshold is None
        else float(nms_iou_threshold)
    )

    if resize_h <= 0 or resize_w <= 0:
        raise HTTPException(
            status_code=400,
            detail="resize_h and resize_w must be positive",
        )
    if max_per_image <= 0:
        raise HTTPException(
            status_code=400,
            detail="max_per_image must be positive",
        )
    if not (0.0 <= nms_iou_threshold <= 1.0):
        raise HTTPException(
            status_code=400,
            detail="nms_iou_threshold must be within [0, 1]",
        )

    input_texts = parse_prompt_tokens(prompt)
    preprocess = build_preprocess(resize_h, resize_w)
    if RUNTIME.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")

    results: list[dict[str, Any]] = []
    request_t0 = time.perf_counter()

    async with RUNTIME.lock:
        _set_roi2det3d_params(
            model=RUNTIME.model,
            score_threshold=score_threshold,
            max_per_image=max_per_image,
            nms=nms,
            class_agnostic_nms=class_agnostic_nms,
            nms_iou_threshold=nms_iou_threshold,
        )
        for upload in files:
            image_bytes = await upload.read()
            if not image_bytes:
                raise HTTPException(
                    status_code=400,
                    detail=f"Uploaded file is empty: {upload.filename}",
                )
            try:
                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            except Exception as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to decode image {upload.filename}: {exc}",
                ) from exc

            k_default = default_intrinsics(pil_image.height, pil_image.width)
            intrinsics = parse_intrinsics(intrinsics_json, k_default)
            data, class_id_mapping = prepare_one(
                pil_image=pil_image,
                intrinsics=intrinsics,
                input_texts=input_texts,
                preprocess=preprocess,
            )

            with torch.no_grad():
                if RUNTIME.device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                (
                    boxes,
                    boxes3d,
                    scores,
                    class_ids,
                    _depth_maps,
                    categories,
                ) = RUNTIME.model(
                    images=data["images"].to(RUNTIME.device),
                    input_hw=[data["input_hw"]],
                    original_hw=[data["original_hw"]],
                    intrinsics=data["intrinsics"].to(RUNTIME.device)[None],
                    padding=[data["padding"]],
                    input_texts=[input_texts],
                )
                if RUNTIME.device == "cuda":
                    torch.cuda.synchronize()
                infer_seconds = time.perf_counter() - t0

            result = {
                "file_name": upload.filename,
                "input_hw": list(data["original_hw"]),
                "inference_seconds": infer_seconds,
                "num_predictions": int(scores[0].numel()) if scores else 0,
                "cuboids": build_cuboids(
                    boxes3d=np.asarray(
                        _to_jsonable(boxes3d[0]), dtype=np.float32
                    )
                    if boxes3d
                    else np.empty((0, 10), dtype=np.float32),
                    scores=np.asarray(_to_jsonable(scores[0]), dtype=np.float32)
                    if scores
                    else np.empty((0,), dtype=np.float32),
                    class_ids=np.asarray(
                        _to_jsonable(class_ids[0]), dtype=np.int64
                    )
                    if class_ids
                    else np.empty((0,), dtype=np.int64),
                    class_id_mapping=class_id_mapping,
                    intrinsics=data["original_intrinsics"].cpu().numpy(),
                ),
                "categories": _to_jsonable(categories),
            }
            if return_vis:
                result["vis_image_base64"] = _render_vis_base64(
                    data=data,
                    boxes3d=boxes3d,
                    scores=scores,
                    class_ids=class_ids,
                    class_id_mapping=class_id_mapping,
                )
            results.append(result)

    return {
        "device": RUNTIME.device,
        "prompt_tokens": input_texts,
        "num_images": len(files),
        "request_seconds": time.perf_counter() - request_t0,
        "params": {
            "score_threshold": score_threshold,
            "max_per_image": max_per_image,
            "nms": nms,
            "class_agnostic_nms": class_agnostic_nms,
            "nms_iou_threshold": nms_iou_threshold,
            "resize_h": resize_h,
            "resize_w": resize_w,
        },
        "results": results,
    }


def _str2bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected bool string, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=SETTINGS.host)
    parser.add_argument("--port", type=int, default=SETTINGS.port)
    parser.add_argument("--ckpt", type=str, default=SETTINGS.ckpt)
    parser.add_argument(
        "--score_threshold", type=float, default=SETTINGS.score_threshold
    )
    parser.add_argument(
        "--max_per_image", type=int, default=SETTINGS.max_per_image
    )
    parser.add_argument("--nms", type=_str2bool, default=SETTINGS.nms)
    parser.add_argument(
        "--class_agnostic_nms",
        type=_str2bool,
        default=SETTINGS.class_agnostic_nms,
    )
    parser.add_argument(
        "--nms_iou_threshold",
        type=float,
        default=SETTINGS.nms_iou_threshold,
    )
    parser.add_argument("--resize_h", type=int, default=SETTINGS.resize_h)
    parser.add_argument("--resize_w", type=int, default=SETTINGS.resize_w)
    return parser.parse_args()


def apply_cli_overrides(args: argparse.Namespace) -> None:
    global SETTINGS
    SETTINGS = RuntimeSettings(
        ckpt=args.ckpt,
        score_threshold=args.score_threshold,
        max_per_image=args.max_per_image,
        nms=args.nms,
        class_agnostic_nms=args.class_agnostic_nms,
        nms_iou_threshold=args.nms_iou_threshold,
        resize_h=args.resize_h,
        resize_w=args.resize_w,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    args = parse_args()
    apply_cli_overrides(args)

    import uvicorn

    uvicorn.run(app, host=SETTINGS.host, port=SETTINGS.port, workers=1)
