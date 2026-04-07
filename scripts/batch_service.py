#!/usr/bin/env python3
"""Batch-call the FastAPI service and save per-image outputs.

The script is designed for a running 3D-MOOD FastAPI server. It can either:

- send one image per POST request, or
- send multiple images in one POST request and split the response back into
  per-image JSON files and per-image visualization PNGs.

Outputs are written under:

  <output_dir>/
    json/       per-image JSON
    vis/        per-image prediction image
    batches/    raw batch responses, one JSON per POST request
    summary.json
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import random
import statistics
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import requests


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _bool_from_str(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected bool string, got {value!r}")


def _guess_content_type(path: Path) -> str:
    content_type, _ = mimetypes.guess_type(path.name)
    return content_type or "application/octet-stream"


def _iter_images(image_dir: Path, image_glob: str, recursive: bool) -> list[Path]:
    if recursive:
        paths = sorted(image_dir.rglob(image_glob))
    else:
        paths = sorted(image_dir.glob(image_glob))
    return [p for p in paths if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]


def _choose_images(paths: list[Path], num_images: int, shuffle: bool, seed: int) -> list[Path]:
    if num_images <= 0 or num_images >= len(paths):
        return paths
    if shuffle:
        rng = random.Random(seed)
        items = paths[:]
        rng.shuffle(items)
        return items[:num_images]
    return paths[:num_images]


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _save_vis_image(path: Path, b64: str | None) -> bool:
    if not b64:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(b64))
    return True


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _normalize_endpoint(endpoint: str) -> str:
    if not endpoint.startswith("/"):
        return "/" + endpoint
    return endpoint


def _build_request_data(args: argparse.Namespace) -> dict[str, str]:
    data: dict[str, str] = {
        "prompt": args.prompt,
        "return_vis": "true" if args.return_vis else "false",
    }
    if args.intrinsics_json:
        data["intrinsics_json"] = args.intrinsics_json
    if args.score_threshold is not None:
        data["score_threshold"] = str(args.score_threshold)
    if args.max_per_image is not None:
        data["max_per_image"] = str(args.max_per_image)
    if args.nms is not None:
        data["nms"] = "true" if args.nms else "false"
    if args.class_agnostic_nms is not None:
        data["class_agnostic_nms"] = "true" if args.class_agnostic_nms else "false"
    if args.nms_iou_threshold is not None:
        data["nms_iou_threshold"] = str(args.nms_iou_threshold)
    if args.resize_h is not None:
        data["resize_h"] = str(args.resize_h)
    if args.resize_w is not None:
        data["resize_w"] = str(args.resize_w)
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="http://127.0.0.1:8000",
        help="FastAPI service base URL.",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/predict",
        help="Path of the inference endpoint.",
    )
    parser.add_argument(
        "--image_dir",
        type=Path,
        required=True,
        help="Directory containing images to send.",
    )
    parser.add_argument(
        "--image_glob",
        type=str,
        default="*.jpg",
        help="Glob pattern used to select images.",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Search for images recursively under image_dir.",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=0,
        help="Maximum number of images to process; 0 means all images.",
    )
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Shuffle images before selecting num_images.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when shuffle is enabled.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("runs/service_batch"),
        help="Directory where JSON and PNG outputs will be written.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Dot-separated text prompt sent to the service.",
    )
    parser.add_argument(
        "--intrinsics_json",
        type=str,
        default=None,
        help="Optional 3x3 intrinsics matrix encoded as JSON.",
    )
    parser.add_argument(
        "--return_vis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ask the service to return base64 visualization images.",
    )
    parser.add_argument(
        "--request_mode",
        choices=("batch", "single"),
        default="batch",
        help="Send multiple files in one POST request, or one file per request.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of images to send in each batch request.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--skip_existing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip files whose JSON output already exists.",
    )
    parser.add_argument(
        "--save_batch_json",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save the raw service response for each POST request.",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=None,
        help="Override service score_threshold for this run.",
    )
    parser.add_argument(
        "--max_per_image",
        type=int,
        default=None,
        help="Override service max_per_image for this run.",
    )
    parser.add_argument(
        "--nms",
        type=_bool_from_str,
        default=None,
        help="Override service nms flag. Accepts true/false.",
    )
    parser.add_argument(
        "--class_agnostic_nms",
        type=_bool_from_str,
        default=None,
        help="Override service class_agnostic_nms flag. Accepts true/false.",
    )
    parser.add_argument(
        "--nms_iou_threshold",
        type=float,
        default=None,
        help="Override service nms_iou_threshold for this run.",
    )
    parser.add_argument(
        "--resize_h",
        type=int,
        default=None,
        help="Override service resize_h for this run.",
    )
    parser.add_argument(
        "--resize_w",
        type=int,
        default=None,
        help="Override service resize_w for this run.",
    )
    return parser.parse_args()


def _post_chunk(
    session: requests.Session,
    url: str,
    chunk: list[Path],
    data: dict[str, str],
    timeout: float,
) -> tuple[dict[str, Any], float]:
    with ExitStack() as stack:
        files = []
        for path in chunk:
            fh = stack.enter_context(path.open("rb"))
            files.append(
                (
                    "files",
                    (
                        path.name,
                        fh,
                        _guess_content_type(path),
                    ),
                )
            )
        t0 = time.perf_counter()
        response = session.post(url, files=files, data=data, timeout=timeout)
        elapsed = time.perf_counter() - t0
    response.raise_for_status()
    return response.json(), elapsed


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch_size must be positive")
    if args.num_images < 0:
        raise SystemExit("--num_images must be >= 0")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")

    base_url = _normalize_base_url(args.base_url)
    endpoint = _normalize_endpoint(args.endpoint)
    url = f"{base_url}{endpoint}"
    image_dir = args.image_dir.resolve()
    output_dir = args.output_dir.resolve()
    json_dir = output_dir / "json"
    vis_dir = output_dir / "vis"
    batch_dir = output_dir / "batches"
    json_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    batch_dir.mkdir(parents=True, exist_ok=True)

    all_images = _iter_images(image_dir, args.image_glob, args.recursive)
    if not all_images:
        raise SystemExit(
            f"No images found under {image_dir} with glob {args.image_glob!r}"
        )
    chosen_images = _choose_images(all_images, args.num_images, args.shuffle, args.seed)

    data = _build_request_data(args)
    session = requests.Session()
    records: list[dict[str, Any]] = []
    request_seconds: list[float] = []
    inference_seconds: list[float] = []
    pred_counts: list[int] = []

    print(f"[service] POST {url}")
    print(f"[service] images={len(chosen_images)} request_mode={args.request_mode}")

    request_index = 0
    if args.request_mode == "single":
        chunks = [[p] for p in chosen_images]
    else:
        chunks = [
            chosen_images[i : i + args.batch_size]
            for i in range(0, len(chosen_images), args.batch_size)
        ]

    for chunk in chunks:
        if args.skip_existing:
            filtered = []
            for path in chunk:
                if not (json_dir / f"{path.stem}.json").is_file():
                    filtered.append(path)
            chunk = filtered
            if not chunk:
                continue

        request_index += 1
        payload, http_seconds = _post_chunk(session, url, chunk, data, args.timeout)
        request_seconds.append(http_seconds)

        if args.save_batch_json:
            batch_path = batch_dir / f"batch_{request_index:04d}.json"
            _save_json(batch_path, payload)

        results = payload.get("results", [])
        if len(results) != len(chunk):
            raise RuntimeError(
                f"Response result count {len(results)} does not match request "
                f"chunk size {len(chunk)} for request {request_index}."
            )

        for local_index, (path, result) in enumerate(zip(chunk, results), start=1):
            infer_s = float(result.get("inference_seconds", 0.0))
            n_pred = int(result.get("num_predictions", 0))
            inference_seconds.append(infer_s)
            pred_counts.append(n_pred)

            record = {
                "request_index": request_index,
                "request_local_index": local_index,
                "batch_size": len(chunk),
                "input_file": path.name,
                "input_path": str(path),
                "output_json": f"json/{path.stem}.json",
                "output_image": f"vis/{path.stem}_pred.png",
                "num_predictions": n_pred,
                "inference_seconds": infer_s,
                "http_seconds": http_seconds,
                "service_device": payload.get("device"),
                "prompt_tokens": payload.get("prompt_tokens"),
                "params": payload.get("params"),
            }
            records.append(record)

            per_image_json = {
                "record": record,
                "result": result,
            }
            _save_json(json_dir / f"{path.stem}.json", per_image_json)
            _save_vis_image(
                vis_dir / f"{path.stem}_pred.png",
                result.get("vis_image_base64"),
            )

            if len(records) == 1 or len(records) % 10 == 0 or len(records) == len(chosen_images):
                print(
                    f"[batch] {len(records)}/{len(chosen_images)} "
                    f"{path.name} infer={infer_s:.3f}s preds={n_pred}"
                )

    summary = {
        "base_url": base_url,
        "endpoint": endpoint,
        "request_url": url,
        "image_dir": str(image_dir),
        "output_dir": str(output_dir),
        "num_images": len(chosen_images),
        "request_mode": args.request_mode,
        "batch_size": args.batch_size,
        "http_requests": len(request_seconds),
        "http_seconds_total": sum(request_seconds),
        "http_seconds_mean": statistics.mean(request_seconds)
        if request_seconds
        else 0.0,
        "http_seconds_median": statistics.median(request_seconds)
        if request_seconds
        else 0.0,
        "inference_seconds_mean": statistics.mean(inference_seconds)
        if inference_seconds
        else 0.0,
        "inference_seconds_median": statistics.median(inference_seconds)
        if inference_seconds
        else 0.0,
        "predictions_per_image_mean": statistics.mean(pred_counts)
        if pred_counts
        else 0.0,
        "predictions_per_image_median": statistics.median(pred_counts)
        if pred_counts
        else 0.0,
        "request_params": {
            "prompt": args.prompt,
            "intrinsics_json": args.intrinsics_json,
            "return_vis": args.return_vis,
            "score_threshold": args.score_threshold,
            "max_per_image": args.max_per_image,
            "nms": args.nms,
            "class_agnostic_nms": args.class_agnostic_nms,
            "nms_iou_threshold": args.nms_iou_threshold,
            "resize_h": args.resize_h,
            "resize_w": args.resize_w,
        },
        "records": records,
    }
    _save_json(output_dir / "summary.json", summary)
    print(f"[done] summary.json -> {output_dir / 'summary.json'}")
    print(f"[done] per-image json -> {json_dir}")
    print(f"[done] per-image vis  -> {vis_dir}")


if __name__ == "__main__":
    main()
