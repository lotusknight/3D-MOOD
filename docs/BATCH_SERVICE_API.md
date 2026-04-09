# Batch Service API Design

This document describes the input and output contract expected by `scripts/batch_service.py`.
It is meant to help design the FastAPI service API that the script calls.

## Purpose

`scripts/batch_service.py` is a thin client around a running 3D-MOOD inference service.
Its job is to:

- collect images from a local folder
- send them to the service through `POST /predict`
- save one JSON result per image
- save one visualization PNG per image when `return_vis=true`
- write raw batch responses and a run summary for later inspection

The script supports both:

- `single` mode: one image per HTTP request
- `batch` mode: multiple images in one HTTP request

## Service Endpoints

The script currently depends on these endpoints:

- `GET /health`
- `POST /predict`

### `GET /health`

Used as a startup check.

Expected response shape:

```json
{
  "status": "ok",
  "device": "cuda",
  "model_loaded": true,
  "model_load_seconds": 8.10,
  "defaults": {
    "score_threshold": 0.1,
    "max_per_image": 100,
    "nms": true,
    "class_agnostic_nms": true,
    "nms_iou_threshold": 0.5,
    "resize_h": 800,
    "resize_w": 1333
  }
}
```

The exact fields can vary a little, but the script assumes the service can be probed before sending requests.

## `POST /predict` Contract

### Request

The script sends `multipart/form-data`.

Required fields:

- `files`: one or more uploaded images
- `prompt`: dot-separated open-vocabulary text prompt, for example `car.vehicle.road`

Optional fields:

- `intrinsics_json`: JSON-encoded 3x3 camera intrinsics matrix
- `return_vis`: `true` or `false`
- `score_threshold`
- `max_per_image`
- `nms`
- `class_agnostic_nms`
- `nms_iou_threshold`
- `resize_h`
- `resize_w`

### Request Semantics

The API should preserve the input order of `files`.

This is important because the client maps each returned result back to the corresponding input file by index.

Recommended behavior:

- `files` length may be `1` or more
- the response `results` array should have the same length as `files`
- `results[i]` should correspond to `files[i]`

### Response

Expected top-level response shape:

```json
{
  "device": "cuda",
  "prompt_tokens": ["car", "vehicle", "road"],
  "num_images": 2,
  "request_seconds": 0.98,
  "params": {
    "score_threshold": 0.1,
    "max_per_image": 100,
    "nms": true,
    "class_agnostic_nms": true,
    "nms_iou_threshold": 0.5,
    "resize_h": 800,
    "resize_w": 1333
  },
  "results": [
    {
      "file_name": "0001.jpg",
      "input_hw": [720, 1280],
      "inference_seconds": 0.77,
      "num_predictions": 3,
      "cuboids": [
        {
          "id": "a0wanluk5hu",
          "direction": "front",
          "front": {
            "tl": {"x": 136.8, "y": 289.0},
            "tr": {"x": 164.1, "y": 287.4},
            "br": {"x": 165.2, "y": 342.8},
            "bl": {"x": 138.1, "y": 349.4}
          },
          "back": {
            "tl": {"x": 10.7, "y": 288.8},
            "tr": {"x": 47.5, "y": 287.3},
            "br": {"x": 49.1, "y": 341.5},
            "bl": {"x": 12.6, "y": 347.8}
          },
          "label": "car",
          "order": 1,
          "score": 0.941779,
          "original_cuboid_label": "car"
        }
      ],
      "categories": [],
      "vis_image_base64": "..."
    }
  ]
}
```

### Per-image result fields

The client currently expects these fields in each element of `results`:

- `file_name`
- `input_hw`
- `inference_seconds`
- `num_predictions`
- `cuboids`
- `categories`
- `vis_image_base64` when `return_vis=true`

Each entry in `cuboids` is now a fully projected 2D cuboid object:

- `id`: generated cuboid identifier
- `direction`: currently always `"front"`
- `front`: closer face in image space, with `tl`, `tr`, `br`, `bl`
- `back`: farther face in image space, with `tl`, `tr`, `br`, `bl`
- `label`: predicted label
- `order`: 1-indexed rank after sorting by descending score
- `score`: confidence score
- `original_cuboid_label`: raw label name preserved for consumers

Each point under `front` and `back` has the form:

```json
{
  "x": 136.8,
  "y": 289.0
}
```

The service computes these corners from the model's internal 3D boxes and camera intrinsics before returning the response. Consumers therefore do not need intrinsics to reconstruct the wireframe in 2D.

## Batch Behavior

`scripts/batch_service.py` supports two request modes.

### `single`

Each image is sent in its own `POST /predict` call.

Pros:

- easiest to debug
- one failing image does not block other images
- response size stays small

Cons:

- more HTTP overhead
- slower for large runs

### `batch`

Several images are sent in one `POST /predict` call.

Pros:

- fewer HTTP round-trips
- better for throughput-oriented runs
- easier to treat a run as one logical batch

Cons:

- larger request and response bodies
- one request failure affects more images
- server must preserve result ordering carefully

## Files Written by the Script

The script writes to the directory passed by `--output_dir`.

### Directory layout

```text
<output_dir>/
  json/
    <image_stem>.json
  vis/
    <image_stem>_pred.png
  batches/
    batch_0001.json
    batch_0002.json
  summary.json
```

### Per-image JSON

Each per-image JSON file contains:

```json
{
  "record": {
    "request_index": 1,
    "request_local_index": 1,
    "batch_size": 8,
    "input_file": "0001.jpg",
    "input_path": "/abs/path/to/0001.jpg",
    "output_json": "json/0001.json",
    "output_image": "vis/0001_pred.png",
    "num_predictions": 3,
    "inference_seconds": 0.77,
    "http_seconds": 1.02,
    "service_device": "cuda",
    "prompt_tokens": ["car", "vehicle", "road"],
    "params": { }
  },
  "result": { }
}
```

This is useful if you want the batch runner to become a stable client for another API implementation.

### Batch JSON

If `--save_batch_json=true`, the raw response of each HTTP request is stored under `batches/`.

This is useful for:

- debugging request-level failures
- comparing server versions
- inspecting all results returned by one batch call

### Summary JSON

`summary.json` stores run-level metadata and aggregate metrics:

- base URL and endpoint
- image source and output directory
- number of images processed
- request mode and batch size
- HTTP latency statistics
- inference latency statistics
- prediction count statistics
- all per-image records

## Important Design Constraints

If you design a service API to work with this script, the following constraints matter:

1. The response should preserve input order.
2. The response should always return one result per input file.
3. `return_vis=true` should embed a base64 PNG or an equivalent image payload.
4. The service should accept extra per-request inference overrides through form fields.
5. `/health` should expose enough information to tell whether the model is loaded and what the defaults are.
6. Each cuboid should already be in projected 2D front/back corner form so the client does not need camera intrinsics to draw it.

## Suggested Minimal API

For a clean service design, the minimal API surface can be:

- `GET /health`
- `POST /predict`

That is enough for `scripts/batch_service.py` to work without any client-side changes.

