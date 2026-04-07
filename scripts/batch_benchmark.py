"""Batch-run 3D-MOOD on a public image set (~100 images) with per-class text prompts.

Default data: **Caltech-101** (torchvision, ~101 object categories, ~125MB download).
Alternatives: `beans` (small / fast, 3 classes), `oxford_iiit_pet` (37 pet breeds, ~750MB),
`bdd100k` (official images under ``data/bdd100k``; see https://doc.bdd100k.com/download.html ),
`bdd100k_hf` (Hugging Face subset, or project copy under ``data/benchmark_datasets/bdd100k_hf/``),
`sun_rgbd` (local SUN RGB-D after MMDet3D-style prep: ``sunrgbd_trainval/image``),
`sun_rgbd_hf` (HF zip or project copy under ``data/benchmark_datasets/sun_rgbd_hf/``),
or your own folder (`--source folder`).

Populate local copies: ``python scripts/materialize_benchmark_datasets.py``.

Usage (from repository root):
  conda activate 3dmood
  export HF_ENDPOINT=https://hf-mirror.com   # optional
  python scripts/batch_benchmark.py --num_images 100

  # Quick smoke test (small download):
  python scripts/batch_benchmark.py --source beans --num_images 30

  # Custom images:
  python scripts/batch_benchmark.py --source folder --image_dir ./my_images --text_prompt chair.table.car

  # BDD100K (100K frames under images/100k/{train,val,test}):
  python scripts/batch_benchmark.py --source bdd100k --bdd100k_root data/bdd100k --num_images 100 --pred_only

  # BDD-style images from Hugging Face (no manual unpack; needs `pip install datasets`):
  python scripts/batch_benchmark.py --source bdd100k_hf --num_images 100 --pred_only

  # SUN RGB-D — local MMDet3D tree, or HF zip (~384MB, huggingface_hub):
  python scripts/batch_benchmark.py --source sun_rgbd --sun_rgbd_root data/sunrgbd --num_images 50 --pred_only
  python scripts/batch_benchmark.py --source sun_rgbd_hf --num_images 50 --pred_only

Outputs under runs/batch_<source>/:
  - vis/*_pred.png      predictions (default; add input/compare unless --pred_only)
  - category_grid.png   one sample per class from preds (or compare if present)
  - summary.json        mean/std inference, per-class means

  Only prediction images (no input, no side-by-side compare):
    python scripts/batch_benchmark.py --num_images 100 --pred_only
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import random
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from PIL import Image, ImageFile

# Caltech-101 trees may contain rare truncated JPEGs; still load for benchmarking.
ImageFile.LOAD_TRUNCATED_IMAGES = True

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
_spec = importlib.util.spec_from_file_location(
    "mood_demo", _SCRIPTS / "demo.py"
)
_mood_demo = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mood_demo)
get_3d_mood_swin_base = _mood_demo.get_3d_mood_swin_base
_save_input_output_preview = _mood_demo._save_input_output_preview

from vis4d.data.transforms.base import compose
from vis4d.data.transforms.normalize import NormalizeImages
from vis4d.data.transforms.resize import ResizeImages, ResizeIntrinsics
from vis4d.data.transforms.to_tensor import ToTensor
from vis4d.common.ckpt import load_model_checkpoint
from vis4d.vis.image.functional import imshow_bboxes3d

from opendet3d.data.transforms.pad import CenterPadImages, CenterPadIntrinsics
from opendet3d.data.transforms.resize import GenResizeParameters


class ImageListSource(Protocol):
    def __len__(self) -> int: ...
    def get(self, i: int) -> tuple[Image.Image, int, str]: ...


def default_intrinsics(h: int, w: int) -> np.ndarray:
    f = 0.72 * float(max(h, w))
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    return np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def class_to_prompt(name: str) -> str:
    """Turn dataset class name into dot-separated open-vocab tokens."""
    t = name.lower().replace("-", " ").replace("/", " ")
    parts = re.split(r"[\s_]+", t.strip())
    parts = [p for p in parts if p]
    return ".".join(parts[:6]) + ".object"


# BDD100K detection classes (approx.); override with --text_prompt
BDD100K_DEFAULT_TEXT_PROMPT = (
    "car.bus.truck.train.person.rider.bicycle.motorcycle."
    "traffic.light.traffic.sign"
)

# SUN RGB-D indoor object names (approx.); override with --text_prompt
SUN_RGB_D_DEFAULT_TEXT_PROMPT = (
    "table.chair.sofa.bed.desk.cabinet.shelf.lamp.monitor.tv."
    "bottle.cup.mug.book.keyboard.mouse.phone.pillow.object"
)


def build_preprocess(resize_shape: tuple[int, int] = (800, 1333)):
    return compose(
        transforms=[
            GenResizeParameters(shape=resize_shape),
            ResizeImages(),
            ResizeIntrinsics(),
            NormalizeImages(),
            CenterPadImages(stride=1, shape=resize_shape, update_input_hw=True),
            CenterPadIntrinsics(),
        ]
    )


def stratified_indices(
    n_total: int,
    labels: list[int],
    k: int,
    seed: int,
) -> list[int]:
    by_label: dict[int, list[int]] = defaultdict(list)
    for i, lb in enumerate(labels):
        by_label[int(lb)].append(i)
    labels_uniq = sorted(by_label.keys())
    rng = random.Random(seed)
    k = min(k, n_total)
    if not labels_uniq:
        return []
    per = k // len(labels_uniq)
    extra = k % len(labels_uniq)
    out: list[int] = []
    for j, lb in enumerate(labels_uniq):
        take = per + (1 if j < extra else 0)
        pool = by_label[lb][:]
        rng.shuffle(pool)
        out.extend(pool[:take])
    rng.shuffle(out)
    return out


def prepare_one(
    pil: Image.Image,
    intrinsics: np.ndarray,
    text_prompts: str,
    preprocess,
    to_tensor,
):
    pil = pil.convert("RGB")
    images = np.array(pil).astype(np.float32)[None, ...]
    h, w = images.shape[1], images.shape[2]
    input_texts = text_prompts.split(".")
    class_id_mapping = {i: t for i, t in enumerate(input_texts)}
    data_dict = {
        "images": images,
        "original_images": images,
        "input_hw": (h, w),
        "original_hw": (h, w),
        "intrinsics": intrinsics,
        "original_intrinsics": intrinsics,
    }
    data = preprocess([data_dict])[0]
    data = to_tensor([data])[0]
    return data, input_texts, class_id_mapping


def make_category_grid(
    records: list[dict],
    out_path: Path,
    repo_root: Path,
    thumb_max: int = 320,
    image_field: str = "compare_png",
) -> None:
    by_label: dict[int, Path] = {}
    for r in records:
        lb = r["label"]
        if lb not in by_label:
            path_key = image_field if image_field in r else "pred_png"
            p = Path(r[path_key])
            if not p.is_absolute():
                p = repo_root / p
            if p.is_file():
                by_label[lb] = p
    if not by_label:
        return
    labels_sorted = sorted(by_label.keys())
    cols = min(6, len(labels_sorted))
    rows = (len(labels_sorted) + cols - 1) // cols
    thumbs: list[Image.Image] = []
    for lb in labels_sorted:
        im = Image.open(by_label[lb]).convert("RGB")
        im.thumbnail((thumb_max * 2, thumb_max), Image.Resampling.LANCZOS)
        thumbs.append(im)
    cell_w = max(t.width for t in thumbs)
    cell_h = max(t.height for t in thumbs)
    grid = Image.new("RGB", (cols * cell_w, rows * cell_h), (48, 48, 48))
    for idx, t in enumerate(thumbs):
        r, c = divmod(idx, cols)
        x = c * cell_w + (cell_w - t.width) // 2
        y = r * cell_h + (cell_h - t.height) // 2
        grid.paste(t, (x, y))
    grid.save(out_path)


class Caltech101Source:
    """Parent of the torchvision layout: ``<parent>/caltech101/101_ObjectCategories``.

    When that folder exists, images are indexed from disk (sorted ``*.jpg`` per class).
    This tolerates missing ``image_0001.jpg`` etc., unlike ``torchvision.datasets.Caltech101``.
    """

    def __init__(self, parent: Path):
        from urllib.error import URLError

        from torchvision.datasets import Caltech101

        parent = Path(parent)
        parent.mkdir(parents=True, exist_ok=True)
        inner = parent / "caltech101" / "101_ObjectCategories"
        self._samples: list[tuple[Path, int, str]] | None = None
        self.class_names: list[str]

        if inner.is_dir():
            categories = sorted(
                p.name for p in inner.iterdir() if p.is_dir()
            )
            if "BACKGROUND_Google" in categories:
                categories.remove("BACKGROUND_Google")
            samples: list[tuple[Path, int, str]] = []
            for ci, cname in enumerate(categories):
                for jpg in sorted((inner / cname).glob("*.jpg")):
                    samples.append((jpg, ci, cname))
            if not samples:
                raise RuntimeError(
                    f"No .jpg images under {inner}. Re-extract Caltech-101 or fix the path."
                )
            self._samples = samples
            self.class_names = categories
            return

        try:
            self._ds = Caltech101(root=str(parent), download=True)
        except (URLError, OSError) as e:
            raise RuntimeError(
                "Caltech-101 download failed (no network or blocked). "
                "Either fix connectivity and retry, or unpack the official archive so that "
                f"{inner} exists, then run again (download will be skipped). "
                "Alternatively use --source folder --image_dir ... --text_prompt ..."
            ) from e
        self.class_names = self._ds.categories

    def __len__(self) -> int:
        if self._samples is not None:
            return len(self._samples)
        return len(self._ds)

    def get(self, i: int) -> tuple[Image.Image, int, str]:
        if self._samples is not None:
            path, idx, name = self._samples[i]
            return Image.open(path).convert("RGB"), idx, name
        pil, y = self._ds[i]
        idx = int(y)
        return pil, idx, self.class_names[idx]


class OxfordPetSource:
    """``<parent>/oxford-iiit-pet/`` (torchvision default subfolder name)."""

    def __init__(self, parent: Path):
        from urllib.error import URLError

        from torchvision.datasets import OxfordIIITPet

        parent = Path(parent)
        parent.mkdir(parents=True, exist_ok=True)
        inner = parent / "oxford-iiit-pet" / "images"
        try:
            self._ds = OxfordIIITPet(
                root=str(parent),
                split="trainval",
                target_types="category",
                download=True,
            )
        except (URLError, OSError) as e:
            if inner.is_dir():
                self._ds = OxfordIIITPet(
                    root=str(parent),
                    split="trainval",
                    target_types="category",
                    download=False,
                )
            else:
                raise RuntimeError(
                    "Oxford-IIIT Pet download failed. With data already on disk, ensure "
                    f"{inner} exists, or use --source caltech101 / folder."
                ) from e
        self.class_names = self._ds.classes

    def __len__(self) -> int:
        return len(self._ds)

    def get(self, i: int) -> tuple[Image.Image, int, str]:
        pil, y = self._ds[i]
        idx = int(y)
        return pil, idx, self.class_names[idx]


class BeansSource:
    def __init__(self) -> None:
        from datasets import load_dataset

        self._ds = load_dataset("beans", split="train")
        names = self._ds.features["labels"].names
        self.class_names = list(names)

    def __len__(self) -> int:
        return len(self._ds)

    def get(self, i: int) -> tuple[Image.Image, int, str]:
        ex = self._ds[i]
        pil = ex["image"]
        if not isinstance(pil, Image.Image):
            pil = Image.fromarray(np.array(pil))
        idx = int(ex["labels"])
        return pil, idx, self.class_names[idx]


class FolderSource:
    def __init__(self, folder: Path, glob: str):
        paths = sorted(folder.glob(glob))
        paths = [p for p in paths if p.is_file()]
        if not paths:
            raise FileNotFoundError(f"No files matching {glob!r} under {folder}")
        self._paths = paths

    def __len__(self) -> int:
        return len(self._paths)

    def get(self, i: int) -> tuple[Image.Image, int, str]:
        p = self._paths[i]
        return Image.open(p), 0, p.stem


class ImageFolderSource:
    """One subdirectory per class (same layout as ``torchvision.datasets.ImageFolder``)."""

    def __init__(self, root: Path):
        from torchvision.datasets import ImageFolder

        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"ImageFolder root not found: {root}")
        self._ds = ImageFolder(str(root))
        self.class_names = self._ds.classes

    def __len__(self) -> int:
        return len(self._ds)

    def get(self, i: int) -> tuple[Image.Image, int, str]:
        pil, y = self._ds[i]
        idx = int(y)
        return pil, idx, self.class_names[idx]


class Bdd100kImageSource:
    """Flat ``.jpg`` / ``.png`` under ``<root>/images/100k/<split>/`` (or ``<root>/bdd100k/...``)."""

    _suffixes = {".jpg", ".jpeg", ".png"}

    def __init__(self, root: Path, splits: tuple[str, ...]):
        root = Path(root)
        bases = [root]
        nested = root / "bdd100k"
        if nested.is_dir():
            bases.append(nested)
        paths: list[Path] = []
        for base in bases:
            for sp in splits:
                d = base / "images" / "100k" / sp
                if not d.is_dir():
                    continue
                for p in sorted(d.iterdir()):
                    if p.is_file() and p.suffix.lower() in self._suffixes:
                        paths.append(p)
        seen: set[Path] = set()
        uniq: list[Path] = []
        for p in paths:
            k = p.resolve()
            if k not in seen:
                seen.add(k)
                uniq.append(p)
        self._paths = uniq
        if not self._paths:
            raise FileNotFoundError(
                f"No BDD100K images under {root}/images/100k/{{{','.join(splits)}}} "
                f"(also tried {root}/bdd100k/images/100k/). "
                "Download the 100K image pack from https://doc.bdd100k.com/download.html "
                "and extract so that paths look like …/images/100k/train/*.jpg."
            )

    def __len__(self) -> int:
        return len(self._paths)

    def get(self, i: int) -> tuple[Image.Image, int, str]:
        p = self._paths[i]
        return Image.open(p).convert("RGB"), 0, p.stem


class Bdd100kHfSource:
    """Driving-scene JPEGs from a Hugging Face dataset (default: 1000 BDD-style frames)."""

    def __init__(
        self,
        repo_id: str = "chdw98/bdd100k_dataset_1000_xy",
        split: str = "train",
        *,
        parquet_file: Path | None = None,
    ):
        from datasets import load_dataset

        if parquet_file is not None:
            pf = Path(parquet_file)
            if not pf.is_file():
                raise FileNotFoundError(pf)
            self._ds = load_dataset("parquet", data_files=str(pf), split="train")
            self._repo_id = f"parquet:{pf.name}"
        else:
            self._ds = load_dataset(repo_id, split=split)
            self._repo_id = repo_id

    def __len__(self) -> int:
        return len(self._ds)

    def get(self, i: int) -> tuple[Image.Image, int, str]:
        row = self._ds[i]
        im = row["images"]
        if not isinstance(im, Image.Image):
            im = Image.fromarray(np.asarray(im))
        name = str(row.get("image_name") or f"{self._repo_id}_{i}")
        safe = re.sub(r"[^\w.\-]+", "_", name)[:80]
        return im.convert("RGB"), 0, safe


def _resolve_sun_rgbd_image_dir(root: Path) -> Path:
    """MMDet3D layout or common unpack paths; see mmdetection3d SUN RGB-D doc."""
    root = Path(root)
    candidates = [
        root / "sunrgbd_trainval" / "image",
        root / "image",
        root / "train_data" / "images",
        root / "images",
        root / "SUNRGBD" / "sunrgbd_trainval" / "image",
    ]
    for d in candidates:
        if d.is_dir() and (any(d.glob("*.jpg")) or any(d.glob("*.png"))):
            return d
    raise FileNotFoundError(
        f"No SUN RGB-D RGB folder under {root}. Expected e.g. "
        f"{root}/sunrgbd_trainval/image/*.jpg (MMDetection3D extract). "
        "See https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/datasets/sunrgbd.html "
        "or use --source sun_rgbd_hf for a Hugging Face zip subset."
    )


class SunRgbdLocalImageSource:
    """RGB frames from a prepared SUN RGB-D directory (flat ``*.jpg`` in ``image/``)."""

    def __init__(self, root: Path):
        img_dir = _resolve_sun_rgbd_image_dir(root)
        self._paths = sorted(img_dir.glob("*.jpg"))
        if not self._paths:
            self._paths = sorted(img_dir.glob("*.png"))
        if not self._paths:
            raise FileNotFoundError(f"No images in {img_dir}")

    def __len__(self) -> int:
        return len(self._paths)

    def get(self, i: int) -> tuple[Image.Image, int, str]:
        p = self._paths[i]
        return Image.open(p).convert("RGB"), 0, p.stem


class SunRgbdZipImageSource:
    """Read ``images/*.{jpg,png}`` from a zip (e.g. HF assignment pack) without full extract."""

    def __init__(self, zip_path: Path, member_prefix: str = "images/"):
        import zipfile

        self._zip_path = Path(zip_path)
        self._zf = zipfile.ZipFile(self._zip_path, "r")
        self._members = sorted(
            n
            for n in self._zf.namelist()
            if n.startswith(member_prefix)
            and not n.endswith("/")
            and Path(n).suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        if not self._members:
            self._zf.close()
            raise RuntimeError(
                f"No {member_prefix}*.jpg in {zip_path}. "
                "Try another zip or use --source sun_rgbd with MMDet3D-prepared files."
            )

    def __len__(self) -> int:
        return len(self._members)

    def get(self, i: int) -> tuple[Image.Image, int, str]:
        data = self._zf.read(self._members[i])
        pil = Image.open(io.BytesIO(data)).convert("RGB")
        stem = Path(self._members[i]).stem
        return pil, 0, stem

    def close(self) -> None:
        if hasattr(self, "_zf") and self._zf:
            self._zf.close()
            self._zf = None


def rel_path(p: Path, repo_root: Path) -> str:
    try:
        return str(p.resolve().relative_to(repo_root))
    except ValueError:
        return str(p.resolve())


def default_benchmark_datasets_dir(repo_root: Path) -> Path:
    return (repo_root / "data" / "benchmark_datasets").resolve()


def ensure_local_caltech101_layout(repo_root: Path, data_root: Path) -> None:
    """If ``<repo>/caltech-101/101_ObjectCategories`` exists, satisfy torchvision layout.

    ``torchvision.datasets.Caltech101(root=data_root)`` expects
    ``<data_root>/caltech101/101_ObjectCategories``.  When that path is missing but the
    repo ships (or the user cloned) ``caltech-101`` with extracted images, symlink
    ``<data_root>/caltech101`` -> ``<repo>/caltech-101``.
    """
    bundle = repo_root / "caltech-101"
    local_categories = bundle / "101_ObjectCategories"
    if not local_categories.is_dir():
        return
    expected = data_root / "caltech101" / "101_ObjectCategories"
    if expected.is_dir():
        return
    data_root.mkdir(parents=True, exist_ok=True)
    link = data_root / "caltech101"
    target = bundle.resolve()
    if link.is_symlink() and link.resolve() == target:
        return
    if link.exists() and not link.is_symlink():
        return
    if link.exists() or link.is_symlink():
        try:
            link.unlink()
        except OSError:
            return
    try:
        link.symlink_to(target, target_is_directory=True)
        print(f"[batch] Using local caltech-101: {link} -> {target}")
    except OSError as e:
        print(f"[batch] Warning: could not symlink caltech101: {e}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=(
            "caltech101",
            "beans",
            "oxford_iiit_pet",
            "bdd100k",
            "bdd100k_hf",
            "sun_rgbd",
            "sun_rgbd_hf",
            "folder",
            "image_folder",
        ),
        default="caltech101",
    )
    parser.add_argument("--num_images", type=int, default=100)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--image_dir",
        type=Path,
        default=None,
        help="With --source folder: directory of images.",
    )
    parser.add_argument(
        "--image_glob",
        type=str,
        default="*.jpg",
        help="Glob under image_dir (also tries *.png if needed).",
    )
    parser.add_argument(
        "--text_prompt",
        type=str,
        default=None,
        help="With --source folder: fixed prompt, e.g. chair.table.person",
    )
    parser.add_argument(
        "--image_folder",
        type=Path,
        default=None,
        help="With --source image_folder: root with class subdirs (ImageFolder layout).",
    )
    parser.add_argument(
        "--bdd100k_root",
        type=Path,
        default=None,
        help="Root containing images/100k/<split>/ (default: <data_root>/bdd100k).",
    )
    parser.add_argument(
        "--bdd100k_split",
        choices=("train", "val", "test", "all"),
        default="train",
        help="Which BDD100K image split to scan (default train; all = train+val+test).",
    )
    parser.add_argument(
        "--bdd100k_hf_repo",
        type=str,
        default="chdw98/bdd100k_dataset_1000_xy",
        help="Hugging Face dataset id for --source bdd100k_hf (requires `pip install datasets`).",
    )
    parser.add_argument(
        "--sun_rgbd_root",
        type=Path,
        default=None,
        help="Root for --source sun_rgbd (default: <data_root>/sunrgbd).",
    )
    parser.add_argument(
        "--sun_rgbd_hf_repo",
        type=str,
        default="ASTASTARIA27/SUN-RGBD-assignment-data",
        help="HF dataset id for --source sun_rgbd_hf (SUN RGB-D style indoor RGB zip).",
    )
    parser.add_argument(
        "--sun_rgbd_hf_zip",
        type=str,
        default="train_data.zip",
        help="Zip member name inside sun_rgbd_hf_repo (default train_data.zip, ~2000 RGB jpgs).",
    )
    parser.add_argument(
        "--benchmark_datasets_dir",
        type=Path,
        default=None,
        help="Directory with project-local copies (bdd100k_hf/*.parquet, sun_rgbd_hf/*.zip). "
        "Default: <repo>/data/benchmark_datasets. If files exist there, they are preferred over Hub download.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="https://huggingface.co/RoyYang0714/3D-MOOD/resolve/main/gdino3d_swin-b_120e_omni3d_834c97.pt",
    )
    parser.add_argument(
        "--pred_only",
        action="store_true",
        help="Save only *_pred.png (skip input and side-by-side compare).",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.1,
        help="Post-process score threshold in RoI2Det3D (lower -> more boxes).",
    )
    parser.add_argument(
        "--max_per_image",
        type=int,
        default=100,
        help="Max boxes kept per image before threshold/NMS.",
    )
    parser.add_argument(
        "--nms",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable NMS in post-process.",
    )
    parser.add_argument(
        "--class_agnostic_nms",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="NMS across classes; disable for per-class NMS.",
    )
    parser.add_argument(
        "--nms_iou_threshold",
        type=float,
        default=0.5,
        help="IoU threshold for NMS.",
    )
    parser.add_argument(
        "--resize_h",
        type=int,
        default=800,
        help="Preprocess resize height (higher may help tiny objects).",
    )
    parser.add_argument(
        "--resize_w",
        type=int,
        default=1333,
        help="Preprocess resize width (higher may help tiny objects).",
    )
    args = parser.parse_args()
    if args.resize_h <= 0 or args.resize_w <= 0:
        raise SystemExit("--resize_h and --resize_w must be positive")
    if args.max_per_image <= 0:
        raise SystemExit("--max_per_image must be positive")
    if not (0.0 <= args.nms_iou_threshold <= 1.0):
        raise SystemExit("--nms_iou_threshold must be in [0, 1]")

    repo_root = Path(__file__).resolve().parent.parent
    data_root = args.data_root.resolve()
    bench_dir = (
        args.benchmark_datasets_dir.resolve()
        if args.benchmark_datasets_dir
        else default_benchmark_datasets_dir(repo_root)
    )
    out_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (repo_root / "runs" / f"batch_{args.source}")
    )
    vis_dir = out_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    source: ImageListSource
    dataset_name = ""

    if args.source == "caltech101":
        ensure_local_caltech101_layout(repo_root, data_root)
        tarball = repo_root / "caltech-101" / "101_ObjectCategories.tar.gz"
        if not (data_root / "caltech101" / "101_ObjectCategories").is_dir():
            if tarball.is_file():
                print(
                    "[batch] caltech-101/101_ObjectCategories not found. "
                    "Extract the archive, e.g.: "
                    f"tar -xzf {tarball} -C {repo_root / 'caltech-101'}",
                    file=sys.stderr,
                )
                print(
                    "[batch] Check file integrity: gzip -t caltech-101/101_ObjectCategories.tar.gz",
                    file=sys.stderr,
                )
        print("[batch] Loading Caltech-101 (torchvision, download if needed) …")
        source = Caltech101Source(data_root)
        dataset_name = "torchvision Caltech101"
    elif args.source == "oxford_iiit_pet":
        print("[batch] Loading Oxford-IIIT Pet (~750MB, download if needed) …")
        source = OxfordPetSource(data_root)
        dataset_name = "torchvision OxfordIIITPet trainval"
    elif args.source == "beans":
        print("[batch] Loading Hugging Face 'beans' (small, plant disease, 3 classes) …")
        source = BeansSource()
        dataset_name = "HuggingFace beans train"
    elif args.source == "bdd100k":
        bdd_root = (
            args.bdd100k_root.resolve()
            if args.bdd100k_root
            else (data_root / "bdd100k").resolve()
        )
        if args.bdd100k_split == "all":
            sp_tuple = ("train", "val", "test")
        else:
            sp_tuple = (args.bdd100k_split,)
        print(f"[batch] Loading BDD100K images from {bdd_root} splits={sp_tuple} …")
        source = Bdd100kImageSource(bdd_root, sp_tuple)
        dataset_name = f"BDD100K images/{','.join(sp_tuple)}"
    elif args.source == "bdd100k_hf":
        local_pq = bench_dir / "bdd100k_hf" / "train-00000-of-00001.parquet"
        if local_pq.is_file():
            print(f"[batch] Using project-local benchmark data {local_pq} …")
            source = Bdd100kHfSource(parquet_file=local_pq)
            dataset_name = "bdd100k_hf (project data/benchmark_datasets)"
        else:
            print(
                f"[batch] Loading Hugging Face dataset {args.bdd100k_hf_repo!r} "
                "(first run downloads ~58MB parquet) …"
            )
            source = Bdd100kHfSource(args.bdd100k_hf_repo, split="train")
            dataset_name = f"HuggingFace {args.bdd100k_hf_repo}"
    elif args.source == "sun_rgbd":
        sroot = (
            args.sun_rgbd_root.resolve()
            if args.sun_rgbd_root
            else (data_root / "sunrgbd").resolve()
        )
        print(f"[batch] Loading SUN RGB-D RGB frames under {sroot} …")
        source = SunRgbdLocalImageSource(sroot)
        dataset_name = f"SUN RGB-D local {sroot}"
    elif args.source == "sun_rgbd_hf":
        local_zip = bench_dir / "sun_rgbd_hf" / args.sun_rgbd_hf_zip
        if local_zip.is_file():
            print(f"[batch] Using project-local benchmark data {local_zip} …")
            zpath = local_zip
            dataset_name = "sun_rgbd_hf (project data/benchmark_datasets)"
        else:
            from huggingface_hub import hf_hub_download

            print(
                f"[batch] Downloading/opening HF zip {args.sun_rgbd_hf_repo}/{args.sun_rgbd_hf_zip} "
                "(first run ~384MB) …"
            )
            zpath = Path(
                hf_hub_download(
                    args.sun_rgbd_hf_repo,
                    args.sun_rgbd_hf_zip,
                    repo_type="dataset",
                )
            )
            dataset_name = f"SUN RGB-D HF zip {args.sun_rgbd_hf_repo}"
        source = SunRgbdZipImageSource(zpath)
    elif args.source == "image_folder":
        if not args.image_folder:
            raise SystemExit("--source image_folder requires --image_folder /path/to/root")
        root_if = args.image_folder.resolve()
        print(f"[batch] Loading ImageFolder {root_if} …")
        source = ImageFolderSource(root_if)
        dataset_name = f"ImageFolder {root_if}"
    else:
        if not args.image_dir or not args.text_prompt:
            raise SystemExit(
                "--source folder requires --image_dir and --text_prompt"
            )
        folder = args.image_dir.resolve()
        try:
            source = FolderSource(folder, args.image_glob)
        except FileNotFoundError:
            alt = "*.png" if args.image_glob == "*.jpg" else "*.jpg"
            source = FolderSource(folder, alt)
        dataset_name = f"folder {folder}"

    n_avail = len(source)
    labels_list: list[int] = []
    for i in range(n_avail):
        _, lb, _ = source.get(i)
        labels_list.append(lb)

    n = min(args.num_images, n_avail)
    if args.source in (
        "folder",
        "bdd100k",
        "bdd100k_hf",
        "sun_rgbd",
        "sun_rgbd_hf",
    ):
        indices = list(range(n))
        random.Random(args.seed).shuffle(indices)
        indices = indices[:n]
    else:
        indices = stratified_indices(n_avail, labels_list, n, args.seed)

    print(f"[batch] dataset={dataset_name} size={n_avail}  running_n={n}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    resize_shape = (args.resize_h, args.resize_w)
    preprocess = build_preprocess(resize_shape=resize_shape)
    to_tensor = ToTensor()

    print(f"[batch] Building model on {device} …")
    t0 = time.perf_counter()
    model = get_3d_mood_swin_base(
        max_per_image=args.max_per_image,
        score_thres=args.score_threshold,
        nms=args.nms,
        class_agnostic_nms=args.class_agnostic_nms,
        iou_thres=args.nms_iou_threshold,
    ).to(device)
    load_model_checkpoint(
        model,
        weights=args.ckpt,
        rev_keys=[(r"^model\.", ""), (r"^module\.", "")],
    )
    model.eval()
    if device == "cuda":
        torch.cuda.synchronize()
    load_s = time.perf_counter() - t0
    print(f"[batch] Model build + checkpoint load: {load_s:.2f} s")
    print(
        "[batch] Inference params: "
        f"score_threshold={args.score_threshold}, "
        f"max_per_image={args.max_per_image}, "
        f"nms={args.nms}, "
        f"class_agnostic_nms={args.class_agnostic_nms}, "
        f"nms_iou_threshold={args.nms_iou_threshold}, "
        f"resize={resize_shape}"
    )

    def run_forward(data, input_texts):
        with torch.no_grad():
            return model(
                images=data["images"].to(device),
                input_hw=[data["input_hw"]],
                original_hw=[data["original_hw"]],
                intrinsics=data["intrinsics"].to(device)[None],
                padding=[data["padding"]],
                input_texts=[input_texts],
            )

    if args.source == "folder":
        fixed_prompt: str | None = args.text_prompt
    elif args.source in ("bdd100k", "bdd100k_hf"):
        fixed_prompt = args.text_prompt or BDD100K_DEFAULT_TEXT_PROMPT
    elif args.source in ("sun_rgbd", "sun_rgbd_hf"):
        fixed_prompt = args.text_prompt or SUN_RGB_D_DEFAULT_TEXT_PROMPT
    else:
        fixed_prompt = None

    # Warmup
    if indices:
        pil_w, y_w, name_w = source.get(indices[0])
        if fixed_prompt:
            p_w = fixed_prompt
        else:
            p_w = class_to_prompt(name_w)
        K_w = default_intrinsics(pil_w.height, pil_w.width)
        d_w, t_w, _ = prepare_one(pil_w, K_w, p_w, preprocess, to_tensor)
        with torch.no_grad():
            _ = run_forward(d_w, t_w)
        if device == "cuda":
            torch.cuda.synchronize()

    records: list[dict] = []
    times: list[float] = []
    pred_counts: list[int] = []
    by_label_times: dict[int, list[float]] = defaultdict(list)
    label_names: dict[int, str] = {}

    for rank, idx in enumerate(indices):
        pil, label, cname = source.get(idx)
        label_names.setdefault(label, cname)
        if fixed_prompt:
            text_prompts = fixed_prompt
            display_name = cname
        else:
            text_prompts = class_to_prompt(cname)
            display_name = cname

        K = default_intrinsics(pil.height, pil.width)
        data, input_texts, class_id_mapping = prepare_one(
            pil, K, text_prompts, preprocess, to_tensor
        )

        slug = re.sub(r"[^a-z0-9]+", "-", display_name.lower())[:40].strip("-")
        stem = f"{rank:03d}_{label:03d}_{slug}_{idx}"
        out_path = vis_dir / f"{stem}_pred.png"
        raw_path = vis_dir / f"{stem}_input.png"
        cmp_path = vis_dir / f"{stem}_compare.png"
        if not args.pred_only:
            pil.convert("RGB").save(raw_path)

        with torch.no_grad():
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            boxes, boxes3d, scores, class_ids, depth_maps, categories = (
                run_forward(data, input_texts)
            )
            if device == "cuda":
                torch.cuda.synchronize()
            infer_s = time.perf_counter() - t0

        times.append(infer_s)
        by_label_times[label].append(infer_s)
        num_predictions = int(scores[0].numel()) if scores else 0
        pred_counts.append(num_predictions)

        imshow_bboxes3d(
            image=data["original_images"].cpu(),
            boxes3d=[b.cpu() for b in boxes3d],
            intrinsics=data["original_intrinsics"].cpu().numpy(),
            scores=[s.cpu() for s in scores],
            class_ids=[c.cpu() for c in class_ids],
            class_id_mapping=class_id_mapping,
            file_path=str(out_path),
            n_colors=len(class_id_mapping),
        )
        if not args.pred_only:
            _save_input_output_preview(raw_path, out_path, cmp_path)

        rec: dict = {
            "index_in_dataset": idx,
            "label": label,
            "class_name": display_name,
            "text_prompts": text_prompts,
            "inference_seconds": infer_s,
            "num_predictions": num_predictions,
            "pred_png": rel_path(out_path, repo_root),
        }
        if not args.pred_only:
            rec["input_png"] = rel_path(raw_path, repo_root)
            rec["compare_png"] = rel_path(cmp_path, repo_root)
        records.append(rec)
        if (rank + 1) % 10 == 0 or rank == 0:
            print(
                f"[batch] {rank + 1}/{n}  {display_name[:36]:<36}  "
                f"infer={infer_s:.3f}s"
            )

    mean_t = statistics.mean(times) if times else 0.0
    std_t = statistics.stdev(times) if len(times) > 1 else 0.0
    per_class = {
        label_names.get(lb, str(lb)): {
            "n": len(v),
            "mean_inference_seconds": float(statistics.mean(v)),
        }
        for lb, v in sorted(by_label_times.items())
    }

    summary = {
        "dataset": dataset_name,
        "source": args.source,
        "num_ran": n,
        "device": device,
        "model_load_seconds": load_s,
        "inference_seconds_mean": mean_t,
        "inference_seconds_stdev": std_t,
        "inference_seconds_median": float(statistics.median(times)) if times else 0.0,
        "predictions_per_image_mean": float(statistics.mean(pred_counts))
        if pred_counts
        else 0.0,
        "predictions_per_image_median": float(statistics.median(pred_counts))
        if pred_counts
        else 0.0,
        "inference_params": {
            "score_threshold": args.score_threshold,
            "max_per_image": args.max_per_image,
            "nms": args.nms,
            "class_agnostic_nms": args.class_agnostic_nms,
            "nms_iou_threshold": args.nms_iou_threshold,
            "resize_h": args.resize_h,
            "resize_w": args.resize_w,
        },
        "per_class": per_class,
        "pred_only": bool(args.pred_only),
        "records": records,
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    grid_path = out_dir / "category_grid.png"
    grid_field = "pred_png" if args.pred_only else "compare_png"
    if (
        args.source
        not in ("folder", "bdd100k", "bdd100k_hf", "sun_rgbd", "sun_rgbd_hf")
        or len(by_label_times) > 1
    ):
        make_category_grid(
            records, grid_path, repo_root, image_field=grid_field
        )
    else:
        if records:
            k = "pred_png" if args.pred_only else "compare_png"
            p0 = Path(records[0][k])
            if not p0.is_absolute():
                p0 = repo_root / p0
            if p0.is_file():
                Image.open(p0).save(grid_path)

    print()
    print("[batch] === Summary ===")
    print(f"  Images:              {n}")
    print(f"  Distinct labels:     {len(by_label_times)}")
    print(f"  Model load:          {load_s:.2f} s (excluded from average)")
    print(f"  Mean inference:      {mean_t:.4f} s")
    print(f"  Std inference:       {std_t:.4f} s")
    print(f"  Median inference:    {summary['inference_seconds_median']:.4f} s")
    print(
        "  Predictions/image:   "
        f"mean={summary['predictions_per_image_mean']:.2f} "
        f"median={summary['predictions_per_image_median']:.2f}"
    )
    print(f"  summary.json:        {summary_path}")
    print(f"  category_grid.png:   {grid_path}")
    vis_hint = "vis/*_pred.png only" if args.pred_only else "vis/*_input/_pred/_compare"
    print(f"  per-image vis:       {vis_dir}/ ({vis_hint})")

    closer = getattr(source, "close", None)
    if callable(closer):
        closer()


if __name__ == "__main__":
    main()
