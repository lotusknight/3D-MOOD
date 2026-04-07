#!/usr/bin/env python3
"""Download (if needed) and copy benchmark datasets into data/benchmark_datasets/.

Populates:
  data/benchmark_datasets/bdd100k_hf/train-00000-of-00001.parquet   (~56MB)
  data/benchmark_datasets/sun_rgbd_hf/train_data.zip                (~367MB)

Requires: pip install huggingface_hub  (and network / HF mirror).

Usage (from repo root):
  export HF_ENDPOINT=https://hf-mirror.com   # optional
  python scripts/materialize_benchmark_datasets.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEST_ROOT = REPO_ROOT / "data" / "benchmark_datasets"


def main() -> int:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Install huggingface_hub: pip install huggingface_hub", file=sys.stderr)
        return 1

    DEST_ROOT.mkdir(parents=True, exist_ok=True)

    bdd_dir = DEST_ROOT / "bdd100k_hf"
    bdd_dir.mkdir(parents=True, exist_ok=True)
    bdd_src = Path(
        hf_hub_download(
            "chdw98/bdd100k_dataset_1000_xy",
            "data/train-00000-of-00001.parquet",
            repo_type="dataset",
        )
    )
    bdd_dst = bdd_dir / "train-00000-of-00001.parquet"
    shutil.copy2(bdd_src.resolve(), bdd_dst)
    print(f"Wrote {bdd_dst} ({bdd_dst.stat().st_size // 1_000_000} MB)")

    sun_dir = DEST_ROOT / "sun_rgbd_hf"
    sun_dir.mkdir(parents=True, exist_ok=True)
    sun_src = Path(
        hf_hub_download(
            "ASTASTARIA27/SUN-RGBD-assignment-data",
            "train_data.zip",
            repo_type="dataset",
        )
    )
    sun_dst = sun_dir / "train_data.zip"
    shutil.copy2(sun_src.resolve(), sun_dst)
    print(f"Wrote {sun_dst} ({sun_dst.stat().st_size // 1_000_000} MB)")

    print("\nbatch_benchmark will use these automatically when present:")
    print("  python scripts/batch_benchmark.py --source bdd100k_hf --num_images 100 --pred_only")
    print("  python scripts/batch_benchmark.py --source sun_rgbd_hf --num_images 50 --pred_only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
