#!/usr/bin/env bash
# Download the official BDD100K *100K images* pack (~5.3GB) from ETH Zurich mirrors.
# Layout after extract should include: bdd100k/images/100k/{train,val,test}/*.jpg
# Doc: https://doc.bdd100k.com/download.html  (md5: 5a0359c86a0b8713adab1eee9a3041cb)
#
# Usage:
#   bash scripts/fetch_bdd100k_official.sh [DEST_DIR]
# DEST_DIR default: data/bdd100k_official

set -euo pipefail
DEST="${1:-data/bdd100k_official}"
BASE="https://dl.cv.ethz.ch/bdd100k/data"
# Filenames vary by mirror version; try common names until one works.
CANDIDATES=(
  "bdd100k_images_100k.tar.gz"
  "bdd100k-images-100k-train-val-test.tar.gz"
  "bdd100k_images_100k.zip"
)

mkdir -p "$DEST"
cd "$DEST"

for name in "${CANDIDATES[@]}"; do
  url="$BASE/$name"
  echo "Trying $url"
  if curl -fL --retry 3 -C - -o "$name" "$url"; then
    echo "Downloaded $name"
    case "$name" in
      *.tar.gz)
        tar -xzf "$name"
        ;;
      *.zip)
        unzip -q "$name"
        ;;
    esac
    echo "Done. Point batch_benchmark at the folder that contains bdd100k/images/100k, e.g.:"
    echo "  python scripts/batch_benchmark.py --source bdd100k --bdd100k_root $DEST --num_images 100 --pred_only"
    exit 0
  fi
  rm -f "$name"
done

echo "None of the candidate URLs worked. Open https://dl.cv.ethz.ch/bdd100k/data/ in a browser," >&2
echo "download the 100K images archive (see doc.bdd100k.com), then extract under $DEST" >&2
exit 1
