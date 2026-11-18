"""Upload finished recordings to a private Kaggle dataset (one new version per run).

    uv run --group recorder python -m backend.recorder.upload            # upload now
    uv run --group recorder python -m backend.recorder.upload --dry-run  # show what would go up

Settings (environment or .env):
  KAGGLE_API_TOKEN   API token from kaggle.com > Settings > API (or KAGGLE_USERNAME + KAGGLE_KEY)
  KAGGLE_DATASET     <your-kaggle-username>/<dataset-slug>, e.g. jdoe/tradematic-ticks
  KAGGLE_TITLE       optional dataset title (default "Tradematic crypto event-contract ticks")

What is uploaded: only finished hour files of the recorder streams (webull, coinbase,
markets, settlements, health), at most one file per hour (compact > zstd > gzip). The
current hour, the Webull token folder, Parquet tables and anything else are never
included. Files are hard-linked (no copies) into <data>/.kaggle_stage/<YYYYMMDD>/, and the
Kaggle CLI uploads each day folder as one uncompressed tar (the files are already zstd).

Each run replaces the dataset's contents with a new version and deletes older versions
(`--delete-old-versions`), so storage used equals the size of the data, not of every
daily copy. The dataset is created PRIVATE on the first run. It contains Webull/Kalshi
market data; check their terms before ever making it public.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .config import REPO_ROOT, load_env_file
from .records import hour_files

STREAMS = ("webull", "coinbase", "markets", "settlements", "health")
STAGE = ".kaggle_stage"


def finished_files(data_dir, min_age_minutes=70):
    """One file per finished hour, for every stream. 'Finished' = untouched for a while."""
    now = time.time()
    out = []
    for s in STREAMS:
        for p in hour_files(data_dir, s):
            if now - p.stat().st_mtime >= min_age_minutes * 60:
                out.append(p)
    return out


def stage(data_dir, files, dataset, title):
    """Rebuild the stage folder of hard links; returns (stage_dir, bytes, file count)."""
    root = Path(data_dir) / STAGE
    if root.exists():
        shutil.rmtree(root)  # only links and a metadata file live here
    root.mkdir(parents=True)
    total = 0
    for p in files:
        day = root / p.parent.name  # YYYYMMDD
        day.mkdir(exist_ok=True)
        os.link(p, day / p.name)
        total += p.stat().st_size
    (root / "dataset-metadata.json").write_text(json.dumps(
        {"title": title, "id": dataset, "licenses": [{"name": "other"}]}, indent=2))
    return root, total, len(files)


def kaggle(*args, env=None):
    cmd = [sys.executable, "-m", "kaggle.cli", *args]
    return subprocess.run(cmd, check=False, env=env, capture_output=True, text=True)


def dataset_exists(dataset, env):
    r = kaggle("datasets", "status", dataset, env=env)
    return r.returncode == 0 and "error" not in (r.stdout + r.stderr).lower()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env-file", default=str(REPO_ROOT / ".env"))
    ap.add_argument("--data", default=None, help="default: RECORDER_DATA_DIR or backend/data/recorder")
    ap.add_argument("--min-age-minutes", type=float, default=70)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    load_env_file(a.env_file)
    data = Path(a.data or os.environ.get("RECORDER_DATA_DIR", REPO_ROOT / "backend/data/recorder"))
    dataset = os.environ.get("KAGGLE_DATASET", "").strip()
    title = os.environ.get("KAGGLE_TITLE", "Tradematic crypto event-contract ticks").strip()
    if dataset.count("/") != 1 or not 3 <= len(dataset.split("/")[1]) <= 50:
        ap.error("set KAGGLE_DATASET=<kaggle-username>/<dataset-slug> (slug 3-50 characters)")
    if not 6 <= len(title) <= 50:
        ap.error("KAGGLE_TITLE must be 6-50 characters (Kaggle rule)")
    if not (os.environ.get("KAGGLE_API_TOKEN") or (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))):
        ap.error("set KAGGLE_API_TOKEN (or KAGGLE_USERNAME and KAGGLE_KEY)")

    files = finished_files(data, a.min_age_minutes)
    if not files:
        print(f"nothing finished to upload yet in {data} "
              f"(files must be untouched for {a.min_age_minutes:g} minutes)")
        return
    days = sorted({p.parent.name for p in files})
    total = sum(p.stat().st_size for p in files)
    print(f"{len(files)} hour files, {total/2**30:.2f} GB, days {days[0]}..{days[-1]} "
          f"from {data} -> {dataset}")
    if a.dry_run:
        return
    root, _, _ = stage(data, files, dataset, title)
    env = dict(os.environ)
    note = f"{len(files)} hour files through {files and max(p.name for p in files)}"
    if dataset_exists(dataset, env):
        r = kaggle("datasets", "version", "-p", str(root), "-m", note, "-r", "tar",
                   "--delete-old-versions", env=env)
    else:
        r = kaggle("datasets", "create", "-p", str(root), "-r", "tar", env=env)  # private by default
    print(r.stdout[-4000:])
    if r.returncode != 0:
        print(r.stderr[-4000:], file=sys.stderr)
        raise SystemExit(r.returncode)


if __name__ == "__main__":
    main()
