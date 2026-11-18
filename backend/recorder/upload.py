"""Upload finished recordings to one private Kaggle dataset (a new version per run).

    uv run --group recorder python -m backend.recorder.upload            # upload now
    uv run --group recorder python -m backend.recorder.upload --dry-run  # show what would go up

Settings (environment or .env):
  KAGGLE_API_TOKEN   API token from kaggle.com > Settings > API (or KAGGLE_USERNAME + KAGGLE_KEY)
  KAGGLE_DATASET     <your-kaggle-username>/<dataset-slug>, e.g. jdoe/tradematic-crypto-ticks
  KAGGLE_TITLE       optional dataset title (default "Tradematic crypto ticks")
  RECORDER_UPLOAD_TMPDIR  optional scratch folder for archives (default: RAM disk /dev/shm)

Layout. Kaggle allows at most 50 top-level files per dataset, so the dataset holds one
folder per UTC day (up to 48 days) plus an `_about/` folder with a format README, each
uploaded by the Kaggle CLI as one uncompressed .tar (the hour files inside are already
zstd). The `_about/` folder is required: Kaggle unpacks a dataset that contains exactly one
folder into top-level files (verified 2026-11-18), so there must always be two. More than
48 days is refused with a clear error rather than uploaded partially.

What is uploaded: only finished hour files of the recorder streams (webull, coinbase,
markets, settlements, health), one file per hour (compact > zstd > gzip). Never the current
hour, the Webull token folder, Parquet tables, logs or anything else. Files are
hard-linked (no copies) into <data>/.kaggle_stage/<YYYYMMDD>/.

SD-card friendly. The CLI builds each day's tar in a temporary folder before sending it,
one day at a time (about 1-1.5 GB). That folder is put on the RAM disk (/dev/shm on
Linux) when it has room, so the archive never touches the SD card. Otherwise it falls back
to the normal temp folder and says so.

Each run replaces the dataset's contents and deletes older versions, so Kaggle storage
equals the data size. The dataset is created PRIVATE. It contains Webull/Kalshi market
data; check their terms before ever making it public. Upload from one machine only.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .config import REPO_ROOT, load_env_file
from .records import hour_files

STREAMS = ("webull", "coinbase", "markets", "settlements", "health")
STAGE = ".kaggle_stage"
MAX_TOP_LEVEL = 50  # Kaggle limit on top-level files per dataset
MAX_DAYS = MAX_TOP_LEVEL - 2  # minus the _about folder and dataset-metadata.json
ABOUT = """# Tradematic crypto ticks

One folder per UTC day (YYYYMMDD). Each holds one zstd-compressed JSONL file per stream
per UTC hour: <stream>_<YYYYMMDDTHH>.jsonl.zst (or .compact.jsonl.zst / older .jsonl.gz).
Streams: webull (event-quote, event-snapshot, event-tick, notice), coinbase (ticker,
match, heartbeat), markets (Kalshi contract metadata), settlements (official results),
health (recorder state; `maintenance` events mark Kalshi maintenance windows).

Each line: {"recv_ns": local receive time ns, "src": source, "type": message type, "data": payload}.
Read with backend.recorder.records.iter_records(path) from the tradematic repo.

This folder also keeps Kaggle from flattening the upload: a dataset containing exactly one
folder is unpacked to top-level files, so there are always at least two folders.
"""


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
    """Rebuild <data>/.kaggle_stage/ as day folders of hard links; returns the folder."""
    root = Path(data_dir) / STAGE
    if root.exists():
        shutil.rmtree(root)  # only links and a metadata file live here
    days = sorted({p.parent.name for p in files})
    if len(days) > MAX_DAYS:
        raise RuntimeError(f"{len(days)} days exceeds Kaggle's {MAX_TOP_LEVEL} top-level entries "
                           f"({MAX_DAYS} day folders + _about + metadata); split the data into a second dataset")
    root.mkdir(parents=True)
    # Always a second folder: Kaggle flattens a dataset that contains exactly one folder.
    (root / "_about").mkdir()
    (root / "_about" / "README.md").write_text(ABOUT)
    for p in files:
        day = root / p.parent.name
        day.mkdir(exist_ok=True)
        os.link(p, day / p.name)
    (root / "dataset-metadata.json").write_text(json.dumps(
        {"title": title, "id": dataset, "licenses": [{"name": "other"}]}, indent=2))
    return root


def largest_day_bytes(files):
    days = {}
    for p in files:
        days[p.parent.name] = days.get(p.parent.name, 0) + p.stat().st_size
    return max(days.values(), default=0)


def scratch_dir(need_bytes):
    """A temp folder for the CLI's archives: RAM disk if it fits, else the system temp."""
    override = os.environ.get("RECORDER_UPLOAD_TMPDIR", "").strip()
    candidates = [Path(override)] if override else [Path("/dev/shm")]
    for c in candidates:
        try:
            if c.is_dir() and os.access(c, os.W_OK) and shutil.disk_usage(c).free > need_bytes * 1.2 + 2**28:
                d = c / "tradematic-upload"
                d.mkdir(exist_ok=True)
                return d, "RAM disk" if str(c).startswith("/dev/shm") else str(c)
        except OSError:
            continue
    return Path(tempfile.gettempdir()), "system temp folder (not enough RAM-disk space)"


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
    ap.add_argument("--summary-file", default=None, help=argparse.SUPPRESS)  # used by the recorder
    a = ap.parse_args()
    load_env_file(a.env_file)
    data = Path(a.data or os.environ.get("RECORDER_DATA_DIR", REPO_ROOT / "backend/data/recorder"))
    dataset = os.environ.get("KAGGLE_DATASET", "").strip()
    title = os.environ.get("KAGGLE_TITLE", "Tradematic crypto ticks").strip()
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
    print(f"{len(files)} hour files in {len(days)} day folders ({days[0]}..{days[-1]}), "
          f"{total/2**30:.2f} GB from {data} -> {dataset}")
    if len(days) > MAX_DAYS:
        raise SystemExit(f"too many days for one dataset ({len(days)} > {MAX_DAYS})")
    if a.dry_run:
        return
    root = stage(data, files, dataset, title)
    tmp, where = scratch_dir(largest_day_bytes(files))
    env = dict(os.environ, TMPDIR=str(tmp), TEMP=str(tmp), TMP=str(tmp))
    print(f"day archives are built in the {where}", flush=True)
    note = f"{len(files)} hour files through {max(p.name for p in files)}"
    if dataset_exists(dataset, env):
        r = kaggle("datasets", "version", "-p", str(root), "-m", note, "-r", "tar",
                   "--delete-old-versions", env=env)
    else:
        r = kaggle("datasets", "create", "-p", str(root), "-r", "tar", env=env)  # private by default
    print(r.stdout[-4000:])
    shutil.rmtree(root, ignore_errors=True)
    if a.summary_file:
        Path(a.summary_file).write_text(json.dumps({"files": len(files), "bytes": total,
                                                    "days": [days[0], days[-1]]}))
    if r.returncode != 0:
        print(r.stderr[-4000:], file=sys.stderr)
        raise SystemExit(r.returncode)


if __name__ == "__main__":
    main()
