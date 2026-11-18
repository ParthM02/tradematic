# Live tick recorder

Records Webull event-contract streams (order book, snapshot, trades) for the BTC/ETH/SOL
Kalshi 15-minute and hourly contracts, Coinbase spot ticks for the underlyings, and the
official settlement of every recorded contract. Finished days can be uploaded to a
private Kaggle dataset.

## Setup

```sh
cp .env.example .env          # fill WEBULL_APP_KEY / WEBULL_APP_SECRET; .env is git-ignored
uv sync --group recorder
uv run --group recorder python -m backend.recorder               # runs until Ctrl-C
uv run --group recorder python -m backend.recorder --no-webull   # no keys: Coinbase + discovery
uv run --group recorder python -m pytest -q backend/recorder/tests
```

On first start, Webull may require approving the API token in the Webull app (its 2FA).
The SDK waits for approval, then caches the token in `backend/data/recorder/.webull_token/`.

## Sources

| Source | Transport | Auth | Content |
| --- | --- | --- | --- |
| Webull market data | MQTT over TCP+TLS `data-api.webull.com:1883`, or WebSocket `wss://data-api.webull.com:8883/mqtt` (`WEBULL_TRANSPORT`) | App key + secret | `event-quote` (yes/no bid ladders), `event-snapshot`, `event-tick`, `notice` |
| Coinbase Exchange | WebSocket `wss://ws-feed.exchange.coinbase.com` | none | `ticker`, `match`, `heartbeat` |
| Kalshi REST | HTTPS, public | none | contract discovery (strikes, open/close) and settlements |

Contracts are chosen every 30 s: every active 15-minute market, plus the 7 strikes
nearest 50 cents in the current hour's ladder per asset. A contract stays subscribed
until it closes; the total is capped by `RECORDER_MAX_SYMBOLS`. Kalshi tickers are
used directly as Webull symbols.

## Output

`backend/data/recorder/<stream>/<YYYYMMDD>/<stream>_<YYYYMMDDTHH>.jsonl.zst`: zstd-compressed
JSONL (level 9 by default), one record per line: `{"recv_ns", "src", "type", "data"}`.
`recv_ns` is local receive time in nanoseconds. Webull's own timestamp is in
`data.basic.timestamp`, Coinbase's in `data.time`. Streams: `webull`, `coinbase`,
`markets`, `settlements`, `health`. Recorder events (connects, disconnects, subscribe
results, Coinbase trade-id gaps, stalls) are written inline, so outage windows can be
masked during analysis.

Read any hour file with `backend.recorder.records.iter_records(path)`. It handles live
`.jsonl.zst`, `.compact.jsonl.zst` and the older `.jsonl.gz` files, and skips a damaged
tail after a crash. `hour_files(root, stream)` lists one file per hour. To build Parquet
tables for analysis: `uv run --group recorder python -m backend.recorder.convert`.

Durability: the writer ends a zstd block about once a second, so a crash loses roughly the
last second. Files are not fsync'd (kinder to SD cards), so a sudden power cut can lose
the last few seconds the OS had not written yet. A restart appends a new frame to the
current hour's file.

Size: the 2026-11-17 overnight recording was 564 MB as gzip. As level-9 zstd it is about
37% smaller (about 0.8 GB per day, more in busy US hours).

## Kaggle upload

Set `KAGGLE_API_TOKEN` and `KAGGLE_DATASET=<username>/<slug>` in `.env`. The recorder then
uploads automatically once a day at `RECORDER_UPLOAD_TIME` (local time, default 03:30).
It also uploads 15 minutes after start if the last success is more than 25 hours old.
A failed upload is retried hourly up to 3 times. Each attempt is one log line; full
output goes to `backend/data/recorder/logs/upload.log`.

Kaggle allows at most 50 top-level files per dataset, so the dataset holds one folder per
UTC day, each uploaded as one `.tar`. That fits up to 49 days of recording. Past that,
the upload stops with a clear error instead of uploading part of the data.

The upload cannot disturb recording. It runs as a separate child process at the lowest
CPU priority, with idle-only disk access on Linux (`ionice -c3`), in its own session, so
Ctrl-C does not interrupt it mid-way. Stopping the recorder stops it cleanly, and the
next run sends everything again. Files are hard-linked into a staging folder (no copies).
The Kaggle CLI builds each day's tar (about 1-1.5 GB) in a temp folder; on the Pi that is
the RAM disk `/dev/shm`, so archives never touch the SD card
(`RECORDER_UPLOAD_TMPDIR` overrides it).
`RECORDER_UPLOAD=off` turns it off.

To upload by hand (for example from a second tmux window):

```sh
uv run --frozen --only-group recorder python -m backend.recorder.upload --dry-run
uv run --frozen --only-group recorder python -m backend.recorder.upload
```

Only finished hour files of the five streams are uploaded: never the current hour, the
Webull token folder, Parquet tables, logs or `.env`. The first run creates the dataset
**private**. Later runs publish a new version and delete older versions, so Kaggle storage
stays equal to the size of the data. To fetch a month for research:
`kaggle datasets download parthmehta10/tradematic-crypto-ticks --unzip`, then
extract the day `.tar` files into `<stream>/<day>/` folders if Kaggle has not already. Upload from one machine only: each upload replaces
the dataset's contents with that machine's files. The data is Webull/Kalshi market data;
check their terms before making the dataset public.

## Optional: compaction

`python -m backend.recorder.compact` rewrites finished hours at zstd level 19 and folds each
Coinbase `match` into its `ticker`. It verifies an exact round trip before deleting the
source. Output is about 40% of gzip for Webull and 70% for Coinbase, but it costs minutes of
CPU per hour, so it is not scheduled.

## Raspberry Pi

The recorder needs only its own dependency group (about 73 MB, no numpy/LightGBM):

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh      # installs uv to ~/.local/bin
cd ~/tradematic && uv sync --frozen --only-group recorder
cp .env.example .env && nano .env                     # Webull + Kaggle keys
uv run --frozen --only-group recorder python -m backend.recorder --seconds 120   # first run: approve token
```

Then install the services (edit `User=` and paths in the three files if not `pi` /
`/home/pi/tradematic`):

```sh
sudo cp backend/recorder/deploy/tradematic-* /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now tradematic-recorder.service tradematic-upload.timer
journalctl -u tradematic-recorder -f                  # live log
systemctl list-timers tradematic-upload.timer         # next upload
```

The log gets one status line every 15 minutes (`RECORDER_LOG_SECONDS`), for example
`last 15 min: webull 154,210 msgs, coinbase 31,882 msgs | webull connected 26/26 subscribed | ...`.
Health records are still saved every 30 s.

The recorder never exits on its own. If Webull's connection dies, or sends nothing for
`RECORDER_STALL_SECONDS` (default 600) while contracts are subscribed, it reconnects only
Webull: after 60 s, doubling up to 15 minutes while reconnects keep failing without data
(e.g. bad credentials). Coinbase, discovery and settlements keep recording throughout.
Each attempt is logged once and saved as `webull_waiting` / `webull_restart` events in
`health`. The systemd units are optional: `tradematic-recorder` restarts the process only
if it crashes. If you install `tradematic-upload.timer`, set `RECORDER_UPLOAD=off` so
uploads do not happen twice.

To run it in tmux instead of systemd:

```sh
tmux new -s recorder
cd ~/tradematic && uv run --frozen --only-group recorder python -m backend.recorder
# detach: Ctrl-b d    reattach: tmux attach -t recorder    stop: Ctrl-c
```

## Kalshi maintenance

Webull routes event-contract trading to Kalshi, which pauses trading every Thursday
3:00–5:00 AM ET and may drop connections then
([Kalshi docs](https://docs.kalshi.com/getting_started/maintenance_and_pauses)). The
recorder treats these as quiet periods:

- the weekly window, plus `RECORDER_MAINTENANCE_BUFFER_MINUTES` (default 15) on each side,
  so 2:45–5:15 AM ET, adjusted for daylight saving;
- extra windows Kalshi announces in `GET /exchange/schedule` (checked every 6 hours);
- any time `GET /exchange/status` reports the exchange or the crypto index paused
  (checked every minute; a crypto-index-only pause must be seen twice in a row).

During a quiet period it keeps recording whatever arrives, never restarts for silence,
and, if Webull's connection dies, waits for the period to end and reconnects in-process
instead of exiting. Repeated discovery errors are logged once per failure streak. Each
start and end is saved as a `maintenance` event in the `health` stream. Every health record
lists the current reasons. For analysis, `exchange.quiet_intervals(records)` returns
the windows to mask. Settlement checks wait up to 6 hours after close, for contracts that
settle late around maintenance.

## Limits to keep in mind

- Webull caps order-book pushes at about 3 per second per contract and allows 5
  connections per App Key. Faster book changes are merged server-side; trades are not
  capped. `notice` messages report drops.
- Coinbase is a proxy for the CF Benchmarks index that Kalshi settles on (measured gap:
  BTC -0.4 bp, ETH -1.9 bp, SOL -2.2 bp).
- Webull's documented crypto event-contract trading hours are weekdays 8:00–18:00 ET.
  Market data streamed outside those hours on 2026-11-17. If it ever stops on
  weekends, the stall check will keep reconnecting Webull (every 10-25 minutes) until data
  resumes. That is harmless, but `RECORDER_STALL_SECONDS=0` turns it off.
