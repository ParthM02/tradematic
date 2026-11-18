"""Recorder settings, read from environment variables (optionally a .env file).

Secrets are only held in memory. `Config.__repr__` never prints them.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SERIES = ("KXBTC15M", "KXETH15M", "KXSOL15M", "KXBTCD", "KXETHD", "KXSOLD")
DEFAULT_PRODUCTS = ("BTC-USD", "ETH-USD", "SOL-USD")

# Webull OpenAPI hosts: https://developer.webull.com/apis/docs/sdk/
WEBULL_HOSTS = {
    "prod": {"http": "api.webull.com", "mqtt": "data-api.webull.com"},
    "sandbox": {"http": "api.sandbox.webull.com", "mqtt": "data-api.sandbox.webull.com"},
}
# MQTT over TCP+TLS uses 1883; MQTT over WebSocket uses wss://...:8883/mqtt.
WEBULL_PORTS = {"tcp": 1883, "websockets": 8883}


def load_env_file(path):
    """Load KEY=VALUE pairs without overriding variables already set in the shell."""
    path = Path(path)
    if not path.exists():
        return False
    from dotenv import load_dotenv

    load_dotenv(path, override=False)
    return True


def _csv(name, default):
    raw = os.environ.get(name, "").strip()
    return tuple(s.strip() for s in raw.split(",") if s.strip()) if raw else tuple(default)


def _hhmm(raw):
    raw = raw.strip()
    h, _, m = raw.partition(":")
    if not (h.isdigit() and m.isdigit() and int(h) < 24 and int(m) < 60):
        raise ValueError("RECORDER_UPLOAD_TIME must be HH:MM (24-hour, local time)")
    return f"{int(h):02d}:{int(m):02d}"


def _int(name, default):
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


@dataclass(frozen=True)
class Config:
    data_dir: Path
    series: tuple = DEFAULT_SERIES
    products: tuple = DEFAULT_PRODUCTS
    hourly_strikes: int = 7
    max_symbols: int = 60
    discovery_seconds: int = 30
    health_seconds: int = 30
    log_seconds: int = 900
    zstd_level: int = 9
    stall_seconds: int = 600
    maintenance_buffer_minutes: int = 15
    upload_time: str = "03:30"
    webull_env: str = "prod"
    webull_region: str = "us"
    webull_transport: str = "tcp"
    webull_depth: int = 10
    webull_app_key: str = field(default="", repr=False)
    webull_app_secret: str = field(default="", repr=False)

    @property
    def webull_enabled(self):
        return bool(self.webull_app_key and self.webull_app_secret)

    @property
    def webull_http_host(self):
        return WEBULL_HOSTS[self.webull_env]["http"]

    @property
    def webull_mqtt_host(self):
        return WEBULL_HOSTS[self.webull_env]["mqtt"]

    @property
    def webull_mqtt_port(self):
        return WEBULL_PORTS[self.webull_transport]

    @property
    def token_dir(self):
        return self.data_dir / ".webull_token"

    def public_summary(self):
        """Settings safe to write into the recorded data (no secrets)."""
        return {
            "series": list(self.series),
            "products": list(self.products),
            "hourly_strikes": self.hourly_strikes,
            "max_symbols": self.max_symbols,
            "webull_enabled": self.webull_enabled,
            "webull_env": self.webull_env,
            "webull_transport": self.webull_transport,
            "webull_depth": self.webull_depth,
            "zstd_level": self.zstd_level,
            "stall_seconds": self.stall_seconds,
            "maintenance_buffer_minutes": self.maintenance_buffer_minutes,
            "upload_time": self.upload_time,
        }


def from_env():
    env = os.environ.get("WEBULL_ENV", "prod").strip().lower()
    transport = os.environ.get("WEBULL_TRANSPORT", "tcp").strip().lower()
    if env not in WEBULL_HOSTS:
        raise ValueError(f"WEBULL_ENV must be one of {sorted(WEBULL_HOSTS)}")
    if transport not in WEBULL_PORTS:
        raise ValueError(f"WEBULL_TRANSPORT must be one of {sorted(WEBULL_PORTS)}")
    data_dir = Path(os.environ.get("RECORDER_DATA_DIR", REPO_ROOT / "backend/data/recorder"))
    return Config(
        data_dir=data_dir.expanduser().resolve(),
        series=_csv("RECORDER_SERIES", DEFAULT_SERIES),
        products=_csv("RECORDER_PRODUCTS", DEFAULT_PRODUCTS),
        hourly_strikes=_int("RECORDER_HOURLY_STRIKES", 7),
        max_symbols=_int("RECORDER_MAX_SYMBOLS", 60),
        discovery_seconds=_int("RECORDER_DISCOVERY_SECONDS", 30),
        health_seconds=_int("RECORDER_HEALTH_SECONDS", 30),
        log_seconds=_int("RECORDER_LOG_SECONDS", 900),
        zstd_level=_int("RECORDER_ZSTD_LEVEL", 9),
        stall_seconds=_int("RECORDER_STALL_SECONDS", 600),
        maintenance_buffer_minutes=_int("RECORDER_MAINTENANCE_BUFFER_MINUTES", 15),
        upload_time=_hhmm(os.environ.get("RECORDER_UPLOAD_TIME", "03:30")),
        webull_env=env,
        webull_region=os.environ.get("WEBULL_REGION", "us").strip(),
        webull_transport=transport,
        webull_depth=_int("WEBULL_DEPTH", 10),
        webull_app_key=os.environ.get("WEBULL_APP_KEY", "").strip(),
        webull_app_secret=os.environ.get("WEBULL_APP_SECRET", "").strip(),
    )
