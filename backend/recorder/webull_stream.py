"""Webull event-contract market data over MQTT (Webull's streaming protocol).

Webull streams through an MQTT 3.1.1 broker, reachable over TCP+TLS (port 1883) or
WebSocket (wss://data-api.webull.com:8883/mqtt). Subscriptions are made with signed
HTTP calls whose session_id equals the MQTT ClientId. Event-contract topics are
`event-quote` (yes/no bid ladders), `event-snapshot` and `event-tick` (trades), all
protobuf. Docs: https://developer.webull.com/apis/docs/market-data-api/data-streaming-api/

Known limits from the docs: at most 5 concurrent connections per App Key, and the
server pushes at most 3 messages per second per connection. That cap means bursts may
be conflated server-side; `notice` messages report drops and are recorded as-is.

This module replaces the SDK's result objects with protobuf -> dict decoding so every
field is kept, and calls the subscribe endpoint directly so HTTP failures are recorded
instead of silently ignored. Subscriptions are not restored by the server after a
reconnect, so the full desired set is re-subscribed on every successful CONNECT.
"""
import json
import logging
import threading
import time
import uuid

from google.protobuf.json_format import MessageToDict

STREAM = "webull"
CATEGORY = "US_EVENT"
SUB_TYPES = ["QUOTE", "SNAPSHOT", "TICK"]
BATCH = 50  # symbols per subscribe call; the server-side limit is undocumented
log = logging.getLogger(__name__)


class ProtoDictDecoder:
    """SDK payload decoder that returns every protobuf field as a plain dict."""

    def __init__(self, message_cls):
        self.message_cls = message_cls

    def parse(self, payload):
        msg = self.message_cls()
        msg.ParseFromString(payload)
        return MessageToDict(msg, preserving_proto_field_name=True)


class JsonDecoder:
    def parse(self, payload):
        text = payload.decode("utf-8") if payload else ""
        try:
            return json.loads(text) if text else None
        except ValueError:
            return {"raw": text}


def register_decoders(client):
    from webull.data.quotes.subscribe import message_pb2 as pb

    for topic, cls in [("event-quote", pb.EventQuote), ("event-snapshot", pb.EventSnapshot),
                       ("event-tick", pb.EventTick), ("quote", pb.Quote),
                       ("snapshot", pb.Snapshot), ("tick", pb.Tick)]:
        client.register_payload_decoder(topic, ProtoDictDecoder(cls))
    client.register_payload_decoder("notice", JsonDecoder())


class WebullStreamer:
    def __init__(self, cfg, writer):
        self.cfg, self.writer = cfg, writer
        self.session_id = uuid.uuid4().hex
        self.desired = set()
        self.subscribed = set()
        self.connected = False
        self.last_message_ns = 0
        self._lock = threading.Lock()
        self._client = None
        self._thread = None
        self.on_fatal = None  # called (from the MQTT thread) if streaming stops for good

    # ---- lifecycle -------------------------------------------------------------
    def start(self):
        from webull.data.data_streaming_client import DataStreamingClient

        cfg = self.cfg
        cfg.token_dir.mkdir(parents=True, exist_ok=True)
        c = DataStreamingClient(cfg.webull_app_key, cfg.webull_app_secret, cfg.webull_region,
                                self.session_id, http_host=cfg.webull_http_host,
                                mqtt_host=cfg.webull_mqtt_host, mqtt_port=cfg.webull_mqtt_port,
                                tls_enable=True, transport=cfg.webull_transport)
        if cfg.webull_transport == "websockets":
            c.ws_set_options(path="/mqtt")
        c.set_token_dir(str(cfg.token_dir))
        register_decoders(c)
        c.on_connect_success = self._on_connect
        c.on_quotes_message = self._on_message
        c.on_disconnect = self._on_disconnect
        self._client = c
        # The SDK's own async starter always enables a log file in the working directory;
        # running its blocking loop in our thread with logger_enable=False avoids that.
        self._thread = threading.Thread(target=self._loop, name="webull-mqtt", daemon=True)
        self._thread.start()
        self._event("starting", {"session_id": self.session_id, "host": cfg.webull_mqtt_host,
                                 "port": cfg.webull_mqtt_port, "transport": cfg.webull_transport})

    def _loop(self):
        error = "streaming loop exited"
        try:
            # First run may block in the SDK's token check until the token is approved
            # in the Webull app (Webull's 2FA). The approved token is cached in token_dir.
            self._client.connect_and_loop_forever(logger_enable=False)
        except Exception as e:  # noqa: BLE001 - any failure is reported and retried
            error = f"{type(e).__name__}: {e}"[:1000]
        if self._client._thread_terminate:
            self._event("stopped", {})  # stopped on purpose by the recorder
            return
        self._event("fatal", {"error": error})
        log.warning("webull streaming stopped: %s", error)
        if self.on_fatal is not None:
            self.on_fatal()

    def stop(self):
        self.connected = False  # no more subscribe calls on this session
        if self._client is None:
            return
        self._client._thread_terminate = True  # stops the SDK retry loop
        try:
            self._client.disconnect()
        except Exception as e:  # noqa: BLE001
            log.warning("webull disconnect failed: %r", e)

    # ---- callbacks (MQTT thread) ----------------------------------------------
    def _on_connect(self, client, api_client, session_id):
        self.connected = True
        with self._lock:
            self.subscribed.clear()
            targets = sorted(self.desired)
        self._event("connected", {"session_id": session_id, "resubscribing": len(targets)})
        self._subscribe(targets)

    def _on_disconnect(self, client, userdata, rc):
        self.connected = False
        self._event("disconnected", {"rc": rc})

    def _on_message(self, client, topic, data):
        now = time.time_ns()
        self.last_message_ns = now
        self.writer.write(STREAM, "webull", topic, data, now)

    # ---- subscription management (any thread) ---------------------------------
    def set_desired(self, symbols):
        """Make the subscribed set match `symbols`; safe to call while disconnected."""
        symbols = set(symbols)
        with self._lock:
            self.desired = symbols
            add = sorted(symbols - self.subscribed)
            drop = sorted(self.subscribed - symbols)
        if self.connected:
            self._subscribe(add)
            self._unsubscribe(drop)

    def _call(self, kind, symbols):
        from webull.data.quotes.market_streaming_data import MarketDataStreaming

        api = MarketDataStreaming(self._client.api_client)
        for i in range(0, len(symbols), BATCH):
            chunk = symbols[i:i + BATCH]
            try:
                if kind == "subscribe":
                    r = api.subscribe(self.session_id, chunk, CATEGORY, SUB_TYPES, self.cfg.webull_depth)
                else:
                    r = api.unsubscribe(self.session_id, chunk, CATEGORY, SUB_TYPES)
                ok = r.status_code == 200
                detail = {"symbols": chunk, "status": r.status_code, "body": r.text[:2000]}
            except Exception as e:  # noqa: BLE001 - SDK raises on non-2xx
                ok, detail = False, {"symbols": chunk, "error": f"{type(e).__name__}: {e}"[:2000]}
            self._event(f"{kind}_{'ok' if ok else 'failed'}", detail)
            if ok:
                with self._lock:
                    if kind == "subscribe":
                        self.subscribed |= set(chunk)
                    else:
                        self.subscribed -= set(chunk)

    def _subscribe(self, symbols):
        if symbols:
            self._call("subscribe", symbols)

    def _unsubscribe(self, symbols):
        if symbols:
            self._call("unsubscribe", symbols)

    def _event(self, kind, data):
        self.writer.write(STREAM, "recorder", kind, data)

    def health(self):
        age = (time.time_ns() - self.last_message_ns) / 1e9 if self.last_message_ns else None
        with self._lock:
            return {"connected": self.connected, "desired": len(self.desired),
                    "subscribed": len(self.subscribed), "last_message_age_s": age}
