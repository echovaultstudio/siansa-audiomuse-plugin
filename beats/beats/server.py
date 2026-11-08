"""Siansa Beats: Beat This! for a test library from Navidrome, with a page to watch it and compare its readings.

Run with ``python -m beats.server``. Set up with environment variables (see the README):

    NAVIDROME_URL, NAVIDROME_USER, NAVIDROME_PASSWORD   the music server (read only)
    PLAYLIST                                             the test library's playlist (default "Beat test")
    AUDIOMUSE_URL, AUDIOMUSE_TOKEN                       optional: the Siansa plugin's readings, for comparison
    BEATS_MODEL (final0 | small0), BEATS_DEVICE (auto | cpu | cuda | cuda:N), BEATS_FLOAT16, BEATS_THREADS
    BEATS_PASSWORD                                       optional: the page and API ask for it (user "siansa")
    DATA_DIR (/data), MODELS_DIR (/models), PORT (8090)

API (JSON):

    GET  /api/health                  {"status": "ok", "model", "model_loaded"}
    GET  /api/status                  the queue, what's being done, setup and recent activity
    GET  /api/playlists               Navidrome's playlists
    GET  /api/songs                   every song with its readings and what they say together
    GET  /api/songs/<id>              one song, with every beat and downbeat and its tempo along the song
    GET  /api/beats/<id>              just the beat grid (the shape the plugin will read later)
    GET  /api/audio/<id>              the song for the page's player (Navidrome's MP3, passed through)
    GET  /api/export.csv              the table as CSV
    POST /api/scan {"playlist"}       read the playlist, queue its new songs
    POST /api/songs/<id>/analyse      analyse it again
    POST /api/analyse-all             analyse every song again
    POST /api/plugin/refresh          read the plugin's readings again
    POST /api/pause, /api/resume
    DELETE /api/songs/<id>            forget it (nothing changes on Navidrome)
"""

import base64
import csv
import hmac
import io
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import __version__, grid
from .audiomuse import AudioMuse, AudioMuseError
from .store import Store
from .subsonic import Subsonic, SubsonicError
from .worker import Worker

PAGE = os.path.join(os.path.dirname(__file__), "page.html")
PASSTHROUGH = ("Content-Type", "Content-Length", "Content-Range", "Accept-Ranges")


class Config:
    def __init__(self, env=None):
        env = os.environ if env is None else env
        self.navidrome_url = env.get("NAVIDROME_URL", "")
        self.navidrome_user = env.get("NAVIDROME_USER", "")
        self.navidrome_password = env.get("NAVIDROME_PASSWORD", "")
        self.playlist = env.get("PLAYLIST", "Beat test")
        self.audiomuse_url = env.get("AUDIOMUSE_URL", "")
        self.audiomuse_token = env.get("AUDIOMUSE_TOKEN", "")
        self.model = env.get("BEATS_MODEL", "final0")
        self.device = env.get("BEATS_DEVICE", "auto")
        self.float16 = env.get("BEATS_FLOAT16", "").strip().lower() in ("1", "true", "yes", "on")
        self.threads = int(env.get("BEATS_THREADS", "0") or 0)
        self.password = env.get("BEATS_PASSWORD", "")
        self.data_dir = env.get("DATA_DIR", "/data")
        self.models_dir = env.get("MODELS_DIR", "/models")
        self.port = int(env.get("PORT", "8090"))


class App:
    """Everything the handler needs, so the tests can make one with stand-ins."""

    def __init__(self, config, tracker=None, decode=None):
        from . import analyser

        self.config = config
        os.makedirs(config.data_dir, exist_ok=True)
        self.store = Store(os.path.join(config.data_dir, "beats.db"))
        self.navidrome = Subsonic(config.navidrome_url, config.navidrome_user, config.navidrome_password)
        self.audiomuse = AudioMuse(config.audiomuse_url, config.audiomuse_token)
        self.tracker = tracker or analyser.BeatThis(config.model, config.device, config.models_dir, config.threads,
                                                    config.float16)
        self.worker = Worker(self.store, self.navidrome, self.audiomuse, self.tracker, decode or analyser.decode,
                             config.data_dir, config.model)
        self.started = time.time()

    # ---- what the API returns ----

    def song_row(self, row):
        genres = [g["label"] for g in row["genres"]]
        row["verdict"] = grid.verdict(row["result"], row["tag_bpm"], row["plugin"], genres) if row["result"] else []
        found = grid.genre_range(genres)
        row["genre_range"] = list(found[:2]) if found else None
        return row

    def songs(self):
        return [self.song_row(r) for r in self.store.overview(self.config.model)]

    def song(self, song_id):
        row = next((r for r in self.store.overview(self.config.model) if r["id"] == song_id), None)
        if row is None:
            return None
        row = self.song_row(row)
        full = self.store.result(song_id, self.config.model, with_beats=True)
        row["beats"] = full["beats"] if full else []
        row["downbeats"] = full["downbeats"] if full else []
        row["tempo_curve"] = grid.tempo_curve(row["beats"])
        row["grid"] = self.steady_grid(full)
        row["end_seconds"] = grid.END_SECONDS
        return row

    @staticmethod
    def steady_grid(full):
        """The even grid of a result with beats, or ``None``."""
        if not full or not full.get("bpm"):
            return None
        return grid.steady_grid(full["beats"], full["downbeats"], full["bpm"], full.get("beats_per_bar") or 4)

    def beats(self, song_id):
        """The beat grid as the plugin will read it: the steady grid when the song is at one tempo (``steady``
        true), else the model's own beats."""
        full = self.store.result(song_id, self.config.model, with_beats=True)
        if not full:
            return None
        steady = self.steady_grid(full)
        return {"track_id": song_id, "model": self.config.model, "bpm": full["bpm"], "intro_bpm": full["intro_bpm"],
                "outro_bpm": full["outro_bpm"], "beats_per_bar": full["beats_per_bar"],
                "first_downbeat_ms": full["first_downbeat_ms"], "steady": bool(steady),
                "on_grid": steady["on_grid"] if steady else None,
                "beats": steady["beats"] if steady else full["beats"],
                "downbeats": steady["downbeats"] if steady else full["downbeats"],
                "model_beats": full["beats"], "model_downbeats": full["downbeats"]}

    def status(self):
        counts = self.store.counts()
        worker = self.worker
        current = dict(worker.current) if worker.current else None
        if current:
            current["for_seconds"] = round(time.time() - current.pop("since"), 1)
            if current.get("step") == "analysing" and not self.tracker_loaded():
                current["step"] = "loading the model"
        return {
            "version": __version__,
            "counts": counts,
            "current": current,
            "paused": worker.paused,
            "last_error": worker.last_error,
            "model": self.config.model,
            "device": getattr(self.tracker, "device", None) or self.config.device,
            "device_name": getattr(self.tracker, "device_name", None),
            "device_note": getattr(self.tracker, "device_note", None),
            "float16": self.config.float16,
            "model_loaded": self.tracker_loaded(),
            "model_load_seconds": getattr(self.tracker, "load_seconds", None),
            "playlist": self.config.playlist,
            "navidrome": self.config.navidrome_url or None,
            "audiomuse": self.config.audiomuse_url or None,
            "plugin": worker.plugin_state,
            "uptime_seconds": round(time.time() - self.started),
            "events": self.store.events(),
        }

    def tracker_loaded(self):
        return bool(getattr(self.tracker, "loaded", True))

    def csv(self):
        out = io.StringIO()
        columns = ["artist", "title", "album", "genres", "tag_bpm", "plugin_intro_bpm", "plugin_outro_bpm", "bpm",
                   "main_share", "bpm_fit", "grid_bpm", "on_grid",
                   "intro_bpm", "outro_bpm", "steady", "grid_share", "beats_per_bar", "bar_share", "first_downbeat_ms",
                   "model_seconds", "state", "notes"]
        w = csv.writer(out)
        w.writerow(columns)
        for row in self.songs():
            r, p = row["result"] or {}, row["plugin"] or {}
            w.writerow([row["artist"], row["title"], row["album"], " / ".join(g["label"] for g in row["genres"]),
                        row["tag_bpm"], p.get("intro_bpm"), p.get("outro_bpm"), _round(r.get("bpm")),
                        _round(r.get("main_share"), 3), _round(r.get("bpm_fit")), _round(r.get("grid_bpm")),
                        _round(r.get("on_grid"), 3),
                        _round(r.get("intro_bpm")), _round(r.get("outro_bpm")), _round(r.get("steady"), 3),
                        _round(r.get("grid_share"), 3), r.get("beats_per_bar"), _round(r.get("bar_share"), 3),
                        r.get("first_downbeat_ms"), _round(r.get("model_seconds"), 1), row["state"],
                        "; ".join(text for _, text in row["verdict"])])
        return out.getvalue()


def _round(value, digits=2):
    return None if value is None else round(value, digits)


class Handler(BaseHTTPRequestHandler):
    app = None  # set by make_server
    server_version = "SiansaBeats"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet: the page's activity list says what matters
        pass

    # ---- plumbing ----

    def _authorised(self):
        password = self.app.config.password
        if not password:
            return True
        header = self.headers.get("Authorization", "")
        if header.startswith("Basic "):
            try:
                user, _, given = base64.b64decode(header[6:]).decode().partition(":")
            except (ValueError, UnicodeDecodeError):
                return False
            return hmac.compare_digest(given.encode(), password.encode())
        return False

    def _send(self, status, body=b"", kind="application/json", headers=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, value, status=HTTPStatus.OK):
        self._send(status, value)

    def _error(self, status, message):
        self._json({"error": message}, status)

    def _body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(min(length, 65536)) or b"{}")
        except ValueError:
            return {}

    def _guard(self):
        if self._authorised():
            return True
        self._send(HTTPStatus.UNAUTHORIZED, {"error": "password needed"},
                   headers={"WWW-Authenticate": 'Basic realm="Siansa Beats", charset="UTF-8"'})
        return False

    # ---- routes ----

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        if self.path == "/api/health":  # always open, for Docker's health check
            return self._json({"status": "ok", "version": __version__, "model": self.app.config.model,
                               "model_loaded": self.app.tracker_loaded()})
        if not self._guard():
            return
        path = self.path.split("?", 1)[0]
        app = self.app
        try:
            if path in ("/", "/index.html"):
                with open(PAGE, "rb") as f:
                    return self._send(HTTPStatus.OK, f.read(), "text/html; charset=utf-8")
            if path == "/api/status":
                return self._json(app.status())
            if path == "/api/playlists":
                return self._json(app.navidrome.playlists())
            if path == "/api/songs":
                return self._json(app.songs())
            if path == "/api/export.csv":
                return self._send(HTTPStatus.OK, app.csv(), "text/csv; charset=utf-8",
                                  {"Content-Disposition": 'attachment; filename="siansa-beats.csv"'})
            m = re.fullmatch(r"/api/songs/([^/]+)", path)
            if m:
                song = app.song(_id(m[1]))
                return self._json(song) if song else self._error(HTTPStatus.NOT_FOUND, "no such song")
            m = re.fullmatch(r"/api/beats/([^/]+)", path)
            if m:
                beats = app.beats(_id(m[1]))
                return self._json(beats) if beats else self._error(HTTPStatus.NOT_FOUND, "not analysed")
            m = re.fullmatch(r"/api/audio/([^/]+)", path)
            if m:
                return self._audio(_id(m[1]))
            return self._error(HTTPStatus.NOT_FOUND, "not found")
        except SubsonicError as e:
            return self._error(HTTPStatus.BAD_GATEWAY, str(e))

    def do_POST(self):
        if not self._guard():
            return
        path = self.path.split("?", 1)[0]
        app, body = self.app, self._body()
        try:
            if path == "/api/scan":
                return self._json(app.worker.scan(body.get("playlist") or app.config.playlist))
            if path == "/api/analyse-all":
                queued = sum(app.store.queue(r["id"], force=True) for r in app.store.overview(app.config.model))
                app.store.event(f"Analysing every song again: {queued} queued")
                app.worker.wake()
                return self._json({"queued": queued})
            if path == "/api/plugin/refresh":
                ids = [r["id"] for r in app.store.overview(app.config.model)]
                return self._json({"found": app.worker.refresh_plugin(ids)})
            if path in ("/api/pause", "/api/resume"):
                app.worker.pause(path == "/api/pause")
                return self._json({"paused": app.worker.paused})
            m = re.fullmatch(r"/api/songs/([^/]+)/analyse", path)
            if m:
                song_id = _id(m[1])
                if not app.store.song(song_id):
                    return self._error(HTTPStatus.NOT_FOUND, "no such song")
                queued = app.store.queue(song_id, force=True)
                app.worker.wake()
                return self._json({"queued": queued})
            return self._error(HTTPStatus.NOT_FOUND, "not found")
        except (SubsonicError, AudioMuseError) as e:
            return self._error(HTTPStatus.BAD_GATEWAY, str(e))

    def do_DELETE(self):
        if not self._guard():
            return
        m = re.fullmatch(r"/api/songs/([^/]+)", self.path.split("?", 1)[0])
        if not m:
            return self._error(HTTPStatus.NOT_FOUND, "not found")
        self.app.store.remove(_id(m[1]))
        return self._json({"removed": True})

    def _audio(self, song_id):
        """Navidrome's MP3 of the song (every browser plays it), passed through with seeking (Range) kept."""
        if not self.app.store.song(song_id):
            return self._error(HTTPStatus.NOT_FOUND, "no such song")
        url = self.app.navidrome.address("stream", id=song_id, format="mp3", maxBitRate=320)
        request = urllib.request.Request(url)
        if self.headers.get("Range"):
            request.add_header("Range", self.headers["Range"])
        try:
            upstream = urllib.request.urlopen(request, timeout=60)
        except urllib.error.HTTPError as e:
            return self._error(e.code, "Navidrome didn't send the song")
        except (urllib.error.URLError, OSError) as e:
            return self._error(HTTPStatus.BAD_GATEWAY, f"Navidrome: {getattr(e, 'reason', e)}")
        with upstream:
            self.send_response(upstream.status)
            for name in PASSTHROUGH:
                if upstream.headers.get(name):
                    self.send_header(name, upstream.headers[name])
            if not upstream.headers.get("Content-Length"):
                self.send_header("Connection", "close")
                self.close_connection = True
            self.end_headers()
            if self.command == "HEAD":
                return
            try:
                while chunk := upstream.read(1 << 16):
                    self.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the listener moved on (seeking)


def _id(text):
    import urllib.parse

    return urllib.parse.unquote(text)


def make_server(app, host="0.0.0.0", port=None):
    handler = type("BoundHandler", (Handler,), {"app": app})
    server = ThreadingHTTPServer((host, app.config.port if port is None else port), handler)
    server.daemon_threads = True
    return server


def main():
    config = Config()
    app = App(config)
    app.worker.start()
    app.store.event(f"Started: model {config.model}, device {config.device}"
                    f"{' (half precision on a GPU)' if config.float16 else ''}, playlist \"{config.playlist}\"")
    if not app.navidrome.configured:
        app.store.event("Navidrome isn't set up: set NAVIDROME_URL, NAVIDROME_USER and NAVIDROME_PASSWORD", "error")
    else:
        try:
            info = app.navidrome.ping()
            app.store.event(f"Navidrome answers ({info.get('type') or 'server'} {info.get('version') or ''})".strip())
        except SubsonicError as e:
            app.store.event(f"Navidrome doesn't answer: {e}", "error")
    app.worker.regrid()

    def plugin():  # in the background: a slow AudioMuse-AI doesn't hold up the page
        if app.worker.check_plugin()["state"] == "ok" and app.store.song_ids():
            app.worker.refresh_plugin(app.store.song_ids())

    threading.Thread(target=plugin, name="plugin", daemon=True).start()
    server = make_server(app)
    print(f"Siansa Beats {__version__} on port {config.port}", flush=True)
    try:
        server.serve_forever()
    finally:
        app.worker.stop()


if __name__ == "__main__":
    main()
