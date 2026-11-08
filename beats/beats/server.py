"""Siansa Beats: Beat This! for the songs of the chosen Navidrome libraries, with a page to watch it and compare
its readings.

Run with ``python -m beats.server``. Set up with environment variables (see the README):

    NAVIDROME_URL, NAVIDROME_USER, NAVIDROME_PASSWORD   the music server (read only)
    AUDIOMUSE_URL, AUDIOMUSE_TOKEN                       optional: the Siansa plugin's readings, for comparison
    BEATS_MODEL (final0 | small0), BEATS_DEVICE (auto | cpu | cuda | cuda:N), BEATS_FLOAT16, BEATS_THREADS
    BEATS_PASSWORD                                       optional: the page and API ask for it (user "siansa")
    DATA_DIR (/data), MODELS_DIR (/models), PORT (8090)

API (JSON):

    GET  /api/health                  {"status": "ok", "model", "model_loaded"}
    GET  /api/status                  the queue, what's being done, setup and recent activity
    GET  /api/libraries               Navidrome's libraries, each with "on" (analysed) and its songs here and done
    GET  /api/songs                   the songs with their readings and what they say together, with ?find=,
                                      ?show=differ|done|failed and ?limit= (200 when not given, at most 1000)
    GET  /api/songs/<id>              one song, with every beat and downbeat and its tempo along the song
    GET  /api/beats/<id>              just the beat grid (the shape the plugin will read later)
    GET  /api/audio/<id>              the song for the page's player (Navidrome's MP3, passed through)
    GET  /api/export.csv              the table as CSV
    GET  /api/grids?ids=a,b           each song's beat grid at both ends, as the Siansa app reads the plugin's
    GET  /api/export.json             the songs' beats and sound levels, to tune the grid and phrases on (?find=
                                      and ?show= as the table, every song without them; the CSV takes them too)
    POST /api/songs/<id>/marks        {"seconds": 67.4}: the owner heard a section start there
    DELETE /api/songs/<id>/marks/<n>  takes a mark away
    POST /api/songs/<id>/analyse      analyse it again
    POST /api/libraries {"ids"}       analyse these libraries (and no others); read them now, and every 6 hours
    POST /api/libraries/read          read the chosen libraries now, for new songs
    DELETE /api/libraries/<id>/songs  forget the songs of a library that's left out, with what was found in them
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
import urllib.parse
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

    def __init__(self, config, tracker=None, decode=None, measure=None):
        from . import analyser

        self.config = config
        os.makedirs(config.data_dir, exist_ok=True)
        self.store = Store(os.path.join(config.data_dir, "beats.db"))
        self.navidrome = Subsonic(config.navidrome_url, config.navidrome_user, config.navidrome_password)
        self.audiomuse = AudioMuse(config.audiomuse_url, config.audiomuse_token)
        self.tracker = tracker or analyser.BeatThis(config.model, config.device, config.models_dir, config.threads,
                                                    config.float16)
        self.worker = Worker(self.store, self.navidrome, self.audiomuse, self.tracker, decode or analyser.decode,
                             config.data_dir, config.model, measure)
        self.started = time.time()

    # ---- what the API returns ----

    def song_row(self, row):
        row["verdict"] = grid.verdict(row["result"], row["tag_bpm"], row["plugin"]) if row["result"] else []
        return row

    SONGS_SHOWN, SONGS_MOST = 200, 1000  # songs in one answer for the table, unless asked for more (at most)

    def songs(self, find=None, show="all", limit=None):
        """The table's songs: those ``find`` finds, of those to ``show`` (all, "differ" or "failed"), at most
        ``limit`` (``0``: every one, for the exports)."""
        limit = None if limit == 0 else min(int(limit or self.SONGS_SHOWN), self.SONGS_MOST)
        rows = self.store.overview(self.config.model, find, show if show in ("done", "failed") else None,
                                   None if show == "differ" else limit)
        rows = [self.song_row(r) for r in rows]
        if show == "differ":
            rows = [r for r in rows if any(level == "warn" for level, _ in r["verdict"])]
        return rows[:limit] if limit else rows

    def song(self, song_id):
        rows = self.store.overview(self.config.model, song_id=song_id)
        row = rows[0] if rows else None
        if row is None:
            return None
        row = self.song_row(row)
        full = self.store.result(song_id, self.config.model, with_beats=True)
        row["beats"] = full["beats"] if full else []
        row["downbeats"] = full["downbeats"] if full else []
        row["tempo_curve"] = grid.tempo_curve(row["beats"])
        row["grid"] = self.steady_grid(full)
        row["end_seconds"] = grid.END_SECONDS
        row["marks"] = self.store.marks(song_id)
        return row

    def steady_grid(self, full):
        """The even grid of a result with beats, or ``None``: as :func:`beats.grid.analyse` made it."""
        if not full or not full.get("bpm"):
            return None
        beats, bpm, per_bar = full["beats"], full["bpm"], full.get("beats_per_bar") or 4
        return grid.with_sound(grid.steady_grid(beats, full["downbeats"], bpm, per_bar), full.get("levels"))

    def export(self, find=None, show="all"):
        """The analysed songs the table finds (every one without ``find`` and ``show``) with what the grid and
        phrases are worked out from (the model's beats and downbeats, the sound's levels) and what they came to: to
        tune them on real music. Song names and genres, no addresses."""
        out = []
        for row in self.songs(find, show, 0):
            full = self.store.result(row["id"], self.config.model, with_beats=True)
            if not full:
                continue
            out.append({"id": row["id"], "artist": row.get("artist"), "title": row.get("title"),
                        "genres": [g["label"] for g in self.store.genres(row["id"])],
                        "duration": row.get("duration"), "tag_bpm": row.get("tag_bpm"),
                        "result": {k: v for k, v in full.items() if k not in ("beats", "downbeats", "levels")},
                        "beats": full["beats"], "downbeats": full["downbeats"], "levels": full.get("levels"),
                        "marks": [m["seconds"] for m in self.store.marks(row["id"])]})
        return {"version": __version__, "model": self.config.model, "songs": out}

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
                "phrases": steady["phrases"] if steady else [], "downbeat_source": full.get("downbeat_source"),
                "model_beats": full["beats"], "model_downbeats": full["downbeats"]}

    def status(self):
        counts = self.store.counts()
        worker = self.worker
        current = dict(worker.current) if worker.current else None
        if current:
            current["for_seconds"] = round(time.time() - current.pop("since"), 1)
            if current.get("step") == "analysing" and not self.tracker_loaded():
                current["step"] = "loading the model"
        preparing = dict(worker.preparing) if worker.preparing else None
        if preparing:
            preparing["for_seconds"] = round(time.time() - preparing.pop("since"), 1)
        return {
            "version": __version__,
            "counts": counts,
            "current": current,
            "next": preparing,
            "regridding": dict(worker.regridding) if worker.regridding else None,
            "libraries": {"chosen": self.store.setting("folders") or [], "read_at": self.store.setting("library_read_at")},
            "paused": worker.paused,
            "last_error": worker.last_error,
            "model": self.config.model,
            "device": getattr(self.tracker, "device", None) or self.config.device,
            "device_name": getattr(self.tracker, "device_name", None),
            "device_note": getattr(self.tracker, "device_note", None),
            "float16": self.config.float16,
            "model_loaded": self.tracker_loaded(),
            "model_load_seconds": getattr(self.tracker, "load_seconds", None),
            "navidrome": self.config.navidrome_url or None,
            "audiomuse": self.config.audiomuse_url or None,
            "plugin": worker.plugin_state,
            "uptime_seconds": round(time.time() - self.started),
            "events": self.store.events(),
        }

    def tracker_loaded(self):
        return bool(getattr(self.tracker, "loaded", True))

    def csv(self, find=None, show="all"):
        out = io.StringIO()
        columns = ["artist", "title", "album", "genres", "tag_bpm", "plugin_intro_bpm", "plugin_outro_bpm", "bpm",
                   "main_share", "bpm_fit", "grid_bpm", "on_grid", "downbeat_source", "phrase_first_ms", "phrase_bars", "phrase_count",
                   "phrase_big", "phrase_big_on",
                   "intro_bpm", "outro_bpm", "steady", "grid_share", "beats_per_bar", "bar_share", "first_downbeat_ms",
                   "model_seconds", "state", "notes"]
        w = csv.writer(out)
        w.writerow(columns)
        for row in self.songs(find, show, 0):
            r, p = row["result"] or {}, row["plugin"] or {}
            w.writerow([row["artist"], row["title"], row["album"], " / ".join(g["label"] for g in row["genres"]),
                        row["tag_bpm"], p.get("intro_bpm"), p.get("outro_bpm"), _round(r.get("bpm")),
                        _round(r.get("main_share"), 3), _round(r.get("bpm_fit")), _round(r.get("grid_bpm")),
                        _round(r.get("on_grid"), 3), r.get("downbeat_source"), r.get("phrase_first_ms"), r.get("phrase_bars"),
                        r.get("phrase_count"), r.get("phrase_big"), r.get("phrase_big_on"),
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
        query = urllib.parse.parse_qs(self.path.split("?", 1)[1]) if "?" in self.path else {}
        app = self.app
        try:
            if path in ("/", "/index.html"):
                with open(PAGE, "rb") as f:
                    return self._send(HTTPStatus.OK, f.read(), "text/html; charset=utf-8")
            if path == "/api/status":
                return self._json(app.status())
            if path == "/api/libraries":
                return self._json(app.worker.libraries())
            find, show = (query.get("find") or [""])[0], (query.get("show") or ["all"])[0]
            if path == "/api/songs":
                try:
                    limit = int((query.get("limit") or ["0"])[0] or 0)
                except ValueError:
                    return self._error(HTTPStatus.BAD_REQUEST, "limit is a number")
                return self._json(app.songs(find, show, limit or None))
            if path == "/api/grids":
                ids = [i for i in dict.fromkeys(p.strip() for p in (query.get("ids") or [""])[0].split(",")) if i]
                if len(ids) > 2000:
                    return self._error(HTTPStatus.BAD_REQUEST, "at most 2000 ids at a time")
                found = app.store.ends(ids, app.config.model)
                return self._json({"version": __version__, "grids": [{"track_id": i, **found[i]} for i in ids if i in found],
                                   "missing": [i for i in ids if i not in found]})
            if path == "/api/export.json":
                body = json.dumps(app.export(find, show)).encode()
                return self._send(HTTPStatus.OK, body, "application/json",
                                  {"Content-Disposition": 'attachment; filename="siansa-beats-data.json"'})
            if path == "/api/export.csv":
                return self._send(HTTPStatus.OK, app.csv(find, show), "text/csv; charset=utf-8",
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
            if path == "/api/libraries":
                ids = body.get("ids")
                if not isinstance(ids, list):
                    return self._error(HTTPStatus.BAD_REQUEST, 'ids needed: a list of library ids ([] for none)')
                try:
                    return self._json(app.worker.choose(ids))
                except ValueError as e:
                    return self._error(HTTPStatus.BAD_REQUEST, str(e))
            if path == "/api/libraries/read":
                return self._json(app.worker.read_libraries())
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
            m = re.fullmatch(r"/api/songs/([^/]+)/marks", path)
            if m:
                song_id = _id(m[1])
                if not app.store.song(song_id):
                    return self._error(HTTPStatus.NOT_FOUND, "no such song")
                try:
                    seconds = float(body.get("seconds"))
                except (TypeError, ValueError):
                    return self._error(HTTPStatus.BAD_REQUEST, "seconds needed")
                if not 0 <= seconds < 24 * 3600:
                    return self._error(HTTPStatus.BAD_REQUEST, "seconds out of range")
                return self._json({"id": app.store.add_mark(song_id, seconds), "marks": app.store.marks(song_id)})
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
        path = self.path.split("?", 1)[0]
        m = re.fullmatch(r"/api/libraries/([^/]+)/songs", path)
        if m:
            try:
                return self._json({"removed": self.app.worker.remove_library(_id(m[1]))})
            except ValueError as e:
                return self._error(HTTPStatus.CONFLICT, str(e))
            except SubsonicError as e:
                return self._error(HTTPStatus.BAD_GATEWAY, str(e))
        m = re.fullmatch(r"/api/songs/([^/]+)/marks/(\d+)", path)
        if m:
            removed = self.app.store.remove_mark(_id(m[1]), int(m[2]))
            return self._json({"removed": removed, "marks": self.app.store.marks(_id(m[1]))})
        m = re.fullmatch(r"/api/songs/([^/]+)", path)
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
                    f"{' (half precision on a GPU)' if config.float16 else ''}")
    if not app.navidrome.configured:
        app.store.event("Navidrome isn't set up: set NAVIDROME_URL, NAVIDROME_USER and NAVIDROME_PASSWORD", "error")
    else:
        try:
            info = app.navidrome.ping()
            app.store.event(f"Navidrome answers ({info.get('type') or 'server'} {info.get('version') or ''})".strip())
        except SubsonicError as e:
            app.store.event(f"Navidrome doesn't answer: {e}", "error")
    def background():  # after the page is up: a big library takes minutes, and a slow AudioMuse-AI can't hold it up
        app.worker.regrid()
        if app.worker.check_plugin()["state"] == "ok" and app.store.song_ids():
            app.worker.refresh_plugin(app.store.song_ids())

    threading.Thread(target=background, name="start", daemon=True).start()
    server = make_server(app)
    print(f"Siansa Beats {__version__} on port {config.port}", flush=True)
    try:
        server.serve_forever()
    finally:
        app.worker.stop()


if __name__ == "__main__":
    main()
