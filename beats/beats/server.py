"""Siansa Beats: Beat This! for the songs of the chosen Navidrome libraries, with a page to watch it, its log and
a debug report.

Run with ``python -m beats.server``. Set up with environment variables (see the README):

    NAVIDROME_URL, NAVIDROME_USER, NAVIDROME_PASSWORD   the music server (read only)
    BEATS_MODEL (final0 | small0), BEATS_DEVICE (auto | cpu | cuda | cuda:N), BEATS_FLOAT16, BEATS_THREADS
    BEATS_PASSWORD                                       optional: the page and API ask for it (user "siansa")
    DATA_DIR (/data), MODELS_DIR (/models), PORT (8090)
    STYLE_THREADS (2)                                   the style model's CPU threads
    LOG_LEVEL (info | debug)                             how much the log says at start (the page can change it)

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
    GET  /api/bad-blends              the song changes marked as bad in Siansa Dev, newest first
    GET  /api/bad-blends/export       all of them with both songs' beats, sound levels and results, as a file
    GET  /api/failed/export           the songs that couldn't be analysed, with why, as a file
    GET  /api/genres                  the style model's progress and agreement with the tags, and every style voted
    GET  /api/genre?name=&limit=      the songs a radio for a genre plays (by vote): {"ids", "styles", "genres"}
    POST /api/bad-blends              {"from_id", "to_id", "technique", "details"}: one more (from the app, through
                                      the Siansa Beats (dev) plugin)
    DELETE /api/bad-blends/<n>        takes one away
    GET  /api/log?after=&limit=&level= the latest log lines (numbered; ``after`` for only newer ones), oldest first
    GET  /api/log/download            every kept log line, as a text file
    POST /api/log/level {"level"}     "debug" or "info": how much the log says (kept across restarts)
    GET  /api/debug                   the debug report: versions, setup (no secrets), threads with where each one is,
                                      the queue, the database and the log, as a file to send
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
from .store import Store
from .subsonic import Subsonic, SubsonicError
from .worker import Worker
from . import genres as genres_module
from . import log

_log = log.get("server")

PAGE = os.path.join(os.path.dirname(__file__), "page.html")
PASSTHROUGH = ("Content-Type", "Content-Length", "Content-Range", "Accept-Ranges")


class Config:
    def __init__(self, env=None):
        env = os.environ if env is None else env
        self.navidrome_url = env.get("NAVIDROME_URL", "")
        self.navidrome_user = env.get("NAVIDROME_USER", "")
        self.navidrome_password = env.get("NAVIDROME_PASSWORD", "")
        self.model = env.get("BEATS_MODEL", "final0")
        self.device = env.get("BEATS_DEVICE", "auto")
        self.float16 = env.get("BEATS_FLOAT16", "").strip().lower() in ("1", "true", "yes", "on")
        self.threads = int(env.get("BEATS_THREADS", "0") or 0)
        self.style_threads = int(env.get("STYLE_THREADS", "2") or 2)
        self.password = env.get("BEATS_PASSWORD", "")
        self.data_dir = env.get("DATA_DIR", "/data")
        self.models_dir = env.get("MODELS_DIR", "/models")
        self.port = int(env.get("PORT", "8090"))
        self.log_level = env.get("LOG_LEVEL", "info")
        # Set but no longer used (the plugin comparison went in 0.14.0): said once in the log, so it can be removed.
        self.unused = [k for k in ("AUDIOMUSE_URL", "AUDIOMUSE_TOKEN") if env.get(k)]


class Kept:
    """A value worked out now and then and handed out at once (the owner's phone waited ~10 s for the genres while songs
    were being analysed, 10 Nov). The first ask works it out. Once ``changes()`` says the data changed, or it's older
    than ``seconds``, it's worked out again in a thread, waiting at most ``WAIT`` for it: a quiet database answers in
    time, so the answer is fresh; a busy one doesn't hold the page up, and the last answer is handed out meanwhile."""

    WAIT = 0.5

    def __init__(self, work, seconds, changes=lambda: 0):
        self.work, self.seconds, self.changes = work, seconds, changes
        self.value, self.at, self.seen = None, 0.0, None
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._refreshing = False

    def get(self):
        if self.value is None:
            with self._lock:
                if self.value is None:
                    seen = self.changes()
                    self.value, self.at, self.seen = self.work(), time.monotonic(), seen
            return self.value
        if self.changes() != self.seen or time.monotonic() - self.at > self.seconds:
            self.refresh(wait=self.WAIT)
        return self.value

    def refresh(self, wait=0.0):
        """Works it out again in a thread (one at a time), waiting at most ``wait`` seconds for it."""
        with self._lock:
            if not self._refreshing:
                self._refreshing = True
                self._done.clear()

                def run():
                    try:
                        seen = self.changes()
                        value = self.work()
                        self.value, self.at, self.seen = value, time.monotonic(), seen
                    except Exception:
                        _log.exception("Working out a kept answer went wrong; the last one stays")
                    finally:
                        self._refreshing = False
                        self._done.set()

                threading.Thread(target=run, name="beats-kept", daemon=True).start()
        if wait:
            self._done.wait(wait)


class App:
    """Everything the handler needs, so the tests can make one with stand-ins."""

    def __init__(self, config, tracker=None, decode=None, measure=None, style=None):
        from . import analyser
        from . import style as style_module

        self.config = config
        os.makedirs(config.data_dir, exist_ok=True)
        self.store = Store(os.path.join(config.data_dir, "beats.db"))
        self.navidrome = Subsonic(config.navidrome_url, config.navidrome_user, config.navidrome_password)
        self.tracker = tracker or analyser.BeatThis(config.model, config.device, config.models_dir, config.threads,
                                                    config.float16)
        self.style = style if style is not None else style_module.StyleModel(config.models_dir, config.style_threads, config.device)
        self.worker = Worker(self.store, self.navidrome, self.tracker, decode or analyser.decode,
                             config.data_dir, config.model, measure, self.style)
        self.started = time.time()
        # Answers the page and the app ask for often, kept ready: the genre list changes only when songs are voted,
        # the styles' counts with each song heard (the page asks every 2 s).
        changes = lambda: self.store.genre_changes  # noqa: E731
        self.kept_genres = Kept(self._genre_list, 60, changes)
        self.kept_style_counts = Kept(self.store.style_counts, 30, changes)  # its total follows songs added, too
        self.worker.on_voted = lambda: self.kept_genres.refresh()

    # ---- what the API returns ----

    def song_row(self, row):
        row["verdict"] = grid.verdict(row["result"], row["tag_bpm"]) if row["result"] else []
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
        out = [s for s in (self.song_data(row["id"], row) for row in self.songs(find, show, 0)) if s]
        return {"version": __version__, "model": self.config.model, "songs": out}

    def song_data(self, song_id, row=None):
        """One song as the data export has it (its beats, downbeats, sound levels, results and marks), or None when
        it isn't analysed."""
        full = self.store.result(song_id, self.config.model, with_beats=True)
        if not full:
            return None
        row = row or self.store.song(song_id) or {}
        return {"id": song_id, "artist": row.get("artist"), "title": row.get("title"),
                "genres": [g["label"] for g in self.store.genres(song_id)],
                "duration": row.get("duration"), "tag_bpm": row.get("tag_bpm"),
                "result": {k: v for k, v in full.items() if k not in ("beats", "downbeats", "levels")},
                "beats": full["beats"], "downbeats": full["downbeats"], "levels": full.get("levels"),
                "marks": [m["seconds"] for m in self.store.marks(song_id)]}

    def failed_export(self):
        """Every song that couldn't be analysed, with why and what kind of file it is: one file to look into them with."""
        rows = self.store.overview(self.config.model, None, "failed", None)
        return {"version": __version__, "model": self.config.model, "failed": [
            {"id": r["id"], "artist": r.get("artist"), "title": r.get("title"), "album": r.get("album"),
             "suffix": r.get("suffix"), "duration": r.get("duration"), "library": r.get("folder"),
             "error": r.get("error"), "failed_at": r.get("finished_at")} for r in rows]}

    def bad_blends_export(self):
        """Every bad blend with both songs' data (:meth:`song_data`): one file to look into them with."""
        blends = self.store.bad_blends(limit=1000)
        ids = {i for b in blends for i in (b["from_id"], b["to_id"])}
        songs = {i: self.song_data(i) for i in ids}
        return {"version": __version__, "model": self.config.model, "bad_blends": [
            {**b, "from": songs.get(b["from_id"]), "to": songs.get(b["to_id"])} for b in blends]}

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
            "bad_blends": self.store.bad_blend_count(),
            "styles": self.styles_status(),
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
            "log_level": log.level(),
            "uptime_seconds": round(time.time() - self.started),
            "events": self.store.events(),
        }

    def styles_status(self):
        """The style model: whether the image has it, how far the songs are, and how often it agrees with the tags."""
        worker, style = self.worker, self.style
        available = bool(style is not None and style.available)
        out = {"available": available, "loaded": bool(available and style.loaded),
               "device": getattr(style, "backend", None), "problem": getattr(style, "problem", None)}
        if available:
            out.update(self.kept_style_counts.get())
            out["agreement"] = worker.genre_report
            styling = dict(worker.styling) if worker.styling else None
            if styling:
                styling["for_seconds"] = round(time.time() - styling.pop("since"), 1)
            out["current"] = styling
        return out

    def genre_list(self):
        """Every style with songs voted into it, most first, kept ready (:class:`Kept`): ``[{"label", "genre", "style",
        "songs", "name"}]``."""
        return self.kept_genres.get()

    def _genre_list(self):
        counts = self.store.vote_counts(genres_module.RADIO_MIN)
        ranked = sorted(counts.items(), key=lambda p: -p[1])
        named = {}
        for label, _ in ranked:
            named.setdefault(label.partition("---")[2] or label, set()).add(label.partition("---")[0])
        out = []
        for label, n in ranked:
            genre, _, style = label.partition("---")
            style = style or label
            # "name": what a radio asks for: the style, or "Genre / Style" when two genres share its name.
            out.append({"label": label, "genre": genre, "style": style, "songs": n,
                        "name": f"{genre} / {style}" if len(named[style]) > 1 else style})
        return out

    def genre_songs(self, name, limit=300):
        """The songs a radio for the genre ``name`` plays (a tag as the app shows it: "Trance", "Electronic /
        Trance"), by vote: ``{"name", "styles", "genres", "ids"}``. Nothing when the style model isn't there or the
        name matches none of its styles, so the app falls back to the tags."""
        taxonomy = self.worker.taxonomy()
        if taxonomy is None:
            return {"name": name, "styles": [], "genres": [], "ids": [], "why": "no style model in this image"}
        styles, families = genres_module.target(taxonomy, name)
        labels = set(styles)
        for f in families:
            labels |= set(taxonomy.by_genre.get(genres_module.key(f), []))
        ids = self.store.songs_voted(labels, genres_module.RADIO_MIN, limit)
        return {"name": name, "styles": sorted(styles), "genres": sorted(families), "ids": ids,
                "why": None if labels else "no style of that name"}

    def debug_report(self):
        """Everything that helps find out why something's wrong, in one file: versions, the setup (addresses, never
        passwords or tokens), where each thread is right now (a stuck one shows where), the status, the queue's
        states, the database's size and the log."""
        import platform
        import sys
        import traceback

        threads = {t.ident: t for t in threading.enumerate()}
        stacks = {}
        for ident, frame in sys._current_frames().items():
            thread = threads.get(ident)
            stacks[thread.name if thread else str(ident)] = "".join(traceback.format_stack(frame)[-12:])
        versions = {"siansa_beats": __version__, "python": sys.version.split()[0], "platform": platform.platform()}
        for name in ("torch", "numpy", "beat_this", "soxr", "soundfile"):
            module = sys.modules.get(name)
            if module is not None:
                versions[name] = getattr(module, "__version__", "?")
        torch = sys.modules.get("torch")
        if torch is not None:
            try:
                versions["cuda"] = torch.version.cuda
                versions["cuda_available"] = torch.cuda.is_available()
                if torch.cuda.is_available():
                    versions["gpu"] = torch.cuda.get_device_name(0)
            except Exception as e:
                versions["cuda"] = f"? ({e})"
        c = self.config
        setup = {"navidrome_url": c.navidrome_url or None, "navidrome_user": c.navidrome_user or None,
                 "navidrome_password": "set" if c.navidrome_password else "not set", "model": c.model,
                 "device": c.device, "float16": c.float16, "threads": c.threads,
                 "page_password": "set" if c.password else "not set", "data_dir": c.data_dir,
                 "models_dir": c.models_dir, "port": c.port, "log_level": log.level(), "unused": c.unused}
        db = os.path.join(c.data_dir, "beats.db")
        status = self.status()
        status.pop("events", None)
        report = {"made_at": time.strftime("%Y-%m-%d %H:%M:%S"), "versions": versions, "setup": setup,
                  "status": status,
                  "database_mb": round(os.path.getsize(db) / 1e6, 1) if os.path.exists(db) else None,
                  "threads": stacks, "activity": self.store.events(200), "log": log.lines(limit=2000)}
        return json.loads(log.redact(json.dumps(report, default=str)))

    def tracker_loaded(self):
        return bool(getattr(self.tracker, "loaded", True))

    def csv(self, find=None, show="all"):
        out = io.StringIO()
        columns = ["artist", "title", "album", "genres", "voted_genres", "sound_top", "tag_bpm", "bpm",
                   "main_share", "bpm_fit", "grid_bpm", "on_grid", "grid_drift", "downbeat_source", "phrase_first_ms", "phrase_bars", "phrase_count",
                   "phrase_big", "phrase_big_on",
                   "intro_bpm", "outro_bpm", "steady", "grid_share", "beats_per_bar", "bar_share", "first_downbeat_ms",
                   "model_seconds", "state", "notes", "error"]
        w = csv.writer(out)
        w.writerow(columns)
        for row in self.songs(find, show, 0):
            r = row["result"] or {}
            by = {}
            for g in row["genres"]:
                by.setdefault(g["source"], []).append(g)
            w.writerow([row["artist"], row["title"], row["album"], " / ".join(g["label"] for g in by.get("tag", [])),
                        " / ".join(f"{g['label'].partition('---')[2]} {g['weight']:.2f}" for g in by.get("vote", [])),
                        " / ".join(f"{g['label'].partition('---')[2]} {g['weight']:.2f}" for g in by.get("sound", [])[:3]),
                        row["tag_bpm"], _round(r.get("bpm")),
                        _round(r.get("main_share"), 3), _round(r.get("bpm_fit")), _round(r.get("grid_bpm")),
                        _round(r.get("on_grid"), 3), _round(r.get("grid_drift"), 4), r.get("downbeat_source"), r.get("phrase_first_ms"), r.get("phrase_bars"),
                        r.get("phrase_count"), r.get("phrase_big"), r.get("phrase_big_on"),
                        _round(r.get("intro_bpm")), _round(r.get("outro_bpm")), _round(r.get("steady"), 3),
                        _round(r.get("grid_share"), 3), r.get("beats_per_bar"), _round(r.get("bar_share"), 3),
                        r.get("first_downbeat_ms"), _round(r.get("model_seconds"), 1), row["state"],
                        "; ".join(text for _, text in row["verdict"]), row.get("error")])
        return out.getvalue()


def _round(value, digits=2):
    return None if value is None else round(value, digits)


class Handler(BaseHTTPRequestHandler):
    app = None  # set by make_server
    server_version = "SiansaBeats"
    protocol_version = "HTTP/1.1"

    # The page asks for these every few seconds: logged only when they fail, so the debug log stays readable.
    POLLED = ("/api/status", "/api/songs", "/api/libraries", "/api/log", "/api/bad-blends", "/api/health")

    def log_message(self, fmt, *args):  # http.server's own complaints (a malformed request); requests: log_request
        _log.debug("%s: %s", self.client_address[0], log.redact(fmt % args if args else fmt))

    SLOW = 1.0  # seconds: a request slower than this is logged as a warning, with where the time went

    def parse_request(self):
        self._started = time.monotonic()
        return super().parse_request()

    def log_request(self, code="-", size="-"):
        path = self.path.split("?", 1)[0]
        took = time.monotonic() - getattr(self, "_started", time.monotonic())
        if took > self.SLOW:
            worker = self.app.worker if self.app else None
            busy = (worker.current or {}).get("step") if worker else None
            _log.warning("%s %s took %.1fs%s", self.command, log.redact(path), took,
                         f" (busy {busy})" if busy else "")
        code = int(getattr(code, "value", code)) if str(getattr(code, "value", code)).isdigit() else 0
        if code >= 500:
            _log.warning("%s %s -> %s", self.command, log.redact(self.path), code)
        elif code >= 400 and path not in ("/favicon.ico",):
            _log.info("%s %s -> %s", self.command, log.redact(self.path), code)
        elif self.command != "GET" or not path.startswith(self.POLLED):
            _log.debug("%s %s -> %s", self.command, log.redact(self.path), code)

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
            if path == "/api/genres":
                return self._json({"styles": app.styles_status(), "genres": app.genre_list()})
            if path == "/api/genre":
                name = (query.get("name") or [""])[0].strip()
                if not name:
                    return self._error(HTTPStatus.BAD_REQUEST, "name needed: the genre, as a tag (Trance)")
                try:
                    limit = max(1, min(2000, int((query.get("limit") or ["300"])[0])))
                except ValueError:
                    return self._error(HTTPStatus.BAD_REQUEST, "limit is a number")
                return self._json(app.genre_songs(name, limit))
            if path == "/api/status":
                return self._json(app.status())
            if path == "/api/log":
                def number(key, default):
                    try:
                        return max(0, int((query.get(key) or [default])[0]))
                    except ValueError:
                        return default
                least = (query.get("level") or ["debug"])[0]
                return self._json({"level": log.level(), "lines": log.lines(number("after", 0), min(number("limit", 300), 5000), least)})
            if path == "/api/log/download":
                stamp = time.strftime("%Y%m%d-%H%M")
                return self._send(HTTPStatus.OK, log.text(), "text/plain; charset=utf-8",
                                  {"Content-Disposition": f'attachment; filename="siansa-beats-{stamp}.log"'})
            if path == "/api/debug":
                stamp = time.strftime("%Y%m%d-%H%M")
                return self._send(HTTPStatus.OK, json.dumps(app.debug_report(), indent=1, default=str), "application/json",
                                  {"Content-Disposition": f'attachment; filename="siansa-beats-debug-{stamp}.json"'})
            if path == "/api/libraries":
                return self._json(app.worker.libraries())
            if path == "/api/bad-blends":
                return self._json(app.store.bad_blends())
            if path == "/api/failed/export":
                body = json.dumps(app.failed_export(), indent=1).encode()
                return self._send(HTTPStatus.OK, body, "application/json",
                                  {"Content-Disposition": 'attachment; filename="siansa-beats-failed.json"'})
            if path == "/api/bad-blends/export":
                body = json.dumps(app.bad_blends_export()).encode()
                return self._send(HTTPStatus.OK, body, "application/json",
                                  {"Content-Disposition": 'attachment; filename="siansa-bad-blends.json"'})
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
            if path == "/api/bad-blends":
                from_id, to_id = str(body.get("from_id") or "").strip(), str(body.get("to_id") or "").strip()
                if not from_id or not to_id:
                    return self._error(HTTPStatus.BAD_REQUEST, "from_id and to_id needed: the two songs")
                details = body.get("details") if isinstance(body.get("details"), dict) else {}
                technique = str(body.get("technique") or "")[:80]
                blend_id = app.store.add_bad_blend(from_id, to_id, technique, details)
                names = {s: (app.store.song(s) or {}).get("title", s) for s in (from_id, to_id)}
                app.store.event(f"Bad blend marked in the app: {names[from_id]} into {names[to_id]}"
                                + (f" ({technique})" if technique else ""), "warn")
                return self._json({"id": blend_id})
            if path == "/api/libraries/read":
                return self._json(app.worker.read_libraries())
            if path == "/api/analyse-all":
                queued = sum(app.store.queue(r["id"], force=True) for r in app.store.overview(app.config.model))
                app.store.event(f"Analysing every song again: {queued} queued")
                app.worker.wake()
                return self._json({"queued": queued})
            if path == "/api/log/level":
                if str(body.get("level") or "").lower() not in ("debug", "info"):
                    return self._error(HTTPStatus.BAD_REQUEST, 'level needed: "debug" or "info"')
                name = log.set_level(body["level"])
                app.store.set_setting("log_level", name)
                app.store.event("Debug details on: the log says every step" if name == "debug" else "Debug details off")
                return self._json({"level": name})
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
        except SubsonicError as e:
            return self._error(HTTPStatus.BAD_GATEWAY, str(e))

    def do_DELETE(self):
        if not self._guard():
            return
        path = self.path.split("?", 1)[0]
        m = re.fullmatch(r"/api/bad-blends/(\d+)", path)
        if m:
            if not self.app.store.remove_bad_blend(int(m[1])):
                return self._error(HTTPStatus.NOT_FOUND, "no such bad blend")
            return self._json({"removed": True})
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


class _Server(ThreadingHTTPServer):
    def handle_error(self, request, client_address):  # a request that broke: the traceback goes to the log
        import sys

        who = client_address[0] if client_address else "?"
        if isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError, TimeoutError)):
            # The other end stopped listening (a health check that gave up, a page closed): nothing broke here.
            _log.debug("%s left before its answer was sent: %s", who, sys.exc_info()[1])
            return
        _log.exception("A request from %s went wrong", who)


def make_server(app, host="0.0.0.0", port=None):
    handler = type("BoundHandler", (Handler,), {"app": app})
    server = _Server((host, app.config.port if port is None else port), handler)
    server.daemon_threads = True
    return server


def main():
    config = Config()
    log.setup(config.log_level)
    # A thread that dies says why in the log, not only on a console nobody reads.
    threading.excepthook = lambda a: _log.error("The thread %s stopped", a.thread.name if a.thread else "?",
                                                exc_info=(a.exc_type, a.exc_value, a.exc_traceback))
    _log.info("Siansa Beats %s starting: model %s, device %s, data in %s", __version__, config.model, config.device,
              config.data_dir)
    app = App(config)
    chosen = app.store.setting("log_level")  # the page's switch wins over LOG_LEVEL
    if chosen:
        log.set_level(chosen)
    _log.info("Log level %s (%s)", log.level(), "chosen on the page" if chosen else "LOG_LEVEL")
    if config.unused:
        _log.info("%s set but no longer used (the plugin comparison went in 0.14.0): it can be removed",
                  " and ".join(config.unused))
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
    def background():  # after the page is up: a big library takes minutes
        try:
            app.worker.regrid()
        except Exception:
            _log.exception("Working out older results again went wrong")
        try:  # every song's genres voted again: the vote's rules or the tags may have changed
            voted = app.worker.revote()
            if voted:
                report = app.worker.genre_report or {}
                app.store.event(f"Voted the genres of {voted} songs" + (
                    f" · the sound agrees with the tags' genre on {report['family'] * 100:.0f} % of {report['songs']}"
                    if report.get("family") is not None else ""))
        except Exception:
            _log.exception("Voting the genres went wrong")

    threading.Thread(target=background, name="start", daemon=True).start()
    server = make_server(app)
    _log.info("Siansa Beats %s on port %s", __version__, config.port)
    try:
        server.serve_forever()
    finally:
        app.worker.stop()


if __name__ == "__main__":
    main()
