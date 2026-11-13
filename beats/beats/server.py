"""Siansa Beats: Beat This! for the songs of the chosen libraries of your music servers (Navidrome, Emby, Jellyfin),
with a page to watch it, its log and a debug report.

Run with ``python -m beats.server``. The connections are made on the page (Setup): the music servers, AudioMuse-AI,
the page's password and the app's key; none of their secrets is kept as plain text or in the container's YAML
(:mod:`beats.vault`). What's left for the container's environment (see the README):

    BEATS_MODEL (final0 | small0), BEATS_DEVICE (auto | cpu | cuda | cuda:N), BEATS_FLOAT16, BEATS_THREADS
    DATA_DIR (/data), MODELS_DIR (/models), PORT (8090)
    STYLE_THREADS (2)                                   the style model's CPU threads
    LOG_LEVEL (info | debug)                             how much the log says at start (the page can change it)
    BEATS_SECRET_KEY                                     optional: the key the kept secrets are encrypted with, to keep
                                                         it out of the data folder (else a key file there)
    BEATS_RESET_ACCESS (yes)                             takes the page's password and the app's key away at the start:
                                                         the way back in when the password is forgotten

    NAVIDROME_URL, NAVIDROME_USER, NAVIDROME_PASSWORD, AUDIOMUSE_URL, AUDIOMUSE_TOKEN, BEATS_PASSWORD, BEATS_API_KEY
        how it was set up until 0.24: each is taken over once (hashed or encrypted) and not read again, so they can go
        from the YAML; the page says which are still there

The Siansa app's API is :mod:`beats.api_v1` (``/api/v1/…``, described in ``beats/openapi.json``). The rest, for the
page (JSON):

    GET  /api/health                  {"status": "ok", "model", "model_loaded"}
    GET  /api/status                  the queue, what's being done, setup and recent activity
    GET  /api/servers                 the music servers, each with its songs here (never a password)
    POST /api/servers                 {"kind", "name", "url", "user", "password"}: one more (kind: navidrome, emby
                                      or jellyfin), once it answers and takes the login
    POST /api/servers/test            the same fields: asks the server and says its version and libraries, adds nothing
    POST /api/servers/<id>            {"name", "url", "user", "password"}: changes one (an empty field stays)
    POST /api/servers/<id>/default    makes it the default: read first, and meant when a request names no server
    DELETE /api/servers/<id>          takes it away, with the songs only it had
    GET  /api/libraries               the server's libraries, each with "on" (analysed) and its songs here and done
    POST /api/audiomuse               {"url", "token"}: connects AudioMuse-AI once its Siansa plugin answers (an empty
                                      token keeps the one there is); /api/audiomuse/test asks without keeping
    DELETE /api/audiomuse             disconnects it
    POST /api/access/password         {"password"}: sets the page's password (8 characters or more); only its hash is kept
    DELETE /api/access/password       takes it away
    POST /api/access/key              makes a new key for the app: {"key"}, said this once (its hash is kept)
    DELETE /api/access/key            takes it away

What's asked about a server's songs (status, libraries, songs, genres, loudness, the exports) takes ``?server=``, its
id or name: the default server's without it.

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
    GET  /api/genres                  the style model's progress and agreement with the tags, and every style heard
    GET  /api/genre?name=&limit=      the songs of a genre, by their sound: {"ids", "styles", "genres", "songs"}
    POST /api/bad-blends              {"from_id", "to_id", "technique", "details"}: one more (from the app, through
                                      the Siansa Beats (dev) plugin)
    DELETE /api/bad-blends/<n>        takes one away
    DELETE /api/bad-blends            takes them all away (a fresh start): {"removed": n}
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

import numpy as np
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import gzip

from . import __version__, api_v1, grid, loudness
from .audiomuse import AudioMuse, AudioMuseError
from .checks import Checks
from .servers import KINDS, MAIN, Servers, folders_key
from .store import Store
from .vault import Vault
from .subsonic import ServerError, SubsonicError
from .worker import Worker
from . import genres as genres_module
from . import log
from . import memory as memory_module

_log = log.get("server")

# A song scores a style this or more: the sound is sure of it (for the log and the app's radio, which prefers these).
SURE = 0.4

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
        # Taken over once into the data folder, hashed (0.24.0): the page sets them since (Setup › Access).
        self.password = env.get("BEATS_PASSWORD", "")
        self.api_key = env.get("BEATS_API_KEY", "")
        # The key the secrets Siansa Beats has to say to other servers are encrypted with, kept out of the data folder;
        # without it, a key file there. And the way back in when the page's password is forgotten.
        self.secret_key = env.get("BEATS_SECRET_KEY", "")
        self.reset_access = env.get("BEATS_RESET_ACCESS", "").strip().lower() in ("1", "true", "yes", "on")
        # The names of the secrets the container still has, for the page to say they can go from the YAML.
        self.secrets_left = [k for k in ("NAVIDROME_PASSWORD", "AUDIOMUSE_TOKEN", "BEATS_PASSWORD", "BEATS_API_KEY")
                             if env.get(k)]
        self.data_dir = env.get("DATA_DIR", "/data")
        self.models_dir = env.get("MODELS_DIR", "/models")
        self.port = int(env.get("PORT", "8090"))
        self.log_level = env.get("LOG_LEVEL", "info")
        # AudioMuse-AI, for similar albums and artists (0.20.0; until 0.14.0 it was for comparing with the plugin).
        self.audiomuse_url = env.get("AUDIOMUSE_URL", "")
        self.audiomuse_token = env.get("AUDIOMUSE_TOKEN", "")


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


def _remove(path):
    try:
        os.remove(path)
    except OSError:
        pass


class App:
    """Everything the handler needs, so the tests can make one with stand-ins."""

    def __init__(self, config, tracker=None, decode=None, measure=None, style=None, sound=None, words=None,
                 hearer=None, style_factory=None):
        from . import analyser
        from . import style as style_module

        self.config = config
        os.makedirs(config.data_dir, exist_ok=True)
        self.store = Store(os.path.join(config.data_dir, "beats.db"))
        self.vault = Vault(config.data_dir, config.secret_key)
        self.servers = Servers(self.store, config, self.vault)
        self._take_from_container()
        saved = self.store.setting("audiomuse") or {}
        self.audiomuse = AudioMuse(saved.get("url"), self.vault.open(saved.get("token")))
        self.tracker = tracker or analyser.BeatThis(config.model, config.device, config.models_dir, config.threads,
                                                    config.float16)
        # The style model (0.39): the one chosen in Setup, Discogs-EffNet unless MAEST was picked (and is in the image).
        chosen = self.store.setting("style_model") or style_module.DEFAULT_MODEL
        spec = style_module.MODELS.get(chosen, style_module.MODELS[style_module.DEFAULT_MODEL])
        if not spec.available(config.models_dir) and style is None:
            spec = style_module.MODELS[style_module.DEFAULT_MODEL]
        self.style_factory = style_factory or (lambda chosen_spec: style_module.StyleModel(config.models_dir, config.style_threads, config.device, chosen_spec))
        self.style = style if style is not None else self.style_factory(spec)
        # Music found by describing it in words (0.26): the text model, loaded when first searched.
        from .hear import Hearer
        from .words import Words

        self.words = words if words is not None else Words(config.models_dir)
        self._families = None  # beats.families.Families, made when first asked
        # How a song sounds, heard here with AudioMuse-AI's models (0.27): new songs need no AudioMuse-AI.
        self.hearer = hearer if hearer is not None else Hearer(config.models_dir, self.words, config.threads)
        # How energetic each song is (0.35, beats.drive): from the sound kept for it; nothing without the model.
        from .drive import Arousal

        self.arousal = Arousal(config.models_dir, self.store, config.threads)
        self.worker = Worker(self.store, self.servers, self.tracker, decode or analyser.decode,
                             config.data_dir, config.model, measure, self.style, sound, self.hearer)
        self.started = time.time()
        # Answers the page and the app ask for often, kept ready: the genre list and the styles' counts change with
        # each song heard (the page asks every 2 s).
        # One of each for every server looked at (its songs alone); with one server, for all the songs.
        self._kept, self._kept_lock = {}, threading.Lock()
        from . import about as about_module

        self.about = about_module.Keeper(self.store, self.about_names)
        # What the Siansa plugin already measured needn't be downloaded again (asked of whichever AudioMuse-AI is set).
        self.worker.importer = lambda ids: self.audiomuse.profiles(ids)
        # What songs sound like (0.25): taken over from AudioMuse-AI's export, and answered from here.
        from .pull import Puller
        from .sound import Sound

        self.sound = Sound(self.store)
        self.puller = Puller(self.store, lambda: self.audiomuse)
        # What was listened to (0.29): from the app as it happens, from exports, and on to Last.fm.
        from .listens import Listens

        self.listens = Listens(self.store, self.vault)
        # Covers and artists' pictures for the listening page (0.30): the music server's own, kept in the data folder.
        from .art import Art

        self.art = Art(self.store, self.servers, config.data_dir)


    PULL_EVERY = 6 * 3600  # AudioMuse-AI's new songs are taken over this often

    def pull_now_and_then(self):
        """Pulls from AudioMuse-AI a minute after the start and every :data:`PULL_EVERY` after (while it's connected)."""
        def loop():
            time.sleep(60)
            while True:
                if self.audiomuse.configured:
                    self.puller.start()
                # The text model goes when nobody has searched for a while (looked at every minute).
                for _ in range(self.PULL_EVERY // 60):
                    time.sleep(60)
                    self.mind_memory()

        threading.Thread(target=loop, name="sound-pull-timer", daemon=True).start()

    def mind_memory(self):
        """Every minute: the text model goes when nobody has searched for a while, the models that analyse when
        there has been no song for a while (beats.memory); what they held goes back to the system."""
        from . import memory

        try:
            if self.words.unload_if_idle():
                memory.trim()
            self.worker.let_models_go_if_idle()
            if self.arousal.catch_up():  # songs heard or pulled since the last minute
                memory.trim()
        except Exception as e:  # never the timer's end
            _log.debug("Minding the memory failed: %s", e)

    RELAY_EVERY = 30  # seconds between tries to send waiting listens on to Last.fm

    def relay_now_and_then(self):
        """Sends listens on to Last.fm in the background, for the listeners who connected it."""
        def loop():
            while True:
                time.sleep(self.RELAY_EVERY)
                try:
                    self.listens.relay()
                except Exception:  # nothing here may take the service down
                    _log.exception("Sending listens on to Last.fm went wrong")

        threading.Thread(target=loop, name="lastfm-relay", daemon=True).start()

    def default_listener(self):
        """Whose listening shows when no one is asked for: the first one with a name (the app's), else the first."""
        users = self.listens.users()
        return next((u["user"] for u in users if u["user"]), users[0]["user"] if users else "")

    def listening(self, user=None, year=None, month=None, tz=None):
        """The listening page: the listeners, and one's numbers for a month, a year or all time (the first listener's
        without one)."""
        users = self.listens.users()
        if user is None:  # the first one with a name (the app's), else the first
            user = next((u["user"] for u in users if u["user"]), users[0]["user"] if users else "")
        answer = {"users": users, "importing": dict(self.listens.importing), "playing": self.listens.all_playing(),
                  **self.listens.summary(user, year, month, tz)}
        # Which song's cover and which song's artist picture each row shows ("cover", "face": a song id here, or
        # null for music that isn't in the library): the page asks /api/art for them as it shows them.
        def pictured(row, cover=True, face=False):
            row = dict(row)
            if cover:
                row["cover"] = self.art.song_for("cover", row.get("song"), row.get("artist"), row.get("album"))
            if face:
                row["face"] = self.art.song_for("artist", row.get("song"), row.get("artist"))
            return row

        top = answer["top"]
        answer["top"] = {"songs": [pictured(r) for r in top["songs"]], "albums": [pictured(r) for r in top["albums"]],
                         "artists": [pictured(r, cover=False, face=True) for r in top["artists"]]}
        answer["recent"] = [pictured(r) for r in answer["recent"]]
        answer["playing"] = [pictured(r, face=True) for r in answer["playing"]]
        answer["new_artists"] = [pictured(r, cover=False, face=True) for r in answer["new_artists"]]
        return answer

    def listening_artist(self, user, name, tz=None):
        """One artist's page in ``user``'s listening, with the pictures each row shows; None when never listened to."""
        if user is None:  # opened straight from its address: the same person Listening shows first
            user = self.default_listener()
        found = self.listens.artist(user, name, tz)
        if not found:
            return None
        found["face"] = self.art.song_for("artist", found["song"], found["artist"])
        for kind in ("songs", "albums", "recent"):
            found[kind] = [dict(r, cover=self.art.song_for("cover", r.get("song"), r.get("artist"), r.get("album")))
                           for r in found[kind]]
        return found

    def sound_status(self, server=None):
        """How much of ``server``'s music has its sound here, and how the pull from AudioMuse-AI is going."""
        # Kept ready (0.35.1): counting 45,000 sounds reads each one's row, vectors and all, and the page asks every
        # few seconds. Worked out again once anything was written, and at most a minute old.
        server = server or self.servers.default_id
        counts = self.kept("sound-counts", server, lambda only: self.store.sound_counts(only), 60, lambda: self.store.writes).get()
        sources = self.kept("sound-sources", server, lambda only: self.store.sound_sources(only), 60, lambda: self.store.writes).get()
        return {**counts, "pull": dict(self.puller.state), "audiomuse": self.audiomuse.configured,
                "text_model": {"available": self.words.available, "steering": self.words.steers,
                               "loaded": self.words.loaded, "load_seconds": self.words.load_seconds},
                # Heard here (0.27): whether this image can, where each song's sound came from, what's being heard.
                "hears": self.hearer.available, "sources": sources,
                "hearing": dict(self.worker.hearing) if self.worker.hearing else None}

    # ---- connections and access, set on the page (0.24.0) ----

    def _take_from_container(self):
        """Once: what the container's settings said until 0.24 (``BEATS_PASSWORD``, ``BEATS_API_KEY``,
        ``AUDIOMUSE_URL``, ``AUDIOMUSE_TOKEN``) goes into the data folder, hashed or encrypted, and isn't read again.
        ``BEATS_RESET_ACCESS`` takes the page's password and the app's key away: the way back in."""
        c, store = self.config, self.store
        if c.reset_access and store.setting("access"):
            store.set_setting("access", {})
            store.event("The page's password and the app's key were taken away (BEATS_RESET_ACCESS): take that setting "
                        "off the container again and set new ones under Setup", "warn")
        taken = store.setting("container_taken_access") or []
        access = dict(store.setting("access") or {})
        if c.password and "password" not in taken:
            access["password"] = Vault.hash_password(c.password)
        if c.api_key and "api_key" not in taken:
            access["api_key"] = Vault.hash_key(c.api_key)
        if access != (store.setting("access") or {}):
            store.set_setting("access", access)
        if c.audiomuse_url and "audiomuse" not in taken:
            store.set_setting("audiomuse", {"url": c.audiomuse_url.rstrip("/"), "token": self.vault.seal(c.audiomuse_token)})
        now = sorted(set(taken) | {name for name, value in (("password", c.password), ("api_key", c.api_key),
                                                              ("audiomuse", c.audiomuse_url)) if value})
        if now != taken:
            store.set_setting("container_taken_access", now)

    def access(self):
        """The hashes the page's password and the app's key are checked against: ``{"password", "api_key"}``."""
        return self.store.setting("access") or {}

    def set_password(self, password):
        """Sets (or with "" takes away) the page's password; only its salted hash is kept."""
        password = str(password or "")
        if password and len(password) < 8:
            raise ValueError("At least 8 characters")
        access = dict(self.access())
        if password:
            access["password"] = Vault.hash_password(password)
        else:
            # The app's key goes with it: a key alone would shut the page to browsers, which can only say a password.
            access = {}
        self.store.set_setting("access", access)
        self.store.event("The page's password was " + ("set" if password else "taken away (and the app's key with it)"))

    def new_api_key(self):
        """Makes a new key for the app (the one before stops working): the key, said this once; its hash is kept."""
        if not self.access().get("password"):
            # Without a password the page is open and a key would only shut browsers out of it.
            raise ValueError("Set the page's password first: the app's key is for a page that asks for one")
        key, hashed = Vault.new_key()
        self.store.set_setting("access", {**self.access(), "api_key": hashed})
        self.store.event("A new key for the app was made")
        return key

    def remove_api_key(self):
        self.store.set_setting("access", {k: v for k, v in self.access().items() if k != "api_key"})
        self.store.event("The app's key was taken away")

    def set_audiomuse(self, url, token=None, check_only=False):
        """Connects AudioMuse-AI once it answers and takes the token (an empty token keeps the one there is for the
        same address): the plugin that answered. With no address, takes the connection away."""
        url = str(url or "").strip().rstrip("/")
        saved = self.store.setting("audiomuse") or {}
        if not url:
            self.store.set_setting("audiomuse", None)
            self.audiomuse = AudioMuse("", "")
            self.store.event("AudioMuse-AI was disconnected")
            return None
        if not re.match(r"^https?://[^\s/]+", url):
            raise ValueError("The address starts with http:// or https://")
        token = str(token or "") or (self.vault.open(saved.get("token")) if saved.get("url") == url else "")
        client = AudioMuse(url, token)
        try:
            plugin = client.check()
        except AudioMuseError as e:
            # No Siansa plugin there: its export for apps is all that's needed since 0.25. A refused token is still said.
            if "plugin" not in str(e):
                raise
            client.sync_check()
            plugin = "export"
        if not check_only:
            self.store.set_setting("audiomuse", {"url": url, "token": self.vault.seal(token)})
            self.audiomuse = client
            self.store.event(f"AudioMuse-AI connected ({plugin} plugin)" if plugin != "export"
                             else "AudioMuse-AI connected (its export for apps; no Siansa plugin there)")
            self.puller.start()  # what it has analysed is taken over straight away
        return plugin

    @property
    def navidrome(self):
        """The default server's client (its name from when Navidrome was the only one)."""
        return self.servers.client()

    def only(self, server):
        """What the store is asked for ``server``'s songs: nothing to leave out while it's the only server."""
        return server if self.servers.several() else None

    def kept(self, name, server, work, seconds, changes):
        """The kept answer ``name`` for ``server``'s songs (:class:`Kept`), made the first time it's asked for."""
        key = (name, self.only(server))
        with self._kept_lock:
            if key not in self._kept:
                self._kept[key] = Kept(lambda: work(key[1]), seconds, changes)
            return self._kept[key]

    def server_list(self):
        """The music servers as the page and the app see them: no passwords, each with its songs here and its chosen
        libraries."""
        counts = self.kept("server-counts", None, lambda _: self.store.server_counts(), 30, lambda: self.store.writes).get()
        return [{**self.servers.public(s), "songs": counts.get(s["id"], 0),
                 "libraries": [str(f) for f in self.store.setting(folders_key(s["id"])) or []]}
                for s in self.servers.all()]

    # ---- what the API returns ----

    def song_row(self, row):
        row["verdict"] = grid.verdict(row["result"], row["tag_bpm"]) if row["result"] else []
        return row

    SONGS_SHOWN, SONGS_MOST = 200, 1000  # songs in one answer for the table, unless asked for more (at most)

    def songs(self, find=None, show="all", limit=None, server=None):
        """The table's songs: those ``find`` finds, of those to ``show`` (all, "differ" or "failed"), at most
        ``limit`` (``0``: every one, for the exports), of ``server``'s (every server's without one)."""
        limit = None if limit == 0 else min(int(limit or self.SONGS_SHOWN), self.SONGS_MOST)
        rows = self.store.overview(self.config.model, find, show if show in ("done", "failed") else None,
                                   None if show == "differ" else limit, server=self.only(server))
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
        row["measure"] = self.measure_of(song_id)
        full = self.store.result(song_id, self.config.model, with_beats=True)
        row["beats"] = full["beats"] if full else []
        row["downbeats"] = full["downbeats"] if full else []
        row["tempo_curve"] = grid.tempo_curve(row["beats"])
        row["grid"] = self.steady_grid(full)
        row["end_seconds"] = grid.END_SECONDS
        row["marks"] = self.store.marks(song_id)
        names = {s["id"]: s["name"] for s in self.servers.all()}
        row["servers"] = [{"id": server, "name": names.get(server, server), "item_id": item_id}
                          for server, item_id in self.store.where(song_id)]
        return row

    def steady_grid(self, full):
        """The even grid of a result with beats, or ``None``: as :func:`beats.grid.analyse` made it."""
        if not full or not full.get("bpm"):
            return None
        beats, bpm, per_bar = full["beats"], full["bpm"], full.get("beats_per_bar") or 4
        return grid.with_sound(grid.steady_grid(beats, full["downbeats"], bpm, per_bar), full.get("levels"))

    def export(self, find=None, show="all", server=None):
        """The analysed songs the table finds (every one without ``find`` and ``show``) with what the grid and
        phrases are worked out from (the model's beats and downbeats, the sound's levels) and what they came to: to
        tune them on real music. Song names and genres, no addresses."""
        out = [s for s in (self.song_data(row["id"], row) for row in self.songs(find, show, 0, server)) if s]
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

    def failed_export(self, server=None):
        """Every song that couldn't be analysed, with why and what kind of file it is: one file to look into them with."""
        rows = self.store.overview(self.config.model, None, "failed", None, server=self.only(server))
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

    def measure_of(self, song_id):
        """One song's loudness, fades and key for its page, with the ramps as ``[[dB, seconds], …]``; None when not
        measured yet."""
        m = self.store.measure(song_id)
        if not m:
            return None

        def points(blob):
            try:
                return [[level, round(chunk * loudness.CHUNK_SECONDS, 1)] for level, chunk in loudness.decode_ramp(blob or "")]
            except ValueError:
                return []

        return {"ref_lufs": m["ref_lufs"], "key": m["key"], "scale": m["scale"], "key_strength": m["key_strength"],
                "duration_ms": m["duration_ms"], "source": m["source"] or "beats", "error": m["error"],
                "measured_at": m["measured_at"], "start": points(m["start_ramp"]), "end": points(m["end_ramp"])}

    FADE_DB = -18  # the level the app fades on (its Crossfade.QUIET_DB)

    def loudness_report(self, server=None):
        """The page's loudness dashboard: how far it is, where the numbers came from, how loud the songs are, how they
        start and end, and their keys (of ``server``'s songs). Kept ready: it reads every measured song."""
        report = self.kept("loudness", server, self._loudness_report, 30, lambda: self.store.measure_changes).get()
        return {**report, "measuring": dict(self.worker.measuring) if self.worker.measuring else None}

    def _loudness_report(self, only):
        summary = self.store.measure_summary(only)
        rows = summary["rows"]
        counts = self.store.measure_counts(loudness.VERSION, only)
        lufs = [r["ref_lufs"] for r in rows if r["ref_lufs"] is not None]

        def bins(values, lo, hi, step):
            out = [0] * int((hi - lo) / step)
            for v in values:
                out[min(len(out) - 1, max(0, int((v - lo) // step)))] += 1
            return {"from": lo, "step": step, "counts": out}

        def at_fade(blob):
            """When the ramp passes the app's fade level (seconds), or None."""
            try:
                return next((chunk * loudness.CHUNK_SECONDS for level, chunk in loudness.decode_ramp(blob or "")
                             if level == self.FADE_DB), None)
            except ValueError:
                return None

        intros = [t for t in (at_fade(r["start_ramp"]) for r in rows) if t is not None]
        outros = [t for t in (at_fade(r["end_ramp"]) for r in rows) if t is not None]
        keys = {}
        for r in rows:
            if r["key"]:
                k = f"{r['key']} {r['scale'] or ''}".strip()
                keys[k] = keys.get(k, 0) + 1
        lufs_sorted = sorted(lufs)
        return {
            **counts,
            "sources": {"beats": sum(1 for r in rows if (r["source"] or "beats") == "beats"),
                        "plugin": sum(1 for r in rows if r["source"] == "plugin")},
            "with_key": sum(1 for r in rows if r["key"]),
            "audiomuse": self.audiomuse.configured,
            "lufs": {**bins(lufs, -30, 0, 1), "median": lufs_sorted[len(lufs_sorted) // 2] if lufs_sorted else None,
                     "quietest": lufs_sorted[0] if lufs_sorted else None, "loudest": lufs_sorted[-1] if lufs_sorted else None},
            "intro_seconds": bins(intros, 0, 30, 2), "outro_seconds": bins(outros, 0, 30, 2), "fade_db": self.FADE_DB,
            "keys": dict(sorted(keys.items(), key=lambda p: -p[1])),
            "failed_songs": summary["failed"],
        }

    def status(self, server=None):
        server = server or self.servers.default_id
        only = self.only(server)
        parts = {}  # where the time went, for the log when the answer is slow

        def timed(name, work):
            started = time.monotonic()
            try:
                return work()
            finally:
                parts[name] = time.monotonic() - started

        counts = timed("queue", lambda: self.store.counts(only))
        styles = timed("styles", lambda: self.styles_status(server))
        measured = timed("loudness", lambda: self.kept("measure-counts", server, lambda o: self.store.measure_counts(loudness.VERSION, o), 30,
                                                       lambda: self.store.writes).get())
        sound = timed("sound", lambda: self.sound_status(server))
        servers = timed("servers", self.server_list)
        events = timed("events", self.store.events)
        self.status_parts = parts
        worker = self.worker
        shown = self.servers.get(server) or {}
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
            "memory_mb": memory_module.rss_mb(),
            "counts": counts,
            "current": current,
            "next": preparing,
            "bad_blends": self.store.bad_blend_count(),
            "styles": styles,
            "connections": {"navidrome": bool(shown), "audiomuse": self.audiomuse.configured,
                            "audiomuse_url": self.audiomuse.url or None, "audiomuse_token": bool(self.audiomuse.token),
                            "api_key": bool(self.access().get("api_key")), "password": bool(self.access().get("password")),
                            # Secrets the container still has though they're no longer read: they can go from the YAML.
                            "left_in_container": self.config.secrets_left,
                            "secret_key": "container" if self.vault.from_env else "data folder"},
            "loudness": {**measured,
                         "measuring": dict(worker.measuring) if worker.measuring else None},
            "regridding": dict(worker.regridding) if worker.regridding else None,
            "reading": dict(worker.reading) if worker.reading else None,
            "sound": sound,
            "libraries": {"chosen": self.store.setting(folders_key(server)) or [],
                          "read_at": self.store.setting("library_read_at")},
            # The server these numbers are of, and all of them (for the page's switch).
            "server": server,
            "server_kind": shown.get("kind"),
            "server_name": shown.get("name"),
            "servers": servers,
            "paused": worker.paused,
            "last_error": worker.last_error,
            "model": self.config.model,
            "device": getattr(self.tracker, "device", None) or self.config.device,
            "device_name": getattr(self.tracker, "device_name", None),
            "device_note": getattr(self.tracker, "device_note", None),
            "float16": self.config.float16,
            "model_loaded": self.tracker_loaded(),
            "model_load_seconds": getattr(self.tracker, "load_seconds", None),
            "navidrome": shown.get("url") or None,  # the shown server's address (its name from before 0.23)
            "log_level": log.level(),
            "uptime_seconds": round(time.time() - self.started),
            "events": events,
        }

    def families_of(self, ids, server=None):
        """CLAP's view of each song's genre families (0.38, a look only): ``{asked id: {"song", "families":
        {"cosine", "share"}, "styles": [(style, score)], "gated": [(style, score)]}}``; a song without a CLAP vector
        says so."""
        from . import families as fam

        if self._families is None:
            self._families = fam.Families(self.words)
        here = self.store.resolve(ids, server)
        rows = self.store.clap_of(list(here.values()))
        styles = self.store.genre_rows(list(here.values()), sources=("sound", "sound-doubted"))
        out = {}
        for asked in ids:
            song_id = here.get(asked)
            if song_id is None:
                out[asked] = {"error": "not a song here"}
                continue
            song = self.store.song(song_id) or {}
            entry = {"song": {k: song.get(k) for k in ("title", "artist", "album")}}
            clap = rows.get(song_id)
            if clap is None:
                entry["error"] = "no CLAP vector yet: not heard here or by AudioMuse-AI"
            else:
                scores = self._families.scores(np.frombuffer(clap, dtype=np.float32))
                had = [(l, w) for src in ("sound", "sound-doubted") for l, w in styles.get(song_id, {}).get(src, [])]
                entry["families"] = scores
                entry["styles"] = [(l.partition("---")[2] or l, w) for l, w in had[:8]]
                entry["gated"] = [(l.partition("---")[2] or l, w) for l, w in fam.gated(had, scores["share"])[:8]]
            out[asked] = entry
        return out

    def style_models(self):
        """The style models to choose from (0.39): each with its name, what it's like, what it costs and whether the
        image has it; and which is in use."""
        from . import style as style_module

        spec = getattr(self.style, "spec", None)
        return {"chosen": spec.id if spec else None,
                "models": [m.summary(self.config.models_dir) for m in style_module.MODELS.values()]}

    def start_analysis(self, ids, server=None):
        """Start analysis (0.40, the owner, 13 Nov): every check first (:mod:`beats.checks`), then, when none failed,
        the ticked libraries are the ones to analyse and are read for new songs in the background. The report, with
        ``reading`` (whether a read started) and ``queued_before``."""
        server = server or self.servers.default_id
        report = Checks(self).run(server, ids)
        report["reading"] = False
        if report["ok"]:
            answer = self.worker.choose(ids, server, background=True)
            report["reading"] = bool(answer.get("reading"))
            if answer.get("reading") is False:
                report["note"] = "The libraries are already being read: this choice is read next time."
        failed = [c["name"] for c in report["checks"] if c["state"] == "fail"]
        warned = [c["name"] for c in report["checks"] if c["state"] == "warn"]
        self.store.event(f"Checked before analysing in {report['seconds']}s: "
                         + ("all well" if not failed and not warned else
                            ", ".join(([f"failed: {', '.join(failed)}"] if failed else []) + ([f"to look at: {', '.join(warned)}"] if warned else [])))
                         + ("" if report["ok"] else ". Nothing started"), "error" if failed else "warn" if warned else "info")
        return report

    def choose_style_model(self, model_id):
        """Switches the style model (0.39, the owner, 13 Nov): the other one is let go, the choice kept, and every
        song's styles are doubted (still used) and heard again with the new model. How many songs that is."""
        from . import style as style_module

        spec = style_module.MODELS.get(model_id)
        if spec is None:
            raise ValueError(f"No style model called {model_id!r}")
        if not spec.available(self.config.models_dir):
            raise ValueError(f"{spec.name} isn't in this image")
        current = getattr(self.style, "spec", None)
        if current is not None and current.id == spec.id:
            return 0
        self.style.unload()
        self.style = self.style_factory(spec)
        self.worker.style = self.style
        self.worker._style_check_said = False
        self.store.set_setting("style_model", spec.id)
        doubted = self.store.doubt_styles()
        self.store.event(f"Style model: {spec.name} from now on; {doubted} songs are heard again with it")
        _log.info("Style model: %s chosen (%d songs to hear again)", spec.name, doubted)
        self.worker.wake()
        return doubted

    def styles_status(self, server=None):
        """The style model: whether the image has it, how far the songs (``server``'s) are, and how often it agrees
        with the tags."""
        worker, style = self.worker, self.style
        available = bool(style is not None and style.available)
        spec = getattr(style, "spec", None)
        out = {"available": available, "loaded": bool(available and style.loaded),
               "device": getattr(style, "backend", None), "problem": getattr(style, "problem", None),
               "checked": getattr(style, "checked", None),
               "model": spec.id if spec else None, "model_name": spec.name if spec else None}
        if available:
            # Kept ready: the page asks every few seconds; its total follows songs added, too.
            out.update(self.kept("styles", server or self.servers.default_id, self.store.style_counts, 30,
                                 lambda: self.store.genre_changes).get())
            out["agreement"] = worker.genre_report
            styling = dict(worker.styling) if worker.styling else None
            if styling:
                styling["for_seconds"] = round(time.time() - styling.pop("since"), 1)
            out["current"] = styling
        return out

    def about_names(self):
        """Every genre the app may show a page for: the sound's as the app names them, then the songs' tags."""
        names, seen = [], set()
        for name in [g["name"] for g in self.genre_list()] + self.store.tag_genres():
            if name and name.lower() not in seen:
                seen.add(name.lower())
                names.append(name)
        return names

    def genre_about(self, name):
        """``{"name", "about", "pending", "status"}``: a kept description at once; a new one waits a few seconds at most.
        ``status``: "found", "pending" (still being looked up), "missing" (Wikipedia has nothing on it) or "failed"
        (Wikipedia couldn't be asked: tried again in half an hour)."""
        found, pending = self.about.answer(name)
        from . import about as about_module

        kept = self.store.setting(about_module.key_of(name)) or {}
        # Why there's none, for the app's log: Wikipedia has nothing on it, or couldn't be asked (tried again soon).
        status = "found" if found else "pending" if pending else "failed" if kept.get("failed") else "missing"
        return {"name": name, "about": found, "pending": pending, "status": status}

    def genre_list(self, server=None):
        """Every style with songs (of ``server``'s) in its genre by their sound, most first, kept ready (:class:`Kept`): ``[{"label",
        "genre", "style", "songs", "name", "libraries"}]``; ``libraries``: the songs in each of the server's libraries,
        by its id."""
        server = server or self.servers.default_id
        return self.kept("genres", server, lambda only: self._genre_list(server, only), 60,
                         lambda: self.store.genre_changes).get()

    def _genre_list(self, server, only):
        counts = self.store.heard_counts(genres_module.HEARD_MIN, genres_module.HEARD_RANK, only)
        by_library = self.store.heard_counts_by_library(genres_module.HEARD_MIN, genres_module.HEARD_RANK, server)
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
                        "name": f"{genre} / {style}" if len(named[style]) > 1 else style,
                        "libraries": by_library.get(label, {})})
        return out

    def genres_of(self, ids, server=None):
        """The genres each of the songs ``ids`` is in by its sound, its best first: ``{asked id: [{"name", "label",
        "genre", "style", "score"}]}``, ``name`` as the genre list names it. Songs with none aren't in it."""
        here = self.store.resolve(ids, server)
        heard = self.store.heard_of(here.values(), genres_module.HEARD_MIN, genres_module.HEARD_RANK)
        names = {g["label"]: g["name"] for g in self.genre_list(server)}
        out = {}
        for asked in ids:
            rows = heard.get(here.get(asked))
            if rows:
                out[asked] = [{"name": names.get(label) or label.partition("---")[2] or label, "label": label,
                               "genre": label.partition("---")[0], "style": label.partition("---")[2] or label,
                               "score": round(weight, 4)} for label, weight in rows]
        return out

    def genre_songs(self, name, limit=None, server=None):
        """The songs of the genre ``name`` (a tag as the app shows it: "Trance", "Electronic / Trance") by their sound
        (:mod:`beats.genres`), the ones that sound most like it first: ``{"name", "styles", "genres", "ids", "songs"}``,
        the ids ``server``'s own for the songs it has; ``songs``: how many there are, of which ``ids`` is a random
        ``limit`` when there are more. Nothing when the style model isn't there or the name matches none of its
        styles, so the app falls back to the tags."""
        taxonomy = self.worker.taxonomy()
        if taxonomy is None:
            return {"name": name, "styles": [], "genres": [], "ids": [], "songs": 0, "why": "no style model in this image"}
        styles, families = genres_module.target(taxonomy, name)
        labels = set(styles)
        for f in families:
            labels |= set(taxonomy.by_genre.get(genres_module.key(f), []))
        # A style: the songs with it among their best few. A whole family ("Electronic"): only the songs whose best
        # style is in it, or the family's radio is everything with a faint touch of any of its styles (0.36).
        rank = genres_module.HEARD_RANK if styles else 1
        found = self.store.songs_heard(labels, genres_module.HEARD_MIN, rank, None, server or self.servers.default_id)
        songs = len(found)
        if limit is not None and songs > limit:
            import random

            keep = set(random.sample(range(songs), limit))
            found = [p for n, p in enumerate(found) if n in keep]  # still the ones that sound most like it first
        sure = sum(1 for _, w in found if w >= SURE)
        _log.info("Genre %s: %d songs by their sound (%d sure, %.2f or more)%s", name, songs, sure, SURE,
                  " for the family's best styles only" if rank == 1 else "")
        return {"name": name, "styles": sorted(styles), "genres": sorted(families), "ids": [i for i, _ in found],
                "weights": [round(w, 3) for _, w in found], "songs": songs, "sure": sure,
                "by": "sound", "why": None if labels else "no style of that name"}

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
        setup = {"model": c.model,
                 "servers": self.server_list(), "default_server": self.servers.default_id,
                 "device": c.device, "float16": c.float16, "threads": c.threads,
                 "page_password": "set" if c.password else "not set", "api_key": "set" if c.api_key else "not set",
                 "audiomuse_url": self.audiomuse.url or None, "audiomuse_token": "set" if self.audiomuse.token else "not set",
                 "page_password": "set" if self.access().get("password") else "not set",
                 "app_key": "set" if self.access().get("api_key") else "not set",
                 "secrets_left_in_container": c.secrets_left, "data_dir": c.data_dir,
                 "models_dir": c.models_dir, "port": c.port, "log_level": log.level()}
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

    def csv(self, find=None, show="all", server=None):
        out = io.StringIO()
        columns = ["artist", "title", "album", "genres", "sound_genres", "sound_top", "tag_bpm", "bpm",
                   "main_share", "bpm_fit", "grid_bpm", "on_grid", "grid_drift", "downbeat_source", "phrase_first_ms", "phrase_bars", "phrase_count",
                   "phrase_big", "phrase_big_on",
                   "intro_bpm", "outro_bpm", "steady", "grid_share", "beats_per_bar", "bar_share", "first_downbeat_ms",
                   "model_seconds", "state", "notes", "error"]
        w = csv.writer(out)
        w.writerow(columns)
        for row in self.songs(find, show, 0, server):
            r = row["result"] or {}
            by = {}
            for g in row["genres"]:
                by.setdefault(g["source"], []).append(g)
            w.writerow([row["artist"], row["title"], row["album"], " / ".join(g["label"] for g in by.get("tag", [])),
                        " / ".join(f"{l.partition('---')[2]} {v:.2f}" for l, v in genres_module.heard(
                            [(g["label"], g["weight"]) for g in by.get("sound", [])])),
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
    POLLED = ("/api/status", "/api/songs", "/api/libraries", "/api/log", "/api/bad-blends", "/api/health", "/api/art/")

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
            parts = getattr(self.app, "status_parts", None) if path == "/api/status" and self.app else None
            where = " · " + ", ".join(f"{k} {v:.1f}s" for k, v in parts.items() if v >= 0.05) if parts else ""
            _log.warning("%s %s took %.1fs%s%s", self.command, log.redact(path), took,
                         f" (busy {busy})" if busy else "", where)
        code = int(getattr(code, "value", code)) if str(getattr(code, "value", code)).isdigit() else 0
        if code >= 500:
            _log.warning("%s %s -> %s", self.command, log.redact(self.path), code)
        elif code >= 400 and path not in ("/favicon.ico",):
            _log.info("%s %s -> %s", self.command, log.redact(self.path), code)
        elif self.command != "GET" or not path.startswith(self.POLLED):
            _log.debug("%s %s -> %s", self.command, log.redact(self.path), code)

    # ---- plumbing ----

    def _authorised(self):
        access, vault = self.app.access(), self.app.vault
        password, api_key = access.get("password"), access.get("api_key")  # hashes: the secrets themselves aren't kept
        if not password and not api_key:
            return True
        header = self.headers.get("Authorization", "")
        if header.startswith("Bearer "):
            given = header[7:].strip()
            return vault.check_key(given, api_key) or vault.check_password(given, password)
        if header.startswith("Basic ") and password:
            try:
                user, _, given = base64.b64decode(header[6:]).decode().partition(":")
            except (ValueError, UnicodeDecodeError):
                return False
            return vault.check_password(given, password)
        return False

    SQUEEZE_FROM = 1024  # bytes: answers this big are sent gzipped to a browser or app that takes it

    def _send(self, status, body=b"", kind="application/json", headers=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, separators=(",", ":")).encode()
        elif isinstance(body, str):
            body = body.encode()
        squeezed = (len(body) >= self.SQUEEZE_FROM and "gzip" in self.headers.get("Accept-Encoding", "")
                    and kind.startswith(("application/json", "text/")))
        if squeezed:  # the songs' table is a megabyte of JSON: a tenth of it over a phone's Wi-Fi
            body = gzip.compress(body, 5)
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        if "Cache-Control" not in (headers or {}):  # a picture says how long it may be kept; nothing else is
            self.send_header("Cache-Control", "no-store")
        if squeezed:
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Vary", "Accept-Encoding")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, value, status=HTTPStatus.OK):
        self._send(status, value)

    def _error(self, status, message):
        self._json({"error": message}, status)

    IMPORT_MOST = 2 * 1024 ** 3  # bytes: no export of a listening history comes near

    def _import_listens(self):
        """Takes an export file as the request's body (``?user=`` whose it is, ``?name=`` its name) and brings it in
        in the background."""
        import tempfile

        app, query = self.app, self._query()
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return self._error(HTTPStatus.BAD_REQUEST, "Choose a file: an export of Koito, Last.fm or ListenBrainz")
        if length > self.IMPORT_MOST:
            return self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "That file is too big")
        if app.listens.importing.get("running"):
            return self._error(HTTPStatus.CONFLICT, "A file is being brought in already")
        folder = os.path.join(app.config.data_dir, "tmp")
        os.makedirs(folder, exist_ok=True)
        handle, path = tempfile.mkstemp(prefix="listens-", dir=folder)
        try:
            with os.fdopen(handle, "wb") as out:
                left = length
                while left:
                    piece = self.rfile.read(min(left, 1 << 20))
                    if not piece:
                        break
                    out.write(piece)
                    left -= len(piece)
            if left:
                raise OSError("the file didn't arrive whole")
        except OSError as e:
            _remove(path)
            return self._error(HTTPStatus.BAD_REQUEST, f"The file couldn't be taken: {e}")
        user = (query.get("user") or [""])[0].strip()
        name = (query.get("name") or ["the file"])[0][:120]
        if not app.listens.start_import(path, user, name, done=lambda: _remove(path)):
            _remove(path)
            return self._error(HTTPStatus.CONFLICT, "A file is being brought in already")
        return self._json({"started": True, "importing": dict(app.listens.importing)})

    def _body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(min(length, 1 << 20)) or b"{}")
        except ValueError:
            return {}

    def _server(self, query):
        """The id of the server a request is about (``?server=``, its id or name; the default without it), or None
        after answering that there's no such server."""
        asked = (query.get("server") or [""])[0]
        found = self.app.servers.find(asked)
        if found is None:
            names = ", ".join(s["name"] for s in self.app.servers.all()) or "none"
            self._error(HTTPStatus.BAD_REQUEST, f"No server {asked} here (the servers: {names})")
        return found

    def _query(self):
        return urllib.parse.parse_qs(self.path.split("?", 1)[1]) if "?" in self.path else {}

    def _guard(self):
        if self._authorised():
            return True
        self._send(HTTPStatus.UNAUTHORIZED, {"error": "password or API key needed"},
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
            if path == "/api/servers":
                return self._json({"servers": app.server_list(), "default": app.servers.default_id,
                                   "kinds": [{"id": k, "name": v} for k, v in KINDS.items()]})
            # Similar music is asked of AudioMuse-AI, whose own name for the server is passed on as it is.
            server = app.servers.find((query.get("server") or [""])[0]) or app.servers.default_id \
                if path.startswith("/api/v1/similar/") else self._server(query)
            if server is None:
                return None
            if path.startswith("/api/v1/") and api_v1.get(self, path, query, server):
                return None
            if path == "/api/listens":
                def number(name):
                    value = (query.get(name) or [""])[0]
                    return int(value) if value.isdigit() and int(value) else None
                user = (query.get("user") or [None])[0]
                month = number("month")
                if month is not None and not 1 <= month <= 12:
                    return self._error(HTTPStatus.BAD_REQUEST, "month is 1 to 12")
                return self._json(app.listening(user, number("year"), month, (query.get("tz") or [None])[0]))
            if path == "/api/listens/import":
                return self._json(dict(app.listens.importing))
            if path == "/api/listens/artist":  # one artist in a listener's history: its own page
                name = (query.get("name") or [""])[0].strip()
                if not name:
                    return self._error(HTTPStatus.BAD_REQUEST, "name needed: the artist")
                found = app.listening_artist((query.get("user") or [None])[0], name, (query.get("tz") or [None])[0])
                return self._json(found) if found else self._error(HTTPStatus.NOT_FOUND, f"No listens of {name}")
            m = re.fullmatch(r"/api/art/(cover|artist)/([^/]+)", path)
            if m:  # a song's cover, or its artist's picture, from the music server (kept here once fetched)
                size = (query.get("size") or ["300"])[0]
                found = app.art.get(m[1], _id(m[2]), int(size) if size.isdigit() else 300)
                if found is None:
                    return self._send(HTTPStatus.NOT_FOUND, b"", "text/plain", {"Cache-Control": "private, max-age=600"})
                return self._send(HTTPStatus.OK, found[0], found[1], {"Cache-Control": "private, max-age=604800"})
            if path == "/api/genres":
                return self._json({"styles": app.styles_status(server), "genres": app.genre_list(server),
                                   "about": app.about.status()})
            if path == "/api/style/models":  # the style models to choose from, and the one in use (0.39)
                return self._json(app.style_models())
            if path == "/api/genre/about":
                name = (query.get("name") or [""])[0].strip()
                if not name:
                    return self._error(HTTPStatus.BAD_REQUEST, "name needed: the genre, as a tag (Trance)")
                return self._json(app.genre_about(name))
            if path == "/api/genre":
                name = (query.get("name") or [""])[0].strip()
                if not name:
                    return self._error(HTTPStatus.BAD_REQUEST, "name needed: the genre, as a tag (Trance)")
                try:
                    limit = max(1, min(2000, int((query.get("limit") or ["300"])[0])))
                except ValueError:
                    return self._error(HTTPStatus.BAD_REQUEST, "limit is a number")
                return self._json(app.genre_songs(name, limit, server))
            if path == "/api/loudness":
                return self._json(app.loudness_report(server))
            if path == "/api/status":
                return self._json(app.status(server))
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
                return self._json(app.worker.libraries(server))
            if path == "/api/analysis/checks":  # the last report (0.40)
                return self._json(app.store.setting("last_checks") or {"checks": [], "ok": None})
            if path == "/api/bad-blends":
                return self._json(app.store.bad_blends())
            if path == "/api/failed/export":
                body = json.dumps(app.failed_export(server), indent=1).encode()
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
                return self._json(app.songs(find, show, limit or None, server))
            if path == "/api/grids":
                ids = [i for i in dict.fromkeys(p.strip() for p in (query.get("ids") or [""])[0].split(",")) if i]
                if len(ids) > 2000:
                    return self._error(HTTPStatus.BAD_REQUEST, "at most 2000 ids at a time")
                here = app.store.resolve(ids, server)
                found = app.store.ends(here.values(), app.config.model)
                return self._json({"version": __version__,
                                   "grids": [{"track_id": i, **found[here[i]]} for i in ids if here.get(i) in found],
                                   "missing": [i for i in ids if here.get(i) not in found]})
            if path == "/api/export.json":
                body = json.dumps(app.export(find, show, server)).encode()
                return self._send(HTTPStatus.OK, body, "application/json",
                                  {"Content-Disposition": 'attachment; filename="siansa-beats-data.json"'})
            if path == "/api/export.csv":
                return self._send(HTTPStatus.OK, app.csv(find, show, server), "text/csv; charset=utf-8",
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
        if path == "/api/listens/import":  # a file, as it is: it can be far bigger than any other request
            return self._import_listens()
        app, body = self.app, self._body()
        try:
            if path.startswith("/api/servers"):
                return self._servers_post(path, body)
            if path in ("/api/audiomuse", "/api/audiomuse/test"):
                from .audiomuse import AudioMuseError
                try:
                    plugin = app.set_audiomuse(body.get("url"), body.get("token"), check_only=path.endswith("/test"))
                except ValueError as e:
                    return self._error(HTTPStatus.BAD_REQUEST, str(e))
                except AudioMuseError as e:
                    return self._error(HTTPStatus.BAD_GATEWAY, str(e))
                return self._json({"plugin": plugin, "url": app.audiomuse.url or None})
            if path == "/api/style/model":  # the style model to hear genres with (0.39): {"model": "effnet" | "maest"}
                try:
                    doubted = app.choose_style_model(str(body.get("model") or ""))
                except ValueError as e:
                    return self._error(HTTPStatus.BAD_REQUEST, str(e))
                return self._json({"doubted": doubted, **app.style_models()})
            if path == "/api/sound/pull":  # everything again with {"everything": true}
                if not app.audiomuse.configured:
                    return self._error(HTTPStatus.BAD_REQUEST, "Connect AudioMuse-AI first (Setup › AudioMuse-AI)")
                started = app.puller.start(bool(body.get("everything")))
                return self._json({"started": started, "pull": dict(app.puller.state)})
            if path == "/api/listens/lastfm":  # on to Last.fm, with the listener's own API account
                from .listens import LastFmError
                user = str(body.get("user") or "")
                try:
                    if body.get("disconnect"):
                        app.listens.disconnect_lastfm(user)
                        return self._json(app.listens.lastfm_status(user))
                    return self._json(app.listens.connect_lastfm(user, body.get("api_key"), body.get("secret"),
                                                                 body.get("name"), body.get("password")))
                except ValueError as e:
                    return self._error(HTTPStatus.BAD_REQUEST, str(e))
                except LastFmError as e:
                    return self._error(HTTPStatus.BAD_GATEWAY, str(e))
            if path == "/api/listens/rename":  # {"from", "to"}: one listener's listens become another's
                if "from" not in body or not str(body.get("to") or "").strip():
                    return self._error(HTTPStatus.BAD_REQUEST, "from and to needed: whose listens, and whose they become")
                return self._json({"moved": app.listens.rename(body["from"], body["to"])})
            if path == "/api/listens/remove":  # a listener's listens from one source (a file brought in by mistake)
                if "user" not in body or body.get("source") not in ("koito", "lastfm", "listenbrainz", "siansa", "all"):
                    return self._error(HTTPStatus.BAD_REQUEST, "user and source needed: koito, lastfm, listenbrainz, siansa or all")
                gone = app.listens.remove(str(body["user"]), None if body["source"] == "all" else body["source"])
                app.store.event(f"Removed {gone} listens of {body['user'] or 'the listener'} ({body['source']})")
                return self._json({"removed": gone})
            if path == "/api/access/password":
                try:
                    app.set_password(body.get("password"))
                except ValueError as e:
                    return self._error(HTTPStatus.BAD_REQUEST, str(e))
                return self._json({"password": bool(app.access().get("password"))})
            if path == "/api/access/key":  # said this once: only its hash is kept
                try:
                    return self._json({"key": app.new_api_key()})
                except ValueError as e:
                    return self._error(HTTPStatus.BAD_REQUEST, str(e))
            server = self._server(self._query())
            if server is None:
                return None
            if path == "/api/libraries":
                ids = body.get("ids")
                if not isinstance(ids, list):
                    return self._error(HTTPStatus.BAD_REQUEST, 'ids needed: a list of library ids ([] for none)')
                try:
                    # The page's: chosen now, read in a thread whose progress the status shows (0.24.1).
                    return self._json(app.worker.choose(ids, server, background=bool(body.get("background"))))
                except ValueError as e:
                    return self._error(HTTPStatus.BAD_REQUEST, str(e))
            if path.startswith("/api/v1/") and api_v1.post(self, path, body, server):
                return None
            if path == "/api/bad-blends":
                answer, problem = api_v1.add_bad_blend(app, body, server)
                return self._error(HTTPStatus.BAD_REQUEST, problem) if problem else self._json(answer)
            if path == "/api/genres/about/fetch":
                return self._json(app.about.fetch_all())
            if path == "/api/analysis/start":
                ids = body.get("ids")
                if not isinstance(ids, list):
                    return self._error(HTTPStatus.BAD_REQUEST, 'ids needed: a list of library ids ([] for none)')
                return self._json(app.start_analysis(ids, server))
            if path == "/api/libraries/read":
                if body.get("background"):
                    return self._json(app.worker.read_in_background(every=True))
                return self._json(app.worker.read_libraries())
            if path == "/api/analyse-all":
                queued = sum(app.store.queue(song_id, force=True) for song_id in app.store.song_ids())
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

    def _servers_post(self, path, body):
        """Adds, tests or changes a music server, or makes one the default. A server is only kept once it answers and
        takes the login; what went wrong is said in plain words, never with the password."""
        app = self.app
        fields = {k: body.get(k) for k in ("name", "url", "user", "password")}
        try:
            if path == "/api/servers/test":
                return self._json(app.servers.test(str(body.get("kind") or "").lower(), str(fields["url"] or "").strip().rstrip("/"),
                                                   str(fields["user"] or "").strip(), fields["password"] or ""))
            if path == "/api/servers":
                added = app.servers.add(body.get("kind"), **fields)
                app.store.event(f"Music server added: {added['name']} ({KINDS[added['kind']]}, user {added['user']})")
                return self._json(added)
            m = re.fullmatch(r"/api/servers/([^/]+)(/default)?", path)
            if not m:
                return self._error(HTTPStatus.NOT_FOUND, "not found")
            server_id = _id(m[1])
            if m[2]:
                app.servers.set_default(server_id)
                app.store.event(f"The default music server is now {app.servers.get(server_id)['name']}")
                return self._json({"default": server_id})
            changed = app.servers.update(server_id, **fields)
            app.store.event(f"Music server changed: {changed['name']}")
            return self._json(changed)
        except KeyError:
            return self._error(HTTPStatus.NOT_FOUND, "no such server")
        except ValueError as e:
            _log.warning("%s: %s", "Testing a server" if path.endswith("/test") else "A server change", e)
            return self._error(HTTPStatus.BAD_REQUEST, str(e))
        except ServerError as e:
            # Said in the log too (0.24.1: a failed test said only "-> 502"); never with the password.
            _log.warning("%s %s at %s as %s: %s", "Testing" if path.endswith("/test") else "Signing in to",
                         KINDS.get(str(body.get("kind") or "").lower(), "the server"), body.get("url") or "its address",
                         body.get("user") or "its user", e)
            return self._error(HTTPStatus.BAD_GATEWAY, str(e))

    def do_DELETE(self):
        if not self._guard():
            return
        path = self.path.split("?", 1)[0]
        if path == "/api/audiomuse":
            self.app.set_audiomuse("")
            return self._json({"url": None})
        if path == "/api/access/password":
            self.app.set_password("")
            return self._json({"password": False})
        if path == "/api/access/key":
            self.app.remove_api_key()
            return self._json({"api_key": False})
        m = re.fullmatch(r"/api/servers/([^/]+)", path)
        if m:
            server_id = _id(m[1])
            try:
                name = (self.app.servers.get(server_id) or {}).get("name")
                removed = self.app.servers.remove(server_id)
            except KeyError:
                return self._error(HTTPStatus.NOT_FOUND, "no such server")
            except ValueError as e:
                return self._error(HTTPStatus.CONFLICT, str(e))
            self.app.store.event(f"Music server removed: {name} (and the {removed} songs' links to it)")
            return self._json({"removed": removed})
        server = self._server(self._query())
        if server is None:
            return None
        if path == "/api/bad-blends":
            removed = self.app.store.clear_bad_blends()
            _log.info("Bad blends cleared: %d taken away", removed)
            return self._json({"removed": removed})
        m = re.fullmatch(r"/api/bad-blends/(\d+)", path)
        if m:
            if not self.app.store.remove_bad_blend(int(m[1])):
                return self._error(HTTPStatus.NOT_FOUND, "no such bad blend")
            return self._json({"removed": True})
        m = re.fullmatch(r"/api/libraries/([^/]+)/songs", path)
        if m:
            try:
                return self._json({"removed": self.app.worker.remove_library(_id(m[1]), server)})
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
        """The song for the page's player, from a server that has it (Navidrome's MP3, which every browser plays; Emby's
        and Jellyfin's own file), passed through with seeking (Range) kept."""
        if not self.app.store.song(song_id):
            return self._error(HTTPStatus.NOT_FOUND, "no such song")
        upstream, problem = None, None
        for server, item_id in self.app.store.where(song_id) or [(MAIN, song_id)]:
            client = self.app.servers.client(server)
            try:
                upstream = client.audio(item_id, self.headers.get("Range"))
                break
            except urllib.error.HTTPError as e:
                problem = problem or (e.code, f"{client.label} didn't send the song")
            except (urllib.error.URLError, OSError) as e:
                problem = problem or (HTTPStatus.BAD_GATEWAY, f"{client.label}: {getattr(e, 'reason', e)}")
        if upstream is None:
            return self._error(*problem)
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
    app.worker.start()
    app.about.start()
    app.pull_now_and_then()
    app.relay_now_and_then()
    app.store.event(f"Started: model {config.model}, device {config.device}"
                    f"{' (half precision on a GPU)' if config.float16 else ''}")
    if not app.servers.all():
        app.store.event("No music server is set up: set NAVIDROME_URL, NAVIDROME_USER and NAVIDROME_PASSWORD, or add "
                        "one on the page (Setup)", "error")
    for server in app.servers.all():
        try:
            info = app.servers.client(server["id"]).ping()
            app.store.event(f"{server['name']} answers ({info.get('type') or 'server'} {info.get('version') or ''})".strip())
        except SubsonicError as e:
            app.store.event(f"{server['name']} doesn't answer: {e}", "error")
    def background():  # after the page is up: a big library takes minutes
        try:
            app.worker.regrid()
        except Exception:
            _log.exception("Working out older results again went wrong")
        try:  # the page's first numbers, ready before anyone opens it (0.35.1)
            started = time.monotonic()
            app.status()
            app.genre_list()
            _log.info("The page's numbers are ready (%.1fs)", time.monotonic() - started)
        except Exception:
            _log.exception("Getting the page's numbers ready went wrong")
        try:  # how energetic each song is (0.35): the whole library, once, from the sound already kept
            scored = app.arousal.catch_up()
            if scored:
                app.store.event(f"Worked out how energetic {scored} songs are, from the sound kept for each (no song was read again)")
        except Exception:
            _log.exception("Working out how energetic the songs are went wrong")
        try:  # genres are the sound's alone from 0.28: the ones voted with the tags go
            gone = app.store.drop_votes()
            if gone:
                app.store.event(f"Genres now come from each song's sound alone: removed {gone} genres voted with the tags")
            compared = app.worker.compare_with_tags()
            report = app.worker.genre_report or {}
            if compared and report.get("family") is not None:
                app.store.event(f"The sound agrees with the tags' genre on {report['family'] * 100:.0f} % of "
                                f"{report['songs']} songs with both")
        except Exception:
            _log.exception("Tidying the genres went wrong")

    threading.Thread(target=background, name="start", daemon=True).start()
    server = make_server(app)
    _log.info("Siansa Beats %s on port %s", __version__, config.port)
    _log.info("Memory at the start: %s", memory_module.said(memory_module.rss_mb()))
    try:
        server.serve_forever()
    finally:
        app.worker.stop()


if __name__ == "__main__":
    main()
