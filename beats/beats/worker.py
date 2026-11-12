"""The queue: from Navidrome to Beat This! to the database, with what it's doing on show.

Each song is downloaded (its original file, to a temporary file in the data folder), decoded, given to the model,
and its beats turned into the readings :mod:`grid` works out. The temporary file is deleted straight after. Two
threads share the work: one downloads, decodes and measures the next song while the other runs the model on this
one, so the model (the GPU) rarely waits for the network.

Each song's style comes from its sound too (:mod:`beats.style`, when the image has the model): for a new song from
the samples already decoded, for songs analysed before from a third thread that downloads them again while the queue
is empty. Those styles are the songs' genres (:mod:`beats.genres`).

Each song's loudness, fades (:mod:`beats.loudness`) and key (:mod:`beats.key`) are measured on the way too, from the
file already downloaded; songs analysed before 0.20 get them from a fourth thread while the queue is empty.

The songs come from the libraries chosen on the page (all, or some: leaving out compilations, say) of each music
server (:mod:`beats.servers`: Navidrome, Emby, Jellyfin, several users of one). They're read again every few hours for
new songs, the default server first: a song another server has too is found by its file or tags and not analysed
again (:mod:`beats.match`). A song is downloaded from the server it was first found on, or any other that has it.
"""

import os
import queue
import tempfile
import threading
import time

from . import features
from . import genres
from . import grid
from . import key as key_finder
from . import log
from . import loudness
from . import memory
from .servers import MAIN, folders_key
from .subsonic import ServerError, SubsonicError

_log = log.get("worker")

# Errors of bugs that are fixed: songs that failed with them are analysed again on start.
FIXED_ERRORS = ("'bars'", "not supported between instances of 'NoneType'")


LIBRARY_RESCAN_SECONDS = 6 * 3600  # how often the chosen libraries are read again for new songs
STYLE_KEEP = 15  # the top styles kept for each song (beats.style.KEEP)
STYLE_RATE = 16_000  # the style model's sample rate: songs analysed before are decoded again at it
STYLE_REPORT_EVERY = 500  # the agreement with the tags is worked out again after this many more songs


def measure_sound(path, samples, rate):
    """A song's loudness and fades from its file (a second decode, at 48 kHz) and its key from the samples already
    decoded: :func:`beats.loudness.profile`'s fields with :func:`beats.key.find`'s."""
    values = loudness.measure_file(path)
    try:
        values.update(key_finder.find(samples, rate) or {})
    except Exception as e:  # the loudness still counts without the key
        _log.debug("Finding the key failed: %s", e)
    return values


class Worker:
    MEASURE_IDLE_SECONDS = 60.0  # between looks for songs to measure, when there were none
    def __init__(self, store, navidrome, tracker, decode, data_dir, model_name, measure=None, style=None,
                 sound=None, hearer=None, trial=None):
        self.store = store
        self.trial = trial  # beats.trial.Trial in the trial image (several measures of energy, side by side), else None
        # How a song sounds (beats.hear.Hearer: AudioMuse-AI's vectors and moods, made here), or None.
        self.hearer = hearer
        self.hearing = None  # {"id", "title", "artist", "since"}: a song analysed before, being heard for its sound
        self.sound = sound or measure_sound  # (path, samples, rate) -> loudness, fades and key
        self.measuring = None  # {"id", "title", "artist", "since"}: a song analysed before, getting its loudness
        # (ids) -> {id: loudness, fades, key}: what AudioMuse-AI's Siansa plugin already measured, taken instead of
        # downloading the song again (AudioMuse.profiles); None without AudioMuse-AI.
        self.importer = None
        self.style = style  # beats.style.StyleModel, or None
        self.styling = None  # {"id", "title", "artist", "since"}: a song analysed before, getting its styles
        self.genre_report = None  # beats.genres.agreement: how often the sound agrees with the tags
        self._taxonomy = None
        # The music servers (beats.servers.Servers), or one client for the container's own (the tests' way).
        self.servers = navidrome if hasattr(navidrome, "client") else _OneServer(navidrome)
        self.measure = measure or features.measure
        self.tracker = tracker  # (samples) -> (beats, downbeats)
        self.decode = decode  # (path) -> samples
        self.decode_rate = 22_050  # the rate decode gives (Beat This!'s)
        try:  # analyser.decode takes a rate; a test's stand-in may not
            import inspect

            self.decode_takes_rate = len(inspect.signature(decode).parameters) >= 2
        except (TypeError, ValueError):
            self.decode_takes_rate = False
        self.data_dir = data_dir
        self.model_name = model_name
        self.paused = False
        self.regridding = None  # {"done", "total"} while older results are worked out again (after an update)
        self.current = None  # {"id", "title", "artist", "step", "since"}: in the model
        self.preparing = None  # the same, for the next song being downloaded and decoded
        self.last_error = None
        self.scan_lock = threading.Lock()
        # What's being read right now, for the page (0.24.1: adding a library took minutes with nothing to show):
        # {"server", "server_name", "library", "libraries", "done", "songs", "started", "why"}, or None.
        self.reading = None
        self.last_work = time.time()  # when a song was last worked on: the models go after a while without one
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._ready = queue.Queue(maxsize=1)
        self._threads = []

    # ---- what the page asks for ----

    def start(self):
        if not self._threads:
            for target, name in ((self._prepare_loop, "beats-prepare"), (self._run, "beats-model"),
                                 (self._library_loop, "beats-library"), (self._style_loop, "beats-style"),
                                 (self._measure_loop, "beats-loudness"), (self._hear_loop, "beats-sound")):
                thread = threading.Thread(target=target, name=name, daemon=True)
                thread.start()
                self._threads.append(thread)

    def stop(self):
        self._stop.set()
        self._wake.set()

    def wake(self):
        self._wake.set()

    def pause(self, paused):
        self.paused = paused
        self.store.event("Paused" if paused else "Carrying on")
        self.wake()

    @property
    def navidrome(self):
        """The default server's client (its name from when Navidrome was the only one)."""
        return self.servers.client()

    def download(self, song_id, path):
        """The song's original file from the server it came from, or another that has it: its size in bytes."""
        problem = None
        for server, item_id in self.store.where(song_id) or [(MAIN, song_id)]:
            try:
                return self.servers.client(server).download(item_id, path)
            except ServerError as e:
                problem = problem or e
        raise problem

    def chosen(self, server=None):
        """The ids of ``server``'s libraries to analyse (kept across restarts): none until some are chosen."""
        server = server or self.servers.default_id
        folders = self.store.setting(folders_key(server))
        if folders is None and server == MAIN and self.store.setting("library"):  # 0.9.0's "whole library": every library
            folders = [f["id"] for f in self.servers.client(MAIN).libraries()]
            self.store.set_setting("folders", folders)
        return [str(f) for f in folders or []]

    def libraries(self, server=None):
        """A server's libraries, each with whether it's chosen and how many of its songs are here and analysed."""
        server = server or self.servers.default_id
        chosen, counts = set(self.chosen(server)), self.store.folder_counts(server)
        return [{**f, "on": f["id"] in chosen, **counts.get(f["id"], {"songs": 0, "done": 0})}
                for f in self.servers.client(server).libraries()]

    def _label(self, server):
        """The server's name in the activity list, when there are several: " on Emby"."""
        found = self.servers.get(server)
        return f" on {found['name']}" if found and self.servers.several() else ""

    def choose(self, ids, server=None, background=False):
        """Analyses ``server``'s libraries ``ids`` (and none of its others): reads them now; the others' songs not yet
        analysed leave the queue (those done stay). Returns ``{"songs", "queued"}``."""
        server = server or self.servers.default_id
        client = self.servers.client(server)
        known = {f["id"]: f["name"] for f in client.libraries()}
        ids = [str(i) for i in dict.fromkeys(ids)]
        unknown = [i for i in ids if i not in known]
        if unknown:
            raise ValueError(f"No library {', '.join(unknown)} on {client.label} (its libraries: "
                             f"{', '.join(f'{k} {v}' for k, v in known.items()) or 'none'})")
        self.store.set_setting(folders_key(server), ids)
        taken = self.store.unqueue_except(ids, server)
        names = ", ".join(known[i] for i in ids) or "none"
        self.store.event(f"Libraries to analyse{self._label(server)}: {names}"
                         + (f" ({taken} songs of the others left the queue)" if taken else ""))
        if not ids:
            return {"songs": 0, "queued": 0}
        if background:
            return self.read_in_background(server)
        return self.read_libraries(**self._what_to_read(server))

    def _what_to_read(self, server):
        """What reading ``server``'s libraries takes: itself alone, unless another server's songs are looked for among
        the default's and those were read before their files' paths were kept (0.23); then the default once first."""
        default = self.servers.default_id
        if server == default or self.store.setting(f"paths_read:{default}"):
            return {"only": server}
        return {"report": server}

    def read_in_background(self, server=None, every=False):
        """Reads ``server``'s chosen libraries (every server's with ``every``) in a thread, so the page can show how
        far it is (:attr:`reading`) instead of waiting: ``{"reading": True}``, or ``{"reading": False}`` when a read
        is already going (it's the one shown)."""
        if self.scan_lock.locked():
            return {"reading": False}
        server = server or self.servers.default_id
        args = {} if every else self._what_to_read(server)
        # Shown at once, before the thread has asked the server anything.
        self.reading = {"server": server, "server_name": (self.servers.get(server) or {}).get("name", server),
                        "library": None, "libraries": None, "done": 0, "songs": None, "started": time.time(), "why": None}

        def run():
            try:
                self.read_libraries(**args)
            except Exception as e:  # said in the activity; the page shows it
                if not isinstance(e, ServerError):
                    _log.exception("Reading the libraries went wrong")

        threading.Thread(target=run, name="beats-read", daemon=True).start()
        return {"reading": True}

    def remove_library(self, folder, server=None):
        """Forgets the songs of a library that's left out (its results too, unless another server has the song): the
        number removed."""
        server = server or self.servers.default_id
        folder = str(folder)
        if folder in self.chosen(server):
            raise ValueError("That library is being analysed: leave it out (and start the analysis) first")
        names = {f["id"]: f["name"] for f in self.servers.client(server).libraries()}
        removed = self.store.remove_folder(folder, server)
        self.store.event(f"Removed the {removed} songs of the library {names.get(folder, folder)}{self._label(server)} "
                         "(and what was found in them)")
        return removed

    def read_libraries(self, only=None, report=None):
        """Reads every song of the chosen libraries (of every server, the default first; of server ``only`` when
        given), adds them and queues the new ones. Returns ``{"songs", "queued"}``: of all that was read, or of server
        ``report`` alone."""
        with self.scan_lock:
            try:
                return self._read_libraries(only, report)
            finally:
                self.reading = None

    def _read_libraries(self, only, report):
        started = time.monotonic()
        total, queued = 0, []
        default = self.servers.default_id
        order = sorted((s["id"] for s in self.servers.all()), key=lambda i: i != default)
        todo = [(s, self.chosen(s)) for s in order if (only is None or s == only)]
        libraries, done = sum(len(c) for _, c in todo), 0
        if report is not None and report != default and any(s == default and c for s, c in todo):
            self.store.event(f"Reading {(self.servers.get(default) or {}).get('name', 'the default server')}'s libraries "
                             "first, this once: its songs' file paths weren't kept before 0.23, and the new server's "
                             "songs are found among them by those")
        for server, chosen in todo:
            client = self.servers.client(server)
            if not chosen:
                continue
            name = (self.servers.get(server) or {}).get("name", server)
            try:
                names = {f["id"]: f["name"] for f in client.libraries()}
            except ServerError as e:
                self.store.event(f"Couldn't read the libraries{self._label(server)}: {e}", "error")
                if only is not None or len(order) == 1:
                    raise
                continue
            for folder in chosen:
                self.reading = {"server": server, "server_name": name, "library": names.get(folder, folder),
                                "libraries": libraries, "done": done, "songs": None, "started": time.time(),
                                "why": None if report is None or server == report else "first"}
                done += 1
                try:
                    songs = [s for s in client.all_songs(folder) if s.get("id")]
                    self.reading = {**self.reading, "songs": len(songs)}
                except ServerError as e:
                    self.store.event(f"Couldn't read the library {names.get(folder, folder)}{self._label(server)}: {e}", "error")
                    if only is not None or len(order) == 1:
                        raise
                    continue
                result = self.store.link_songs(server, folder, songs, self.servers.twins(server))
                self._clean(server, folder, songs, names.get(folder, folder))
                if report is None or report == server:
                    total += len(songs)
                    queued += result["queued"]
                found = result["matched"]
                here = sum(found.values())
                self.store.event(
                    f"Library {names.get(folder, folder)}{self._label(server)}: {len(songs)} songs, "
                    f"{len(result['queued'])} new in the queue"
                    + (f", {here} already analysed from another server ("
                       + ", ".join(f"{n} by {'their file' if how == 'path' else 'their tags' if how == 'tags' else 'title and album' if how == 'title' else 'the same id'}"
                                   for how, n in sorted(found.items())) + ")" if here else ""))
        if only is None or only == default:
            self.store.set_setting(f"paths_read:{default}", True)  # its songs' paths are kept now
        self.store.set_setting("library_read_at", time.time())
        self.store.event(f"Read the libraries in {time.monotonic() - started:.0f}s")
        self.compare_with_tags()  # their tags may have changed
        self.wake()
        return {"songs": total, "queued": len(queued)}

    # A library that suddenly lists far fewer songs is more likely a server that's still scanning (or a disk that
    # isn't mounted) than a clear-out: nothing is removed then, and it's said.
    CLEAN_MOST = 25  # songs that may always go at once
    CLEAN_SHARE = 0.10  # or this share of the library, when that's more

    def _clean(self, server, folder, songs, name):
        """Forgets the songs a library no longer has (0.27, as AudioMuse-AI's cleaning does): removed from the
        server, they go from here too, with their analysis, unless another server still has them."""
        gone = self.store.gone_from(server, folder, [s["id"] for s in songs])
        if not gone:
            return 0
        had = len(gone) + len(songs)
        if not songs or len(gone) > max(self.CLEAN_MOST, self.CLEAN_SHARE * had):
            self.store.event(
                f"Library {name}{self._label(server)} lists {len(songs)} songs where it had {had}: {len(gone)} would "
                "be removed here, which is too many to be sure of. Nothing was removed; if they're really gone, untick "
                "the library and tick it again", "warn")
            return 0
        removed = self.store.remove_items(server, gone)
        self.store.event(f"Library {name}{self._label(server)}: {len(gone)} songs are gone from the server"
                         + (f"; {removed} forgotten here with their analysis" if removed else "")
                         + (f", {len(gone) - removed} kept (another server still has them)" if len(gone) - removed else ""))
        return removed

    def _any_chosen(self):
        return any(self.store.setting(folders_key(s["id"])) for s in self.servers.all()) or bool(self.chosen(MAIN))

    def _library_loop(self):
        """Reads the chosen libraries again every few hours, for songs added to the servers since."""
        while not self._stop.wait(60.0):
            try:
                if not self._any_chosen():
                    continue
            except ServerError:
                continue
            last = self.store.setting("library_read_at", 0) or 0
            if time.time() - last < LIBRARY_RESCAN_SECONDS:
                continue
            try:
                self.read_libraries()
            except Exception as e:  # the server away: try again next time
                self.store.set_setting("library_read_at", time.time())
                if isinstance(e, ServerError):
                    _log.warning("Couldn't read the libraries (tried again in %d hours): %s", LIBRARY_RESCAN_SECONDS // 3600, e)
                else:
                    _log.exception("Reading the libraries went wrong")

    def regrid(self):
        """Works out again, from their saved beats, the results of an older Siansa Beats (no model needed)."""
        stale = self.store.stale_results(grid.ANALYSIS_VERSION)
        if stale:
            self.store.event(f"Working out {len(stale)} songs again from their saved beats (an update changed how)…")
            self.regridding = {"done": 0, "total": len(stale)}
        started = time.monotonic()
        for song_id, model in stale:
            try:
                row = self.store.result(song_id, model, with_beats=True)
                song = self.store.song(song_id) or {}
                if row and row.get("beats"):
                    self.store.update_result(song_id, model, grid.analyse(row["beats"], row["downbeats"],
                                                                          song.get("duration") or row["beats"][-1],
                                                                          row.get("levels")))
            except Exception as e:  # one odd song never stops the rest
                _log.debug("Working out %s again failed", song_id, exc_info=True)
                self.store.event(f"Couldn't work out {song_id} again: {e}", "warn")
            self.regridding["done"] += 1
        self.regridding = None
        if stale:
            self.store.event(f"Worked out {len(stale)} songs again from their saved beats in "
                             f"{time.monotonic() - started:.0f}s")
        # Songs that failed on a bug since fixed (0.15.0: short tracks off mixed CDs, bars past the measured sound)
        # are analysed again, once: a song that fails again keeps its new error.
        retry = [r["id"] for r in self.store.failed() if any(sign in (r.get("error") or "") for sign in FIXED_ERRORS)]
        for song_id in retry:
            self.store.queue(song_id, force=True)
        if retry:
            self.store.event(f"Analysing {len(retry)} failed songs again: what failed them is fixed")
            self.wake()
        unheard = self.store.without_levels()
        for song_id in unheard:
            self.store.queue(song_id, force=True)
        if unheard:
            self.store.event(f"Analysing {len(unheard)} songs again to measure their sound, for their phrases")
            self.wake()
        return len(stale)

    # ---- genres: styles from the sound ----

    def taxonomy(self):
        """The style model's 400 styles, findable by name, or ``None`` without the model."""
        if self._taxonomy is None and self.style is not None and self.style.available:
            from . import style as style_module

            self._taxonomy = genres.Taxonomy(style_module.labels(self.style.models_dir))
        return self._taxonomy

    def _style(self, song, samples, rate):
        """Gives a song its styles from ``samples``: its genres (:mod:`beats.genres`). Never fails the song."""
        if self.style is None or not self.style.available:
            return
        first = not self.style.loaded
        try:
            started = time.monotonic()
            ranked = self.style(samples, rate)
            seconds = time.monotonic() - started
            if first:
                where = getattr(self.style, "backend", None) or "?"
                self.store.event(f"Style model (Discogs-EffNet) loaded on the {where} in {self.style.load_seconds or 0:.1f}s")
                if getattr(self.style, "problem", None):
                    self.store.event(self.style.problem, "warn")
            top = ranked[:STYLE_KEEP]
            self.store.set_genres(song["id"], "sound", [l for l, _ in top], [round(v, 4) for _, v in top])
            _log.debug("%s – %s: sounds like %s (%.2fs)", song.get("artist"), song.get("title"),
                       ", ".join(f"{l.partition('---')[2]} {v:.2f}" for l, v in top[:3]), seconds)
        except Exception as e:
            _log.warning("%s – %s: couldn't hear its style: %s", song.get("artist"), song.get("title"), e,
                         exc_info=not isinstance(e, RuntimeError))
            self.store.set_genres(song["id"], "sound-failed", [str(e)[:200] or e.__class__.__name__])

    def compare_with_tags(self):
        """Works out how often the songs' sound agrees with their tags (:func:`beats.genres.agreement`, shown on the
        page; the tags play no part in a song's genres). The number of songs with styles."""
        taxonomy = self.taxonomy()
        if taxonomy is None:
            return 0
        rows = self.store.genre_rows()
        report = [(by["sound"], _merge(taxonomy.match(label) for label, _ in by.get("tag", [])))
                  for by in rows.values() if by.get("sound")]
        self.genre_report = genres.agreement(report)
        return len(report)

    def _style_loop(self):
        """Gives the songs analysed before the style model their styles, one at a time, while the queue is empty."""
        done_since_report = 0
        while not self._stop.wait(2.0):
            if self.style is None or not self.style.available or self.paused:
                continue
            if self.current or self.preparing or self.store.counts().get("queued"):
                continue  # new songs first: they get their styles on the way
            waiting = self.store.songs_without_style(limit=20)
            if not waiting:
                if done_since_report:
                    self.compare_with_tags()
                    done_since_report = 0
                self._stop.wait(60.0)
                continue
            for song_id in waiting:
                if self._stop.is_set() or self.paused or self.store.counts().get("queued"):
                    break
                self._style_again(song_id)
                done_since_report += 1
                if done_since_report >= STYLE_REPORT_EVERY:
                    self.compare_with_tags()
                    done_since_report = 0

    def _style_again(self, song_id):
        self.last_work = time.time()
        song = self.store.song(song_id) or {"id": song_id, "title": song_id, "artist": ""}
        self.styling = {"id": song_id, "title": song.get("title"), "artist": song.get("artist"), "since": time.time()}
        path = None
        try:
            fd, path = tempfile.mkstemp(prefix="style-", dir=self.data_dir)
            os.close(fd)
            started = time.monotonic()
            self.download(song_id, path)
            downloaded = time.monotonic() - started
            samples = self.decode(path, STYLE_RATE) if self.decode_takes_rate else self.decode(path)
            decoded = time.monotonic() - started - downloaded
            self._style(song, samples, STYLE_RATE if self.decode_takes_rate else self.decode_rate)
            _log.debug("%s – %s: style of a song analysed before: download %.2fs, decode %.2fs, model %.2fs",
                       song.get("artist"), song.get("title"), downloaded, decoded,
                       time.monotonic() - started - downloaded - decoded)
        except Exception as e:  # Navidrome away, a file ffmpeg can't read: marked, so it isn't tried again and again
            _log.warning("%s – %s: couldn't get its sound for its style: %s", song.get("artist"), song.get("title"), e)
            self.store.set_genres(song_id, "sound-failed", [str(e)[:200] or e.__class__.__name__])
        finally:
            self.styling = None
            if path and os.path.exists(path):
                os.remove(path)

    def _measure(self, song, path, samples, rate):
        """Measures the song's loudness, fades and key, keeping why when it can't: never stops its analysis."""
        started = time.monotonic()
        try:
            values = self.sound(path, samples, rate)
            self.store.save_measure(song["id"], values, loudness.VERSION)
            _log.debug("%s – %s: %s LUFS, key %s %s, in %.2fs", song.get("artist"), song.get("title"),
                       values.get("ref_lufs"), values.get("key"), values.get("scale"), time.monotonic() - started)
        except Exception as e:
            _log.warning("%s – %s: couldn't measure its loudness: %s", song.get("artist"), song.get("title"), e)
            self.store.save_measure(song["id"], {}, loudness.VERSION, str(e) or e.__class__.__name__)

    # ---- how a song sounds (0.27): AudioMuse-AI's vectors and moods, made here ----

    HEAR_IDLE_SECONDS = 60.0

    def _hear(self, song, path):
        """Hears the song at ``path`` for its sound and keeps it (or why not): never stops its analysis. True when
        it was heard."""
        if self.hearer is None or not self.hearer.available:
            return False
        started, first = time.monotonic(), not self.hearer.loaded
        try:
            values = self.hearer(path)
            self.store.save_sound([{"song_id": song["id"], **values}], source="beats")
            if first:
                self.store.event(f"The models that hear a song's sound loaded in {self.hearer.load_seconds or 0:.1f}s")
            _log.debug("%s – %s: heard in %.2fs (%s)", song.get("artist"), song.get("title"),
                       time.monotonic() - started, values.get("moods"))
            return True
        except Exception as e:
            _log.warning("%s – %s: couldn't hear its sound: %s", song.get("artist"), song.get("title"), e)
            self.store.sound_failed(song["id"], str(e) or e.__class__.__name__)
            return False

    def _hear_loop(self):
        """Hears the songs whose sound isn't known (analysed before 0.27, and never by AudioMuse-AI), one at a time,
        while nothing else is being done (new songs are heard on the way)."""
        while not self._stop.wait(5.0):
            if self.hearer is None or not self.hearer.available:
                self._stop.wait(self.HEAR_IDLE_SECONDS)
                continue
            if self.paused or self.current or self.preparing or self.styling or self.measuring \
                    or self.store.counts().get("queued"):
                continue
            waiting = self.store.without_sound(limit=20)
            if not waiting:
                self._stop.wait(self.HEAR_IDLE_SECONDS)
                continue
            for song_id in waiting:
                if self._stop.is_set() or self.paused or self.store.counts().get("queued"):
                    break
                self._hear_again(song_id)

    def _hear_again(self, song_id):
        self.last_work = time.time()
        song = self.store.song(song_id) or {"id": song_id, "title": song_id, "artist": ""}
        self.hearing = {"id": song_id, "title": song.get("title"), "artist": song.get("artist"), "since": time.time()}
        path = None
        try:
            fd, path = tempfile.mkstemp(prefix="sound-", dir=self.data_dir)
            os.close(fd)
            self.download(song_id, path)
            self._hear(song, path)
        except Exception as e:  # the server away, a file gone: kept, so it isn't tried again and again
            _log.warning("%s – %s: couldn't get the song to hear its sound: %s", song.get("artist"), song.get("title"), e)
            self.store.sound_failed(song_id, str(e) or e.__class__.__name__)
        finally:
            self.hearing = None
            if path and os.path.exists(path):
                os.remove(path)

    def _measure_loop(self):
        """Measures the loudness, fades and key of songs analysed before 0.20, one at a time, while nothing else is
        being done (new songs get them on the way)."""
        while not self._stop.wait(3.0):
            if self.paused or self.current or self.preparing or self.styling or self.store.counts().get("queued"):
                continue
            waiting = self.store.without_measure(loudness.VERSION, limit=20)
            if not waiting:
                self._stop.wait(self.MEASURE_IDLE_SECONDS)
                continue
            if self.importer and self._import(waiting):
                continue  # some came from the plugin: look again for the rest
            for song_id in waiting:
                if self._stop.is_set() or self.paused or self.styling or self.store.counts().get("queued"):
                    break
                self._measure_again(song_id)

    def _import(self, waiting):
        """Takes what the Siansa plugin on AudioMuse-AI already measured of the songs left to measure, a few hundred at a
        time, each asked for once: True while there were songs to ask about. Then the rest are measured here."""
        tried = self.__dict__.setdefault("_import_tried", set())
        ids = [i for i in self.store.without_measure(loudness.VERSION, limit=len(tried) + 500) if i not in tried][:500]
        if not ids:
            self.importer = None
            _log.info("Took what AudioMuse-AI's Siansa plugin had measured; the rest are measured here")
            return False
        tried.update(ids)
        try:
            found = self.importer(ids)
        except Exception as e:  # AudioMuse-AI away: the songs are measured here instead
            _log.info("Couldn't take loudness from AudioMuse-AI's Siansa plugin: %s", e)
            self.importer = None
            return False
        for song_id, values in found.items():
            self.store.save_measure(song_id, values, loudness.VERSION, source="plugin")
        if found:
            self.store.event(f"Loudness, fades and key of {len(found)} songs taken from AudioMuse-AI's Siansa plugin")
        return True

    def _measure_again(self, song_id):
        self.last_work = time.time()
        song = self.store.song(song_id) or {"id": song_id, "title": song_id, "artist": ""}
        self.measuring = {"id": song_id, "title": song.get("title"), "artist": song.get("artist"), "since": time.time()}
        path = None
        try:
            fd, path = tempfile.mkstemp(prefix="loudness-", dir=self.data_dir)
            os.close(fd)
            self.download(song_id, path)
            samples = self.decode(path)
            self._measure(song, path, samples, self.decode_rate)
        except Exception as e:  # Navidrome away, a file ffmpeg can't read: kept, so it isn't tried again and again
            _log.warning("%s – %s: couldn't get its sound for its loudness: %s", song.get("artist"), song.get("title"), e)
            self.store.save_measure(song_id, {}, loudness.VERSION, str(e) or e.__class__.__name__)
        finally:
            self.measuring = None
            if path and os.path.exists(path):
                os.remove(path)

    # ---- memory (0.34): the models go when there's nothing to analyse ----

    def busy(self):
        """Whether a song is being worked on right now, in any of the ways."""
        return bool(self.current or self.preparing or self.styling or self.hearing or self.measuring or self.regridding)

    def let_models_go_if_idle(self, now=None, idle=memory.MODELS_IDLE):
        """Lets Beat This!, the style model and the models that hear a song go after ``idle`` seconds without a song
        to analyse, and hands their memory back; they load again with the next song. True when any went."""
        now = now or time.time()
        if self.busy() or now - self.last_work < idle:
            return False
        before = memory.rss_mb()
        gone = [name for name, model in (("Beat This!", self.tracker), ("the style model", self.style),
                                         ("the models that hear a song", self.hearer), ("the trial's models", self.trial))
                if model is not None and callable(getattr(model, "unload", None)) and model.unload()]
        if not gone:
            return False
        after = memory.trim()
        held = f" · memory {memory.said(before)}, now {memory.said(after)}" if before is not None and after is not None else ""
        self.store.event(f"Let go after {int(idle // 60)} minutes without a song to analyse: {', '.join(gone)}{held}")
        return True

    # ---- the work ----

    def _prepare_loop(self):
        """Downloads, decodes and measures the next song while the model works on this one."""
        while not self._stop.is_set():
            song_id = None if self.paused else self.store.claim_next()
            if song_id is not None:
                _log.debug("Next song: %s", song_id)
            if song_id is None:
                self._wake.wait(timeout=5.0)
                self._wake.clear()
                continue
            prepared = self.prepare(song_id)
            while prepared and not self._stop.is_set():
                try:
                    self._ready.put(prepared, timeout=1.0)
                    break
                except queue.Full:
                    continue
            self.preparing = None  # handed to the model (shown as "next" until then)

    def _run(self):
        """Runs the model on each prepared song in turn."""
        while not self._stop.is_set():
            try:
                prepared = self._ready.get(timeout=1.0)
            except queue.Empty:
                continue
            self.finish(prepared)

    def _step(self, holder, song_id, step):
        self.store.set_state(song_id, step)
        if holder:
            holder["step"] = step
            holder["since"] = time.time()

    def work(self, song_id):
        """One song from start to end, in this thread."""
        prepared = self.prepare(song_id)
        self.preparing = None
        if prepared:
            self.finish(prepared)

    def prepare(self, song_id):
        """Downloads, decodes and measures the song: what :meth:`finish` needs, or ``None`` when it failed."""
        self.last_work = time.time()
        song = self.store.song(song_id) or {"id": song_id, "title": song_id, "artist": "", "duration": 0}
        holder = self.preparing = {"id": song_id, "title": song["title"], "artist": song["artist"], "step": None,
                                   "since": time.time()}
        path = None
        label = f"{song['artist']} – {song['title']} ({song_id})"
        _log.debug("%s: preparing (%s, %.0fs long)", label, song.get("suffix") or "?", song.get("duration") or 0)
        try:
            self._step(holder, song_id, "downloading")
            started = time.monotonic()
            fd, path = tempfile.mkstemp(prefix="song-", dir=self.data_dir)
            os.close(fd)
            size = self.download(song_id, path)
            download_seconds = time.monotonic() - started
            _log.debug("%s: downloaded %.1f MB in %.2fs", label, size / 1e6, download_seconds)

            self._step(holder, song_id, "decoding")
            started = time.monotonic()
            samples = self.decode(path)
            decode_seconds = time.monotonic() - started
            _log.debug("%s: decoded %d samples (%.1fs of sound) in %.2fs", label, len(samples), len(samples) / 22_050.0,
                       decode_seconds)
            self._style(song, samples, self.decode_rate)
            self._measure(song, path, samples, self.decode_rate)
            self._hear(song, path)
            if self.trial is not None:
                self.trial.hear(song, path)
            try:
                levels = self.measure(samples)
                _log.debug("%s: measured the sound", label)
            except Exception as e:  # the phrases need it, the beats don't: carry on without
                levels = None
                _log.debug("%s: measuring the sound failed", label, exc_info=True)
                self.store.event(f"{song['artist']} – {song['title']}: couldn't measure the sound for phrases: {e}", "warn")
            self._step(holder, song_id, "ready")
            return {"song": song, "samples": samples, "levels": levels, "size": size,
                    "download_seconds": download_seconds, "decode_seconds": decode_seconds}
        except Exception as e:  # one song failing never stops the queue
            self.preparing = None
            self._failed(song, e)
            return None
        finally:
            if path and os.path.exists(path):
                os.remove(path)

    def _failed(self, song, e):
        message = str(e) or e.__class__.__name__
        self.last_error = message
        self.store.set_state(song["id"], "failed", message)
        self.store.event(f"{song['artist']} – {song['title']} failed: {message}", "error")
        # The whole story for anything unexpected; Navidrome's and the decoder's own errors say enough.
        if isinstance(e, (SubsonicError, RuntimeError)):
            _log.debug("%s (%s): %s", song.get("title"), song["id"], e.__class__.__name__)
        else:
            _log.error("Unexpected error on %s (%s)", song.get("title"), song["id"], exc_info=e)

    def finish(self, prepared):
        """Runs the model on a prepared song and saves what it found."""
        song, samples, levels = prepared["song"], prepared["samples"], prepared["levels"]
        song_id, size = song["id"], prepared["size"]
        download_seconds, decode_seconds = prepared["download_seconds"], prepared["decode_seconds"]
        holder = self.current = {"id": song_id, "title": song["title"], "artist": song["artist"], "step": None,
                                 "since": time.time()}
        try:
            duration = len(samples) / 22_050.0
            self._step(holder, song_id, "analysing")
            first = not getattr(self.tracker, "loaded", True)
            started = time.monotonic()
            beats, downbeats = self.tracker(samples)
            model_seconds = time.monotonic() - started
            if first:  # the first song also loaded the model: its time is the song's alone
                model_seconds -= getattr(self.tracker, "load_seconds", 0) or 0
            if first:
                where = getattr(self.tracker, "device_name", None) or getattr(self.tracker, "device", None) or "?"
                self.store.event(f"Model {self.model_name} loaded on {where} in "
                                 f"{getattr(self.tracker, 'load_seconds', 0) or 0:.1f}s")
                note = getattr(self.tracker, "device_note", None)
                if note:
                    self.store.event(note, "warn")

            _log.debug("%s – %s: the model found %d beats, %d downbeats in %.2fs", song["artist"], song["title"],
                       len(beats), len(downbeats), model_seconds)
            values = grid.analyse(beats, downbeats, duration or song["duration"], levels)
            _log.debug("%s – %s: %s", song["artist"], song["title"],
                       ", ".join(f"{k} {round(values[k], 3) if isinstance(values[k], float) else values[k]}" for k in ("bpm", "grid_bpm", "steady", "on_grid", "beats_per_bar",
                                                                    "first_downbeat_ms", "phrase_bars", "phrase_count",
                                                                    "downbeat_source") if k in values))
            values.update(download_seconds=download_seconds, decode_seconds=decode_seconds, model_seconds=model_seconds)
            self.store.save_result(song_id, self.model_name, values, beats, downbeats, levels)
            self.store.set_state(song_id, "done")
            bpm = f"{values['bpm']:.2f} BPM" if values.get("bpm") else "no steady beat"
            self.store.event(f"{song['artist']} – {song['title']}: {bpm}, {len(beats)} beats, "
                             f"{values.get('beats_per_bar') or '?'} a bar · {size / 1e6:.1f} MB in "
                             f"{download_seconds:.1f}s, decoded in {decode_seconds:.1f}s, model {model_seconds:.1f}s")
        except Exception as e:  # one song failing never stops the queue
            self._failed(song, e)
        finally:
            self.current = None
            self.last_work = time.time()
            # The song's sound, back to the system before the next: whoever handed it over may still hold the dict.
            prepared.clear()
            del samples, levels
            memory.trim()


def _merge(matches):
    styles, found = set(), set()
    for s, g in matches:
        styles |= s
        found |= g
    return styles, found


class _OneServer:
    """The music servers when there's only a client for one (the tests make a Worker with one): the container's own."""

    def __init__(self, client):
        self._client = client
        self.default_id = MAIN

    def client(self, server_id=None):
        return self._client

    def all(self):
        return [{"id": MAIN, "kind": "navidrome", "name": "Navidrome"}]

    def get(self, server_id):
        return self.all()[0] if server_id == MAIN else None

    def several(self):
        return False

    def twins(self, server_id):
        return []
