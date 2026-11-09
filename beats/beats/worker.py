"""The queue: from Navidrome to Beat This! to the database, with what it's doing on show.

Each song is downloaded (its original file, to a temporary file in the data folder), decoded, given to the model,
and its beats turned into the readings :mod:`grid` works out. The temporary file is deleted straight after. Two
threads share the work: one downloads, decodes and measures the next song while the other runs the model on this
one, so the model (the GPU) rarely waits for the network.

The songs come from the Navidrome libraries chosen on the page (all, or some: leaving out compilations, say). They're
read again every few hours for new songs.
"""

import os
import queue
import tempfile
import threading
import time

from . import features
from . import grid
from . import log
from .subsonic import SubsonicError

_log = log.get("worker")

# Errors of bugs that are fixed: songs that failed with them are analysed again on start.
FIXED_ERRORS = ("'bars'", "not supported between instances of 'NoneType'")


LIBRARY_RESCAN_SECONDS = 6 * 3600  # how often the chosen libraries are read again for new songs


class Worker:
    def __init__(self, store, navidrome, tracker, decode, data_dir, model_name, measure=None):
        self.store = store
        self.navidrome = navidrome
        self.measure = measure or features.measure
        self.tracker = tracker  # (samples) -> (beats, downbeats)
        self.decode = decode  # (path) -> samples
        self.data_dir = data_dir
        self.model_name = model_name
        self.paused = False
        self.regridding = None  # {"done", "total"} while older results are worked out again (after an update)
        self.current = None  # {"id", "title", "artist", "step", "since"}: in the model
        self.preparing = None  # the same, for the next song being downloaded and decoded
        self.last_error = None
        self.scan_lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._ready = queue.Queue(maxsize=1)
        self._threads = []

    # ---- what the page asks for ----

    def start(self):
        if not self._threads:
            for target, name in ((self._prepare_loop, "beats-prepare"), (self._run, "beats-model"),
                                 (self._library_loop, "beats-library")):
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

    def chosen(self):
        """The ids of the Navidrome libraries to analyse (kept across restarts): none until some are chosen."""
        folders = self.store.setting("folders")
        if folders is None and self.store.setting("library"):  # 0.9.0's "whole library": every library
            folders = [f["id"] for f in self.navidrome.libraries()]
            self.store.set_setting("folders", folders)
        return [str(f) for f in folders or []]

    def libraries(self):
        """Navidrome's libraries, each with whether it's chosen and how many of its songs are here and analysed."""
        chosen, counts = set(self.chosen()), self.store.folder_counts()
        return [{**f, "on": f["id"] in chosen, **counts.get(f["id"], {"songs": 0, "done": 0})}
                for f in self.navidrome.libraries()]

    def choose(self, ids):
        """Analyses the libraries ``ids`` (and no others): reads them now; the others' songs not yet analysed leave
        the queue (those done stay). Returns ``{"songs", "queued"}``."""
        known = {f["id"]: f["name"] for f in self.navidrome.libraries()}
        ids = [str(i) for i in dict.fromkeys(ids)]
        unknown = [i for i in ids if i not in known]
        if unknown:
            raise ValueError(f"No library {', '.join(unknown)} on Navidrome (its libraries: "
                             f"{', '.join(f'{k} {v}' for k, v in known.items()) or 'none'})")
        self.store.set_setting("folders", ids)
        taken = self.store.unqueue_except(ids)
        names = ", ".join(known[i] for i in ids) or "none"
        self.store.event(f"Libraries to analyse: {names}" + (f" ({taken} songs of the others left the queue)" if taken else ""))
        return self.read_libraries() if ids else {"songs": 0, "queued": 0}

    def remove_library(self, folder):
        """Forgets the songs of a library that's left out (its results too): the number removed."""
        folder = str(folder)
        if folder in self.chosen():
            raise ValueError("That library is being analysed: leave it out (and start the analysis) first")
        names = {f["id"]: f["name"] for f in self.navidrome.libraries()}
        removed = self.store.remove_folder(folder)
        self.store.event(f"Removed the {removed} songs of the library {names.get(folder, folder)} (and what was found "
                         "in them)")
        return removed

    def read_libraries(self):
        """Reads every song of the chosen libraries from Navidrome, adds them and queues the new ones. Returns
        ``{"songs", "queued"}``."""
        with self.scan_lock:
            started = time.monotonic()
            names = {f["id"]: f["name"] for f in self.navidrome.libraries()}
            total, queued = 0, []
            for folder in self.chosen():
                try:
                    songs = [s for s in self.navidrome.all_songs(folder) if s.get("id")]
                except SubsonicError as e:
                    self.store.event(f"Couldn't read the library {names.get(folder, folder)}: {e}", "error")
                    raise
                added = self.store.add_songs(songs, folder)
                total += len(songs)
                queued += added
                self.store.event(f"Library {names.get(folder, folder)}: {len(songs)} songs, {len(added)} new in the queue")
            self.store.set_setting("library_read_at", time.time())
            self.store.event(f"Read the libraries in {time.monotonic() - started:.0f}s")
            self.wake()
            return {"songs": total, "queued": len(queued)}

    def _library_loop(self):
        """Reads the chosen libraries again every few hours, for songs added to Navidrome since."""
        while not self._stop.wait(60.0):
            try:
                if not self.chosen():
                    continue
            except SubsonicError:
                continue
            last = self.store.setting("library_read_at", 0) or 0
            if time.time() - last < LIBRARY_RESCAN_SECONDS:
                continue
            try:
                self.read_libraries()
            except Exception as e:  # Navidrome away: try again next time
                self.store.set_setting("library_read_at", time.time())
                if isinstance(e, SubsonicError):
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
        retry = [r["id"] for r in self.store.overview(None, None, "failed", None)
                 if any(sign in (r.get("error") or "") for sign in FIXED_ERRORS)]
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
        song = self.store.song(song_id) or {"id": song_id, "title": song_id, "artist": "", "duration": 0}
        holder = self.preparing = {"id": song_id, "title": song["title"], "artist": song["artist"], "step": None,
                                   "since": time.time()}
        path = None
        label = f"{song['artist']} – {song['title']} ({song_id})"
        _log.debug("%s: preparing (%s, %.0fs on Navidrome)", label, song.get("suffix") or "?", song.get("duration") or 0)
        try:
            self._step(holder, song_id, "downloading")
            started = time.monotonic()
            fd, path = tempfile.mkstemp(prefix="song-", dir=self.data_dir)
            os.close(fd)
            size = self.navidrome.download(song_id, path)
            download_seconds = time.monotonic() - started
            _log.debug("%s: downloaded %.1f MB in %.2fs", label, size / 1e6, download_seconds)

            self._step(holder, song_id, "decoding")
            started = time.monotonic()
            samples = self.decode(path)
            decode_seconds = time.monotonic() - started
            _log.debug("%s: decoded %d samples (%.1fs of sound) in %.2fs", label, len(samples), len(samples) / 22_050.0,
                       decode_seconds)
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
