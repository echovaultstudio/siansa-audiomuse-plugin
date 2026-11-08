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
import traceback

from . import grid
from . import features
from .audiomuse import AudioMuseError
from .subsonic import SubsonicError


LIBRARY_RESCAN_SECONDS = 6 * 3600  # how often the chosen libraries are read again for new songs


class Worker:
    def __init__(self, store, navidrome, audiomuse, tracker, decode, data_dir, model_name, measure=None):
        self.store = store
        self.navidrome = navidrome
        self.audiomuse = audiomuse
        self.measure = measure or features.measure
        self.plugin_state = {"state": "waiting", "text": "Not asked yet"}
        self.tracker = tracker  # (samples) -> (beats, downbeats)
        self.decode = decode  # (path) -> samples
        self.data_dir = data_dir
        self.model_name = model_name
        self.paused = False
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
            if queued:  # the plugin's readings of the new songs, for comparison: in the background, as they can be many
                threading.Thread(target=self.refresh_plugin, args=(queued,), name="plugin", daemon=True).start()
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
                if not isinstance(e, SubsonicError):
                    traceback.print_exc()

    def regrid(self):
        """Works out again, from their saved beats, the results of an older Siansa Beats (no model needed)."""
        stale = self.store.stale_results(grid.ANALYSIS_VERSION)
        for song_id, model in stale:
            row = self.store.result(song_id, model, with_beats=True)
            song = self.store.song(song_id) or {}
            if row and row.get("beats"):
                self.store.update_result(song_id, model, grid.analyse(row["beats"], row["downbeats"],
                                                                      song.get("duration") or row["beats"][-1],
                                                                      row.get("levels"),
                                                                      [g["label"] for g in self.store.genres(song_id)]))
        if stale:
            self.store.event(f"Worked out the tempo again for {len(stale)} songs from their saved beats")
        unheard = self.store.without_levels()
        for song_id in unheard:
            self.store.queue(song_id, force=True)
        if unheard:
            self.store.event(f"Analysing {len(unheard)} songs again to measure their sound, for their phrases")
            self.wake()
        return len(stale)

    def check_plugin(self):
        """Asks the Siansa plugin whether it's there, for the page's plugin line."""
        if not self.audiomuse.configured:
            self.plugin_state = {"state": "off", "text": "Set AUDIOMUSE_URL (and AUDIOMUSE_TOKEN if AudioMuse-AI "
                                                         "has a login) to compare with the plugin"}
            return self.plugin_state
        try:
            health = self.audiomuse.health()
            if health.get("status") == "error":
                raise AudioMuseError(health.get("error") or "The plugin says it isn't working: AudioMuse-AI's log says more")
            version = health.get("plugin_version", "?")
            self.plugin_state = {"state": "ok", "version": version, "text": f"Siansa plugin {version} answers"}
            self.store.event(f"Siansa plugin {version} answers on AudioMuse-AI")
        except AudioMuseError as e:
            self.plugin_state = {"state": "error", "text": f"The Siansa plugin doesn't answer: {e}"}
            self.store.event(self.plugin_state["text"], "warn")
        return self.plugin_state

    def refresh_plugin(self, ids):
        """The Siansa plugin's readings of ``ids``, kept for comparison. Quietly skipped without AudioMuse-AI."""
        if not self.audiomuse.configured or not ids:
            self.check_plugin()
            return 0
        try:
            found = self.audiomuse.profiles(ids)
        except AudioMuseError as e:
            self.plugin_state = {"state": "error", "text": f"Couldn't read the plugin's readings: {e}"}
            self.store.event(self.plugin_state["text"], "warn")
            return 0
        with_beats = 0
        for song_id, values in found.items():
            if any(values.get(f) is not None for f in ("intro_bpm", "outro_bpm")):
                self.store.save_plugin(song_id, values)
                with_beats += 1
        if not found:
            text = (f"The plugin knows none of these {len(ids)} songs: has AudioMuse-AI analysed them, and is it "
                    "set up with this Navidrome?")
            state = "warn"
        elif with_beats < len(found):
            text = (f"Plugin readings for {with_beats} of {len(ids)} songs ({len(found) - with_beats} measured "
                    "before the plugin read beats: it does them as it goes)")
            state = "ok" if with_beats else "warn"
        else:
            text, state = f"Plugin readings for {with_beats} of {len(ids)} songs", "ok"
        self.plugin_state = {"state": state, "text": text, "found": with_beats, "asked": len(ids)}
        self.store.event(text, "info" if state == "ok" else "warn")
        return with_beats

    # ---- the work ----

    def _prepare_loop(self):
        """Downloads, decodes and measures the next song while the model works on this one."""
        while not self._stop.is_set():
            song_id = None if self.paused else self.store.claim_next()
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
        try:
            self._step(holder, song_id, "downloading")
            started = time.monotonic()
            fd, path = tempfile.mkstemp(prefix="song-", dir=self.data_dir)
            os.close(fd)
            size = self.navidrome.download(song_id, path)
            download_seconds = time.monotonic() - started

            self._step(holder, song_id, "decoding")
            started = time.monotonic()
            samples = self.decode(path)
            decode_seconds = time.monotonic() - started
            try:
                levels = self.measure(samples)
            except Exception as e:  # the phrases need it, the beats don't: carry on without
                levels = None
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
        if not isinstance(e, (SubsonicError, RuntimeError)):
            traceback.print_exc()

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

            genres = [g["label"] for g in self.store.genres(song_id)]
            values = grid.analyse(beats, downbeats, duration or song["duration"], levels, genres)
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
