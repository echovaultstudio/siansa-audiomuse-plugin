"""The queue: one song at a time, from Navidrome to Beat This! to the database, with what it's doing on show.

Each song is downloaded (its original file, to a temporary file in the data folder), decoded, given to the model,
and its beats turned into the readings :mod:`grid` works out. The temporary file is deleted straight after.
"""

import os
import tempfile
import threading
import time
import traceback

from . import grid
from .audiomuse import AudioMuseError
from .subsonic import SubsonicError


class Worker:
    def __init__(self, store, navidrome, audiomuse, tracker, decode, data_dir, model_name):
        self.store = store
        self.navidrome = navidrome
        self.audiomuse = audiomuse
        self.tracker = tracker  # (samples) -> (beats, downbeats)
        self.decode = decode  # (path) -> samples
        self.data_dir = data_dir
        self.model_name = model_name
        self.paused = False
        self.current = None  # {"id", "title", "artist", "step", "since"}
        self.last_error = None
        self.scan_lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = None

    # ---- what the page asks for ----

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="beats-worker", daemon=True)
            self._thread.start()

    def stop(self):
        self._stop.set()
        self._wake.set()

    def wake(self):
        self._wake.set()

    def pause(self, paused):
        self.paused = paused
        self.store.event("Paused" if paused else "Carrying on")
        self.wake()

    def scan(self, playlist_name):
        """Reads the playlist ``playlist_name`` from Navidrome, adds its songs, queues the new ones, and fetches the
        plugin's readings for them. Returns ``{"songs", "queued"}``."""
        with self.scan_lock:
            try:
                found = self.navidrome.playlist_named(playlist_name)
                if not found:
                    names = ", ".join(p["name"] for p in self.navidrome.playlists()) or "none"
                    raise SubsonicError(f'No playlist called "{playlist_name}" on Navidrome (its playlists: {names})')
                songs = self.navidrome.playlist_songs(found["id"])
            except SubsonicError as e:
                self.store.event(f"Couldn't read the playlist: {e}", "error")
                raise
            queued = 0
            for song in songs:
                if not song.get("id"):
                    continue
                self.store.add_song(song, playlist=found["name"])
                queued += self.store.queue(song["id"])
            self.store.event(f'Playlist "{found["name"]}": {len(songs)} songs, {queued} new in the queue')
            self.refresh_plugin([s["id"] for s in songs if s.get("id")])
            self.wake()
            return {"songs": len(songs), "queued": queued}

    def refresh_plugin(self, ids):
        """The Siansa plugin's readings of ``ids``, kept for comparison. Quietly skipped without AudioMuse-AI."""
        if not self.audiomuse.configured or not ids:
            return 0
        try:
            found = self.audiomuse.profiles(ids)
        except AudioMuseError as e:
            self.store.event(f"Couldn't read the plugin's readings from AudioMuse-AI: {e}", "warn")
            return 0
        for song_id, values in found.items():
            if any(values.get(f) is not None for f in ("intro_bpm", "outro_bpm")):
                self.store.save_plugin(song_id, values)
        self.store.event(f"Plugin readings: {len(found)} of {len(ids)} songs measured by the Siansa plugin")
        return len(found)

    # ---- the work ----

    def _run(self):
        while not self._stop.is_set():
            song_id = None if self.paused else self.store.next_queued()
            if song_id is None:
                self._wake.wait(timeout=5.0)
                self._wake.clear()
                continue
            self.work(song_id)

    def _step(self, song_id, step):
        self.store.set_state(song_id, step)
        if self.current:
            self.current["step"] = step
            self.current["since"] = time.time()

    def work(self, song_id):
        song = self.store.song(song_id) or {"id": song_id, "title": song_id, "artist": "", "duration": 0}
        self.current = {"id": song_id, "title": song["title"], "artist": song["artist"], "step": None,
                        "since": time.time()}
        path = None
        try:
            self._step(song_id, "downloading")
            started = time.monotonic()
            fd, path = tempfile.mkstemp(prefix="song-", dir=self.data_dir)
            os.close(fd)
            size = self.navidrome.download(song_id, path)
            download_seconds = time.monotonic() - started

            self._step(song_id, "decoding")
            started = time.monotonic()
            samples = self.decode(path)
            decode_seconds = time.monotonic() - started
            duration = len(samples) / 22_050.0

            self._step(song_id, "analysing")
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

            values = grid.analyse(beats, downbeats, duration or song["duration"])
            values.update(download_seconds=download_seconds, decode_seconds=decode_seconds, model_seconds=model_seconds)
            self.store.save_result(song_id, self.model_name, values, beats, downbeats)
            self.store.set_state(song_id, "done")
            bpm = f"{values['bpm']:.2f} BPM" if values.get("bpm") else "no steady beat"
            self.store.event(f"{song['artist']} – {song['title']}: {bpm}, {len(beats)} beats, "
                             f"{values.get('beats_per_bar') or '?'} a bar · {size / 1e6:.1f} MB in "
                             f"{download_seconds:.1f}s, decoded in {decode_seconds:.1f}s, model {model_seconds:.1f}s")
        except Exception as e:  # one song failing never stops the queue
            message = str(e) or e.__class__.__name__
            self.last_error = message
            self.store.set_state(song_id, "failed", message)
            self.store.event(f"{song['artist']} – {song['title']} failed: {message}", "error")
            if not isinstance(e, (SubsonicError, RuntimeError)):
                traceback.print_exc()
        finally:
            if path and os.path.exists(path):
                os.remove(path)
            self.current = None
