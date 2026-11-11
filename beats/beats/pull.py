"""Takes over what AudioMuse-AI has worked out about each song (0.25, the owner, 11 Nov), so nothing is analysed
twice: its vectors (how a song sounds; the one the search by words uses), key, moods, genres and energy. Not its
tempo: a song's tempo here is Beat This!'s alone (the owner, 11 Nov: "the big reason for moving to Beat This!").

AudioMuse-AI has a read-only export for apps (``/api/sync``). Its index says, for every song it has analysed, the music
server's id and a fingerprint of the analysis; only songs that are new here, or whose fingerprint moved, are asked for
in full. So the first pull takes a few minutes for a big library and every later one seconds.

A song is matched by its id on the music server: AudioMuse-AI gives the ids of its default server, and this service
knows every song by the ids of each of its servers. A song this service doesn't have (a library not chosen here) is
left out and counted.
"""

import base64
import threading
import time

from . import log
from .audiomuse import AudioMuseError

_log = log.get("pull")

BATCH = 100
VECTOR_BYTES = {"embedding": 200 * 4, "clap_embedding": 512 * 4}


def row_of(song_id, track):
    """A song's sound as kept here, from AudioMuse-AI's ``track``; None when it has no usable vector."""
    def vector(name):
        try:
            raw = base64.b64decode(track.get(name) or "")
        except (ValueError, TypeError):
            return None
        return raw if len(raw) == VECTOR_BYTES[name] else None

    musicnn = vector("embedding")
    if musicnn is None:
        return None

    def number(name):
        try:
            return float(track[name]) if track.get(name) is not None else None
        except (TypeError, ValueError):
            return None

    year = track.get("year")
    # Not its tempo (the owner, 11 Nov): a song's tempo is the one Beat This! finds, never AudioMuse-AI's.
    return {"song_id": song_id, "musicnn": musicnn, "clap": vector("clap_embedding"),
            "key": track.get("key"), "scale": track.get("scale"), "moods": track.get("mood_vector"),
            "other": track.get("other_features"), "energy": number("energy"), "album_artist": track.get("album_artist"),
            "year": year if isinstance(year, int) else None, "fp": track.get("fp")}


class Puller:
    """Pulls in a thread; :attr:`state` says how it's going, for the page."""

    def __init__(self, store, audiomuse, on_done=None):
        self.store, self.audiomuse, self.on_done = store, audiomuse, on_done
        self.state = dict(self.store.setting("sound_pull") or {}, running=False)
        self._lock = threading.Lock()

    def start(self, everything=False):
        """Starts a pull unless one runs; ``everything`` asks for every song again, not only new and changed ones."""
        with self._lock:
            if self.state.get("running"):
                return False
            self.state = {"running": True, "started": time.time(), "done": 0, "total": 0, "kept": 0, "unknown": 0}
        threading.Thread(target=self._run, args=(everything,), name="sound-pull", daemon=True).start()
        return True

    def _run(self, everything):
        try:
            result = self.pull(everything)
            self.state = {"running": False, "finished": time.time(), **result}
            if result["kept"]:
                self.store.event(f"The sound of {result['kept']} songs was taken over from AudioMuse-AI"
                                 + (f" ({result['unknown']} of its songs aren't in the libraries here)"
                                    if result["unknown"] else ""))
        except AudioMuseError as e:
            self.state = {"running": False, "finished": time.time(), "error": str(e)}
            _log.warning("Pulling from AudioMuse-AI: %s", e)
        except Exception as e:  # nothing here may take the service down
            self.state = {"running": False, "finished": time.time(), "error": f"{type(e).__name__}: {e}"}
            _log.exception("Pulling from AudioMuse-AI failed")
        try:
            self.store.set_setting("sound_pull", self.state)
        except Exception:  # the service is closing
            _log.debug("The pull's result wasn't kept", exc_info=True)
        if self.on_done:
            self.on_done()

    def pull(self, everything=False):
        """One pull, in this thread: ``{"total" (AudioMuse-AI's songs), "kept" (taken over now), "unknown" (not
        here), "same" (already here, unchanged), "no_vector"}``."""
        have = {} if everything else self.store.sound_prints()
        wanted, total, unknown, same = [], 0, 0, 0
        for tracks, total in self.audiomuse().sync_index():
            here = self.store.resolve([t["id"] for t in tracks if t.get("id")])
            for t in tracks:
                song_id = here.get(t.get("id"))
                if song_id is None:
                    unknown += 1
                elif have.get(song_id) == t.get("fp") and t.get("fp"):
                    same += 1
                else:
                    wanted.append((t["id"], song_id))
            self.state.update(total=total, unknown=unknown)
        kept = no_vector = 0
        self.state.update(total=len(wanted), done=0)
        for start in range(0, len(wanted), BATCH):
            chunk = dict(wanted[start:start + BATCH])
            rows = [row_of(chunk[t["id"]], t) for t in self.audiomuse().sync_songs(list(chunk)) if t.get("id") in chunk]
            no_vector += sum(1 for r in rows if r is None)
            rows = [r for r in rows if r is not None]
            if rows:
                kept += self.store.save_sound(rows)
            self.state.update(done=min(start + BATCH, len(wanted)), kept=kept)
        _log.info("Pulled from AudioMuse-AI: %d of its %d songs taken over, %d unchanged, %d not here, %d without a vector",
                  kept, total, same, unknown, no_vector)
        return {"total": total, "kept": kept, "unknown": unknown, "same": same, "no_vector": no_vector}
