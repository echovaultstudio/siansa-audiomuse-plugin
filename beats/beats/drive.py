"""How energetic a song is, as a listener means it (0.35): its arousal, from Essentia's emoMusic model.

Until 0.35 a song's energy was AudioMuse-AI's: how loud its frames are. The owner's trial (12 Nov: 164 songs, five
measures side by side, judged by ear) showed loudness says little about energy: a bass-heavy, slow song mastered loud
came out above hard drum and bass, and once every song is on one scale nearly all of them are within a quarter of each
other. Of the measures tried, emoMusic's arousal matched his ear on five songs of six where it and loudness disagreed
most.

The model (MTG, UPF; CC BY-NC-SA 4.0; ``emomusic-msd-musicnn-2``, as ONNX in the image) reads the MusiCNN vector that
is kept for every song with a sound, the one AudioMuse-AI made or :mod:`beats.hear` makes, and answers valence and
arousal, each about 1 to 9. So a whole library is scored from what is already kept, in seconds, with no song decoded:
in the trial the arousal from a song's one kept vector and from every patch of its sound agreed at 0.98.

Kept in the table ``arousal`` beside the sound, not in it (a sound is replaced whole when it is pulled or heard
again): each row remembers which sound it was made from, and :meth:`Arousal.catch_up` scores what is new or changed.
An image without the model (an older one, a test) has no arousal and everything else is as before.
"""

import os
import threading

import numpy as np

from . import log, memory

_log = log.get("drive")

MODEL = "emomusic-msd-musicnn-2"
FILE = MODEL + ".onnx"
AROUSAL = 1  # the model answers (valence, arousal)
BATCH = 2000


class Arousal:
    def __init__(self, models_dir, store, threads=None):
        self.path, self.store, self.threads = os.path.join(models_dir, FILE), store, threads
        self._session = None
        self._lock = threading.Lock()
        with store.lock:
            store.db.execute("CREATE TABLE IF NOT EXISTS arousal (song_id TEXT PRIMARY KEY, value REAL, sound_at REAL)")
            store.db.commit()

    @property
    def available(self):
        return os.path.exists(self.path)

    @property
    def loaded(self):
        return self._session is not None

    def unload(self):
        with self._lock:
            had, self._session = self._session is not None, None
            return had

    def score(self, vectors):
        """The arousal of each MusiCNN vector (rows of 200 float32), about 1..9."""
        with self._lock:
            if self._session is None:
                import onnxruntime

                self._session = onnxruntime.InferenceSession(self.path, memory.session_options(onnxruntime, self.threads),
                                                             providers=["CPUExecutionProvider"])
            session = self._session
            rows = np.asarray(vectors, dtype=np.float32).reshape(-1, 200)
            return session.run(None, {session.get_inputs()[0].name: rows})[0][:, AROUSAL]

    def catch_up(self, most=None):
        """Scores every song whose sound is new or changed since it was scored; the number scored. Nothing without
        the model."""
        if not self.available:
            return 0
        done = 0
        while most is None or done < most:
            rows = self.store._rows(
                "SELECT o.song_id, o.musicnn, o.at FROM sound o LEFT JOIN arousal a ON a.song_id = o.song_id "
                "WHERE o.musicnn IS NOT NULL AND length(o.musicnn) = 800 AND (a.song_id IS NULL OR a.sound_at IS NOT o.at) "
                "LIMIT ?", (BATCH,))
            if not rows:
                break
            values = self.score(np.stack([np.frombuffer(r["musicnn"], dtype=np.float32) for r in rows]))
            with self.store.lock:
                self.store.db.executemany("INSERT OR REPLACE INTO arousal (song_id, value, sound_at) VALUES (?, ?, ?)",
                                          [(r["song_id"], float(v), r["at"]) for r, v in zip(rows, values)])
                self.store.db.commit()
            done += len(rows)
        if done:
            self.unload()
            _log.info("Worked out how energetic %d songs are, from the sound kept for each", done)
        return done

    def of(self, song_ids):
        """``{song id: arousal}`` of the songs among ``song_ids`` that have one."""
        out, ids = {}, list(dict.fromkeys(song_ids))
        for start in range(0, len(ids), 500):
            chunk = ids[start:start + 500]
            for r in self.store._rows(f"SELECT song_id, value FROM arousal WHERE song_id IN ({', '.join('?' for _ in chunk)})", chunk):
                out[r["song_id"]] = r["value"]
        return out

    def count(self):
        return self.store._rows("SELECT COUNT(*) AS n FROM arousal")[0]["n"]


# ---- the loudness of songs taken over from an older AudioMuse-AI, on today's scale ----

OLD_SCALE_BELOW = 0.45  # a library whose taken-over energies sit under this in the middle was measured the old way
OFFSET = 0.035  # the trial: today's measure is this much under the old one put through today's mapping (±0.015)


def old_scale(energies):
    """Whether these taken-over energies are on the old scale (the plain mean level, about 0.05 to 0.4 for music)
    and not today's (the level in dB on 0..1, about 0.5 to 0.9): by their middle, as a library is one or the other."""
    known = sorted(e for e in energies if e is not None and e > 0)
    return bool(known) and known[len(known) // 2] < OLD_SCALE_BELOW


def converted(old):
    """An old-scale energy on today's scale: its level in dB mapped from -60..0 to 0..1, less :data:`OFFSET` (today's
    is the mean of the frames' dB, a little under the dB of their mean). In the trial, right to about 0.015."""
    if old is None or old <= 0:
        return old
    return float(min(1.0, max(0.0, (20.0 * np.log10(old) + 60.0) / 60.0 - OFFSET)))
