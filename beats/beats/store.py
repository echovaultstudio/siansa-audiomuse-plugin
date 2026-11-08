"""What's been analysed, in one SQLite file in the data folder (``/data/beats.db``).

Tables:

- ``songs``: each song in the test library (from Navidrome), with its BPM tag;
- ``genres``: a song's genres, each with where it came from (``source``: ``tag`` for now). Room for better genres
  later: another source (a genre model, AudioMuse-AI's labels…) adds its own rows, and the page and the checks use
  every source;
- ``jobs``: each song's place in the queue and what's happening to it;
- ``results``: Beat This!'s readings, one row per song and model, with every beat and downbeat;
- ``plugin``: the Siansa plugin's current readings, for comparison;
- ``events``: what the service did, for the page's activity list.

Nothing here is written back to Navidrome, AudioMuse-AI or the music files.
"""

import json
import sqlite3
import threading
import time

SCHEMA_VERSION = 1
EVENTS_KEPT = 500

SCHEMA = """
CREATE TABLE IF NOT EXISTS songs (
    id TEXT PRIMARY KEY, title TEXT, artist TEXT, album TEXT, duration REAL, suffix TEXT, tag_bpm REAL,
    playlist TEXT, added_at REAL
);
CREATE TABLE IF NOT EXISTS genres (
    song_id TEXT, source TEXT, label TEXT, weight REAL DEFAULT 1.0, PRIMARY KEY (song_id, source, label)
);
CREATE TABLE IF NOT EXISTS jobs (
    song_id TEXT PRIMARY KEY, state TEXT, step TEXT, queued_at REAL, started_at REAL, finished_at REAL, error TEXT
);
CREATE TABLE IF NOT EXISTS results (
    song_id TEXT, model TEXT, bpm REAL, bpm_median REAL, steady REAL, grid_ms REAL, grid_share REAL,
    intro_bpm REAL, intro_steady REAL, outro_bpm REAL, outro_steady REAL, beats_per_bar INTEGER, bar_share REAL,
    first_downbeat_ms REAL, beat_count INTEGER, downbeat_count INTEGER, download_seconds REAL, decode_seconds REAL,
    model_seconds REAL, analysed_at REAL, beats TEXT, downbeats TEXT, PRIMARY KEY (song_id, model)
);
CREATE TABLE IF NOT EXISTS plugin (
    song_id TEXT PRIMARY KEY, intro_bpm REAL, outro_bpm REAL, intro_steady REAL, outro_steady REAL,
    intro_bar_ms REAL, outro_bar_ms REAL, fetched_at REAL
);
CREATE TABLE IF NOT EXISTS events (at REAL, level TEXT, message TEXT);
"""

RESULT_FIELDS = ("bpm", "bpm_median", "steady", "grid_ms", "grid_share", "intro_bpm", "intro_steady", "outro_bpm",
                 "outro_steady", "beats_per_bar", "bar_share", "first_downbeat_ms", "beat_count", "downbeat_count",
                 "download_seconds", "decode_seconds", "model_seconds")
PLUGIN_FIELDS = ("intro_bpm", "outro_bpm", "intro_steady", "outro_steady", "intro_bar_ms", "outro_bar_ms")
BUSY = ("downloading", "decoding", "analysing")


class Store:
    def __init__(self, path):
        self.path = path
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        with self.lock, self.db:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.executescript(SCHEMA)
            if self.db.execute("PRAGMA user_version").fetchone()[0] < SCHEMA_VERSION:
                self.db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            # Songs that were being worked on when the service stopped go back in the queue.
            self.db.execute(f"UPDATE jobs SET state='queued', step=NULL WHERE state IN {BUSY}")

    def close(self):
        with self.lock:
            self.db.close()

    def _rows(self, sql, args=()):
        with self.lock:
            return [dict(r) for r in self.db.execute(sql, args).fetchall()]

    # ---- songs and genres ----

    def add_song(self, song, playlist=None, source="tag"):
        """Adds or updates ``song`` (from :func:`subsonic.song_from`); its genres replace that source's."""
        with self.lock, self.db:
            self.db.execute(
                "INSERT INTO songs (id, title, artist, album, duration, suffix, tag_bpm, playlist, added_at) "
                "VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET title=excluded.title, artist=excluded.artist, "
                "album=excluded.album, duration=excluded.duration, suffix=excluded.suffix, tag_bpm=excluded.tag_bpm, "
                "playlist=COALESCE(excluded.playlist, songs.playlist)",
                (song["id"], song["title"], song["artist"], song["album"], song["duration"], song["suffix"],
                 song["tag_bpm"], playlist, time.time()))
            self.set_genres(song["id"], source, song.get("genres") or [])

    def set_genres(self, song_id, source, labels, weights=None):
        with self.lock, self.db:
            self.db.execute("DELETE FROM genres WHERE song_id=? AND source=?", (song_id, source))
            for i, label in enumerate(labels):
                weight = weights[i] if weights else 1.0
                self.db.execute("INSERT OR REPLACE INTO genres VALUES (?,?,?,?)", (song_id, source, label, weight))

    def genres(self, song_id):
        return self._rows("SELECT source, label, weight FROM genres WHERE song_id=? ORDER BY source, weight DESC, label",
                          (song_id,))

    def song(self, song_id):
        rows = self._rows("SELECT * FROM songs WHERE id=?", (song_id,))
        return rows[0] if rows else None

    def remove(self, song_id):
        with self.lock, self.db:
            for table, key in (("songs", "id"), ("genres", "song_id"), ("jobs", "song_id"), ("results", "song_id"),
                               ("plugin", "song_id")):
                self.db.execute(f"DELETE FROM {table} WHERE {key}=?", (song_id,))

    # ---- the queue ----

    def queue(self, song_id, force=False):
        """Puts the song in the queue; with ``force`` also when it's done or failed. True when it was queued."""
        with self.lock, self.db:
            row = self.db.execute("SELECT state FROM jobs WHERE song_id=?", (song_id,)).fetchone()
            if row and (row["state"] == "queued" or row["state"] in BUSY):
                return False
            if row and not force:
                return False
            self.db.execute("INSERT OR REPLACE INTO jobs (song_id, state, step, queued_at, started_at, finished_at, error)"
                            " VALUES (?, 'queued', NULL, ?, NULL, NULL, NULL)", (song_id, time.time()))
            return True

    def next_queued(self):
        rows = self._rows("SELECT song_id FROM jobs WHERE state='queued' ORDER BY queued_at, song_id LIMIT 1")
        return rows[0]["song_id"] if rows else None

    def set_state(self, song_id, state, error=None):
        now = time.time()
        with self.lock, self.db:
            if state in BUSY:
                self.db.execute("UPDATE jobs SET state=?, step=?, started_at=COALESCE(started_at, ?) WHERE song_id=?",
                                (state, state, now, song_id))
            else:
                self.db.execute("UPDATE jobs SET state=?, step=NULL, finished_at=?, error=? WHERE song_id=?",
                                (state, now, error, song_id))

    def counts(self):
        rows = self._rows("SELECT state, COUNT(*) AS n FROM jobs GROUP BY state")
        counts = {r["state"]: r["n"] for r in rows}
        counts["songs"] = self._rows("SELECT COUNT(*) AS n FROM songs")[0]["n"]
        return counts

    # ---- results ----

    def save_result(self, song_id, model, values, beats, downbeats):
        with self.lock, self.db:
            self.db.execute(
                f"INSERT OR REPLACE INTO results (song_id, model, {', '.join(RESULT_FIELDS)}, analysed_at, beats, downbeats)"
                f" VALUES (?, ?, {', '.join('?' for _ in RESULT_FIELDS)}, ?, ?, ?)",
                (song_id, model, *[values.get(f) for f in RESULT_FIELDS], time.time(),
                 json.dumps([round(b, 3) for b in beats]), json.dumps([round(b, 3) for b in downbeats])))

    def result(self, song_id, model, with_beats=False):
        rows = self._rows("SELECT * FROM results WHERE song_id=? AND model=?", (song_id, model))
        if not rows:
            return None
        row = rows[0]
        beats, downbeats = row.pop("beats"), row.pop("downbeats")
        if with_beats:
            row["beats"], row["downbeats"] = json.loads(beats or "[]"), json.loads(downbeats or "[]")
        return row

    def save_plugin(self, song_id, values):
        with self.lock, self.db:
            self.db.execute(f"INSERT OR REPLACE INTO plugin (song_id, {', '.join(PLUGIN_FIELDS)}, fetched_at) "
                            f"VALUES (?, {', '.join('?' for _ in PLUGIN_FIELDS)}, ?)",
                            (song_id, *[values.get(f) for f in PLUGIN_FIELDS], time.time()))

    def plugin(self, song_id):
        rows = self._rows("SELECT * FROM plugin WHERE song_id=?", (song_id,))
        return rows[0] if rows else None

    def overview(self, model):
        """Every song with its job, result (no beats) and plugin readings, for the table."""
        rows = self._rows(
            "SELECT s.*, j.state, j.step, j.error, j.finished_at, "
            + ", ".join(f"r.{f} AS r_{f}" for f in RESULT_FIELDS) + ", r.analysed_at AS r_analysed_at, "
            + ", ".join(f"p.{f} AS p_{f}" for f in PLUGIN_FIELDS)
            + " FROM songs s LEFT JOIN jobs j ON j.song_id=s.id LEFT JOIN results r ON r.song_id=s.id AND r.model=? "
            "LEFT JOIN plugin p ON p.song_id=s.id ORDER BY s.artist COLLATE NOCASE, s.title COLLATE NOCASE", (model,))
        for row in rows:
            row["result"] = {f: row.pop(f"r_{f}") for f in RESULT_FIELDS}
            row["result"]["analysed_at"] = row.pop("r_analysed_at")
            if row["result"]["analysed_at"] is None:
                row["result"] = None
            row["plugin"] = {f: row.pop(f"p_{f}") for f in PLUGIN_FIELDS}
            if all(v is None for v in row["plugin"].values()):
                row["plugin"] = None
            row["genres"] = self.genres(row["id"])
        return rows

    # ---- events ----

    def event(self, message, level="info"):
        with self.lock, self.db:
            self.db.execute("INSERT INTO events VALUES (?,?,?)", (time.time(), level, message))
            self.db.execute("DELETE FROM events WHERE rowid NOT IN (SELECT rowid FROM events ORDER BY at DESC LIMIT ?)",
                            (EVENTS_KEPT,))

    def events(self, limit=60):
        return self._rows("SELECT at, level, message FROM events ORDER BY at DESC, rowid DESC LIMIT ?", (limit,))
