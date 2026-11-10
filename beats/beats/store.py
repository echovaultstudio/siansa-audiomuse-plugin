"""What's been analysed, in one SQLite file in the data folder (``/data/beats.db``).

Tables:

- ``songs``: each song from the chosen Navidrome libraries, with its BPM tag and its library (``folder``);
- ``genres``: a song's genres, each with where it came from (``source``): ``tag`` (its own tags), ``sound`` (the
  styles :mod:`beats.style` hears, with their scores; ``sound-failed`` when its sound couldn't be read) and ``vote``
  (:mod:`beats.genres`: what the song's genres are, for the radios);
- ``jobs``: each song's place in the queue and what's happening to it;
- ``results``: Beat This!'s readings, one row per song and model, with every beat and downbeat;
- ``events``: what the service did, for the page's activity list;
- ``measures``: each song's loudness, fade ramps and key (:mod:`beats.loudness`, :mod:`beats.key`), for the app's
  volume levelling, fades and blends in key (``error`` when it couldn't be measured);
- ``bad_blends``: song changes the owner marked as bad in Siansa Dev, with how the app blended them;
- ``settings``: what the page chose (the libraries to analyse), kept across restarts.

Nothing here is written back to Navidrome, AudioMuse-AI or the music files.
"""

import json
import sqlite3
import threading
import time

from . import log

_activity = log.get("activity")
_LOG_LEVEL = {"info": 20, "warn": 30, "error": 40}

SCHEMA_VERSION = 4  # 2: bpm_fit, main_share (0.3.0); 3: the steady grid (0.4.0); 4: levels, phrases (0.5.0)
EVENTS_KEPT = 500

SCHEMA = """
CREATE TABLE IF NOT EXISTS songs (
    id TEXT PRIMARY KEY, title TEXT, artist TEXT, album TEXT, duration REAL, suffix TEXT, tag_bpm REAL,
    playlist TEXT, added_at REAL  -- playlist: before 0.10.0, which reads libraries instead (the folder column)
);
CREATE TABLE IF NOT EXISTS genres (
    song_id TEXT, source TEXT, label TEXT, weight REAL DEFAULT 1.0, PRIMARY KEY (song_id, source, label)
);
CREATE INDEX IF NOT EXISTS genres_by_label ON genres (source, label, weight);
CREATE TABLE IF NOT EXISTS jobs (
    song_id TEXT PRIMARY KEY, state TEXT, step TEXT, queued_at REAL, started_at REAL, finished_at REAL, error TEXT
);
CREATE TABLE IF NOT EXISTS results (
    song_id TEXT, model TEXT, bpm REAL, bpm_median REAL, steady REAL, grid_ms REAL, grid_share REAL,
    intro_bpm REAL, intro_steady REAL, outro_bpm REAL, outro_steady REAL, beats_per_bar INTEGER, bar_share REAL,
    first_downbeat_ms REAL, beat_count INTEGER, downbeat_count INTEGER, download_seconds REAL, decode_seconds REAL,
    model_seconds REAL, analysed_at REAL, beats TEXT, downbeats TEXT, PRIMARY KEY (song_id, model)
);
DROP TABLE IF EXISTS plugin;  -- the Siansa plugin's readings, compared with up to 0.13: no longer asked for
CREATE TABLE IF NOT EXISTS events (at REAL, level TEXT, message TEXT);
CREATE TABLE IF NOT EXISTS marks (
    id INTEGER PRIMARY KEY AUTOINCREMENT, song_id TEXT, seconds REAL, created_at REAL
);
CREATE INDEX IF NOT EXISTS marks_song ON marks (song_id);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS measures (
    song_id TEXT PRIMARY KEY, ref_lufs REAL, start_ramp TEXT, end_ramp TEXT, duration_ms INTEGER, key TEXT, scale TEXT,
    key_strength REAL, version INTEGER, measured_at REAL, error TEXT
);
CREATE TABLE IF NOT EXISTS bad_blends (
    id INTEGER PRIMARY KEY AUTOINCREMENT, at REAL, from_id TEXT, to_id TEXT, technique TEXT, details TEXT
);
"""

RESULT_FIELDS = ("bpm", "bpm_fit", "main_share", "grid_bpm", "on_grid", "grid_drift", "analysis_version", "downbeat_source", "grid_counted",
                 "changes_share", "phrase_first_ms", "phrase_count", "phrase_bars", "phrase_big", "phrase_big_on", "phrase_source", "from_mix", "ends", "bpm_median", "steady", "grid_ms", "grid_share", "intro_bpm", "intro_steady", "outro_bpm",
                 "outro_steady", "beats_per_bar", "bar_share", "first_downbeat_ms", "beat_count", "downbeat_count",
                 "download_seconds", "decode_seconds", "model_seconds")
BUSY = ("downloading", "decoding", "ready", "analysing")  # ready: downloaded and decoded, waiting for the model


def _column(name, value):
    """A result's value as kept in its column: the app's ends as JSON, the rest as they are."""
    if name == "ends":
        return json.dumps(value) if value else None
    return value


class Store:
    def __init__(self, path):
        self.path = path
        self.lock = threading.RLock()
        self.genre_changes = 0  # counts every change to the genres table, so kept answers know when they're old
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        with self.lock, self.db:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.executescript(SCHEMA)
            if "folder" not in {r[1] for r in self.db.execute("PRAGMA table_info(songs)")}:
                self.db.execute("ALTER TABLE songs ADD COLUMN folder TEXT")
            columns = {r[1] for r in self.db.execute("PRAGMA table_info(results)")}
            for column, kind in (("bpm_fit", "REAL"), ("main_share", "REAL"), ("grid_bpm", "REAL"), ("on_grid", "REAL"), ("grid_drift", "REAL"),
                                 ("analysis_version", "REAL"), ("downbeat_source", "TEXT"), ("grid_counted", "TEXT"), ("changes_share", "REAL"),
                                 ("phrase_first_ms", "REAL"), ("phrase_count", "INTEGER"), ("phrase_bars", "INTEGER"), ("phrase_big", "INTEGER"),
                                 ("phrase_big_on", "INTEGER"), ("levels", "TEXT"), ("phrase_source", "TEXT"),
                                 ("from_mix", "INTEGER"), ("ends", "TEXT")):
                if column not in columns:
                    self.db.execute(f"ALTER TABLE results ADD COLUMN {column} {kind}")
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

    def add_song(self, song, folder=None, source="tag"):
        """Adds or updates ``song`` (from :func:`subsonic.song_from`) of library ``folder``; its genres replace that
        source's."""
        with self.lock, self.db:
            self._add_song(song, folder, source)

    def add_songs(self, songs, folder=None, source="tag"):
        """Adds or updates many songs at once (a whole library) and queues the ones never analysed: their ids."""
        queued = []
        now = time.time()
        with self.lock, self.db:
            for song in songs:
                self._add_song(song, folder, source)
                if self.db.execute("INSERT OR IGNORE INTO jobs (song_id, state, queued_at) VALUES (?, 'queued', ?)",
                                   (song["id"], now)).rowcount:
                    queued.append(song["id"])
        return queued

    def _add_song(self, song, folder, source):
        self.db.execute(
            "INSERT INTO songs (id, title, artist, album, duration, suffix, tag_bpm, folder, added_at) "
            "VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET title=excluded.title, artist=excluded.artist, "
            "album=excluded.album, duration=excluded.duration, suffix=excluded.suffix, tag_bpm=excluded.tag_bpm, "
            "folder=COALESCE(excluded.folder, songs.folder)",
            (song["id"], song["title"], song["artist"], song["album"], song["duration"], song["suffix"],
             song["tag_bpm"], folder, time.time()))
        self.db.execute("DELETE FROM genres WHERE song_id=? AND source=?", (song["id"], source))
        for label in song.get("genres") or []:
            self.db.execute("INSERT OR REPLACE INTO genres VALUES (?,?,?,?)", (song["id"], source, label, 1.0))

    def set_genres(self, song_id, source, labels, weights=None):
        self.genre_changes += 1
        with self.lock, self.db:
            self.db.execute("DELETE FROM genres WHERE song_id=? AND source=?", (song_id, source))
            for i, label in enumerate(labels):
                weight = weights[i] if weights else 1.0
                self.db.execute("INSERT OR REPLACE INTO genres VALUES (?,?,?,?)", (song_id, source, label, weight))

    # ---- genres by vote (beats.genres): the sound's styles, the tags, and the vote of both ----

    def songs_without_style(self, limit=50):
        """Songs of the chosen libraries whose sound hasn't been given styles yet (nor failed to): the next ones for
        the style model, analysed songs first."""
        return [r["id"] for r in self._rows(
            "SELECT s.id FROM songs s LEFT JOIN jobs j ON j.song_id=s.id WHERE NOT EXISTS (SELECT 1 FROM genres g "
            "WHERE g.song_id=s.id AND g.source IN ('sound', 'sound-failed')) AND COALESCE(j.state, '') != 'failed' "
            "ORDER BY CASE j.state WHEN 'done' THEN 0 ELSE 1 END, s.id LIMIT ?", (limit,))]

    def style_counts(self):
        """``{"done", "failed", "total"}``: songs with styles from their sound, songs whose sound couldn't be, of all."""
        rows = self._rows("SELECT source, COUNT(DISTINCT song_id) AS n FROM genres WHERE source IN ('sound', "
                          "'sound-failed') GROUP BY source")
        by = {r["source"]: r["n"] for r in rows}
        total = self._rows("SELECT COUNT(*) AS n FROM songs s LEFT JOIN jobs j ON j.song_id=s.id "
                           "WHERE COALESCE(j.state, '') != 'failed'")[0]["n"]  # a song that failed has no sound to hear
        return {"done": by.get("sound", 0), "failed": by.get("sound-failed", 0), "total": total}

    def genre_rows(self, song_ids=None, sources=("sound", "tag")):
        """``{song_id: {source: [(label, weight)]}}`` of ``song_ids`` (every song when ``None``), for voting."""
        out = {}
        marks = ", ".join("?" for _ in sources)
        where, args = f"source IN ({marks})", list(sources)
        if song_ids is not None:
            song_ids = list(song_ids)[:900]
            where += f" AND song_id IN ({', '.join('?' for _ in song_ids)})"
            args += song_ids
        for r in self._rows(f"SELECT song_id, source, label, weight FROM genres WHERE {where} ORDER BY weight DESC",
                            tuple(args)):
            out.setdefault(r["song_id"], {}).setdefault(r["source"], []).append((r["label"], r["weight"]))
        return out

    def replace_votes(self, votes):
        """Writes each song's ``vote`` rows (``{song_id: [(label, score)]}``) in one go."""
        self.genre_changes += 1
        with self.lock, self.db:
            for song_id, voted in votes.items():
                self.db.execute("DELETE FROM genres WHERE song_id=? AND source='vote'", (song_id,))
                self.db.executemany("INSERT OR REPLACE INTO genres VALUES (?, 'vote', ?, ?)",
                                    [(song_id, label, score) for label, score in voted])

    def vote_counts(self, minimum):
        """``{label: songs}``: how many songs have each style in their vote at ``minimum`` or more."""
        return {r["label"]: r["n"] for r in self._rows(
            "SELECT label, COUNT(*) AS n FROM genres WHERE source='vote' AND weight >= ? GROUP BY label", (minimum,))}

    def songs_voted(self, labels, minimum, limit):
        """Song ids (random order) with any of ``labels`` in their vote at ``minimum`` or more, at most ``limit``."""
        labels = list(labels)
        if not labels:
            return []
        marks = ", ".join("?" for _ in labels)
        return [r["song_id"] for r in self._rows(
            f"SELECT DISTINCT song_id FROM genres WHERE source='vote' AND weight >= ? AND label IN ({marks}) "
            "ORDER BY RANDOM() LIMIT ?", (minimum, *labels, limit))]

    def genres(self, song_id):
        return self._rows("SELECT source, label, weight FROM genres WHERE song_id=? ORDER BY source, weight DESC, label",
                          (song_id,))

    def song(self, song_id):
        rows = self._rows("SELECT * FROM songs WHERE id=?", (song_id,))
        return rows[0] if rows else None

    def song_ids(self):
        return [r["id"] for r in self._rows("SELECT id FROM songs")]

    def remove(self, song_id):
        self.genre_changes += 1
        with self.lock, self.db:
            for table, key in (("songs", "id"), ("genres", "song_id"), ("jobs", "song_id"), ("results", "song_id"),
                               ("marks", "song_id")):
                self.db.execute(f"DELETE FROM {table} WHERE {key}=?", (song_id,))

    def remove_folder(self, folder):
        """Forgets every song of library ``folder``: its results, marks and place in the queue. The number removed
        (nothing changes on Navidrome)."""
        with self.lock, self.db:
            ids = [r[0] for r in self.db.execute("SELECT id FROM songs WHERE folder=?", (folder,))]
            for table, key in (("genres", "song_id"), ("jobs", "song_id"), ("results", "song_id"),
                               ("marks", "song_id")):
                self.db.execute(f"DELETE FROM {table} WHERE {key} IN (SELECT id FROM songs WHERE folder=?)", (folder,))
            self.db.execute("DELETE FROM songs WHERE folder=?", (folder,))
            return len(ids)

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

    NEXT = "SELECT song_id FROM jobs WHERE state='queued' ORDER BY queued_at, song_id LIMIT 1"

    def next_queued(self):
        rows = self._rows(self.NEXT)
        return rows[0]["song_id"] if rows else None

    def claim_next(self):
        """The next song in the queue, marked as being downloaded in the same step, so no two workers take it."""
        with self.lock, self.db:
            row = self.db.execute(self.NEXT).fetchone()
            if not row:
                return None
            self.db.execute("UPDATE jobs SET state='downloading', step='downloading', started_at=? WHERE song_id=?",
                            (time.time(), row["song_id"]))
            return row["song_id"]

    def unqueue_except(self, folders):
        """Takes the songs of every library but ``folders`` out of the queue (those done stay): the number taken."""
        folders = list(folders)
        with self.lock, self.db:
            return self.db.execute(
                "DELETE FROM jobs WHERE state='queued' AND song_id IN (SELECT id FROM songs WHERE folder IS NOT NULL"
                + (f" AND folder NOT IN ({', '.join('?' for _ in folders)})" if folders else "") + ")",
                folders).rowcount

    def folder_counts(self):
        """``{folder: {"songs", "done"}}``: each library's songs here, and how many are analysed."""
        rows = self._rows("SELECT s.folder, COUNT(*) AS songs, SUM(j.state='done') AS done FROM songs s "
                          "LEFT JOIN jobs j ON j.song_id=s.id GROUP BY s.folder")
        return {r["folder"]: {"songs": r["songs"], "done": r["done"] or 0} for r in rows if r["folder"] is not None}

    # ---- settings ----

    def setting(self, key, default=None):
        rows = self._rows("SELECT value FROM settings WHERE key=?", (key,))
        return json.loads(rows[0]["value"]) if rows else default

    def set_setting(self, key, value):
        with self.lock, self.db:
            self.db.execute("INSERT OR REPLACE INTO settings VALUES (?, ?)", (key, json.dumps(value)))

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

    def save_result(self, song_id, model, values, beats, downbeats, levels=None):
        """``levels``: the sound's levels (:func:`beats.features.measure`); ``None`` saves an empty ``""``, so a
        song whose sound couldn't be measured isn't asked for again."""
        with self.lock, self.db:
            self.db.execute(
                f"INSERT OR REPLACE INTO results (song_id, model, {', '.join(RESULT_FIELDS)}, analysed_at, beats, downbeats,"
                f" levels) VALUES (?, ?, {', '.join('?' for _ in RESULT_FIELDS)}, ?, ?, ?, ?)",
                (song_id, model, *[_column(f, values.get(f)) for f in RESULT_FIELDS], time.time(),
                 json.dumps([round(b, 3) for b in beats]), json.dumps([round(b, 3) for b in downbeats]),
                 json.dumps(levels) if levels else ""))

    def result(self, song_id, model, with_beats=False):
        rows = self._rows("SELECT * FROM results WHERE song_id=? AND model=?", (song_id, model))
        if not rows:
            return None
        row = rows[0]
        row["ends"] = json.loads(row["ends"]) if row.get("ends") else None
        beats, downbeats, levels = row.pop("beats"), row.pop("downbeats"), row.pop("levels", None)
        if with_beats:
            row["beats"], row["downbeats"] = json.loads(beats or "[]"), json.loads(downbeats or "[]")
            row["levels"] = json.loads(levels) if levels else None
        return row

    def stale_results(self, version):
        """``(song_id, model)`` of results worked out by an older Siansa Beats (``analysis_version`` below
        ``version``), to be worked out again from their beats."""
        return [(r["song_id"], r["model"]) for r in self._rows(
            "SELECT song_id, model FROM results WHERE (analysis_version IS NULL OR analysis_version < ?) "
            "AND beats IS NOT NULL", (version,))]

    def without_levels(self):
        """Songs analysed before their sound was measured (0.5.0), to be analysed again for their phrases."""
        return [r["song_id"] for r in self._rows("SELECT song_id FROM results WHERE levels IS NULL")]

    def update_result(self, song_id, model, values):
        """Rewrites the readings worked out from the beats, keeping the beats and the times."""
        keys = [k for k in RESULT_FIELDS if k in values and not k.endswith("_seconds")]
        with self.lock, self.db:
            self.db.execute(f"UPDATE results SET {', '.join(f'{k}=?' for k in keys)} WHERE song_id=? AND model=?",
                            (*[_column(k, values[k]) for k in keys], song_id, model))

    def ends(self, song_ids, model):
        """``{song_id: ends}`` (:func:`beats.grid.app_ends`) of the songs among ``song_ids`` that have them."""
        out = {}
        ids = list(dict.fromkeys(song_ids))
        for start in range(0, len(ids), 500):  # SQLite's limit on ? in one statement
            chunk = ids[start:start + 500]
            for r in self._rows(f"SELECT song_id, ends FROM results WHERE model=? AND ends IS NOT NULL AND ends<>'' "
                                f"AND song_id IN ({', '.join('?' for _ in chunk)})", (model, *chunk)):
                out[r["song_id"]] = json.loads(r["ends"])
        return out

    # ---- loudness, fades and key ----

    def save_measure(self, song_id, values, version, error=None):
        """``values``: :func:`beats.loudness.profile` with :func:`beats.key.find`'s fields (either may be missing);
        ``error`` when the song couldn't be measured, so it isn't tried again and again."""
        with self.lock, self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO measures (song_id, ref_lufs, start_ramp, end_ramp, duration_ms, key, scale, "
                "key_strength, version, measured_at, error) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (song_id, values.get("ref_lufs"), values.get("start_ramp"), values.get("end_ramp"),
                 values.get("duration_ms"), values.get("key"), values.get("scale"), values.get("key_strength"),
                 version, time.time(), (error or None) and str(error)[:300]))

    def measures(self, song_ids):
        """``{song_id: {"ref_lufs", "start_ramp", "end_ramp", "duration_ms", "key", "scale"}}`` of the songs among
        ``song_ids`` that were measured (not those that failed)."""
        out = {}
        ids = list(dict.fromkeys(song_ids))
        for start in range(0, len(ids), 500):
            chunk = ids[start:start + 500]
            for r in self._rows("SELECT song_id, ref_lufs, start_ramp, end_ramp, duration_ms, key, scale FROM measures "
                                f"WHERE error IS NULL AND song_id IN ({', '.join('?' for _ in chunk)})", chunk):
                out[r.pop("song_id")] = r
        return out

    def without_measure(self, version, limit=20):
        """Songs analysed for their beats but not yet measured by this ``version`` (failed ones aren't asked for again,
        unless measured by an older version)."""
        return [r["song_id"] for r in self._rows(
            "SELECT j.song_id FROM jobs j LEFT JOIN measures m ON m.song_id = j.song_id WHERE j.state = 'done' AND "
            "(m.song_id IS NULL OR (m.version < ? AND m.error IS NULL)) ORDER BY j.finished_at DESC LIMIT ?",
            (version, limit))]

    def measure_counts(self, version):
        """``{"measured", "failed", "waiting"}`` for the page and the capabilities."""
        row = self._rows(
            "SELECT SUM(CASE WHEN m.error IS NULL AND m.version >= ? THEN 1 ELSE 0 END) AS measured, "
            "SUM(CASE WHEN m.error IS NOT NULL THEN 1 ELSE 0 END) AS failed, "
            "SUM(CASE WHEN m.song_id IS NULL OR (m.version < ? AND m.error IS NULL) THEN 1 ELSE 0 END) AS waiting "
            "FROM jobs j LEFT JOIN measures m ON m.song_id = j.song_id WHERE j.state = 'done'", (version, version))[0]
        return {k: row[k] or 0 for k in ("measured", "failed", "waiting")}

    def overview(self, model, find=None, state=None, limit=None, song_id=None):
        """The songs with their job and result (no beats), for the table: those whose name, artist,
        album or genre has ``find`` in it, only those in ``state`` ("done", "failed"), at most ``limit`` of them, or just ``song_id``."""
        where, args = [], [model]
        if song_id is not None:
            where.append("s.id=?")
            args.append(song_id)
        if state:
            where.append("j.state=?")
            args.append(state)
        for word in (find or "").split():
            like = "%" + word.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
            where.append("(s.title LIKE ? ESCAPE '\\' OR s.artist LIKE ? ESCAPE '\\' OR s.album LIKE ? ESCAPE '\\' OR "
                         "EXISTS (SELECT 1 FROM genres g WHERE g.song_id=s.id AND g.label LIKE ? ESCAPE '\\'))")
            args += [like] * 4
        rows = self._rows(
            "SELECT s.*, j.state, j.step, j.error, j.finished_at, "
            + ", ".join(f"r.{f} AS r_{f}" for f in RESULT_FIELDS) + ", r.analysed_at AS r_analysed_at"
            + " FROM songs s LEFT JOIN jobs j ON j.song_id=s.id LEFT JOIN results r ON r.song_id=s.id AND r.model=?"
            + (" WHERE " + " AND ".join(where) if where else "")
            + " ORDER BY s.artist COLLATE NOCASE, s.title COLLATE NOCASE" + (" LIMIT ?" if limit else ""),
            (*args, *([int(limit)] if limit else [])))
        genres = self._genres_of([r["id"] for r in rows])
        for row in rows:
            row["result"] = {f: row.pop(f"r_{f}") for f in RESULT_FIELDS}
            row["result"].pop("ends", None)  # what the app fetches (GET /api/grids), not the table
            row["result"]["analysed_at"] = row.pop("r_analysed_at")
            if row["result"]["analysed_at"] is None:
                row["result"] = None
            row["genres"] = genres.get(row["id"], [])
        counts = self.mark_counts()
        for row in rows:
            row["marks"] = counts.get(row["id"], 0)
        return rows

    def _genres_of(self, ids):
        """``{song_id: genres}`` of ``ids`` in one go (all of them for a big list)."""
        sql = "SELECT song_id, source, label, weight FROM genres"
        rows = self._rows(sql + f" WHERE song_id IN ({', '.join('?' for _ in ids)})", ids) if len(ids) <= 500 \
            else self._rows(sql)
        out = {}
        for r in sorted(rows, key=lambda r: (r["source"], -(r["weight"] or 0), r["label"])):
            out.setdefault(r.pop("song_id"), []).append(r)
        return out

    # ---- marks: where the owner heard a section start, to check the phrases against ----

    def add_mark(self, song_id, seconds):
        with self.lock, self.db:
            cursor = self.db.execute("INSERT INTO marks (song_id, seconds, created_at) VALUES (?, ?, ?)",
                                     (song_id, round(float(seconds), 3), time.time()))
            return cursor.lastrowid

    def marks(self, song_id):
        return self._rows("SELECT id, seconds FROM marks WHERE song_id=? ORDER BY seconds", (song_id,))

    def remove_mark(self, song_id, mark_id):
        with self.lock, self.db:
            return self.db.execute("DELETE FROM marks WHERE song_id=? AND id=?", (song_id, mark_id)).rowcount > 0

    def mark_counts(self):
        return {r["song_id"]: r["n"] for r in self._rows("SELECT song_id, COUNT(*) AS n FROM marks GROUP BY song_id")}

    # ---- events ----

    def event(self, message, level="info"):
        _activity.log(_LOG_LEVEL.get(level, 20), message)
        with self.lock, self.db:
            self.db.execute("INSERT INTO events VALUES (?,?,?)", (time.time(), level, message))
            self.db.execute("DELETE FROM events WHERE rowid NOT IN (SELECT rowid FROM events ORDER BY at DESC LIMIT ?)",
                            (EVENTS_KEPT,))

    # ---- bad blends: song changes marked as bad in the app ----

    def add_bad_blend(self, from_id, to_id, technique, details):
        with self.lock, self.db:
            return self.db.execute("INSERT INTO bad_blends (at, from_id, to_id, technique, details) VALUES (?,?,?,?,?)",
                                   (time.time(), from_id, to_id, technique, json.dumps(details or {}))).lastrowid

    def bad_blends(self, limit=200):
        """The latest marks, each with both songs' names (when they're here)."""
        rows = self._rows(
            "SELECT b.*, f.title AS from_title, f.artist AS from_artist, t.title AS to_title, t.artist AS to_artist "
            "FROM bad_blends b LEFT JOIN songs f ON f.id=b.from_id LEFT JOIN songs t ON t.id=b.to_id "
            "ORDER BY b.at DESC, b.id DESC LIMIT ?", (limit,))
        for r in rows:
            r["details"] = json.loads(r["details"] or "{}")
        return rows

    def remove_bad_blend(self, blend_id):
        with self.lock, self.db:
            return self.db.execute("DELETE FROM bad_blends WHERE id=?", (blend_id,)).rowcount > 0

    def bad_blend_count(self):
        return self._rows("SELECT COUNT(*) AS n FROM bad_blends")[0]["n"]

    def events(self, limit=60):
        return self._rows("SELECT at, level, message FROM events ORDER BY at DESC, rowid DESC LIMIT ?", (limit,))
