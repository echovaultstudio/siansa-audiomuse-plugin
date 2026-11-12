"""What's been analysed, in one SQLite file in the data folder (``/data/beats.db``).

Tables:

- ``songs``: each song from the chosen Navidrome libraries, with its BPM tag and its library (``folder``);
- ``genres``: a song's genres, each with where it came from (``source``): ``tag`` (its own tags) and ``sound`` (the
  styles :mod:`beats.style` hears, with their scores: the song's genres, :mod:`beats.genres`; ``sound-failed`` when its
  sound couldn't be read);
- ``jobs``: each song's place in the queue and what's happening to it;
- ``results``: Beat This!'s readings, one row per song and model, with every beat and downbeat;
- ``events``: what the service did, for the page's activity list;
- ``measures``: each song's loudness, fade ramps and key (:mod:`beats.loudness`, :mod:`beats.key`), for the app's
  volume levelling, fades and blends in key (``error`` when it couldn't be measured);
- ``bad_blends``: song changes the owner marked as bad in Siansa Dev, with how the app blended them;
- ``settings``: what the page chose (the libraries to analyse, the music servers added there), kept across restarts;
- ``links``: each music server's own id for a song (``server``, ``item_id``) and the library it's in there. A song two
  servers have is one row of ``songs``, analysed once, with a link from each (:mod:`beats.match`, 0.23.0).

Nothing here is written back to the music servers, AudioMuse-AI or the music files.

Reading doesn't wait for writing (0.23.0): the database is in WAL mode, changes go through one connection behind a
lock, and what the page and the app ask for is read through a few connections of their own. Until then every request
waited for whatever was being read or written, for seconds on a big library.
"""

import contextlib
import json
import queue
import sqlite3
import threading
import time

from . import log
from . import match

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
    key_strength REAL, version INTEGER, measured_at REAL, error TEXT, source TEXT
);
CREATE TABLE IF NOT EXISTS bad_blends (
    id INTEGER PRIMARY KEY AUTOINCREMENT, at REAL, from_id TEXT, to_id TEXT, technique TEXT, details TEXT
);
CREATE TABLE IF NOT EXISTS links (
    server TEXT, item_id TEXT, song_id TEXT, folder TEXT, PRIMARY KEY (server, item_id)
);
-- The sound of each song as AudioMuse-AI's models hear it (0.25): pulled from AudioMuse-AI, later made here.
-- Never its tempo (the owner, 11 Nov): a song's tempo is Beat This!'s alone, in results.
CREATE TABLE IF NOT EXISTS sound (
    song_id TEXT PRIMARY KEY, musicnn BLOB, clap BLOB, key TEXT, scale TEXT, moods TEXT, other TEXT,
    energy REAL, album_artist TEXT, year INTEGER, fp TEXT, source TEXT, at REAL
);
CREATE INDEX IF NOT EXISTS links_by_song ON links (song_id, server);
CREATE INDEX IF NOT EXISTS links_by_server ON links (server, song_id);
CREATE INDEX IF NOT EXISTS links_by_item ON links (item_id);
CREATE INDEX IF NOT EXISTS genres_by_source_song ON genres (source, song_id);  -- the styles' counts, in the index
CREATE INDEX IF NOT EXISTS jobs_by_state ON jobs (state, queued_at, song_id);  -- the queue's next song and its counts
CREATE INDEX IF NOT EXISTS songs_by_name ON songs (artist COLLATE NOCASE, title COLLATE NOCASE);  -- the table's order
"""

MAIN = "main"  # the container's own server (beats.servers.MAIN): every song before 0.23 is its
READERS = 4  # connections for reading: the page asks for a few things at once

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
        self.sound_changes = 0  # the same for the sound table: the vectors in memory are read again when it moves
        self.measure_changes = 0  # the same for the loudness measures
        self.writes = 0  # every change to anything (0.35.1): answers kept while nothing is written are still right
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self._writing = threading.local()  # how deep this thread is in a change: it reads its own change, not a reader's
        self._readers = None
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")  # safe with WAL, and each change no longer waits for the disk twice
        with self._write():
            self.db.executescript(SCHEMA)
            if "tempo" in {r[1] for r in self.db.execute("PRAGMA table_info(sound)")}:
                # 0.25's first builds kept AudioMuse-AI's tempo: it goes, and is never asked for again.
                try:
                    self.db.execute("ALTER TABLE sound DROP COLUMN tempo")
                except sqlite3.OperationalError:  # an SQLite from before columns could be dropped: emptied instead
                    self.db.execute("UPDATE sound SET tempo=NULL")
            if "error" not in {r[1] for r in self.db.execute("PRAGMA table_info(sound)")}:
                self.db.execute("ALTER TABLE sound ADD COLUMN error TEXT")  # 0.27: why a song couldn't be heard here
            if "source" not in {r[1] for r in self.db.execute("PRAGMA table_info(measures)")}:
                self.db.execute("ALTER TABLE measures ADD COLUMN source TEXT")  # 0.20.0's first builds had none
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
            self._to_links()
        if path != ":memory:":
            self._readers = queue.LifoQueue()
            for _ in range(READERS):
                reader = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
                reader.row_factory = sqlite3.Row
                reader.execute("PRAGMA query_only=ON")
                reader.execute("PRAGMA busy_timeout=10000")
                self._readers.put(reader)

    def _to_links(self):
        """0.23.0: what the songs' files and tags are compared by, and each song's link to the container's own server
        (whose id for it is the song's own, as it always was). Done once, in the change that opens the database."""
        columns = {r[1] for r in self.db.execute("PRAGMA table_info(songs)")}
        for column in ("path", "tail", "meta", "loose", "server"):
            if column not in columns:
                self.db.execute(f"ALTER TABLE songs ADD COLUMN {column} TEXT")
        for name in ("tail", "meta", "loose"):
            self.db.execute(f"CREATE INDEX IF NOT EXISTS songs_by_{name} ON songs ({name})")
        rows = self.db.execute("SELECT id, title, artist, album FROM songs WHERE meta IS NULL AND title <> ''").fetchall()
        self.db.executemany("UPDATE songs SET meta=?, loose=? WHERE id=?",
                            [(match.meta(dict(r)), match.loose(dict(r)), r["id"]) for r in rows])
        if not self.db.execute("SELECT 1 FROM settings WHERE key='links_made'").fetchone():
            self.db.execute("INSERT OR IGNORE INTO links (server, item_id, song_id, folder) "
                            "SELECT ?, id, id, folder FROM songs", (MAIN,))
            self.db.execute("INSERT OR REPLACE INTO settings VALUES ('links_made', 'true')")

    def close(self):
        with self.lock:
            while self._readers is not None and not self._readers.empty():
                self._readers.get().close()
            self.db.close()

    @contextlib.contextmanager
    def _write(self):
        """One change, written whole or not at all, one at a time. What this thread reads meanwhile is its own change."""
        with self.lock:
            depth = getattr(self._writing, "depth", 0)
            self._writing.depth = depth + 1
            try:
                with self.db:
                    yield self.db
            finally:
                self._writing.depth = depth
                self.writes += 1

    def _rows(self, sql, args=()):
        if self._readers is None or getattr(self._writing, "depth", 0):
            with self.lock:
                return [dict(r) for r in self.db.execute(sql, args).fetchall()]
        reader = self._readers.get()
        try:
            return [dict(r) for r in reader.execute(sql, args).fetchall()]
        finally:
            self._readers.put(reader)

    # ---- the sound of songs (0.25) ----

    SOUND_FIELDS = ("musicnn", "clap", "key", "scale", "moods", "other", "energy", "album_artist", "year", "fp")

    def save_sound(self, rows, source="audiomuse"):
        """Keeps the sound of songs: each row ``{"song_id", …SOUND_FIELDS}`` (vectors as float32 bytes). The number kept."""
        now = time.time()
        with self._write():
            self.db.executemany(
                "INSERT OR REPLACE INTO sound (song_id, " + ", ".join(self.SOUND_FIELDS) + ", source, at) VALUES (?"
                + ", ?" * len(self.SOUND_FIELDS) + ", ?, ?)",
                [(r["song_id"], *(r.get(f) for f in self.SOUND_FIELDS), source, now) for r in rows])
        self.sound_changes += 1
        return len(rows)

    def sound_failed(self, song_id, error):
        """Keeps why ``song_id`` couldn't be heard here, so it isn't tried again and again (a pull from AudioMuse-AI,
        or hearing it after an update, replaces this)."""
        with self._write():
            self.db.execute("INSERT OR REPLACE INTO sound (song_id, source, at, error) VALUES (?, 'beats', ?, ?)",
                            (song_id, time.time(), str(error)[:300]))

    def without_sound(self, limit=20):
        """Songs analysed for their beats whose sound isn't known (and that didn't fail to be heard), newest first."""
        return [r["song_id"] for r in self._rows(
            "SELECT j.song_id FROM jobs j LEFT JOIN sound o ON o.song_id = j.song_id WHERE j.state = 'done' AND "
            "o.song_id IS NULL ORDER BY j.finished_at DESC LIMIT ?", (limit,))]

    def sound_sources(self, server=None):
        """``{"audiomuse": n, "beats": n, "failed": n}``: where each of ``server``'s songs' sound came from."""
        of, args = self._of(server, "o.song_id")
        out = {"audiomuse": 0, "beats": 0, "failed": 0}
        for r in self._rows("SELECT CASE WHEN o.musicnn IS NULL THEN 'failed' ELSE COALESCE(o.source, 'audiomuse') END "
                            f"AS source, COUNT(*) AS n FROM sound o WHERE 1=1{of} GROUP BY 1", args):
            out[r["source"]] = out.get(r["source"], 0) + r["n"]
        return out

    def sound_prints(self):
        """``{song id: fp}``: what AudioMuse-AI's analysis of each song looked like when it was pulled."""
        return {r["song_id"]: r["fp"] for r in self._rows("SELECT song_id, fp FROM sound")}

    def sound_vectors(self, column="musicnn"):
        """Every song with a ``column`` vector, with what's said of it in an answer: rows of ``{"song_id", "vector"
        (bytes), "title", "artist", "album", "album_artist", "other"}``, by song id."""
        if column not in ("musicnn", "clap"):
            raise ValueError(column)
        return self._rows(f"SELECT o.song_id, o.{column} AS vector, s.title, s.artist, s.album, o.album_artist, o.other "
                          f"FROM sound o JOIN songs s ON s.id=o.song_id WHERE o.{column} IS NOT NULL ORDER BY o.song_id")

    def sound_rows(self, song_ids):
        """``{song id: its sound without the vectors}`` of the songs among ``song_ids`` that have one."""
        out, ids = {}, list(dict.fromkeys(song_ids))
        for start in range(0, len(ids), 500):
            chunk = ids[start:start + 500]
            for r in self._rows("SELECT o.song_id, o.key, o.scale, o.moods, o.other, o.energy, o.album_artist, "
                                "o.year, o.fp, o.source, s.title, s.artist, s.album FROM sound o JOIN songs s ON s.id=o.song_id "
                                f"WHERE o.song_id IN ({', '.join('?' for _ in chunk)})", chunk):
                out[r["song_id"]] = r
        return out

    def energy_old_scale(self):
        """Whether the energies taken over from AudioMuse-AI are on its old scale (beats.drive.old_scale): looked at
        once for each change of the sound table."""
        seen = getattr(self, "_old_scale", None)
        if seen is None or seen[0] != self.sound_changes:
            from .drive import old_scale

            energies = [r["energy"] for r in self._rows("SELECT energy FROM sound WHERE source != 'beats' AND energy IS NOT NULL")]
            seen = self._old_scale = (self.sound_changes, old_scale(energies))
        return seen[1]

    def sound_counts(self, server=None):
        """``{"songs", "sound", "words"}``: ``server``'s songs, how many have their sound, how many the vector the
        search by words needs."""
        of, args = self._of(server)
        row = self._rows("SELECT COUNT(*) AS songs, COUNT(o.musicnn) AS sound, COUNT(o.clap) AS words FROM songs s "
                         f"LEFT JOIN sound o ON o.song_id=s.id WHERE 1=1{of}", args)[0]
        return row

    def server_song_ids(self, server):
        """The ids here of the songs ``server`` has."""
        return {r["song_id"] for r in self._rows("SELECT DISTINCT song_id FROM links WHERE server=?", (server,))}

    # ---- songs and genres ----

    def add_song(self, song, folder=None, source="tag", server=MAIN):
        """Adds or updates ``song`` (from a server's ``song_from``) of ``server``'s library ``folder``, without
        queueing it; its genres replace that source's. The song's id here."""
        with self._write():
            return self._link(server, folder, song, source, (), None)[0]

    def add_songs(self, songs, folder=None, source="tag", server=MAIN):
        """Adds or updates many songs at once (a whole library) and queues the ones never analysed: their ids."""
        return self.link_songs(server, folder, songs, source=source)["queued"]

    def link_songs(self, server, folder, songs, twins=(), source="tag"):
        """A library of ``server`` as just read: each song is one already here (it keeps its link, or is found by its
        file or tags, :mod:`beats.match`; for another user of the same server, ``twins``, by its id) or a new one,
        queued. ``{"songs", "queued": [ids], "matched": {"path", "tags", "title", "user"}}``: how many were found
        here for the first time, by what."""
        queued, matched = [], {}
        with self._write():
            for song in songs:
                song_id, how = self._link(server, folder, song, source, twins, queued)
                if how:
                    matched[how] = matched.get(how, 0) + 1
        return {"songs": len(songs), "queued": queued, "matched": matched}

    def _link(self, server, folder, song, source, twins, queued):
        """One song of a server's library, inside a change: ``(its id here, how it was found here or None)``."""
        item_id = song["id"]
        row = self.db.execute("SELECT song_id FROM links WHERE server=? AND item_id=?", (server, item_id)).fetchone()
        how = None
        if row is not None:
            song_id = row["song_id"]
            self.db.execute("UPDATE links SET folder=COALESCE(?, folder) WHERE server=? AND item_id=?",
                            (folder, server, item_id))
        else:
            song_id, how = self._find(server, song, twins)
            if song_id is None:
                # Its id here is the server's own, as always; with the server's name before it when another has that id.
                taken = self.db.execute("SELECT 1 FROM songs WHERE id=?", (item_id,)).fetchone()
                song_id = f"{server}:{item_id}" if taken else item_id
                self._save_song(song_id, song, folder, server)
            self.db.execute("INSERT OR REPLACE INTO links (server, item_id, song_id, folder) VALUES (?,?,?,?)",
                            (server, item_id, song_id, folder))
        home = self.db.execute("SELECT COALESCE(server, ?) FROM songs WHERE id=?", (MAIN, song_id)).fetchone()
        if home is None:  # a link whose song was removed: the song again
            self._save_song(song_id, song, folder, server)
        elif home[0] == server and how is None:
            self._save_song(song_id, song, folder, server)  # its home's tags are the song's
        elif song.get("path"):
            # A song found on another server keeps its own tags; a path it lacked (read before 0.23) is taken.
            self.db.execute("UPDATE songs SET path=?, tail=? WHERE id=? AND path IS NULL",
                            (song["path"], match.tail(song["path"]), song_id))
        if home is None or home[0] == server and how is None:
            self.db.execute("DELETE FROM genres WHERE song_id=? AND source=?", (song_id, source))
            for label in song.get("genres") or []:
                self.db.execute("INSERT OR REPLACE INTO genres VALUES (?,?,?,?)", (song_id, source, label, 1.0))
        if queued is not None and self.db.execute(
                "INSERT OR IGNORE INTO jobs (song_id, state, queued_at) VALUES (?, 'queued', ?)",
                (song_id, time.time())).rowcount:
            queued.append(song_id)
        return song_id, how

    def _find(self, server, song, twins):
        """The song here that a server's new item is: ``(id, how)``, or ``(None, None)`` for a song of its own. Never
        a song the server already has under another id: two items of one server are two songs."""
        def free(song_id):
            return not self.db.execute("SELECT 1 FROM links WHERE server=? AND song_id=?", (server, song_id)).fetchone()

        for twin in twins:  # another user of the same server: the same id is the same song
            row = self.db.execute("SELECT song_id FROM links WHERE server=? AND item_id=?", (twin, song["id"])).fetchone()
            if row is not None and free(row["song_id"]):
                return row["song_id"], "user"

        def by(column):
            return lambda key: [dict(r) for r in self.db.execute(
                f"SELECT id, path, duration FROM songs WHERE {column}=?", (key,)) if free(r["id"])] if key else []

        return match.choose(song, by("tail"), by("meta"), by("loose"))

    def _save_song(self, song_id, song, folder, server):
        self.db.execute(
            "INSERT INTO songs (id, title, artist, album, duration, suffix, tag_bpm, folder, added_at, path, tail, meta, "
            "loose, server) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET title=excluded.title, "
            "artist=excluded.artist, album=excluded.album, duration=excluded.duration, suffix=excluded.suffix, "
            "tag_bpm=excluded.tag_bpm, folder=COALESCE(excluded.folder, songs.folder), "
            "path=COALESCE(excluded.path, songs.path), tail=COALESCE(excluded.tail, songs.tail), meta=excluded.meta, "
            "loose=excluded.loose",
            (song_id, song["title"], song["artist"], song["album"], song["duration"], song["suffix"],
             song["tag_bpm"], folder, time.time(), song.get("path"), match.tail(song.get("path")), match.meta(song),
             match.loose(song), None if server == MAIN else server))

    # ---- a server's ids for the songs, and the songs of a server's ids ----

    def resolve(self, ids, server=None):
        """``{given id: song id here}`` of the ids a caller has (a server's own; ``server``'s first when two servers
        use the same id), leaving out those nothing here goes by."""
        out, ids = {}, list(dict.fromkeys(ids))
        for start in range(0, len(ids), 500):
            chunk = ids[start:start + 500]
            marks = ", ".join("?" for _ in chunk)
            for r in self._rows(f"SELECT item_id, server, song_id FROM links WHERE item_id IN ({marks})", chunk):
                if r["item_id"] not in out or r["server"] == server:
                    out[r["item_id"]] = r["song_id"]
            for r in self._rows(f"SELECT id FROM songs WHERE id IN ({marks})", chunk):
                out.setdefault(r["id"], r["id"])
        return out

    def item_ids(self, song_ids, server):
        """``{song id here: server's id for it}`` of the songs ``server`` has."""
        out, ids = {}, list(dict.fromkeys(song_ids))
        for start in range(0, len(ids), 500):
            chunk = ids[start:start + 500]
            for r in self._rows("SELECT song_id, item_id FROM links WHERE server=? AND song_id IN "
                                f"({', '.join('?' for _ in chunk)})", (server, *chunk)):
                out.setdefault(r["song_id"], r["item_id"])
        return out

    def where(self, song_id):
        """Where a song can be had: ``[(server, its id there)]``, the server it came from first."""
        rows = self._rows("SELECT l.server, l.item_id, COALESCE(s.server, ?) AS home FROM links l "
                          "LEFT JOIN songs s ON s.id=l.song_id WHERE l.song_id=?", (MAIN, song_id))
        rows.sort(key=lambda r: (r["server"] != r["home"], r["server"] != MAIN, r["server"]))
        return [(r["server"], r["item_id"]) for r in rows]

    def server_counts(self):
        """``{server: songs}``: how many songs each server has here."""
        return {r["server"]: r["n"] for r in self._rows("SELECT server, COUNT(*) AS n FROM links GROUP BY server")}

    @staticmethod
    def _of(server, column="s.id"):
        """SQL (and its value) that keeps only ``server``'s songs; nothing for every server's."""
        return ("", ()) if server is None else (f" AND {column} IN (SELECT song_id FROM links WHERE server=?)", (server,))

    def set_genres(self, song_id, source, labels, weights=None):
        self.genre_changes += 1
        with self._write():
            self.db.execute("DELETE FROM genres WHERE song_id=? AND source=?", (song_id, source))
            for i, label in enumerate(labels):
                weight = weights[i] if weights else 1.0
                self.db.execute("INSERT OR REPLACE INTO genres VALUES (?,?,?,?)", (song_id, source, label, weight))

    # ---- genres (beats.genres): the styles each song's sound has, and its tags beside them ----

    def songs_without_style(self, limit=50):
        """Songs of the chosen libraries whose sound hasn't been given styles yet (nor failed to): the next ones for
        the style model, analysed songs first."""
        return [r["id"] for r in self._rows(
            "SELECT s.id FROM songs s LEFT JOIN jobs j ON j.song_id=s.id WHERE NOT EXISTS (SELECT 1 FROM genres g "
            "WHERE g.song_id=s.id AND g.source IN ('sound', 'sound-failed')) AND COALESCE(j.state, '') != 'failed' "
            "ORDER BY CASE j.state WHEN 'done' THEN 0 ELSE 1 END, s.id LIMIT ?", (limit,))]

    def style_counts(self, server=None):
        """``{"done", "failed", "total"}``: songs with styles from their sound, songs whose sound couldn't be, of all
        (``server``'s songs only, when given)."""
        of, value = self._of(server, "song_id")
        rows = self._rows("SELECT source, COUNT(DISTINCT song_id) AS n FROM genres WHERE source IN ('sound', "
                          f"'sound-failed'){of} GROUP BY source", value)
        by = {r["source"]: r["n"] for r in rows}
        of, value = self._of(server)
        total = self._rows("SELECT COUNT(*) AS n FROM songs s LEFT JOIN jobs j ON j.song_id=s.id "
                           f"WHERE COALESCE(j.state, '') != 'failed'{of}", value)[0]["n"]  # a failed song has no sound to hear
        return {"done": by.get("sound", 0), "failed": by.get("sound-failed", 0), "total": total}

    def tag_genres(self, limit=300):
        """The songs' own genre tags, most songs first: ``["Hard Rock", …]`` (the genres Siansa shows without Siansa Beats)."""
        return [r["label"] for r in self._rows(
            "SELECT label, COUNT(*) AS n FROM genres WHERE source='tag' GROUP BY label ORDER BY n DESC LIMIT ?", (limit,))]

    def genre_rows(self, song_ids=None, sources=("sound", "tag")):
        """``{song_id: {source: [(label, weight)]}}`` of ``song_ids`` (every song when ``None``)."""
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

    # A song is in a style's genre when its sound scores it ``minimum`` or more, and always in its best one.
    HEARD = ("g.source='sound' AND (g.weight >= ? OR NOT EXISTS (SELECT 1 FROM genres b WHERE b.song_id = g.song_id "
             "AND b.source='sound' AND b.weight > g.weight))")

    def heard_of(self, song_ids, minimum):
        """``{song id: [(label, weight)]}``: the genres each of ``song_ids`` is in by its sound, its best first."""
        out, ids = {}, list(dict.fromkeys(song_ids))
        for start in range(0, len(ids), 500):
            chunk = ids[start:start + 500]
            marks = ", ".join("?" for _ in chunk)
            for r in self._rows(f"SELECT g.song_id AS id, g.label AS label, g.weight AS w FROM genres g WHERE {self.HEARD} "
                                f"AND g.song_id IN ({marks}) ORDER BY g.weight DESC, g.label", (minimum, *chunk)):
                out.setdefault(r["id"], []).append((r["label"], r["w"]))
        return out

    def drop_votes(self):
        """Removes the genres voted from sound and tags together (before 0.28). How many rows went."""
        with self._write():
            gone = self.db.execute("DELETE FROM genres WHERE source='vote'").rowcount
        if gone:
            self.genre_changes += 1
        return gone

    def heard_counts(self, minimum, server=None):
        """``{label: songs}``: how many songs (``server``'s only, when given) are in each style's genre by their
        sound."""
        if server is None:
            return {r["label"]: r["n"] for r in self._rows(
                f"SELECT g.label, COUNT(*) AS n FROM genres g WHERE {self.HEARD} GROUP BY g.label", (minimum,))}
        # The server's songs first, then each one's styles (CROSS JOIN keeps that order: five times quicker than the
        # styles first on a big library).
        return {r["label"]: r["n"] for r in self._rows(
            "SELECT g.label, COUNT(*) AS n FROM links l CROSS JOIN genres g ON g.song_id = l.song_id "
            f"WHERE l.server = ? AND {self.HEARD} GROUP BY g.label", (server, minimum))}

    def heard_counts_by_library(self, minimum, server=MAIN):
        """``{label: {library: songs}}``: :meth:`heard_counts` split by the library the songs are in on ``server``, so
        the app can leave out a genre whose songs are all in libraries it hides."""
        out = {}
        for r in self._rows(
                "SELECT g.label, COALESCE(l.folder, '') AS folder, COUNT(*) AS n FROM links l CROSS JOIN genres g "
                f"ON g.song_id = l.song_id WHERE l.server = ? AND {self.HEARD} GROUP BY g.label, l.folder",
                (server, minimum)):
            out.setdefault(r["label"], {})[str(r["folder"])] = r["n"]
        return out

    def songs_heard(self, labels, minimum, limit=None, server=None):
        """Every song id in the genre of any of ``labels`` by their sound, the ones that sound most like it first (the
        first ``limit`` when given): the ids here, or ``server``'s own for the songs it has."""
        labels = list(labels)
        if not labels:
            return []
        marks = ", ".join("?" for _ in labels)
        if server is None:
            rows = self._rows(
                f"SELECT g.song_id AS id, MAX(g.weight) AS w FROM genres g WHERE {self.HEARD} "
                f"AND g.label IN ({marks}) GROUP BY g.song_id", (minimum, *labels))
        else:
            rows = self._rows(
                "SELECT l.item_id AS id, MAX(g.weight) AS w FROM genres g JOIN links l ON l.song_id = g.song_id "
                f"AND l.server = ? WHERE {self.HEARD} AND g.label IN ({marks}) GROUP BY l.item_id",
                (server, minimum, *labels))
        return [r["id"] for r in sorted(rows, key=lambda r: -r["w"])][:limit]

    def genres(self, song_id):
        return self._rows("SELECT source, label, weight FROM genres WHERE song_id=? ORDER BY source, weight DESC, label",
                          (song_id,))

    def song(self, song_id):
        rows = self._rows("SELECT * FROM songs WHERE id=?", (song_id,))
        return rows[0] if rows else None

    def song_ids(self):
        return [r["id"] for r in self._rows("SELECT id FROM songs")]

    GONE_WITH_A_SONG = (("songs", "id"), ("genres", "song_id"), ("jobs", "song_id"), ("results", "song_id"),
                        ("marks", "song_id"), ("measures", "song_id"), ("links", "song_id"), ("sound", "song_id"))

    def remove(self, song_id):
        self.genre_changes += 1
        with self._write():
            for table, key in self.GONE_WITH_A_SONG:
                self.db.execute(f"DELETE FROM {table} WHERE {key}=?", (song_id,))

    def remove_folder(self, folder, server=MAIN):
        """Forgets library ``folder`` of ``server``: the songs only it had go, with their results, marks and place in
        the queue; a song another server (or library) has too stays, analysed. The number of songs it had (nothing
        changes on the server)."""
        return self._unlink("server=? AND folder=?", (server, folder))

    def remove_server(self, server):
        """Forgets ``server``: as :meth:`remove_folder`, for all of its libraries."""
        return self._unlink("server=?", (server,))

    def gone_from(self, server, folder, item_ids):
        """The ids ``server``'s library ``folder`` had here that aren't among ``item_ids`` (what it has now)."""
        now = set(item_ids)
        return [r["item_id"] for r in self._rows("SELECT item_id FROM links WHERE server=? AND folder=?",
                                                 (server, str(folder))) if r["item_id"] not in now]

    def remove_items(self, server, item_ids):
        """Forgets ``server``'s songs ``item_ids`` (gone from it): a song no server has any more goes with everything
        known about it; one another server still has stays. How many songs went altogether."""
        before = len(self._rows("SELECT id FROM songs"))
        ids = list(dict.fromkeys(item_ids))
        for start in range(0, len(ids), 400):
            chunk = ids[start:start + 400]
            self._unlink(f"server=? AND item_id IN ({', '.join('?' for _ in chunk)})", (server, *chunk))
        if ids:
            self.sound_changes += 1
        return before - len(self._rows("SELECT id FROM songs"))

    def _unlink(self, where, args):
        self.genre_changes += 1
        with self._write():
            ids = [r[0] for r in self.db.execute(f"SELECT DISTINCT song_id FROM links WHERE {where}", args)]
            self.db.execute(f"DELETE FROM links WHERE {where}", args)
            for song_id in ids:
                left = self.db.execute("SELECT server, folder FROM links WHERE song_id=? ORDER BY server=? DESC, server "
                                       "LIMIT 1", (song_id, MAIN)).fetchone()
                if left is None:
                    for table, key in self.GONE_WITH_A_SONG:
                        self.db.execute(f"DELETE FROM {table} WHERE {key}=?", (song_id,))
                else:  # it lives on where it still is
                    self.db.execute("UPDATE songs SET server=?, folder=? WHERE id=?",
                                    (None if left["server"] == MAIN else left["server"], left["folder"], song_id))
            return len(ids)

    # ---- the queue ----

    def queue(self, song_id, force=False):
        """Puts the song in the queue; with ``force`` also when it's done or failed. True when it was queued."""
        with self._write():
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
        with self._write():
            row = self.db.execute(self.NEXT).fetchone()
            if not row:
                return None
            self.db.execute("UPDATE jobs SET state='downloading', step='downloading', started_at=? WHERE song_id=?",
                            (time.time(), row["song_id"]))
            return row["song_id"]

    def unqueue_except(self, folders, server=MAIN):
        """Takes ``server``'s songs of every library but ``folders`` out of the queue (those done stay, and those
        another server has in a chosen library of its own): the number taken."""
        folders = list(folders)
        with self._write():
            return self.db.execute(
                "DELETE FROM jobs WHERE state='queued' AND song_id IN (SELECT song_id FROM links WHERE server=? "
                "AND folder IS NOT NULL" + (f" AND folder NOT IN ({', '.join('?' for _ in folders)})" if folders else "")
                + ") AND song_id NOT IN (SELECT song_id FROM links WHERE server<>?"
                + (f" OR folder IN ({', '.join('?' for _ in folders)})" if folders else "") + ")",
                (server, *folders, server, *folders)).rowcount

    def folder_counts(self, server=MAIN):
        """``{folder: {"songs", "done"}}``: each of ``server``'s libraries' songs here, and how many are analysed."""
        rows = self._rows("SELECT l.folder, COUNT(*) AS songs, SUM(j.state='done') AS done FROM links l "
                          "LEFT JOIN jobs j ON j.song_id=l.song_id WHERE l.server=? GROUP BY l.folder", (server,))
        return {r["folder"]: {"songs": r["songs"], "done": r["done"] or 0} for r in rows if r["folder"] is not None}

    # ---- settings ----

    def setting(self, key, default=None):
        rows = self._rows("SELECT value FROM settings WHERE key=?", (key,))
        return json.loads(rows[0]["value"]) if rows else default

    def set_setting(self, key, value):
        with self._write():
            self.db.execute("INSERT OR REPLACE INTO settings VALUES (?, ?)", (key, json.dumps(value)))

    def set_state(self, song_id, state, error=None):
        now = time.time()
        with self._write():
            if state in BUSY:
                self.db.execute("UPDATE jobs SET state=?, step=?, started_at=COALESCE(started_at, ?) WHERE song_id=?",
                                (state, state, now, song_id))
            else:
                self.db.execute("UPDATE jobs SET state=?, step=NULL, finished_at=?, error=? WHERE song_id=?",
                                (state, now, error, song_id))

    def counts(self, server=None):
        """``{state: songs, "songs": all}`` of the queue (``server``'s songs only, when given)."""
        of, value = self._of(server, "song_id")
        rows = self._rows(f"SELECT state, COUNT(*) AS n FROM jobs WHERE 1{of} GROUP BY state", value)
        counts = {r["state"]: r["n"] for r in rows}
        of, value = self._of(server, "id")
        counts["songs"] = self._rows(f"SELECT COUNT(*) AS n FROM songs WHERE 1{of}", value)[0]["n"]
        return counts

    # ---- results ----

    def save_result(self, song_id, model, values, beats, downbeats, levels=None):
        """``levels``: the sound's levels (:func:`beats.features.measure`); ``None`` saves an empty ``""``, so a
        song whose sound couldn't be measured isn't asked for again."""
        with self._write():
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

    def failed(self):
        """``[{"id", "error"}]`` of the songs that couldn't be analysed."""
        return self._rows("SELECT song_id AS id, error FROM jobs WHERE state='failed'")

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
        with self._write():
            self.db.execute(f"UPDATE results SET {', '.join(f'{k}=?' for k in keys)} WHERE song_id=? AND model=?",
                            (*[_column(k, values[k]) for k in keys], song_id, model))

    def tempos(self, song_ids, model):
        """``{song_id: bpm}`` of the songs among ``song_ids`` Beat This! has heard: the only tempo a song has here."""
        out, ids = {}, list(dict.fromkeys(song_ids))
        for start in range(0, len(ids), 500):
            chunk = ids[start:start + 500]
            for r in self._rows("SELECT song_id, bpm FROM results WHERE model=? AND bpm IS NOT NULL AND song_id IN "
                                f"({', '.join('?' for _ in chunk)})", (model, *chunk)):
                out[r["song_id"]] = r["bpm"]
        return out

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

    def save_measure(self, song_id, values, version, error=None, source="beats"):
        """``values``: :func:`beats.loudness.profile` with :func:`beats.key.find`'s fields (either may be missing);
        ``error`` when the song couldn't be measured, so it isn't tried again and again. ``source``: "beats" (measured
        here) or "plugin" (taken from AudioMuse-AI's Siansa plugin)."""
        self.measure_changes += 1
        with self._write():
            self.db.execute(
                "INSERT OR REPLACE INTO measures (song_id, ref_lufs, start_ramp, end_ramp, duration_ms, key, scale, "
                "key_strength, version, measured_at, error, source) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (song_id, values.get("ref_lufs"), values.get("start_ramp"), values.get("end_ramp"),
                 values.get("duration_ms"), values.get("key"), values.get("scale"), values.get("key_strength"),
                 version, time.time(), (error or None) and str(error)[:300], source))

    def measure(self, song_id):
        """One song's row of ``measures`` (with ``error`` and ``source``), or None."""
        rows = self._rows("SELECT * FROM measures WHERE song_id=?", (song_id,))
        return rows[0] if rows else None

    def measure_summary(self, server=None):
        """For the page's loudness dashboard: every measured song's loudness, key and where it came from, and the
        songs that couldn't be measured with why: ``{"rows": [{"ref_lufs", "key", "scale", "source", "start_ramp",
        "end_ramp"}], "failed": [{"id", "title", "artist", "error"}]}``."""
        of, value = self._of(server, "song_id")
        rows = self._rows("SELECT ref_lufs, key, scale, source, start_ramp, end_ramp FROM measures "
                          f"WHERE error IS NULL{of}", value)
        of, value = self._of(server, "m.song_id")
        failed = self._rows("SELECT m.song_id AS id, s.title, s.artist, m.error, m.measured_at FROM measures m "
                            f"LEFT JOIN songs s ON s.id = m.song_id WHERE m.error IS NOT NULL{of} "
                            "ORDER BY m.measured_at DESC LIMIT 200", value)
        return {"rows": rows, "failed": failed}

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

    def measure_counts(self, version, server=None):
        """``{"measured", "failed", "waiting"}`` for the page and the capabilities (``server``'s songs, when given)."""
        of, value = self._of(server, "j.song_id")
        row = self._rows(
            "SELECT SUM(CASE WHEN m.error IS NULL AND m.version >= ? THEN 1 ELSE 0 END) AS measured, "
            "SUM(CASE WHEN m.error IS NOT NULL THEN 1 ELSE 0 END) AS failed, "
            "SUM(CASE WHEN m.song_id IS NULL OR (m.version < ? AND m.error IS NULL) THEN 1 ELSE 0 END) AS waiting "
            f"FROM jobs j LEFT JOIN measures m ON m.song_id = j.song_id WHERE j.state = 'done'{of}",
            (version, version, *value))[0]
        return {k: row[k] or 0 for k in ("measured", "failed", "waiting")}

    def overview(self, model, find=None, state=None, limit=None, song_id=None, server=None):
        """The songs with their job and result (no beats), for the table: those whose name, artist,
        album or genre has ``find`` in it, only those in ``state`` ("done", "failed"), at most ``limit`` of them, or just ``song_id``."""
        where, args = [], [model]
        if server is not None:
            where.append("s.id IN (SELECT song_id FROM links WHERE server=?)")
            args.append(server)
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
            + ", ".join(f"r.{f} AS r_{f}" for f in RESULT_FIELDS) + ", r.analysed_at AS r_analysed_at, "
            + "m.ref_lufs AS m_ref_lufs, m.key AS m_key, m.scale AS m_scale, m.source AS m_source, m.error AS m_error"
            + " FROM songs s LEFT JOIN jobs j ON j.song_id=s.id LEFT JOIN results r ON r.song_id=s.id AND r.model=?"
            + " LEFT JOIN measures m ON m.song_id=s.id"
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
            # Loudness and key (for the app), or why they couldn't be measured; None when not measured yet.
            m = {k: row.pop(f"m_{k}") for k in ("ref_lufs", "key", "scale", "source", "error")}
            row["sound"] = m if (m["ref_lufs"] is not None or m["error"] or m["source"]) else None
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
        with self._write():
            cursor = self.db.execute("INSERT INTO marks (song_id, seconds, created_at) VALUES (?, ?, ?)",
                                     (song_id, round(float(seconds), 3), time.time()))
            return cursor.lastrowid

    def marks(self, song_id):
        return self._rows("SELECT id, seconds FROM marks WHERE song_id=? ORDER BY seconds", (song_id,))

    def remove_mark(self, song_id, mark_id):
        with self._write():
            return self.db.execute("DELETE FROM marks WHERE song_id=? AND id=?", (song_id, mark_id)).rowcount > 0

    def mark_counts(self):
        return {r["song_id"]: r["n"] for r in self._rows("SELECT song_id, COUNT(*) AS n FROM marks GROUP BY song_id")}

    # ---- events ----

    def event(self, message, level="info"):
        _activity.log(_LOG_LEVEL.get(level, 20), message)
        with self._write():
            self.db.execute("INSERT INTO events VALUES (?,?,?)", (time.time(), level, message))
            self.db.execute("DELETE FROM events WHERE rowid NOT IN (SELECT rowid FROM events ORDER BY at DESC LIMIT ?)",
                            (EVENTS_KEPT,))

    # ---- bad blends: song changes marked as bad in the app ----

    def add_bad_blend(self, from_id, to_id, technique, details):
        with self._write():
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
        with self._write():
            return self.db.execute("DELETE FROM bad_blends WHERE id=?", (blend_id,)).rowcount > 0

    def bad_blend_count(self):
        return self._rows("SELECT COUNT(*) AS n FROM bad_blends")[0]["n"]

    def events(self, limit=60):
        return self._rows("SELECT at, level, message FROM events ORDER BY at DESC, rowid DESC LIMIT ?", (limit,))
