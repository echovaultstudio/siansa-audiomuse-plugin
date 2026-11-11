"""What songs sound like, and what's found from it (0.25, the owner, 11 Nov: Siansa Beats as Siansa's only companion):
songs like a song, a path from one song to another, a blend of songs, moods, "your sound", similar albums and artists.

Every song has a vector from AudioMuse-AI's first model (MusiCNN, 200 numbers) that says how it sounds: two songs sound
alike when their vectors point the same way (cosine distance, AudioMuse-AI's default). The vectors come from
AudioMuse-AI's own export for apps (:mod:`beats.audiomuse`, ``/api/sync``), so nothing is analysed twice, and are kept
in the ``sound`` table by this service's song ids: the answers are given in the ids of whichever server the app plays
from.

All of them in memory are a matrix (46,000 songs: 37 MB), and a search is one multiplication over it (a few
milliseconds), so no index is built or kept. It's read again when the table changes.

The answers follow AudioMuse-AI's (so the app reads them the same, and can be compared with it), with its defaults:

- **like a song**: the nearest songs, without the song itself or another copy of it (same title and artist), without
  songs that are the same recording again (closer than :data:`SAME_RECORDING` to one already in the answer), at most
  :data:`PER_ARTIST` by one artist; with ``same_mood``, only songs whose mood (danceable, aggressive, happy, party,
  relaxed, sad) is within :data:`MOOD_APART` of the song's;
- **a path**: the line between the two songs in steps; at each step the nearest song not used yet;
- **a blend** ("alchemy"): the songs nearest each song to blend, in turns, leaving out those within
  :data:`AWAY` of a song to avoid;
- **a mood**: the songs nearest one of AudioMuse-AI's mood centres (``mood_centroids.json``, its file);
- **your sound**: the songs nearest the average of the songs given (the most played).
"""

import json
import os
import threading
import time

import numpy as np

from . import log, similarity

_log = log.get("sound")

DIMENSIONS = 200
WORD_DIMENSIONS = 512  # CLAP's: the vector a description in words is compared with (beats.words)
PER_ARTIST = 3  # AudioMuse-AI's MAX_SONGS_PER_ARTIST
SAME_RECORDING = 0.01  # its DUPLICATE_DISTANCE_THRESHOLD_COSINE
MOOD_APART = 0.15  # its MOOD_SIMILARITY_THRESHOLD
AWAY = 0.2  # its ALCHEMY_SUBTRACT_DISTANCE_ANGULAR
MOODS = ("danceable", "aggressive", "happy", "party", "relaxed", "sad")  # its OTHER_FEATURE_LABELS
MOST = 200  # the most songs one answer holds
MASK_SECONDS = 60  # how long "which songs a server has" is kept: libraries are read again now and then
CENTRES = os.path.join(os.path.dirname(__file__), "mood_centroids.json")


class Unknown(Exception):
    """A song (or mood) asked about isn't known here, or its sound isn't."""


def pairs(text):
    """``"rock:0.61,pop:0.2"`` (how AudioMuse-AI writes moods and genres) as ``{"rock": 0.61, "pop": 0.2}``."""
    out = {}
    for part in str(text or "").split(","):
        name, _, value = part.rpartition(":")
        try:
            if name.strip():
                out[name.strip()] = float(value)
        except ValueError:
            pass
    return out


def unit(matrix):
    """Each row at length one (a row of nothing stays nothing), so a product of two rows is their cosine."""
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    lengths = np.linalg.norm(matrix, axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0
    return matrix / lengths


class Sound:
    def __init__(self, store, centres=CENTRES):
        self.store = store
        self._lock = threading.Lock()
        self._seen = None  # the store's sound_changes these were read at
        self._links = {}  # server -> (when it was read, which rows the server has)
        self.ids, self.rows, self.index = [], [], {}
        self.matrix = np.zeros((0, DIMENSIONS), np.float32)
        self.moods = np.zeros((0, len(MOODS)), np.float32)
        self._albums = None  # (albums, artists) grouped, their lists made when first asked for
        self._centres_file, self._centres = centres, None
        self._words = None  # (the store's sound_changes it was read at, ids, rows, matrix): read when first searched

    # ---- the songs in memory ----

    def _load(self):
        with self._lock:
            if self._seen == self.store.sound_changes:
                return
            seen = self.store.sound_changes
            rows = [r for r in self.store.sound_vectors() if r["vector"] and len(r["vector"]) == DIMENSIONS * 4]
            self.ids = [r["song_id"] for r in rows]
            self.index = {song_id: i for i, song_id in enumerate(self.ids)}
            raw = np.frombuffer(b"".join(r["vector"] for r in rows), dtype=np.float32).reshape(-1, DIMENSIONS) \
                if rows else np.zeros((0, DIMENSIONS), np.float32)
            self.matrix = unit(np.nan_to_num(raw))
            self.raw = raw
            feelings = [pairs(r["other"]) for r in rows]
            self.moods = np.array([[f.get(m, 0.0) for m in MOODS] for f in feelings], np.float32).reshape(-1, len(MOODS))
            self.rows = [{k: r[k] for k in ("song_id", "title", "artist", "album", "album_artist")} for r in rows]
            self._albums, self._links = None, {}
            self._seen = seen
            _log.info("The sound of %d songs is in memory", len(self.ids))

    def count(self):
        self._load()
        return len(self.ids)

    def _mask(self, server):
        """Which rows ``server`` has (all of them without one)."""
        if server is None:
            return None
        at, mask = self._links.get(server, (0.0, None))
        if mask is None or time.monotonic() - at > MASK_SECONDS:
            have = self.store.server_song_ids(server)
            mask = np.array([song_id in have for song_id in self.ids], bool)
            self._links[server] = (time.monotonic(), mask)
        return mask

    def _vector(self, song_id):
        self._load()
        at = self.index.get(song_id)
        if at is None:
            raise Unknown(f"The sound of {song_id} isn't known yet")
        return self.matrix[at]

    # ---- finding ----

    def near(self, vector, server=None, wanted=MOST):
        """The rows nearest ``vector``, nearest first, as ``[(row, distance)]`` (cosine distance: 0 the same)."""
        self._load()
        if not len(self.ids):
            return []
        scores = self.matrix @ unit(vector)[0]
        mask = self._mask(server)
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        wanted = min(wanted, len(self.ids))
        top = np.argpartition(-scores, wanted - 1)[:wanted]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(int(i), float(1.0 - scores[i])) for i in top if np.isfinite(scores[i])]

    def _pick(self, found, n, taken=(), per_artist=PER_ARTIST, mood_of=None, away=None):
        """From ``found`` (``[(row, distance)]``, nearest first) the first ``n`` that make a varied answer: no song
        twice (by title and artist, nor one of ``taken``'s), no recording again, at most ``per_artist`` by one artist;
        with ``mood_of`` (a row), only songs in its mood; with ``away`` (vectors), none near one of them."""
        signatures = {self._signature(i) for i in taken}
        artists, picked = {}, []
        for row, distance in found:
            if len(picked) >= n:
                break
            if row in taken:
                continue
            signature = self._signature(row)
            if signature in signatures:
                continue
            if picked and 1.0 - float((self.matrix[[r for r, _ in picked]] @ self.matrix[row]).max()) < SAME_RECORDING:
                continue
            artist = signature[1]
            if per_artist and artists.get(artist, 0) >= per_artist:
                continue
            if mood_of is not None and float(np.abs(self.moods[row] - self.moods[mood_of]).mean()) > MOOD_APART:
                continue
            if away is not None and len(away) and float((away @ self.matrix[row]).max()) > 1.0 - AWAY:
                continue
            signatures.add(signature)
            artists[artist] = artists.get(artist, 0) + 1
            picked.append((row, distance))
        return picked

    def _signature(self, row):
        r = self.rows[row]
        return (similarity.norm(r["title"]), similarity.norm(r["artist"]))

    def _songs(self, picked):
        """An answer's songs, as AudioMuse-AI gives them (``item_id`` is this service's id: the caller turns it into
        the server's)."""
        return [{"item_id": self.rows[row]["song_id"], "title": self.rows[row]["title"] or "",
                 "author": self.rows[row]["artist"] or "", "album": self.rows[row]["album"] or "",
                 "distance": round(max(0.0, distance), 6)} for row, distance in picked]

    # ---- what's asked ----

    def similar(self, song_id, n=50, same_mood=False, server=None):
        """Songs that sound like ``song_id``."""
        vector = self._vector(song_id)
        here = self.index[song_id]
        n = max(1, min(MOST, int(n)))
        found = self.near(vector, server, wanted=min(len(self.ids), n * 8 + 50))
        return self._songs(self._pick(found, n, taken={here}, mood_of=here if same_mood else None))

    def around(self, vector, n=50, server=None):
        """Songs nearest a point in the sound (a mood's centre, an average)."""
        n = max(1, min(MOST, int(n)))
        return self._songs(self._pick(self.near(vector, server, wanted=n * 8 + 50), n))

    def path(self, start, end, length=25, server=None):
        """``length`` songs from ``start`` to ``end``, both counted; each a song id, or a vector (a mood's centre)."""
        self._load()
        length = max(2, min(MOST, int(length)))

        def end_of(what):
            if isinstance(what, str):
                return self._vector(what), self.index[what]
            return unit(what)[0], None

        (a, a_row), (b, b_row) = end_of(start), end_of(end)
        taken = {r for r in (a_row, b_row) if r is not None}
        signatures_of_ends = set(taken)
        steps = length - len(taken)
        middle, artists = [], {}
        for row in taken:
            artists[self._signature(row)[1]] = 1
        for step in range(1, steps + 1):
            point = a + (b - a) * (step / (steps + 1))
            found = self.near(point, server, wanted=60 + len(taken) + len(middle))
            used = signatures_of_ends | {r for r, _ in middle}
            pick = None
            signatures = {self._signature(r) for r in used}
            for row, distance in found:
                signature = self._signature(row)
                if row in used or signature in signatures or artists.get(signature[1], 0) >= PER_ARTIST:
                    continue
                pick = (row, distance)
                break
            if pick is None:  # a small library: the artists' limit is let go before the path is cut short
                pick = next(((r, d) for r, d in found if r not in used and self._signature(r) not in signatures), None)
            if pick is None:
                break
            middle.append(pick)
            artists[self._signature(pick[0])[1]] = artists.get(self._signature(pick[0])[1], 0) + 1
        out = ([(a_row, 0.0)] if a_row is not None else []) + middle + ([(b_row, 0.0)] if b_row is not None else [])
        return self._songs(out)

    def blend(self, add, avoid=(), n=50, server=None):
        """Songs nearest each of ``add`` (song ids) in turns, without any near one of ``avoid``."""
        self._load()
        add = [s for s in dict.fromkeys(add) if s in self.index]
        if not add:
            raise Unknown("The sound of none of those songs is known yet")
        n = max(1, min(MOST, int(n)))
        seeds = {self.index[s] for s in add} | {self.index[s] for s in avoid if s in self.index}
        away = self.matrix[[self.index[s] for s in avoid if s in self.index]] if avoid else None
        # One list for each song to blend; their songs are then taken in turns, so each has its share.
        lists = [self._pick(self.near(self.matrix[self.index[s]], server, wanted=n * 6 + 50), n, taken=seeds,
                            per_artist=0, away=away) for s in add]
        merged, seen, signatures, artists = [], set(seeds), set(), {}
        for turn in range(max((len(l) for l in lists), default=0)):
            for found in lists:
                if turn >= len(found) or len(merged) >= n:
                    continue
                row, distance = found[turn]
                signature = self._signature(row)
                if row in seen or signature in signatures or artists.get(signature[1], 0) >= PER_ARTIST:
                    continue
                seen.add(row)
                signatures.add(signature)
                artists[signature[1]] = artists.get(signature[1], 0) + 1
                merged.append((row, distance))
        merged.sort(key=lambda found: found[1])
        return self._songs(merged)

    def yours(self, song_ids, n=50, server=None):
        """"Your sound": songs nearest the average of ``song_ids`` (the most played, the most played first, which
        count for more), the songs themselves among them."""
        self._load()
        rows = [self.index[s] for s in dict.fromkeys(song_ids) if s in self.index]
        if not rows:
            raise Unknown("The sound of none of your most played songs is known yet")
        weights = np.linspace(1.0, 0.5, len(rows), dtype=np.float32)[:, None]
        return self.around((self.matrix[rows] * weights).sum(axis=0), n, server)

    # ---- by words (the vector comes from beats.words) ----

    def _word_songs(self):
        with self._lock:
            if self._words is None or self._words[0] != self.store.sound_changes:
                seen = self.store.sound_changes
                rows = [r for r in self.store.sound_vectors("clap") if r["vector"] and len(r["vector"]) == WORD_DIMENSIONS * 4]
                raw = np.frombuffer(b"".join(r["vector"] for r in rows), dtype=np.float32).reshape(-1, WORD_DIMENSIONS) \
                    if rows else np.zeros((0, WORD_DIMENSIONS), np.float32)
                self._words = (seen, [r["song_id"] for r in rows],
                               [{k: r[k] for k in ("song_id", "title", "artist", "album")} for r in rows],
                               unit(np.nan_to_num(raw)))
                _log.info("The vectors of %d songs for the search by words are in memory", len(rows))
            return self._words[1:]

    def word_count(self):
        return len(self._word_songs()[0])

    def described(self, vector, n=50, server=None):
        """The songs nearest ``vector`` (a description, from :meth:`beats.words.Words.embed`), as AudioMuse-AI's text
        search gives them: ``{"item_id", "title", "author", "album", "similarity"}``, the most alike first, no song
        twice (by title and artist), the same recording never twice, at most :data:`PER_ARTIST` by one artist (none
        of that limit from 1000 songs up, as AudioMuse-AI)."""
        ids, rows, matrix = self._word_songs()
        if not ids:
            return []
        n = max(1, int(n))
        scores = matrix @ unit(vector)[0]
        if server is not None:
            have = self.store.server_song_ids(server)
            scores = np.where(np.array([song_id in have for song_id in ids], bool), scores, -np.inf)
        wanted = min(len(ids), n * 8 + 50)
        top = np.argpartition(-scores, wanted - 1)[:wanted]
        top = top[np.argsort(-scores[top], kind="stable")]
        per_artist = 0 if n >= 1000 else PER_ARTIST
        picked, signatures, artists = [], set(), {}
        for i in top:
            if len(picked) >= n or not np.isfinite(scores[i]):
                break
            signature = (similarity.norm(rows[i]["title"]), similarity.norm(rows[i]["artist"]))
            if signature in signatures or (per_artist and artists.get(signature[1], 0) >= per_artist):
                continue
            if picked and 1.0 - float((matrix[picked] @ matrix[i]).max()) < SAME_RECORDING:
                continue
            signatures.add(signature)
            artists[signature[1]] = artists.get(signature[1], 0) + 1
            picked.append(int(i))
        return [{"item_id": rows[i]["song_id"], "title": rows[i]["title"] or "", "author": rows[i]["artist"] or "",
                 "album": rows[i]["album"] or "", "similarity": round(float(scores[i]), 6)} for i in picked]

    # ---- moods ----

    def centres(self):
        """AudioMuse-AI's mood centres: ``{mood: [{"centroid": vector, "top_tags": {tag: share}, …}]}``."""
        if self._centres is None:
            try:
                with open(self._centres_file, encoding="utf-8") as f:
                    self._centres = {mood: info.get("centroids") or [] for mood, info in json.load(f).items()}
            except (OSError, ValueError) as e:
                _log.warning("The mood centres couldn't be read: %s", e)
                self._centres = {}
        return self._centres

    def centre_list(self):
        """The centres as AudioMuse-AI lists them, without their vectors."""
        return {mood: [{"index": i, "top_tags": [t for t, _ in sorted((c.get("top_tags") or {}).items(),
                                                                       key=lambda tag: -tag[1])[:5]],
                        "n_songs": c.get("n_songs", 0), "mood_score": c.get("mood_score", 0),
                        "cluster_id": c.get("cluster_id", i)} for i, c in enumerate(centres)]
                for mood, centres in self.centres().items()}

    def centre(self, mood, index=None):
        """One mood centre's vector; without ``index``, the average of the mood's (a path's end)."""
        centres = self.centres().get(str(mood or "").lower())
        if not centres:
            raise Unknown(f"Unknown mood '{mood}': one of {', '.join(self.centres()) or 'none'}")
        if index is None:
            return np.mean(unit([c["centroid"] for c in centres]), axis=0)
        if not 0 <= index < len(centres):
            raise Unknown(f"Mood '{mood}' has centres 0 to {len(centres) - 1}")
        return np.array(centres[index]["centroid"], np.float32)

    def mood(self, mood, index, n=50, server=None):
        return self.around(self.centre(mood, index), n, server)

    # ---- albums and artists ----

    def _grouped(self):
        self._load()
        if self._albums is None:
            tracks = [(r["song_id"], r["album"], r["album_artist"], r["artist"], self.raw[i])
                      for i, r in enumerate(self.rows)]
            albums, artists = similarity.group_tracks(tracks)
            self._albums = {"albums": albums, "artists": artists, "album_lists": None, "artist_lists": None}
        return self._albums

    def similar_albums(self, song_id):
        """``(name, artist, [{album, artist, track_id, score}])`` of the albums like ``song_id``'s."""
        self._load()
        at = self.index.get(song_id)
        if at is None or not self.rows[at]["album"]:
            raise Unknown("The sound of this song or its album isn't known yet")
        grouped, r = self._grouped(), self.rows[at]
        if grouped["album_lists"] is None:
            grouped["album_lists"] = similarity.rank_albums(grouped["albums"])
        key = similarity.album_key(similarity.artist_of(r["album_artist"], r["artist"]), r["album"])
        group = grouped["albums"].get(key)
        return (group.name if group else r["album"], group.artist if group else r["artist"],
                grouped["album_lists"].get(key) or [])

    def similar_artists(self, song_id=None, artist=None):
        """``(name, [{artist, track_id, score}])`` of the artists like ``song_id``'s, or like ``artist`` by name."""
        self._load()
        if song_id:
            at = self.index.get(song_id)
            if at is None:
                raise Unknown("The sound of this song isn't known yet")
            artist = (self.rows[at]["artist"] or self.rows[at]["album_artist"] or "").strip()
        if not artist:
            raise Unknown("A song or an artist is needed")
        grouped = self._grouped()
        if grouped["artist_lists"] is None:
            grouped["artist_lists"] = similarity.rank_artists(grouped["artists"])
        return artist, grouped["artist_lists"].get(similarity.norm(artist)) or []
