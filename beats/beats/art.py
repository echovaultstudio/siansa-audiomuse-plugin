"""Covers and artists' pictures for the listening page (0.30, the owner, 11 Nov: "covers and artist images just like
Koito does").

Nothing is fetched from the internet: a picture is the music server's own (the cover the app shows; the artist's
picture the server keeps), asked through the song it belongs to. So there's one for whatever is in the library, and a
coloured tile with the name's first letters for the rest (listens of music that isn't on the server any more).

- A listen says its artist, title and album as text. :class:`Art` finds a song here to ask the server about: the
  listen's own song when it's linked, else any song of that album, else any song of that artist.
- A picture is kept in the data folder (``art/``) once fetched, at the three sizes the page uses, and "none" is
  remembered for a day, so a page of fifty rows asks the server once.
"""

import hashlib
import os
import threading
import time

from . import log
from .listens import key

_log = log.get("art")

SIZES = (96, 300, 640)  # thumbnails, cards, the big one
KINDS = ("cover", "artist")
NONE_FOR = 24 * 3600  # seconds "the server has none" is believed
MOST_BYTES = 8 * 1024 * 1024


class Art:
    def __init__(self, store, servers, data_dir, now=time.time):
        self.store, self.servers, self.now = store, servers, now
        self.folder = os.path.join(data_dir, "art")
        self._none = {}  # (kind, song id) -> when the server had none
        self._lock = threading.Lock()
        self._index = None  # (the listens' and songs' change marks, by artist, by artist and album)

    # ---- which song to ask about ----

    def _songs(self):
        """``(by artist, by (artist, album))``: a song here for each artist and album, as names are compared."""
        with self._lock:
            mark = self.store.song_changes if hasattr(self.store, "song_changes") else None
            if self._index is None or self._index[0] != mark or self.now() - self._index[3] > 600:
                by_artist, by_album = {}, {}
                for r in self.store._rows("SELECT id, artist, album FROM songs WHERE artist IS NOT NULL"):
                    artist = key(r["artist"])
                    by_artist.setdefault(artist, r["id"])
                    if r["album"]:
                        by_album.setdefault((artist, key(r["album"])), r["id"])
                self._index = (mark, by_artist, by_album, self.now())
            return self._index[1], self._index[2]

    def song_for(self, kind, song_id=None, artist=None, album=None):
        """The song to ask the server about for a listen's cover or artist: its own, else one of that album (for a
        cover) or of that artist. None when nothing here is theirs."""
        if song_id and kind == "cover":
            return song_id
        by_artist, by_album = self._songs()
        if kind == "cover":
            return by_album.get((key(artist), key(album))) if artist and album else None
        return song_id or (by_artist.get(key(artist)) if artist else None)

    # ---- the picture ----

    def _path(self, kind, song_id, size):
        name = hashlib.sha256(f"{kind}|{song_id}|{size}".encode()).hexdigest()[:32]
        return os.path.join(self.folder, name)

    def get(self, kind, song_id, size=300):
        """``(bytes, content type)`` of ``song_id``'s cover or artist's picture at one of :data:`SIZES` (the nearest
        one up), from the data folder or the music server; None when there's none."""
        if kind not in KINDS or not song_id:
            return None
        size = next((s for s in SIZES if s >= int(size or 0)), SIZES[-1])
        path = self._path(kind, song_id, size)
        try:
            with open(path, "rb") as f, open(path + ".type") as t:
                return f.read(), t.read().strip()
        except OSError:
            pass
        since = self._none.get((kind, song_id))
        if since and self.now() - since < NONE_FOR:
            return None
        found = self._fetch(kind, song_id, size)
        if found is None:
            self._none[(kind, song_id)] = self.now()
            return None
        os.makedirs(self.folder, exist_ok=True)
        try:
            with open(path + ".part", "wb") as f:
                f.write(found[0])
            with open(path + ".type", "w") as t:
                t.write(found[1])
            os.replace(path + ".part", path)
        except OSError as e:
            _log.warning("A picture couldn't be kept: %s", e)
        return found

    def _fetch(self, kind, song_id, size):
        """From the first server that has the song and a picture for it."""
        for server, item_id in self.store.where(song_id) or []:
            client = self.servers.client(server)
            if not hasattr(client, "picture"):
                continue
            try:
                found = client.picture(item_id, kind, size)
            except Exception as e:  # a server that's away: no picture now, asked again next time
                _log.debug("No %s picture of %s from %s: %s", kind, song_id, getattr(client, "label", server), e)
                return None
            if found and found[0] and len(found[0]) <= MOST_BYTES:
                return found
        return None

    def kept(self):
        """``{"pictures", "bytes"}`` kept in the data folder."""
        try:
            files = [f for f in os.listdir(self.folder) if not f.endswith((".type", ".part"))]
            return {"pictures": len(files), "bytes": sum(os.path.getsize(os.path.join(self.folder, f)) for f in files)}
        except OSError:
            return {"pictures": 0, "bytes": 0}
