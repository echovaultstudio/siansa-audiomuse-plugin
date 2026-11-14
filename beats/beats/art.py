"""Covers and artists' pictures for the listening page (0.30, the owner, 11 Nov: "covers and artist images just like
Koito does").

Nothing is fetched from the internet: a picture is the music server's own (the cover the app shows; the artist's
picture the server keeps), asked through the song it belongs to. So there's one for whatever is in the library, and a
coloured tile with the name's first letters for the rest (listens of music that isn't on the server any more).

- A listen says its artist, title and album as text. :class:`Art` finds a song here to ask the server about: the
  listen's own song when it's linked, else any song of that album, else any song of that artist.
- An animated cover (a GIF, an animated WebP) shows as a still: its first frame, made once at the three sizes
  (0.34.1). Navidrome gives one back as the whole file, many megabytes of it, which used to count as no picture.
- A picture is kept in the data folder (``art/``) once fetched, at the three sizes the page uses, and "none" is
  remembered for a day, so a page of fifty rows asks the server once.
"""

import hashlib
import io
import os
import threading
import time

from . import log
from .listens import key

_log = log.get("art")

SIZES = (96, 300, 640)  # thumbnails, cards, the big one
KINDS = ("cover", "artist")
NONE_FOR = 30 * 60  # seconds "the server has none" is believed: a photo it's still fetching comes soon after
INDEX_AT_MOST_EVERY = 60  # seconds: the index of songs by artist and album is not made again sooner than this
MOST_BYTES = 8 * 1024 * 1024  # a still picture is never bigger than this
MOST_ANIMATED = 256 * 1024 * 1024  # an animated cover may be (the owner's library has one of 130 MB): only its first frame is kept


def stills(data, sizes=SIZES):
    """``{size: JPEG bytes}`` of the first frame of an animated picture (or of one too big to keep as it is), no wider
    or taller than each size; None for an ordinary still picture, or when it can't be read (no Pillow, a broken file)."""
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(io.BytesIO(data)) as picture:
            if not getattr(picture, "is_animated", False) and len(data) <= MOST_BYTES:
                return None
            picture.seek(0)
            frame = picture.convert("RGB")
        out = {}
        for size in sizes:
            small = frame.copy()
            small.thumbnail((size, size))
            kept = io.BytesIO()
            small.save(kept, "JPEG", quality=88)
            out[size] = kept.getvalue()
        return out
    except Exception as e:  # not a picture Pillow reads: as before
        _log.debug("A picture's first frame couldn't be made: %s", e)
        return None


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
            # Made again when the songs changed, but not more often than every minute (0.42.3: during an analysis they
            # change with every song, and reading 46,000 of them again held each Listening page up for seconds).
            age = self.now() - self._index[3] if self._index is not None else None
            if self._index is None or age > 600 or (self._index[0] != mark and age > INDEX_AT_MOST_EVERY):
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
        # An animated cover's first frame, at every size from the one download; else the picture as the server gave it.
        made = found[2] if len(found) > 2 else None
        keep = {s: (made[s], "image/jpeg") for s in made} if made else {size: found[:2]}
        os.makedirs(self.folder, exist_ok=True)
        for one, (data, kind_of) in keep.items():
            target = self._path(kind, song_id, one)
            try:
                with open(target + ".part", "wb") as f:
                    f.write(data)
                with open(target + ".type", "w") as t:
                    t.write(kind_of)
                os.replace(target + ".part", target)
            except OSError as e:
                _log.warning("A picture couldn't be kept: %s", e)
        return keep[size]

    def _fetch(self, kind, song_id, size):
        """From the first server that has the song and a picture for it: ``(bytes, type)``, with a third,
        ``{size: JPEG}``, when it's animated (:func:`stills`)."""
        for server, item_id in self.store.where(song_id) or []:
            client = self.servers.client(server)
            if not hasattr(client, "picture"):
                continue
            try:
                found = client.picture(item_id, kind, size)
            except Exception as e:  # a server that's away: no picture now, asked again next time
                _log.debug("No %s picture of %s from %s: %s", kind, song_id, getattr(client, "label", server), e)
                return None
            if not found or not found[0] or len(found[0]) > MOST_ANIMATED:
                continue
            made = stills(found[0])
            if made:
                _log.info("The %s picture of %s is animated (%.1f MB): its first frame is shown", kind, song_id, len(found[0]) / 1e6)
                return found[0], found[1], made
            if len(found[0]) <= MOST_BYTES:
                return found
        _log.info("No %s picture of %s: none of its servers has one now (asked again in %d min)", kind, song_id, NONE_FOR // 60)
        return None

    def kept(self):
        """``{"pictures", "bytes"}`` kept in the data folder."""
        try:
            files = [f for f in os.listdir(self.folder) if not f.endswith((".type", ".part"))]
            return {"pictures": len(files), "bytes": sum(os.path.getsize(os.path.join(self.folder, f)) for f in files)}
        except OSError:
            return {"pictures": 0, "bytes": 0}
