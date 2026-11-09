"""Navidrome (or any OpenSubsonic server), read only: its libraries, their songs and their audio.

Logs in with a salted token (``t``/``s``), never sending the password itself. Nothing on the server is changed.
"""

import hashlib
import json
import secrets
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request

from . import log

_log = log.get("navidrome")

CLIENT = "siansa-beats"
API = "1.16.1"


class SubsonicError(Exception):
    """The server answered with an error, or couldn't be reached."""


class Subsonic:
    def __init__(self, url, user, password, timeout=30.0):
        self.url = (url or "").rstrip("/")
        self.user = user or ""
        self.password = password or ""
        self.timeout = timeout

    @property
    def configured(self):
        return bool(self.url and self.user and self.password)

    def _params(self, **extra):
        salt = secrets.token_hex(6)
        token = hashlib.md5((self.password + salt).encode()).hexdigest()
        params = {"u": self.user, "t": token, "s": salt, "v": API, "c": CLIENT, "f": "json"}
        params.update({k: v for k, v in extra.items() if v is not None})
        return params

    def address(self, method, **extra):
        """The full address of ``method`` with the login in it (never logged: it holds the token)."""
        return f"{self.url}/rest/{method}?{urllib.parse.urlencode(self._params(**extra))}"

    def _get(self, method, **extra):
        if not self.configured:
            raise SubsonicError("Navidrome isn't set up (NAVIDROME_URL, NAVIDROME_USER, NAVIDROME_PASSWORD)")
        started = time.monotonic()
        shown = {k: v for k, v in extra.items() if v is not None}
        try:
            with urllib.request.urlopen(self.address(method, **extra), timeout=self.timeout) as response:
                body = json.load(response)
        except urllib.error.HTTPError as e:
            _log.warning("%s %s: HTTP %s after %.2fs", method, shown, e.code, time.monotonic() - started)
            raise SubsonicError(f"{method}: HTTP {e.code}") from None
        except (urllib.error.URLError, OSError, ValueError) as e:
            _log.warning("%s %s: %s after %.2fs", method, shown, getattr(e, "reason", e), time.monotonic() - started)
            raise SubsonicError(f"{method}: {getattr(e, 'reason', e)}") from None
        answer = body.get("subsonic-response") or {}
        if answer.get("status") != "ok":
            error = answer.get("error") or {}
            _log.warning("%s %s: Navidrome says %s", method, shown, error)
            raise SubsonicError(f"{method}: {error.get('message') or 'the server said no'}")
        _log.debug("%s %s: ok in %.2fs", method, shown, time.monotonic() - started)
        return answer

    def ping(self):
        answer = self._get("ping")
        return {"version": answer.get("serverVersion") or answer.get("version"), "type": answer.get("type")}

    def libraries(self):
        """Navidrome's libraries (Subsonic's music folders): ``[{id, name}]``, ids as text."""
        folders = (self._get("getMusicFolders").get("musicFolders") or {}).get("musicFolder") or []
        return [{"id": str(f["id"]), "name": f.get("name") or str(f["id"])} for f in folders if f.get("id") is not None]

    def all_songs(self, library=None, page=500):
        """Every song in ``library`` (an id from :meth:`libraries`; ``None``: every library), ``page`` at a time:
        ``search3`` with an empty query (as clients that sync the whole library ask Navidrome), or album by album
        when the server finds nothing for an empty query."""
        songs, offset = [], 0
        while True:
            found = (self._get("search3", query="", songCount=page, songOffset=offset, artistCount=0,
                               albumCount=0, musicFolderId=library).get("searchResult3") or {}).get("song") or []
            songs += [song_from(e) for e in found]
            if len(found) < page:
                break
            offset += page
        return songs or self._songs_by_album(library)

    def _songs_by_album(self, library, page=500):
        songs, offset = [], 0
        while True:
            albums = (self._get("getAlbumList2", type="alphabeticalByName", size=page, offset=offset,
                                musicFolderId=library)
                      .get("albumList2") or {}).get("album") or []
            for album in albums:
                songs += [song_from(e) for e in (self._get("getAlbum", id=album["id"]).get("album") or {}).get("song") or []]
            if len(albums) < page:
                return songs
            offset += page

    def song(self, song_id):
        return song_from(self._get("getSong", id=song_id).get("song") or {})

    def download(self, song_id, path):
        """The song's original file, saved at ``path``. Returns its size in bytes."""
        started = time.monotonic()
        try:
            with urllib.request.urlopen(self.address("stream", id=song_id, format="raw"), timeout=self.timeout) as r, \
                    open(path, "wb") as out:
                kind = r.headers.get("Content-Type", "")
                if "json" in kind or "xml" in kind:
                    said = r.read(300).decode(errors="replace")
                    _log.warning("stream %s: Navidrome sent %s instead of the file: %s", song_id, kind, said)
                    raise SubsonicError(f"stream: {said}")
                shutil.copyfileobj(r, out, 1 << 20)
                size = out.tell()
                _log.debug("stream %s: %.1f MB (%s) in %.2fs", song_id, size / 1e6, kind or "no type",
                           time.monotonic() - started)
                return size
        except urllib.error.HTTPError as e:
            _log.warning("stream %s: HTTP %s after %.2fs", song_id, e.code, time.monotonic() - started)
            raise SubsonicError(f"stream: HTTP {e.code}") from None
        except (urllib.error.URLError, OSError) as e:
            _log.warning("stream %s: %s after %.2fs", song_id, getattr(e, "reason", e), time.monotonic() - started)
            raise SubsonicError(f"stream: {getattr(e, 'reason', e)}") from None


def song_from(e):
    """The fields this service keeps, from a Subsonic song entry."""
    genres = [g.get("name") for g in (e.get("genres") or []) if g.get("name")]
    if not genres and e.get("genre"):
        genres = [g.strip() for g in str(e["genre"]).replace(";", "/").split("/") if g.strip()]
    return {
        "id": e.get("id"),
        "title": e.get("title") or "",
        "artist": e.get("artist") or e.get("displayArtist") or "",
        "album": e.get("album") or "",
        "duration": float(e.get("duration") or 0),
        "suffix": e.get("suffix") or "",
        "tag_bpm": float(e.get("bpm") or 0) or None,
        "genres": genres,
    }
