"""Navidrome (or any OpenSubsonic server), read only: its libraries, their songs and their audio.

Logs in with a salted token (``t``/``s``), never sending the password itself. Nothing on the server is changed.

Each song comes with its file's path, to find the same file on another server (:mod:`beats.match`): Navidrome's own
API says where the file really is (Subsonic's ``path`` is made up from the tags unless the player is set to report
real paths), so it's asked first, with the same user and password; Subsonic's path is the fallback.
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


class ServerError(Exception):
    """A music server answered with an error, or couldn't be reached."""


SubsonicError = ServerError  # its name until 0.23, when Emby and Jellyfin came


class Subsonic:
    kind, label = "navidrome", "Navidrome"

    def __init__(self, url, user, password, timeout=30.0):
        self.url = (url or "").rstrip("/")
        self.user = user or ""
        self.password = password or ""
        self.timeout = timeout
        self._native_token = None

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
            raise SubsonicError("Navidrome isn't set up: add it under Setup › Music servers")
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
        songs = songs or self._songs_by_album(library)
        try:  # where each file really is, for finding it on another server
            real = self.real_paths(library)
        except ServerError as e:
            real = {}
            _log.debug("Navidrome's own API didn't give the files' paths (Subsonic's are used): %s", e)
        for song in songs:
            if real.get(song["id"]):
                song["path"] = real[song["id"]]
        return songs

    def _native(self, path, token=None, body=None):
        """One call of Navidrome's own API (the one its web page uses), as JSON."""
        request = urllib.request.Request(self.url + path, data=json.dumps(body).encode() if body is not None else None,
                                         headers={"Content-Type": "application/json"})
        if token:
            request.add_header("X-ND-Authorization", "Bearer " + token)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as e:
            raise ServerError(f"{path.split('?')[0]}: HTTP {e.code}") from None
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise ServerError(f"{path.split('?')[0]}: {getattr(e, 'reason', e)}") from None

    def real_paths(self, library=None, page=1000):
        """``{song id: path}`` of ``library``'s songs (every library's with ``None``), as Navidrome's own API has them:
        the file's real place, under its library's folder."""
        if not self.configured:
            raise ServerError("Navidrome isn't set up")
        if not self._native_token:
            answer = self._native("/auth/login", body={"username": self.user, "password": self.password})
            self._native_token = (answer or {}).get("token")
            if not self._native_token:
                raise ServerError("/auth/login: no token")
        out, start = {}, 0
        while True:
            query = {"_start": start, "_end": start + page, "_sort": "id", "_order": "ASC"}
            if library is not None:
                query["library_id"] = library
            try:
                rows = self._native("/api/song?" + urllib.parse.urlencode(query), self._native_token)
            except ServerError:
                self._native_token = None  # an old token: a new login next time
                raise
            if not isinstance(rows, list):
                raise ServerError("/api/song: not a list")
            for row in rows:
                if row.get("id") and row.get("path"):
                    root, path = str(row.get("libraryPath") or ""), str(row["path"])
                    out[row["id"]] = path if path.startswith("/") or not root else root.rstrip("/") + "/" + path
            if len(rows) < page:
                return out
            start += page

    def most_played(self, limit=20):
        """The ids of the songs this user has played most, the most played first (Navidrome's own API)."""
        if not self.configured:
            raise ServerError("Navidrome isn't set up")
        if not self._native_token:
            answer = self._native("/auth/login", body={"username": self.user, "password": self.password})
            self._native_token = (answer or {}).get("token")
            if not self._native_token:
                raise ServerError("/auth/login: no token")
        query = {"_start": 0, "_end": limit, "_sort": "play_count", "_order": "DESC"}
        try:
            rows = self._native("/api/song?" + urllib.parse.urlencode(query), self._native_token)
        except ServerError:
            self._native_token = None
            raise
        if not isinstance(rows, list):
            raise ServerError("/api/song: not a list")
        return [str(r["id"]) for r in rows if r.get("id") and (r.get("playCount") or 0) > 0]

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

    def audio(self, song_id, byte_range=None):
        """The song as MP3 for the page's player (every browser plays it), opened: the caller reads and closes it."""
        request = urllib.request.Request(self.address("stream", id=song_id, format="mp3", maxBitRate=320))
        if byte_range:
            request.add_header("Range", byte_range)
        return urllib.request.urlopen(request, timeout=60)

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
        "path": e.get("path") or None,
    }
