"""Navidrome (or any OpenSubsonic server), read only: playlists, songs and their audio, as the Siansa app asks.

Logs in with a salted token (``t``/``s``), never sending the password itself. Nothing on the server is changed.
"""

import hashlib
import json
import secrets
import shutil
import urllib.error
import urllib.parse
import urllib.request

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
        try:
            with urllib.request.urlopen(self.address(method, **extra), timeout=self.timeout) as response:
                body = json.load(response)
        except urllib.error.HTTPError as e:
            raise SubsonicError(f"{method}: HTTP {e.code}") from None
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise SubsonicError(f"{method}: {getattr(e, 'reason', e)}") from None
        answer = body.get("subsonic-response") or {}
        if answer.get("status") != "ok":
            error = answer.get("error") or {}
            raise SubsonicError(f"{method}: {error.get('message') or 'the server said no'}")
        return answer

    def ping(self):
        answer = self._get("ping")
        return {"version": answer.get("serverVersion") or answer.get("version"), "type": answer.get("type")}

    def playlists(self):
        """``[{id, name, songs}]``."""
        lists = (self._get("getPlaylists").get("playlists") or {}).get("playlist") or []
        return [{"id": p["id"], "name": p.get("name", ""), "songs": p.get("songCount", 0)} for p in lists]

    def playlist_named(self, name):
        """The playlist called ``name`` (case doesn't matter), or ``None``."""
        for p in self.playlists():
            if p["name"].strip().lower() == (name or "").strip().lower():
                return p
        return None

    def playlist_songs(self, playlist_id):
        entries = (self._get("getPlaylist", id=playlist_id).get("playlist") or {}).get("entry") or []
        return [song_from(e) for e in entries]

    def song(self, song_id):
        return song_from(self._get("getSong", id=song_id).get("song") or {})

    def download(self, song_id, path):
        """The song's original file, saved at ``path``. Returns its size in bytes."""
        try:
            with urllib.request.urlopen(self.address("stream", id=song_id, format="raw"), timeout=self.timeout) as r, \
                    open(path, "wb") as out:
                kind = r.headers.get("Content-Type", "")
                if "json" in kind or "xml" in kind:
                    raise SubsonicError(f"stream: {r.read(300).decode(errors='replace')}")
                shutil.copyfileobj(r, out, 1 << 20)
                return out.tell()
        except urllib.error.HTTPError as e:
            raise SubsonicError(f"stream: HTTP {e.code}") from None
        except (urllib.error.URLError, OSError) as e:
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
