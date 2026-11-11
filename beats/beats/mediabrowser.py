"""Emby and Jellyfin, read only: their music libraries, the songs in them and their audio (0.23.0, the owner, 10 Nov).

Jellyfin forked from Emby and kept most of its API, so one class does both, as the Siansa app's own does:

- it logs in once with the user name and password (``Users/AuthenticateByName``) and keeps the access token; a
  request the server refuses (HTTP 401) logs in again once. The token goes in a header (Emby's
  ``X-Emby-Authorization`` and ``X-Emby-Token``, Jellyfin's ``Authorization: MediaBrowser …``), never in an address,
  so it's never in a log;
- the libraries are the user's views of music; a library's songs come a page at a time, each with its file's path
  (to find the same file on another server, :mod:`beats.match`);
- a song's audio is its untouched original file (``/Audio/<id>/stream?Static=true``).

Nothing on the server is changed.
"""

import json
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request

from . import __version__, log
from .subsonic import ServerError

_log = log.get("mediaserver")

TICKS_A_SECOND = 10_000_000
FIELDS = "Path,Genres,MediaSources"


class MediaBrowser:
    def __init__(self, url, user, password, jellyfin=False, timeout=30.0):
        self.url = (url or "").rstrip("/")
        self.user = user or ""
        self.password = password or ""
        self.jellyfin = jellyfin
        self.kind = "jellyfin" if jellyfin else "emby"
        self.label = "Jellyfin" if jellyfin else "Emby"
        self.timeout = timeout
        self._login = None  # (user id, token)
        # Called with (user id, token) after each sign-in, so the token is kept and the password needn't be (0.24).
        self.on_login = None

    def signed_in(self, user_id, token):
        """Carries on with a sign-in kept from before: no password is needed while the server takes its token."""
        self._login = (user_id, token) if user_id and token else None
        return self

    @property
    def configured(self):
        return bool(self.url and self.user)

    # ---- plumbing ----

    def _headers(self, token):
        who = (f'{"MediaBrowser" if self.jellyfin else "Emby"} Client="Siansa Beats", Device="Siansa Beats", '
               f'DeviceId="siansa-beats", Version="{__version__}"') + (f', Token="{token}"' if token else "")
        if self.jellyfin:
            return {"Authorization": who}
        return {"X-Emby-Authorization": who, **({"X-Emby-Token": token} if token else {})}

    def _open(self, path, token=None, body=None, headers=None, timeout=None):
        """One request, opened. Raises :class:`_Refused` for HTTP 401 with a token (an old login)."""
        request = urllib.request.Request(self.url + path, data=json.dumps(body).encode() if body is not None else None,
                                         headers={"Content-Type": "application/json", "Accept": "application/json",
                                                  **self._headers(token), **(headers or {})})
        try:
            return urllib.request.urlopen(request, timeout=timeout or self.timeout)
        except urllib.error.HTTPError as e:
            if e.code == 401 and token:
                raise _Refused() from None
            if e.code == 401:
                raise ServerError(f"{self.label} refused the user name or password") from None
            raise ServerError(f"{path.split('?')[0]}: HTTP {e.code}") from None
        except (urllib.error.URLError, OSError) as e:
            raise ServerError(f"{path.split('?')[0]}: {getattr(e, 'reason', e)}") from None

    def _json(self, path, token=None, body=None):
        started = time.monotonic()
        with self._open(path, token, body) as response:
            try:
                answer = json.loads(response.read().decode("utf-8-sig") or "null")
            except ValueError:
                raise ServerError(f"That address doesn't look like {self.label}") from None
        _log.debug("%s %s: ok in %.2fs", self.label, path.split("?")[0], time.monotonic() - started)
        return answer

    def _session(self, again=False):
        if not self.configured:
            raise ServerError(f"{self.label} isn't set up")
        if self._login is None or again:
            if not self.password:
                # Signed in with a kept token (the password isn't kept) that the server no longer takes.
                raise ServerError(f"{self.label} no longer takes the saved sign-in: enter the password again "
                                  "(Setup › Music servers › Edit)")
            answer = self._json("/Users/AuthenticateByName", body={"Username": self.user, "Pw": self.password}) or {}
            token, user_id = answer.get("AccessToken"), (answer.get("User") or {}).get("Id")
            if not token or not user_id:
                raise ServerError(f"{self.label} refused the login")
            self._login = (user_id, token)
            _log.info("Logged in to %s as %s", self.label, self.user)
            if self.on_login:
                self.on_login(user_id, token)
        return self._login

    def _as_user(self, call):
        """``call(user id, token)``, logging in again once when the kept login is refused."""
        try:
            return call(*self._session())
        except _Refused:
            _log.info("%s refused the kept login; logging in again", self.label)
            try:
                return call(*self._session(again=True))
            except _Refused:
                raise ServerError(f"{self.label} refused the login") from None

    def _items_path(self, user_id, **query):
        query = urllib.parse.urlencode({k: v for k, v in query.items() if v is not None})
        return f"/Items?userId={user_id}&{query}" if self.jellyfin else f"/Users/{user_id}/Items?{query}"

    # ---- what Siansa Beats asks ----

    def ping(self):
        info = self._json("/System/Info/Public") or {}
        if not info.get("Version"):
            raise ServerError(f"That address doesn't look like {self.label}")
        if self.password:
            self._session(again=True)  # the login is checked too
        else:  # a kept sign-in: checked by asking for something only a signed-in user gets
            self.libraries()
        return {"version": info.get("Version"), "type": (info.get("ProductName") or self.label)}

    def libraries(self):
        """The user's music libraries: ``[{id, name}]``."""
        def views(user_id, token):
            path = f"/UserViews?userId={user_id}" if self.jellyfin else f"/Users/{user_id}/Views"
            return (self._json(path, token) or {}).get("Items") or []

        return [{"id": str(v["Id"]), "name": v.get("Name") or "Music"} for v in self._as_user(views)
                if v.get("Id") and v.get("CollectionType") == "music"]

    def all_songs(self, library=None, page=500):
        """Every song in ``library`` (an id from :meth:`libraries`; ``None``: every music library), ``page`` at a time."""
        if library is None:
            return [s for f in self.libraries() for s in self.all_songs(f["id"], page)]
        songs, start = [], 0
        while True:
            def items(user_id, token):
                return (self._json(self._items_path(user_id, ParentId=library, IncludeItemTypes="Audio", Recursive="true",
                                                    Fields=FIELDS, SortBy="SortName", SortOrder="Ascending",
                                                    StartIndex=start, Limit=page), token) or {}).get("Items") or []

            found = self._as_user(items)
            songs += [song_from(e) for e in found if e.get("Id")]
            if len(found) < page:
                return songs
            start += page

    def song(self, song_id):
        def one(user_id, token):
            item = urllib.parse.quote(str(song_id), safe="")
            path = f"/Items/{item}?userId={user_id}" if self.jellyfin else f"/Users/{user_id}/Items/{item}"
            return self._json(path, token) or {}

        return song_from(self._as_user(one))

    def _stream(self, song_id, token, byte_range=None, timeout=None):
        item = urllib.parse.quote(str(song_id), safe="")
        return self._open(f"/Audio/{item}/stream?Static=true", token,
                          headers={"Accept": "*/*", **({"Range": byte_range} if byte_range else {})}, timeout=timeout)

    def audio(self, song_id, byte_range=None):
        """The song's own file for the page's player, opened: the caller reads and closes it."""
        try:
            return self._as_user(lambda _, token: self._stream(song_id, token, byte_range, 60))
        except ServerError as e:  # the page's player expects urllib's errors, as from Navidrome
            raise urllib.error.URLError(str(e)) from None

    def download(self, song_id, path):
        """The song's original file, saved at ``path``. Returns its size in bytes."""
        started = time.monotonic()

        def fetch(_, token):
            with self._stream(song_id, token) as r, open(path, "wb") as out:
                kind = r.headers.get("Content-Type", "")
                if "json" in kind or "html" in kind:
                    raise ServerError(f"stream: {self.label} sent {kind} instead of the file")
                shutil.copyfileobj(r, out, 1 << 20)
                return out.tell(), kind

        try:
            size, kind = self._as_user(fetch)
        except OSError as e:
            raise ServerError(f"stream: {getattr(e, 'reason', e)}") from None
        _log.debug("%s stream %s: %.1f MB (%s) in %.2fs", self.label, song_id, size / 1e6, kind or "no type",
                   time.monotonic() - started)
        return size


class _Refused(Exception):
    """The server no longer takes the kept login."""


def song_from(e):
    """The fields this service keeps, from an Emby or Jellyfin song."""
    sources = e.get("MediaSources") or []
    path = e.get("Path") or next((m.get("Path") for m in sources if m.get("Path")), None)
    container = e.get("Container") or next((m.get("Container") for m in sources if m.get("Container")), "") or ""
    artists = [a for a in (e.get("Artists") or []) if a] or [e.get("AlbumArtist") or ""]
    suffix = path.rsplit(".", 1)[1].lower() if path and "." in path.rsplit("/", 1)[-1] else container.split(",")[0]
    return {
        "id": e.get("Id"),
        "title": e.get("Name") or "",
        "artist": " • ".join(artists),
        "album": e.get("Album") or "",
        "duration": round((e.get("RunTimeTicks") or 0) / TICKS_A_SECOND, 3),
        "suffix": suffix,
        "tag_bpm": None,  # neither server reads the BPM tag
        "genres": [g for g in (e.get("Genres") or []) if g],
        "path": path,
    }
