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
    def __init__(self, url, user, password, jellyfin=False, timeout=30.0, empty_password=False):
        self.url = (url or "").rstrip("/")
        self.user = user or ""
        self.password = password or ""
        # A user without a password (common on a home Emby): signing in again needs none (0.24.1).
        self.empty_password = empty_password
        self.jellyfin = jellyfin
        self.kind = "jellyfin" if jellyfin else "emby"
        self.label = "Jellyfin" if jellyfin else "Emby"
        self.timeout = timeout
        # The device this sign-in is as. Jellyfin keeps one sign-in a device and user: a new one under the same name
        # ends the one before, so a check of a server (its Test button) signs in as another device than the kept
        # sign-in, which would stop working otherwise (0.25, found by the real-server checks).
        self.device = "siansa-beats"
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
               f'DeviceId="{self.device}", Version="{__version__}"') + (f', Token="{token}"' if token else "")
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
            sent = ""
            try:
                sent = f" ({what_it_sent(e, e.read(2048))})"
            except OSError:
                pass
            raise ServerError(f"{path.split('?')[0]}: HTTP {e.code}{sent}") from None
        except (urllib.error.URLError, OSError) as e:
            raise ServerError(why_not(self.label, getattr(e, "reason", e))) from None

    def _json(self, path, token=None, body=None):
        started = time.monotonic()
        with self._open(path, token, body) as response:
            raw = response.read()
            try:
                answer = json.loads(raw.decode("utf-8-sig") or "null")
            except ValueError:
                # What did answer, so the wrong port can be told from a proxy's page (0.24.2).
                raise ServerError(f"That address doesn't look like {self.label}: it sent {what_it_sent(response, raw)}") from None
        _log.debug("%s %s: ok in %.2fs", self.label, path.split("?")[0], time.monotonic() - started)
        return answer

    def _session(self, again=False):
        if not self.configured:
            raise ServerError(f"{self.label} isn't set up")
        if self._login is None or again:
            if not self.password and not self.empty_password:
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
        """What the server is: ``{"version", "type"}``, once it answers and takes the sign-in. Emby set up behind
        ``/emby`` answers there only: that's tried too, and :attr:`url` is then the one that answered."""
        info, problem, tried = None, None, []
        for base in dict.fromkeys([self.url, self.url + "/emby"] if not self.url.endswith("/emby") else [self.url]):
            was, self.url = self.url, base
            try:
                info = self._json("/System/Info/Public") or {}
                if not (isinstance(info, dict) and info.get("Version")):
                    tried.append(f"{base}/System/Info/Public answered, but not with a server's version")
            except ServerError as e:
                problem, info = problem or e, None
                tried.append(f"{base}{e}" if str(e).startswith("/") else f"{base}: {e}")
            if isinstance(info, dict) and info.get("Version"):
                break
            self.url = was
        if not isinstance(info, dict) or not info.get("Version"):
            if problem and not str(problem).startswith(("/", "That address")):
                raise problem  # nothing answers there at all: said as it is
            if any(": HTTP 403" in t for t in tried):
                # The server itself turns this container away before any sign-in (the owner's Emby, 11 Nov: the app
                # on the same network got in, Siansa Beats got "Forbidden" for a page that needs no login).
                raise ServerError(
                    f"{self.label} answers at {self.url} but refuses Siansa Beats (HTTP 403, before any sign-in). "
                    f"It lets in only addresses it counts as local: in {self.label}'s settings under Network, add "
                    "this container's network to the LAN networks (Docker's is usually 172.16.0.0/12) or allow "
                    f"remote connections; or use the address {self.label} is reached at through your proxy. "
                    f"What answered: {'; '.join(tried)}")
            # Something answers, but not as this server: what it said is given, so the cause can be seen (0.24.2: an
            # Emby reached through its proxy's address but not by IP and port, with nothing to say why).
            raise ServerError(f"No {self.label} answers at {self.url}: is it {self.label}'s own address and port, "
                              f"like http://192.168.1.10:8096? What answered: {'; '.join(tried)}")
        product = str(info.get("ProductName") or "")
        other = "Emby" if self.jellyfin else "Jellyfin"
        if other.lower() in product.lower():
            raise ServerError(f"That's {other} ({product} {info.get('Version')}): choose {other} as its kind")
        if self.password or self.empty_password:
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

    def most_played(self, limit=20):
        """The ids of the songs this user has played most, the most played first."""
        def items(user_id, token):
            return (self._json(self._items_path(user_id, IncludeItemTypes="Audio", Recursive="true", Filters="IsPlayed",
                                                SortBy="PlayCount", SortOrder="Descending", Limit=limit), token)
                    or {}).get("Items") or []

        return [str(e["Id"]) for e in self._as_user(items) if e.get("Id")]

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


def what_it_sent(response, raw):
    """What a server that isn't the one asked for sent, in a few words: its kind, who it says it is, how it starts."""
    headers = response.headers
    kind = (headers.get("Content-Type") or "no content type").split(";")[0]
    who = headers.get("Server")
    start = " ".join((raw or b"")[:300].decode("utf-8", "replace").split())[:80]
    return kind + (f" from {who}" if who else "") + (f', starting "{start}"' if start else ", empty")


def why_not(label, reason):
    """Why a server couldn't be reached, in plain words."""
    text = str(reason)
    if "CERTIFICATE_VERIFY_FAILED" in text or "SSL" in text:
        return (f"{label}'s certificate isn't trusted ({text}): use its local http:// address and port, or a "
                "certificate the container trusts")
    if "refused" in text.lower():
        return f"Nothing answers at that address and port ({text}): is {label} running there, and is the port right?"
    if "timed out" in text.lower() or "timeout" in text.lower():
        return f"{label} didn't answer in time ({text}): can the container reach that address?"
    if "Name or service not known" in text or "nodename" in text or "resolve" in text.lower():
        return f"That name isn't known to the container ({text}): try the server's IP address"
    return f"Couldn't reach {label}: {text}"


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
