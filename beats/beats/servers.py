"""The music servers Siansa Beats reads (0.23.0, the owner, 10 Nov): Navidrome, Emby and Jellyfin, several of them,
and several users of one (each sees the libraries that user may).

The first is the one in the container's settings (``NAVIDROME_URL``, ``NAVIDROME_USER``, ``NAVIDROME_PASSWORD``), its
id ``main``. The others are added on the page (Setup › Music servers) and kept in the data folder with their
passwords, which are never sent back, logged or put in a report.

One of them is the default: asked first when the libraries are read, and the one a request means when it names none
(``?server=``), so an app that knows nothing of this keeps working.

Every server's songs share one catalogue: a song two of them have is analysed once (:mod:`beats.match`), and each
keeps its own id for it (:meth:`beats.store.Store.link_songs`).
"""

import re
import threading
import urllib.parse

from .mediabrowser import MediaBrowser
from .subsonic import ServerError, Subsonic

MAIN = "main"
KINDS = {"navidrome": "Navidrome", "emby": "Emby", "jellyfin": "Jellyfin"}
MOST = 12  # servers and users: more is a mistake


def make_client(kind, url, user, password):
    if kind == "navidrome":
        return Subsonic(url, user, password)
    if kind in ("emby", "jellyfin"):
        return MediaBrowser(url, user, password, jellyfin=kind == "jellyfin")
    raise ValueError(f"kind is one of: {', '.join(KINDS)}")


def same_place(a, b):
    """Whether two addresses are the same server (``http://nas:4533/`` and ``http://NAS:4533``)."""
    def parts(url):
        u = urllib.parse.urlsplit(url if "//" in (url or "") else "//" + (url or ""))
        return (u.hostname or "").lower(), u.port or (443 if u.scheme == "https" else 80), u.path.rstrip("/")

    return parts(a) == parts(b)


class Servers:
    def __init__(self, store, config):
        self.store, self.config = store, config
        self._clients = {}
        self._lock = threading.RLock()

    # ---- which there are ----

    def _saved(self):
        return [s for s in self.store.setting("servers") or [] if isinstance(s, dict) and s.get("id")]

    def all(self):
        """Every server, the container's own first: ``[{"id", "kind", "name", "url", "user", "password", "env"}]``."""
        c = self.config
        out = []
        if c.navidrome_url:
            out.append({"id": MAIN, "kind": "navidrome", "name": self.store.setting("main_name") or "Navidrome",
                        "url": c.navidrome_url, "user": c.navidrome_user, "password": c.navidrome_password, "env": True})
        return out + [{**s, "env": False} for s in self._saved()]

    def get(self, server_id):
        return next((s for s in self.all() if s["id"] == server_id), None)

    @property
    def default_id(self):
        ids = [s["id"] for s in self.all()]
        chosen = self.store.setting("default_server")
        return chosen if chosen in ids else (ids[0] if ids else MAIN)

    def find(self, asked):
        """The id of the server a request names (its id or its name, any case), the default's for none; None when
        there's no such server."""
        if asked is None or not str(asked).strip():
            return self.default_id
        asked = str(asked).strip()
        servers = self.all()
        for s in servers:
            if s["id"] == asked:
                return s["id"]
        named = [s["id"] for s in servers if s["name"].casefold() == asked.casefold()]
        return named[0] if len(named) == 1 else None

    def several(self):
        return len(self.all()) > 1

    def public(self, server):
        """A server as the page and the app see it: never its password."""
        return {"id": server["id"], "kind": server["kind"], "name": server["name"], "url": server["url"],
                "user": server["user"], "default": server["id"] == self.default_id, "env": bool(server.get("env"))}

    def client(self, server_id=None):
        """The server's client (kept: it holds its login). A stand-in that says so for one that isn't there."""
        server_id = server_id or self.default_id
        with self._lock:
            server = self.get(server_id)
            if server is None:
                return Subsonic("", "", "")
            key = (server["kind"], server["url"], server["user"], server["password"])
            kept = self._clients.get(server_id)
            if kept is None or kept[0] != key:
                kept = (key, make_client(*key))
                self._clients[server_id] = kept
            return kept[1]

    def twins(self, server_id):
        """The other users of the same server (same kind and address): its songs have the same ids for all of them."""
        me = self.get(server_id)
        if me is None:
            return []
        return [s["id"] for s in self.all() if s["id"] != server_id and s["kind"] == me["kind"]
                and same_place(s["url"], me["url"])]

    # ---- changing them (from the page) ----

    def _check(self, kind, name, url, user, password, but=None):
        kind = str(kind or "").strip().lower()
        if kind not in KINDS:
            raise ValueError(f"kind is one of: {', '.join(KINDS)}")
        url, user = str(url or "").strip().rstrip("/"), str(user or "").strip()
        if not re.match(r"^https?://[^\s/]+", url):
            raise ValueError("The address starts with http:// or https://")
        if not user or not password:
            raise ValueError("A user name and its password are needed")
        name = " ".join(str(name or "").split())[:60] or f"{KINDS[kind]} ({user})"
        for s in self.all():
            if s["id"] == but:
                continue
            if s["kind"] == kind and same_place(s["url"], url) and s["user"].casefold() == user.casefold():
                raise ValueError(f"{s['user']} of that {KINDS[kind]} is already here, as {s['name']}")
        for s in self.all():
            if s["id"] != but and s["name"].casefold() == name.casefold():
                raise ValueError(f"There's already a server named {s['name']}")
        return kind, name, url, user

    def test(self, kind, url, user, password):
        """Asks the server and logs in: ``{"version", "type", "libraries"}``. :class:`ServerError` says why not."""
        client = make_client(kind, url, user, password)
        info = client.ping()
        return {**info, "libraries": client.libraries()}

    def add(self, kind, name, url, user, password):
        """Adds a server (or another user of one) once it answers and takes the login: its public form."""
        with self._lock:
            kind, name, url, user = self._check(kind, name, url, user, password)
            saved = self._saved()
            if len(saved) + 1 >= MOST:
                raise ValueError(f"At most {MOST} servers")
            self.test(kind, url, user, password)
            taken = {s["id"] for s in self.all()}
            server_id = next(i for i in (kind if n == 1 else f"{kind}-{n}" for n in range(1, MOST * 2)) if i not in taken)
            server = {"id": server_id, "kind": kind, "name": name, "url": url, "user": user, "password": password}
            self.store.set_setting("servers", saved + [server])
            return self.public({**server, "env": False})

    def update(self, server_id, name=None, url=None, user=None, password=None):
        """Changes a server added on the page (its name, address or login; an empty password keeps the one it has).
        The container's own takes a new name only: the rest is its settings."""
        with self._lock:
            server = self.get(server_id)
            if server is None:
                raise KeyError(server_id)
            if server["env"]:
                if any(v for v in (url, user, password)):
                    raise ValueError("This server is set on the container (NAVIDROME_URL, NAVIDROME_USER, "
                                     "NAVIDROME_PASSWORD): change it there")
                new = " ".join(str(name or "").split())[:60]
                if new and any(s["name"].casefold() == new.casefold() for s in self.all() if s["id"] != server_id):
                    raise ValueError(f"There's already a server named {new}")
                if new:
                    self.store.set_setting("main_name", new)
                return self.public(self.get(server_id))
            password = password or server["password"]
            kind, name, url, user = self._check(server["kind"], name or server["name"], url or server["url"],
                                                user or server["user"], password, but=server_id)
            self.test(kind, url, user, password)
            changed = {"id": server_id, "kind": kind, "name": name, "url": url, "user": user, "password": password}
            self.store.set_setting("servers", [changed if s["id"] == server_id else s for s in self._saved()])
            self._clients.pop(server_id, None)
            return self.public({**changed, "env": False})

    def remove(self, server_id):
        """Takes a server added on the page away, with its songs' ids there; songs only it had go with it. The number
        of songs forgotten."""
        with self._lock:
            server = self.get(server_id)
            if server is None:
                raise KeyError(server_id)
            if server["env"]:
                raise ValueError("This server is set on the container: take NAVIDROME_URL away there")
            self.store.set_setting("servers", [s for s in self._saved() if s["id"] != server_id])
            self._clients.pop(server_id, None)
            if self.store.setting("default_server") == server_id:
                self.store.set_setting("default_server", None)
            self.store.set_setting(folders_key(server_id), None)
            return self.store.remove_server(server_id)

    def set_default(self, server_id):
        if self.get(server_id) is None:
            raise KeyError(server_id)
        self.store.set_setting("default_server", server_id)


def folders_key(server_id):
    """Where a server's chosen libraries are kept: ``folders`` for the container's own, as before 0.23."""
    return "folders" if server_id == MAIN else f"folders:{server_id}"


__all__ = ["KINDS", "MAIN", "ServerError", "Servers", "folders_key", "make_client", "same_place"]
