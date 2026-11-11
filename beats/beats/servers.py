"""The music servers Siansa Beats reads (0.23.0, the owner, 10 Nov): Navidrome, Emby and Jellyfin, several of them,
and several users of one (each sees the libraries that user may).

All are added on the page (Setup › Music servers) and kept in the data folder. No password is kept as plain text
(0.24.0, :mod:`beats.vault`): Navidrome's is encrypted (it has to be said to Navidrome each time); Emby's and
Jellyfin's aren't kept at all, only the encrypted sign-in token the server gave. None is ever sent back, logged or put
in a report.

Until 0.24 the first server was the container's settings (``NAVIDROME_URL``, ``NAVIDROME_USER``,
``NAVIDROME_PASSWORD``). They're taken over once, as the server ``main`` (:meth:`Servers.take_from_container`), and
not read again: they can go from the YAML.

One of them is the default: asked first when the libraries are read, and the one a request means when it names none
(``?server=``), so an app that knows nothing of this keeps working.

Every server's songs share one catalogue: a song two of them have is analysed once (:mod:`beats.match`), and each
keeps its own id for it (:meth:`beats.store.Store.link_songs`).
"""

import json
import re
import threading
import urllib.parse

from .mediabrowser import MediaBrowser
from .subsonic import ServerError, Subsonic

MAIN = "main"
KINDS = {"navidrome": "Navidrome", "emby": "Emby", "jellyfin": "Jellyfin"}
MOST = 12  # servers and users: more is a mistake


def make_client(kind, url, user, password, login=None):
    """A server's client. ``login``: Emby's or Jellyfin's kept sign-in, ``(user id, token)``, in place of a password."""
    if kind == "navidrome":
        return Subsonic(url, user, password)
    if kind in ("emby", "jellyfin"):
        return MediaBrowser(url, user, password, jellyfin=kind == "jellyfin").signed_in(*(login or (None, None)))
    raise ValueError(f"kind is one of: {', '.join(KINDS)}")


def same_place(a, b):
    """Whether two addresses are the same server (``http://nas:4533/`` and ``http://NAS:4533``)."""
    def parts(url):
        u = urllib.parse.urlsplit(url if "//" in (url or "") else "//" + (url or ""))
        return (u.hostname or "").lower(), u.port or (443 if u.scheme == "https" else 80), u.path.rstrip("/")

    return parts(a) == parts(b)


class Servers:
    def __init__(self, store, config, vault):
        self.store, self.config, self.vault = store, config, vault
        self._clients = {}
        self._lock = threading.RLock()
        self.take_from_container()

    # ---- which there are ----

    def _saved(self):
        return [s for s in self.store.setting("servers") or [] if isinstance(s, dict) and s.get("id")]

    def take_from_container(self):
        """Once: the container's ``NAVIDROME_…`` settings become the server ``main``, its password encrypted, and a
        password kept as plain text by 0.23 is encrypted. True when the container's settings were taken."""
        saved, changed = self._saved(), False
        for s in saved:  # 0.23 kept the page's servers' passwords as they were typed
            if "password" in s:
                s["secret"] = self.vault.seal(s.pop("password"))
                changed = True
        c, taken = self.config, False
        if c.navidrome_url and not self.store.setting("container_taken") and not any(s["id"] == MAIN for s in saved):
            saved.insert(0, {"id": MAIN, "kind": "navidrome", "name": self.store.setting("main_name") or "Navidrome",
                             "url": c.navidrome_url.rstrip("/"), "user": c.navidrome_user,
                             "secret": self.vault.seal(c.navidrome_password)})
            changed = taken = True
        if changed:
            self.store.set_setting("servers", saved)
        if c.navidrome_url:
            self.store.set_setting("container_taken", True)
        return taken

    def all(self):
        """Every server: ``[{"id", "kind", "name", "url", "user", "password", "login"}]``, the password and the
        sign-in (Emby's and Jellyfin's ``(user id, token)``) as they're needed, never to be shown."""
        out = []
        for s in self._saved():
            login = None
            if s.get("login"):
                try:
                    kept = json.loads(self.vault.open(s["login"]) or "null") or {}
                    login = (kept.get("user_id"), kept.get("token")) if kept.get("token") else None
                except ValueError:
                    login = None
            out.append({"id": s["id"], "kind": s["kind"], "name": s["name"], "url": s["url"], "user": s["user"],
                        "password": self.vault.open(s.get("secret")), "login": login})
        return out

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
                "user": server["user"], "default": server["id"] == self.default_id,
                # Whether it can still sign in: False when its secret was encrypted with a key that's gone.
                "signed_in": bool(server.get("password") or server.get("login"))}

    def client(self, server_id=None):
        """The server's client (kept: it holds its login). A stand-in that says so for one that isn't there."""
        server_id = server_id or self.default_id
        with self._lock:
            server = self.get(server_id)
            if server is None:
                return Subsonic("", "", "")
            key = (server["kind"], server["url"], server["user"], server["password"], server["login"])
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
        return self._signed_in(kind, url, user, password)[0]

    def _signed_in(self, kind, url, user, password):
        """Signs in to check a server: ``(what it is and has, what to keep for it)``. What's kept is never the password
        as it was typed: Navidrome's encrypted, or the encrypted token Emby or Jellyfin gave for it."""
        client = make_client(kind, url, user, password)
        got = []
        if kind != "navidrome":
            client.on_login = lambda user_id, token: got.append({"user_id": user_id, "token": token})
        info = client.ping()
        info = {**info, "libraries": client.libraries()}
        if kind == "navidrome":
            return info, {"secret": self.vault.seal(password)}
        if not got:
            raise ServerError(f"{KINDS[kind]} refused the login")
        return info, {"login": self.vault.seal(json.dumps(got[-1]))}

    def add(self, kind, name, url, user, password):
        """Adds a server (or another user of one) once it answers and takes the login: its public form."""
        with self._lock:
            kind, name, url, user = self._check(kind, name, url, user, password)
            saved = self._saved()
            if len(saved) >= MOST:
                raise ValueError(f"At most {MOST} servers")
            _, keep = self._signed_in(kind, url, user, password)
            taken = {s["id"] for s in saved}
            server_id = next(i for i in (kind if n == 1 else f"{kind}-{n}" for n in range(1, MOST * 2)) if i not in taken)
            self.store.set_setting("servers", saved + [{"id": server_id, "kind": kind, "name": name, "url": url,
                                                        "user": user, **keep}])
            return self.public(self.get(server_id))

    def update(self, server_id, name=None, url=None, user=None, password=None):
        """Changes a server: its name alone, or its address or login too. A new address or user needs the password
        again (only Navidrome's is kept, and it isn't said to another address unasked); the password alone signs in
        again, for a server that stopped taking the kept sign-in."""
        with self._lock:
            server = self.get(server_id)
            if server is None:
                raise KeyError(server_id)
            url = str(url or "").strip().rstrip("/") or server["url"]
            user = str(user or "").strip() or server["user"]
            moved = url != server["url"] or user != server["user"]
            if moved and not password:
                raise ValueError("Its password is needed to change its address or user")
            kind, name, url, user = self._check(server["kind"], name or server["name"], url, user,
                                                password or "kept", but=server_id)
            keep = {}
            if password:
                _, keep = self._signed_in(kind, url, user, password)
            self.store.set_setting("servers", [
                {**{k: v for k, v in s.items() if k not in (("secret", "login") if keep else ())},
                 "name": name, "url": url, "user": user, **keep} if s["id"] == server_id else s
                for s in self._saved()])
            self._clients.pop(server_id, None)
            return self.public(self.get(server_id))

    def remove(self, server_id):
        """Takes a server away, with its songs' ids there; songs only it had go with it. The number of songs
        forgotten."""
        with self._lock:
            if self.get(server_id) is None:
                raise KeyError(server_id)
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
