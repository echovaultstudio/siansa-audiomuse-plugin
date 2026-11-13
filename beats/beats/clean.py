"""Tidying the database (0.40, the owner, 13 Nov): what's here that needn't be, looked for first and shown, then
removed only as shown.

``preview(server)`` asks the server for every ticked library right now and finds, as groups: songs the server no
longer has, songs of libraries that are left out, songs linked to no server, and songs that couldn't be analysed.
Each song is named. The preview comes with a token good for a few minutes; ``apply(server, token, groups)`` removes
exactly the songs that preview showed in the chosen groups (the token is bound to that list, so a stale page, or a
request that never saw the preview, removes nothing), after a copy of the database is kept beside it
(``beats-before-clean.db``), and checks the database over afterwards.
"""
import logging
import os
import secrets
import shutil
import threading
import time

from .subsonic import ServerError

_log = logging.getLogger("beats.clean")

OFFER_SECONDS = 15 * 60  # a preview is good for this long
BACKUP = "beats-before-clean.db"
NAMED = 200  # songs named in each group of the preview; the rest are counted
FEWER_WARNING = 0.5  # a library listing under half the songs it had is more likely a server mid-scan than a clear-out


class Cleaner:
    def __init__(self, app):
        self.app = app
        self._offers = {}
        self._lock = threading.Lock()

    # ---- looking ----

    def preview(self, server=None):
        app = self.app
        server = server or app.servers.default_id
        client = app.servers.client(server)
        name = (app.servers.get(server) or {}).get("name", server)
        chosen = set(app.worker.chosen(server))
        groups = []
        try:
            libraries = {f["id"]: f["name"] for f in client.libraries()}
        except ServerError as e:
            raise ServerError(f"{name} doesn't answer, so nothing can be told gone: {e}") from None
        counts = app.store.folder_counts(server)
        # Gone from the server: each ticked library listed right now.
        for folder in sorted(chosen, key=lambda f: libraries.get(f, f)):
            try:
                now = [s["id"] for s in client.all_songs(folder) if s.get("id")]
            except ServerError as e:
                groups.append(self._group(f"gone:{folder}", f"Gone from {libraries.get(folder, folder)}", [],
                                          warning=f"{name} couldn't list this library ({e}): nothing of it is removed"))
                continue
            gone = app.store.gone_from(server, folder, now)
            had = counts.get(folder, {}).get("songs", 0)
            warning = None
            if gone and len(now) < FEWER_WARNING * had:
                warning = (f"The server lists {len(now):,} songs here where it had {had:,}: it may still be scanning, or a "
                           "disk may not be mounted. Remove these only if they're really gone.")
            groups.append(self._group(f"gone:{folder}", f"Gone from {libraries.get(folder, folder)}",
                                      self._named_items(server, gone), warning=warning,
                                      detail="On the server no more: their beats, loudness, sound, styles and marks go too, "
                                             "unless another server still has the song."))
        # Libraries that are left out, but still have songs here.
        for folder, count in sorted(counts.items(), key=lambda p: libraries.get(p[0], p[0])):
            if folder in chosen or not count.get("songs"):
                continue
            items = [r["item_id"] for r in app.store._rows("SELECT item_id FROM links WHERE server=? AND folder=?", (server, folder))]
            groups.append(self._group(f"left-out:{folder}", f"Left out: {libraries.get(folder, f'library {folder}')}",
                                      self._named_items(server, items),
                                      detail="The library isn't ticked any more. Its songs go with what was found in them "
                                             "(a song another server or library has stays)."))
        # Songs linked to no server at all (from before 0.23, or whose links went).
        unlinked = app.store.unlinked_songs()
        if unlinked:
            groups.append(self._group("unlinked", "Linked to no server", self._named_songs(unlinked),
                                      detail="Nothing says which server or library these are from, so they're never "
                                             "read again and never updated."))
        # Songs that couldn't be analysed.
        failed = [r["id"] for r in app.store.failed()]
        if failed:
            groups.append(self._group("failed", "Couldn't be analysed", self._named_songs(failed),
                                      detail="Removed here, they're found again on the next read and tried once more."))
        token = secrets.token_urlsafe(24)
        with self._lock:
            now = time.time()
            self._offers = {t: o for t, o in self._offers.items() if o["until"] > now}
            self._offers[token] = {"until": now + OFFER_SECONDS, "server": server,
                                   "groups": {g["id"]: g["_songs"] for g in groups}}
        shown = [{k: v for k, v in g.items() if k != "_songs"} for g in groups]
        total = sum(g["count"] for g in shown)
        return {"token": token, "good_for_seconds": OFFER_SECONDS, "server": server, "server_name": name,
                "groups": shown, "total": total}

    def _group(self, group_id, title, songs, detail=None, warning=None):
        """``songs``: ``[(what to remove, "Artist – Title")]``."""
        return {"id": group_id, "title": title, "count": len(songs), "names": [n for _, n in songs[:NAMED]],
                "detail": detail, "warning": warning, "_songs": [key for key, _ in songs]}

    def _named_items(self, server, item_ids):
        rows = {}
        store = self.app.store
        for start in range(0, len(item_ids), 400):
            chunk = item_ids[start:start + 400]
            for r in store._rows("SELECT l.item_id, s.artist, s.title FROM links l LEFT JOIN songs s ON s.id=l.song_id "
                                 f"WHERE l.server=? AND l.item_id IN ({', '.join('?' for _ in chunk)})", (server, *chunk)):
                rows[r["item_id"]] = f"{r['artist'] or '?'} – {r['title'] or r['item_id']}"
        return sorted(((i, rows.get(i, i)) for i in item_ids), key=lambda p: p[1].lower())

    def _named_songs(self, song_ids):
        rows = {}
        store = self.app.store
        for start in range(0, len(song_ids), 400):
            chunk = song_ids[start:start + 400]
            for r in store._rows(f"SELECT id, artist, title FROM songs WHERE id IN ({', '.join('?' for _ in chunk)})", chunk):
                rows[r["id"]] = f"{r['artist'] or '?'} – {r['title'] or r['id']}"
        return sorted(((i, rows.get(i, i)) for i in song_ids), key=lambda p: p[1].lower())

    # ---- removing ----

    def apply(self, server, token, groups):
        """Removes the songs of ``groups`` as the preview with ``token`` showed them. ``{"removed", "groups", "backup",
        "integrity"}``; ``ValueError`` for a token that's unknown, used or too old, or a group the preview didn't have."""
        with self._lock:
            offer = self._offers.pop(str(token or ""), None)
        if offer is None or offer["until"] < time.time():
            raise ValueError("That look at the database is too old (or was used): look again, then remove")
        if offer["server"] != server:
            raise ValueError("That look was at another server's songs")
        groups = [str(g) for g in (groups or [])]
        unknown = [g for g in groups if g not in offer["groups"]]
        if unknown:
            raise ValueError(f"The look didn't show {', '.join(unknown)}")
        app = self.app
        backup = self._backup()
        removed, by_group = 0, {}
        for group in groups:
            keys = offer["groups"][group]
            if not keys:
                continue
            if group.startswith("gone:") or group.startswith("left-out:"):
                n = app.store.remove_items(server, keys)
                if group.startswith("left-out:"):
                    app.store.remove_folder(group.split(":", 1)[1], server)  # its links too, every one
            else:
                n = 0
                for song_id in keys:
                    if app.store.song(song_id) is not None:
                        app.store.remove(song_id)
                        n += 1
            by_group[group] = n
            removed += n
        integrity = app.store.integrity()
        app.store.event(f"Tidied the database: {removed} songs removed ("
                        + ", ".join(f"{g}: {n}" for g, n in by_group.items()) + ")"
                        + (f"; a copy from before is {BACKUP}" if backup else "")
                        + f"; SQLite quick_check {integrity['quick_check']}", "warn" if integrity["quick_check"] != "ok" else "info")
        app.worker.wake()
        return {"removed": removed, "groups": by_group, "backup": backup, "integrity": integrity}

    def _backup(self):
        """A copy of the database from before, beside it (the previous copy is replaced). Its name, or None when there
        isn't room (it's said in the activity, and the removal goes ahead)."""
        store = self.app.store
        folder = os.path.dirname(store.path)
        target = os.path.join(folder, BACKUP)
        try:
            size = sum(os.path.getsize(store.path + x) for x in ("", "-wal") if os.path.exists(store.path + x))
            if shutil.disk_usage(folder).free < size * 1.5 + 200e6:
                store.event(f"No copy of the database before tidying: not enough room for {size / 1e6:,.0f} MB", "warn")
                return None
            with store.lock:
                if os.path.exists(target):
                    os.remove(target)
                store.db.execute("VACUUM INTO ?", (target,))
            return BACKUP
        except Exception as e:  # the copy is a courtesy; the removal is what was asked for
            _log.warning("Couldn't copy the database before tidying: %s", e)
            store.event(f"No copy of the database before tidying: {e}", "warn")
            return None
