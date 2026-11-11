"""Stand-ins for Navidrome and the Siansa plugin on AudioMuse-AI, for the tests and the end-to-end check.

``FakeServer(folder)`` serves the songs listed in ``folder/catalog.json`` (made by make_test_songs.py) as a Navidrome
with two libraries, "Music" (every song, unless a song's ``library`` says "2") and "Compilations", and the plugin's
profiles from each song's ``plugin`` entry. It checks the login
(user ``test``, password ``test``; AudioMuse-AI token ``token``) the way the real ones do.

``FakeServer`` also answers Navidrome's own API (``/auth/login``, ``/api/song``) with where each file really is
(``/music/<artist>/<album>/<file>``), while its Subsonic ``path`` is made up from the tags, as Navidrome's is; a
second user, ``second``, sees the same songs. ``FakeMediaServer(folder)`` is an Emby (or, with ``jellyfin=True``, a
Jellyfin) with the same files under its own folder (``/mnt/media/music/…``) and its own ids (``e-`` or ``j-`` before
the catalogue's): each takes only its own way of logging in and its own addresses, as the real ones do.

Run on its own for the end-to-end check or to try the page: ``python3 tests/fakes.py <folder> [port]``; add
``--emby <port>`` and ``--jellyfin <port>`` for those two with the same songs.
"""

import hashlib
import json
import os
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

USER, PASSWORD, TOKEN = "test", "test", "token"
SECOND_PASSWORD = "second-pass-9431"
USERS = {USER: PASSWORD, "second": SECOND_PASSWORD}  # Navidrome's users: both see every song


def file_path(song):
    """Where a song's file is under the music folder, the same on every server: artist/album/file."""
    return f"{song['artist']}/{song.get('album', 'Test')}/{song['file']}"


PICTURE = b"\x89PNG\r\n\x1a\n"  # how a picture starts: the tests' pictures are this and the id they're of


def entry(song):
    e = {"id": song["id"], "title": song["title"], "artist": song["artist"], "album": song.get("album", "Test"),
         "duration": int(song.get("duration", 60)), "suffix": os.path.splitext(song["file"])[1].lstrip(".")}
    if song.get("bpm"):
        e["bpm"] = song["bpm"]
    if song.get("genre"):
        e["genre"] = song["genre"]
        e["genres"] = [{"name": g.strip()} for g in song["genre"].split("/")]
    e["coverArt"], e["artistId"] = "mf-" + song["id"], "artist-of-" + song["id"]
    # Made up from the tags, as Navidrome's Subsonic path is unless the player reports real paths.
    e["path"] = f"{song['artist']}/{e['album']}/01 - {song['title']}.{e['suffix']}"
    return e


class FakeServer:
    def __init__(self, folder, port=0):
        self.folder = folder
        with open(os.path.join(folder, "catalog.json")) as f:
            self.songs = json.load(f)
        self.requests = []
        self.search_all = True  # whether search3 finds every song for an empty query (else: album by album)
        self.native = True  # whether Navidrome's own API answers (else only Subsonic's made-up paths)
        # AudioMuse-AI's export for apps (/api/sync): {song id: its track, as AudioMuse-AI gives it}; None: it has none.
        self.sync = None
        # AudioMuse-AI's music servers (/api/servers), each with the song ids its user sees; None: it has no list.
        # [{"server_id", "name", "server_type", "is_default", "sees": {ids}}]
        self.am_servers = None
        self.played = {}  # song id -> plays, for Navidrome's own API's play counts
        # Pictures (0.30): the cover ids ("mf-<song>") and artist ids ("ar-artist-of-<song>") that have one; any other
        # is answered with the stand-in, as Navidrome does: an ordinary picture marked "don't keep", with no version.
        self.pictures = set()
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def reply(self, status, body, kind="application/json", headers=None):
                data = json.dumps(body).encode() if not isinstance(body, bytes) else body
                self.send_response(status)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(data)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(data)

            def subsonic(self, body=None, error=None):
                answer = {"status": "failed", "version": "1.16.1", "error": {"code": 40, "message": error}} if error \
                    else {"status": "ok", "version": "1.16.1", "type": "navidrome", "serverVersion": "0.0-fake",
                          **(body or {})}
                self.reply(200, {"subsonic-response": answer})

            def do_GET(self):
                url = urllib.parse.urlparse(self.path)
                q = {k: v[0] for k, v in urllib.parse.parse_qs(url.query, keep_blank_values=True).items()}
                fake.requests.append((url.path, q))
                if url.path.startswith("/plugins/siansa_analysis/"):
                    if len(self.requestline) > 4094:  # as AudioMuse-AI's web server (gunicorn's limit_request_line)
                        return self.reply(400, {"error": "Request Line is too large"})
                    return self.plugin(url.path, q)
                if url.path == "/api/sync":
                    if self.headers.get("Authorization") != f"Bearer {TOKEN}":
                        return self.reply(401, {"error": "login needed"})
                    if fake.sync is None:
                        return self.reply(404, {"error": "not found"})
                    if len(self.requestline) > 4094:
                        return self.reply(400, {"error": "Request Line is too large"})
                    tracks = [fake.sync[i] for i in sorted(fake.sync)]
                    if fake.am_servers is not None:  # only what the asked server's user sees; its default's without one
                        asked = [m for m in fake.am_servers if q.get("server") in (m["server_id"], m["name"])] \
                            if q.get("server") else [m for m in fake.am_servers if m["is_default"]]
                        if not asked:
                            return self.reply(400, {"error": "Invalid server selection."})
                        tracks = [t for t in tracks if t["id"] in asked[0]["sees"]]
                    if "ids" in q:
                        tracks = [t for t in tracks if t["id"] in q["ids"].split(",")]
                    total, limit, page = len(tracks), int(q.get("limit", 500)), int(q.get("page", 1))
                    tracks = tracks[(page - 1) * limit:page * limit]
                    if q.get("fields") == "index":
                        tracks = [{"id": t["id"], "fp": t["fp"]} for t in tracks]
                    more = page * limit < total and "ids" not in q
                    return self.reply(200, {"tracks": tracks, "total_tracks": total, "has_more": more,
                                            "next_page": page + 1 if more else None, "provider_type": "navidrome"})
                if url.path == "/api/servers":
                    if self.headers.get("Authorization") != f"Bearer {TOKEN}":
                        return self.reply(401, {"error": "login needed"})
                    if fake.am_servers is None:
                        return self.reply(404, {"error": "not found"})
                    return self.reply(200, {"servers": [{k: v for k, v in m.items() if k != "sees"} | {"music_libraries": ""}
                                                        for m in fake.am_servers], "multi_server_enabled": True})
                if url.path == "/api/song" and fake.native:
                    if self.headers.get("X-ND-Authorization") not in [f"Bearer nd-{u}" for u in USERS]:
                        return self.reply(401, {"error": "login needed"})
                    rows = [s for s in fake.songs if "library_id" not in q or str(s.get("library", "1")) == q["library_id"]]
                    if q.get("_sort") == "play_count":
                        rows = sorted(rows, key=lambda s: -fake.played.get(s["id"], 0))
                    rows = rows[int(q.get("_start", 0)):int(q.get("_end", len(rows)))]
                    return self.reply(200, [{"id": s["id"], "libraryId": int(s.get("library", "1")),
                                             "libraryPath": "/music", "path": file_path(s),
                                             "playCount": fake.played.get(s["id"], 0)} for s in rows])
                if not url.path.startswith("/rest/"):
                    return self.reply(404, {"error": "not found"})
                if q.get("t") != hashlib.md5((USERS.get(q.get("u"), "\0") + q.get("s", "")).encode()).hexdigest() \
                        or "p" in q:
                    return self.subsonic(error="Wrong username or password")
                method = url.path[len("/rest/"):].removesuffix(".view")
                by_id = {s["id"]: s for s in fake.songs}
                if method == "ping":
                    return self.subsonic()
                if method == "getMusicFolders":
                    return self.subsonic({"musicFolders": {"musicFolder": [{"id": 1, "name": "Music"},
                                                                           {"id": 2, "name": "Compilations"}]}})
                songs = [s for s in fake.songs if "musicFolderId" not in q
                         or str(s.get("library", "1")) == q["musicFolderId"]]
                if method == "search3":
                    found = songs if q.get("query") == "" and fake.search_all else []
                    start, count = int(q.get("songOffset", 0)), int(q.get("songCount", 20))
                    return self.subsonic({"searchResult3": {"song": [entry(s) for s in found[start:start + count]]}})
                # An album is in one library, as on Navidrome: its id says which.
                albums = sorted({(str(s.get("library", "1")), s.get("album", "Test")) for s in songs})
                if method == "getAlbumList2":
                    start, size = int(q.get("offset", 0)), int(q.get("size", 10))
                    return self.subsonic({"albumList2": {"album": [{"id": f"al-{lib}-{a}", "name": a}
                                                                   for lib, a in albums[start:start + size]]}})
                if method == "getAlbum":
                    lib, _, name = q.get("id", "").removeprefix("al-").partition("-")
                    return self.subsonic({"album": {"id": q.get("id"), "song": [
                        entry(s) for s in fake.songs
                        if s.get("album", "Test") == name and str(s.get("library", "1")) == lib]}})
                if method == "getSong":
                    if q.get("id") not in by_id:
                        return self.subsonic(error="Song not found")
                    return self.subsonic({"song": entry(by_id[q["id"]])})
                if method == "getCoverArt":
                    if q.get("id") in fake.pictures:
                        return self.reply(200, PICTURE + q["id"].encode(), "image/png",
                                          {"ETag": '"v1"', "Cache-Control": "public, max-age=315360000"})
                    return self.reply(200, PICTURE + b"stand-in", "image/webp", {"Cache-Control": "no-store"})
                if method == "stream":
                    song = by_id.get(q.get("id"))
                    if not song:
                        return self.reply(404, b"", "text/plain")
                    with open(os.path.join(fake.folder, song["file"]), "rb") as f:
                        data = f.read()
                    kind = "audio/flac" if song["file"].endswith(".flac") else "audio/wav"
                    rng = self.headers.get("Range")
                    if rng and rng.startswith("bytes="):
                        start = int(rng[6:].split("-")[0] or 0)
                        part = data[start:]
                        return self.reply(206, part, kind, {"Content-Range": f"bytes {start}-{len(data) - 1}/{len(data)}",
                                                            "Accept-Ranges": "bytes"})
                    return self.reply(200, data, kind, {"Accept-Ranges": "bytes"})
                return self.subsonic(error=f"Unknown method {method}")

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                fake.requests.append((self.path, {}))
                if self.path == "/auth/login" and fake.native:
                    if USERS.get(body.get("username")) != body.get("password"):
                        return self.reply(401, {"error": "Invalid username or password"})
                    return self.reply(200, {"token": f"nd-{body['username']}", "name": body["username"]})
                return self.reply(404, {"error": "not found"})

            def plugin(self, path, q):
                if self.headers.get("Authorization") != f"Bearer {TOKEN}":
                    return self.reply(401, {"error": "login needed"})
                if path.endswith("/api/health"):
                    return self.reply(200, {"plugin": "siansa_analysis", "plugin_version": "0.5.7-fake", "status": "ok"})
                if path.endswith("/api/profiles") and "/siansa_beats/" in path:
                    return self.reply(404, {"error": "not installed"})
                if path.endswith("/api/profiles"):
                    ids = [i for i in q.get("ids", "").split(",") if i]
                    profiles = [{"track_id": s["id"], **s["plugin"]} for s in fake.songs
                                if s["id"] in ids and s.get("plugin")]
                    return self.reply(200, {"schema_version": 1, "profiles": profiles,
                                            "pending": [i for i in ids if i not in {p["track_id"] for p in profiles}]})
                # Only the Siansa Analysis plugin is installed here: Siansa Beats asks the dev plugin first.
                if "/siansa_analysis/" in path and "/api/similar/" in path:
                    if q.get("track_id") == "nope":
                        return self.reply(404, {"error": "unknown song"})
                    kind = path.rsplit("/", 1)[1]
                    key = "album" if kind == "albums" else "artist"
                    return self.reply(200, {"similar": [{key: "Deep", "artist": "House Test", "track_id": "t122",
                                                         "score": 0.91, "asked": q}]})
                return self.reply(404, {"error": "not found"})

        self.server = ThreadingHTTPServer(("127.0.0.1" if port == 0 else "0.0.0.0", port), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def start(self):
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


class FakeMediaServer:
    """An Emby or Jellyfin with the catalogue's songs: ``only`` (ids) for some of them, ``extra`` for songs only it
    has (as the catalogue's entries), ``paths=False`` for one that doesn't say where its files are."""

    def __init__(self, folder, jellyfin=False, port=0, only=None, extra=(), paths=True, root="/mnt/media/music",
                 under_emby=False):
        self.folder, self.jellyfin, self.paths, self.root = folder, jellyfin, paths, root
        self.under_emby = under_emby  # an Emby set up to answer under /emby only (a reverse proxy's way)
        self.pictures = set()  # the item ids (an album's, an artist's) that have a primary picture; the rest: HTTP 404
        self.forbids = False  # whether it turns every request away (HTTP 403), as Emby does a caller it takes for remote
        self.prefix = "j-" if jellyfin else "e-"
        with open(os.path.join(folder, "catalog.json")) as f:
            self.songs = [s for s in json.load(f) if only is None or s["id"] in only] + list(extra)
        self.requests, self.token, self.logins = [], "tok-1", 0
        self.password = PASSWORD  # a test may give it one that's told apart from the user name
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def reply(self, status, body, kind="application/json", headers=None):
                data = json.dumps(body).encode() if not isinstance(body, bytes) else body
                self.send_response(status)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(data)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(data)

            def said(self):
                """The login header as this server takes it: Jellyfin's standard one, Emby's own."""
                return self.headers.get("Authorization" if fake.jellyfin else "X-Emby-Authorization") or ""

            def logged_in(self):
                if not self.said().startswith("MediaBrowser " if fake.jellyfin else "Emby "):
                    return False
                if fake.jellyfin:
                    return f'Token="{fake.token}"' in self.said()
                return self.headers.get("X-Emby-Token") == fake.token

            def unprefixed(self):
                """The path as Emby takes it: with or without /emby (only with it, ``under_emby``); None for neither."""
                if self.path.startswith("/emby/"):
                    self.path = self.path[5:]
                    return True
                return not fake.under_emby

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                if not self.unprefixed():
                    return self.reply(404, b"<html>Not found</html>", "text/html")
                fake.requests.append((self.path, {}))
                if self.path == "/Users/AuthenticateByName":
                    if not self.said().startswith("MediaBrowser " if fake.jellyfin else "Emby "):
                        return self.reply(400, {"error": "no client named"})
                    if body.get("Username") != USER or body.get("Pw") != fake.password:
                        return self.reply(401, b"", "text/plain")
                    fake.logins += 1
                    return self.reply(200, {"AccessToken": fake.token, "User": {"Id": "u1", "Name": USER}})
                return self.reply(404, {"error": "not found"})

            def item(self, song):
                e = {"Id": fake.prefix + song["id"], "Name": song["title"], "Album": song.get("album", "Test"),
                     "AlbumArtist": song["artist"], "Artists": [song["artist"]], "Type": "Audio",
                     "RunTimeTicks": int(song.get("duration", 60)) * 10_000_000 + 4_000_000,  # to the tick, not the second
                     "Container": os.path.splitext(song["file"])[1].lstrip("."),
                     "Genres": [g.strip() for g in (song.get("genre") or "").split("/") if g.strip()]}
                if fake.paths:
                    e["Path"] = f"{fake.root}/{file_path(song)}"
                e["AlbumId"], e["ArtistItems"] = "album-of-" + song["id"], [{"Id": "artist-of-" + song["id"], "Name": song["artist"]}]
                if e["AlbumId"] in fake.pictures:
                    e["AlbumPrimaryImageTag"] = "tag1"
                return e

            def do_GET(self):
                if fake.forbids:  # an Emby that lets in local addresses only, asked from elsewhere
                    return self.reply(403, b"Forbidden", "text/plain", {"Server": "UPnP/1.0 DLNADOC/1.50"})
                if not self.unprefixed():
                    return self.reply(404, b"<html>Not found</html>", "text/html")
                url = urllib.parse.urlparse(self.path)
                q = {k: v[0] for k, v in urllib.parse.parse_qs(url.query, keep_blank_values=True).items()}
                fake.requests.append((url.path, q))
                if url.path == "/System/Info/Public":
                    return self.reply(200, {"Version": "10.10.7" if fake.jellyfin else "4.8.10.0",
                                            "ProductName": "Jellyfin Server" if fake.jellyfin else "Emby Server"})
                if not self.logged_in():
                    return self.reply(401, b"", "text/plain")
                views = "/UserViews" if fake.jellyfin else "/Users/u1/Views"
                items = "/Items" if fake.jellyfin else "/Users/u1/Items"
                if fake.jellyfin and url.path in (views, items) and q.get("userId") != "u1":
                    return self.reply(400, {"error": "userId needed"})
                if url.path == views:
                    return self.reply(200, {"Items": [{"Id": "lib1", "Name": "Music", "CollectionType": "music"},
                                                      {"Id": "lib9", "Name": "Films", "CollectionType": "movies"}]})
                if url.path == items:
                    if q.get("IncludeItemTypes") != "Audio" or q.get("Recursive") != "true" or q.get("ParentId") != "lib1":
                        return self.reply(200, {"Items": [], "TotalRecordCount": 0})
                    start, count = int(q.get("StartIndex", 0)), int(q.get("Limit", 100))
                    return self.reply(200, {"Items": [self.item(s) for s in fake.songs[start:start + count]],
                                            "TotalRecordCount": len(fake.songs)})
                parts = url.path.strip("/").split("/")
                if len(parts) == 4 and parts[0] == "Items" and parts[2:] == ["Images", "Primary"]:
                    if parts[1] in fake.pictures:
                        return self.reply(200, PICTURE + parts[1].encode(), "image/jpeg")
                    return self.reply(404, b"", "text/plain")
                one = parts[-1] if (parts[:1] == ["Items"] and len(parts) == 2) or (parts[:3] == ["Users", "u1", "Items"] and len(parts) == 4) else None
                if one:
                    song = next((s for s in fake.songs if fake.prefix + s["id"] == one), None)
                    return self.reply(200, self.item(song)) if song else self.reply(404, b"", "text/plain")
                if len(parts) == 3 and parts[0] == "Audio" and parts[2] == "stream" and q.get("Static") == "true":
                    song = next((s for s in fake.songs if fake.prefix + s["id"] == parts[1]), None)
                    if not song:
                        return self.reply(404, b"", "text/plain")
                    with open(os.path.join(fake.folder, song["file"]), "rb") as f:
                        data = f.read()
                    kind = "audio/flac" if song["file"].endswith(".flac") else "audio/wav"
                    rng = self.headers.get("Range")
                    if rng and rng.startswith("bytes="):
                        start = int(rng[6:].split("-")[0] or 0)
                        return self.reply(206, data[start:], kind, {"Content-Range": f"bytes {start}-{len(data) - 1}/{len(data)}",
                                                                   "Accept-Ranges": "bytes"})
                    return self.reply(200, data, kind, {"Accept-Ranges": "bytes"})
                return self.reply(404, {"error": "not found"})

        self.server = ThreadingHTTPServer(("127.0.0.1" if port == 0 else "0.0.0.0", port), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def start(self):
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


if __name__ == "__main__":
    args = sys.argv[1:]
    others = {}
    for flag in ("--emby", "--jellyfin"):
        if flag in args:
            at = args.index(flag)
            others[flag] = int(args[at + 1])
            del args[at:at + 2]
    for flag, port in others.items():
        media = FakeMediaServer(args[0], jellyfin=flag == "--jellyfin", port=port).start()
        print(f"Fake {'Jellyfin' if media.jellyfin else 'Emby'} on port {port}, {len(media.songs)} songs", flush=True)
    server = FakeServer(args[0], int(args[1]) if len(args) > 1 else 4533)
    print(f"Fake Navidrome and AudioMuse-AI on port {server.server.server_address[1]}, {len(server.songs)} songs",
          flush=True)
    server.server.serve_forever()
