"""Stand-ins for Navidrome and the Siansa plugin on AudioMuse-AI, for the tests and the end-to-end check.

``FakeServer(folder)`` serves the songs listed in ``folder/catalog.json`` (made by make_test_songs.py) as a Navidrome
with one playlist, "Beat test", and the plugin's profiles from each song's ``plugin`` entry. It checks the login
(user ``test``, password ``test``; AudioMuse-AI token ``token``) the way the real ones do.

Run on its own for the end-to-end check or to try the page: ``python3 tests/fakes.py <folder> [port]``.
"""

import hashlib
import json
import os
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

USER, PASSWORD, TOKEN = "test", "test", "token"
PLAYLIST = "Beat test"


def entry(song):
    e = {"id": song["id"], "title": song["title"], "artist": song["artist"], "album": song.get("album", "Test"),
         "duration": int(song.get("duration", 60)), "suffix": os.path.splitext(song["file"])[1].lstrip(".")}
    if song.get("bpm"):
        e["bpm"] = song["bpm"]
    if song.get("genre"):
        e["genre"] = song["genre"]
        e["genres"] = [{"name": g.strip()} for g in song["genre"].split("/")]
    return e


class FakeServer:
    def __init__(self, folder, port=0):
        self.folder = folder
        with open(os.path.join(folder, "catalog.json")) as f:
            self.songs = json.load(f)
        self.requests = []
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
                q = {k: v[0] for k, v in urllib.parse.parse_qs(url.query).items()}
                fake.requests.append((url.path, q))
                if url.path.startswith("/plugins/siansa_analysis/"):
                    if len(self.requestline) > 4094:  # as AudioMuse-AI's web server (gunicorn's limit_request_line)
                        return self.reply(400, {"error": "Request Line is too large"})
                    return self.plugin(url.path, q)
                if not url.path.startswith("/rest/"):
                    return self.reply(404, {"error": "not found"})
                if q.get("u") != USER or q.get("t") != hashlib.md5((PASSWORD + q.get("s", "")).encode()).hexdigest() \
                        or "p" in q:
                    return self.subsonic(error="Wrong username or password")
                method = url.path[len("/rest/"):].removesuffix(".view")
                by_id = {s["id"]: s for s in fake.songs}
                if method == "ping":
                    return self.subsonic()
                if method == "getPlaylists":
                    return self.subsonic({"playlists": {"playlist": [
                        {"id": "pl-1", "name": PLAYLIST, "songCount": len(fake.songs)},
                        {"id": "pl-2", "name": "Something else", "songCount": 0}]}})
                if method == "getPlaylist":
                    if q.get("id") != "pl-1":
                        return self.subsonic({"playlist": {"id": q.get("id"), "entry": []}})
                    return self.subsonic({"playlist": {"id": "pl-1", "name": PLAYLIST,
                                                       "entry": [entry(s) for s in fake.songs]}})
                if method == "getSong":
                    if q.get("id") not in by_id:
                        return self.subsonic(error="Song not found")
                    return self.subsonic({"song": entry(by_id[q["id"]])})
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

            def plugin(self, path, q):
                if self.headers.get("Authorization") != f"Bearer {TOKEN}":
                    return self.reply(401, {"error": "login needed"})
                if path.endswith("/api/health"):
                    return self.reply(200, {"plugin": "siansa_analysis", "plugin_version": "0.5.7-fake", "status": "ok"})
                if path.endswith("/api/profiles"):
                    ids = [i for i in q.get("ids", "").split(",") if i]
                    profiles = [{"track_id": s["id"], **s["plugin"]} for s in fake.songs
                                if s["id"] in ids and s.get("plugin")]
                    return self.reply(200, {"schema_version": 1, "profiles": profiles,
                                            "pending": [i for i in ids if i not in {p["track_id"] for p in profiles}]})
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
    server = FakeServer(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 4533)
    print(f"Fake Navidrome and AudioMuse-AI on port {server.server.server_address[1]}, {len(server.songs)} songs",
          flush=True)
    server.server.serve_forever()
