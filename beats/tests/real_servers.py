#!/usr/bin/env python3
"""Siansa Beats against real Navidrome, Emby and Jellyfin servers with the same music (the Siansa app's test servers:
its scripts/servers/start.sh; user max, password sesame). No model: a stand-in gives every song the same beats, as
what's checked is the servers' side.

    python3 tests/real_servers.py navidrome=http://127.0.0.1:4533,emby=http://127.0.0.1:8096,jellyfin=http://127.0.0.1:8097 <library.json>

Checks, on each server: the login, its music libraries, every song with its tags and where its file is, and the
song's own file downloaded whole. Then the point of it (0.23.0): the songs analysed from Navidrome are found again on
Emby and on Jellyfin by their files (each server has the folder under another path), none is analysed twice, and
the app is answered by each server's own ids.
Exits 1 on a failure.
"""

import json
import os
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from beats import match  # noqa: E402
from beats.server import App, Config, make_server  # noqa: E402
from beats.servers import make_client  # noqa: E402

USER, PASSWORD = "max", "sesame"
problems = []


def check(ok, text):
    print(("  ok    " if ok else "  WRONG ") + text)
    if not ok:
        problems.append(text)


class Samples:
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return 30 * 22_050


class Tracker:
    loaded, load_seconds, calls = True, 0.0, 0

    def __call__(self, samples):
        Tracker.calls += 1
        beats = [round(0.3 + i * 0.5, 2) for i in range(55)]
        return beats, beats[::4]


def main(servers, library_path):
    servers = dict(pair.split("=", 1) for pair in servers.split(","))
    with open(library_path) as f:
        library = json.load(f)
    wanted = library["songs"] if isinstance(library, dict) else library
    print(f"{len(wanted)} songs in the test library")

    # ---- each server on its own ----
    sizes = {}
    tmp = tempfile.mkdtemp()
    for kind, url in servers.items():
        print(f"\n{kind} at {url}")
        client = make_client(kind, url, USER, PASSWORD)
        info = client.ping()
        check(bool(info.get("version")), f"answers: {info}")
        libraries = client.libraries()
        check(len(libraries) >= 1, f"music libraries: {libraries}")
        songs = [s for f in libraries for s in client.all_songs(f["id"])]
        check(len(songs) == len(wanted), f"{len(songs)} songs listed")
        check(all(s["title"] and s["artist"] and s["album"] and s["duration"] > 0 for s in songs),
              "every song has its title, artist, album and length")
        with_path = [s for s in songs if s.get("path")]
        check(len(with_path) == len(songs), f"every song says where its file is ({len(with_path)} of {len(songs)}), "
              f"such as {with_path[0]['path'] if with_path else None}")
        if kind == "navidrome":
            real = client.real_paths()
            check(len(real) == len(songs), f"Navidrome's own API gives the files' real paths ({len(real)})")
            check(all(os.path.splitext(p)[1] for p in real.values()), "and they end in the files' names")
        try:
            wrong = make_client(kind, url, USER, "not the password")
            wrong.libraries()
            check(False, "a wrong password is refused")
        except Exception as e:  # noqa: BLE001
            check("refused" in str(e) or "Wrong" in str(e) or "password" in str(e).lower() or "401" in str(e),
                  f"a wrong password is refused: {e}")
        for song in songs:
            target = os.path.join(tmp, "file")
            size = client.download(song["id"], target)
            sizes.setdefault(match.tail(song["path"]) or match.meta(song), {})[kind] = size
        with client.audio(songs[0]["id"], "bytes=0-99") as r:
            check(r.status in (200, 206) and len(r.read()) > 0, f"the page's player gets the song (HTTP {r.status})")
    same = [by for by in sizes.values() if len(by) == len(servers) and len(set(by.values())) == 1]
    check(len(same) == len(wanted), f"every server sends the same file, whole: {len(same)} of {len(wanted)} songs "
          f"({[by for by in sizes.values() if by not in same][:2]})")

    # ---- the three together: analysed once ----
    print("\nSiansa Beats with all three")
    first = servers.pop("navidrome")
    env = {"NAVIDROME_URL": first, "NAVIDROME_USER": USER, "NAVIDROME_PASSWORD": PASSWORD,
           "DATA_DIR": os.path.join(tmp, "data"), "PORT": "0"}

    class NoStyle:
        available, loaded = False, False

    app = App(Config(env), tracker=Tracker(), decode=lambda path: Samples(os.path.getsize(path)),
              measure=lambda samples: None, style=NoStyle(),
              sound=lambda path, samples, rate: {"ref_lufs": -12.0, "start_ramp": "", "end_ramp": "", "duration_ms": 30000})
    server = make_server(app, host="127.0.0.1", port=0)
    base = f"http://127.0.0.1:{server.server_address[1]}"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    app.worker.start()

    def call(path, method="GET", body=None):
        request = urllib.request.Request(base + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=120) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise SystemExit(f"{method} {path} -> {e.code}: {e.read().decode()[:300]}") from None

    def wait_done():
        end = time.time() + 120
        while time.time() < end:
            c = call("/api/status")["counts"]
            if c.get("songs") and c["songs"] == c.get("done", 0) + c.get("failed", 0):
                return c
            time.sleep(0.3)
        raise SystemExit(f"not done: {call('/api/status')['counts']}")

    ids = [f["id"] for f in call("/api/libraries")]
    read = call("/api/libraries", "POST", {"ids": ids})
    check(read == {"songs": len(wanted), "queued": len(wanted)}, f"Navidrome's songs are queued: {read}")
    counts = wait_done()
    check(counts.get("done") == len(wanted) and Tracker.calls == len(wanted), f"all analysed once: {counts}")

    for kind, url in servers.items():
        # The address as the browser shows it: cut back to the server's own (0.24.1).
        tested = call("/api/servers/test", "POST", {"kind": kind, "url": url + "/web/index.html#!/home", "user": USER,
                                                    "password": PASSWORD})
        check(tested["url"] in (url, url + "/emby") and tested["libraries"],
              f"{kind}: found from the address the browser shows ({tested['url']})")
        added = call("/api/servers", "POST", {"kind": kind, "name": "", "url": url, "user": USER, "password": PASSWORD})
        libraries = call(f"/api/libraries?server={added['id']}")
        read = call(f"/api/libraries?server={added['id']}", "POST", {"ids": [f["id"] for f in libraries]})
        check(read == {"songs": len(wanted), "queued": 0}, f"{kind}: its songs are found here, none queued: {read}")
        said = [e["message"] for e in call("/api/status")["events"] if added["name"] in e["message"] and "Library" in e["message"]]
        check(any(f"{len(wanted)} by their file" in e for e in said), f"{kind}: all found by their files: {said[:1]}")
        mine = [s for f in libraries for s in app.servers.client(added["id"]).all_songs(f["id"])]  # the kept sign-in
        profiles = call(f"/api/v1/profiles?server={added['id']}&ids=" + ",".join(s["id"] for s in mine))
        check(len(profiles["profiles"]) == len(wanted) and not profiles["pending"],
              f"{kind}: the app gets every song's profile by {kind}'s own ids ({len(profiles['profiles'])})")
        status = call(f"/api/status?server={added['id']}")
        check(status["counts"].get("songs") == len(wanted) and status["counts"].get("done") == len(wanted),
              f"{kind}: its own numbers: {status['counts']}")
    # What songs sound like (0.25): each song a made-up vector, as if taken over from AudioMuse-AI. Asked by each
    # server's own ids, the answer is in that server's ids; "your sound" reads the server's most played songs.
    import numpy

    here = app.store.song_ids()
    rng = numpy.random.RandomState(7)
    app.store.save_sound([{"song_id": i, "musicnn": rng.rand(200).astype(numpy.float32).tobytes(), "fp": "x"} for i in here])
    for shown in call("/api/v1/capabilities")["servers"]:
        kind, name = shown["kind"], shown["id"]
        client = app.servers.client(name)
        mine = {s["id"] for s in client.all_songs()}
        one = sorted(mine)[0]
        like = call(f"/api/v1/audiomuse/api/similar_tracks?server={name}&item_id={urllib.parse.quote(one)}&n=5")
        check(len(like) == min(5, len(mine) - 1) and {s["item_id"] for s in like} <= mine - {one},
              f"{kind}: songs like a song come in {kind}'s own ids ({len(like)})")
        if kind == "navidrome":
            client._get("scrobble", id=one, submission="true")
        else:
            client._as_user(lambda user_id, token: client._json(
                f"/UserPlayedItems/{one}?userId={user_id}" if client.jellyfin else f"/Users/{user_id}/PlayedItems/{one}",
                token, body={}))
        played = client.most_played(5)
        check(played[:1] == [one], f"{kind}: the most played song is the one just played ({played[:2]})")
        yours = call(f"/api/v1/audiomuse/api/sonic_fingerprint/generate?server={name}", "POST", {"n": 3})
        check(len(yours) == 3 and yours[0]["item_id"] == one and {s["item_id"] for s in yours} <= mine,
              f"{kind}: your sound starts from the most played song, in {kind}'s ids")

    time.sleep(1)
    check(Tracker.calls == len(wanted), f"nothing was analysed again ({Tracker.calls} runs of the model for {len(wanted)} songs)")
    check(call("/api/status")["counts"].get("songs") == len(wanted), "the catalogue still has each song once")
    caps = call("/api/v1/capabilities")
    check([s["kind"] for s in caps["servers"]] == ["navidrome", *servers], f"the app is told the servers: {[s['id'] for s in caps['servers']]}")

    print()
    if problems:
        print(f"{len(problems)} wrong:")
        for p in problems:
            print("  " + p)
        raise SystemExit(1)
    print("All right on every server.")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
