"""Several music servers (0.23.0): Emby and Jellyfin beside Navidrome, and a second user of one. A song two servers
have is analysed once; each server keeps its own ids, and what's asked with ``?server=`` is about that server's songs.
Navidrome, Emby and Jellyfin are tests/fakes.py; the real ones are checked by real_servers.py in CI."""

import json
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from beats.mediabrowser import MediaBrowser  # noqa: E402
from beats.store import Store  # noqa: E402
from beats.subsonic import ServerError, Subsonic  # noqa: E402
from fakes import PASSWORD, SECOND_PASSWORD, TOKEN, USER, FakeMediaServer  # noqa: E402
from test_service import CATALOG, Base, FakeStyle, Service  # noqa: E402

ONLY_EMBY = {"id": "t128", "title": "Only Here", "artist": "Emby Artist", "album": "Elsewhere", "genre": "Techno",
             "duration": 45, "file": "t140.flac"}  # its file's content says t140: the stand-in decode reads it as that


class ClientTest(Base):
    def both(self):
        for jellyfin in (False, True):
            media = FakeMediaServer(self.music, jellyfin=jellyfin).start()
            self.addCleanup(media.stop)
            yield media, MediaBrowser(media.url, USER, PASSWORD, jellyfin=jellyfin)

    def test_libraries_songs_and_files_each_in_its_servers_own_way(self):
        for media, client in self.both():
            with self.subTest(client.label):
                self.assertIn(client.ping()["type"], ("Emby Server", "Jellyfin Server"))
                self.assertEqual(client.libraries(), [{"id": "lib1", "name": "Music"}])  # the films aren't music
                songs = client.all_songs("lib1", page=3)  # two pages
                self.assertEqual([s["id"] for s in songs], [media.prefix + s["id"] for s in CATALOG])
                first = songs[0]
                self.assertEqual((first["title"], first["artist"], first["album"], first["suffix"], first["genres"]),
                                 ("Control", "Ecsape", "Test", "flac", ["Trance"]))
                self.assertEqual(first["path"], "/mnt/media/music/Ecsape/Test/t140.flac")
                self.assertAlmostEqual(first["duration"], 60.4)
                self.assertEqual(client.all_songs(None), songs)
                target = os.path.join(self.tmp, "song")
                self.assertEqual(client.download(media.prefix + "t140", target), 4)
                with open(target) as f:
                    self.assertEqual(f.read(), "t140")
                with client.audio(media.prefix + "t140", "bytes=2-") as r:
                    self.assertEqual((r.status, r.read()), (206, b"40"))
                # The token is in a header, never in an address (so never in a log).
                self.assertFalse(any("tok-1" in json.dumps(r) for r in media.requests))

    def test_a_login_the_server_stops_taking_is_made_again(self):
        for media, client in self.both():
            with self.subTest(client.label):
                client.libraries()
                media.token = "tok-2"
                self.assertEqual(len(client.all_songs("lib1")), len(CATALOG))
                self.assertEqual(media.logins, 2)

    def test_errors_say_what_went_wrong(self):
        for media, _ in self.both():
            with self.assertRaisesRegex(ServerError, "refused the user name or password"):
                MediaBrowser(media.url, USER, "wrong", jellyfin=media.jellyfin).libraries()
        with self.assertRaisesRegex(ServerError, "Nothing answers at that address and port"):
            MediaBrowser("http://127.0.0.1:1", USER, PASSWORD).ping()
        # A Navidrome isn't an Emby.
        with self.assertRaises(ServerError):
            MediaBrowser(self.fake.url, USER, PASSWORD).ping()

    def test_navidrome_says_where_its_files_really_are(self):
        nd = Subsonic(self.fake.url, USER, PASSWORD)
        songs = nd.all_songs("1")
        self.assertEqual(songs[0]["path"], "/music/Ecsape/Test/t140.flac")
        self.fake.native = False  # an older Navidrome, or its own API closed off: Subsonic's made-up path
        self.assertEqual(Subsonic(self.fake.url, USER, PASSWORD).all_songs("1")[0]["path"], "Ecsape/Test/01 - Control.flac")


class ServersTest(Service):
    def media(self, **options):
        media = FakeMediaServer(self.music, **options).start()
        self.addCleanup(media.stop)
        return media

    def analysed(self):
        self.start(style=FakeStyle(os.path.join(self.tmp, "models")))
        self.call("/api/libraries", "POST", {"ids": ["1"]})
        self.wait_done()
        self.assertEqual(self.tracker.calls, 3)  # the fourth can't be decoded

    def add(self, media, name=None, **fields):
        return self.call("/api/servers", "POST", {"kind": "jellyfin" if media.jellyfin else "emby", "name": name,
                                                  "url": media.url, "user": USER, "password": PASSWORD, **fields})

    def events(self):
        return [e["message"] for e in self.call("/api/status")["events"]]

    def test_a_song_another_server_has_too_is_not_analysed_again(self):
        self.analysed()
        emby = self.media()
        added = self.add(emby, "Living room Emby")
        self.assertEqual(added, {"id": "emby", "kind": "emby", "name": "Living room Emby", "url": emby.url, "user": USER,
                                 "default": False, "signed_in": True})
        self.assertEqual(self.call("/api/libraries?server=emby"),
                         [{"id": "lib1", "name": "Music", "on": False, "songs": 0, "done": 0}])
        self.assertEqual(self.call("/api/libraries?server=emby", "POST", {"ids": ["lib1"]}), {"songs": 4, "queued": 0})
        self.assertIn("Library Music on Living room Emby: 4 songs, 0 new in the queue, 4 already analysed from another "
                      "server (4 by their file)", self.events())
        time.sleep(0.3)
        self.assertEqual(self.tracker.calls, 3)
        self.assertFalse(any("/Audio/" in path for path, _ in emby.requests))  # nothing downloaded from it

        # What Emby's ids are asked about is what Navidrome's songs were found to be.
        profiles = self.call("/api/v1/profiles?ids=e-t140,e-t122,nope&server=emby")
        self.assertEqual([p["track_id"] for p in profiles["profiles"]], ["e-t140", "e-t122"])
        self.assertEqual(profiles["pending"], ["nope"])
        mine = self.call("/api/v1/profiles?ids=t140")["profiles"][0]
        self.assertEqual({**profiles["profiles"][0], "track_id": "t140"}, mine)
        # Without naming the server too: an id is understood from any of them.
        self.assertEqual(self.call("/api/v1/profiles?ids=e-t140")["profiles"][0]["track_id"], "e-t140")
        self.assertEqual(self.call("/api/grids?ids=e-t140,t122")["missing"], [])

        # What's answered is the named server's own ids, by its id or its name.
        self.assertEqual(sorted(self.call("/api/v1/genre?name=Trance")["ids"]), ["t140", "t96w"])
        self.assertEqual(sorted(self.call("/api/v1/genre?name=Trance&server=emby")["ids"]), ["e-t140", "e-t96w"])
        self.assertEqual(sorted(self.call("/api/v1/genre?name=Trance&server=living%20room%20emby")["ids"]), ["e-t140", "e-t96w"])
        genres = {g["name"]: g for g in self.call("/api/v1/genres?server=emby")["genres"]}
        self.assertEqual(genres["Trance"]["libraries"], {"lib1": 2})
        self.assertEqual({g["name"]: g for g in self.call("/api/v1/genres")["genres"]}["Trance"]["libraries"], {"1": 2})

        caps = self.call("/api/v1/capabilities?server=emby")
        self.assertEqual(caps["music_server"], {"kind": "emby", "set_up": True, "id": "emby", "name": "Living room Emby"})
        self.assertEqual([(s["id"], s["kind"], s["default"], s["songs"]) for s in caps["servers"]],
                         [("main", "navidrome", True, 4), ("emby", "emby", False, 4)])
        self.assertEqual(caps["songs"], {"known": 4, "analysed": 3, "failed": 1, "waiting": 0})
        self.assertNotIn(PASSWORD + '"', json.dumps(caps).replace('"user": "test"', ""))

        # The page's song says where it is, and plays from either.
        song = self.call("/api/songs/t140")
        self.assertEqual(song["servers"], [{"id": "main", "name": "Navidrome", "item_id": "t140"},
                                           {"id": "emby", "name": "Living room Emby", "item_id": "e-t140"}])
        # A bad blend marked while playing from Emby names the songs here.
        self.call("/api/v1/bad-blends?server=emby", "POST", {"from_id": "e-t140", "to_id": "e-t122", "technique": "x"})
        self.assertEqual(self.call("/api/bad-blends")[0]["from_title"], "Control")

    def test_songs_are_found_by_their_tags_when_the_paths_say_nothing(self):
        self.fake.native = False
        self.analysed()
        jellyfin = self.media(jellyfin=True, paths=False)
        self.add(jellyfin)
        self.assertEqual(self.call("/api/libraries?server=jellyfin", "POST", {"ids": ["lib1"]}), {"songs": 4, "queued": 0})
        self.assertTrue(any("4 already analysed from another server (4 by their tags)" in e for e in self.events()))
        self.assertEqual(self.call("/api/v1/profiles?ids=j-t122&server=jellyfin")["pending"], [])

    def test_a_song_only_the_other_server_has_is_analysed_from_it(self):
        self.analysed()
        emby = self.media(only=["t140", "t122"], extra=[ONLY_EMBY])
        self.add(emby)
        self.assertEqual(self.call("/api/libraries?server=emby", "POST", {"ids": ["lib1"]}), {"songs": 3, "queued": 1})
        end = time.time() + 20
        while self.tracker.calls < 4 and time.time() < end:
            time.sleep(0.05)
        self.assertEqual(self.tracker.calls, 4)
        self.assertTrue(any(path == "/Audio/e-t128/stream" for path, _ in emby.requests))
        self.wait_done()
        # Each server's numbers are its own.
        self.assertEqual(self.call("/api/status?server=emby")["counts"], {"done": 3, "songs": 3})
        self.assertEqual(self.call("/api/status")["counts"], {"done": 3, "failed": 1, "songs": 4})
        self.assertEqual(sorted(s["id"] for s in self.call("/api/songs?server=emby")), ["e-t128", "t122", "t140"])
        self.assertEqual(len(self.call("/api/songs")), 4)
        status = self.call("/api/status?server=emby")
        self.assertEqual((status["server"], status["server_kind"], status["navidrome"]), ("emby", "emby", emby.url))
        self.assertEqual([(s["id"], s["songs"], s["libraries"]) for s in status["servers"]],
                         [("main", 4, ["1"]), ("emby", 3, ["lib1"])])
        self.assertEqual(self.call("/api/v1/profiles?ids=e-t128")["profiles"][0]["track_id"], "e-t128")
        raw, data = self.call("/api/audio/e-t128", raw=True)
        self.assertEqual(data, b"t140")
        raw, data = self.call("/api/export.csv?server=emby", raw=True)
        self.assertEqual(len(data.decode().strip().splitlines()), 4)

        # Taken away again: the song only it had goes, the others stay analysed.
        self.assertEqual(self.call("/api/servers/emby", "DELETE"), {"removed": 3})
        self.assertEqual([s["id"] for s in self.call("/api/servers")["servers"]], ["main"])
        self.assertEqual(self.call("/api/status")["counts"], {"done": 3, "failed": 1, "songs": 4})
        self.assertEqual(self.call("/api/v1/profiles?ids=e-t128,e-t140,t140")["pending"], ["e-t128", "e-t140"])

    def test_a_second_user_of_the_same_server(self):
        self.analysed()
        added = self.call("/api/servers", "POST", {"kind": "navidrome", "name": "", "url": self.fake.url + "/",
                                                   "user": "second", "password": SECOND_PASSWORD})
        self.assertEqual((added["id"], added["name"]), ("navidrome", "Navidrome (second)"))
        self.assertEqual(self.call("/api/libraries?server=navidrome", "POST", {"ids": ["2", "1"]})["queued"], 0)
        self.assertTrue(any("4 already analysed from another server (4 by the same id)" in e for e in self.events()))
        self.assertEqual(self.call("/api/status?server=navidrome")["counts"]["songs"], 4)
        self.assertEqual(self.tracker.calls, 3)
        # The same user twice is a mistake.
        with self.assertRaises(urllib.error.HTTPError) as e:
            self.call("/api/servers", "POST", {"kind": "navidrome", "url": self.fake.url, "user": "SECOND", "password": SECOND_PASSWORD})
        self.assertEqual(e.exception.code, 400)
        self.assertIn("already here", json.loads(e.exception.read())["error"])

    def test_the_default_server_and_what_cant_be_done(self):
        self.start()
        jellyfin = self.media(jellyfin=True)

        def refused(path, method="POST", body=None):
            with self.assertRaises(urllib.error.HTTPError) as e:
                self.call(path, method, body)
            return e.exception.code, json.loads(e.exception.read())["error"]

        fields = {"kind": "jellyfin", "url": jellyfin.url, "user": USER, "password": PASSWORD}
        self.assertEqual(refused("/api/servers", body={**fields, "kind": "plex"})[0], 400)
        self.assertIn("http://", refused("/api/servers", body={**fields, "url": "nas:8096"})[1])
        self.assertIn("password", refused("/api/servers", body={**fields, "password": ""})[1])
        code, why = refused("/api/servers", body={**fields, "password": "wrong"})
        self.assertEqual(code, 502)
        self.assertIn("refused the user name or password", why)
        self.assertNotIn("wrong", why)
        self.assertEqual(refused("/api/servers", body={**fields, "url": "http://127.0.0.1:1"})[0], 502)
        self.assertEqual(self.call("/api/servers")["servers"], [
            {"id": "main", "kind": "navidrome", "name": "Navidrome", "url": self.fake.url, "user": USER, "default": True,
             "signed_in": True, "songs": 0, "libraries": []}])
        # Asked before it's added: what it is and what it has.
        self.assertEqual(self.call("/api/servers/test", "POST", fields)["libraries"], [{"id": "lib1", "name": "Music"}])
        self.assertEqual(self.add(jellyfin)["name"], "Jellyfin (test)")
        self.assertIn("named", refused("/api/servers", body={**fields, "name": "navidrome", "user": "other"})[1])

        self.assertEqual(self.call("/api/servers/jellyfin/default", "POST"), {"default": "jellyfin"})
        self.assertEqual(self.call("/api/status")["server"], "jellyfin")
        self.assertEqual(self.call("/api/v1/capabilities")["music_server"]["kind"], "jellyfin")
        self.assertEqual(self.call("/api/libraries")[0]["id"], "lib1")
        self.assertEqual(self.call("/api/libraries?server=main")[0]["id"], "1")
        self.assertEqual(self.call("/api/servers/jellyfin", "POST", {"name": "Den"})["name"], "Den")
        self.assertEqual(self.call("/api/servers/main", "POST", {"name": "Home"})["name"], "Home")
        # The first server is like any other since 0.24: changed and removed on the page. A new address or user
        # needs the password again (it isn't said to another address unasked).
        self.assertIn("password is needed", refused("/api/servers/main", body={"url": "http://elsewhere"})[1])
        self.assertEqual(refused("/api/servers/main", body={"url": "http://127.0.0.1:1", "password": PASSWORD})[0], 502)
        self.assertEqual(self.call("/api/servers")["servers"][0]["url"], self.fake.url)  # not changed by a failed try
        self.assertEqual(refused("/api/servers/nope", "DELETE")[0], 404)
        self.assertEqual(refused("/api/servers/nope/default")[0], 404)
        code, why = refused("/api/status?server=nope", "GET")
        self.assertEqual(code, 400)
        self.assertIn("Home, Den", why)
        # Passwords are kept, never said: not in the list, the status, the debug report or the log.
        for path in ("/api/servers", "/api/status", "/api/debug", "/api/log?limit=5000", "/api/v1/capabilities"):
            raw, data = self.call(path, raw=True)
            self.assertNotIn('"password": "' + PASSWORD, data.decode())
            self.assertNotIn("tok-1", data.decode())
        self.call("/api/servers/jellyfin", "DELETE")
        self.assertEqual(self.call("/api/status")["server"], "main")

    def test_the_page_reads_in_the_background_and_shows_how_far(self):
        self.analysed()
        emby = self.media()
        self.add(emby)
        slow = threading.Event()
        real = self.app.worker.servers.client("emby").all_songs

        def held(library, *a, **k):
            slow.wait(10)
            return real(library, *a, **k)

        self.app.worker.servers.client("emby").all_songs = held
        # Chosen at once, read in a thread: the answer doesn't wait for the read.
        started = time.monotonic()
        self.assertEqual(self.call("/api/libraries?server=emby", "POST", {"ids": ["lib1"], "background": True}), {"reading": True})
        self.assertLess(time.monotonic() - started, 2)
        reading = self.call("/api/status")["reading"]
        self.assertIsNotNone(reading)  # shown at once, before the thread has asked anything
        end = time.time() + 5
        while reading.get("library") is None and time.time() < end:
            time.sleep(0.05)
            reading = self.call("/api/status")["reading"]
        self.assertEqual({k: reading[k] for k in ("server", "library", "libraries", "done", "songs", "why")},
                         {"server": "emby", "library": "Music", "libraries": 1, "done": 0, "songs": None, "why": None})
        # Already reading: a second press doesn't start another.
        self.assertEqual(self.call("/api/libraries/read", "POST", {"background": True}), {"reading": False})
        slow.set()
        end = time.time() + 10
        while self.call("/api/status")["reading"] and time.time() < end:
            time.sleep(0.05)
        self.assertIsNone(self.call("/api/status")["reading"])
        self.assertEqual(self.call("/api/status?server=emby")["counts"]["songs"], 4)
        # Navidrome's songs were read in this run with their paths, so Emby was read alone: Navidrome wasn't again.
        said = [e["message"] for e in self.call("/api/status")["events"]]
        self.assertFalse(any("first, this once" in e for e in said))
        self.assertEqual(sum(e.startswith("Library Music on") for e in said), 1)

    def test_the_default_server_is_read_first_only_once(self):
        self.analysed()
        self.app.store.set_setting("paths_read:main", None)  # as a database from before 0.23 left it
        emby, jellyfin = self.media(), self.media(jellyfin=True)
        self.add(emby)
        self.call("/api/libraries?server=emby", "POST", {"ids": ["lib1"]})
        said = [e["message"] for e in self.call("/api/status")["events"]]
        self.assertTrue(any("first, this once" in e for e in said))
        self.assertTrue(self.app.store.setting("paths_read:main"))
        self.add(jellyfin)
        before = len([p for p, _ in self.fake.requests if "search3" in p])
        self.call("/api/libraries?server=jellyfin", "POST", {"ids": ["lib1"]})
        self.assertEqual(len([p for p, _ in self.fake.requests if "search3" in p]), before)  # Navidrome not read again

    def test_emby_as_it_is_found_at_home(self):
        """What made a real Emby fail (the owner, 11 Nov): the address as the browser shows it, Emby behind /emby only,
        a user without a password, the other kind chosen, nothing at that port. Each is either taken or said plainly,
        in the answer and in the log."""
        self.start()
        emby = self.media(under_emby=True)
        fields = {"kind": "emby", "url": emby.url + "/web/index.html#!/home", "user": USER, "password": PASSWORD}
        tested = self.call("/api/servers/test", "POST", fields)
        self.assertEqual((tested["url"], tested["libraries"]), (emby.url + "/emby", [{"id": "lib1", "name": "Music"}]))
        added = self.call("/api/servers", "POST", {**fields, "name": "Emby"})
        self.assertEqual(added["url"], emby.url + "/emby")
        self.call("/api/libraries?server=emby", "POST", {"ids": ["lib1"]})
        self.assertEqual(self.call("/api/status?server=emby")["counts"]["songs"], 4)

        def refused(body):
            with self.assertRaises(urllib.error.HTTPError) as e:
                self.call("/api/servers/test", "POST", body)
            return e.exception.code, json.loads(e.exception.read())["error"]

        jellyfin = self.media(jellyfin=True)
        self.assertEqual(refused({**fields, "url": jellyfin.url}),
                         (502, "That's Jellyfin (Jellyfin Server 10.10.7): choose Jellyfin as its kind"))
        code, why = refused({**fields, "url": self.fake.url})
        self.assertEqual(code, 502)
        self.assertIn("No Emby answers at", why)
        # What did answer is said (0.24.2): which address, how, and how its answer starts.
        self.assertIn(f"What answered: {self.fake.url}/System/Info/Public: HTTP ", why)
        self.assertIn(f"; {self.fake.url}/emby/System/Info/Public: HTTP ", why)
        # An Emby that lets in local addresses only (the owner's, 11 Nov): Siansa Beats' container isn't one to it.
        emby.forbids = True
        code, why = refused({**fields, "url": emby.url})
        emby.forbids = False
        self.assertEqual(code, 502)
        self.assertIn("refuses Siansa Beats (HTTP 403, before any sign-in)", why)
        self.assertIn("LAN networks", why)
        self.assertIn("HTTP 403 (text/plain from ", why)
        self.assertIn('starting "Forbidden")', why)
        code, why = refused({**fields, "url": "http://127.0.0.1:1"})
        self.assertEqual(code, 502)
        self.assertIn("Nothing answers at that address and port", why)
        with self.assertLogs("siansa_beats.server", "WARNING") as said:
            code, why = refused({**fields, "password": "wrong"})
        self.assertEqual((code, why), (502, "Emby refused the user name or password"))
        # Why is in the log too (it said only "-> 502" before), never with the password.
        self.assertTrue(any("Testing Emby at " in m and "refused the user name or password" in m for m in said.output))
        self.assertFalse(any("wrong" in m for m in said.output))

        # A user without a password: taken, and signed in again with none when the token is refused.
        bare = self.media()
        bare.password = ""
        added = self.call("/api/servers", "POST", {"kind": "emby", "name": "Bare", "url": bare.url, "user": USER, "password": ""})
        bare.token = "tok-9"
        self.assertEqual(self.call(f"/api/libraries?server={added['id']}")[0]["id"], "lib1")
        self.assertEqual(bare.logins, 2)  # when it was added, and again with no password when the token was refused
        # Navidrome still needs one.
        self.assertEqual(refused({"kind": "navidrome", "url": self.fake.url, "user": USER, "password": ""}),
                         (400, "Its password is needed"))

    def test_answers_are_squeezed_for_those_that_take_it(self):
        self.start()
        import gzip

        raw, data = self.call("/", headers={"Accept-Encoding": "gzip"}, raw=True)
        self.assertEqual(raw.headers["Content-Encoding"], "gzip")
        self.assertIn(b"Siansa Beats", gzip.decompress(data))
        raw, data = self.call("/", raw=True)
        self.assertIsNone(raw.headers["Content-Encoding"])
        raw, data = self.call("/api/health", headers={"Accept-Encoding": "gzip"}, raw=True)
        self.assertIsNone(raw.headers["Content-Encoding"])  # too small to be worth it


class StoreLinksTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "beats.db")

    def song(self, song_id, **fields):
        return {"id": song_id, "title": "Song " + song_id, "artist": "A", "album": "B", "duration": 100.0, "suffix": "flac",
                "tag_bpm": None, "genres": ["House"], "path": f"/music/A/B/{song_id}.flac", **fields}

    def test_a_database_from_before_links_gets_them(self):
        import sqlite3

        old = Store(self.path)
        old.add_songs([self.song("a"), self.song("b")], "1")
        old.close()
        db = sqlite3.connect(self.path)  # as 0.22 left it: no links, no keys to compare by
        db.execute("DELETE FROM links")
        db.execute("DELETE FROM settings WHERE key='links_made'")
        db.execute("UPDATE songs SET meta=NULL, loose=NULL, tail=NULL, path=NULL")
        db.commit()
        db.close()
        store = Store(self.path)
        self.assertEqual(store.resolve(["a", "b", "c"]), {"a": "a", "b": "b"})
        self.assertEqual(store.where("a"), [("main", "a")])
        self.assertEqual(store.folder_counts("main"), {"1": {"songs": 2, "done": 0}})
        # Found by its tags by another server straight away; its path comes with the next read of its own.
        found = store.link_songs("emby", "lib", [self.song("e1", title="Song a", path="/mnt/A/B/a.flac")])
        self.assertEqual((found["queued"], found["matched"]), ([], {"tags": 1}))
        store.close()

    def test_two_items_of_one_server_are_two_songs(self):
        store = Store(self.path)
        store.add_songs([self.song("a")], "1")
        # The same tags and length, another file: its own song, not the first one's.
        result = store.link_songs("main", "1", [self.song("a2", title="Song a", path="/music/A/Other/a.flac")])
        self.assertEqual(result["queued"], ["a2"])
        # On another server both are found, each by its file.
        result = store.link_songs("emby", "lib", [self.song("e1", title="Song a", path="/mnt/A/B/a.flac"),
                                                 self.song("e2", title="Song a", path="/mnt/A/Other/a.flac")])
        self.assertEqual(result["matched"], {"path": 2})
        self.assertEqual(store.resolve(["e1", "e2"]), {"e1": "a", "e2": "a2"})
        store.close()

    def test_ids_two_servers_share_stay_apart(self):
        store = Store(self.path)
        store.add_songs([self.song("7")], "1")
        result = store.link_songs("emby", "lib", [self.song("7", title="Another", path="/mnt/X/Y/z.flac")])
        self.assertEqual(result["queued"], ["emby:7"])
        self.assertEqual(store.resolve(["7"]), {"7": "7"})
        self.assertEqual(store.resolve(["7"], "emby"), {"7": "emby:7"})
        self.assertEqual(store.item_ids(["emby:7", "7"], "emby"), {"emby:7": "7"})
        self.assertEqual(store.where("emby:7"), [("emby", "7")])
        store.close()

    def test_a_library_left_takes_only_the_songs_nothing_else_has(self):
        store = Store(self.path)
        store.add_songs([self.song("a"), self.song("b")], "1")
        store.link_songs("emby", "lib", [self.song("e1", title="Song a", path="/mnt/A/B/a.flac")])
        self.assertEqual(store.remove_folder("1", "main"), 2)
        self.assertEqual(store.song_ids(), ["a"])  # Emby still has it
        self.assertEqual(store.where("a"), [("emby", "e1")])
        self.assertEqual(store.counts("emby")["songs"], 1)
        store.close()


if __name__ == "__main__":
    unittest.main()
