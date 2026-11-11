"""The listening history (beats/listens.py): listens kept once, exports of Koito, Last.fm and ListenBrainz read, the
numbers of a month or a year on the listener's clock, and listens sent on to Last.fm."""

import datetime
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from beats import listens as L  # noqa: E402
from beats.store import Store  # noqa: E402
from beats.vault import Vault  # noqa: E402
from test_service import Service  # noqa: E402


def at(text):
    """Seconds since 1970 of ``text`` (UTC)."""
    return int(datetime.datetime.fromisoformat(text).replace(tzinfo=datetime.timezone.utc).timestamp())


class FakeLastFm(L.LastFm):
    """Last.fm's stand-in: keeps what it's sent; ``fail`` is the error it answers with."""

    def __init__(self):
        super().__init__()
        self.sent, self.playing, self.signed_in, self.fail, self.loved = [], [], [], None, []

    def session(self, api_key, secret, name, password):
        if password != "right":
            raise L.LastFmError("Last.fm: Authentication Failed - You do not have permissions to access the service", 4)
        self.signed_in.append((api_key, secret, name))
        return "session-" + name

    def scrobble(self, keys, rows):
        if self.fail:
            raise self.fail
        assert keys == {"api_key": "key", "secret": "shh", "session": "session-max"}, keys
        self.sent.append([(r["at"], r["artist"], r["title"]) for r in rows])

    def now_playing(self, keys, row):
        self.playing.append((row["artist"], row["title"]))

    def love(self, keys, artist, title, loved=True):
        self.loved.append((artist, title, loved))


class Files(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def read(self, name, content):
        path = os.path.join(self.tmp, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content if isinstance(content, str) else json.dumps(content))
        kind, rows = L.read_export(path)
        return kind, list(rows)

    def test_koitos_export(self):
        alias = lambda name, primary=True: {"alias": name, "source": "Canonical", "is_primary": primary}  # noqa: E731
        kind, rows = self.read("koito_export.json", {"version": "1", "exported_at": "2026-11-11T10:00:00Z", "user": "max", "listens": [
            {"listened_at": "2022-01-08T22:32:08Z", "client": "Siansa",
             "track": {"mbid": None, "duration": 354, "aliases": [alias("Every Wave", False), alias("Every Wave To Ever Rise")]},
             "album": {"image_url": "", "mbid": None, "aliases": [alias("LP3")], "various_artists": False},
             "artists": [{"mbid": None, "is_primary": False, "aliases": [alias("Elizabeth Powell")]},
                         {"mbid": None, "is_primary": True, "aliases": [alias("American Football")]}]}]})
        self.assertEqual(kind, "koito")
        self.assertEqual(rows, [{"at": at("2022-01-08T22:32:08"), "artist": "American Football", "title": "Every Wave To Ever Rise",
                                 "album": "LP3", "duration": 354, "client": "Siansa"}])

    def test_last_fms_exports(self):
        track = {"artist": {"mbid": "", "#text": "Röyksopp"}, "album": {"mbid": "", "#text": "Melody A.M."},
                 "name": "Eple", "mbid": "", "image": [], "date": {"uts": "1262304000", "#text": "01 Jan 2010, 00:00"}}
        playing = {"artist": {"#text": "Now"}, "name": "Playing", "album": {"#text": ""}, "@attr": {"nowplaying": "true"}}
        want = [{"at": 1262304000, "artist": "Röyksopp", "title": "Eple", "album": "Melody A.M."}]
        # The export tool's JSON: pages of Last.fm's own answer (what's playing while it ran isn't a listen).
        self.assertEqual(self.read("recenttracks-max.json", [{"track": [playing, track]}]), ("lastfm", want))
        self.assertEqual(self.read("pages.json", [[track]]), ("lastfm", want))
        # Its CSV, and the older tool's four columns without a first line.
        self.assertEqual(self.read("a.csv", "uts,utc_time,artist,artist_mbid,album,album_mbid,track,track_mbid\n"
                                            '1262304000,"01 Jan 2010, 00:00",Röyksopp,,Melody A.M.,,Eple,\n'), ("lastfm", want))
        self.assertEqual(self.read("b.csv", "Röyksopp,Melody A.M.,Eple,01 Jan 2010 00:00\n"), ("lastfm", want))
        # Another site's JSON: milliseconds.
        self.assertEqual(self.read("stats.json", {"username": "max", "scrobbles": [
            {"track": "Eple", "artist": "Röyksopp", "album": "Melody A.M.", "date": 1262304000000}]}), ("lastfm", want))

    def test_listenbrainz_export(self):
        listen = {"listened_at": 1262304000, "track_metadata": {"artist_name": "Röyksopp", "track_name": "Eple",
                  "release_name": "Melody A.M.", "additional_info": {"duration_ms": 216000, "media_player": "Siansa"}}}
        want = [{"at": 1262304000, "artist": "Röyksopp", "title": "Eple", "album": "Melody A.M.", "duration": 216, "client": "Siansa"}]
        self.assertEqual(self.read("lb.json", [listen]), ("listenbrainz", want))
        self.assertEqual(self.read("lb.jsonl", json.dumps(listen) + "\n" + json.dumps(listen) + "\n")[1], want * 2)

    def test_a_file_thats_none_of_them_says_so(self):
        for name, content in (("x.json", {"hello": 1}), ("x.csv", "one,two\n1,2\n"), ("x.json", "{broken")):
            with self.assertRaises(L.CantRead):
                self.read(name, content)
        with open(os.path.join(self.tmp, "x.zip"), "wb") as f:
            f.write(b"PK\x03\x04rest")
        with self.assertRaisesRegex(L.CantRead, "unpack"):
            L.read_export(os.path.join(self.tmp, "x.zip"))


class History(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.store = Store(os.path.join(self.tmp, "beats.db"))
        self.addCleanup(self.store.close)
        self.now = at("2026-11-11T12:00:00")
        self.lastfm = FakeLastFm()
        self.listens = L.Listens(self.store, Vault(self.tmp), self.lastfm, now=lambda: self.now)

    def row(self, when, artist="Röyksopp", title="Eple", **more):
        return {"at": at(when), "artist": artist, "title": title, **more}

    def test_a_listen_is_kept_once_however_its_written(self):
        first = self.listens.add("max", [self.row("2026-11-01T10:00:00"), self.row("2026-11-01T10:00:00", "RÖYKSOPP", "eple!")])
        self.assertEqual(first, {"kept": 1, "known": 1, "skipped": 0})
        again = self.listens.add("max", [self.row("2026-11-01T10:00:00")], source="lastfm")  # the same listen in an export
        self.assertEqual(again, {"kept": 0, "known": 1, "skipped": 0})
        other = self.listens.add("kids", [self.row("2026-11-01T10:00:00")])  # another listener's is their own
        self.assertEqual(other["kept"], 1)
        bad = self.listens.add("max", [{"at": None, "artist": "A", "title": "B"}, self.row("2026-11-01T10:00:00", ""),
                                       self.row("1999-01-01T00:00:00"), self.row("2031-01-01T00:00:00")])
        self.assertEqual(bad, {"kept": 0, "known": 0, "skipped": 4})
        self.assertEqual([(u["user"], u["listens"]) for u in self.listens.users()], [("kids", 1), ("max", 1)])

    def test_the_numbers_of_a_month_a_year_and_all_time_on_the_listeners_clock(self):
        rows = [self.row("2025-12-31T22:30:00", duration=200),  # 00:30 on 1 Jan 2026 in Helsinki
                self.row("2026-01-10T08:00:00", duration=200), self.row("2026-01-10T09:00:00", "Moby", "Porcelain", album="Play"),
                self.row("2026-01-11T09:00:00", "Moby", "Porcelain", album="Play", duration=240),
                self.row("2026-03-05T20:00:00", "Moby", "Honey", album="Play")]
        self.listens.add("max", rows)
        year = self.listens.summary("max", 2026, None, "Europe/Helsinki")
        self.assertEqual(year["period"]["label"], "2026")
        self.assertEqual(year["totals"]["listens"], 5)  # the one just after midnight there is 2026's
        self.assertEqual((year["totals"]["artists"], year["totals"]["songs"], year["totals"]["albums"], year["totals"]["days"]), (2, 3, 1, 4))
        # Lengths: its own, its song's from another listen (Porcelain: 240), else a typical one (Honey).
        self.assertEqual((year["totals"]["seconds"], year["totals"]["guessed"]), (200 + 200 + 240 + 240 + L.TYPICAL_SECONDS, 1))
        self.assertEqual(year["months"][:3], [4, 0, 1])
        self.assertEqual([p["listens"] for p in year["chart"]["points"]][:3], [4, 0, 1])
        self.assertEqual([(s["artist"], s["title"], s["listens"]) for s in year["top"]["songs"]],
                         [("Moby", "Porcelain", 2), ("Röyksopp", "Eple", 2), ("Moby", "Honey", 1)])  # a tie: the later one first
        self.assertEqual([(a["artist"], a["listens"]) for a in year["top"]["artists"]], [("Moby", 3), ("Röyksopp", 2)])
        self.assertEqual(year["top"]["albums"], [{"album": "Play", "artist": "Moby", "listens": 3}])
        self.assertEqual(year["hours"][0], 1)  # half past midnight, Helsinki's clock
        self.assertEqual(year["records"]["streak"], 2)
        self.assertEqual(year["records"]["busiest_day"], {"date": "2026-01-10", "listens": 2})
        self.assertEqual([y["year"] for y in year["years"]], [2026])
        utc = self.listens.summary("max", 2026, None, "UTC")
        self.assertEqual((utc["totals"]["listens"], [y["year"] for y in utc["years"]]), (4, [2025, 2026]))

        january = self.listens.summary("max", 2026, 1, "Europe/Helsinki")
        self.assertEqual((january["period"]["label"], january["totals"]["listens"], len(january["chart"]["points"])), ("January 2026", 4, 31))
        self.assertEqual(january["totals"]["new_artists"], 2)
        march = self.listens.summary("max", 2026, 3, "Europe/Helsinki")
        self.assertEqual((march["totals"]["listens"], march["totals"]["new_artists"]), (1, 0))  # Moby isn't new in March
        everything = self.listens.summary("max", None, None, "Europe/Helsinki")
        self.assertEqual((everything["period"]["label"], everything["totals"]["listens"], everything["chart"]["unit"]), ("All time", 5, "year"))
        # A new listen shows at once: the kept numbers are worked out again.
        self.listens.add("max", [self.row("2026-03-06T20:00:00", "Moby", "Honey")])
        self.assertEqual(self.listens.summary("max", 2026, 3, "Europe/Helsinki")["totals"]["listens"], 2)

    def test_listens_without_a_name_become_the_apps_listeners(self):
        # The owner's: Koito's export brought in without a name, then the app scrobbling as "max".
        self.listens.add("", [self.row("2026-11-01T10:00:00"), self.row("2026-11-02T10:00:00")], source="koito")
        self.listens.add("max", [self.row("2026-11-02T10:00:00")])
        self.listens.connect_lastfm("", "key", "shh", "max", "right")
        self.listens.set_playing("max", {"artist": "Moby", "title": "Honey"})
        self.assertEqual([u["user"] for u in self.listens.users()], ["", "max"])
        self.assertEqual(self.listens.rename("", "max"), 1)  # the one max had already is kept once
        self.assertEqual([(u["user"], u["listens"]) for u in self.listens.users()], [("max", 2)])
        self.assertTrue(self.listens.lastfm_status("max")["connected"])  # sending on moved with them
        self.assertFalse(self.listens.relays(""))
        self.assertEqual(self.listens.rename("max", "max"), 0)

    def test_the_apps_listens_since_connecting_go_on_once_joined(self):
        # Last.fm was connected while the listens without a name were shown; the app's came in as "max" meanwhile.
        self.listens.add("", [self.row("2026-11-01T10:00:00")], source="koito")
        self.now = at("2026-11-11T10:30:00")
        self.listens.connect_lastfm("", "key", "shh", "max", "right")
        self.listens.add("max", [self.row("2026-11-11T10:00:00", "Before", "Connecting")])  # not sent: from before
        self.now = at("2026-11-11T12:00:00")
        self.listens.add("max", [self.row("2026-11-11T11:00:00", "Moby", "Honey")])
        self.assertEqual(self.listens.relay(), {"": 0})  # nothing of max's is sent to the account of ""
        self.listens.rename("", "max")
        self.assertEqual(self.listens.relay(), {"max": 1})
        self.assertEqual(self.lastfm.sent, [[(at("2026-11-11T11:00:00"), "Moby", "Honey")]])

    def test_hearts_go_on_to_last_fm(self):
        self.assertFalse(self.listens.love("max", "Moby", "Honey"))  # not connected: nothing to pass on to
        self.connect()
        self.assertTrue(self.listens.love("max", "Moby", "Honey"))
        self.assertTrue(self.listens.love("max", "Moby", "Honey", loved=False))
        self.assertFalse(self.listens.love("max", "", "Honey"))
        end = time.time() + 5
        while time.time() < end and len(self.lastfm.loved) < 2:
            time.sleep(0.02)
        self.assertEqual(sorted(self.lastfm.loved), [("Moby", "Honey", False), ("Moby", "Honey", True)])

    def test_the_app_listener_shows_before_it_has_a_listen(self):
        self.listens.add("", [self.row("2026-11-01T10:00:00")], source="koito")
        self.listens.set_playing("max", {"artist": "Moby", "title": "Honey", "duration": 200})
        self.assertEqual([u["user"] for u in self.listens.users()], ["", "max"])
        self.assertEqual([(p["user"], p["title"]) for p in self.listens.all_playing()], [("max", "Honey")])

    def test_whats_playing_shows_while_it_can_be(self):
        self.assertTrue(self.listens.set_playing("max", {"artist": "Moby", "title": "Honey", "duration": 200}))
        self.assertFalse(self.listens.set_playing("max", {"artist": "", "title": "Honey"}))
        self.assertEqual(self.listens.now_playing("max")["title"], "Honey")
        self.now += 300
        self.assertIsNone(self.listens.now_playing("max"))

    def connect(self):
        return self.listens.connect_lastfm("max", "key", "shh", "max", "right")

    def test_new_listens_go_on_to_last_fm_and_files_never(self):
        with self.assertRaises(L.LastFmError):
            self.listens.connect_lastfm("max", "key", "shh", "max", "wrong")
        with self.assertRaises(ValueError):
            self.listens.connect_lastfm("max", "key", "", "max", "right")
        self.listens.add("max", [self.row("2026-11-10T10:00:00")])  # before it was connected: not sent
        self.assertEqual(self.connect()["connected"], True)
        saved = json.dumps(self.store.setting("lastfm_relay"))
        self.assertNotIn("shh", saved)  # the keys are encrypted, the password isn't kept at all
        self.assertNotIn("right", saved)
        self.assertNotIn("session-max", saved)
        self.listens.add("max", [self.row("2026-11-11T10:00:00"), self.row("2026-11-11T10:05:00", "Moby", "Honey"),
                                 self.row("2026-10-01T10:00:00", "Old", "Listen")])  # the last one: over two weeks old
        self.listens.add("max", [self.row("2026-11-11T09:00:00", "From", "A file")], source="koito")
        self.listens.add("kids", [self.row("2026-11-11T11:00:00")])  # another listener's never go to max's account
        self.assertEqual(self.listens.lastfm_status("max")["waiting"], 3)
        self.assertEqual(self.listens.relay(), {"max": 2})
        self.assertEqual(self.lastfm.sent, [[(at("2026-11-11T10:00:00"), "Röyksopp", "Eple"), (at("2026-11-11T10:05:00"), "Moby", "Honey")]])
        status = self.listens.lastfm_status("max")
        self.assertEqual((status["waiting"], status["sent"], status["turned_away"], status["error"]), (0, 2, 1, None))
        self.assertEqual(self.listens.relay(), {"max": 0})  # nothing twice

    def test_listens_wait_while_last_fm_cant_be_reached(self):
        self.connect()
        self.listens.add("max", [self.row("2026-11-11T10:00:00")])
        self.lastfm.fail = L.LastFmError("Last.fm couldn't be reached: timed out")
        self.assertEqual(self.listens.relay(), {"max": 0})
        self.assertIn("tried again later", self.listens.lastfm_status("max")["error"])
        self.assertEqual(self.listens.lastfm_status("max")["waiting"], 1)
        self.lastfm.fail = L.LastFmError("Last.fm: Invalid session key - Please re-authenticate", 9)
        self.listens.relay()
        self.assertIn("connect Last.fm again", self.listens.lastfm_status("max")["error"])
        self.assertEqual(self.listens.lastfm_status("max")["waiting"], 1)
        self.lastfm.fail = None
        self.assertEqual(self.listens.relay(), {"max": 1})
        self.assertIsNone(self.listens.lastfm_status("max")["error"])
        # Disconnected: what's waiting isn't sent later.
        self.listens.add("max", [self.row("2026-11-11T11:00:00")])
        self.assertTrue(self.listens.disconnect_lastfm("max"))
        self.assertEqual(self.listens.relay(), {})
        self.assertEqual(self.listens.lastfm_status("max")["waiting"], 0)

    def test_whats_playing_is_said_to_last_fm(self):
        self.connect()
        self.listens.set_playing("max", {"artist": "Moby", "title": "Honey"})
        end = time.time() + 5
        while time.time() < end and not self.lastfm.playing:
            time.sleep(0.02)
        self.assertEqual(self.lastfm.playing, [("Moby", "Honey")])

    def test_last_fms_requests_are_signed(self):
        seen = {}

        class Answer:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return b'{"scrobbles": {"@attr": {"accepted": 1}}}'

        def opener(request, timeout=None):
            seen.update(urllib.parse.parse_qsl(request.data.decode()))
            return Answer()

        L.LastFm(opener).scrobble({"api_key": "key", "secret": "shh", "session": "sk"},
                                  [{"at": 1, "artist": "Moby", "title": "Honey", "album": "Play", "duration": 200}])
        import hashlib
        signed = {k: v for k, v in seen.items() if k not in ("api_sig", "format")}
        self.assertEqual(seen["api_sig"], hashlib.md5(("".join(k + signed[k] for k in sorted(signed)) + "shh").encode()).hexdigest())
        self.assertEqual((seen["artist[0]"], seen["track[0]"], seen["timestamp[0]"], seen["method"]), ("Moby", "Honey", "1", "track.scrobble"))


class ListeningService(Service):
    """Through the running service: the app's scrobbles, a file brought in on the page, the numbers."""

    def started(self):
        self.start()
        self.call("/api/libraries", "POST", {"ids": ["1"]})
        self.wait_done()
        self.lastfm = self.app.listens.lastfm = FakeLastFm()

    def test_the_app_scrobbles_and_the_page_shows_it(self):
        self.started()
        caps = self.call("/api/v1/capabilities")["features"]["scrobbling"]
        self.assertEqual((caps["available"], caps["listens"]), (True, 0))
        now = int(time.time())
        sent = self.call("/api/v1/listens", "POST", {"user": "max", "listens": [
            {"id": "t140", "at": now - 600, "artist": "Ecsape", "title": "Control", "album": "Test", "duration": 60},
            {"id": "t140", "at": now - 300, "artist": "Ecsape", "title": "Control", "duration": 60},
            {"id": "not-here", "at": now - 100, "artist": "Someone", "title": "Elsewhere"}]})
        self.assertEqual(sent, {"kept": 3, "known": 0, "skipped": 0})
        self.assertEqual(self.call("/api/v1/listens", "POST", {"user": "max", "listens": [
            {"id": "t140", "at": now - 600, "artist": "Ecsape", "title": "Control"}]})["known"], 1)  # sent again after a failure
        with self.assertRaises(urllib.error.HTTPError) as e:
            self.call("/api/v1/listens", "POST", {"user": "max", "listens": "all of them"})
        self.assertEqual(e.exception.code, 400)
        self.call("/api/v1/listens/now", "POST", {"user": "max", "artist": "House Test", "title": "Deep", "duration": 50})

        top = self.call("/api/v1/listens/top?user=max&kind=songs")
        self.assertEqual([(t["title"], t["listens"], t["id"]) for t in top["top"]], [("Control", 2, "t140"), ("Elsewhere", 1, None)])
        self.assertEqual(self.call(f"/api/v1/listens/top?user=max&kind=artists&from={now - 200}")["top"], [{"artist": "Someone", "listens": 1}])
        page = self.call("/api/listens?tz=Europe/Helsinki")  # the first listener's, all time
        self.assertEqual((page["user"], page["totals"]["listens"], page["now"]["title"]), ("max", 3, "Deep"))
        self.assertEqual([u["user"] for u in page["users"]], ["max"])
        self.assertEqual(len(page["recent"]), 3)
        self.assertEqual(self.call("/api/v1/capabilities")["features"]["scrobbling"]["listens"], 3)

    def test_a_file_is_brought_in_on_the_page(self):
        self.started()
        export = json.dumps({"version": "1", "user": "max", "listens": [
            {"listened_at": "2024-05-01T10:00:00Z", "client": "x", "track": {"duration": 60, "aliases": [{"alias": "Control", "is_primary": True}]},
             "album": {"aliases": [{"alias": "Test", "is_primary": True}]}, "artists": [{"is_primary": True, "aliases": [{"alias": "Ecsape", "is_primary": True}]}]},
            {"listened_at": "2024-05-02T10:00:00Z", "client": "x", "track": {"duration": 60, "aliases": [{"alias": "Unknown song", "is_primary": True}]},
             "album": {"aliases": []}, "artists": [{"is_primary": True, "aliases": [{"alias": "Nobody", "is_primary": True}]}]}]}).encode()

        def upload(data, name="koito_export.json"):
            request = urllib.request.Request(self.base + "/api/listens/import?user=max&name=" + name, data=data, method="POST")
            with urllib.request.urlopen(request, timeout=10) as r:
                json.loads(r.read())
            end = time.time() + 10
            while time.time() < end:
                state = self.call("/api/listens/import")
                if not state.get("running"):
                    return state
                time.sleep(0.05)
            self.fail("the file wasn't brought in")

        state = upload(export)
        self.assertEqual((state["format"], state["kept"], state["known"], state.get("error")), ("koito", 2, 0, None))
        self.assertEqual(upload(export)["known"], 2)  # the same file again adds nothing
        self.assertIn("isn't an export", upload(b'{"hello": 1}', "other.json")["error"])
        self.assertEqual(os.listdir(os.path.join(self.tmp, "data", "tmp")), [])  # the files aren't kept
        may = self.call("/api/listens?user=max&year=2024&month=5&tz=UTC")
        self.assertEqual((may["period"]["label"], may["totals"]["listens"]), ("May 2024", 2))
        # A listen from a file is linked to the song here by artist and title, so the app gets an id to play.
        top = self.call("/api/v1/listens/top?user=max")["top"]
        self.assertEqual({t["title"]: t["id"] for t in top}, {"Control": "t140", "Unknown song": None})
        self.assertEqual(self.call("/api/listens/remove", "POST", {"user": "max", "source": "koito"}), {"removed": 2})

    def test_last_fm_is_connected_on_the_page_without_showing_a_key(self):
        self.started()
        with self.assertRaises(urllib.error.HTTPError) as e:
            self.call("/api/listens/lastfm", "POST", {"user": "max", "api_key": "key", "secret": "shh", "name": "max", "password": "wrong"})
        self.assertEqual(e.exception.code, 502)
        status = self.call("/api/listens/lastfm", "POST", {"user": "max", "api_key": "key", "secret": "shh", "name": "max", "password": "right"})
        self.assertEqual((status["connected"], status["name"]), (True, "max"))
        self.call("/api/v1/listens", "POST", {"user": "max", "listens": [{"at": int(time.time()) - 60, "artist": "Moby", "title": "Honey"}]})
        self.assertEqual(self.app.listens.relay(), {"max": 1})
        page = json.dumps(self.call("/api/listens?user=max"))
        self.assertIn('"sent": 1', page)
        for secret in ("shh", "session-max", "right"):
            self.assertNotIn(secret, page)
        _, report = self.call("/api/debug", raw=True)
        for secret in ("shh", "session-max"):
            self.assertNotIn(secret, report.decode())
        self.assertEqual(self.call("/api/v1/loves", "POST", {"user": "max", "artist": "Moby", "title": "Honey", "loved": True}), {"passed_on": True})
        self.assertEqual(self.call("/api/v1/loves", "POST", {"user": "kids", "artist": "Moby", "title": "Honey"}), {"passed_on": False})
        self.assertFalse(self.call("/api/listens/lastfm", "POST", {"user": "max", "disconnect": True})["connected"])


if __name__ == "__main__":
    unittest.main()
