"""Start analysis checks everything first (0.40): the server, the libraries, the database, the disk, the models and
the page's safety; a song's download has a size cap on every server."""

import io
import os
import urllib.error
import shutil
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from beats import checks, subsonic  # noqa: E402
from beats.mediabrowser import MediaBrowser  # noqa: E402
from beats.store import Store  # noqa: E402
from beats.subsonic import ServerError, Subsonic  # noqa: E402
from fakes import PASSWORD, USER, FakeMediaServer  # noqa: E402
from test_service import Base, FakeStyle, Service  # noqa: E402


class PrivateAddressTest(unittest.TestCase):
    def test_a_home_address_is_told_from_a_public_one(self):
        for url in ("http://192.168.1.10:4533", "http://10.0.0.2", "http://172.16.5.5", "http://localhost:4533",
                    "http://navidrome:4533", "http://nas.local", "http://music.lan", "http://127.0.0.1", "http://[::1]:4533"):
            self.assertTrue(checks.is_private(url), url)
        for url in ("http://music.example.com", "http://8.8.8.8:4533", "http://my-nas.duckdns.org"):
            self.assertFalse(checks.is_private(url), url)


class FingerprintTest(unittest.TestCase):
    def test_every_model_file_is_hashed(self):
        folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, folder, True)
        with open(os.path.join(folder, "a.onnx"), "wb") as f:
            f.write(b"abc")
        os.makedirs(os.path.join(folder, "sub"))
        found = checks.fingerprints(folder)
        self.assertEqual(list(found), ["a.onnx"])
        self.assertEqual(found["a.onnx"]["size"], 3)
        self.assertEqual(found["a.onnx"]["sha256"], "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        self.assertEqual(checks.fingerprints(os.path.join(folder, "none")), {})


class IntegrityTest(unittest.TestCase):
    def test_quick_check_and_the_rows_of_gone_songs(self):
        folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, folder, True)
        db = Store(os.path.join(folder, "beats.db"))
        self.addCleanup(db.close)
        db.add_song({"id": "a", "title": "T", "artist": "A", "album": "B", "duration": 60.0, "suffix": "flac",
                     "tag_bpm": None, "genres": ["Techno"]}, folder="1")
        with db._write():
            db.db.execute("INSERT INTO genres VALUES ('ghost', 'tag', 'Pop', 1.0)")
            db.db.execute("INSERT INTO jobs (song_id, state) VALUES ('ghost', 'queued')")
            db.db.execute("INSERT INTO marks (song_id, seconds, created_at) VALUES ('ghost', 1.0, 1.0)")
        found = db.integrity()
        self.assertEqual(found["quick_check"], "ok")
        self.assertEqual(found["orphans"], {"genres": 1, "jobs": 1, "marks": 1})
        self.assertGreater(found["size"], 0)
        self.assertEqual(db.integrity()["orphans"], {})  # tidy now
        self.assertEqual([g["label"] for g in db.overview("final0", song_id="a")[0]["genres"]], ["Techno"])  # a's rows stay


class DownloadCapTest(Base):
    """A song's file is never read past the cap, on every server (0.40)."""

    def test_a_file_the_server_says_is_too_big_is_refused_on_every_server(self):
        was = subsonic.MOST_SONG_BYTES
        self.addCleanup(setattr, subsonic, "MOST_SONG_BYTES", was)
        clients = [(Subsonic(self.fake.url, USER, PASSWORD), "t140")]
        for jellyfin in (False, True):
            media = FakeMediaServer(self.music, jellyfin=jellyfin).start()
            self.addCleanup(media.stop)
            clients.append((MediaBrowser(media.url, USER, PASSWORD, jellyfin=jellyfin), media.prefix + "t140"))
        for client, song_id in clients:
            with self.subTest(client.label):
                target = os.path.join(self.tmp, "song")
                subsonic.MOST_SONG_BYTES = 4
                self.assertEqual(client.download(song_id, target), 4)  # just fits
                subsonic.MOST_SONG_BYTES = 3
                with self.assertRaisesRegex(ServerError, "says the file is 0 MB, over the 0 MB"):
                    client.download(song_id, target)

    def test_a_stream_without_a_length_is_cut_off_at_the_cap(self):
        was = subsonic.MOST_SONG_BYTES
        self.addCleanup(setattr, subsonic, "MOST_SONG_BYTES", was)
        subsonic.MOST_SONG_BYTES = 5

        class Endless:
            headers = {}

            def read(self, n):
                return b"x" * n

        out = io.BytesIO()
        with self.assertRaisesRegex(ServerError, "goes past"):
            subsonic.save(Endless(), out, "s", "Navidrome")
        self.assertLessEqual(len(out.getvalue()), 5)

        class Short:
            headers = {"Content-Length": "3"}
            data = io.BytesIO(b"abc")

            def read(self, n):
                return self.data.read(n)

        out = io.BytesIO()
        self.assertEqual(subsonic.save(Short(), out), 3)
        self.assertEqual(out.getvalue(), b"abc")


class StartAnalysisTest(Service):
    def test_start_analysis_checks_everything_then_reads_the_ticked_libraries(self):
        """0.40 (the owner, 13 Nov): one button. The checks come back as a list; with none failed the libraries are
        chosen and read for new songs in the background; the report is kept for the page."""
        models = os.path.join(self.tmp, "models")
        self.start(style=FakeStyle(models))
        data = self.app.config.data_dir
        stale = os.path.join(data, "song-old")
        with open(stale, "wb") as f:
            f.write(b"x" * 10)
        os.utime(stale, (time.time() - 7200, time.time() - 7200))
        fresh = os.path.join(data, "song-new")
        with open(fresh, "wb") as f:
            f.write(b"x")
        r = self.call("/api/analysis/start", "POST", {"ids": ["1"]})
        self.assertTrue(r["ok"])
        self.assertTrue(r["reading"])
        by = {c["id"]: c for c in r["checks"]}
        self.assertEqual(list(by), ["server", "libraries", "database", "space", "models", "access"])
        self.assertEqual(by["server"]["state"], "ok")  # a 127.0.0.1 address over http is fine
        self.assertIn("Nothing is ever written", by["server"]["detail"])
        self.assertEqual((by["libraries"]["state"], by["database"]["state"], by["space"]["state"]), ("ok", "ok", "ok"))
        self.assertIn("Music", by["libraries"]["detail"])
        self.assertIn("1 temporary files of a run that was cut off removed", by["space"]["detail"])
        self.assertFalse(os.path.exists(stale))
        self.assertTrue(os.path.exists(fresh))  # one in use stays
        self.assertEqual(by["models"]["state"], "warn")  # the test image has no sound models
        self.assertIn("fingerprinted", by["models"]["detail"])
        self.assertEqual(by["access"]["state"], "warn")  # no password
        self.assertIn("no password", by["access"]["detail"])
        self.wait_done()
        self.assertEqual(self.call("/api/status")["libraries"]["chosen"], ["1"])
        kept = self.call("/api/analysis/checks")
        self.assertEqual([c["id"] for c in kept["checks"]], list(by))
        self.assertIn("Checked before analysing", " ".join(e["message"] for e in self.call("/api/status")["events"]))
        # Pressed again: the models are compared with their fingerprints, and new songs looked for.
        r = self.call("/api/analysis/start", "POST", {"ids": ["1"]})
        self.assertIn("unchanged since first checked", {c["id"]: c for c in r["checks"]}["models"]["detail"])

    def test_a_failed_check_starts_nothing(self):
        models = os.path.join(self.tmp, "models")
        self.start(style=FakeStyle(models))
        # An unknown library.
        r = self.call("/api/analysis/start", "POST", {"ids": ["9"]})
        self.assertFalse(r["ok"])
        self.assertFalse(r["reading"])
        self.assertEqual({c["id"]: c["state"] for c in r["checks"]}["libraries"], "fail")
        self.assertEqual(self.call("/api/status")["libraries"]["chosen"], [])
        # A model file changed since it was fingerprinted.
        self.call("/api/analysis/start", "POST", {"ids": ["1"]})
        with open(os.path.join(models, "discogs-effnet-bs64-1.json"), "a") as f:
            f.write(" ")
        r = self.call("/api/analysis/start", "POST", {"ids": ["1"]})
        self.assertFalse(r["ok"])
        models_check = {c["id"]: c for c in r["checks"]}["models"]
        self.assertEqual(models_check["state"], "fail")
        self.assertIn("discogs-effnet-bs64-1.json", models_check["detail"])
        self.assertIn("pull the image again", models_check["detail"])

    def test_a_server_that_is_away_fails_the_first_check(self):
        self.start(style=FakeStyle(os.path.join(self.tmp, "models")))
        self.fake.stop()
        r = self.call("/api/analysis/start", "POST", {"ids": ["1"]})
        self.assertFalse(r["ok"])
        by = {c["id"]: c for c in r["checks"]}
        self.assertEqual(by["server"]["state"], "fail")
        self.assertIn("doesn't answer", by["server"]["detail"])
        self.assertEqual(by["libraries"]["state"], "fail")
        self.fake = type(self.fake)(self.music).start()  # for tearDown


class LibraryTasksTest(Service):
    """What a library gets (0.40): a library of two-hour mixes wants its loudness, not its beats."""

    def test_a_library_left_out_of_the_beats_still_gets_its_loudness(self):
        for song in self.fake.songs:
            song["library"] = "2" if song["id"] in ("t96w", "t122") else "1"  # 2: the mixes
        self.start(style=FakeStyle(os.path.join(self.tmp, "models")))
        r = self.call("/api/analysis/start", "POST", {"ids": ["1", "2"], "skips": {"2": ["beats", "sound", "styles"]}})
        self.assertTrue(r["ok"])
        end = time.time() + 20
        while time.time() < end:
            counts = self.call("/api/status")["counts"]
            if counts.get("done", 0) + counts.get("failed", 0) == 2 and counts.get("skipped") == 2:
                break
            time.sleep(0.1)
        counts = self.call("/api/status")["counts"]
        self.assertEqual((counts.get("done"), counts.get("failed"), counts.get("skipped")), (1, 1, 2))
        libraries = {l["id"]: l for l in self.call("/api/libraries")}
        self.assertEqual(libraries["2"]["skip"], ["beats", "sound", "styles"])
        self.assertEqual(libraries["1"]["skip"], [])
        self.assertEqual((libraries["2"]["songs"], libraries["2"]["done"]), (2, 2))  # settled, by choice
        self.assertEqual({s["id"]: s["state"] for s in self.call("/api/songs")}["t122"], "skipped")
        self.assertIn("left out of beats, sound, genres by sound", " ".join(e["message"] for e in self.call("/api/status")["events"]))
        # The loudness is measured all the same, by the measure loop, beats or no beats.
        end = time.time() + 20
        while time.time() < end and self.call("/api/status")["loudness"]["waiting"]:
            time.sleep(0.2)
        measured = self.call("/api/status")["loudness"]
        self.assertEqual(measured["waiting"], 0)
        self.assertGreaterEqual(measured["measured"] + measured["failed"], 3)  # t140, t122, t96w
        self.assertIsNotNone(self.call("/api/songs/t122")["measure"])
        # Analyse again doesn't force the beats on a library that doesn't want them.
        self.assertEqual(self.call("/api/analyse-all", "POST")["queued"], 2)  # t140 and the failed one; not the mixes
        self.wait_done()
        # Put the beats back: the two join the queue and are analysed.
        r = self.call("/api/analysis/start", "POST", {"ids": ["1", "2"], "skips": {}})
        self.assertTrue(r["ok"])
        self.wait_done()
        counts = self.call("/api/status")["counts"]
        self.assertEqual((counts.get("done"), counts.get("skipped", 0)), (3, 0))
        self.assertIn("in everything again (2 songs joined the queue)", " ".join(e["message"] for e in self.call("/api/status")["events"]))

    def test_a_song_in_two_libraries_is_left_out_only_when_both_leave_it_out(self):
        from beats.store import Store

        db = Store(os.path.join(self.tmp, "two.db"))
        self.addCleanup(db.close)
        song = {"id": "x", "title": "T", "artist": "A", "album": "B", "duration": 60.0, "suffix": "flac", "tag_bpm": None, "genres": []}
        db.link_songs("main", "1", [song])
        db.link_songs("emby", "2", [song])  # the same song on a second server (a song is in one library of a server)
        self.assertEqual(db.set_skips("main", "1", ["beats"]), None)  # the other library still wants them
        self.assertEqual(db.wanted("x"), {"beats", "loudness", "sound", "styles"})
        self.assertEqual(db.set_skips("emby", "2", ["beats", "sound"]), {"skipped": 1, "queued": 0})
        self.assertEqual(db.wanted("x"), {"loudness", "sound", "styles"})  # sound: the first library still wants it
        self.assertEqual(db.set_skips("main", "1", []), {"skipped": 0, "queued": 1})
        self.assertEqual(db.skips("emby"), {"2": ["beats", "sound"]})
        self.assertFalse(db.set_skips("emby", "2", ["nonsense"]))  # unknown tasks are ignored
        self.assertEqual(db.skips("emby"), {})


class TidyTest(Service):
    """Tidying the database (0.40): looked at first, removed as shown, a copy kept."""

    def test_what_the_server_no_longer_has_is_shown_then_removed_with_the_looks_token(self):
        for song in self.fake.songs:
            song["library"] = "2" if song["id"] in ("t96w",) else "1"
        self.start(style=FakeStyle(os.path.join(self.tmp, "models")))
        self.call("/api/libraries", "POST", {"ids": ["1", "2"]})
        self.wait_done()
        self.call("/api/songs/t122/marks", "POST", {"seconds": 12.0})
        look = self.call("/api/clean/preview")
        self.assertEqual(look["total"], 1)  # the failed song alone
        by = {g["id"]: g for g in look["groups"]}
        self.assertEqual(by["failed"]["names"], ["Nobody – Broken"])
        self.assertEqual(by["gone:1"]["count"], 0)
        # Now library 2 is unticked, and a song goes from the server (a read would tidy one song by itself; the
        # look doesn't read, so it shows what's gone).
        self.call("/api/libraries", "POST", {"ids": ["1"]})
        self.fake.songs[:] = [s for s in self.fake.songs if s["id"] != "t122"]
        look = self.call("/api/clean/preview")
        by = {g["id"]: g for g in look["groups"]}
        self.assertEqual([g["id"] for g in look["groups"]], ["gone:1", "left-out:2", "failed"])
        self.assertEqual((by["gone:1"]["count"], by["gone:1"]["names"]), (1, ["House Test – Deep"]))
        self.assertEqual((by["left-out:2"]["count"], by["left-out:2"]["names"]), (1, ["Three – Waltz"]))
        self.assertIsNone(by["gone:1"]["warning"])
        self.assertEqual(sorted(s["id"] for s in self.call("/api/songs")), ["bad", "t122", "t140", "t96w"])  # looking removed nothing
        # A wrong token, a group the look didn't show, then the real thing.
        with self.assertRaises(urllib.error.HTTPError) as e:
            self.call("/api/clean", "POST", {"token": "nope", "groups": ["gone:1"]})
        self.assertEqual(e.exception.code, 409)
        with self.assertRaises(urllib.error.HTTPError) as e:
            self.call("/api/clean", "POST", {"token": look["token"], "groups": ["gone:9"]})
        self.assertEqual(e.exception.code, 409)
        self.assertEqual(sorted(s["id"] for s in self.call("/api/songs")), ["bad", "t122", "t140", "t96w"])
        look = self.call("/api/clean/preview")  # a failed try spends the token: look again
        r = self.call("/api/clean", "POST", {"token": look["token"], "groups": ["gone:1", "left-out:2"]})
        self.assertEqual((r["removed"], r["groups"]), (2, {"gone:1": 1, "left-out:2": 1}))
        self.assertEqual(r["backup"], "beats-before-clean.db")
        self.assertEqual(r["integrity"]["quick_check"], "ok")
        self.assertEqual(sorted(s["id"] for s in self.call("/api/songs")), ["bad", "t140"])
        self.assertEqual(self.call("/api/grids?ids=t122")["missing"], ["t122"])
        # The copy is the database from before, whole.
        import sqlite3

        copy = sqlite3.connect(os.path.join(self.app.config.data_dir, "beats-before-clean.db"))
        self.assertEqual(copy.execute("SELECT COUNT(*) FROM songs").fetchone()[0], 4)
        self.assertEqual(copy.execute("SELECT COUNT(*) FROM marks WHERE song_id='t122'").fetchone()[0], 1)
        copy.close()
        # The same token twice removes nothing more.
        with self.assertRaises(urllib.error.HTTPError) as e:
            self.call("/api/clean", "POST", {"token": look["token"], "groups": ["gone:1"]})
        self.assertEqual(e.exception.code, 409)
        self.assertIn("Tidied the database: 2 songs removed", " ".join(e["message"] for e in self.call("/api/status")["events"]))

    def test_a_library_the_server_suddenly_lists_half_empty_is_warned_about(self):
        self.start(style=FakeStyle(os.path.join(self.tmp, "models")))
        self.call("/api/libraries", "POST", {"ids": ["1"]})
        self.wait_done()
        self.fake.songs[:] = self.fake.songs[:1]
        look = self.call("/api/clean/preview")
        gone = {g["id"]: g for g in look["groups"]}["gone:1"]
        self.assertEqual(gone["count"], 3)
        self.assertIn("may still be scanning", gone["warning"])


if __name__ == "__main__":
    unittest.main()
