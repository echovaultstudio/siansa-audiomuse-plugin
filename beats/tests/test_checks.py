"""Start analysis checks everything first (0.40): the server, the libraries, the database, the disk, the models and
the page's safety; a song's download has a size cap on every server."""

import io
import os
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


if __name__ == "__main__":
    unittest.main()
