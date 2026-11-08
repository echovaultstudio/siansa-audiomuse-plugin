"""The service end to end without the model: Navidrome and AudioMuse-AI are tests/fakes.py, and a stand-in tracker
gives each song the beats of its known tempo. The model itself is checked by e2e.py in the Docker image."""

import base64
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from beats import grid  # noqa: E402
from beats.audiomuse import AudioMuse, AudioMuseError  # noqa: E402
from beats.server import App, Config, make_server  # noqa: E402
from beats.store import Store  # noqa: E402
from beats.subsonic import Subsonic, SubsonicError  # noqa: E402
from fakes import PASSWORD, TOKEN, USER, FakeServer  # noqa: E402

CATALOG = [
    {"id": "t140", "title": "Control", "artist": "Ecsape", "genre": "Trance", "bpm": 140, "duration": 60,
     "expect_bpm": 140.0, "expect_bar": 4, "plugin": {"intro_bpm": 187.1, "outro_bpm": 140.1}},
    {"id": "t122", "title": "Deep", "artist": "House Test", "genre": "House", "bpm": 61, "duration": 50,
     "expect_bpm": 122.0, "expect_bar": 4, "plugin": None},
    {"id": "t96w", "title": "Waltz", "artist": "Three", "genre": "Pop", "bpm": None, "duration": 40,
     "expect_bpm": 96.0, "expect_bar": 3, "plugin": None},
    {"id": "bad", "title": "Broken", "artist": "Nobody", "genre": "", "bpm": None, "duration": 30,
     "expect_bpm": None, "expect_bar": None, "plugin": None},
]


class Samples:
    """What the stand-in decode gives: the song's id, and its length in samples."""

    def __init__(self, song_id, seconds):
        self.song_id, self.seconds = song_id, seconds

    def __len__(self):
        return int(self.seconds * 22_050)


def fake_decode(path):
    with open(path) as f:
        song_id = f.read().strip()
    song = next(s for s in CATALOG if s["id"] == song_id)
    if song_id == "bad":
        raise RuntimeError("ffmpeg couldn't read the song: invalid data")
    return Samples(song_id, song["duration"])


class FakeTracker:
    loaded = True
    load_seconds = 0.1

    def __init__(self):
        self.calls = 0
        self.delay = 0.0

    def __call__(self, samples):
        self.calls += 1
        time.sleep(self.delay)
        song = next(s for s in CATALOG if s["id"] == samples.song_id)
        period = 60.0 / song["expect_bpm"]
        beats = [round((0.3 + i * period) / 0.02) * 0.02 for i in range(int((song["duration"] - 0.5) / period))]
        return beats, beats[::song["expect_bar"]]


def make_library(folder):
    for song in CATALOG:
        song["file"] = f"{song['id']}.flac"
        with open(os.path.join(folder, song["file"]), "w") as f:
            f.write(song["id"])
    with open(os.path.join(folder, "catalog.json"), "w") as f:
        json.dump(CATALOG, f)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.music = os.path.join(self.tmp, "music")
        os.makedirs(self.music)
        make_library(self.music)
        self.fake = FakeServer(self.music).start()

    def tearDown(self):
        self.fake.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)


class SubsonicTest(Base):
    def test_reads_libraries_songs_and_files_with_a_token_login(self):
        nd = Subsonic(self.fake.url, USER, PASSWORD)
        self.assertEqual(nd.ping()["type"], "navidrome")
        self.assertEqual(nd.libraries(), [{"id": "1", "name": "Music"}, {"id": "2", "name": "Compilations"}])
        songs = nd.all_songs("1")
        self.assertEqual([s["id"] for s in songs], ["t140", "t122", "t96w", "bad"])
        self.assertEqual(songs[0]["tag_bpm"], 140.0)
        self.assertEqual(songs[0]["genres"], ["Trance"])
        self.assertIsNone(songs[2]["tag_bpm"])
        self.assertEqual(nd.song("t122")["title"], "Deep")
        path = os.path.join(self.tmp, "x")
        self.assertEqual(nd.download("t140", path), 4)
        with open(path) as f:
            self.assertEqual(f.read(), "t140")
        # The password itself is never sent: only a salted token.
        self.assertTrue(all("p" not in q and q["u"] == USER and "t" in q for _, q in self.fake.requests
                            if _.startswith("/rest/")))

    def test_errors_say_what_went_wrong(self):
        with self.assertRaisesRegex(SubsonicError, "Wrong username or password"):
            Subsonic(self.fake.url, USER, "wrong").ping()
        with self.assertRaisesRegex(SubsonicError, "isn't set up"):
            Subsonic("", "", "").ping()
        with self.assertRaisesRegex(SubsonicError, "getSong: Song not found"):
            Subsonic(self.fake.url, USER, PASSWORD).song("missing")
        with self.assertRaises(SubsonicError):
            Subsonic("http://127.0.0.1:9", USER, PASSWORD, timeout=2).ping()

    def test_genres_from_a_single_genre_field(self):
        from beats.subsonic import song_from

        self.assertEqual(song_from({"id": "a", "genre": "Electronic/Trance; Dance"})["genres"],
                         ["Electronic", "Trance", "Dance"])
        self.assertEqual(song_from({"id": "a", "genre": "Pop", "genres": [{"name": "Pop"}, {"name": "Rock"}]})["genres"],
                         ["Pop", "Rock"])


class AudioMuseTest(Base):
    def test_reads_the_plugins_profiles_with_a_token(self):
        am = AudioMuse(self.fake.url, TOKEN)
        found = am.profiles(["t140", "t122"])
        self.assertEqual(found["t140"]["intro_bpm"], 187.1)
        self.assertNotIn("t122", found)
        self.assertEqual(am.health()["plugin_version"], "0.5.7-fake")

    def test_many_songs_are_asked_in_requests_short_enough_for_audiomuse(self):
        # Navidrome's ids are 22 characters: 200 of them made a request line AudioMuse-AI turned away (HTTP 400).
        ids = ["t140"] + [f"{n:022d}" for n in range(600)]
        found = AudioMuse(self.fake.url, TOKEN).profiles(ids)
        self.assertEqual(found["t140"]["intro_bpm"], 187.1)
        asked = [q["ids"].split(",") for path, q in self.fake.requests if path.endswith("/api/profiles")]
        self.assertGreater(len(asked), 3)
        self.assertEqual(sum(len(a) for a in asked), len(ids))

    def test_batches_keep_to_the_address_budget_and_the_id_limit(self):
        from beats.audiomuse import PROFILES, batches
        for chunk in batches([f"{n:022d}" for n in range(1000)]):
            self.assertLessEqual(len(PROFILES + "?ids=" + ",".join(chunk)), 3500)
        self.assertEqual([len(c) for c in batches([str(n) for n in range(1200)], budget=10 ** 6)], [500, 500, 200])
        self.assertEqual(batches(["a", "", "a", "b"]), [["a", "b"]])

    def test_problems_say_what_to_check(self):
        with self.assertRaisesRegex(AudioMuseError, "http://"):
            AudioMuse("audiomuse:8000").health()
        with self.assertRaisesRegex(AudioMuseError, "Nothing answers.*container itself"):
            AudioMuse("http://127.0.0.1:9", timeout=3).health()
        with self.assertRaisesRegex(AudioMuseError, "set AUDIOMUSE_TOKEN"):
            AudioMuse(self.fake.url).health()
        with self.assertRaisesRegex(AudioMuseError, "HTTP 404: No Siansa plugin there"):
            AudioMuse(self.fake.url + "/audiomuse", TOKEN).health()

    def test_a_wrong_token_says_so(self):
        with self.assertRaisesRegex(AudioMuseError, "AUDIOMUSE_TOKEN"):
            AudioMuse(self.fake.url, "wrong").profiles(["t140"])
        self.assertFalse(AudioMuse("").configured)


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "beats.db")
        self.opened = []

    def tearDown(self):
        for db in self.opened:
            db.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def store(self):
        db = Store(self.path)
        self.opened.append(db)
        return db

    def song(self, sid="a"):
        return {"id": sid, "title": "T", "artist": "A", "album": "B", "duration": 60.0, "suffix": "flac",
                "tag_bpm": 128.0, "genres": ["Techno", "Electronic"]}

    def test_the_queue_and_its_states(self):
        db = self.store()
        db.add_song(self.song(), folder="1")
        self.assertTrue(db.queue("a"))
        self.assertFalse(db.queue("a"))  # already queued
        self.assertEqual(db.next_queued(), "a")
        db.set_state("a", "analysing")
        self.assertIsNone(db.next_queued())
        self.assertFalse(db.queue("a", force=True))  # being worked on
        db.set_state("a", "done")
        self.assertFalse(db.queue("a"))  # done stays done…
        self.assertTrue(db.queue("a", force=True))  # …unless asked again
        self.assertEqual(db.counts(), {"queued": 1, "songs": 1})

    def test_libraries_queue_in_turn_and_one_worker_claims_each_song(self):
        db = self.store()
        self.assertEqual(db.add_songs([self.song("m1"), self.song("m2")], "1"), ["m1", "m2"])
        self.assertEqual(db.add_songs([self.song("c1")], "2"), ["c1"])
        db.add_song(self.song("old"))  # from the playlist of 0.9.0: no library yet
        self.assertEqual(db.add_songs([self.song("m1")], "1"), [])  # nothing new
        self.assertEqual(db.claim_next(), "m1")
        self.assertEqual(db.counts()["downloading"], 1)
        self.assertEqual(db.unqueue_except(["1"]), 1)  # c1; m1 is being worked on, m2 is in library 1
        self.assertEqual(db.claim_next(), "m2")
        self.assertIsNone(db.claim_next())
        db.set_state("m1", "done")
        self.assertEqual(db.folder_counts(), {"1": {"songs": 2, "done": 1}, "2": {"songs": 1, "done": 0}})
        db.add_song(self.song("old"), folder="2")  # found in a library later
        self.assertEqual(db.song("old")["folder"], "2")

    def test_the_table_finds_and_limits(self):
        db = self.store()
        db.add_songs([dict(self.song("a"), title="Stars Align"), dict(self.song("b"), genres=["Drum & Bass"]),
                      dict(self.song("c"), title="100%_sure"), self.song("d")], "1")
        ids = lambda **k: [r["id"] for r in db.overview("final0", **k)]  # noqa: E731
        self.assertEqual(sorted(ids()), ["a", "b", "c", "d"])
        self.assertEqual(ids(find="stars ALIGN"), ["a"])
        self.assertEqual(ids(find="drum"), ["b"])  # genres too
        self.assertEqual(ids(find="%_"), ["c"])  # as written, not as a pattern
        self.assertEqual(len(ids(limit=2)), 2)
        db.set_state("b", "failed", "no")
        self.assertEqual(ids(state="failed"), ["b"])
        self.assertEqual(db.overview("final0", song_id="b")[0]["genres"][0]["label"], "Drum & Bass")

    def test_settings_are_kept(self):
        db = self.store()
        self.assertIsNone(db.setting("library"))
        db.set_setting("library", True)
        db.close()
        self.opened.remove(db)
        self.assertTrue(self.store().setting("library"))

    def test_work_cut_off_by_a_restart_goes_back_in_the_queue(self):
        db = self.store()
        db.add_song(self.song())
        db.queue("a")
        db.set_state("a", "downloading")
        db.close()
        self.opened.remove(db)
        self.assertEqual(self.store().next_queued(), "a")

    def test_results_genres_and_removal(self):
        db = self.store()
        db.add_song(self.song())
        db.set_genres("a", "model", ["Trance"], [0.8])
        self.assertEqual([(g["source"], g["label"]) for g in db.genres("a")],
                         [("model", "Trance"), ("tag", "Electronic"), ("tag", "Techno")])
        db.save_result("a", "final0", {"bpm": 128.01, "beats_per_bar": 4}, [0.5, 0.97], [0.5])
        self.assertEqual(db.result("a", "final0", with_beats=True)["beats"], [0.5, 0.97])
        self.assertNotIn("beats", db.result("a", "final0"))
        self.assertIsNone(db.result("a", "small0"))
        row = db.overview("final0")[0]
        self.assertEqual(row["result"]["bpm"], 128.01)
        self.assertIsNone(row["plugin"])
        db.save_plugin("a", {"intro_bpm": 128.0})
        self.assertEqual(db.overview("final0")[0]["plugin"]["intro_bpm"], 128.0)
        db.remove("a")
        self.assertEqual(db.overview("final0"), [])
        self.assertEqual(db.genres("a"), [])

    def test_events_are_kept_to_a_limit(self):
        import beats.store as store

        db = self.store()
        for i in range(store.EVENTS_KEPT + 20):
            db.event(f"e{i}")
        self.assertEqual(len(db.events(limit=10_000)), store.EVENTS_KEPT)
        self.assertEqual(db.events(limit=1)[0]["message"], f"e{store.EVENTS_KEPT + 19}")


class ServiceTest(Base):
    def start(self, password=""):
        env = {"NAVIDROME_URL": self.fake.url, "NAVIDROME_USER": USER, "NAVIDROME_PASSWORD": PASSWORD,
               "AUDIOMUSE_URL": self.fake.url, "AUDIOMUSE_TOKEN": TOKEN, "DATA_DIR": os.path.join(self.tmp, "data"),
               "BEATS_PASSWORD": password, "PORT": "0"}
        self.tracker = FakeTracker()
        self.app = App(Config(env), tracker=self.tracker, decode=fake_decode)
        self.server = make_server(self.app, host="127.0.0.1", port=0)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        import threading

        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.app.worker.start()
        self.password = password

    def tearDown(self):
        if hasattr(self, "server"):
            self.app.worker.stop()
            self.server.shutdown()
            self.server.server_close()
            self.app.store.close()
        super().tearDown()

    def call(self, path, method="GET", body=None, password=None, headers=None, raw=False):
        request = urllib.request.Request(self.base + path, method=method,
                                         data=json.dumps(body).encode() if body is not None else None)
        request.add_header("Content-Type", "application/json")
        for k, v in (headers or {}).items():
            request.add_header(k, v)
        password = self.password if password is None else password
        if password:
            request.add_header("Authorization", "Basic " + base64.b64encode(f"siansa:{password}".encode()).decode())
        with urllib.request.urlopen(request, timeout=10) as r:
            data = r.read()
            return (r, data) if raw else json.loads(data)

    def wait_done(self, timeout=20):
        end = time.time() + timeout
        while time.time() < end:
            counts = self.call("/api/status")["counts"]
            if counts.get("songs") and counts.get("songs") == counts.get("done", 0) + counts.get("failed", 0):
                return counts
            time.sleep(0.1)
        self.fail(f"not done: {self.call('/api/status')}")

    def test_a_library_analysed_and_compared(self):
        self.start()
        self.assertEqual(self.call("/api/libraries", "POST", {"ids": ["1"]}), {"songs": 4, "queued": 4})
        counts = self.wait_done()
        self.assertEqual((counts["done"], counts["failed"]), (3, 1))
        songs = {s["id"]: s for s in self.call("/api/songs")}

        control = songs["t140"]
        self.assertAlmostEqual(control["result"]["bpm"], 140.0, delta=0.1)
        self.assertEqual(control["result"]["beats_per_bar"], 4)
        self.assertEqual(control["plugin"]["intro_bpm"], 187.1)
        texts = [t for _, t in control["verdict"]]
        self.assertIn("Plugin's intro 187.1 is 4/3 of this", texts)
        self.assertIn("Matches the BPM tag", texts)
        self.assertNotIn("genre_range", control)

        self.assertIn(["warn", "BPM tag 61 is half of this"], songs["t122"]["verdict"])
        self.assertEqual(songs["t96w"]["result"]["beats_per_bar"], 3)
        self.assertEqual(songs["bad"]["state"], "failed")
        self.assertIn("ffmpeg couldn't read", songs["bad"]["error"])

        detail = self.call("/api/songs/t140")
        self.assertGreater(len(detail["beats"]), 100)
        self.assertEqual(len(detail["downbeats"]), len(detail["beats"][::4]))
        self.assertTrue(detail["tempo_curve"])
        self.assertEqual(detail["end_seconds"], grid.END_SECONDS)
        beats = self.call("/api/beats/t140")
        self.assertEqual((beats["beats_per_bar"], beats["track_id"]), (4, "t140"))
        self.assertTrue(beats["steady"])
        self.assertGreater(beats["on_grid"], 0.9)
        self.assertAlmostEqual(len(beats["downbeats"]), len(beats["beats"]) / 4, delta=1)
        self.assertTrue(detail["grid"]["beats"])

        # Reading the library again queues nothing new; Analyse again does.
        self.assertEqual(self.call("/api/libraries/read", "POST")["queued"], 0)
        self.assertEqual(self.call("/api/songs/t140/analyse", "POST"), {"queued": True})
        self.wait_done()
        self.assertEqual(self.tracker.calls, 4)
        self.assertEqual(self.call("/api/analyse-all", "POST")["queued"], 4)
        self.wait_done()

        status = self.call("/api/status")
        self.assertEqual(status["plugin"]["state"], "ok")
        self.assertEqual((status["plugin"]["found"], status["plugin"]["asked"]), (1, 4))
        self.assertTrue(any("Library Music: 4 songs, 4 new" in e["message"] for e in status["events"]))
        self.assertTrue(any("Broken failed" in e["message"] for e in status["events"]))
        self.assertTrue(status["model_loaded"])

        _, csv = self.call("/api/export.csv", raw=True)
        lines = csv.decode().splitlines()
        self.assertTrue(lines[0].startswith("artist,title"))
        self.assertEqual(len(lines), 5)
        self.assertIn("4/3", csv.decode())

        self.assertEqual(self.call("/api/songs/bad", "DELETE"), {"removed": True})
        self.assertEqual(len(self.call("/api/songs")), 3)
        # Temporary downloads are gone.
        self.assertEqual([f for f in os.listdir(self.app.config.data_dir) if f.startswith("song-")], [])

    def test_results_of_an_older_version_are_worked_out_again_from_their_beats(self):
        self.start()
        self.call("/api/libraries", "POST", {"ids": ["1"]})
        self.wait_done()
        db = self.app.store
        with db.lock, db.db:  # as 0.2: no main tempo yet, and the whole-song line as the tempo
            db.db.execute("UPDATE results SET bpm=149.1, bpm_fit=NULL, main_share=NULL, analysis_version=NULL")
        self.assertEqual(self.app.worker.regrid(), 3)
        control = {s["id"]: s for s in self.call("/api/songs")}["t140"]["result"]
        self.assertAlmostEqual(control["bpm"], 140.0, delta=0.1)
        self.assertIsNotNone(control["main_share"])
        self.assertEqual(self.app.worker.regrid(), 0)
        self.assertEqual(self.tracker.calls, 3)  # no model needed

    def test_the_data_comes_out_for_tuning(self):
        self.start()
        self.call("/api/libraries", "POST", {"ids": ["1"]})
        self.wait_done()
        data = self.call("/api/export.json")
        self.assertEqual(len(data["songs"]), 3)
        song = {s["id"]: s for s in data["songs"]}["t140"]
        self.assertTrue(song["beats"])
        self.assertIn("bpm", song["result"])
        self.assertNotIn(TOKEN, json.dumps(data))

    def test_section_marks_are_kept_shown_and_exported(self):
        self.start()
        self.call("/api/libraries", "POST", {"ids": ["1"]})
        self.wait_done()
        added = self.call("/api/songs/t140/marks", "POST", {"seconds": 67.4})
        self.call("/api/songs/t140/marks", "POST", {"seconds": 12.25})
        marks = self.call("/api/songs/t140")["marks"]
        self.assertEqual([m["seconds"] for m in marks], [12.25, 67.4])
        self.assertEqual({s["id"]: s["marks"] for s in self.call("/api/songs")}["t140"], 2)
        self.call("/api/analyse-all", "POST")  # analysing again keeps them
        self.wait_done()
        song = {s["id"]: s for s in self.call("/api/export.json")["songs"]}["t140"]
        self.assertEqual(song["marks"], [12.25, 67.4])
        left = self.call(f"/api/songs/t140/marks/{added['id']}", "DELETE")
        self.assertEqual([m["seconds"] for m in left["marks"]], [12.25])
        for bad in ({"seconds": "soon"}, {}, {"seconds": -1}):
            with self.assertRaises(urllib.error.HTTPError) as caught:
                self.call("/api/songs/t140/marks", "POST", bad)
            self.assertEqual(caught.exception.code, 400)
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.call("/api/songs/nope/marks", "POST", {"seconds": 1})
        self.assertEqual(caught.exception.code, 404)

    def test_grids_for_the_app(self):
        self.start()
        self.call("/api/libraries", "POST", {"ids": ["1"]})
        self.wait_done()
        answer = self.call("/api/grids?ids=t140,nope,t122,t140")
        self.assertEqual(answer["missing"], ["nope"])
        grids = {g["track_id"]: g for g in answer["grids"]}
        self.assertEqual(set(grids), {"t140", "t122"})
        g = grids["t140"]
        for end in ("intro", "outro"):
            for field in ("bpm", "beat_ms", "bar_ms", "rhythm", "steady"):
                self.assertIsNotNone(g[f"{end}_{field}"], f"{end}_{field}")
        self.assertAlmostEqual(g["intro_bpm"], 140.0, delta=0.1)
        self.assertLess(g["intro_beat_ms"], g["outro_beat_ms"])
        self.assertEqual(g["intro_phrase_bars"], 8)
        self.assertIsNone(g["beats_until_ms"])  # it beats to its end: no beatless outro
        self.assertNotIn("ends", self.call("/api/songs")[0]["result"])
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.call("/api/grids?ids=" + ",".join(str(n) for n in range(2001)))
        self.assertEqual(caught.exception.code, 400)

    def test_the_plugin_line_says_what_went_wrong(self):
        self.start()
        self.app.audiomuse.token = "wrong"
        self.call("/api/plugin/refresh", "POST")
        plugin = self.call("/api/status")["plugin"]
        self.assertEqual(plugin["state"], "error")
        self.assertIn("AUDIOMUSE_TOKEN", plugin["text"])
        self.app.audiomuse.url = ""
        self.assertEqual(self.app.worker.check_plugin()["state"], "off")

    def test_the_audio_is_passed_through_with_seeking(self):
        self.start()
        self.call("/api/libraries", "POST", {"ids": ["1"]})
        r, data = self.call("/api/audio/t122", raw=True)
        self.assertEqual((r.status, data), (200, b"t122"))
        r, data = self.call("/api/audio/t122", headers={"Range": "bytes=2-"}, raw=True)
        self.assertEqual((r.status, data, r.headers["Content-Range"]), (206, b"22", "bytes 2-3/4"))
        stream = [q for p, q in self.fake.requests if p == "/rest/stream" and q.get("format") == "mp3"]
        self.assertTrue(stream and stream[0]["maxBitRate"] == "320")
        with self.assertRaises(urllib.error.HTTPError) as e:
            self.call("/api/audio/unknown")
        self.assertEqual(e.exception.code, 404)

    def test_an_unknown_library_lists_the_ones_there_are(self):
        self.start()
        with self.assertRaises(urllib.error.HTTPError) as e:
            self.call("/api/libraries", "POST", {"ids": ["9"]})
        self.assertEqual(e.exception.code, 400)
        self.assertIn("1 Music, 2 Compilations", json.loads(e.exception.read())["error"])
        with self.assertRaises(urllib.error.HTTPError) as e:
            self.call("/api/libraries", "POST", {})
        self.assertEqual(e.exception.code, 400)
        self.assertEqual(self.call("/api/status")["counts"]["songs"], 0)

    def test_pause_and_carry_on(self):
        self.start()
        self.assertEqual(self.call("/api/pause", "POST"), {"paused": True})
        self.call("/api/libraries", "POST", {"ids": ["1"]})
        time.sleep(0.5)
        self.assertEqual(self.tracker.calls, 0)
        self.assertEqual(self.call("/api/resume", "POST"), {"paused": False})
        self.wait_done()
        self.assertEqual(self.tracker.calls, 3)

    def test_a_password_guards_everything_but_health(self):
        self.start(password="s3cret")
        self.assertEqual(self.call("/api/health", password="")["status"], "ok")
        for path in ("/", "/api/status", "/api/songs", "/api/audio/t140", "/api/export.csv"):
            with self.assertRaises(urllib.error.HTTPError) as e:
                self.call(path, password="")
            self.assertEqual(e.exception.code, 401, path)
            self.assertIn("Basic", e.exception.headers["WWW-Authenticate"])
        with self.assertRaises(urllib.error.HTTPError):
            self.call("/api/status", password="wrong")
        self.assertIn("counts", self.call("/api/status"))
        _, page = self.call("/", raw=True)
        self.assertIn(b"<title>Siansa Beats</title>", page)

    def test_chosen_libraries(self):
        for song in self.fake.songs:
            song["library"] = "2" if song["id"] == "t96w" else "1"  # the waltz is on a compilation
        self.start()
        self.assertEqual([(l["name"], l["on"]) for l in self.call("/api/libraries")],
                         [("Music", False), ("Compilations", False)])
        self.assertEqual(self.call("/api/libraries", "POST", {"ids": ["1"]}), {"songs": 3, "queued": 3})
        counts = self.wait_done()
        self.assertEqual((counts["songs"], counts["done"], counts["failed"]), (3, 2, 1))
        self.assertNotIn("t96w", [s["id"] for s in self.call("/api/songs")])
        status = self.call("/api/status")
        self.assertEqual(status["libraries"]["chosen"], ["1"])
        self.assertIsNotNone(status["libraries"]["read_at"])
        libraries = {l["name"]: l for l in self.call("/api/libraries")}
        self.assertEqual((libraries["Music"]["on"], libraries["Music"]["songs"], libraries["Music"]["done"]), (True, 3, 2))
        self.assertFalse(libraries["Compilations"]["on"])

        self.assertEqual(self.call("/api/libraries", "POST", {"ids": ["1", "2"]}), {"songs": 4, "queued": 1})
        self.wait_done()
        self.assertEqual([s["id"] for s in self.call("/api/songs?find=waltz")], ["t96w"])
        self.assertEqual([s["id"] for s in self.call("/api/songs?show=failed")], ["bad"])
        self.assertEqual(sorted(s["id"] for s in self.call("/api/songs?show=done")), ["t122", "t140", "t96w"])
        self.assertEqual(len(self.call("/api/songs?limit=2")), 2)
        self.assertEqual(len(self.call("/api/songs")), 4)
        self.assertEqual(len(self.call("/api/export.json")["songs"]), 3)
        self.assertEqual([s["id"] for s in self.call("/api/export.json?find=waltz")["songs"]], ["t96w"])
        _, csv_body = self.call("/api/export.csv?find=waltz", raw=True)
        self.assertEqual(len(csv_body.decode().strip().splitlines()), 2)
        self.assertEqual(self.call("/api/grids?ids=t122")["grids"][0]["track_id"], "t122")  # the app finds them all
        self.assertEqual(self.call("/api/libraries/read", "POST"), {"songs": 4, "queued": 0})

    def test_the_songs_of_a_library_left_out_can_be_removed(self):
        for song in self.fake.songs:
            song["library"] = "2" if song["id"] in ("t96w", "t122") else "1"  # 2: long DJ mixes, say
        self.start()
        self.call("/api/libraries", "POST", {"ids": ["1", "2"]})
        self.wait_done()
        self.call("/api/songs/t122/marks", "POST", {"seconds": 12.0})
        with self.assertRaises(urllib.error.HTTPError) as e:  # still being analysed
            self.call("/api/libraries/2/songs", "DELETE")
        self.assertEqual(e.exception.code, 409)
        self.call("/api/libraries", "POST", {"ids": ["1"]})
        self.assertEqual(self.call("/api/libraries/2/songs", "DELETE"), {"removed": 2})
        self.assertEqual(sorted(s["id"] for s in self.call("/api/songs")), ["bad", "t140"])
        self.assertEqual(self.call("/api/grids?ids=t122")["missing"], ["t122"])
        libraries = {l["id"]: l for l in self.call("/api/libraries")}
        self.assertEqual((libraries["2"]["songs"], libraries["1"]["songs"]), (0, 2))
        self.assertEqual(self.app.store.marks("t122"), [])

    def test_the_libraries_of_0_9_0s_whole_library(self):
        self.start()
        self.app.store.set_setting("library", True)
        self.assertEqual(self.app.worker.chosen(), ["1", "2"])
        self.assertEqual(self.call("/api/status")["libraries"]["chosen"], ["1", "2"])

    def test_the_library_album_by_album_when_search_finds_nothing(self):
        self.fake.search_all = False
        nd = Subsonic(self.fake.url, USER, PASSWORD)
        self.assertEqual(sorted(s["id"] for s in nd.all_songs()), ["bad", "t122", "t140", "t96w"])
        self.fake.songs[0]["library"] = "2"
        self.assertEqual([s["id"] for s in nd.all_songs("2")], ["t140"])
        self.fake.search_all = True
        self.assertEqual(len(nd.all_songs(page=3)), 4)  # paged
        self.assertEqual(len(nd.all_songs("1", page=2)), 3)

    def test_leaving_a_library_out_empties_its_queue(self):
        for song in self.fake.songs:
            song["library"] = "2" if song["id"] in ("t96w", "t122") else "1"
        self.start()
        self.call("/api/pause", "POST")
        self.call("/api/libraries", "POST", {"ids": ["1", "2"]})
        self.assertEqual(self.call("/api/status")["counts"]["queued"], 4)
        self.assertEqual(self.call("/api/libraries", "POST", {"ids": ["1"]}), {"songs": 2, "queued": 0})
        self.assertEqual(self.call("/api/status")["counts"]["queued"], 2)
        self.assertEqual(self.call("/api/libraries", "POST", {"ids": []}), {"songs": 0, "queued": 0})
        self.assertEqual(self.call("/api/status")["counts"].get("queued", 0), 0)

    def test_the_next_song_is_downloaded_while_the_model_works(self):
        self.start()
        self.tracker.delay = 0.4
        self.call("/api/libraries", "POST", {"ids": ["1"]})
        end, seen = time.time() + 10, False
        while time.time() < end and not seen:
            status = self.call("/api/status")
            seen = bool(status["current"] and status["next"])
            time.sleep(0.05)
        self.assertTrue(seen, "never a song in the model and the next one being prepared at once")
        self.wait_done()

    def test_unknown_routes_and_songs(self):
        self.start()
        for path in ("/api/nothing", "/api/songs/none", "/api/beats/none"):
            with self.assertRaises(urllib.error.HTTPError) as e:
                self.call(path)
            self.assertEqual(e.exception.code, 404, path)
        with self.assertRaises(urllib.error.HTTPError) as e:
            self.call("/api/songs/none/analyse", "POST")
        self.assertEqual(e.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
