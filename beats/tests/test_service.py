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
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from beats import __version__, grid  # noqa: E402
from beats.server import App, Config, make_server  # noqa: E402
from beats.store import Store  # noqa: E402
from beats.worker import Worker  # noqa: E402
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


class FakeStyle:
    """The style model's stand-in: Control sounds like trance, Deep like deep house, Waltz like trance too (its Pop
    tag is wrong)."""

    HEARD = {"t140": [("Electronic---Trance", 0.71), ("Electronic---Progressive Trance", 0.4)],
             "t122": [("Electronic---Deep House", 0.66), ("Electronic---House", 0.5)],
             "t96w": [("Electronic---Trance", 0.55), ("Pop---Ballad", 0.1)]}

    def __init__(self, models_dir, on=True):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        labels = sorted({l for heard in self.HEARD.values() for l, _ in heard} | {"Rock---Punk", "Pop---Pop Rock"})
        with open(os.path.join(models_dir, "discogs-effnet-bs64-1.json"), "w") as f:
            json.dump({"classes": labels}, f)
        self.on, self.loaded, self.load_seconds, self.calls = on, False, 0.01, []

    @property
    def available(self):
        return self.on

    def __call__(self, samples, rate=16_000):
        self.loaded = True
        self.calls.append((samples.song_id, rate))
        return self.HEARD.get(samples.song_id, [])


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


def fake_sound(path, samples, rate):
    """Stands in for the loudness, fades and key measure: the stand-in files aren't audio."""
    song_id = samples.song_id
    if song_id == "t96w":
        raise RuntimeError("ffmpeg couldn't read the song for its loudness: nope")
    return {"ref_lufs": -9.5 if song_id == "t140" else -12.25, "start_ramp": "xAAA", "end_ramp": "xAAB",
            "duration_ms": int(samples.seconds * 1000), "key": "A", "scale": "minor", "key_strength": 0.8}


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


class LogTest(unittest.TestCase):
    def test_secrets_are_taken_out(self):
        from beats.log import redact

        self.assertEqual(redact("http://nd:4533/rest/stream?u=max&t=abc123&s=x9&v=1.16.1&id=5"),
                         "http://nd:4533/rest/stream?u=max&t=***&s=***&v=1.16.1&id=5")
        self.assertEqual(redact("Authorization: Bearer eyJhbGciOi.x.y"), "Authorization: Bearer ***")
        self.assertEqual(redact("password=hunter2 token: abc"), "password=*** token: ***")
        self.assertEqual(redact("/api/songs?find=password&show=done"), "/api/songs?find=password&show=done")


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
        self.assertNotIn("plugin", row)
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
    def start(self, password="", style=None, api_key=""):
        env = {"NAVIDROME_URL": self.fake.url, "NAVIDROME_USER": USER, "NAVIDROME_PASSWORD": PASSWORD,
               "AUDIOMUSE_URL": self.fake.url, "AUDIOMUSE_TOKEN": TOKEN, "DATA_DIR": os.path.join(self.tmp, "data"),
               "BEATS_PASSWORD": password, "BEATS_API_KEY": api_key, "PORT": "0"}
        self.tracker = FakeTracker()
        self.app = App(Config(env), tracker=self.tracker, decode=fake_decode, style=style or FakeStyle(
            os.path.join(self.tmp, "models"), on=False), sound=fake_sound)
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
        self.assertNotIn("plugin", control)
        texts = [t for _, t in control["verdict"]]
        self.assertFalse([t for t in texts if "Plugin" in t])
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
        self.assertNotIn("plugin", status)
        # Nothing is asked of AudioMuse-AI any more.
        self.assertFalse([path for path, _ in self.fake.requests if path.startswith("/plugins/")])
        self.assertTrue(any("Library Music: 4 songs, 4 new" in e["message"] for e in status["events"]))
        self.assertTrue(any("Broken failed" in e["message"] for e in status["events"]))
        self.assertTrue(status["model_loaded"])

        _, csv = self.call("/api/export.csv", raw=True)
        lines = csv.decode().splitlines()
        self.assertTrue(lines[0].startswith("artist,title"))
        self.assertEqual(len(lines), 5)
        self.assertIn("BPM tag 61 is half of this", csv.decode())
        self.assertNotIn("plugin", csv.decode().splitlines()[0])

        self.assertEqual(self.call("/api/songs/bad", "DELETE"), {"removed": True})
        self.assertEqual(len(self.call("/api/songs")), 3)
        # Temporary downloads are gone.
        self.assertEqual([f for f in os.listdir(self.app.config.data_dir) if f.startswith("song-")], [])

    def test_status_shows_songs_being_worked_out_again(self):
        self.start()
        self.app.worker.regridding = {"done": 3, "total": 10}
        self.assertEqual(self.call("/api/status")["regridding"], {"done": 3, "total": 10})
        self.app.worker.regridding = None
        self.assertIsNone(self.call("/api/status")["regridding"])

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

    def test_v1_capabilities_and_profiles_for_the_app(self):
        self.start()
        empty = self.call("/api/v1/capabilities")
        self.assertEqual((empty["service"], empty["api"]), ("siansa-beats", 1))
        self.assertEqual(empty["features"]["beat_grids"]["songs"], 0)
        self.assertTrue(empty["features"]["similar"]["available"])  # AUDIOMUSE_URL is set
        self.call("/api/libraries", "POST", {"ids": ["1"]})
        self.wait_done()
        caps = self.call("/api/v1/capabilities")
        self.assertEqual(caps["songs"]["analysed"], 3)
        self.assertEqual(caps["features"]["beat_grids"]["songs"], 3)
        self.assertEqual(caps["features"]["loudness"]["songs"], 2)  # t96w couldn't be measured
        self.assertEqual(caps["limits"]["profile_ids"], 500)

        answer = self.call("/api/v1/profiles?ids=t140,nope,t96w,t140")
        self.assertEqual(answer["schema_version"], 1)
        self.assertEqual(answer["pending"], ["nope"])
        profiles = {p["track_id"]: p for p in answer["profiles"]}
        self.assertEqual(set(profiles), {"t140", "t96w"})
        p = profiles["t140"]
        self.assertEqual((p["ref_lufs"], p["start_ramp"], p["end_ramp"], p["duration_ms"]), (-9.5, "xAAA", "xAAB", 60000))
        self.assertEqual((p["key"], p["scale"], p["beat_grid"]), ("A", "minor", "siansa_beats"))
        self.assertAlmostEqual(p["intro_bpm"], 140.0, delta=0.1)
        self.assertNotIn("beats_until_ms", p)  # nulls are left out, as the plugin did
        grid_only = profiles["t96w"]  # its beats, without loudness: the app levels it by its tags
        self.assertEqual((grid_only["ref_lufs"], grid_only["start_ramp"], grid_only["duration_ms"]), (None, "", 0))
        self.assertNotIn("key", grid_only)
        self.assertEqual(grid_only["beat_grid"], "siansa_beats")
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.call("/api/v1/profiles?ids=" + ",".join(str(n) for n in range(501)))
        self.assertEqual(caught.exception.code, 400)
        self.assertEqual(self.call("/api/status")["loudness"]["measured"], 2)

    def test_v1_genres_about_and_bad_blends(self):
        self.start(style=FakeStyle(os.path.join(self.tmp, "models"), on=True))
        self.call("/api/libraries", "POST", {"ids": ["1"]})
        self.wait_done()
        self.assertEqual(self.call("/api/v1/genres")["genres"], self.call("/api/genres")["genres"])
        listed = {g["style"]: g for g in self.call("/api/v1/genres")["genres"]}
        self.assertEqual(listed["Trance"]["libraries"], {"1": listed["Trance"]["songs"]})  # each library's songs
        trance = self.call("/api/v1/genre?name=Trance")
        self.assertEqual((set(trance["ids"]), trance["styles"]), ({"t140", "t96w"}, ["Electronic---Trance"]))
        for path in ("/api/v1/genre", "/api/v1/genre/about", "/api/v1/genre?name=Trance&limit=x"):
            with self.assertRaises(urllib.error.HTTPError) as caught:
                self.call(path)
            self.assertEqual(caught.exception.code, 400, path)
        self.app.store.set_setting("about:trance", {"summary": "Fast.", "url": "https://en.wikipedia.org/wiki/Trance_music",
                                                    "source": "Wikipedia", "at": time.time()})
        about = self.call("/api/v1/genre/about?name=Trance")  # kept: not looked up on Wikipedia in the tests
        self.assertEqual((about["name"], about["about"]["summary"]), ("Trance", "Fast."))
        made = self.call("/api/v1/bad-blends", "POST", {"from_id": "t140", "to_id": "t122", "technique": "echo out"})
        self.assertEqual(self.call("/api/bad-blends")[0]["id"], made["id"])
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.call("/api/v1/bad-blends", "POST", {"from_id": "t140"})
        self.assertEqual(caught.exception.code, 400)
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.call("/api/v1/nothing")
        self.assertEqual(caught.exception.code, 404)
        spec = self.call("/api/v1/openapi.json")
        self.assertEqual(spec["info"]["version"], "1")
        for path in ("/api/v1/capabilities", "/api/v1/profiles", "/api/v1/genres", "/api/v1/genre", "/api/v1/genre/about",
                     "/api/v1/bad-blends"):
            self.assertIn(path, spec["paths"])

    def test_v1_similar_albums_and_artists_asked_of_audiomuse(self):
        self.start()
        self.assertTrue(self.call("/api/v1/capabilities")["features"]["similar"]["available"])
        albums = self.call("/api/v1/similar/albums?track_id=t140")
        self.assertEqual((albums["source"], albums["similar"][0]["album"], albums["similar"][0]["track_id"]), ("audiomuse", "Deep", "t122"))
        self.assertEqual(albums["similar"][0]["asked"], {"track_id": "t140"})
        artists = self.call("/api/v1/similar/artists?artist=Ecsape&server=Home")
        self.assertEqual(artists["similar"][0]["artist"], "House Test")
        self.assertEqual(artists["similar"][0]["asked"], {"artist": "Ecsape", "server": "Home"})
        for path, code in (("/api/v1/similar/albums", 400), ("/api/v1/similar/albums?artist=x", 400),
                           ("/api/v1/similar/albums?track_id=nope", 502)):
            with self.assertRaises(urllib.error.HTTPError) as caught:
                self.call(path)
            self.assertEqual(caught.exception.code, code, path)

    def test_an_api_key_for_the_app(self):
        self.start(password="s3cret", api_key="k3y")
        self.assertEqual(self.call("/api/v1/capabilities", password="", headers={"Authorization": "Bearer k3y"})["api"], 1)
        self.assertEqual(self.call("/api/v1/capabilities", password="", headers={"Authorization": "Bearer s3cret"})["api"], 1)
        self.assertEqual(self.call("/api/v1/capabilities")["api"], 1)  # Basic with the password
        for header in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": "Basic " + base64.b64encode(b"siansa:k3y").decode()}):
            with self.assertRaises(urllib.error.HTTPError) as caught:
                self.call("/api/v1/profiles?ids=t140", password="", headers=header)
            self.assertEqual(caught.exception.code, 401, header)
        self.assertEqual(self.call("/api/health", password="")["status"], "ok")

    def test_songs_analysed_before_get_their_loudness_while_the_queue_is_empty(self):
        Worker.MEASURE_IDLE_SECONDS = 0.5
        self.addCleanup(setattr, Worker, "MEASURE_IDLE_SECONDS", 60.0)
        self.start()
        self.call("/api/libraries", "POST", {"ids": ["1"]})
        self.wait_done()
        with self.app.store.lock, self.app.store.db:  # as 0.19 left them: beats, no loudness
            self.app.store.db.execute("DELETE FROM measures")
        end = time.time() + 20
        while time.time() < end and self.app.store.measure_counts(1)["waiting"]:
            time.sleep(0.2)
        self.assertEqual(self.app.store.measure_counts(1), {"measured": 2, "failed": 1, "waiting": 0})
        self.assertEqual(self.call("/api/v1/profiles?ids=t122")["profiles"][0]["ref_lufs"], -12.25)

    def test_the_log_shows_every_step_with_debug_details_and_no_secrets(self):
        from beats import log

        self.start()
        log.setup("info")
        self.assertEqual(self.call("/api/log/level", "POST", {"level": "debug"}), {"level": "debug"})
        self.assertEqual(self.app.store.setting("log_level"), "debug")
        self.assertEqual(self.call("/api/status")["log_level"], "debug")
        self.call("/api/libraries", "POST", {"ids": ["1"]})
        self.wait_done()
        lines = self.call("/api/log?limit=5000")["lines"]
        text = "\n".join(l["message"] for l in lines)
        self.assertIn("downloaded", text)
        self.assertIn("the model found", text)
        self.assertIn("getMusicFolders", text)  # the requests to Navidrome, with their times
        self.assertTrue(any(l["level"] == "error" and "Broken failed" in l["message"] for l in lines))
        self.assertNotRegex(text, r"[?&]t=[0-9a-f]{6}")  # Subsonic's login token never reaches the log
        newer = self.call(f"/api/log?after={lines[-1]['n']}")["lines"]
        self.assertTrue(all(l["n"] > lines[-1]["n"] for l in newer))
        self.assertEqual(self.call("/api/log?level=warn&limit=5000")["lines"],
                         [l for l in self.call("/api/log?limit=5000")["lines"] if l["level"] in ("warn", "error")])
        r, body = self.call("/api/log/download", raw=True)
        self.assertIn("attachment", r.headers["Content-Disposition"])
        self.assertIn(b"ERROR", body)
        with self.assertRaises(urllib.error.HTTPError) as e:
            self.call("/api/log/level", "POST", {"level": "loud"})
        self.assertEqual(e.exception.code, 400)
        self.call("/api/log/level", "POST", {"level": "info"})
        before = self.call("/api/log?limit=1")["lines"][-1]["n"]
        self.call("/api/songs/t140/analyse", "POST")
        self.wait_done()
        self.assertFalse([l for l in self.call(f"/api/log?after={before}&limit=5000")["lines"] if l["level"] == "debug"])

    def test_songs_that_failed_on_a_fixed_bug_are_analysed_again(self):
        self.start()
        self.call("/api/libraries", "POST", {"ids": ["1"]})
        self.wait_done()
        self.app.store.set_state("t140", "failed", "'bars'")
        calls = self.tracker.calls
        self.app.worker.regrid()
        self.wait_done()
        self.assertEqual(self.tracker.calls, calls + 1)
        states = {s["id"]: s["state"] for s in self.call("/api/songs")}
        self.assertEqual(states["t140"], "done")
        self.assertEqual(states["bad"], "failed")  # its own error, not a fixed bug: left alone

    def test_genres_are_voted_from_the_sound_and_the_tags(self):
        style = FakeStyle(os.path.join(self.tmp, "models"))
        self.start(style=style)
        self.call("/api/libraries", "POST", {"ids": ["1"]})
        self.wait_done()
        songs = {s["id"]: s for s in self.call("/api/songs")}
        vote = lambda sid: [g["label"] for g in songs[sid]["genres"] if g["source"] == "vote"]
        self.assertEqual(vote("t140")[0], "Electronic---Trance")  # its sound and its tag agree
        self.assertEqual(vote("t96w")[0], "Electronic---Trance")  # its sound wins over its Pop tag
        self.assertEqual(vote("t122")[:2], ["Electronic---House", "Electronic---Deep House"])  # its House tag tips it
        self.assertEqual({sid for sid, _ in style.calls}, {"t140", "t122", "t96w"})  # from the samples on hand
        radio = self.call("/api/genre?name=" + urllib.parse.quote("Electronic / Trance"))
        self.assertEqual(radio["styles"], ["Electronic---Trance"])
        self.assertEqual(set(radio["ids"]), {"t140", "t96w"})
        self.assertEqual(set(self.call("/api/genre?name=Electronic")["ids"]), {"t140", "t122", "t96w"})
        self.assertEqual(self.call("/api/genre?name=Opera")["ids"], [])
        listed = self.call("/api/genres")
        self.assertEqual(listed["genres"][0], {"label": "Electronic---Trance", "genre": "Electronic", "style": "Trance",
                                               "songs": 2, "name": "Trance", "libraries": {"1": 2}})
        status = self.call("/api/status")["styles"]
        self.assertTrue(status["available"])
        self.assertEqual(status["done"], 3)
        self.assertEqual(status["failed"], 0)  # the broken one failed before: not asked for its style
        _, csv_body = self.call("/api/export.csv", raw=True)
        self.assertIn("Trance 0.71", csv_body.decode())

    def test_songs_analysed_before_get_their_styles_while_the_queue_is_empty(self):
        style = FakeStyle(os.path.join(self.tmp, "models"), on=False)
        self.start(style=style)
        self.call("/api/libraries", "POST", {"ids": ["1"]})
        self.wait_done()
        self.assertEqual(style.calls, [])
        style.on = True  # an update brings the model
        end = time.time() + 20
        while time.time() < end and self.call("/api/status")["styles"].get("done", 0) < 3:
            time.sleep(0.2)
        self.assertEqual(self.call("/api/status")["styles"]["done"], 3)
        self.assertEqual(set(self.call("/api/genre?name=Trance")["ids"]), {"t140", "t96w"})

    def test_without_the_style_model_the_app_falls_back_to_tags(self):
        self.start()
        self.call("/api/libraries", "POST", {"ids": ["1"]})
        self.wait_done()
        radio = self.call("/api/genre?name=Trance")
        self.assertEqual(radio["ids"], [])
        self.assertIn("no style model", radio["why"])
        self.assertFalse(self.call("/api/status")["styles"]["available"])

    def test_the_debug_report_says_where_everything_is_without_secrets(self):
        self.start(password="pagepass")
        self.call("/api/libraries", "POST", {"ids": ["1"]})
        self.wait_done()
        r, body = self.call("/api/debug", raw=True)
        self.assertIn("attachment", r.headers["Content-Disposition"])
        report = json.loads(body)
        self.assertEqual(report["versions"]["siansa_beats"], __version__)
        self.assertEqual(report["setup"]["navidrome_password"], "set")
        self.assertEqual(report["setup"]["page_password"], "set")
        self.assertEqual(report["setup"]["audiomuse_token"], "set")
        self.assertNotIn(f'"{TOKEN}"', body.decode())
        self.assertIn("beats-model", report["threads"])
        self.assertEqual(report["status"]["counts"]["failed"], 1)
        self.assertNotIn("pagepass", body.decode())
        self.assertNotRegex(body.decode(), r"[?&]t=[0-9a-f]{6}")

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
        failed = self.call("/api/failed/export")["failed"]
        self.assertEqual([f["id"] for f in failed], ["bad"])
        self.assertTrue(failed[0]["error"])
        self.assertEqual(failed[0]["library"], "1")
        self.assertIn("error", self.call("/api/export.csv?show=failed", raw=True)[1].decode().splitlines()[0].split(","))
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

    def test_bad_blends_from_the_app(self):
        self.start()
        self.call("/api/libraries", "POST", {"ids": ["1"]})
        self.wait_done()
        self.assertEqual(self.call("/api/status")["bad_blends"], 0)
        with self.assertRaises(urllib.error.HTTPError) as e:
            self.call("/api/bad-blends", "POST", {"from_id": "t140"})
        self.assertEqual(e.exception.code, 400)
        made = self.call("/api/bad-blends", "POST", {"from_id": "t140", "to_id": "t122", "technique": "bass swap",
                                                    "details": {"bars": 16, "from_rhythm": 0.56}})
        self.call("/api/bad-blends", "POST", {"from_id": "gone", "to_id": "t140", "details": "not a dict"})
        rows = self.call("/api/bad-blends")
        self.assertEqual(len(rows), 2)
        first = rows[1]
        self.assertEqual((first["id"], first["technique"], first["details"]["bars"]), (made["id"], "bass swap", 16))
        self.assertTrue(first["from_title"])
        self.assertIsNone(rows[0]["from_title"])  # a song that isn't here
        self.assertEqual(rows[0]["details"], {})
        self.assertEqual(self.call("/api/status")["bad_blends"], 2)
        self.assertTrue(any("Bad blend marked" in e["message"] for e in self.call("/api/status")["events"]))
        export = self.call("/api/bad-blends/export")
        marked = next(b for b in export["bad_blends"] if b["id"] == made["id"])
        self.assertEqual((marked["from"]["id"], marked["to"]["id"]), ("t140", "t122"))
        self.assertTrue(marked["from"]["beats"])
        self.assertIsNone(next(b for b in export["bad_blends"] if b["from_id"] == "gone")["from"])
        self.assertEqual(self.call(f"/api/bad-blends/{made['id']}", "DELETE"), {"removed": True})
        with self.assertRaises(urllib.error.HTTPError) as e:
            self.call(f"/api/bad-blends/{made['id']}", "DELETE")
        self.assertEqual(e.exception.code, 404)
        grids = self.call("/api/grids?ids=t140")["grids"][0]
        for key in ("intro_lock_ms", "outro_lock_ms", "mix_track", "intro_bass_bars", "drops_ms", "intro_energy"):
            self.assertIn(key, grids)

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


class KeptTest(unittest.TestCase):
    """Answers kept ready: fresh when the data changed and the database is quiet, the last one at once when it's busy."""

    def test_a_change_is_seen_when_the_database_answers_in_time(self):
        from beats.server import Kept

        state = {"changes": 0, "value": 1}
        kept = Kept(lambda: state["value"], seconds=300, changes=lambda: state["changes"])
        self.assertEqual(kept.get(), 1)
        state["value"] = 2
        self.assertEqual(kept.get(), 1)  # nothing changed: the kept one, without asking
        state["changes"] += 1
        self.assertEqual(kept.get(), 2)

    def test_a_busy_database_doesnt_hold_the_answer_up(self):
        import threading

        from beats.server import Kept

        gate = threading.Event()
        state = {"changes": 0, "calls": 0}

        def work():
            state["calls"] += 1
            if state["calls"] > 1:
                gate.wait(5)  # a database busy with analysis
            return state["calls"]

        kept = Kept(work, seconds=300, changes=lambda: state["changes"])
        self.assertEqual(kept.get(), 1)
        state["changes"] += 1
        started = time.monotonic()
        self.assertEqual(kept.get(), 1)  # the last one, after at most Kept.WAIT
        self.assertLess(time.monotonic() - started, Kept.WAIT + 0.4)
        gate.set()
        end = time.monotonic() + 5
        while kept.value != 2 and time.monotonic() < end:
            time.sleep(0.01)
        self.assertEqual(kept.value, 2)
