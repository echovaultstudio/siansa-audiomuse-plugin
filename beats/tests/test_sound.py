"""What songs sound like (0.25): taken over from AudioMuse-AI's export, and answered as AudioMuse-AI answers it, so the
app can ask Siansa Beats the same things. The vectors here are made up: songs of a "style" point the same way."""

import base64
import os
import sys
import time
import unittest
import urllib.error

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from beats import sound  # noqa: E402
from beats.store import Store  # noqa: E402
from test_service import CATALOG, Base, Service  # noqa: E402

AXES = {"trance": 0, "house": 1, "jazz": 2, "rock": 3}


def vector(style, lean=None, amount=0.0, wobble=0):
    """A song's vector: along its style's axis, leaning ``amount`` towards ``lean``'s, with a little of its own."""
    v = np.zeros(sound.DIMENSIONS, np.float32)
    v[AXES[style]] = 1.0
    if lean:
        v[AXES[lean]] = amount
    v[10 + wobble] = 0.3  # enough that two songs of a style are not taken for one recording
    return v


def song(song_id, title, artist, album, style, lean=None, amount=0.0, wobble=0, other="danceable:0.5,happy:0.5"):
    return {"id": song_id, "title": title, "artist": artist, "album": album, "duration": 200.0, "suffix": "flac",
            "tag_bpm": None, "genres": [], "path": f"{artist}/{album}/{title}.flac",
            "vector": vector(style, lean, amount, wobble), "other": other}


LIBRARY = [
    song("t1", "Lift", "Ana", "Up", "trance", wobble=1),
    song("t2", "Rise", "Ana", "Up", "trance", wobble=2),
    song("t3", "Soar", "Ana", "Up", "trance", wobble=3),
    song("t4", "Climb", "Ana", "Higher", "trance", wobble=4),
    song("t5", "Peak", "Bo", "Tops", "trance", wobble=5),
    song("t6", "Lift", "Ana", "Up (Deluxe)", "trance", wobble=1),  # the same song again, on another edition
    song("m1", "Halfway", "Cy", "Between", "trance", "house", 1.0, wobble=6),
    song("m2", "Nearly", "Cy", "Between", "trance", "house", 3.0, wobble=7),
    song("h1", "Groove", "Di", "Floor", "house", wobble=8, other="danceable:0.9,happy:0.9,party:0.9"),
    song("h2", "Swing", "Di", "Floor", "house", wobble=9),
    song("h3", "Step", "Ed", "Rooms", "house", wobble=10),
    song("j1", "Blue", "Flo", "Late", "jazz", wobble=11, other="relaxed:0.9,sad:0.6"),
    song("j2", "Smoke", "Flo", "Late", "jazz", wobble=12, other="relaxed:0.9,sad:0.6"),
    song("r1", "Riff", "Gus", "Loud", "rock", wobble=13, other="aggressive:0.9"),
]


def filled(store, server="main", songs=LIBRARY):
    store.link_songs(server, "1", [{k: s[k] for k in s if k not in ("vector", "other")} for s in songs])
    store.save_sound([{"song_id": s["id"], "musicnn": s["vector"].tobytes(), "other": s["other"], "fp": "f-" + s["id"],
                       "album_artist": s["artist"]} for s in songs])


class SoundTest(Base):
    def setUp(self):
        super().setUp()
        self.store = Store(os.path.join(self.tmp, "beats.db"))
        filled(self.store)
        self.sound = sound.Sound(self.store)

    def tearDown(self):
        self.store.close()
        super().tearDown()

    def ids(self, songs):
        return [s["item_id"] for s in songs]

    def test_songs_like_a_song_are_its_style_varied(self):
        found = self.sound.similar("t1", 5)
        ids = self.ids(found)
        self.assertNotIn("t1", ids)  # never the song itself
        self.assertNotIn("t6", ids)  # nor the same song on another edition (same title and artist)
        self.assertEqual(set(ids[:4]), {"t2", "t3", "t4", "t5"})  # the other trance songs first
        self.assertEqual(ids[4], "m1")  # then the one leaning towards house
        self.assertLess(found[0]["distance"], 0.1)
        self.assertEqual(found, sorted(found, key=lambda s: s["distance"]))
        self.assertEqual(set(found[0]), {"item_id", "title", "author", "album", "distance"})  # AudioMuse-AI's layout

    def test_at_most_three_by_one_artist(self):
        ids = self.ids(self.sound.similar("t5", 10))
        self.assertEqual(sum(1 for i in ids if i in ("t1", "t2", "t3", "t4", "t6")), sound.PER_ARTIST)

    def test_the_same_recording_twice_is_left_out(self):
        twin = song("t9", "Rise (Remaster)", "Zed", "Again", "trance", wobble=2)  # t2's vector, another name
        filled(self.store, songs=[twin])
        ids = self.ids(sound.Sound(self.store).similar("t5", 10))
        self.assertTrue(("t2" in ids) != ("t9" in ids), ids)

    def test_same_mood_keeps_only_songs_that_feel_alike(self):
        self.assertIn("h1", self.ids(self.sound.similar("h2", 10)))
        self.assertNotIn("h1", self.ids(self.sound.similar("h2", 10, same_mood=True)))  # much happier than Swing

    def test_a_song_whose_sound_isnt_known_says_so(self):
        with self.assertRaises(sound.Unknown):
            self.sound.similar("nope")

    def test_a_path_goes_from_one_song_to_the_other(self):
        path = self.ids(self.sound.path("t1", "h1", 5))
        self.assertEqual((path[0], path[-1], len(path)), ("t1", "h1", 5))
        self.assertEqual(len(set(path)), 5)
        # Trance first, then the songs in between, then house.
        self.assertLess(path.index("m1"), path.index("m2"))
        self.assertTrue(path[1].startswith(("t", "m")) and path[-2].startswith(("h", "m")), path)

    def test_a_path_can_end_in_a_mood(self):
        centres = {"relaxed": [{"centroid": [float(x) for x in vector("jazz")], "top_tags": {"jazz": 0.9, "mellow": 0.5}}]}
        self.sound._centres = centres
        path = self.ids(self.sound.path("t1", self.sound.centre("relaxed"), 4))
        self.assertEqual((path[0], len(path)), ("t1", 4))
        self.assertIn(path[-1], ("j1", "j2"))
        self.assertEqual(self.sound.centre_list(), {"relaxed": [
            {"index": 0, "top_tags": ["jazz", "mellow"], "n_songs": 0, "mood_score": 0, "cluster_id": 0}]})
        self.assertEqual(self.ids(self.sound.mood("relaxed", 0, 2)), ["j1", "j2"])
        with self.assertRaises(sound.Unknown):
            self.sound.centre("cheerful")

    def test_a_blend_takes_from_each_song_and_keeps_away_from_others(self):
        ids = self.ids(self.sound.blend(["t1", "j1"], n=6))
        self.assertTrue({"t2", "j2"} <= set(ids), ids)
        self.assertFalse({"t1", "j1"} & set(ids))  # never the songs blended
        away = self.ids(self.sound.blend(["m1"], avoid=["h1"], n=10))
        self.assertFalse({"h1", "h2", "h3"} & set(away), away)  # nothing near the song to avoid
        self.assertIn("t5", away)

    def test_your_sound_is_near_what_you_play_most(self):
        ids = self.ids(self.sound.yours(["j1", "j2"], 3))
        self.assertEqual(set(ids[:2]), {"j1", "j2"})

    def test_only_the_servers_own_songs_answer(self):
        self.store.link_songs("emby", "lib", [{**{k: s[k] for k in s if k not in ("vector", "other")}, "id": "e-" + s["id"]}
                                              for s in LIBRARY if s["id"] in ("t1", "t5", "h1")])
        ids = self.ids(sound.Sound(self.store).similar("t1", 10, server="emby"))
        self.assertEqual(ids, ["t5", "h1"])

    def test_similar_albums_and_artists(self):
        name, artist, albums = self.sound.similar_albums("t1")
        self.assertEqual((name, artist), ("Up", "Ana"))
        names = [a["album"] for a in albums]
        self.assertNotIn("Up (Deluxe)", names)  # another edition of the album itself
        self.assertEqual(names[:2], ["Higher", "Tops"])
        self.assertEqual(set(albums[0]), {"album", "artist", "track_id", "score"})
        who, artists = self.sound.similar_artists("t1")
        self.assertEqual((who, artists[0]["artist"]), ("Ana", "Bo"))
        self.assertEqual(self.sound.similar_artists(artist="flo")[1][0]["artist"] in ("Cy", "Di", "Ed", "Gus", "Ana", "Bo"), True)

    def test_new_sound_is_read_when_the_table_changes(self):
        self.assertEqual(self.sound.count(), len(LIBRARY))
        filled(self.store, songs=[song("r2", "Chord", "Hal", "Louder", "rock", wobble=14)])
        self.assertEqual(self.sound.count(), len(LIBRARY) + 1)
        self.assertEqual(self.ids(self.sound.similar("r1", 1)), ["r2"])

    def test_audiomuses_tempo_kept_by_the_first_builds_goes(self):
        """0.25's first builds kept AudioMuse-AI's tempo beside the sound: opening the database removes it."""
        self.store.close()
        import sqlite3

        path = os.path.join(self.tmp, "beats.db")
        db = sqlite3.connect(path)
        db.execute("ALTER TABLE sound ADD COLUMN tempo REAL")
        db.execute("UPDATE sound SET tempo=89.0")
        db.commit()
        db.close()
        self.store = Store(path)
        self.assertNotIn("tempo", {r[1] for r in self.store.db.execute("PRAGMA table_info(sound)")})
        self.assertEqual(sound.Sound(self.store).count(), len(LIBRARY))  # the sound itself stays

    def test_a_song_another_server_still_has_isnt_forgotten(self):
        emby = [{**{k: s[k] for k in s if k not in ("vector", "other")}, "id": "e-" + s["id"]} for s in LIBRARY[:2]]
        self.store.link_songs("emby", "lib", emby)
        self.assertEqual(self.store.gone_from("main", "1", [s["id"] for s in LIBRARY[2:]]), ["t1", "t2"])
        self.assertEqual(self.store.remove_items("main", ["t1", "t2"]), 0)  # Emby still has both
        self.assertIsNotNone(self.store.song("t1"))
        self.assertEqual(self.store.remove_items("emby", ["e-t1"]), 1)  # now nobody has t1
        self.assertIsNone(self.store.song("t1"))
        self.assertEqual(sound.Sound(self.store).count(), len(LIBRARY) - 1)

    def test_moods_as_audiomuse_writes_them(self):
        self.assertEqual(sound.pairs("rock:0.61, pop:0.2,bad,x:y"), {"rock": 0.61, "pop": 0.2})
        self.assertEqual(sound.pairs(None), {})


def word_vector(axis, lean=None):
    """A vector in the space words share with songs: along ``axis``, a little towards ``lean``."""
    v = np.zeros(sound.WORD_DIMENSIONS, np.float32)
    v[axis] = 1.0
    if lean is not None:
        v[lean] = 0.4
    return v


class FakeWords:
    """The text model's stand-in: "control", "deep" and "waltz" are the catalogue's three songs' directions."""

    available = steers = True
    loaded, load_seconds = True, 0.1
    AXES = {"control": 0, "deep": 1, "waltz": 2}

    def __init__(self):
        self.steered = []

    def embed(self, text):
        v = np.zeros(sound.WORD_DIMENSIONS, np.float32)
        for word, axis in self.AXES.items():
            if word in text.lower():
                v[axis] += 1.0
        if not v.any():
            v[100] = 1.0
        return v / np.linalg.norm(v)

    def terms(self, asked):
        known = [{"term": t["term"], "weight": 3.0, "direction": t.get("direction", "more")}
                 for t in asked or [] if t.get("term") in self.AXES]
        return known, [f'"{t.get("term")}" is not a validated concept for this library'
                       for t in asked or [] if t.get("term") not in self.AXES]

    def steer(self, vector, terms):
        self.steered.append(terms)
        out = vector.copy()
        for t in terms:
            out[self.AXES[t["term"]]] += 5.0 if t["direction"] == "more" else -5.0
        return out

    def catalogue(self):
        return {"available": True, "reason": None, "max_terms": 10, "alpha_steps": [1.0, 3.0], "default_alpha": 3.0,
                "categories": [{"category": "genre", "label": "Genre", "terms": [{"term": "deep", "grounding": 0.4}]}]}

    def unload_if_idle(self, now=None):
        return False


class FakeHearer:
    """The stand-in for the models that hear a song: each of the catalogue's songs sounds like its own direction."""

    loaded, load_seconds = True, 0.1

    def __init__(self, available=True):
        self.available, self.heard = available, []

    def __call__(self, path):
        with open(path) as f:
            song_id = f.read().strip()
        self.heard.append(song_id)
        if song_id == "t96w":
            raise RuntimeError("The song is too short to hear (under 3 seconds)")
        at = [s["id"] for s in CATALOG].index(song_id)
        return {"musicnn": vector("trance", wobble=at + 1).tobytes(), "clap": word_vector(at).tobytes(),
                "moods": "electronic:0.600,dance:0.550", "other": "danceable:0.60,happy:0.50", "energy": 0.61}


def track(s, **changes):
    """A song's sound and feel, as the hearer gives them, for :func:`kept_row`."""
    return {"id": s["id"], "title": s["title"], "artist": s["artist"], "album_artist": s["artist"], "album": s.get("album", "Test"),
            "year": 2020, "tempo": 128.0, "key": "A", "scale": "minor", "mood_vector": "trance:0.6,house:0.2",
            "energy": 0.4, "other_features": "danceable:0.7,happy:0.4", "rating": None, "umap_x": None, "umap_y": None,
            "fp": "fp-" + s["id"], "embedding": base64.b64encode(vector("trance", wobble=CATALOG.index(s) + 1).tobytes()).decode(),
            "clap_embedding": base64.b64encode(word_vector(CATALOG.index(s)).tobytes()).decode(), **changes}


def kept_row(song_id, t):
    """``t`` (:func:`track`) as a row of the sound table."""
    return {"song_id": song_id, "musicnn": base64.b64decode(t["embedding"]),
            "clap": base64.b64decode(t["clap_embedding"]) if t.get("clap_embedding") else None,
            "key": t.get("key"), "scale": t.get("scale"), "moods": t.get("mood_vector"), "other": t.get("other_features"),
            "energy": t.get("energy"), "album_artist": t.get("album_artist"), "year": t.get("year"), "fp": t.get("fp")}


class AnswersTest(Service):
    """What songs sound like, answered to the app through the running service."""

    def heard(self):
        """Every song of the catalog (but the broken one) with its sound here, as if heard; the status' sound."""
        self.app.store.save_sound([kept_row(s["id"], track(s)) for s in CATALOG if s["id"] != "bad"], source="beats")
        end = time.time() + 5
        while time.time() < end:
            state = self.call("/api/status")["sound"]
            if state["sound"] == 3:
                return state
            time.sleep(0.05)
        self.fail("the sound didn't show")

    def test_the_songs_sound_answers_the_app(self):
        self.start()
        self.call("/api/libraries", "POST", {"ids": ["1"]})
        self.wait_done()
        state = self.heard()
        self.assertEqual((state["sound"], state["words"], state["songs"]), (3, 3, 4))
        self.assertNotIn("pull", state)  # 0.43: nothing is taken over from anywhere

        caps = self.call("/api/v1/capabilities")["features"]
        self.assertEqual((caps["sound"]["available"], caps["sound"]["songs"], caps["sound"]["prefix"]),
                         (True, 3, "/api/v1/audiomuse"))
        self.assertIn("/api/similar_tracks", caps["sound"]["answers"])
        self.assertEqual(caps["similar"]["source"], "beats")

        am = "/api/v1/audiomuse"
        like = self.call(f"{am}/api/similar_tracks?item_id=t140&n=5")
        self.assertEqual([s["item_id"] for s in like], ["t122", "t96w"])
        self.assertEqual(like[0]["author"], "House Test")
        feats = self.call(f"{am}/api/sync?ids=t122,t140,bad&include_embeddings=false")["tracks"]
        # The tempo is Beat This!'s (the owner, 11 Nov).
        self.assertEqual([(t["id"], t["tempo"], t["tempo_source"], t["other_features"]) for t in feats],
                         [("t122", 122.0, "beat_this", "danceable:0.7,happy:0.4"),
                          ("t140", 140.0, "beat_this", "danceable:0.7,happy:0.4")])
        self.assertEqual({(t["key"], t["scale"]) for t in feats}, {("A", "minor")})  # this service's own measure
        self.assertNotIn("tempo", {r[1] for r in self.app.store.db.execute("PRAGMA table_info(sound)")})
        self.assertNotIn("embedding", feats[0])
        path = self.call(f"{am}/api/find_path?start_song_id=t140&end_song_id=t96w&max_steps=3&path_fix_size=true")["path"]
        self.assertEqual([s["item_id"] for s in path], ["t140", "t122", "t96w"])
        blend = self.call(f"{am}/api/alchemy", "POST", {"items": [{"id": "t140", "op": "ADD", "type": "song"}], "n": 5})
        self.assertEqual([s["item_id"] for s in blend["results"]], ["t122", "t96w"])
        self.assertIn("happy", self.call(f"{am}/api/mood_centroids"))
        self.assertTrue(self.call(f"{am}/api/similar_tracks?mood=happy&centroid_index=0&n=2"))
        # "Your sound": from what this server's user has played most.
        self.fake.played = {"t96w": 9, "t122": 2}
        yours = self.call(f"{am}/api/sonic_fingerprint/generate", "POST", {"n": 3})
        self.assertEqual([s["item_id"] for s in yours][0], "t96w")
        self.assertEqual(self.call(f"{am}/api/sonic_fingerprint/generate", "POST", {"n": 2, "ids": ["t140"]})[0]["item_id"], "t140")

        # Siansa Beats' own paths (0.44) answer exactly as the older ones, and the capabilities say where each is.
        own = "/api/v1/sound"
        self.assertEqual(self.call(f"{own}/similar?item_id=t140&n=5"), like)
        self.assertEqual(self.call(f"{own}/songs?ids=t122,t140,bad&include_embeddings=false")["tracks"], feats)
        self.assertEqual(self.call(f"{own}/path?start_song_id=t140&end_song_id=t96w&max_steps=3&path_fix_size=true")["path"], path)
        self.assertEqual(self.call(f"{own}/blend", "POST", {"items": [{"id": "t140", "op": "ADD", "type": "song"}], "n": 5}), blend)
        self.assertEqual(self.call(f"{own}/moods"), self.call(f"{am}/api/mood_centroids"))
        self.assertEqual(self.call(f"{own}/yours", "POST", {"n": 2, "ids": ["t140"]})[0]["item_id"], "t140")
        paths = self.call("/api/v1/capabilities")["features"]["sound"]["paths"]
        self.assertEqual(paths["/api/similar_tracks"], "/api/v1/sound/similar")
        self.assertEqual(set(paths), set(caps["sound"]["answers"]))  # one for every answer, by the name older apps know
        with self.assertRaises(urllib.error.HTTPError) as e:
            self.call(f"{own}/nothing")
        self.assertEqual(e.exception.code, 404)

        # Similar albums and artists come from here.
        artists = self.call("/api/v1/similar/artists?track_id=t140")
        self.assertEqual((artists["source"], {a["artist"] for a in artists["similar"]}), ("beats", {"House Test", "Three"}))
        self.assertEqual(self.call("/api/v1/similar/albums?track_id=t140")["source"], "beats")

        with self.assertRaises(urllib.error.HTTPError) as e:
            self.call(f"{am}/api/similar_tracks?item_id=bad")  # known, but its sound isn't
        self.assertEqual(e.exception.code, 404)
        with self.assertRaises(urllib.error.HTTPError) as e:
            self.call(f"{am}/api/similar_tracks?item_id=who")
        self.assertEqual(e.exception.code, 404)

    def test_claps_view_of_a_songs_families_is_a_look_only(self):
        """0.38: the families' shares and what the styles would be, from the stored CLAP vector; nothing kept."""
        self.words = FakeWords()
        self.start()
        self.call("/api/libraries", "POST", {"ids": ["1"]})
        self.wait_done()
        self.heard()
        got = self.call("/api/v1/songs/families?ids=t140,nope")["songs"]
        self.assertEqual(got["nope"]["error"], "not a song here")
        fam = got["t140"]["families"]
        self.assertEqual(len(fam["share"]), 15)
        self.assertAlmostEqual(sum(fam["share"].values()), 1.0, places=2)
        self.assertEqual(got["t140"]["song"]["title"], "Control")
        self.assertIn("gated", got["t140"])

    def test_music_is_found_by_describing_it(self):
        """The search by words (0.26): the songs nearest the words, steered by concepts."""
        self.words = FakeWords()
        self.start()
        self.call("/api/libraries", "POST", {"ids": ["1"]})
        self.wait_done()
        am = "/api/v1/audiomuse"
        caps = self.call("/api/v1/capabilities")["features"]
        self.assertEqual((caps["words"]["available"], caps["sound"]["answers"]), (False, []))  # nothing heard yet
        with self.assertRaises(urllib.error.HTTPError) as e:
            self.call(f"{am}/api/clap/search", "POST", {"query": "deep"})
        self.assertEqual(e.exception.code, 503)

        self.heard()
        caps = self.call("/api/v1/capabilities")["features"]
        self.assertEqual((caps["words"]["available"], caps["words"]["songs"], caps["words"]["steering"]), (True, 3, True))
        self.assertIn("/api/clap/search", caps["sound"]["answers"])
        self.assertIn("/api/clap/concepts", caps["sound"]["answers"])

        found = self.call(f"{am}/api/clap/search", "POST", {"query": "something deep", "limit": 2})
        self.assertEqual(self.call("/api/v1/sound/search", "POST", {"query": "something deep", "limit": 2}), found)  # 0.44's own path
        self.assertEqual(self.call("/api/v1/sound/concepts"), self.call(f"{am}/api/clap/concepts"))
        self.assertEqual(caps["sound"]["paths"]["/api/clap/search"], "/api/v1/sound/search")
        self.assertEqual((found["query"], found["count"]), ("something deep", 2))
        self.assertEqual(found["results"][0]["item_id"], "t122")
        self.assertEqual(set(found["results"][0]), {"item_id", "title", "author", "album", "similarity"})
        self.assertAlmostEqual(found["results"][0]["similarity"], 1.0, places=4)
        # Steered towards another concept; one it doesn't know is left out and said.
        steered = self.call(f"{am}/api/clap/search", "POST", {"query": "deep", "limit": 1, "steering": [
            {"term": "waltz", "direction": "more"}, {"term": "bagpipes"}]})
        self.assertEqual(steered["results"][0]["item_id"], "t96w")
        self.assertEqual(steered["steering"], [{"term": "waltz", "weight": 3.0, "direction": "more"}])
        self.assertEqual(steered["steering_warnings"], ['"bagpipes" is not a validated concept for this library'])
        with self.assertRaises(urllib.error.HTTPError) as e:  # nothing usable to steer by: refused
            self.call(f"{am}/api/clap/search", "POST", {"query": "deep", "steering": [{"term": "bagpipes"}]})
        self.assertEqual(e.exception.code, 400)
        with self.assertRaises(urllib.error.HTTPError) as e:
            self.call(f"{am}/api/clap/search", "POST", {"query": "  "})
        self.assertEqual(e.exception.code, 400)
        self.assertEqual(self.call(f"{am}/api/clap/concepts")["categories"][0]["terms"][0]["term"], "deep")
        self.assertEqual(self.call("/api/status")["sound"]["text_model"]["available"], True)

    def test_new_songs_are_heard_here(self):
        """0.27: Siansa Beats hears each song's sound itself while it analyses it; what can't be heard is kept and
        not tried again."""
        self.hearer, self.words = FakeHearer(), FakeWords()
        self.start()
        self.call("/api/libraries", "POST", {"ids": ["1"]})
        self.wait_done()
        state = self.call("/api/status")["sound"]
        self.assertEqual((state["hears"], state["sound"], state["words"], state["sources"]),
                         (True, 2, 2, {"audiomuse": 0, "beats": 2, "failed": 1}))
        self.assertEqual(sorted(self.hearer.heard), ["t122", "t140", "t96w"])  # "bad" can't even be decoded
        am = "/api/v1/audiomuse"
        self.assertEqual([s["item_id"] for s in self.call(f"{am}/api/similar_tracks?item_id=t140&n=5")], ["t122"])
        feats = self.call(f"{am}/api/sync?ids=t140")["tracks"][0]
        self.assertEqual((feats["tempo"], feats["mood_vector"], feats["energy"], feats["other_features"]),
                         (140.0, "electronic:0.600,dance:0.550", 0.61, "danceable:0.60,happy:0.50"))
        self.assertEqual(feats["energy_source"], "beats")  # heard here, not taken over (0.34)
        self.assertIn("arousal", feats)  # 0.35: how energetic it is; None here, without the model
        self.assertIsNone(feats["arousal"])
        self.assertEqual(self.call(f"{am}/api/clap/search", "POST", {"query": "deep"})["results"][0]["item_id"], "t122")
        # The one that couldn't be heard isn't tried again and again.
        self.assertEqual(self.app.store.without_sound(), [])

    def test_songs_analysed_before_are_heard_when_theres_nothing_else_to_do(self):
        self.hearer = FakeHearer(available=False)  # an image from before 0.27
        self.start()
        self.call("/api/libraries", "POST", {"ids": ["1"]})
        self.wait_done()
        self.assertEqual(self.call("/api/status")["sound"]["sound"], 0)
        self.assertEqual(sorted(self.app.store.without_sound()), ["t122", "t140", "t96w"])
        self.hearer.available = True  # after the update
        self.app.worker.HEAR_IDLE_SECONDS = 0.2
        end = time.time() + 90
        while time.time() < end and self.app.store.without_sound():
            time.sleep(0.2)
        self.assertEqual(self.app.store.without_sound(), [])
        self.assertEqual(self.call("/api/status")["sound"]["sources"], {"audiomuse": 0, "beats": 2, "failed": 1})

    def read_again(self):
        self.call("/api/libraries/read", "POST", {})
        end = time.time() + 15
        while time.time() < end and self.call("/api/status")["reading"]:
            time.sleep(0.05)
        return [e["message"] for e in self.call("/api/status")["events"]]

    def test_songs_gone_from_the_server_are_forgotten(self):
        """0.27's cleaning: a song removed from the server goes from here with its analysis when its
        library is read again; a library that suddenly lists far fewer songs is left alone, and that's said."""
        self.hearer = FakeHearer()
        self.start()
        self.call("/api/libraries", "POST", {"ids": ["1"]})
        self.wait_done()
        self.assertEqual(self.call("/api/status")["counts"]["songs"], 4)
        everything = list(self.fake.songs)
        self.fake.songs[:] = [s for s in everything if s["id"] != "t122"]
        said = self.read_again()
        self.assertTrue(any("1 songs are gone from the server; 1 forgotten here with their analysis" in m for m in said), said[:4])
        self.assertEqual(self.call("/api/status")["counts"]["songs"], 3)
        self.assertIsNone(self.app.store.song("t122"))
        self.assertEqual(self.app.store.sound_rows(["t122"]), {})
        self.assertEqual(list(self.app.store.tempos(["t122", "t140"], self.app.config.model)), ["t140"])
        with self.assertRaises(urllib.error.HTTPError) as e:
            self.call("/api/v1/audiomuse/api/similar_tracks?item_id=t122")
        self.assertEqual(e.exception.code, 404)

        # Half the library missing at once: more likely the server than a clear-out. Nothing goes.
        self.app.worker.CLEAN_MOST = 0
        self.fake.songs[:] = [s for s in everything if s["id"] == "t140"]
        said = self.read_again()
        self.assertTrue(any("too many to be sure of. Nothing was removed" in m for m in said), said[:4])
        self.assertEqual(self.call("/api/status")["counts"]["songs"], 3)
        self.fake.songs[:] = everything

    def test_before_a_songs_sound_is_known_similar_music_is_empty_not_an_error(self):
        """0.43: nothing else is asked; the app gets an empty list with why."""
        self.start()
        self.call("/api/libraries", "POST", {"ids": ["1"]})
        self.wait_done()
        caps = self.call("/api/v1/capabilities")["features"]
        self.assertEqual((caps["sound"]["available"], caps["similar"]), (False, {"available": False, "source": "beats"}))
        answer = self.call("/api/v1/similar/artists?artist=Ecsape")
        self.assertEqual((answer["similar"], answer["source"]), ([], "beats"))
        for gone in ("/api/audiomuse", "/api/sound/pull"):
            with self.assertRaises(urllib.error.HTTPError) as e:
                self.call(gone, "POST", {})
            self.assertEqual(e.exception.code, 404, gone)


class WordsTest(unittest.TestCase):
    """The text model's own code, without the model (the image's check runs it for real): what's asked to steer by,
    and the concepts' list."""

    def setUp(self):
        from beats.words import Words

        self.words = Words("/nowhere")

    def test_an_image_without_the_model_says_so(self):
        from beats.words import NotThere

        self.assertFalse(self.words.available)
        with self.assertRaises(NotThere):
            self.words.embed("chill")
        self.assertEqual(self.words.catalogue()["available"], False)
        self.assertEqual(self.words.terms([{"term": "piano"}]), ([], ["concept steering is unavailable"]))

    def test_terms_are_cleaned_as_audiomuse_cleans_them(self):
        self.words.models_dir = os.path.dirname(__file__)  # the two steering models' names, as empty files
        made = [os.path.join(self.words.models_dir, n) for n in
                ("clap_text_model.onnx", "dclap_sae_k20_d1024_best_encoder.onnx", "dclap_sae_k20_d1024_best_decoder.onnx")]
        for path in made:
            open(path, "w").close()
        try:
            terms, problems = self.words.terms([{"term": "piano", "weight": 4, "direction": "LESS"}, "trance",
                                                {"term": "piano"}, {"term": "bagpipes"}, {"term": "choir", "weight": "x"},
                                                {"term": "guitar", "direction": "sideways"}, 7, {"term": " "}])
            self.assertEqual(terms, [{"term": "piano", "weight": 3.0, "direction": "less"},
                                     {"term": "trance", "weight": 3.0, "direction": "more"}])
            self.assertEqual(len(problems), 6)
            self.assertIn('"bagpipes" is not a validated concept for this library', problems)
            self.assertEqual(self.words.terms("piano")[1], ["steering must be a list of {term, weight, direction} objects"])
            catalogue = self.words.catalogue()
            self.assertEqual([c["category"] for c in catalogue["categories"]],
                             ["genre", "instrument", "vocals", "mood", "production", "tempo", "era"])
            self.assertEqual(sum(len(c["terms"]) for c in catalogue["categories"]), 109)
            self.assertEqual((catalogue["max_terms"], catalogue["default_alpha"]), (10, 3.0))
        finally:
            for path in made:
                os.remove(path)


if __name__ == "__main__":
    unittest.main()
