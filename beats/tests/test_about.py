"""Genre descriptions from Wikipedia (beats.about), without the network."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from beats import about  # noqa: E402
from beats.store import Store  # noqa: E402


def page(title, extract, description="", kind="standard"):
    return {"type": kind, "title": title, "description": description, "extract": extract,
            "content_urls": {"desktop": {"page": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"}}}


PAGES = {
    "House music": page("House music", "House is a genre of electronic dance music that originated in Chicago in the 1980s.",
                        "Genre of electronic dance music"),
    "House": page("House", "A house is a single-unit residential building.", "Building"),
    "Electro (music)": page("Electro (music)", "Electro is a genre of electronic music and early hip hop.", "Music genre"),
    "Dubstep": page("Dubstep", "Dubstep is a genre of electronic dance music that originated in South London in the early 2000s.",
                    "Genre of electronic dance music"),
    "Mercury": page("Mercury", "Mercury may refer to:", kind="disambiguation"),
}


class AboutTest(unittest.TestCase):
    def test_finds_the_music_article_not_a_namesake(self):
        found = about.lookup("House", PAGES.get)
        self.assertEqual("House music", found["title"])
        self.assertIn("Chicago", found["summary"])
        self.assertEqual("Wikipedia", found["source"])
        self.assertTrue(found["url"].startswith("https://en.wikipedia.org/wiki/"))

    def test_a_style_with_its_family_is_looked_up_by_the_style(self):
        self.assertEqual("Electro (music)", about.lookup("Hip Hop / Electro", PAGES.get)["title"])
        self.assertEqual("Dubstep", about.lookup("dubstep", PAGES.get)["title"])

    def test_nothing_on_it_is_none(self):
        self.assertIsNone(about.lookup("Mercury", PAGES.get))

    def test_long_text_ends_at_a_sentence(self):
        text = "First sentence here. " * 60
        cut = about.trim(text, 100)
        self.assertTrue(cut.endswith("."))
        self.assertLessEqual(len(cut), 100)

    def test_kept_and_asked_again_when_old(self):
        store = Store(os.path.join(tempfile.mkdtemp(), "b.db"))
        asked = []

        def get(title):
            asked.append(title)
            return PAGES.get(title)

        now = [1_000_000.0]
        self.assertEqual("House music", about.about(store, "House", get, lambda: now[0])["title"])
        before = len(asked)
        about.about(store, "House", get, lambda: now[0])
        self.assertEqual(before, len(asked))  # kept
        self.assertIsNone(about.about(store, "Mercury", get, lambda: now[0]))
        before = len(asked)
        about.about(store, "Mercury", get, lambda: now[0] + 60)
        self.assertEqual(before, len(asked))  # nothing found is kept a while too
        now[0] += about.AGAIN_MISSING + 1
        about.about(store, "Mercury", get, lambda: now[0])
        self.assertGreater(len(asked), before)

    def test_wikipedia_out_of_reach_isnt_nothing_on_it(self):
        # A timeout used to be kept as "nothing on it" for a week: the genre page showed none.
        store = Store(os.path.join(tempfile.mkdtemp(), "b.db"))
        now = [1_000_000.0]
        self.assertIsNone(about.about(store, "House", lambda title: about.FAILED, lambda: now[0]))
        self.assertTrue(store.setting(about.key_of("House"))["failed"])
        self.assertIsNone(about.about(store, "House", PAGES.get, lambda: now[0] + 60))  # not asked again at once
        now[0] += about.AGAIN_FAILED + 1
        self.assertEqual("House music", about.about(store, "House", PAGES.get, lambda: now[0])["title"])
        # One title out of reach and none of the others it: tried again, not "nothing".
        self.assertIs(about.lookup("Mercury", lambda t: about.FAILED if t == "Mercury (music)" else PAGES.get(t)), about.FAILED)

    def test_the_keeper_fetches_every_genre_and_answers_at_once(self):
        store = Store(os.path.join(tempfile.mkdtemp(), "b.db"))
        asked = []

        def get(title):
            asked.append(title)
            return PAGES.get(title)

        keeper = about.Keeper(store, lambda: ["House", "Dubstep", "Mercury"], get)
        keeper.PAUSE = 0
        self.assertEqual(keeper.counts()["waiting"], 3)
        self.assertEqual(keeper.run_round(), 3)
        self.assertEqual(keeper.counts(), {"genres": 3, "found": 2, "missing": 1, "failed": 0, "waiting": 0})
        before = len(asked)
        found, pending = keeper.answer("House")
        self.assertEqual((found["title"], pending), ("House music", False))
        self.assertEqual(before, len(asked))  # kept: Wikipedia isn't asked
        self.assertEqual(keeper.run_round(), 0)  # nothing left to look up

    def test_a_slow_lookup_says_pending_and_carries_on(self):
        import threading

        store = Store(os.path.join(tempfile.mkdtemp(), "b.db"))
        go = threading.Event()

        def slow(title):
            go.wait(5)
            return PAGES.get(title)

        keeper = about.Keeper(store, lambda: [], slow)
        keeper.WAIT = 0.2
        self.assertEqual(keeper.answer("Dubstep"), (None, True))
        go.set()
        for _ in range(50):
            found, pending = keeper.answer("Dubstep")
            if found:
                break
            import time
            time.sleep(0.1)
        self.assertEqual(found["title"], "Dubstep")


if __name__ == "__main__":
    unittest.main()
