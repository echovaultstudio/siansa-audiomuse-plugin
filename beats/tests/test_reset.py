"""The way back in from Docker (beats/reset.py): ``docker exec siansa-beats python -m beats.reset``."""
import io
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from beats import reset  # noqa: E402
from beats.art import Art, INDEX_AT_MOST_EVERY  # noqa: E402
from beats.store import Store  # noqa: E402
from beats.vault import Vault  # noqa: E402


class ResetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.store = Store(os.path.join(self.tmp, "beats.db"))
        self.addCleanup(self.store.close)

    def test_it_takes_the_password_and_the_key_away_while_the_service_runs(self):
        self.store.set_setting("access", {"password": Vault.hash_password("correct horse"), "api_key": Vault.hash_key("k" * 32)})
        self.store.set_setting("audiomuse", {"url": "http://am"})
        out = io.StringIO()
        self.assertEqual(reset.main(env={"DATA_DIR": self.tmp}, out=out), 0)
        self.assertIn("the page's password and the app's key", out.getvalue())
        self.assertEqual(self.store.setting("access"), {})  # the running service reads it at its next request
        self.assertEqual(self.store.setting("audiomuse"), {"url": "http://am"})  # nothing else is touched
        self.assertIn("taken away from Docker", self.store.events()[0]["message"])
        # Again: nothing to take, said plainly, and no second line in Activity.
        out = io.StringIO()
        self.assertEqual(reset.main(env={"DATA_DIR": self.tmp}, out=out), 0)
        self.assertIn("open already", out.getvalue())
        self.assertEqual(len([e for e in self.store.events() if "from Docker" in e["message"]]), 1)

    def test_a_wrong_folder_is_said_and_changes_nothing(self):
        out = io.StringIO()
        self.assertEqual(reset.main(env={"DATA_DIR": os.path.join(self.tmp, "nowhere")}, out=out), 1)
        self.assertIn("No Siansa Beats database", out.getvalue())
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "nowhere")))


class PictureIndexTest(unittest.TestCase):
    """The songs by artist and album, for the Listening page's pictures: not read again with every song analysed."""

    def test_it_is_not_made_again_within_a_minute_of_songs_changing(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        store = Store(os.path.join(tmp, "beats.db"))
        self.addCleanup(store.close)
        clock = [1000.0]
        art = Art(store, None, tmp, now=lambda: clock[0])
        reads = []
        rows = store._rows
        store._rows = lambda sql, *a: (reads.append(sql) if "FROM songs WHERE artist" in sql else None) or rows(sql, *a)
        art._songs()
        store.song_changes = getattr(store, "song_changes", 0) + 1  # a song was analysed
        art._songs()
        self.assertEqual(len(reads), 1)  # not again yet
        clock[0] += INDEX_AT_MOST_EVERY + 1
        art._songs()
        self.assertEqual(len(reads), 2)  # a minute on, with the songs changed
        art._songs()
        self.assertEqual(len(reads), 2)  # nothing changed since


if __name__ == "__main__":
    unittest.main()
