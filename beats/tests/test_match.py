"""Finding the same song on two music servers without hearing it (beats.match): by its file, then its tags, never
by a guess."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from beats import match  # noqa: E402


def finder(songs):
    """``choose`` over a list of songs here (``id``, ``path``, ``title``, ``artist``, ``album``, ``duration``)."""
    def by(key_of):
        return lambda key: [s for s in songs if key and key_of(s) == key]

    return lambda item: match.choose(item, by(lambda s: match.tail(s.get("path"))), by(match.meta), by(match.loose))


HERE = [
    {"id": "nd1", "path": "/music/Karnivool/Sound Awake/01 - Simple Boy.flac", "title": "Simple Boy",
     "artist": "Karnivool", "album": "Sound Awake", "duration": 346},
    {"id": "nd2", "path": "/music/Karnivool/Sound Awake/02 - Goliath.flac", "title": "Goliath", "artist": "Karnivool",
     "album": "Sound Awake", "duration": 277},
    {"id": "nd3", "path": None, "title": "Last Day of Summer", "artist": "Katy B • Jax Jones", "album": "Singles",
     "duration": 146},
]


class PathTest(unittest.TestCase):
    def test_paths_are_compared_by_their_ends(self):
        self.assertEqual(match.tail("/music/Artist/Album/01 - Song.flac"), "album/01 - song.flac")
        self.assertEqual(match.tail("Album\\01 - Song.flac"), "album/01 - song.flac")
        self.assertEqual(match.tail("song.flac"), "song.flac")
        self.assertIsNone(match.tail(None))
        self.assertEqual(match.shared_end("/music/A/B/c.flac", "/mnt/media/music/A/B/c.flac"), 4)
        self.assertEqual(match.shared_end("/music/A/B/c.flac", "/music/X/B/c.flac"), 2)
        # é as one sign (Navidrome) or as e and an accent (a server on another system): the same name.
        self.assertEqual(match.tail("/m/Yé-Yé/caf\u00e9.mp3"), match.tail("/m/Ye\u0301-Ye\u0301/cafe\u0301.mp3"))

    def test_the_same_file_under_another_folder_is_the_same_song(self):
        find = finder(HERE)
        self.assertEqual(find({"path": "/mnt/media/music/Karnivool/Sound Awake/01 - Simple Boy.flac", "title": "x",
                               "duration": 346.4}), ("nd1", "path"))
        # Navidrome's own paths start at its library: no folder before the artist.
        self.assertEqual(find({"path": "Karnivool/Sound Awake/02 - Goliath.flac", "duration": 277}), ("nd2", "path"))
        # Windows' way of writing it.
        self.assertEqual(find({"path": "D:\\Music\\Karnivool\\Sound Awake\\02 - Goliath.flac"}), ("nd2", "path"))

    def test_the_file_alone_is_not_enough(self):
        find = finder(HERE)
        self.assertEqual(find({"path": "/other/Live Album/01 - Simple Boy.flac", "title": "Other", "duration": 346}),
                         (None, None))

    def test_another_length_is_another_song(self):
        find = finder(HERE)
        self.assertEqual(find({"path": "/x/Karnivool/Sound Awake/01 - Simple Boy.flac", "title": "Simple Boy",
                               "artist": "Karnivool", "album": "Sound Awake", "duration": 400}), (None, None))

    def test_two_files_that_end_alike_are_not_guessed_between(self):
        twice = HERE + [{"id": "copy", "path": "/backup/Karnivool/Sound Awake/01 - Simple Boy.flac", "title": "Simple Boy",
                         "artist": "Karnivool", "album": "Sound Awake", "duration": 346}]
        find = finder(twice)
        self.assertEqual(find({"path": "Karnivool/Sound Awake/01 - Simple Boy.flac", "title": "Simple Boy",
                               "artist": "Karnivool", "album": "Sound Awake", "duration": 346}), (None, None))
        # One folder more decides it.
        self.assertEqual(find({"path": "/srv/music/Karnivool/Sound Awake/01 - Simple Boy.flac"}), ("nd1", "path"))
        self.assertEqual(find({"path": "/srv/backup/Karnivool/Sound Awake/01 - Simple Boy.flac"}), ("copy", "path"))


class TagsTest(unittest.TestCase):
    def test_artists_written_differently_are_the_same(self):
        self.assertEqual(match.artists("Katy B • Jax Jones"), match.artists("Jax Jones; Katy B"))
        self.assertEqual(match.artists("Katy B feat. Jax Jones"), match.artists("Katy B & Jax Jones"))
        self.assertNotEqual(match.artists("Katy B"), match.artists("Katy B, Jax Jones"))

    def test_a_song_without_a_path_is_found_by_its_tags(self):
        find = finder(HERE)
        self.assertEqual(find({"title": "last day of summer", "artist": "Jax Jones; Katy B", "album": "Singles",
                               "duration": 146.3}), ("nd3", "tags"))
        # A path that fits nothing here falls back to the tags too.
        self.assertEqual(find({"path": "/elsewhere/a/b.flac", "title": "Goliath", "artist": "Karnivool",
                               "album": "Sound Awake", "duration": 277}), ("nd2", "tags"))

    def test_title_and_album_need_both_lengths(self):
        find = finder(HERE)
        self.assertEqual(find({"title": "Last Day of Summer", "artist": "Various", "album": "Singles", "duration": 146}),
                         ("nd3", "title"))
        self.assertEqual(find({"title": "Last Day of Summer", "artist": "Various", "album": "Singles"}), (None, None))
        self.assertEqual(find({"title": "Last Day of Summer", "artist": "Various", "album": "Singles", "duration": 200}),
                         (None, None))

    def test_two_songs_with_the_same_tags_are_not_guessed_between(self):
        twice = HERE + [dict(HERE[2], id="again")]
        self.assertEqual(finder(twice)({"title": "Last Day of Summer", "artist": "Katy B • Jax Jones", "album": "Singles",
                                        "duration": 146}), (None, None))

    def test_nothing_to_go_by(self):
        self.assertEqual(finder(HERE)({"title": "", "artist": "", "album": ""}), (None, None))


if __name__ == "__main__":
    unittest.main()
