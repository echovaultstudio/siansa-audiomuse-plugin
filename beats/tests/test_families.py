"""CLAP's second opinion on a song's genre family (0.38): a look at what it would do to the Discogs styles."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from beats import families  # noqa: E402


class AxisWords:
    """A text model whose space has one axis per family that matters here: country on 0, electronic on 1."""

    def embed(self, text):
        v = np.zeros(512, np.float32)
        if "country" in text:
            v[0] = 1.0
        elif "electronic" in text:
            v[1] = 1.0
        else:
            v[2 + hash(text) % 400] = 1.0
        return v


class FamiliesTest(unittest.TestCase):
    def test_a_country_sounding_song_keeps_its_country_style_and_loses_electro_house(self):
        fam = families.Families(AxisWords())
        song = np.zeros(512, np.float32)
        song[0], song[1] = 0.9, 0.2  # mostly country, a touch electronic
        scores = fam.scores(song)
        self.assertGreater(scores["share"]["Folk, World, & Country"], 0.9)
        self.assertLess(scores["share"]["Electronic"], 0.05)
        # The Discogs model's view of Kacey Musgraves' "Mother" (the owner, 13 Nov), weighed by CLAP's.
        styles = [("Electronic---Dance-pop", 0.70), ("Electronic---Electroclash", 0.69), ("Electronic---Electro House", 0.67),
                  ("Folk, World, & Country---Country", 0.63)]
        gated = families.gated(styles, scores["share"])
        self.assertEqual(gated[0][0], "Folk, World, & Country---Country")
        self.assertAlmostEqual(gated[0][1], 0.63, places=3)
        self.assertLess(gated[1][1], 0.05)

    def test_an_electronic_song_is_left_as_it_was(self):
        fam = families.Families(AxisWords())
        song = np.zeros(512, np.float32)
        song[1] = 1.0
        scores = fam.scores(song)
        styles = [("Electronic---Bassline", 0.44), ("Electronic---UK Garage", 0.31)]
        self.assertEqual(families.gated(styles, scores["share"]), [("Electronic---Bassline", 0.44), ("Electronic---UK Garage", 0.31)])

    def test_every_discogs_family_is_asked_about(self):
        self.assertEqual(len(families.FAMILIES), 15)
        self.assertEqual(families.family_of("Electronic---Electro House"), "Electronic")
        self.assertIsNone(families.family_of("Rock"))


if __name__ == "__main__":
    unittest.main()
