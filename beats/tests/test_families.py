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
        song[0], song[1] = 0.9, 0.1  # mostly country, a touch electronic
        scores = fam.scores(song)
        self.assertGreater(scores["share"]["Folk, World, & Country"], 0.9)
        self.assertLess(scores["share"]["Electronic"], 0.05)
        # The Discogs model's view of Kacey Musgraves' "Mother" (the owner, 13 Nov), with CLAP's veto: the electronic
        # family isn't heard (cosine 0.2 of the song's 0.9/0.2 mix → a quarter up the ramp), country is.
        styles = [("Electronic---Dance-pop", 0.70), ("Electronic---Electroclash", 0.69), ("Electronic---Electro House", 0.67),
                  ("Folk, World, & Country---Country", 0.63)]
        gated = families.gated(styles, scores["cosine"])
        self.assertEqual(gated[0][0], "Folk, World, & Country---Country")
        self.assertAlmostEqual(gated[0][1], 0.63, places=3)
        self.assertLess(gated[1][1], 0.2)

    def test_the_veto_is_absolute_not_against_the_strongest_family(self):
        # Vocal house (the owner's Gorgon City, 13 Nov): pop 0.43, electronic 0.34 — both heard, (nearly) nothing lost.
        styles = [("Electronic---House", 0.52), ("Electronic---Bassline", 0.27)]
        gated = families.gated(styles, {"Pop": 0.43, "Electronic": 0.34})
        self.assertEqual([l for l, _ in gated], ["Electronic---House", "Electronic---Bassline"])
        self.assertGreater(gated[0][1], 0.52 * 0.9)
        self.assertEqual(families.gated(styles, {"Pop": 0.43, "Electronic": 0.36}), styles)
        # A piano ballad (Mother: electronic 0.03): the electronic styles go entirely.
        self.assertEqual(families.gated(styles, {"Classical": 0.32, "Electronic": 0.03}), [("Electronic---House", 0.0), ("Electronic---Bassline", 0.0)])
        self.assertEqual((families.heard(0.15), round(families.heard(0.25), 6), families.heard(0.35)), (0.0, 0.5, 1.0))

    def test_an_electronic_song_is_left_as_it_was(self):
        fam = families.Families(AxisWords())
        song = np.zeros(512, np.float32)
        song[1] = 1.0
        scores = fam.scores(song)
        styles = [("Electronic---Bassline", 0.44), ("Electronic---UK Garage", 0.31)]
        self.assertEqual(families.gated(styles, scores["cosine"]), [("Electronic---Bassline", 0.44), ("Electronic---UK Garage", 0.31)])
        self.assertEqual(fam.ask(song, ["electronic dance music"]), {"electronic dance music": 1.0})

    def test_every_discogs_family_is_asked_about(self):
        self.assertEqual(len(families.FAMILIES), 15)
        self.assertEqual(families.family_of("Electronic---Electro House"), "Electronic")
        self.assertIsNone(families.family_of("Rock"))


if __name__ == "__main__":
    unittest.main()
