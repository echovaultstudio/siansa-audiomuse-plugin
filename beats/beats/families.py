"""A second opinion on a song's genre, by CLAP (0.38, the owner, 13 Nov): the Discogs style model read a country album's
piano ballads as Dance-pop and Electro House at 0.7, on the right file, with the spectrogram Essentia would make and
the GPU agreeing with the CPU. The model is at home among electronic releases and misreads clean, sparse productions.

CLAP judges the coarse genre well: how near a song's CLAP vector (kept for every analysed song, for the search by
words) is to "country music", "electronic dance music", "rock music"… Each Discogs style belongs to one of fifteen
families (``Electronic---Electro House``), so a song's styles can be weighed by how much CLAP hears the family.

This module only *looks*, for now: :func:`family_scores` gives the families' scores for one vector, and
:func:`gated` what the song's styles would be, scaled by them. Nothing is kept until the owner has seen it on real
songs (``/api/v1/songs/families``).
"""

import numpy as np

# Discogs's fifteen genres, each as CLAP is asked about it. The label before "---" in a style's name is its family.
FAMILIES = {
    "Blues": "blues music",
    "Brass & Military": "brass band and military marching music",
    "Children's": "children's music",
    "Classical": "classical music",
    "Electronic": "electronic dance music",
    "Folk, World, & Country": "folk, country and world music",
    "Funk / Soul": "funk and soul music",
    "Hip Hop": "hip hop and rap music",
    "Jazz": "jazz music",
    "Latin": "latin music",
    "Non-Music": "spoken word, not music",
    "Pop": "pop music",
    "Reggae": "reggae music",
    "Rock": "rock music",
    "Stage & Screen": "film soundtrack and musical theatre",
}

# How sharply the cosines are turned into shares: CLAP's own logit scale is about this.
SCALE = 25.0

# Whether a family is heard at all, from its cosine alone (the owner's library, 13 Nov: ten electronic tracks scored
# 0.27 to 0.50 for "electronic dance music", a piano ballad 0.03): nothing below LOW, fully above HIGH, a ramp between.
# Absolute, not against the strongest family: vocal house reads as pop first and electronic second, and is still both.
LOW = 0.15
HIGH = 0.35


def family_of(label):
    """The family a Discogs style belongs to ("Electronic---Electro House" → "Electronic")."""
    return label.partition("---")[0] if "---" in label else None


class Families:
    """The families' vectors in CLAP's space, made once with the text model (:class:`beats.words.Words`)."""

    def __init__(self, words):
        self.words = words
        self._vectors = None

    def vectors(self):
        if self._vectors is None:
            self._vectors = {name: self.words.embed(prompt) for name, prompt in FAMILIES.items()}
        return self._vectors

    def ask(self, clap, prompts):
        """The cosine of a song's CLAP vector to each of ``prompts`` (any words), to try wordings by."""
        v = np.asarray(clap, dtype=np.float32)
        v = v / max(float(np.linalg.norm(v)), 1e-8)
        return {p: round(float(np.dot(v, self.words.embed(p))), 4) for p in prompts}

    def scores(self, clap):
        """``{"cosine": {family: cos}, "share": {family: 0..1}}`` for a song's CLAP vector (512 float32), the shares
        a softmax of the cosines at :data:`SCALE`."""
        v = np.asarray(clap, dtype=np.float32)
        length = float(np.linalg.norm(v))
        if length <= 1e-8:
            raise ValueError("an empty vector")
        v = v / length
        cos = {name: float(np.dot(v, vec)) for name, vec in self.vectors().items()}
        logits = np.array([cos[n] for n in cos]) * SCALE
        exp = np.exp(logits - logits.max())
        share = exp / exp.sum()
        return {"cosine": {n: round(c, 4) for n, c in cos.items()},
                "share": {n: round(float(s), 4) for n, s in zip(cos, share)}}


def heard(cosine):
    """How much a family is heard, 0 to 1, from its cosine: the ramp from :data:`LOW` to :data:`HIGH`."""
    return min(1.0, max(0.0, (float(cosine) - LOW) / (HIGH - LOW)))


def gated(styles, cosine):
    """What a song's styles would be with CLAP's veto: each ``(label, score)`` scaled by how much its family is heard
    (:func:`heard` of the family's cosine). A family CLAP doesn't hear takes its styles with it; one it hears keeps
    them whole. Sorted, best first."""
    out = []
    for label, score in styles:
        fam = family_of(label)
        weight = heard(cosine.get(fam, 0.0)) if fam else 1.0
        out.append((label, round(float(score) * weight, 4)))
    return sorted(out, key=lambda p: -p[1])
