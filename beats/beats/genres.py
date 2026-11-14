"""A song's genres: what its sound says (:mod:`beats.style`, Essentia's Discogs-EffNet), and nothing else.

Every genre is one of the model's 400 Discogs styles (``Electronic---Progressive Trance``). A song is in a genre when
its sound scores that style :data:`HEARD_MIN` or more *and* the style is among its :data:`HEARD_RANK` best (0.36, the
owner, 12 Nov: a country album's faint fourth and fifth guesses had put it in Electro House), and always in the one it
scores highest; there is no limit on how many songs a genre has. A radio for a whole family ("Electronic") takes only
the songs whose best style is in it. The songs' own genre tags play no part (before 0.28
they were voted together with the sound): they stay what Siansa shows on an album's or an artist's page.

Names are still matched onto the styles, for what a radio is asked for: "Drum And Bass", "drum & bass" and "DnB" are
all ``Electronic---Drum n Bass``, "Electronic / Trance" names a genre and a style. The tags are matched the same way
only to report how often the sound agrees with them (:func:`agreement`).

Plain Python.
"""

import re

HEARD_MIN = 0.15  # a song is in a style's genre when its sound scores it this or more (and always in its best one)
HEARD_RANK = 3  # …and the style is among the song's best this many; a family radio takes the best style only (1)

# Other names for the same thing, after normalising (lower case, "&" as "and", only letters and digits).
ALIASES = {
    "drumandbass": "drumnbass", "dnb": "drumnbass", "dandb": "drumnbass", "drumbass": "drumnbass",
    "jungledrumandbass": "jungle", "hiphoprap": "hiphop", "rap": "hiphop", "rnb": "rhythmandblues",
    "randb": "rhythmandblues", "electronica": "electronic", "edm": "electronic", "dance": "electronic",
    "2step": "ukgarage", "psychedelictrance": "psytrance", "upliftingtrance": "trance", "vocaltrance": "trance",
    "liquidfunk": "drumnbass", "liquiddrumandbass": "drumnbass", "neurofunk": "drumnbass", "indie": "indierock",
    "alternative": "alternativerock", "altrock": "alternativerock", "metal": "heavymetal",
    "singersongwriter": "folk", "rocknroll": "rockandroll", "classic": "classical", "countrymusic": "country",
    "ambientmusic": "ambient", "chillout": "downtempo", "chill": "downtempo",
}

SPLIT = re.compile(r"\s*[/;|]\s*|\s*,\s*(?![^()]*\))")


def key(name):
    """The name as letters and digits only, "&" read as "and" (so "Drum & Bass" and "Drum and Bass" agree)."""
    text = (name or "").lower().replace("&", " and ").replace("'n'", "n").replace("’n’", "n")
    plain = re.sub(r"[^a-z0-9]", "", text)
    return ALIASES.get(plain, plain)


def parts(tag):
    """The genres one tag names: "Electronic / Trance" is two, "Drum And Bass; Liquid Funk" two."""
    return [p for p in SPLIT.split(tag or "") if p.strip()]


class Taxonomy:
    """The model's styles, findable by a style's or a genre's name."""

    def __init__(self, labels):
        self.labels = list(labels)
        self.by_style, self.by_genre = {}, {}
        for label in self.labels:
            genre, _, style = label.partition("---")
            self.by_style.setdefault(key(style or genre), []).append(label)
            self.by_genre.setdefault(key(genre), []).append(label)

    def genre_size(self, genre):
        """How many styles a genre has: Electronic's many make it the likelier home of a name shared with another."""
        return len(self.by_genre.get(key(genre), []))

    def match(self, tag):
        """``(styles, genres)``: the labels a tag names as a style, and the genres (families) it names. A style name
        under two genres (Electro: Electronic and Hip Hop) gives both."""
        styles, genres = set(), set()
        for part in parts(tag):
            k = key(part)
            if k in self.by_style:
                styles.update(self.by_style[k])
            if k in self.by_genre:
                genres.add(self.by_genre[k][0].partition("---")[0])
        return styles, genres


def heard(ranked, minimum=HEARD_MIN, rank=HEARD_RANK):
    """The genres of a song whose sound gave ``ranked`` (``[(label, score)]``, best first): every style at
    ``minimum`` or more among its ``rank`` best, and always the best one."""
    ranked = sorted(ranked or [], key=lambda p: -p[1])
    return [p for i, p in enumerate(ranked) if (p[1] >= minimum and i < rank) or p[1] == ranked[0][1]]


def target(taxonomy, name):
    """What a radio for ``name`` (a tag as the app shows it: "Trance", "Electronic / Trance") plays: ``(styles,
    genres)``. The most precise wins: a name with a style is that style, one that is only a genre is the genre."""
    styles, genres = taxonomy.match(name)
    if styles and genres:  # "Hip Hop / Electro": the Electro of Hip Hop, not Electronic's too
        within = {s for s in styles if s.partition("---")[0] in genres}
        styles = within or styles
    return (styles, set()) if styles else (set(), genres)


def agreement(rows):
    """How often the sound agrees with the tags, over songs with both: ``{"songs", "style", "family"}``, the shares
    whose tagged style is in the sound's top 3, and whose tagged genre is the sound's top genre. ``rows``:
    ``[(sound ranked [(label, score)], taxonomy.match(tags) result)]``."""
    songs = style_hits = family_hits = with_style = 0
    for ranked, (styles, genres) in rows:
        if not ranked or not (styles or genres):
            continue
        songs += 1
        if styles:
            with_style += 1
            if styles & {label for label, _ in ranked[:3]}:
                style_hits += 1
        tagged = genres | {s.partition("---")[0] for s in styles}
        if ranked[0][0].partition("---")[0] in tagged:
            family_hits += 1
    return {"songs": songs, "style": round(style_hits / with_style, 3) if with_style else None,
            "family": round(family_hits / songs, 3) if songs else None}
