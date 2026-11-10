"""A song's genres by vote: what its sound says (:mod:`beats.style`, Essentia's Discogs-EffNet) first, its own genre
tags second (partly set by hand, partly guessed by tools from the album), and room for more sources later (Last.fm).

Every genre is one of the model's 400 Discogs styles (``Electronic---Progressive Trance``), so tags are matched onto
them: "Drum And Bass", "drum & bass" and "DnB" are all ``Electronic---Drum n Bass``, "Electronic / Trance" names a
genre and a style. The vote for each style is its sound score (0 to 1), plus :data:`TAG_STYLE` when a tag names that
style, plus :data:`TAG_GENRE` when a tag names its genre (the family). A tag the sound doesn't hear at all still counts,
less than the sound: a song without sound scores is voted on its tags alone. Kept as the ``vote`` rows of the genres
table, so a radio asks for a genre with one query.

Plain Python.
"""

import re

TAG_STYLE = 0.35  # a tag naming the style
TAG_GENRE = 0.10  # a tag naming its genre (Electronic, Rock…)
VOTE_MIN = 0.15  # a style below this (sound and tags together) isn't one of the song's genres…
VOTE_KEEP = 5  # …and a song keeps at most this many
RADIO_MIN = 0.30  # a song is in a genre's radio from this vote on

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
        under two genres (Electro: Electronic and Hip Hop) gives both: :func:`vote` picks one."""
        styles, genres = set(), set()
        for part in parts(tag):
            k = key(part)
            if k in self.by_style:
                styles.update(self.by_style[k])
            if k in self.by_genre:
                genres.add(self.by_genre[k][0].partition("---")[0])
        return styles, genres


def vote(taxonomy, sound, tags):
    """``[(label, score)]``: the song's genres, best first, at most :data:`VOTE_KEEP`, each at least :data:`VOTE_MIN`
    (but always the best one, when there's any vote at all). ``sound``: ``{label: score}`` from the model (may be
    empty); ``tags``: the song's genre tags as they are."""
    scores = {label: float(v) for label, v in (sound or {}).items()}
    tag_styles, tag_genres, shared = set(), set(), []
    for tag in tags or []:
        s, g = taxonomy.match(tag)
        tag_genres |= g
        for part in parts(tag):  # each name once: a style named under two genres counts for one of them
            named = s & set(taxonomy.by_style.get(key(part), []))
            if len(named) > 1:
                shared.append(named)
            else:
                tag_styles |= named
    for named in shared:
        tag_styles.add(_likeliest(taxonomy, named, scores, tag_genres))
    for label in tag_styles:
        scores[label] = scores.get(label, 0.0) + TAG_STYLE
    if tag_genres:
        for label in list(scores):
            if label.partition("---")[0] in tag_genres:
                scores[label] += TAG_GENRE
    ranked = sorted(((l, round(v, 4)) for l, v in scores.items() if v > 0), key=lambda p: -p[1])
    if not ranked:
        return []
    kept = [p for p in ranked[:VOTE_KEEP] if p[1] >= VOTE_MIN]
    return kept or ranked[:1]


def _likeliest(taxonomy, labels, sound, tag_genres):
    """Of one style name under several genres, the one meant: the one the song's sound hears most, else the one in a
    genre its tags name, else the one in the genre with the most styles (Electronic)."""
    heard = [l for l in labels if sound.get(l, 0.0) > 0]
    if heard:
        return max(heard, key=lambda l: sound[l])
    tagged = [l for l in labels if l.partition("---")[0] in tag_genres]
    if tagged:
        return sorted(tagged)[0]
    return max(sorted(labels), key=lambda l: taxonomy.genre_size(l.partition("---")[0]))


def family(voted):
    """The genre (Electronic, Rock…) a song's vote leans to most, or ``None``."""
    totals = {}
    for label, score in voted or []:
        g = label.partition("---")[0]
        totals[g] = totals.get(g, 0.0) + score
    return max(totals, key=totals.get) if totals else None


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
