"""The same song on two music servers (or for two users of one), found without hearing it: so a song analysed from
Navidrome isn't analysed again when Emby or Jellyfin serve the same file (the owner, 10 Nov; AudioMuse-AI's way since
its 3.0: "analysed only once and mapped to both").

A server's item is matched to a song already here by, in turn:

1. its file: the same path, or paths that end alike (``/music/Artist/Album/01 - Song.flac`` on one server and
   ``Artist/Album/01 - Song.flac`` on the other: each server mounts the folder where it likes), by at least the file
   and the folder it's in;
2. its tags: the same title, artists and album;
3. its title and album alone (servers write several artists differently), when their lengths are known and agree.

Lengths more than :data:`SECONDS_APART` apart never match (another cut of the song), and when two songs here fit
equally well none is taken: an item that can't be placed for sure is its own song, analysed like any new one. A song
analysed twice costs a few seconds; two songs taken for one would give one of them the other's beats.
"""

import re
import unicodedata

SECONDS_APART = 3.0  # servers round a song's length differently: Navidrome to the second, Emby to the tick
TAIL_PARTS = 2  # the file and its folder: what two paths must share at the least


def _text(value):
    """Text as compared: one form of each letter (é as one sign or two), no case, single spaces."""
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).casefold().split())


def path_parts(path):
    """A file's path as its folders and name, as compared: ``["artist", "album", "01 - song.flac"]``."""
    return [p for p in _text(str(path or "").replace("\\", "/")).split("/") if p and p != "."]


def tail(path):
    """The end of a path two servers must share: its file and the folder it's in (the file alone at the top)."""
    parts = path_parts(path)
    return "/".join(parts[-TAIL_PARTS:]) if parts else None


def shared_end(a, b):
    """How many names two paths share from the file up."""
    a, b = path_parts(a), path_parts(b)
    n = 0
    while n < len(a) and n < len(b) and a[-1 - n] == b[-1 - n]:
        n += 1
    return n


_ARTISTS = re.compile(r"\s*(?:[•;/,&+]|\bfeat\.?|\bft\.?|\bfeaturing\b|\bwith\b|\bvs\.?|\bx\b)\s*")


def artists(text):
    """The artists of a song as compared: ``"A • B"``, ``"A; B"`` and ``"B feat. A"`` are the same two."""
    return " | ".join(sorted({a for a in _ARTISTS.split(_text(text)) if a}))


def meta(song):
    """A song's tags as compared (title, artists, album), or None without a title."""
    title = _text(song.get("title"))
    return f"{title}\n{artists(song.get('artist'))}\n{_text(song.get('album'))}" if title else None


def loose(song):
    """A song's title and album as compared, or None without both."""
    title, album = _text(song.get("title")), _text(song.get("album"))
    return f"{title}\n{album}" if title and album else None


def lengths_agree(a, b, both=False):
    """Whether two lengths (seconds) may be the same recording's; an unknown one agrees unless ``both`` are needed."""
    if not a or not b:
        return not both
    return abs(float(a) - float(b)) <= SECONDS_APART


def choose(item, by_tail, by_meta, by_loose):
    """The song here that ``item`` (a server's song: ``path``, ``title``, ``artist``, ``album``, ``duration``) is, as
    ``(song id, how)`` with how "path", "tags" or "title", or ``(None, None)``. ``by_tail``, ``by_meta`` and
    ``by_loose`` give the songs here with that path end, those tags, that title and album: each
    ``{"id", "path", "duration"}``."""
    length = item.get("duration")
    path = item.get("path")
    if path:
        fits = [(shared_end(path, c.get("path")), c) for c in by_tail(tail(path))
                if lengths_agree(length, c.get("duration"))]
        fits = [(n, c) for n, c in fits if n >= min(TAIL_PARTS, len(path_parts(path)))]
        if fits:
            best = max(n for n, _ in fits)
            ids = {c["id"] for n, c in fits if n == best}
            if len(ids) == 1:
                return ids.pop(), "path"
            return None, None  # two files here end alike: not guessed
    key = meta(item)
    if key:
        ids = {c["id"] for c in by_meta(key) if lengths_agree(length, c.get("duration"))}
        if len(ids) == 1:
            return ids.pop(), "tags"
        if len(ids) > 1:
            return None, None
    key = loose(item)
    if key:
        ids = {c["id"] for c in by_loose(key) if lengths_agree(length, c.get("duration"), both=True)}
        if len(ids) == 1:
            return ids.pop(), "title"
    return None, None
