"""What a song's beats say: its tempo, how steady it is, its bars, and how it compares with other readings.

Beat This! gives the times of every beat and downbeat (in seconds, on a 20 ms grid). From those this works out
what Siansa's DJ blends need, the way the Siansa plugin reports it: the tempo of the whole song and of each end
(the first and last :data:`END_SECONDS`, as the plugin measures), how steady it is, how many beats a bar has and
where the first downbeat falls. It also says how two tempos relate (the same, double, half, 4/3…), which is what
tells a misread tempo from a different one.

Pure Python, so it's tested without numpy or the model.
"""

import math
import statistics

END_SECONDS = 30.0  # each end, as the plugin's beats.SECONDS
MIN_BEATS = 8  # fewer, and there's no tempo worth reporting
GRID_MS = 30.0  # a beat this close to the straight grid sits on it
STEADY_SHARE = 0.04  # a gap within 4 % of the usual one is steady…
STEADY_MS = 25.0  # …or within 25 ms: Beat This! gives beats on a 20 ms grid, so a gap can be one step off

# How two tempos can relate when one of them is misread: the beat counted at another level of the bar.
RELATIONS = (
    ("same", 1.0),
    ("double", 2.0),
    ("half", 0.5),
    ("4/3", 4.0 / 3.0),
    ("3/4", 0.75),
    ("3/2", 1.5),
    ("2/3", 2.0 / 3.0),
)
RELATION_TOLERANCE = 0.03  # 3 %: two readings of one song rarely differ by more when they agree

# The tempos each style is usually made at, for telling which of two related readings is likely right. Keys are
# matched as words in a genre (lower case); the first that matches wins, so the more specific come first. Room for
# better genres later: anything that adds a genre (genres table, any source) gets checked against this.
GENRE_RANGES = (
    ("drum and bass", 160, 180), ("drum & bass", 160, 180), ("dnb", 160, 180), ("jungle", 155, 180),
    ("dubstep", 135, 150), ("trap", 130, 170),
    ("psytrance", 135, 150), ("hardstyle", 145, 160), ("hardcore", 160, 200), ("gabber", 160, 200),
    ("trance", 125, 150), ("techno", 120, 150), ("house", 115, 132), ("garage", 125, 140),
    ("eurodance", 120, 145), ("hi-nrg", 120, 140), ("disco", 105, 135), ("electro", 110, 135),
    ("downtempo", 70, 115), ("trip hop", 70, 110), ("chillout", 70, 120), ("ambient", 50, 120),
    ("hip hop", 70, 105), ("hip-hop", 70, 105), ("rap", 70, 110), ("r&b", 60, 110),
    ("reggaeton", 85, 105), ("reggae", 60, 100), ("dancehall", 85, 110),
    ("metal", 80, 200), ("punk", 140, 200), ("rock", 70, 180), ("pop", 70, 140),
    ("jazz", 50, 250), ("classical", 30, 200), ("soundtrack", 30, 200),
)


def _fit(xs, ys):
    """Least squares line ``ys = a + b * xs``: (a, b, rms of what's left)."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return my, 0.0, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    rms = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / n)
    return a, b, rms


def tempo(beats):
    """The tempo of ``beats`` (seconds) as a dict, or ``None`` when there are too few.

    ``bpm`` comes from a straight line through the beats, numbered so a missed or extra beat doesn't bend it (each
    gap is counted in whole beats of the median gap). ``bpm_median`` is from the median gap alone. ``steady`` (0..1)
    is the share of gaps within 4 % (or 25 ms, the model's grid) of the median; ``grid_ms`` how far, at most, a beat lies from the straight grid
    and ``grid_share`` the share within :data:`GRID_MS` of it (a drum machine: near 1; a live drummer: lower).
    """
    beats = sorted(beats)
    if len(beats) < MIN_BEATS:
        return None
    gaps = [b - a for a, b in zip(beats, beats[1:]) if b > a]
    if not gaps:
        return None
    median = statistics.median(gaps)
    if median <= 0:
        return None
    index, numbers = 0, [0]
    for gap in gaps:
        index += max(1, round(gap / median))
        numbers.append(index)
    a, period, _ = _fit(numbers, beats)
    if period <= 0:
        return None
    off = [abs(t - (a + period * n)) * 1000.0 for n, t in zip(numbers, beats)]
    allowed = max(STEADY_SHARE * median, STEADY_MS / 1000.0)
    steady = sum(1 for g in gaps if abs(g - median) <= allowed) / len(gaps)
    return {
        "bpm": 60.0 / period,
        "bpm_median": 60.0 / median,
        "steady": steady,
        "grid_ms": max(off),
        "grid_share": sum(1 for o in off if o <= GRID_MS) / len(off),
        "beats": len(beats),
    }


def section(beats, start, end):
    """:func:`tempo` of the beats between ``start`` and ``end`` (seconds)."""
    return tempo([b for b in beats if start <= b <= end])


def bars(beats, downbeats):
    """``(beats_per_bar, share)``: how many beats a bar usually has, and the share of bars that have that many.
    ``(None, 0.0)`` when there are too few downbeats."""
    downbeats = sorted(downbeats)
    if len(downbeats) < 3:
        return None, 0.0
    counts = []
    for start, end in zip(downbeats, downbeats[1:]):
        counts.append(sum(1 for b in beats if start - 0.01 <= b < end - 0.01))
    counts = [c for c in counts if c > 0]
    if not counts:
        return None, 0.0
    usual = statistics.mode(counts)
    return usual, sum(1 for c in counts if c == usual) / len(counts)


def tempo_curve(beats, points=240, window=16):
    """The local tempo along the song for a chart: ``[[seconds, bpm], …]``, each over ``window`` beats (the span
    they cover, so the model's 20 ms grid averages out), at most ``points`` of them."""
    beats = sorted(beats)
    if len(beats) < window + 1:
        return []
    out = []
    for i in range(len(beats) - window):
        gaps = [b - a for a, b in zip(beats[i:i + window], beats[i + 1:i + window + 1])]
        local = statistics.median(gaps)
        span = beats[i + window] - beats[i]
        if local <= 0 or span <= 0:
            continue
        count = sum(max(1, round(g / local)) for g in gaps)  # a missed beat counts as the two it spans
        out.append([round(beats[i + window // 2], 2), round(60.0 * count / span, 2)])
    if len(out) > points:
        step = len(out) / points
        out = [out[int(i * step)] for i in range(points)]
    return out


def relation(a, b, tolerance=RELATION_TOLERANCE):
    """How tempo ``a`` relates to tempo ``b``: ``"same"``, ``"double"`` (a is twice b), ``"half"``, ``"4/3"``,
    ``"3/4"``, ``"3/2"``, ``"2/3"``, ``"different"``, or ``None`` when either is missing."""
    if not a or not b or a <= 0 or b <= 0:
        return None
    ratio = a / b
    name, value = min(RELATIONS, key=lambda r: abs(math.log(ratio / r[1])))
    return name if abs(ratio / value - 1.0) <= tolerance else "different"


def genre_range(genres):
    """``(low, high, genre)``: the usual tempos of the first of ``genres`` (labels) that :data:`GENRE_RANGES`
    knows, or ``None``."""
    for label in genres or ():
        text = " " + " ".join(label.lower().replace("/", " ").replace(",", " ").replace("_", " ").split()) + " "
        for key, low, high in GENRE_RANGES:
            if f" {key} " in text:
                return low, high, label
    return None


def analyse(beats, downbeats, duration):
    """Everything Siansa's comparison shows for one song, from Beat This!'s beats and downbeats (seconds) and the
    song's length (seconds)."""
    whole = tempo(beats) or {}
    intro = section(beats, 0.0, END_SECONDS) or {}
    outro = section(beats, max(0.0, duration - END_SECONDS), duration + 1.0) or {}
    per_bar, bar_share = bars(beats, downbeats)
    return {
        "bpm": whole.get("bpm"),
        "bpm_median": whole.get("bpm_median"),
        "steady": whole.get("steady"),
        "grid_ms": whole.get("grid_ms"),
        "grid_share": whole.get("grid_share"),
        "intro_bpm": intro.get("bpm"),
        "intro_steady": intro.get("steady"),
        "outro_bpm": outro.get("bpm"),
        "outro_steady": outro.get("steady"),
        "beats_per_bar": per_bar,
        "bar_share": bar_share,
        "first_downbeat_ms": round(min(downbeats) * 1000.0) if len(downbeats) else None,
        "beat_count": len(beats),
        "downbeat_count": len(downbeats),
    }


def verdict(result, tag_bpm=None, plugin=None, genres=()):
    """What the readings say together, for the table: a list of ``(level, text)``, level ``"ok"``, ``"warn"`` or
    ``"info"``. ``result`` is :func:`analyse`'s, ``plugin`` the plugin's profile (``intro_bpm``, ``outro_bpm``)."""
    notes = []
    bpm = (result or {}).get("bpm")
    if not bpm:
        return [("warn", "No steady beat found")]
    if tag_bpm:
        r = relation(tag_bpm, bpm)
        if r == "same":
            notes.append(("ok", "Matches the BPM tag"))
        elif r == "different":
            notes.append(("warn", f"BPM tag {tag_bpm:g} differs"))
        elif r:
            notes.append(("warn", f"BPM tag {tag_bpm:g} is {r} of this"))
    for end in ("intro", "outro"):
        theirs = (plugin or {}).get(f"{end}_bpm")
        ours = result.get(f"{end}_bpm") or bpm
        r = relation(theirs, ours)
        if r == "same":
            notes.append(("ok", f"Plugin's {end} agrees"))
        elif r == "different":
            notes.append(("warn", f"Plugin's {end} {theirs:.1f} differs"))
        elif r:
            notes.append(("warn", f"Plugin's {end} {theirs:.1f} is {r} of this"))
    found = genre_range(genres)
    if found:
        low, high, label = found
        if not low <= bpm <= high:
            notes.append(("warn", f"{bpm:.1f} is outside {label}'s usual {low}–{high}"))
    if (result.get("steady") or 0) < 0.6:
        notes.append(("info", "Tempo varies"))
    return notes
