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

from . import ends, features, phrase

ANALYSIS_VERSION = 10  # what's worked out from the beats: older results are worked out again from their saved beats
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


MAIN_BAND = 0.02  # local tempos within 2 % of each other are the same tempo
MAIN_WINDOW = 16  # beats in each local reading, as the tempo curve


def local_tempos(beats, window=MAIN_WINDOW):
    """``[(seconds, bpm, length), …]``: the tempo over every run of ``window`` beats along the song, with the
    middle beat's time and how long the run lasts. A missed beat counts as the two it spans."""
    beats = sorted(beats)
    out = []
    for i in range(len(beats) - window):
        gaps = [b - a for a, b in zip(beats[i:i + window], beats[i + 1:i + window + 1])]
        local = statistics.median(gaps)
        span = beats[i + window] - beats[i]
        if local <= 0 or span <= 0:
            continue
        count = sum(max(1, round(g / local)) for g in gaps)
        out.append((beats[i + window // 2], 60.0 * count / span, span))
    return out


def main_tempo(beats, band=MAIN_BAND):
    """``(bpm, share)``: the tempo the song spends most of its time at, and the share of its beat it lasts, or
    ``(None, 0.0)``. Unlike one line through every beat, a breakdown, a half-time intro or a tempo change doesn't
    drag it off: each stretch of the song counts for its own tempo."""
    local = local_tempos(beats)
    if not local:
        return None, 0.0
    # Each run counts for its length, spread over the window's beats so overlapping runs don't count twice.
    points = sorted((math.log(bpm), span / MAIN_WINDOW, bpm) for _, bpm, span in local)
    total = sum(w for _, w, _ in points)
    width = math.log(1.0 + band)
    best, best_weight, lo, weight = (0, 0), -1.0, 0, 0.0
    for hi, (x, w, _) in enumerate(points):
        weight += w
        while x - points[lo][0] > 2 * width:
            weight -= points[lo][1]
            lo += 1
        if weight > best_weight:
            best, best_weight = (lo, hi), weight
    chosen = points[best[0]:best[1] + 1]
    bpm = _weighted_median([(b, w) for _, w, b in chosen])
    return bpm, best_weight / total if total else 0.0


def _weighted_median(pairs):
    pairs = sorted(pairs)
    half, run = sum(w for _, w in pairs) / 2.0, 0.0
    for value, w in pairs:
        run += w
        if run >= half:
            return value
    return pairs[-1][0]


GRID_HIT = 0.040  # a model beat this close (seconds) to the steady grid sits on it
GRID_MIN_SHARE = 0.7  # less on one steady grid (of the model's beats, or of the grid's beats heard), and the song isn't at one tempo
END_MIN_SHARE = 0.5  # an end that holds one tempo for less than half its beat has no steady beat


def steady_grid(beats, downbeats, bpm, per_bar=4):
    """One even grid of beats at ``bpm`` laid over the model's ``beats``, as DJ software does for a song made at
    one tempo: it carries on through breakdowns and quiet intros where the model hears extra beats or none.

    Its bars come from a vote: each of the model's downbeats on the grid says which beat of ``per_bar`` starts
    the bar, and the most voted wins; the bars then follow every ``per_bar`` beats, so they can't drift.

    Returns ``None`` when fewer than :data:`GRID_MIN_SHARE` of the model's beats sit on one grid (a live band, a
    tempo change), else a dict: ``bpm``, ``on_grid`` (that share), ``beats`` and ``downbeats`` (seconds),
    ``bar_share`` (the share of the model's downbeats that voted for the winner) and ``first_downbeat_ms``."""
    beats = sorted(beats)
    if not bpm or len(beats) < MIN_BEATS:
        return None
    period = 60.0 / bpm
    # Number the beats one after another from a beat in the middle of the song, each by its gap to the last one that
    # fitted, so a tempo a fraction off (174.44 for a song at 174) can't make one stretch a beat off from the next,
    # and extra beats heard in a breakdown (half a beat off) are left out. Then one line through them gives the
    # tempo and phase; twice, as the sharper tempo numbers long gaps better.
    best = None
    for anchor in _anchors(len(beats)):
        fit = period
        for _ in range(3):
            numbered = _number(beats, fit, anchor)
            if len(numbered) < MIN_BEATS:
                break
            origin, fit = _robust_fit(numbered)
            if fit <= 0:
                break
        else:
            hits = sum(1 for b in beats if abs(_offset(b, origin, fit)) <= GRID_HIT)
            if best is None or hits > best[0]:
                best = (hits, origin, fit)
        if best and best[0] >= 0.9 * len(beats):
            break
    if best is None:
        return None
    _, origin, period = best
    hits = sum(1 for b in beats if abs(_offset(b, origin, period)) <= GRID_HIT)
    # The share on the grid, two ways: of the model's beats (extra beats heard in a breakdown count against it), and
    # of the grid's beats the model heard (a quiet stretch counts against it). A song at one tempo does well on at
    # least one; a song that changes tempo on neither.
    span = [origin + n * period for n in range(math.ceil((beats[0] - origin) / period),
                                                   math.floor((beats[-1] - origin) / period) + 1)]
    heard = _heard(span, beats)
    share = max(hits / len(beats), heard / len(span) if span else 0.0)
    if share < GRID_MIN_SHARE:
        return None
    first = math.ceil((beats[0] - period / 2 - origin) / period)
    last = math.floor((beats[-1] + period / 2 - origin) / period)
    numbers = list(range(first, last + 1))
    per_bar = per_bar if per_bar in (2, 3, 4, 6, 7) else 4
    tally = [0] * per_bar
    for d in downbeats:
        if abs(_offset(d, origin, period)) <= GRID_HIT * 1.5:
            tally[round((d - origin) / period) % per_bar] += 1
    voted = sum(tally)
    start = tally.index(max(tally)) if voted else (first % per_bar)
    grid_beats = [origin + n * period for n in numbers]
    grid_downs = [origin + n * period for n in numbers if n % per_bar == start]
    return {
        "bpm": 60.0 / period,
        "on_grid": share,
        "beats": [round(b, 3) or 0.0 for b in grid_beats],  # "or 0.0": no -0.0
        "downbeats": [round(b, 3) or 0.0 for b in grid_downs],
        "per_bar": per_bar,
        "bar_share": max(tally) / voted if voted else 0.0,
        "first_downbeat_ms": round(grid_downs[0] * 1000.0) if grid_downs else None,
    }


PHRASE_SHIFTS = (0, 4)  # where the sound may put a normal track's phrases: from bar 1, or a 4-bar intro later


def with_sound(steady, levels):
    """``steady`` (a :func:`steady_grid`) with its bars checked against where the music changes, and its phrases,
    from the song's ``levels`` (:func:`beats.features.measure`). Without levels it's returned as it is, with
    ``downbeat_source`` ``"model"`` and phrases every 8 bars from bar 1.

    The phrases (``phrase_source``): a dance track is made in 8-bar blocks from its first bar, so they're every 8
    bars from bar 1 (``"bar 1"``), unless the sound says clearly that they start 4 bars later or run 16 bars
    (``"sound"``): the owner's marks (21 of 22 on bar 1 + 8k where the bars were clear) and the songs where the
    sound was sure (61 of 78 on bar 1, 9 more 4 bars later) both say so. A track from a mixed CD (``from_mix``:
    full volume at both ends, cut wherever the DJ put the track marker) has no bar 1 to count from: its phrases
    come from the sound or there are none (``"mix: sound"``, ``"mix: none"``)."""
    if not steady:
        return steady
    steady = dict(steady, downbeat_source="model", phrases=steady["downbeats"][::phrase.PHRASE_BARS],
                  phrase_bars=phrase.PHRASE_BARS, phrase_source="bar 1", from_mix=None, phrase_big=None,
                  phrase_big_on=None, changes_share=None)
    hop, frames = features.unpack(levels)
    if frames:
        steady["from_mix"] = features.full_volume_ends(hop, frames)
    beat_levels = phrase.per_beat(hop, frames, steady["beats"])
    per_bar = steady["per_bar"]
    if len(beat_levels) < 8 * per_bar:
        if steady["from_mix"]:
            steady.update(phrases=[], phrase_source="mix: none")
        return steady
    vote = steady["beats"].index(steady["downbeats"][0]) % per_bar if steady["downbeats"] else 0
    start, source, details = phrase.downbeats(beat_levels, per_bar, vote, steady["bar_share"])
    downs = steady["beats"][start::per_bar]
    found, about = phrase.phrases(phrase.bars_of(beat_levels, start, per_bar))
    if steady["from_mix"]:
        phrases = downs[found::about["bars"]] if found is not None else []
        bars, origin = about["bars"], "mix: sound" if found is not None else "mix: none"
    elif found is not None and found % 8 in PHRASE_SHIFTS:
        phrases, bars, origin = downs[found::about["bars"]], about["bars"], "sound"
    else:
        phrases, bars, origin = downs[::phrase.PHRASE_BARS], phrase.PHRASE_BARS, "bar 1"
    steady.update(
        downbeats=downs, downbeat_source=source, changes_share=details["changes_share"],
        first_downbeat_ms=round(downs[0] * 1000.0) if downs else None,
        phrases=phrases, phrase_bars=bars, phrase_source=origin,
        phrase_big=about["big"], phrase_big_on=about["big_on"])
    return steady


def _anchors(count):
    """Beats to number a song from: its middle, then further out, in case the middle one is an extra beat."""
    return sorted({min(count - 1, max(0, int(count * f))) for f in (0.5, 0.3, 0.7, 0.15, 0.85)},
                  key=lambda i: abs(i - count / 2))


def _number(beats, period, anchor):
    """``[(n, seconds)]``: the beats that fit a grid of ``period`` counted from ``beats[anchor]`` (n = 0), each by
    its gap to the last one that fitted, both ways along the song."""
    allow = min(GRID_HIT * 1.5, period * 0.2)
    out = [(0, beats[anchor])]
    for direction in (1, -1):
        last_n, last_b = 0, beats[anchor]
        i = anchor + direction
        while 0 <= i < len(beats):
            k = (beats[i] - last_b) / period
            n = round(k)
            if n != 0 and abs(k - n) * period <= allow:
                last_n, last_b = last_n + n, beats[i]
                out.append((last_n, last_b))
            i += direction
    return sorted(out)


def _robust_fit(numbered):
    """``(origin, period)`` of a line through ``[(n, seconds)]``, leaving out those more than :data:`GRID_HIT` off
    it (a beat the model placed late in a fill) and fitting again."""
    ns, ts = [n for n, _ in numbered], [t for _, t in numbered]
    origin, period, _ = _fit(ns, ts)
    for _ in range(3):
        kept = [(n, t) for n, t in numbered if abs(t - (origin + n * period)) <= GRID_HIT]
        if len(kept) < MIN_BEATS or len(kept) == len(ns):
            break
        ns, ts = [n for n, _ in kept], [t for _, t in kept]
        origin, period, _ = _fit(ns, ts)
    return origin, period


def _heard(grid_beats, beats):
    """How many of ``grid_beats`` have one of the model's ``beats`` within :data:`GRID_HIT` (both sorted)."""
    found, j = 0, 0
    for g in grid_beats:
        while j < len(beats) and beats[j] < g - GRID_HIT:
            j += 1
        if j < len(beats) and abs(beats[j] - g) <= GRID_HIT:
            found += 1
    return found


def _offset(t, origin, period):
    """How far (seconds) ``t`` is from the nearest beat of the grid ``origin + n * period``."""
    return t - (origin + round((t - origin) / period) * period)


def end_tempo(beats, start, end):
    """``(bpm, share, beats)`` for one end of a song: the tempo it holds for most of it, as :func:`main_tempo`.
    The tempo is ``None`` when no tempo lasts :data:`END_MIN_SHARE` of it (a quiet or beatless intro)."""
    part = [b for b in beats if start <= b <= end]
    if len(part) < MIN_BEATS:
        return None, 0.0, len(part)
    bpm, share = main_tempo(part)
    if not bpm or share < END_MIN_SHARE:
        return None, share, len(part)
    return bpm, share, len(part)


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


def tempo_curve(beats, points=240, window=MAIN_WINDOW):
    """The local tempo along the song for a chart: ``[[seconds, bpm], …]``, each over ``window`` beats (the span
    they cover, so the model's 20 ms grid averages out), at most ``points`` of them."""
    out = [[round(t, 2), round(bpm, 2)] for t, bpm, _ in local_tempos(beats, window)]
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


def app_ends(steady, model_beats, duration, levels=None, end_seconds=END_SECONDS):
    """The beat grid at each end of the song, in the shape the Siansa app reads from the Siansa plugin (its
    ``BeatEnd``): ``intro_bpm``, ``intro_beat_ms`` (the first beat), ``intro_bar_ms`` (the first downbeat),
    ``intro_phrase_ms`` and ``intro_phrase_bars`` (the first phrase start, when there are phrases), ``intro_rhythm``
    (the share of the grid's beats in the first :data:`END_SECONDS` the model heard: low for a beatless intro) and
    ``intro_steady`` (the share on the grid); the same for ``outro_`` (the last beat, downbeat and phrase start, from
    the last :data:`END_SECONDS`); and ``beats_until_ms``, the model's last beat. From the steady grid, so the
    tempo is the song's, to a hundredth of a BPM.

    Each end is taken inside the song's lock (:func:`beats.ends.lock_points`): the first beat, bar and phrase from
    where the beat locks onto the grid, the last ones up to where it last holds, and the rhythm over the
    :data:`END_SECONDS` inside it. So a track cut from a mixed CD, whose first and last seconds carry the DJ's own
    transitions, is blended where its own beat is. With the other fields of :func:`beats.ends.describe`."""
    grid_beats, downs, phrases = steady["beats"], steady["downbeats"], steady.get("phrases") or []
    if not grid_beats:
        return None
    model_beats = sorted(model_beats)
    hits = [bool(_heard([g], model_beats)) for g in grid_beats]
    extra = ends.describe(steady, hits, duration, levels)
    lock_in = extra["intro_lock_ms"] / 1000.0 if extra["intro_lock_ms"] is not None else 0.0
    lock_out = extra["outro_lock_ms"] / 1000.0 if extra["outro_lock_ms"] is not None else duration

    def heard(lo, hi):
        inside = [b for b in grid_beats if lo <= b <= hi]
        return (_heard(inside, model_beats) / len(inside)) if inside else 0.0

    def ms(t):
        return round(t * 1000.0, 1) if t is not None else None

    last_beat = max((b for b in grid_beats if b <= min(duration, lock_out + 0.05)), default=grid_beats[-1])
    first_beat = next((b for b in grid_beats if b >= lock_in - 0.05), grid_beats[0])
    # Where the beat stops, only when a beatless outro follows (as the plugin sends it): the app then fades after
    # the last beat instead of beat-matching. Every song's last beat is a few seconds before its end, where its
    # fade-out gets too quiet to hear the beat; sent for all, the app did no beat-matched blends at all.
    last_heard = model_beats[-1] if model_beats else None
    beatless = last_heard is not None and duration - last_heard >= BEATLESS_OUTRO_SECONDS
    out = {"beats_until_ms": ms(last_heard) if beatless else None}
    for end, beat, bar, phrase_start, lo, hi in (
            ("intro", first_beat, next((d for d in downs if d >= first_beat - 0.05), None),
             next((p for p in phrases if p >= first_beat - 0.05), None), first_beat, first_beat + end_seconds),
            ("outro", last_beat, max((d for d in downs if d <= last_beat + 0.05), default=None),
             max((p for p in phrases if p <= last_beat + 0.05), default=None), max(0.0, last_beat - end_seconds),
             last_beat + 0.05)):
        out.update({
            f"{end}_bpm": round(steady["bpm"], 3),
            f"{end}_beat_ms": ms(beat),
            f"{end}_bar_ms": ms(bar),
            f"{end}_phrase_ms": ms(phrase_start),
            f"{end}_phrase_bars": steady.get("phrase_bars") if phrase_start is not None else None,
            f"{end}_rhythm": round(heard(lo, hi), 3),
            f"{end}_steady": round(steady["on_grid"], 3),
        })
    out.update(extra)
    return out


BEATLESS_OUTRO_SECONDS = 16.0  # this long without a beat at the end is a beatless outro, as the plugin reads it


def analyse(beats, downbeats, duration, levels=None):
    """Everything Siansa's comparison shows for one song, from Beat This!'s beats and downbeats (seconds), the
    song's length (seconds) and its sound's ``levels`` (:func:`beats.features.measure`, for bars and phrases).

    The grid is the tempo the model heard: genre tags never change it (they're partly guessed). A song heard at 87
    stays at 87; the app counts half and double as the same tempo, and doubles anything under 100
    itself, so it blends into 174 drum and bass as it would by ear."""
    whole = tempo(beats) or {}
    main, main_share = main_tempo(beats)
    fit = whole.get("bpm")
    if main and fit and abs(fit / main - 1.0) <= 0.005:
        main = fit  # a steady song: the line through all its beats is the finer reading of the same tempo
    bpm = main if main else fit
    intro_end = END_SECONDS
    outro_start = max(0.0, duration - END_SECONDS)
    intro_bpm, intro_share, _ = end_tempo(beats, 0.0, intro_end)
    outro_bpm, outro_share, _ = end_tempo(beats, outro_start, duration + 1.0)
    per_bar, bar_share = bars(beats, downbeats)
    steady = with_sound(steady_grid(beats, downbeats, bpm, per_bar or 4), levels) or {}
    if steady:  # the even grid's bars, not the model's downbeats, which wander in breakdowns
        per_bar = steady["per_bar"]
    return {
        "bpm": bpm,
        "bpm_fit": fit,
        "main_share": main_share if main else None,
        "bpm_median": whole.get("bpm_median"),
        "steady": whole.get("steady"),
        "grid_ms": whole.get("grid_ms"),
        "grid_share": whole.get("grid_share"),
        "intro_bpm": intro_bpm,
        "intro_steady": intro_share,
        "outro_bpm": outro_bpm,
        "outro_steady": outro_share,
        "beats_per_bar": per_bar,
        "bar_share": steady.get("bar_share", bar_share),
        "first_downbeat_ms": steady.get("first_downbeat_ms",
                                        round(min(downbeats) * 1000.0) if len(downbeats) else None),
        "grid_bpm": steady.get("bpm"),
        "on_grid": steady.get("on_grid"),
        "downbeat_source": steady.get("downbeat_source"),
        "grid_counted": "same" if steady else None,
        "phrase_source": steady.get("phrase_source"),
        "from_mix": steady.get("from_mix"),
        "ends": app_ends(steady, beats, duration, levels) if steady else None,
        "changes_share": steady.get("changes_share"),
        "phrase_first_ms": round(steady["phrases"][0] * 1000.0) if steady.get("phrases") else None,
        "phrase_count": len(steady["phrases"]) if steady.get("phrases") is not None and levels else None,
        "phrase_bars": steady.get("phrase_bars") if steady.get("phrases") else None,
        "phrase_big": steady.get("phrase_big"),
        "phrase_big_on": steady.get("phrase_big_on"),
        "beat_count": len(beats),
        "downbeat_count": len(downbeats),
        "analysis_version": ANALYSIS_VERSION,
    }


def verdict(result, tag_bpm=None, plugin=None):
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
    for end in ("intro", "outro"):
        theirs = result.get(f"{end}_bpm")
        r = relation(theirs, bpm)
        name = "Starts" if end == "intro" else "Ends"
        if not theirs:
            if f"{end}_bpm" in result:
                notes.append(("info", f"{name} without a steady beat"))
        elif r in ("half", "double"):
            notes.append(("info", f"{name} at {r} time ({theirs:.1f})"))
        elif r and r != "same":  # how the song is made, not two readings disagreeing: noted, not flagged
            notes.append(("info", f"{name} at {theirs:.1f}, not the song's {bpm:.1f}"))
    if result.get("downbeat_source") == "changes":
        notes.append(("warn", "Bars moved: the model's downbeats and the music's changes disagree"))
    if result.get("phrase_count") == 0 and result.get("on_grid"):
        big, on = result.get("phrase_big") or 0, result.get("phrase_big_on") or 0
        notes.append(("warn", f"No clear phrases ({on} of {big} big changes on one bar of 8)" if big else
                      "No clear phrases (no big changes)"))
    if result.get("on_grid") is None and "on_grid" in result:
        notes.append(("info", "Not at one tempo: no steady grid"))
    if (result.get("steady") or 0) < 0.6:
        notes.append(("info", "Tempo varies"))
    return notes
