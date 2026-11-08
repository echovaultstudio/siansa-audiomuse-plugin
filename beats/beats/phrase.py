"""Where the bars start and where the phrases start, from where the music changes.

The model's downbeats wander in breakdowns, and a phrase (8 bars, often 16) isn't something the model reads at
all. Both show in the sound: in dance music the bass comes in, the drop hits, the hats stop, on beat 1 of a bar
and at the start of a phrase. So this measures how much the music changes at each beat of the steady grid (the
sub bass, bass, mids and highs before it against after it, from :mod:`beats.features`) and asks:

* **downbeats**: on which beat of the bar do the big changes land? Set against the model's vote, the stronger
  of the two wins, and a clear disagreement is reported;
* **phrases**: on which bar of every 8 do they land? The answer comes with how many of the song's big changes
  are on a phrase start, so a song without clear sections says so instead of guessing.

Plain Python, so it's tested without numpy.
"""

import math
import statistics

PHRASE_BARS = 8
SIDE_BEATS = 4  # the change at a beat: the 4 beats after it against the 4 before
SIDE_BARS = 2  # the change at a bar: the 2 bars after it against the 2 before
BIG = 2.5  # a change this many spreads above the usual is a big one (a new section)…
MIN_CHANGE_DB = 3.0  # …and at least this big (dB, over the four bands): smaller is noise, not a section
SURE_PEAKS = 3  # this many big changes on one beat of the bar outweigh the model's downbeats


def per_beat(hop, frames, beats):
    """The mean level of each band between each beat and the next (dB), a list per beat but the last."""
    if not hop or not frames:
        return []
    out = []
    for a, b in zip(beats, beats[1:]):
        lo, hi = int(a / hop), max(int(a / hop) + 1, int(b / hop))
        part = frames[lo:hi]
        if not part:
            break
        out.append([sum(f[k] for f in part) / len(part) for k in range(len(part[0]))])
    return out


def _mean(rows):
    return [sum(r[k] for r in rows) / len(rows) for k in range(len(rows[0]))]


def _distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def changes(rows, side):
    """How much the music changes at each row: the mean of the ``side`` rows after against the ``side`` before
    (0 where there aren't enough on either side)."""
    out = [0.0] * len(rows)
    for i in range(side, len(rows) - side + 1):
        out[i] = _distance(_mean(rows[i - side:i]), _mean(rows[i:i + side]))
    return out


def _excess(values):
    """How far each value is above the usual (the median), in spreads (median absolute deviation); 0 below it."""
    if not values:
        return []
    middle = statistics.median(values)
    spread = statistics.median(abs(v - middle) for v in values) or 1e-9
    return [max(0.0, (v - middle) / spread) for v in values]


def peaks(values, least=BIG):
    """``[(index, excess)]``: where a change is at its biggest (no bigger one beside it) and at least ``least``
    spreads above the usual. A change measured over a bar either side rises and falls around the moment it
    happens; its peak is that moment."""
    excess = _excess(values)
    out = []
    for i, e in enumerate(excess):
        if e < least or values[i] < MIN_CHANGE_DB:
            continue
        left = excess[i - 1] if i > 0 else -1.0
        right = excess[i + 1] if i + 1 < len(excess) else -1.0
        if e > left and e >= right:
            out.append((i, e))
    return out


def pick(values, period):
    """``(best, share, peaks)``: the position (0 .. ``period``-1) where the big changes land most (weighted by
    how big), and the share of them it has (1/``period`` is chance)."""
    found = peaks(values)
    scores = [0.0] * period
    for i, e in found:
        scores[i % period] += e
    total = sum(scores)
    if total <= 0:
        return None, 0.0, found
    best = max(range(period), key=lambda k: scores[k])
    return best, scores[best] / total, found


def strength(share, period):
    """How far ``share`` is from chance (0) to all of it (1)."""
    return max(0.0, (share - 1.0 / period) / (1.0 - 1.0 / period))


def downbeats(beat_levels, per_bar, vote, vote_share):
    """Which beat of the grid starts the bar: ``(start, source, details)``. ``vote`` is the beat the model's
    downbeats voted for (0 .. ``per_bar``-1, counted from the grid's first beat) and ``vote_share`` its share.

    The changes win when at least :data:`SURE_PEAKS` big ones agree on a beat (a new section starts on beat 1:
    that's how the music is made, while the model's downbeats can be a beat off all through a song or wander in
    breakdowns), or when they're clearly stronger than the vote. Otherwise the vote stands. ``source`` is
    ``"both"`` (they agree), ``"model"`` or ``"changes"``."""
    if len(beat_levels) > 4 * SIDE_BEATS:
        found, share, big = pick(changes(beat_levels, SIDE_BEATS), per_bar)
    else:
        found, share, big = None, 0.0, []
    by_changes = strength(share, per_bar)
    by_vote = strength(vote_share, per_bar)
    agreeing = sum(1 for i, _ in big if found is not None and i % per_bar == found)
    details = {"changes_beat": found, "changes_share": share, "changes_peaks": len(big), "vote_beat": vote,
               "vote_share": vote_share}
    if found is None:
        return vote, "model", details
    if found == vote:
        return vote, "both", details
    if (agreeing >= SURE_PEAKS and by_changes >= 0.6) or by_changes > by_vote + 0.15:
        return found, "changes", details
    return vote, "model", details


def phrases(bar_levels, phrase_bars=PHRASE_BARS):
    """Which bar starts the phrases: ``(start, details)``, ``start`` counted from the first bar (0 ..
    ``phrase_bars``-1), or ``None`` when the song has no clear sections. ``details``: ``share`` of the changes on
    phrase starts, ``big`` (bars with a big change) and ``big_on`` (how many of them are on a phrase start)."""
    if len(bar_levels) < 2 * phrase_bars:
        return None, {"share": 0.0, "big": 0, "big_on": 0}
    start, share, found = pick(changes(bar_levels, SIDE_BARS), phrase_bars)
    big = [i for i, _ in found]
    on = [i for i in big if start is not None and i % phrase_bars == start]
    details = {"share": share, "big": len(big), "big_on": len(on), "bars": phrase_bars}
    if start is None or not big or len(on) * 2 < len(big):
        return None, details  # fewer than half the big changes on one bar of 8: no clear phrases
    # 16 bars when three quarters of them (at least 4) fall on every other start of 8.
    even = [i for i in on if (i - start) % (2 * phrase_bars) == 0]
    if len(on) >= 4 and max(len(even), len(on) - len(even)) * 4 >= len(on) * 3:
        details["bars"] = 2 * phrase_bars
        if len(even) * 2 < len(on):
            start += phrase_bars
    return start, details


def bars_of(beat_levels, first, per_bar):
    """Each bar's levels: the mean of each band over the bar, beat by beat (so a bar with the same loudness but
    another pattern still differs). ``first`` is the grid beat the first full bar starts on."""
    out = []
    for start in range(first, len(beat_levels) - per_bar + 1, per_bar):
        out.append([v for row in beat_levels[start:start + per_bar] for v in row])
    return out
