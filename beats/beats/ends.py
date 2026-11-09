"""What the Siansa app needs to choose how to blend at each end of a song, beyond its beat grid.

- **Lock-in** (``intro_lock_ms``, ``outro_lock_ms``): where the song's beat locks onto its grid at the start, and
  where it last holds at the end. A track cut from a mixed CD carries the DJ's own transitions there (the record
  before or after it still playing), so its first and last seconds don't follow its own grid; a blend belongs
  inside the lock. For most songs the lock is their first and last bar.
- **Bass** (``intro_bass_bars``, ``outro_bass_bars``): how many bars from the lock the bass stays out (an intro of
  drums alone) or is already gone (an outro of drums alone): those bars blend without a bass swap.
- **Drops and breakdowns** (``drops_ms``, ``breakdowns_ms``): the phrase starts where the bass comes back in after
  being out, or drops out.
- **Energy** (``intro_energy``, ``outro_energy``): how loud each end is next to the song's loud part, in dB
  (0 the same, -6 a quieter intro), for the direction of a blend.

All from the steady grid and the sound's levels (:mod:`beats.features`); plain Python.
"""

import statistics

from . import features

LOCK_BEATS = 16  # 4 bars of 4: the stretch that has to sit on the grid
LOCK_MISSES = 1  # of them the model may miss
BASS_IN_DB = 6.0  # bass within this of the song's usual loud bass is "in"
BAND_BASS = 1  # sub bass, bass, mids, highs: the bass band (120–500 Hz, kick and bassline)
BAND_SUB = 0


def lock_points(grid_beats, heard, downbeats):
    """``(intro, outro)`` seconds: the first downbeat that starts 4 bars all on the grid, and the last downbeat that
    ends them; ``(None, None)`` when no 4 bars are. ``heard`` says, for each of ``grid_beats``, whether the model
    heard it."""
    starts = [i for i in range(len(grid_beats) - LOCK_BEATS + 1)
              if LOCK_BEATS - sum(heard[i:i + LOCK_BEATS]) <= LOCK_MISSES]
    if not starts:
        return None, None
    first = grid_beats[starts[0]]
    last = grid_beats[starts[-1] + LOCK_BEATS - 1]
    downs = sorted(downbeats) or grid_beats
    intro = next((d for d in downs if d >= first - 0.05), first)
    outro = max((d for d in downs if d <= last + 0.05), default=last)
    if outro <= intro:
        return first, last
    return intro, outro


def _bars(downbeats, start, stop):
    return [(a, b) for a, b in zip(downbeats, downbeats[1:]) if a >= start - 0.01 and b <= stop + 0.01]


def _level(frames, hop, start, stop, band):
    lo, hi = int(start / hop), max(int(start / hop) + 1, int(stop / hop))
    values = [f[band] for f in frames[lo:hi]]
    return statistics.fmean(values) if values else None


def _total(f):
    import math
    return 10.0 * math.log10(sum(10.0 ** (v / 10.0) for v in f))


def describe(steady, heard, duration, levels):
    """The fields above for the app, from ``steady`` (:func:`beats.grid.steady_grid` with its phrases), ``heard``
    (per grid beat, whether the model heard it), the song's ``duration`` and its ``levels``."""
    beats, downs, phrases = steady["beats"], steady["downbeats"], steady.get("phrases") or []
    intro, outro = lock_points(beats, heard, downs)
    out = {"intro_lock_ms": _ms(intro), "outro_lock_ms": _ms(outro), "mix_track": bool(steady.get("from_mix")),
           "intro_bass_bars": None, "outro_bass_bars": None, "drops_ms": [], "breakdowns_ms": [],
           "intro_energy": None, "outro_energy": None}
    hop, frames = features.unpack(levels)
    if not frames or intro is None:
        return out
    bars = _bars(downs, intro, outro)
    if not bars:
        return out
    bass = [max(_level(frames, hop, a, b, BAND_BASS), _level(frames, hop, a, b, BAND_SUB)) for a, b in bars]
    loud_bass = statistics.median(sorted(bass)[len(bass) // 2:])  # the usual bass of the song's fuller half
    is_in = [v >= loud_bass - BASS_IN_DB for v in bass]
    out["intro_bass_bars"] = next((i for i, v in enumerate(is_in) if v), len(is_in))
    out["outro_bass_bars"] = next((i for i, v in enumerate(reversed(is_in)) if v), len(is_in))

    def bass_at(t):  # the bass of the 4 bars from t (or before it, with ``before``)
        inside = [v for (a, _), v in zip(bars, bass) if t - 0.01 <= a < t + 4 * (bars[0][1] - bars[0][0])]
        return statistics.fmean(inside) if inside else None

    def bass_before(t):
        size = 4 * (bars[0][1] - bars[0][0])
        inside = [v for (a, _), v in zip(bars, bass) if t - size - 0.01 <= a < t - 0.01]
        return statistics.fmean(inside) if inside else None

    for p in phrases:
        if p <= intro + 0.01 or p >= outro - 0.01:
            continue
        now, then = bass_at(p), bass_before(p)
        if now is None or then is None:
            continue
        if now - then >= BASS_IN_DB and now >= loud_bass - BASS_IN_DB:
            out["drops_ms"].append(_ms(p))
        elif then - now >= BASS_IN_DB and then >= loud_bass - BASS_IN_DB:
            out["breakdowns_ms"].append(_ms(p))
    total = [_total(f) for f in frames]
    loud = statistics.median(sorted(total)[len(total) // 2:])
    span = 30.0
    for name, a, b in (("intro", intro, min(outro, intro + span)), ("outro", max(intro, outro - span), outro)):
        lo, hi = int(a / hop), max(int(a / hop) + 1, int(b / hop))
        part = total[lo:hi]
        out[f"{name}_energy"] = round(statistics.fmean(part) - loud, 1) if part else None
    return out


def _ms(t):
    return round(t * 1000.0, 1) if t is not None else None
