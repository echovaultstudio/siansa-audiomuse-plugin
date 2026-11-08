#!/usr/bin/env python3
"""The real thing: the built image with Beat This! against the fake Navidrome (tests/fakes.py) and the songs from
make_test_songs.py, whose tempos and bars are known. Checks every part a user touches: the scan, the queue, the
model's readings, the comparison with the plugin and the BPM tag, the song's page, the audio and the CSV.

    python3 tests/e2e.py http://127.0.0.1:8090 <songs folder>/catalog.json

Songs marked strict must read within 1.5 % of their tempo with the right bar, their bars starting on beat 1 and
their phrases where their sections do (within 60 ms); the others are reported (a model may count drum & bass at
half time, which a DJ would too). Prints a table either way; exits 1 on a failure.
"""

import json
import sys
import time
import urllib.request

STRICT = {"t140", "t122", "t128"}  # the steady 4/4 dance songs: what DJ blends need right


def call(base, path, method="GET", body=None, raw=False):
    request = urllib.request.Request(base + path, method=method, data=json.dumps(body).encode() if body else None,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as r:
        data = r.read()
        return (r, data) if raw else json.loads(data)


def main(base, catalog_path):
    with open(catalog_path) as f:
        catalog = {s["id"]: s for s in json.load(f)}
    problems, notes = [], []

    end = time.time() + 120
    while True:
        try:
            health = call(base, "/api/health")
            break
        except OSError:
            if time.time() > end:
                raise SystemExit("The service didn't start")
            time.sleep(1)
    print(f"Service up: model {health['model']}, loaded {health['model_loaded']}")

    print("Libraries:", call(base, "/api/libraries", "POST", {"ids": ["1"]}))
    started, end = time.time(), time.time() + 900
    while True:
        status = call(base, "/api/status")
        c = status["counts"]
        if c.get("songs") and c["songs"] == c.get("done", 0) + c.get("failed", 0):
            break
        if time.time() > end:
            raise SystemExit(f"Not finished after 15 minutes: {c}")
        time.sleep(2)
    print(f"All analysed in {time.time() - started:.0f} s (model loaded in {status['model_load_seconds']:.1f} s "
          f"on {status.get('device_name') or status.get('device')})")
    if status.get("device_note"):
        problems.append(f"device: {status['device_note']}")
    if not status.get("device_name"):
        problems.append("the status doesn't say where the model runs")

    songs = {s["id"]: s for s in call(base, "/api/songs")}
    print(f"\n{'song':<14}{'expect':>8}{'read':>9}{'intro':>8}{'outro':>8}{'bar':>5}{'steady':>8}{'1 at':>7}"
          f"{'phrase':>8}{'model s':>9}  notes")
    for sid, want in catalog.items():
        s = songs.get(sid)
        if not s or s["state"] != "done":
            problems.append(f"{sid}: {s and s['state']} {s and s.get('error')}")
            continue
        r = s["result"]
        line = (f"{sid:<14}{want['expect_bpm']:>8.1f}{(r['bpm'] or 0):>9.2f}{(r['intro_bpm'] or 0):>8.1f}"
                f"{(r['outro_bpm'] or 0):>8.1f}{str(r['beats_per_bar']):>5}{(r['steady'] or 0):>8.2f}"
                f"{str(r['first_downbeat_ms']):>7}{str(r['phrase_first_ms']):>8}"
                f"{r['model_seconds']:>9.1f}  " + "; ".join(t for _, t in s["verdict"]))
        print(line)
        off = abs((r["bpm"] or 0) / want["expect_bpm"] - 1)
        bar_ok = r["beats_per_bar"] == want["expect_bar"]
        target = problems if sid in STRICT else notes
        if off > 0.015:
            target.append(f"{sid}: read {r['bpm']} BPM, expected {want['expect_bpm']}")
        if not bar_ok:
            target.append(f"{sid}: {r['beats_per_bar']} beats a bar, expected {want['expect_bar']}")
        # Each song's first kick, on beat 1, is at 0 s; its phrases start after its lead-in.
        bar_ms = want["expect_bar"] * 60_000.0 / want["expect_bpm"]
        first = r.get("first_downbeat_ms")
        if first is None or min(first % bar_ms, bar_ms - first % bar_ms) > 60:
            target.append(f"{sid}: the bars start at {first} ms, not on beat 1 (every {bar_ms:.0f} ms from 0)")
        phrase = r.get("phrase_first_ms")
        expected = want.get("expect_phrase_bar", 0) * bar_ms
        if phrase is None or abs(phrase - expected) > 60:
            target.append(f"{sid}: the phrases start at {phrase} ms, expected {expected:.0f} "
                          f"({r.get('phrase_big_on')} of {r.get('phrase_big')} big changes on them)")

    # The comparison: t140 carries the plugin's 4/3 misreading, t122 a BPM tag at half its tempo.
    texts = {sid: [t for _, t in s["verdict"]] for sid, s in songs.items()}
    if not any("is 4/3 of this" in t for t in texts.get("t140", [])):
        problems.append(f"t140: the plugin's 4/3 misreading isn't shown ({texts.get('t140')})")
    if not any("is half of this" in t for t in texts.get("t122", [])):
        problems.append(f"t122: the BPM tag at half isn't shown ({texts.get('t122')})")

    try:
        checks(base, catalog, problems)
    except OSError as e:  # urllib's HTTPError is one
        problems.append(f"a page or the API failed: {e}")

    for n in notes:
        print(f"::warning::{n}")
    for p in problems:
        print(f"::error::{p}")
    print(f"\n{len(problems)} problems, {len(notes)} notes")
    return 1 if problems else 0


def checks(base, catalog, problems):
    """The song's page, its beat grid, the audio, the CSV and the page itself."""
    detail = call(base, "/api/songs/t140")
    if len(detail["beats"]) < 150 or not detail["downbeats"] or not detail["tempo_curve"]:
        problems.append("t140's page lacks its beats, downbeats or tempo curve")
    beats = call(base, "/api/beats/t140")
    if not beats["beats"] or beats["beats_per_bar"] != 4:
        problems.append("/api/beats/t140 isn't right")
    r, audio = call(base, "/api/audio/t140", raw=True)
    if r.status != 200 or len(audio) < 100_000:
        problems.append(f"the audio didn't come through ({r.status}, {len(audio)} bytes)")
    r, csv = call(base, "/api/export.csv", raw=True)
    if len(csv.decode().splitlines()) != len(catalog) + 1:
        problems.append("the CSV doesn't have a row a song")
    r, page = call(base, "/", raw=True)
    if b"Siansa Beats" not in page:
        problems.append("the page didn't load")
    if not call(base, "/api/health")["model_loaded"]:
        problems.append("the model isn't reported as loaded")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
