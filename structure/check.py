"""Siansa Structure on the test songs (beats/tests/make_test_songs.py): each must come back with its sections, and
how near they start to where the song's sections do (its lead-in, then every 8 bars) is reported.

    python3 structure/check.py http://127.0.0.1:8091 songs/catalog.json
"""

import json
import os
import sys
import urllib.request


def main(url, catalog_path):
    folder = os.path.dirname(catalog_path)
    problems = []
    for song in json.load(open(catalog_path)):
        path = os.path.join(folder, song["file"])
        data = open(path, "rb").read()
        request = urllib.request.Request(url + "/analyse", data=data, method="POST",
                                         headers={"X-Filename": song["file"], "Content-Type": "application/octet-stream"})
        try:
            with urllib.request.urlopen(request, timeout=900) as r:
                found = json.load(r)
        except Exception as e:  # noqa: BLE001
            problems.append(f"{song['id']}: {e}")
            continue
        bar = song["expect_bar"] * 60.0 / song["expect_bpm"]
        lead = song.get("expect_phrase_bar", 0)
        expected = [(lead + 8 * k) * bar for k in range(0, 40) if (lead + 8 * k) * bar < found["beats"][-1] - bar] \
            if found["beats"] else []
        starts = [s["start"] for s in found["segments"]]
        near = [t for t in expected if any(abs(t - s) < 1.0 for s in starts)]
        print(f"{song['id']}: {found['bpm']} BPM (expect {song['expect_bpm']:g}), {len(found['downbeats'])} downbeats, "
              f"{len(found['segments'])} sections in {found['seconds']} s on {found['device']}: "
              + ", ".join(f"{s['label']} {s['start']:.1f}" for s in found["segments"]))
        print(f"    section starts expected at {', '.join(f'{t:.1f}' for t in expected)}: {len(near)} of {len(expected)} found within 1 s")
        if not found["segments"]:
            problems.append(f"{song['id']}: no sections")
    if problems:
        print("\n".join(problems))
        sys.exit(1)


if __name__ == "__main__":
    main(*sys.argv[1:3])
