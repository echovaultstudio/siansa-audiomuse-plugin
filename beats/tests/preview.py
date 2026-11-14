#!/usr/bin/env python3
"""The service with a stand-in for the model, on the songs from make_test_songs.py and the fake Navidrome: for
working on the page without PyTorch. The stand-in gives each song the beats of its known tempo (with a few beats
missed and a 20 ms grid, like the model's), and reads the 174 BPM song at half time, as a model might.

    python3 tests/preview.py <songs folder> [port]      then open http://127.0.0.1:<port> (default 8090)
"""

import hashlib
import json
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from beats.server import App, Config, make_server  # noqa: E402
from fakes import PASSWORD, USER, FakeServer  # noqa: E402


def main(folder, port=8090):
    with open(os.path.join(folder, "catalog.json")) as f:
        catalog = json.load(f)
    by_hash = {}
    for song in catalog:
        with open(os.path.join(folder, song["file"]), "rb") as f:
            by_hash[hashlib.sha1(f.read()).hexdigest()] = song

    class Samples:
        def __init__(self, song, path):
            self.song, self.path = song, path

        def __len__(self):
            return int(self.song["duration"] * 22_050)

    def decode(path):
        with open(path, "rb") as f:
            return Samples(by_hash[hashlib.sha1(f.read()).hexdigest()], path)

    class Tracker:
        loaded, load_seconds, device, device_name, device_note = True, 0.0, "cpu", "a stand-in", None

        def __call__(self, samples):
            song = samples.song
            rng = random.Random(song["id"])
            bpm = song["expect_bpm"] / (2 if song["expect_bpm"] > 170 else 1)
            period = 60.0 / bpm
            beats = [round((i * period) / 0.02) * 0.02 for i in range(int((song["duration"] - 0.5) / period))]
            beats = [b for b in beats if rng.random() > 0.02]
            return beats, beats[::song["expect_bar"]]

    fake = FakeServer(folder).start()
    data = tempfile.mkdtemp(prefix="beats-preview-")
    env = {"NAVIDROME_URL": fake.url, "NAVIDROME_USER": USER, "NAVIDROME_PASSWORD": PASSWORD,
           "DATA_DIR": data, "PORT": str(port)}
    def measure(samples):  # the real sound, for the phrases: needs ffmpeg and numpy
        from beats import analyser, features

        return features.measure(analyser.decode(samples.path))

    from beats import log

    log.setup("debug")
    app = App(Config(env), tracker=Tracker(), decode=decode, measure=measure)
    app.worker.start()
    app.store.event("Preview: a stand-in instead of the model")
    print(f"Preview on http://127.0.0.1:{port} (fake Navidrome {fake.url}, data {data})", flush=True)
    make_server(app, host="127.0.0.1", port=port).serve_forever()


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8090)
