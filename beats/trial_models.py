"""Essentia's small models (TensorFlow) as ONNX, for the energy trial's image (Dockerfile.trial).

Each model's metadata says its input and output; the usual names are the fallback."""

import json
import subprocess
import sys
import urllib.request

BASE = "https://essentia.upf.edu/models/classification-heads"
MODELS = ("deam/deam-msd-musicnn-2", "emomusic/emomusic-msd-musicnn-2", "muse/muse-msd-musicnn-2",
          "engagement/engagement_regression-discogs-effnet-1",
          "approachability/approachability_regression-discogs-effnet-1")


def fetch(address, to):
    for attempt in range(5):
        try:
            urllib.request.urlretrieve(address, to)
            return
        except OSError as e:
            print(f"download {attempt + 1} of {address} failed: {e}", flush=True)
    raise SystemExit(f"couldn't download {address}")


def main(out):
    for model in MODELS:
        name = model.split("/")[1]
        fetch(f"{BASE}/{model}.pb", f"/tmp/{name}.pb")
        fetch(f"{BASE}/{model}.json", f"/tmp/{name}.json")
        schema = json.load(open(f"/tmp/{name}.json")).get("schema") or {}
        inputs = [i["name"] for i in schema.get("inputs") or []] or ["model/Placeholder"]
        outputs = [o["name"] for o in schema.get("outputs") or [] if o.get("output_purpose", "predictions") == "predictions"] \
            or ["model/Identity"]
        print(name, "inputs", inputs, "outputs", outputs, flush=True)
        subprocess.run([sys.executable, "-m", "tf2onnx.convert", "--graphdef", f"/tmp/{name}.pb",
                        "--inputs", ",".join(f"{i}:0" for i in inputs[:1]), "--outputs", ",".join(f"{o}:0" for o in outputs[:1]),
                        "--opset", "13", "--output", f"{out}/{name}.onnx"], check=True)


if __name__ == "__main__":
    main(sys.argv[1])
