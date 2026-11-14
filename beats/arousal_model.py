"""Essentia's emoMusic arousal and valence model (TensorFlow) as ONNX, for the image (Dockerfile, the "arousal"
stage). Its metadata says its input and output; the usual names are the fallback."""

import json
import subprocess
import sys
import urllib.request

BASE = "https://essentia.upf.edu/models/classification-heads/emomusic/emomusic-msd-musicnn-2"


def fetch(address, to):
    for attempt in range(5):
        try:
            urllib.request.urlretrieve(address, to)
            return
        except OSError as e:
            print(f"download {attempt + 1} of {address} failed: {e}", flush=True)
    raise SystemExit(f"couldn't download {address}")


def main(out):
    fetch(BASE + ".pb", "/tmp/model.pb")
    fetch(BASE + ".json", "/tmp/model.json")
    schema = json.load(open("/tmp/model.json")).get("schema") or {}
    inputs = [i["name"] for i in schema.get("inputs") or []] or ["model/Placeholder"]
    outputs = [o["name"] for o in schema.get("outputs") or [] if o.get("output_purpose", "predictions") == "predictions"] \
        or ["model/Identity"]
    print("inputs", inputs, "outputs", outputs, flush=True)
    subprocess.run([sys.executable, "-m", "tf2onnx.convert", "--graphdef", "/tmp/model.pb", "--inputs", f"{inputs[0]}:0",
                    "--outputs", f"{outputs[0]}:0", "--opset", "13", "--output", out], check=True)


if __name__ == "__main__":
    main(sys.argv[1])
