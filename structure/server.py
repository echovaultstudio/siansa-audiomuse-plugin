"""Siansa Structure: where a song's sections start, from All-In-One, for Siansa Beats.

    POST /analyse   the song's audio file as the body (X-Filename: "song.flac" for its kind) →
                    {"bpm", "beats", "downbeats", "segments": [{"start", "end", "label"}], "seconds", "device"}
    GET  /health    {"status": "ok", "version", "device", "loaded"}

All-In-One (github.com/mir-aidj/all-in-one, MIT) separates the song into drums, bass, vocals and the rest (Demucs),
then its eight models (trained on the Harmonix Set) find the beats, the downbeats and the sections, labelled intro,
verse, chorus, break, bridge, inst, solo, outro. The models load once, with the first song; songs go one at a time.

Settings (environment): PORT (8091), STRUCTURE_DEVICE (auto: the GPU when there is one), STRUCTURE_TOKEN (when set,
requests need "Authorization: Bearer <it>").
"""

import hmac
import json
import os
import sys
import tempfile
import threading
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

VERSION = "0.1.0"
PORT = int(os.environ.get("PORT", "8091"))
TOKEN = os.environ.get("STRUCTURE_TOKEN", "")
MAX_BYTES = 400 * 1024 * 1024
KINDS = {".flac", ".mp3", ".wav", ".ogg", ".opus", ".m4a", ".aac", ".wma", ".aiff", ".aif"}


def log(message):
    print(time.strftime("%H:%M:%S"), message, file=sys.stderr, flush=True)


class Analyser:
    def __init__(self, device):
        self.wanted = device or "auto"
        self.device = None
        self.model = None
        self.lock = threading.Lock()

    def load(self):
        import torch
        from allin1.models import load_pretrained_model

        if self.wanted == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.wanted
        started = time.time()
        self.model = load_pretrained_model("harmonix-all", device=self.device)
        log(f"All-In-One's models loaded on {self.device} in {time.time() - started:.1f} s")

    def __call__(self, path):
        import torch
        from allin1.demix import demix
        from allin1.helpers import run_inference
        from allin1.spectrogram import extract_spectrograms

        with self.lock:
            if self.model is None:
                self.load()
            started = time.time()
            with tempfile.TemporaryDirectory() as work:
                work = Path(work)
                demixed = demix([path], work / "demix", self.device)
                specs = extract_spectrograms(demixed, work / "spec", multiprocess=False)
                with torch.no_grad():
                    result = run_inference(path=path, spec_path=specs[0], model=self.model, device=self.device,
                                           include_activations=False, include_embeddings=False)
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
            return {
                "bpm": result.bpm,
                "beats": [round(float(b), 3) for b in result.beats],
                "downbeats": [round(float(b), 3) for b in result.downbeats],
                "segments": [{"start": round(float(s.start), 3), "end": round(float(s.end), 3), "label": s.label}
                             for s in result.segments],
                "seconds": round(time.time() - started, 2),
                "device": self.device,
            }


ANALYSER = Analyser(os.environ.get("STRUCTURE_DEVICE", "auto"))


class Handler(BaseHTTPRequestHandler):
    server_version = f"SiansaStructure/{VERSION}"

    def log_message(self, fmt, *args):  # one line a request, without the body
        log(f"{self.command} {self.path.split('?')[0]} {args[1] if len(args) > 1 else ''}")

    def _json(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _allowed(self):
        if not TOKEN:
            return True
        given = self.headers.get("Authorization", "")
        if hmac.compare_digest(given.encode(), f"Bearer {TOKEN}".encode()):
            return True
        self._json(HTTPStatus.UNAUTHORIZED, {"error": "STRUCTURE_TOKEN needed"})
        return False

    def do_GET(self):
        if self.path.split("?")[0] == "/health":
            return self._json(HTTPStatus.OK, {"status": "ok", "version": VERSION, "device": ANALYSER.device or ANALYSER.wanted,
                                              "loaded": ANALYSER.model is not None})
        self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self):
        if self.path.split("?")[0] != "/analyse":
            return self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        if not self._allowed():
            return
        size = int(self.headers.get("Content-Length") or 0)
        if not 0 < size <= MAX_BYTES:
            return self._json(HTTPStatus.BAD_REQUEST, {"error": "the song's audio file is needed as the body"})
        kind = Path(self.headers.get("X-Filename") or "song.flac").suffix.lower()
        if kind not in KINDS:
            kind = ".flac"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / f"song{kind}"
            with open(path, "wb") as f:
                left = size
                while left:
                    chunk = self.rfile.read(min(left, 1 << 20))
                    if not chunk:
                        return self._json(HTTPStatus.BAD_REQUEST, {"error": "the body ended early"})
                    f.write(chunk)
                    left -= len(chunk)
            try:
                found = ANALYSER(path)
            except Exception as e:  # noqa: BLE001 - the reason goes back to Siansa Beats
                traceback.print_exc()
                return self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(e).__name__}: {e}"})
        log(f"{len(found['segments'])} sections, {found['bpm']} BPM, in {found['seconds']} s on {found['device']}")
        self._json(HTTPStatus.OK, found)


if __name__ == "__main__":
    log(f"Siansa Structure {VERSION} on port {PORT}")
    ThreadingHTTPServer(("", PORT), Handler).serve_forever()
