"""How a song sounds, heard here (0.27, the owner, 11 Nov: "Beats running its own analysis"): the two vectors and the
moods AudioMuse-AI gave each song, made by Siansa Beats itself, so a new song needs no AudioMuse-AI.

It's AudioMuse-AI's own recipe with AudioMuse-AI's own models, step for step, so a song heard here lands where
AudioMuse-AI would have put it and sits among the songs taken over from it (the page's check compares the two on
songs both have heard). Nothing is "improved": the same numbers are the point.

- **How it sounds** (the ``musicnn`` vector, 200 numbers): the song at 16 kHz mono, a mel spectrogram (96 bands,
  512-sample frames every 256, ``log10(1 + 10000 x)``), cut into patches of 187 frames; MusiCNN's embedding of each
  patch, averaged. From the same embeddings its 50 tags (rock, chillout, 80s…), of which the five strongest are kept
  as ``moods``, as AudioMuse-AI keeps them.
- **For the search by words** (the ``clap`` vector, 512 numbers): the song at 48 kHz, in 10-second pieces every 5
  seconds; each piece's mel spectrogram (128 bands, 0 to 14 kHz, in dB) through the distilled CLAP audio model
  (AudioMuse-AI's DCLAP); the pieces' vectors averaged and brought to length one.
- **How it feels** (``other``: danceable, aggressive, happy, party, relaxed, sad): how near the song's CLAP vector is
  to each of those words (CLAP's text model, :mod:`beats.words`), from 0 to 1.
- **Energy**: the mean loudness of its frames, as AudioMuse-AI measures it (0 at -60 dB, 1 at full scale).

Only the first :data:`MOST_SECONDS` of a file are heard, as AudioMuse-AI does. The tempo is never taken from here:
it's Beat This!'s. The key is this service's own measure.

The models (``musicnn_embedding.onnx``, ``musicnn_prediction.onnx``, ``model_epoch_36.onnx`` with its ``.data``) are
in the image, from AudioMuse-AI's releases, and run on the CPU with ONNX Runtime; librosa makes the spectrograms,
as in AudioMuse-AI, so they're the same to the last digit.
"""

import os
import subprocess
import threading
import time

import numpy as np

from . import log

_log = log.get("hear")

EMBEDDING = "musicnn_embedding.onnx"
PREDICTION = "musicnn_prediction.onnx"
CLAP_AUDIO = "model_epoch_36.onnx"
MOST_SECONDS = 600  # AudioMuse-AI's AUDIO_LOAD_TIMEOUT: the part of a file that's heard
FALLBACK_RATE = 48_000  # the rate ffmpeg decodes at when librosa can't read a file itself
TOP_MOODS = 5
FEELINGS = ("danceable", "aggressive", "happy", "party", "relaxed", "sad")  # AudioMuse-AI's OTHER_FEATURE_LABELS
# MusiCNN's 50 tags, in the model's order (AudioMuse-AI's MOOD_LABELS).
TAGS = (
    "rock", "pop", "alternative", "indie", "electronic", "female vocalists", "dance", "00s", "alternative rock", "jazz",
    "beautiful", "metal", "chillout", "male vocalists", "classic rock", "soul", "indie rock", "Mellow", "electronica",
    "80s", "folk", "90s", "chill", "instrumental", "punk", "oldies", "blues", "hard rock", "ambient", "acoustic",
    "experimental", "female vocalist", "guitar", "Hip-Hop", "70s", "party", "country", "easy listening", "sexy",
    "catchy", "funk", "electro", "heavy metal", "Progressive rock", "60s", "rnb", "indie pop", "sad", "House", "happy")


class CantHear(Exception):
    """A song couldn't be heard: why."""


def load(path, seconds=MOST_SECONDS):
    """The first ``seconds`` of the file at ``path`` as ``(mono float32, its own rate)``: read as AudioMuse-AI reads it
    (librosa, at the file's own rate), by ffmpeg when librosa can't (a format its reader doesn't know)."""
    import librosa

    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            audio, rate = librosa.load(path, sr=None, mono=True, duration=seconds)
        if audio is not None and audio.size:
            return audio.astype(np.float32), int(rate)
    except Exception as e:  # whatever its readers raise: ffmpeg is asked instead
        _log.debug("librosa couldn't read %s (%s); asking ffmpeg", os.path.basename(path), e)
    return decode(path, FALLBACK_RATE, seconds), FALLBACK_RATE


def decode(path, rate=FALLBACK_RATE, seconds=MOST_SECONDS):
    """The first ``seconds`` of the file at ``path`` as mono float32 at ``rate`` (ffmpeg's own high-quality resampler)."""
    command = ["ffmpeg", "-v", "error", "-nostdin", "-t", str(seconds), "-i", path, "-vn", "-ac", "1",
               "-af", f"aresample={rate}:resampler=soxr:precision=28", "-f", "f32le", "-"]
    try:
        done = subprocess.run(command, capture_output=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise CantHear(f"ffmpeg couldn't read the song: {e}") from None
    if done.returncode != 0 or not done.stdout:
        why = done.stderr.decode("utf-8", "replace").strip().splitlines()
        raise CantHear("ffmpeg couldn't read the song: " + (why[-1] if why else f"exit {done.returncode}"))
    return np.frombuffer(done.stdout, dtype=np.float32).copy()


def patches(audio, rate=16_000):
    """MusiCNN's input, as AudioMuse-AI makes it (``prepare_spectrogram_patches``): ``(patches, 187, 96)``, or None
    for a clip too short for one patch."""
    import librosa

    mel = librosa.feature.melspectrogram(y=audio, sr=rate, n_fft=512, hop_length=256, n_mels=96, window="hann",
                                         center=False, power=2.0, norm="slaney", htk=False)
    log_mel = np.log10(1 + 10000 * np.maximum(mel, 0.0))
    cut = [log_mel[:, i:i + 187] for i in range(0, log_mel.shape[1] - 187 + 1, 187)]
    if not cut:
        return None
    return np.array(cut).transpose(0, 2, 1).astype(np.float32)


def energy(audio):
    """AudioMuse-AI's ``_estimate_energy``: the mean of the frames' loudness, 0 at -60 dB and 1 at full scale."""
    import librosa

    if audio is None or not audio.size:
        return 0.0
    rms = librosa.feature.rms(y=audio)
    rms_db = librosa.amplitude_to_db(rms, ref=1.0, top_db=None)
    return float(np.mean(np.clip((rms_db + 60.0) / 60.0, 0.0, 1.0)))


def pieces(audio, length=480_000, step=240_000):
    """The 10-second pieces CLAP hears, every 5 seconds, the last one flush with the end (AudioMuse-AI's
    ``analyze_audio_file``); a shorter clip is one piece, padded with silence."""
    audio = np.clip(audio, -1.0, 1.0)
    audio = ((audio * 32767.0).astype(np.int16) / 32767.0).astype(np.float32)  # as 16-bit, as AudioMuse-AI hears it
    if len(audio) <= length:
        return [np.pad(audio, (0, length - len(audio)), mode="constant")]
    out = [audio[start:start + length] for start in range(0, len(audio) - length + 1, step)]
    if len(out) * step < len(audio):
        out.append(audio[-length:])
    return out


def clap_mel(piece, rate=48_000):
    """One piece's mel spectrogram for the CLAP audio model (AudioMuse-AI's ``compute_mel_spectrogram``)."""
    import librosa

    mel = librosa.feature.melspectrogram(y=piece, sr=rate, n_fft=2048, hop_length=480, win_length=2048, window="hann",
                                         center=True, pad_mode="reflect", power=2.0, n_mels=128, fmin=0, fmax=14000)
    mel = librosa.power_to_db(mel, ref=1.0, amin=1e-10, top_db=None)
    return mel[np.newaxis, np.newaxis, :, :].astype(np.float32)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Hearer:
    """Hears songs with AudioMuse-AI's models. ``words`` (:class:`beats.words.Words`) gives the six feelings' vectors."""

    def __init__(self, models_dir, words=None, threads=0):
        self.models_dir, self.words, self.threads = models_dir, words, threads
        self._lock = threading.Lock()
        self._sessions = None
        self._feelings = None
        self.load_seconds = None

    def _path(self, name):
        return os.path.join(self.models_dir, name)

    @property
    def available(self):
        """Whether the models are in the image (one from before 0.27, or the tests, has none)."""
        return all(os.path.exists(self._path(n)) for n in (EMBEDDING, PREDICTION, CLAP_AUDIO, CLAP_AUDIO + ".data"))

    @property
    def loaded(self):
        return self._sessions is not None

    def _load(self):
        if self._sessions is not None:
            return self._sessions
        if not self.available:
            raise CantHear("This Siansa Beats has no models to hear a song's sound with: its image is from before 0.27")
        import onnxruntime

        started = time.monotonic()
        options = onnxruntime.SessionOptions()
        if self.threads:
            options.intra_op_num_threads = self.threads
        self._sessions = tuple(onnxruntime.InferenceSession(self._path(n), options, providers=["CPUExecutionProvider"])
                               for n in (EMBEDDING, PREDICTION, CLAP_AUDIO))
        self.load_seconds = round(time.monotonic() - started, 1)
        _log.info("The models that hear a song's sound are loaded (%.1fs)", self.load_seconds)
        return self._sessions

    def feelings(self):
        """The six feelings as vectors in CLAP's space (worked out once, with the text model), or None without it."""
        if self._feelings is None and self.words is not None and self.words.available:
            self._feelings = {word: self.words.embed(word) for word in FEELINGS}
        return self._feelings

    def __call__(self, path):
        """The sound of the song at ``path``, as kept in the ``sound`` table: ``{"musicnn", "clap" (float32 bytes),
        "moods", "other" (AudioMuse-AI's packed scores), "energy"}``. :class:`CantHear` says why not."""
        import librosa

        with self._lock:
            embedding, prediction, clap = self._load()
            native, rate = load(path)
            if not native.size or not np.any(native):
                raise CantHear("The song is silent, or nothing of it could be read")
            # Each model's rate from the file's own, with the resampler AudioMuse-AI uses.
            low = native if rate == 16_000 else librosa.resample(native, orig_sr=rate, target_sr=16_000, res_type="soxr_hq")
            high = native if rate == 48_000 else librosa.resample(native, orig_sr=rate, target_sr=48_000, res_type="soxr_hq")
            cut = patches(low)
            if cut is None:
                raise CantHear("The song is too short to hear (under 3 seconds)")
            per_patch = np.concatenate([embedding.run(None, {embedding.get_inputs()[0].name: cut[i:i + 64]})[0]
                                        for i in range(0, len(cut), 64)], axis=0)
            logits = prediction.run(None, {prediction.get_inputs()[0].name: per_patch})[0]
            scores = sigmoid(np.mean(sigmoid(logits), axis=0))  # twice, as AudioMuse-AI does
            top = sorted(zip(TAGS, (float(s) for s in scores)), key=lambda pair: pair[1], reverse=True)[:TOP_MOODS]
            musicnn = np.mean(per_patch, axis=0).astype(np.float32)

            heard = np.vstack([clap.run(None, {"mel_spectrogram": clap_mel(piece)})[0] for piece in pieces(high)])
            vector = np.mean(heard, axis=0)
            vector = (vector / (np.linalg.norm(vector) + 1e-9)).astype(np.float32)
        feelings = self.feelings()
        other = ",".join(f"{word}:{(float(vector @ feelings[word]) + 1.0) / 2.0:.2f}" if feelings else f"{word}:0.00"
                         for word in FEELINGS)
        return {"musicnn": musicnn.tobytes(), "clap": vector.tobytes(),
                "moods": ",".join(f"{tag}:{score:.3f}" for tag, score in top), "other": other,
                "energy": energy(low)}
