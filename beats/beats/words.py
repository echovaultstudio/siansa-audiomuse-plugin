"""Music found by describing it in words (0.26, the owner, 11 Nov): "chill relaxed mellow grooves", the mood radios,
the vibes and their sliders. Until now AudioMuse-AI did this; Siansa Beats does it the same way, with the same models,
so the words land among the songs exactly as they did there.

Every song has a second vector, from CLAP (512 numbers; taken over from AudioMuse-AI, later made here), in a space
that words share: CLAP's text model turns a description into a vector of the same kind, and the songs nearest it are
the ones that sound like the description.

- **The text model** is LAION CLAP's text tower as ONNX (``clap_text_model.onnx``, 500 MB, in the image), unchanged,
  with RoBERTa's tokenizer (``roberta-tokenizer.json``, here): 77 tokens, padded, as AudioMuse-AI asks it.
- **Steering** ("more piano", "less vocals": the vibes' sliders) is AudioMuse-AI's: a small pair of models (a sparse
  autoencoder over CLAP's space, ``dclap_sae_…_encoder.onnx`` and ``…_decoder.onnx``) and its catalogue of concepts
  (``sound_concepts.json``, AudioMuse-AI's ``dclap_sae_concepts.json``: 109 terms in 7 groups, each a few of the
  autoencoder's latents and how much each counts). A concept is added to (or taken from) the query's latents, and the
  change that makes, decoded, is added to the query.

The models are loaded when first asked for and let go after :data:`IDLE` seconds without a search (the text model
takes about 600 MB while loaded). They run on the CPU: a query takes a fraction of a second.
"""

import json
import os
import threading
import time

import numpy as np

from . import log

_log = log.get("words")

HERE = os.path.dirname(__file__)
TEXT_MODEL = "clap_text_model.onnx"
ENCODER = "dclap_sae_k20_d1024_best_encoder.onnx"
DECODER = "dclap_sae_k20_d1024_best_decoder.onnx"
TOKENS = 77
IDLE = 600  # seconds without a search before the models are let go
STEPS = (1.0, 2.0, 3.0, 5.0, 10.0)  # AudioMuse-AI's CLAP_SAE_ALPHA_STEPS: how strongly a concept can steer
DEFAULT_STEP = 3.0
MOST_TERMS = 10


class NotThere(Exception):
    """The search by words can't run here: why."""


class Words:
    def __init__(self, models_dir, tokenizer=os.path.join(HERE, "roberta-tokenizer.json"),
                 concepts=os.path.join(HERE, "sound_concepts.json")):
        self.models_dir, self._tokenizer_file, self._concepts_file = models_dir, tokenizer, concepts
        self._lock = threading.RLock()
        self._text = self._tokenizer = self._encoder = self._decoder = None
        self._catalogue = None
        self.last_used = 0.0
        self.load_seconds = None

    def _path(self, name):
        return os.path.join(self.models_dir, name)

    @property
    def available(self):
        """Whether the text model is in the image (an image from before 0.26, or the tests, has none)."""
        return os.path.exists(self._path(TEXT_MODEL)) and os.path.exists(self._tokenizer_file)

    @property
    def steers(self):
        return self.available and all(os.path.exists(p) for p in (self._path(ENCODER), self._path(DECODER),
                                                                  self._concepts_file))

    @property
    def loaded(self):
        return self._text is not None

    # ---- the models ----

    def _load(self):
        if self._text is not None:
            return
        if not self.available:
            raise NotThere("This Siansa Beats has no text model: its image is from before 0.26")
        import onnxruntime
        from tokenizers import Tokenizer

        started = time.monotonic()
        tokenizer = Tokenizer.from_file(self._tokenizer_file)
        tokenizer.enable_truncation(TOKENS)
        tokenizer.enable_padding(length=TOKENS, pad_id=1, pad_token="<pad>")
        self._tokenizer = tokenizer
        self._text = onnxruntime.InferenceSession(self._path(TEXT_MODEL), providers=["CPUExecutionProvider"])
        self.load_seconds = round(time.monotonic() - started, 1)
        _log.info("The text model is loaded (%.1fs)", self.load_seconds)

    def _load_steering(self):
        if self._encoder is not None:
            return True
        if not self.steers:
            return False
        import onnxruntime

        self._encoder = onnxruntime.InferenceSession(self._path(ENCODER), providers=["CPUExecutionProvider"])
        self._decoder = onnxruntime.InferenceSession(self._path(DECODER), providers=["CPUExecutionProvider"])
        return True

    def unload_if_idle(self, now=None):
        """Lets the models go after :data:`IDLE` seconds without a search; True when it did."""
        with self._lock:
            if self._text is None or (now or time.time()) - self.last_used < IDLE:
                return False
            self._text = self._tokenizer = self._encoder = self._decoder = None
            _log.info("The text model was let go after %d minutes without a search", IDLE // 60)
            return True

    # ---- words to a vector ----

    def embed(self, text):
        """``text`` as a vector in the songs' CLAP space, at length one."""
        with self._lock:
            self._load()
            self.last_used = time.time()
            encoded = self._tokenizer.encode(str(text))
            out = self._text.run(None, {"input_ids": np.array([encoded.ids], np.int64),
                                        "attention_mask": np.array([encoded.attention_mask], np.int64)})[0][0]
        length = float(np.linalg.norm(out))
        if not np.isfinite(length) or length <= 1e-8:
            raise NotThere("The text model gave nothing for those words")
        return (out / length).astype(np.float32)

    # ---- concepts and steering (AudioMuse-AI's) ----

    def _concepts(self):
        if self._catalogue is None:
            with open(self._concepts_file, encoding="utf-8") as f:
                self._catalogue = json.load(f)
            self._by_term = {c["term"]: c for c in self._catalogue.get("concepts") or []}
        return self._catalogue

    def catalogue(self):
        """The concepts a search can be steered by, in AudioMuse-AI's layout: groups of ``{term, grounding}``."""
        if not self.steers:
            return {"available": False, "reason": "This Siansa Beats has no steering models: its image is from before "
                                                  "0.26", "categories": [], "alpha_steps": list(STEPS),
                    "default_alpha": DEFAULT_STEP, "max_terms": MOST_TERMS}
        catalogue, grouped = self._concepts(), {}
        for entry in catalogue.get("concepts") or []:
            group = grouped.setdefault(entry["category"], {"category": entry["category"], "label": entry["label"],
                                                           "terms": []})
            group["terms"].append({"term": entry["term"], "grounding": entry["grounding"]})
        order = catalogue.get("category_order") or sorted(grouped)
        return {"available": True, "reason": None, "categories": [grouped[name] for name in order if name in grouped],
                "alpha_steps": list(STEPS), "default_alpha": DEFAULT_STEP, "max_terms": MOST_TERMS}

    def terms(self, asked):
        """What's asked to steer by, cleaned as AudioMuse-AI cleans it: ``([{"term", "weight", "direction"}],
        [what was wrong with the rest])``. An unknown term is left out and said, not refused."""
        if not asked:
            return [], []
        if not isinstance(asked, list):
            return [], ["steering must be a list of {term, weight, direction} objects"]
        if not self.steers:
            return [], ["concept steering is unavailable"]
        self._concepts()
        cleaned, problems, seen = [], [], set()
        for item in asked[:MOST_TERMS + 1]:
            if len(cleaned) >= MOST_TERMS:
                problems.append(f"at most {MOST_TERMS} concepts can be combined")
                break
            if isinstance(item, str):
                item = {"term": item}
            if not isinstance(item, dict):
                problems.append("each steering entry must be a string or an object")
                continue
            term = str(item.get("term") or "").strip()
            if not term:
                problems.append("a steering entry has no term")
            elif term not in self._by_term:
                problems.append(f'"{term}" is not a validated concept for this library')
            elif term in seen:
                problems.append(f'"{term}" was passed more than once')
            else:
                try:
                    weight = float(item.get("weight", DEFAULT_STEP))
                except (TypeError, ValueError):
                    problems.append(f'"{term}" has a non numeric weight')
                    continue
                direction = str(item.get("direction") or "more").lower()
                if direction not in ("more", "less"):
                    problems.append(f'"{term}" has direction "{direction}", expected more or less')
                    continue
                seen.add(term)
                cleaned.append({"term": term, "weight": min(STEPS, key=lambda step: abs(step - abs(weight))),
                                "direction": direction})
        return cleaned, problems

    def steer(self, vector, terms):
        """``vector`` (a query) moved towards or away from ``terms`` (from :meth:`terms`), at the length it had;
        as it was when steering can't run."""
        if not terms:
            return vector
        with self._lock:
            if not self._load_steering():
                return vector
            self._concepts()
            self.last_used = time.time()
            query = np.asarray(vector, np.float32).reshape(1, -1)
            length = float(np.linalg.norm(query))
            if length <= 1e-8:
                return vector
            edit = np.zeros((1, int(self._catalogue.get("d_sae") or 1024)), np.float32)
            less = False
            for term in terms:
                entry = self._by_term.get(term["term"])
                if entry is None:
                    continue
                alpha = term["weight"] if term["direction"] == "more" else -term["weight"]
                less = less or alpha < 0
                edit[0, np.asarray(entry["support"], np.int64)] += alpha * np.asarray(entry["mask"], np.float32)
            try:
                original = self._encoder.run(None, {self._encoder.get_inputs()[0].name: query})[0].astype(np.float32)
                edited = np.maximum(original + edit, 0.0) if less else original + edit
                decoded = self._decoder.run(None, {self._decoder.get_inputs()[0].name:
                                                   np.concatenate([original, edited], axis=0)})[0].astype(np.float32)
            except Exception:
                _log.exception("Steering failed: the words are searched as they are")
                return vector
        steered = query.reshape(-1) + (decoded[1] - decoded[0])
        norm = float(np.linalg.norm(steered))
        if norm <= 1e-8 or not np.isfinite(steered).all():
            return vector
        return (steered / norm * length).astype(np.float32)
