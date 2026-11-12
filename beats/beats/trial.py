"""The energy trial (the owner, 12 Nov): several ways to say how energetic a song is, side by side, to choose from.

Only in the trial image (``Dockerfile.trial``, tag ``trial``), which has Essentia's small models in ``/models/trial``;
any other image has none and this does nothing. It is not part of Siansa Beats' analysis and changes none of it.

For each song analysed, :class:`Trial` works out and keeps (table ``trial``):

- **today's energy** (``energy_now``: AudioMuse-AI's current measure, the loudness of the frames on 0..1) and **the
  old one** (``energy_old``: the plain mean of the frames' RMS, what the songs taken over from an older AudioMuse-AI
  carry), so the two scales can be seen on the same songs;
- **arousal and valence** from Essentia's three models on MusiCNN (DEAM, emoMusic, MuSe; each 1..9), twice: from every
  patch of the song and averaged (``…_arousal``: how Essentia means them to be used), and from the song's one mean
  vector (``…_arousal_stored``: what could be worked out for the whole library from the vectors already kept, without
  decoding a song). How close the two are says whether the cheap way is good enough;
- **engagement and approachability** (Essentia's regressions on the Discogs-EffNet embedding the style model makes);
- **the ingredients of a blend**: how often a note starts (``onsets``, a second), how much of the sound is bass
  (``bass``, under 150 Hz), how bright it is (``bright``, the spectral centroid in Hz), and how much its loudness
  moves (``range_db``, the loud frames over the quiet ones).

The page ``/trial`` lists them, sortable, with each song's tempo and loudness; ``/api/trial.csv`` is the same as a file.
"""

import json
import os
import threading
import time

import numpy as np

from . import hear, log, memory
from . import style as style_module

_log = log.get("trial")

FOLDER = "trial"
# name -> (file, the embedding it reads, its outputs in order)
HEADS = {
    "deam": ("deam-msd-musicnn-2.onnx", "musicnn", ("valence", "arousal")),
    "emomusic": ("emomusic-msd-musicnn-2.onnx", "musicnn", ("valence", "arousal")),
    "muse": ("muse-msd-musicnn-2.onnx", "musicnn", ("valence", "arousal")),
    "engagement": ("engagement_regression-discogs-effnet-1.onnx", "effnet", ("engagement",)),
    "approachability": ("approachability_regression-discogs-effnet-1.onnx", "effnet", ("approachability",)),
}
COLUMNS = (
    "energy_now", "energy_old",
    "deam_arousal", "emomusic_arousal", "muse_arousal",
    "deam_arousal_stored", "emomusic_arousal_stored", "muse_arousal_stored",
    "deam_valence", "emomusic_valence", "muse_valence",
    "engagement", "approachability",
    "onsets", "bass", "bright", "range_db",
)


def ingredients(low, rate=16_000):
    """What a blend could be made of, from 16 kHz mono ``low``: ``{"energy_now", "energy_old", "onsets", "bass",
    "bright", "range_db"}``."""
    import librosa

    rms = librosa.feature.rms(y=low)[0]
    db = librosa.amplitude_to_db(rms, ref=1.0, top_db=None)
    spectrum = np.abs(librosa.stft(low, n_fft=2048, hop_length=512)) ** 2
    freqs = librosa.fft_frequencies(sr=rate, n_fft=2048)
    total = float(spectrum.sum()) or 1.0
    seconds = max(len(low) / float(rate), 1e-9)
    starts = librosa.onset.onset_detect(y=low, sr=rate, hop_length=512)
    heard = db[db > -60.0]
    return {
        "energy_now": hear.energy(low),
        "energy_old": float(np.mean(rms)) if rms.size else 0.0,
        "onsets": float(len(starts) / seconds),
        "bass": float(spectrum[freqs < 150.0].sum() / total),
        "bright": float((freqs[:, None] * spectrum).sum() / total),
        "range_db": float(np.percentile(heard, 95) - np.percentile(heard, 10)) if heard.size else 0.0,
    }


class Trial:
    def __init__(self, models_dir, store=None, threads=None):
        self.models_dir, self.store, self.threads = models_dir, store, threads
        self.folder = os.path.join(models_dir, FOLDER)
        self._sessions = None
        self._lock = threading.Lock()
        if store is not None:
            with store.lock:
                store.db.execute("CREATE TABLE IF NOT EXISTS trial (song_id TEXT PRIMARY KEY, found TEXT, at REAL, error TEXT)")
                store.db.commit()

    @property
    def available(self):
        """Whether this is the trial image: Essentia's models are there."""
        return all(os.path.exists(os.path.join(self.folder, HEADS[name][0])) for name in HEADS)

    @property
    def loaded(self):
        return self._sessions is not None

    def unload(self):
        with self._lock:
            had, self._sessions = self._sessions is not None, None
            return had

    def _load(self):
        if self._sessions is None:
            import onnxruntime

            def session(path):
                return onnxruntime.InferenceSession(path, memory.session_options(onnxruntime, self.threads),
                                                    providers=["CPUExecutionProvider"])

            sessions = {name: session(os.path.join(self.folder, HEADS[name][0])) for name in HEADS}
            sessions["musicnn"] = session(os.path.join(self.models_dir, hear.EMBEDDING))
            effnet = session(os.path.join(self.models_dir, style_module.MODEL_FILE))
            # The output that is the embedding: not the 400 styles' scores.
            sessions["effnet"] = effnet
            sessions["effnet_output"] = next(o.name for o in effnet.get_outputs() if o.shape and o.shape[-1] == 1280)
            self._sessions = sessions
            _log.info("The trial's models are loaded")
        return self._sessions

    @staticmethod
    def _run(session, rows):
        return session.run(None, {session.get_inputs()[0].name: np.asarray(rows, dtype=np.float32)})[0]

    def measure(self, path):
        """Every candidate for the song at ``path``: ``{column: number}`` (see the module's words)."""
        import librosa

        with self._lock:
            sessions = self._load()
            native, rate = hear.load(path)
            if not native.size or not np.any(native):
                raise hear.CantHear("The song is silent, or nothing of it could be read")
            low = native if rate == 16_000 else librosa.resample(native, orig_sr=rate, target_sr=16_000, res_type="soxr_hq")
            found = ingredients(low)

            cut = hear.patches(low)
            if cut is not None:
                musicnn = sessions["musicnn"]
                per_patch = np.concatenate([self._run(musicnn, cut[i:i + 64]) for i in range(0, len(cut), 64)], axis=0)
                mean = np.mean(per_patch, axis=0, keepdims=True)
                for name, (_, reads, outputs) in HEADS.items():
                    if reads != "musicnn":
                        continue
                    each = np.mean(self._run(sessions[name], per_patch), axis=0)
                    stored = self._run(sessions[name], mean)[0]
                    for i, what in enumerate(outputs):
                        found[f"{name}_{what}"] = float(each[i])
                        if what == "arousal":
                            found[f"{name}_arousal_stored"] = float(stored[i])

            batch = style_module.patches(style_module.mel(low))
            effnet = sessions["effnet"]
            embedding = effnet.run([sessions["effnet_output"]], {effnet.get_inputs()[0].name: batch})[0]
            for name, (_, reads, outputs) in HEADS.items():
                if reads == "effnet":
                    found[outputs[0]] = float(np.mean(self._run(sessions[name], embedding), axis=0).reshape(-1)[0])
        return found

    # ---- kept, and shown ----

    def keep(self, song_id, found=None, error=None):
        with self.store.lock:
            self.store.db.execute("INSERT OR REPLACE INTO trial (song_id, found, at, error) VALUES (?, ?, ?, ?)",
                                  (song_id, json.dumps(found) if found else None, time.time(), error))
            self.store.db.commit()

    def hear(self, song, path):
        """Measures the song at ``path`` and keeps it (or why not): never stops its analysis. True when it was kept."""
        if not self.available:
            return False
        started = time.monotonic()
        try:
            found = self.measure(path)
            self.keep(song["id"], found)
            _log.info("%s – %s: %s (%.1fs)", song.get("artist"), song.get("title"),
                      ", ".join(f"{k} {found[k]:.3f}" for k in ("energy_now", "energy_old", "deam_arousal", "engagement") if k in found),
                      time.monotonic() - started)
            return True
        except Exception as e:
            _log.warning("%s – %s: the trial couldn't measure it: %s", song.get("artist"), song.get("title"), e)
            self.keep(song["id"], error=str(e) or e.__class__.__name__)
            return False

    def rows(self, model=None):
        """Every song measured, with its title, artist, album, tempo and loudness: a list of dicts, by artist and album."""
        kept = self.store._rows("SELECT t.song_id, t.found, s.title, s.artist, s.album FROM trial t JOIN songs s ON s.id=t.song_id "
                                "WHERE t.found IS NOT NULL ORDER BY s.artist, s.album, s.title")
        ids = [r["song_id"] for r in kept]
        tempos = self.store.tempos(ids, model) if model else {}
        measured = self.store.measures(ids)
        out = []
        for r in kept:
            row = {"id": r["song_id"], "artist": r["artist"], "album": r["album"], "title": r["title"],
                   "tempo": tempos.get(r["song_id"]), "lufs": (measured.get(r["song_id"]) or {}).get("ref_lufs")}
            row.update(json.loads(r["found"]))
            out.append(row)
        return out

    def csv(self, model=None):
        import csv
        import io

        head = ("artist", "album", "title", "tempo", "lufs") + COLUMNS
        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow(head)
        for row in self.rows(model):
            writer.writerow(["" if row.get(k) is None else (round(row[k], 4) if isinstance(row[k], float) else row[k]) for k in head])
        return out.getvalue()


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Siansa Beats: the energy trial</title><style>
body{margin:0;background:#0b0d12;color:#e8e8ea;font:14px/1.4 system-ui,sans-serif}
header{padding:16px}h1{margin:0 0 4px;font-size:20px}p{margin:4px 0;color:#a8abb3;max-width:70ch}a{color:#ffb547}
.wrap{overflow:auto;max-height:calc(100vh - 150px)}table{border-collapse:collapse;white-space:nowrap}
th,td{padding:6px 10px;border-bottom:1px solid #20242c;text-align:right}
th{position:sticky;top:0;background:#151922;cursor:pointer;user-select:none}th.on{color:#ffb547}
td.t,th.t{text-align:left;max-width:220px;overflow:hidden;text-overflow:ellipsis}
td.bar{background:linear-gradient(90deg,rgba(255,181,71,.28) var(--w),transparent var(--w))}
</style></head><body><header><h1>The energy trial</h1>
<p>Each column is one way to say how energetic a song is. Press a column to order the songs by it, then judge by ear
which order feels right. The bar in a cell is where the song stands among these songs. <a href="/api/trial.csv">As a file</a>.</p>
<p id="count"></p></header><div class="wrap"><table id="t"></table></div><script>
const text=["artist","album","title"];let rows=[],by="deam_arousal",down=true;
const cols=["artist","album","title","tempo","lufs",%s];
function draw(){const t=document.getElementById("t");const lo={},hi={};
for(const c of cols){if(text.includes(c))continue;const v=rows.map(r=>r[c]).filter(v=>typeof v==="number");lo[c]=Math.min(...v);hi[c]=Math.max(...v)}
rows.sort((a,b)=>{const x=a[by],y=b[by];if(x==null)return 1;if(y==null)return -1;return (x<y?-1:x>y?1:0)*(down?-1:1)});
let h="<tr>"+cols.map(c=>`<th class="${text.includes(c)?"t ":""}${c===by?"on":""}" data-c="${c}">${c.replace(/_/g," ")}${c===by?(down?" ↓":" ↑"):""}</th>`).join("")+"</tr>";
for(const r of rows){h+="<tr>"+cols.map(c=>{const v=r[c];if(text.includes(c))return `<td class="t" title="${String(v??"").replace(/"/g,"&quot;")}">${String(v??"").replace(/</g,"&lt;")}</td>`;
if(typeof v!=="number")return "<td></td>";const w=hi[c]>lo[c]?Math.round(100*(v-lo[c])/(hi[c]-lo[c])):0;
return `<td class="bar" style="--w:${w}%%">${Math.abs(v)>=100?v.toFixed(0):v.toFixed(3)}</td>`}).join("")+"</tr>"}
t.innerHTML=h;t.querySelectorAll("th").forEach(e=>e.onclick=()=>{const c=e.dataset.c;if(c===by)down=!down;else{by=c;down=!text.includes(c)}draw()})}
async function load(){const r=await fetch("/api/trial");rows=(await r.json()).songs;
document.getElementById("count").textContent=rows.length+" songs measured so far. This page reads again every 20 seconds.";draw()}
load();setInterval(load,20000);
</script></body></html>
""" % ",".join(f'"{c}"' for c in COLUMNS)
