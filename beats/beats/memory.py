"""How much memory Siansa Beats holds, and handing back what it no longer needs (0.34).

A service that analyses songs grows and stays grown for three reasons, each met here:

* **The models stay loaded** after the last song (Beat This!, the style model, the models that hear a song's sound:
  a few gigabytes together). They are let go after :data:`MODELS_IDLE` seconds without a song to analyse
  (:meth:`beats.worker.Worker.let_models_go_if_idle`), as the text model already was, and loaded again by the next
  song in a few seconds.
* **ONNX Runtime keeps its largest working memory** for the next run (its "arena"): one long song, and it holds
  that much for good. :func:`session_options` turns the arena off, so each run's memory goes back when it ends. The
  numbers that come out are the same.
* **The C library keeps freed memory** for itself instead of giving it back to the system, most of all with several
  threads (one pool each). :func:`trim` asks it to give it back, after each song and when the models go, and the
  image sets ``MALLOC_ARENA_MAX=2`` so there are two pools, not one a thread.

What can't be given back while the service runs: the GPU's own libraries, once a model has used the GPU (CUDA keeps
them for the life of the process). :func:`rss_mb` is in the log at each of these moments, so a log says what was
held when.
"""

import ctypes
import ctypes.util
import gc
import logging

_log = logging.getLogger("memory")

MODELS_IDLE = 600  # seconds without a song to analyse before the models are let go


def rss_mb():
    """The memory the service holds right now (resident, in MB), or None where the system doesn't say (not Linux)."""
    try:
        with open("/proc/self/status", encoding="ascii") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024.0)
    except (OSError, ValueError, IndexError):
        pass
    return None


_libc = None


def _malloc_trim():
    """glibc's ``malloc_trim(0)``: freed memory back to the system. False where there's no glibc (musl, macOS)."""
    global _libc
    if _libc is None:
        try:
            _libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
        except OSError:
            _libc = False
    if not _libc or not hasattr(_libc, "malloc_trim"):
        return False
    try:
        _libc.malloc_trim(0)
        return True
    except Exception:  # never worth a failed song
        return False


def trim():
    """Hands back what's no longer used: Python's own unreachable objects, then the C library's freed memory.
    Returns the memory held after (MB, or None). A few milliseconds."""
    gc.collect()
    _malloc_trim()
    return rss_mb()


def session_options(onnxruntime, threads=None, inter_threads=None):
    """ONNX Runtime's options for every model here: no arena and no memory pattern, so a run's working memory is
    given back when it ends instead of being kept at its largest for the next (see the module's words)."""
    options = onnxruntime.SessionOptions()
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    if threads:
        options.intra_op_num_threads = threads
    if inter_threads:
        options.inter_op_num_threads = inter_threads
    return options


def said(mb):
    """``mb`` for a sentence: "1.2 GB", "640 MB", or "an unknown amount"."""
    if mb is None:
        return "an unknown amount"
    return f"{mb / 1024.0:.1f} GB" if mb >= 1024 else f"{mb} MB"
