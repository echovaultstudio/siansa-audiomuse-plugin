"""The log: what Siansa Beats does, in the container's log (``docker logs``) and on the page.

Every line goes to standard output, so TrueNAS and ``docker logs siansa-beats`` show it, and to a ring of the
latest lines the page shows and downloads. ``LOG_LEVEL`` (``info`` by default, or ``debug``) sets how much at
start; the page's **Debug details** switch changes it while running, and it's kept across restarts.

At ``info``: starting up, the libraries read, each song done or failed, the plugin, errors with their traceback.
At ``debug`` as well: each song's steps with their times and sizes, what the model and the grid found, every request
to Navidrome and AudioMuse-AI (with the time it took) and every request to this service but the page's own polling.

Passwords, tokens and API keys never reach the log: :func:`redact` takes them out of every line.
"""

import collections
import logging
import re
import sys
import threading
import time

logger = logging.getLogger("siansa_beats")

LEVELS = {"debug": logging.DEBUG, "info": logging.INFO, "warn": logging.WARNING, "warning": logging.WARNING,
          "error": logging.ERROR}
KEPT = 5000  # lines kept for the page and the download

# Query parameters and headers that carry a secret: Subsonic's token, salt and password, API keys, bearer tokens.
_SECRET_PARAM = re.compile(r"(?i)([?&](?:t|s|p|password|token|api_?key|apikey|jwt|access_token)=)[^&\s\"']*")
_BEARER = re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]+")
_SECRET_KEY = re.compile(r"(?i)\b((?:password|token|api_?key|secret)\s*[=:]\s*)[^\s,;\"']+")


def redact(text):
    """``text`` without the secrets that addresses and headers carry."""
    text = _SECRET_PARAM.sub(r"\1***", str(text))
    text = _BEARER.sub(r"\1 ***", text)
    return _SECRET_KEY.sub(r"\1***", text)


class _Redacting(logging.Formatter):
    def format(self, record):
        return redact(super().format(record))


class Ring(logging.Handler):
    """The latest lines, for the page."""

    def __init__(self, size=KEPT):
        super().__init__(logging.DEBUG)
        self.lines = collections.deque(maxlen=size)
        self.lock_ = threading.Lock()
        self.count = 0

    def emit(self, record):
        try:
            message = redact(record.getMessage())
            if record.exc_info:
                message += "\n" + redact(logging.Formatter().formatException(record.exc_info))
            with self.lock_:
                self.count += 1
                self.lines.append({"n": self.count, "at": record.created, "level": _name(record.levelno),
                                   "where": record.name.rpartition(".")[2], "message": message})
        except Exception:  # a log line never breaks the service
            self.handleError(record)


def _name(levelno):
    return {logging.DEBUG: "debug", logging.INFO: "info", logging.WARNING: "warn"}.get(levelno, "error")


ring = Ring()
_set_up = False


def setup(level="info", stream=None):
    """Sends the log to ``stream`` (standard output) and the ring, at ``level``. Safe to call more than once."""
    global _set_up
    if not _set_up:
        handler = logging.StreamHandler(stream or sys.stdout)
        handler.setFormatter(_Redacting("%(asctime)s %(levelname)-5s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
        logger.addHandler(handler)
        logger.addHandler(ring)
        logger.propagate = False
        _set_up = True
    set_level(level)


def set_level(level):
    """``debug``, ``info``, ``warn`` or ``error`` (an unknown name is ``info``). Returns the name set."""
    name = str(level or "info").strip().lower()
    if name not in LEVELS:
        name = "info"
    logger.setLevel(LEVELS[name])
    return "warn" if name == "warning" else name


def level():
    return _name(logger.getEffectiveLevel())


def get(name):
    """The logger of one part (``worker``, ``navidrome``, …)."""
    return logger.getChild(name)


def lines(after=0, limit=500, least="debug"):
    """The latest ``limit`` lines numbered above ``after``, at ``least`` that level, oldest first."""
    floor = LEVELS.get(least, logging.DEBUG)
    with ring.lock_:
        kept = [l for l in ring.lines if l["n"] > after and LEVELS.get(l["level"], logging.ERROR) >= floor]
    return kept[-limit:] if limit else kept


def text():
    """Every kept line as plain text, for the download."""
    out = []
    for l in lines(limit=0):
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(l["at"]))
        out.append(f"{stamp} {l['level'].upper():5} {l['where']}: {l['message']}")
    return "\n".join(out) + ("\n" if out else "")


class timed:
    """``with log.timed(logger, "decoding"):`` logs at debug how long the block took (and its failure)."""

    def __init__(self, log, what):
        self.log, self.what = log, what

    def __enter__(self):
        self.started = time.monotonic()
        return self

    @property
    def seconds(self):
        return time.monotonic() - self.started

    def __exit__(self, kind, value, tb):
        if kind is None:
            self.log.debug("%s took %.2fs", self.what, self.seconds)
        else:
            self.log.debug("%s failed after %.2fs: %s", self.what, self.seconds, value)
        return False
