"""A genre's short description, with a little of its history: the opening of its Wikipedia article (the owner, 10 Nov),
for the app's genre page.

Looked up once and kept (``about:<genre>`` in the settings table): a timed cache. A description is looked up again
after :data:`AGAIN_FOUND`, a genre Wikipedia has nothing on after :data:`AGAIN_MISSING`, and one Wikipedia couldn't be
asked about (no internet, a slow answer) after :data:`AGAIN_FAILED`: a failure isn't taken for "nothing on it".

:class:`Keeper` fetches every genre's description in the background (the owner, 10 Nov: some genre pages waited ages
for theirs), so the app's genre page has it at once; :meth:`Keeper.answer` never holds an answer up for long. The text is Wikipedia's (CC BY-SA): the answer names it and links the article, and the
app shows both. Only the genre's name leaves the service.
"""

import json
import re
import time
import urllib.parse
import urllib.request

from . import __version__
from . import log

_log = log.get("about")

SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/{}?redirect=true"
AGAIN_MISSING = 7 * 24 * 3600
AGAIN_FOUND = 90 * 24 * 3600
AGAIN_FAILED = 30 * 60
RECORD = 2  # records before it kept a timeout as "nothing on it": those are looked up again
FAILED = object()  # fetch's answer when Wikipedia couldn't be asked (not "no such article")
MOST_CHARS = 700  # the page shows a few lines: the opening, cut at a sentence
MUSIC = re.compile(r"\b(genre|music|musical|subgenre|style|sound|dance|song|bands?|rock|jazz|electronic)\b", re.I)


def candidates(name):
    """The article titles to try for genre ``name``, most likely first: "House" is "House music", "Electro" is
    "Electro (music)". A name with a family ("Hip Hop / Electro") is looked up by its style."""
    style = name.split("/")[-1].strip() or name.strip()
    titled = style[:1].upper() + style[1:]
    seen, out = set(), []
    for t in (f"{titled} (music)", f"{titled} (genre)", f"{titled} (music genre)", f"{titled} music", titled):
        if t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out


def trim(text, most=MOST_CHARS):
    """``text`` cut at the last sentence that ends within ``most`` characters (or at a word, with "…")."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= most:
        return text
    cut = text[:most]
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    if end > most // 3:
        return cut[:end + 1]
    return cut.rsplit(" ", 1)[0] + "…"


def usable(page):
    """Whether a Wikipedia summary is about music: a real article (not a list of meanings) that says so."""
    if not page or page.get("type") != "standard":
        return False
    words = f"{page.get('description') or ''} {(page.get('extract') or '')[:300]}"
    return bool(MUSIC.search(words))


def fetch(title, timeout=6.0):
    """Wikipedia's summary of ``title``, None when there's no such article, or :data:`FAILED` when it couldn't be asked."""
    url = SUMMARY.format(urllib.parse.quote(title.replace(" ", "_"), safe=""))
    request = urllib.request.Request(url, headers={
        "User-Agent": f"SiansaBeats/{__version__} (https://siansa.app; genre descriptions)",
        "Accept": "application/json",
    })
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        _log.debug("Wikipedia %s: HTTP %s", title, e.code)
        return FAILED
    except Exception as e:  # no internet from here, a slow answer: asked again later
        _log.debug("Wikipedia %s: %s", title, e)
        return FAILED


def lookup(name, get=fetch):
    """``{"summary", "url", "source", "title"}`` for genre ``name``; None when Wikipedia has nothing on it; :data:`FAILED`
    when it couldn't be asked about one of the titles and none of the others was it."""
    failed = False
    for title in candidates(name):
        page = get(title)
        if page is FAILED:
            failed = True
            continue
        if usable(page):
            url = ((page.get("content_urls") or {}).get("desktop") or {}).get("page")
            return {"summary": trim(page.get("extract")), "url": url, "source": "Wikipedia", "title": page.get("title")}
    return FAILED if failed else None


def key_of(name):
    return "about:" + name.strip().lower()


def fresh(kept, now):
    """Whether a kept record still stands, or it's time to look the genre up again."""
    if not kept:
        return False
    if not kept.get("summary") and kept.get("v") != RECORD:
        return False  # "nothing on it" from before 0.22 may have been Wikipedia not answering
    again = AGAIN_FOUND if kept.get("summary") else AGAIN_FAILED if kept.get("failed") else AGAIN_MISSING
    return now - kept.get("at", 0) < again


def about(store, name, get=fetch, now=time.time):
    """Genre ``name``'s description, kept in ``store``: looked up when not kept yet, or kept too long."""
    kept = store.setting(key_of(name))
    if fresh(kept, now()):
        return kept if kept.get("summary") else None
    found = lookup(name, get)
    if found is FAILED:
        store.set_setting(key_of(name), {"failed": True, "at": now(), "v": RECORD})
        _log.info("About %s: Wikipedia couldn't be asked; again in %d min", name, AGAIN_FAILED // 60)
        return None
    record = dict(found or {}, at=now(), v=RECORD)
    store.set_setting(key_of(name), record)
    if found:
        _log.info("About %s: %s (%s)", name, found["title"], found["source"])
    else:
        _log.info("About %s: nothing found", name)
    return record if found else None


class Keeper:
    """Every genre's description, fetched in the background a genre at a time and kept: the sound's genres and the songs'
    tags, the most songs first. ``names()`` gives them; :meth:`fetch_all` starts a round now (the page's button), and
    one runs by itself every :data:`ROUND_SECONDS`."""

    ROUND_SECONDS = 6 * 3600
    PAUSE = 1.0  # between genres: Wikipedia is asked politely
    WAIT = 3.0  # the most an answer waits for a genre not kept yet; then the app is told to ask again shortly

    def __init__(self, store, names, get=fetch, now=time.time):
        import threading

        self.store, self.names, self.get, self.now = store, names, get, now
        self.lock = threading.Lock()
        self.busy = {}  # name (lower case) -> threading.Event, while it's being looked up
        self.round = None  # {"done", "total", "current", "started"} while a round runs
        self.last_round = None  # {"at", "looked_up", "seconds"}
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        import threading

        if not self._thread:
            self._thread = threading.Thread(target=self._loop, name="beats-about", daemon=True)
            self._thread.start()

    def stop(self):
        self._stop.set()
        self._wake.set()

    def fetch_all(self):
        """Starts a round now (or says one is running)."""
        self._wake.set()
        return {"started": self.round is None, "round": self.round}

    def _look_up(self, name):
        """Looks ``name`` up once, however many ask at the same time: the description, or None."""
        import threading

        k = name.strip().lower()
        with self.lock:
            event = self.busy.get(k)
            mine = event is None
            if mine:
                event = self.busy[k] = threading.Event()
        if not mine:
            event.wait(30)
            kept = self.store.setting(key_of(name))
            return kept if kept and kept.get("summary") else None
        try:
            return about(self.store, name, self.get, self.now)
        finally:
            with self.lock:
                self.busy.pop(k, None)
            event.set()

    def answer(self, name):
        """For the app: ``(description or None, pending)``. A kept one at once; else it's looked up, waiting at most
        :data:`WAIT` seconds: when it's not there by then, ``pending`` is True and the lookup carries on, so asking
        again shortly finds it."""
        import threading

        kept = self.store.setting(key_of(name))
        if fresh(kept, self.now()):
            return (kept if kept.get("summary") else None), False
        box = {}
        worker = threading.Thread(target=lambda: box.setdefault("v", self._look_up(name)), name="beats-about-ask", daemon=True)
        worker.start()
        worker.join(self.WAIT)
        if worker.is_alive():
            return None, True
        return box.get("v"), False

    def counts(self):
        """``{"genres", "found", "missing", "failed", "waiting"}`` over the genres there are now."""
        out = {"genres": 0, "found": 0, "missing": 0, "failed": 0, "waiting": 0}
        now = self.now()
        for name in self.names():
            out["genres"] += 1
            kept = self.store.setting(key_of(name))
            if not fresh(kept, now):
                out["waiting"] += 1
            elif kept.get("summary"):
                out["found"] += 1
            elif kept.get("failed"):
                out["failed"] += 1
            else:
                out["missing"] += 1
        return out

    def status(self):
        return {**self.counts(), "round": dict(self.round) if self.round else None, "last_round": self.last_round}

    def run_round(self):
        """Looks up every genre not kept (or kept too long), one at a time: how many were looked up."""
        names = [n for n in self.names() if not fresh(self.store.setting(key_of(n)), self.now())]
        started = time.monotonic()
        self.round = {"done": 0, "total": len(names), "current": None, "started": self.now()}
        looked = 0
        try:
            for name in names:
                if self._stop.is_set():
                    break
                self.round["current"] = name
                self._look_up(name)
                looked += 1
                self.round["done"] = looked
                self._stop.wait(self.PAUSE)
        finally:
            self.round = None
            self.last_round = {"at": self.now(), "looked_up": looked, "seconds": round(time.monotonic() - started, 1)}
        if looked:
            _log.info("Genre descriptions: looked up %d in %.0f s", looked, time.monotonic() - started)
        return looked

    def _loop(self):
        self._stop.wait(30)  # after the start, when the genres are known
        while not self._stop.is_set():
            try:
                self.run_round()
            except Exception:
                _log.exception("Fetching the genre descriptions went wrong; trying again later")
            self._wake.wait(self.ROUND_SECONDS)
            self._wake.clear()
