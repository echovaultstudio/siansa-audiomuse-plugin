"""A genre's short description, with a little of its history: the opening of its Wikipedia article (the owner, 10 Nov),
for the app's genre page.

Looked up once and kept (``about:<genre>`` in the settings table); a genre Wikipedia has nothing on is asked again after
:data:`AGAIN_MISSING` seconds. The text is Wikipedia's (CC BY-SA): the answer names it and links the article, and the
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
    """Wikipedia's summary of ``title``, or None."""
    url = SUMMARY.format(urllib.parse.quote(title.replace(" ", "_"), safe=""))
    request = urllib.request.Request(url, headers={
        "User-Agent": f"SiansaBeats/{__version__} (https://siansa.app; genre descriptions)",
        "Accept": "application/json",
    })
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code != 404:
            _log.debug("Wikipedia %s: HTTP %s", title, e.code)
        return None
    except Exception as e:  # no internet from here, a slow answer: no description, the page shows without it
        _log.debug("Wikipedia %s: %s", title, e)
        return None


def lookup(name, get=fetch):
    """``{"summary", "url", "source", "title"}`` for genre ``name``, or None when Wikipedia has nothing on it."""
    for title in candidates(name):
        page = get(title)
        if usable(page):
            url = ((page.get("content_urls") or {}).get("desktop") or {}).get("page")
            return {"summary": trim(page.get("extract")), "url": url, "source": "Wikipedia", "title": page.get("title")}
    return None


def about(store, name, get=fetch, now=time.time):
    """Genre ``name``'s description, kept in ``store``: looked up when not kept yet, or kept too long."""
    key = "about:" + name.strip().lower()
    kept = store.setting(key)
    if kept and now() - kept.get("at", 0) < (AGAIN_FOUND if kept.get("summary") else AGAIN_MISSING):
        return kept if kept.get("summary") else None
    found = lookup(name, get)
    record = dict(found or {}, at=now())
    store.set_setting(key, record)
    if found:
        _log.info("About %s: %s (%s)", name, found["title"], found["source"])
    else:
        _log.info("About %s: nothing found", name)
    return record if found else None
