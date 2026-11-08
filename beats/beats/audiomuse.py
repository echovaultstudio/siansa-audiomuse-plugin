"""The Siansa plugin's current beat readings, from AudioMuse-AI, to set beside Beat This!'s.

Optional: without AUDIOMUSE_URL the comparison just leaves those columns empty. Logs in with an AudioMuse-AI API
token (AUDIOMUSE_TOKEN) when it has a login; nothing is changed there.
"""

import json
import urllib.error
import urllib.parse
import urllib.request

FIELDS = ("intro_bpm", "outro_bpm", "intro_steady", "outro_steady", "intro_bar_ms", "outro_bar_ms")
MAX_IDS = 500  # the plugin takes up to 500 ids a call…
URL_BUDGET = 3500  # …but AudioMuse-AI's web server turns away request lines over about 4 KB (HTTP 400), as the app knows
PROFILES = "/plugins/siansa_analysis/api/profiles"


def batches(ids, budget=URL_BUDGET, max_ids=MAX_IDS):
    """``ids`` split so each request's address stays within ``budget`` bytes and ``max_ids`` ids."""
    out, current, start = [], [], len(PROFILES) + len("?ids=")
    length = start
    for i in dict.fromkeys(i for i in ids if i):
        size = len(urllib.parse.quote(i, safe=""))
        add = size + (1 if current else 0)
        if current and (length + add > budget or len(current) >= max_ids):
            out.append(current)
            current, length, add = [], start, size
        current.append(i)
        length += add
    if current:
        out.append(current)
    return out


class AudioMuseError(Exception):
    pass


class AudioMuse:
    def __init__(self, url, token="", timeout=30.0):
        self.url = (url or "").rstrip("/")
        self.token = token or ""
        self.timeout = timeout

    @property
    def configured(self):
        return bool(self.url)

    def _get(self, path):
        request = urllib.request.Request(self.url + path)
        if self.token:
            request.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as e:
            hint = " (check AUDIOMUSE_TOKEN)" if e.code in (401, 403) else ""
            try:
                said = json.load(e).get("error") or ""
            except Exception:
                said = ""
            raise AudioMuseError(f"HTTP {e.code}{hint}" + (f": {said}" if said else "")) from None
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise AudioMuseError(str(getattr(e, "reason", e))) from None

    def health(self):
        return self._get("/plugins/siansa_analysis/api/health")

    def profiles(self, ids):
        """``{song id: {intro_bpm, …}}`` for the songs the plugin has measured; songs it hasn't are left out."""
        found = {}
        for chunk in batches(ids):
            answer = self._get(PROFILES + "?ids=" + ",".join(urllib.parse.quote(i, safe="") for i in chunk))
            for p in answer.get("profiles") or []:
                found[p.get("track_id")] = {k: p.get(k) for k in FIELDS}
        return found
