"""The Siansa plugin's current beat readings, from AudioMuse-AI, to set beside Beat This!'s.

Optional: without AUDIOMUSE_URL the comparison just leaves those columns empty. Logs in with an AudioMuse-AI API
token (AUDIOMUSE_TOKEN) when it has a login; nothing is changed there.
"""

import json
import urllib.error
import urllib.parse
import urllib.request

FIELDS = ("intro_bpm", "outro_bpm", "intro_steady", "outro_steady", "intro_bar_ms", "outro_bar_ms")
BATCH = 200  # the plugin takes up to 500 ids a call


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
            raise AudioMuseError(f"HTTP {e.code}{hint}") from None
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise AudioMuseError(str(getattr(e, "reason", e))) from None

    def health(self):
        return self._get("/plugins/siansa_analysis/api/health")

    def profiles(self, ids):
        """``{song id: {intro_bpm, …}}`` for the songs the plugin has measured; songs it hasn't are left out."""
        found = {}
        ids = [i for i in ids if i]
        for start in range(0, len(ids), BATCH):
            chunk = ids[start:start + BATCH]
            answer = self._get("/plugins/siansa_analysis/api/profiles?" + urllib.parse.urlencode({"ids": ",".join(chunk)}))
            for p in answer.get("profiles") or []:
                found[p.get("track_id")] = {k: p.get(k) for k in FIELDS}
        return found
