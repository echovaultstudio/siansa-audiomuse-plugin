"""The Siansa plugin's current beat readings, from AudioMuse-AI, to set beside Beat This!'s.

Optional: without AUDIOMUSE_URL the comparison just leaves those columns empty. Logs in with an AudioMuse-AI API
token (AUDIOMUSE_TOKEN) when it has a login; nothing is changed there.
"""

import json
import socket
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
        """The JSON answer at ``path``, or an :class:`AudioMuseError` that says what to check."""
        parts = urllib.parse.urlsplit(self.url)
        host = parts.hostname or self.url
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise AudioMuseError("AUDIOMUSE_URL needs to start with http:// or https://, like http://192.168.1.10:8000")
        here = " (inside the container that's the container itself: use the server's address)" \
            if host in ("localhost", "127.0.0.1", "::1") else ""
        request = urllib.request.Request(self.url + path, headers={"Accept": "application/json"})
        if self.token:
            request.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read()
                final = response.geturl()
        except urllib.error.HTTPError as e:
            try:
                said = json.load(e).get("error") or ""
            except Exception:
                said = ""
            if e.code in (401, 403):
                hint = ("AudioMuse-AI asks for a login: set AUDIOMUSE_TOKEN to an API token from AudioMuse-AI"
                        if not self.token else "AudioMuse-AI doesn't take this AUDIOMUSE_TOKEN: make a new API token")
            elif e.code == 404:
                hint = ("No Siansa plugin there: check AUDIOMUSE_URL is AudioMuse-AI's own address (no /plugins/… at "
                        "the end) and that the plugin is installed and turned on")
            elif e.code == 400 and "too large" in said.lower():
                hint = "AudioMuse-AI turned the request away as too long"
            elif e.code >= 500:
                hint = "AudioMuse-AI had an error: its log says more"
            else:
                hint = "AudioMuse-AI turned the request away"
            raise AudioMuseError(f"HTTP {e.code}: {hint}" + (f" ({said})" if said else "")) from None
        except urllib.error.URLError as e:
            reason = e.reason
            if isinstance(reason, ConnectionRefusedError):
                raise AudioMuseError(f"Nothing answers at {host}:{parts.port or ''}: check the port, and that "
                                     f"AudioMuse-AI is running{here}") from None
            if isinstance(reason, socket.gaierror):
                raise AudioMuseError(f"The name {host} isn't known inside the container: use the server's IP "
                                     "address instead") from None
            if isinstance(reason, (socket.timeout, TimeoutError)):
                raise AudioMuseError(f"No answer from {host} within {self.timeout:.0f} s: check the address, and "
                                     "that nothing blocks the port") from None
            raise AudioMuseError(f"Couldn't reach {host}: {reason}{here}") from None
        except (socket.timeout, TimeoutError):
            raise AudioMuseError(f"No answer from {host} within {self.timeout:.0f} s") from None
        except OSError as e:
            raise AudioMuseError(f"Couldn't reach {host}: {e}{here}") from None
        try:
            return json.loads(body)
        except ValueError:
            login = "login" in final.lower() or b"<form" in body[:20000].lower()
            raise AudioMuseError("AudioMuse-AI answered with its login page: set AUDIOMUSE_TOKEN to an API token "
                                 "from AudioMuse-AI" if login else
                                 "The answer isn't the plugin's: check AUDIOMUSE_URL is AudioMuse-AI's address") from None

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
