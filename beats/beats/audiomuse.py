"""Similar albums and artists, asked of AudioMuse-AI (the owner, 10 Nov: Beats answers the app for everything; at first by
asking AudioMuse, later from Beats' own fingerprints). AudioMuse-AI's Siansa plugin works them out from its sound
analysis; Siansa Beats passes the app's question on (``AUDIOMUSE_URL``, ``AUDIOMUSE_TOKEN``) and hands back the answer
as it is: ``{"similar": [{"album"|"artist", "artist", "track_id", "score"}, …]}``.

It asks the Siansa Beats (dev) plugin first, then the Siansa Analysis plugin (both answer the same way).

:meth:`AudioMuse.profiles` takes the loudness, fades and key the plugin already measured, so songs Siansa Beats analysed
before 0.20 needn't be downloaded again for them.
"""

import json
import time
import urllib.error
import urllib.parse
import urllib.request

from . import log

_log = log.get("audiomuse")
PLUGINS = ("/plugins/siansa_beats/", "/plugins/siansa_analysis/")
TIMEOUT = 20
KINDS = ("albums", "artists")
PASSED_ON = ("track_id", "artist", "server", "limit")  # the app's question, as the plugin takes it


class AudioMuseError(Exception):
    """AudioMuse-AI couldn't answer: why, in plain words."""


class AudioMuse:
    def __init__(self, url, token, opener=None):
        self.url = (url or "").rstrip("/")
        self.token = token or ""
        self.open = opener or urllib.request.urlopen
        self.plugin = None  # the plugin's prefix that answered last

    @property
    def configured(self):
        return bool(self.url)

    def _get(self, path):
        request = urllib.request.Request(self.url + path, headers={"Accept": "application/json"})
        if self.token:
            request.add_header("Authorization", "Bearer " + self.token)
        started = time.monotonic()
        try:
            with self.open(request, timeout=TIMEOUT) as response:
                answer = json.load(response)
        except urllib.error.HTTPError as e:
            _log.info("AudioMuse-AI %s -> HTTP %s (%.2fs)", path.split("?")[0], e.code, time.monotonic() - started)
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise AudioMuseError(f"Couldn't reach AudioMuse-AI: {getattr(e, 'reason', e)}") from None
        except ValueError:
            raise AudioMuseError("AudioMuse-AI's answer isn't JSON: is AUDIOMUSE_URL its API port?") from None
        _log.debug("AudioMuse-AI %s (%.2fs)", path.split("?")[0], time.monotonic() - started)
        return answer

    PROFILE_FIELDS = ("ref_lufs", "start_ramp", "end_ramp", "duration_ms", "key", "scale")

    def profiles(self, ids):
        """``{song_id: {"ref_lufs", "start_ramp", "end_ramp", "duration_ms", "key", "scale"}}`` of the songs among ``ids``
        the plugin has measured (loudness found); ``{}`` when there's no AudioMuse-AI or no plugin answers."""
        if not self.configured or not ids:
            return {}
        out = {}
        for prefix in PLUGINS:
            for start in range(0, len(ids), 100):  # AudioMuse-AI turns away long addresses
                chunk = [i for i in ids[start:start + 100] if i not in out]
                if not chunk:
                    continue
                try:
                    answer = self._get(f"{prefix}api/profiles?" + urllib.parse.urlencode({"ids": ",".join(chunk)}))
                except (urllib.error.HTTPError, AudioMuseError) as e:
                    _log.debug("No profiles from %s: %s", prefix, e)
                    break
                for p in (answer or {}).get("profiles") or []:
                    if p.get("track_id") in chunk and p.get("ref_lufs") is not None and p.get("start_ramp"):
                        out[p["track_id"]] = {k: p.get(k) for k in self.PROFILE_FIELDS}
        return out

    def similar(self, kind, query):
        """AudioMuse-AI's ``kind`` ("albums" or "artists") like ``query``'s ``track_id`` or ``artist``."""
        if kind not in KINDS:
            raise ValueError(kind)
        if not self.configured:
            raise AudioMuseError("AudioMuse-AI isn't set up for Siansa Beats: set AUDIOMUSE_URL (and AUDIOMUSE_TOKEN)")
        asked = urllib.parse.urlencode({k: query[k] for k in PASSED_ON if query.get(k)})
        tried = [self.plugin] + [p for p in PLUGINS if p != self.plugin] if self.plugin else list(PLUGINS)
        for prefix in tried:
            try:
                answer = self._get(f"{prefix}api/similar/{kind}?{asked}")
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    continue  # this plugin isn't installed (or doesn't know the song): the other one may
                if e.code in (401, 403):
                    raise AudioMuseError("AudioMuse-AI wants a login: set AUDIOMUSE_TOKEN to one of its API tokens") from None
                raise AudioMuseError(f"AudioMuse-AI answered HTTP {e.code}") from None
            if isinstance(answer, dict) and isinstance(answer.get("similar"), list):
                self.plugin = prefix
                return {"similar": answer["similar"], "source": "audiomuse"}
        raise AudioMuseError("AudioMuse-AI has neither Siansa plugin, or doesn't know the song")
