"""Similar albums and artists, asked of AudioMuse-AI (the owner, 10 Nov: Beats answers the app for everything; at first by
asking AudioMuse, later from Beats' own fingerprints). AudioMuse-AI's Siansa plugin works them out from its sound
analysis; Siansa Beats passes the app's question on (its address and token, set on the page) and hands back the answer
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
            raise AudioMuseError("AudioMuse-AI's answer isn't JSON: is the address its API port?") from None
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

    def check(self):
        """Asks AudioMuse-AI's Siansa plugin something harmless: the plugin that answered ("siansa_beats" or
        "siansa_analysis"). :class:`AudioMuseError` says why not: unreachable, a token wanted or refused, no plugin."""
        if not self.configured:
            raise AudioMuseError("An address is needed")
        for prefix in PLUGINS:
            try:
                answer = self._get(f"{prefix}api/profiles?ids=siansa-beats-check")
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    continue
                if e.code in (401, 403):
                    raise AudioMuseError("AudioMuse-AI didn't take the token" if self.token
                                         else "AudioMuse-AI wants a token: one of its API tokens") from None
                raise AudioMuseError(f"AudioMuse-AI answered HTTP {e.code}") from None
            if isinstance(answer, dict) and "profiles" in answer:
                self.plugin = prefix
                return prefix.strip("/").split("/")[-1]
        raise AudioMuseError("That address answers, but neither Siansa plugin is installed in AudioMuse-AI there")

    # ---- its export for apps: every song's analysis and vectors (0.25) ----

    def _sync(self, query):
        try:
            return self._get("/api/sync?" + urllib.parse.urlencode(query))
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise AudioMuseError("AudioMuse-AI didn't take the token" if self.token
                                     else "AudioMuse-AI wants a token: one of its API tokens") from None
            if e.code == 404:
                raise AudioMuseError("This AudioMuse-AI has no export for apps (/api/sync): it needs a newer version") from None
            raise AudioMuseError(f"AudioMuse-AI answered HTTP {e.code}") from None

    def sync_check(self):
        """How many songs AudioMuse-AI's export has; :class:`AudioMuseError` says why it can't be read."""
        if not self.configured:
            raise AudioMuseError("An address is needed")
        answer = self._sync({"fields": "index", "limit": 1})
        if not isinstance(answer, dict) or "tracks" not in answer:
            raise AudioMuseError("That address answers, but not as AudioMuse-AI's export does")
        return int(answer.get("total_tracks") or 0)

    def sync_index(self):
        """Every song AudioMuse-AI has analysed, a page at a time: lists of ``{"id" (the music server's), "fp" (what
        its analysis looks like: it changes when the analysis does)}``."""
        page = 1
        while page:
            answer = self._sync({"fields": "index", "limit": 1000, "page": page}) or {}
            yield answer.get("tracks") or [], int(answer.get("total_tracks") or 0)
            page = answer.get("next_page") if answer.get("has_more") else None

    def sync_songs(self, ids):
        """The whole analysis of the songs ``ids`` (at most 100: AudioMuse-AI turns away long addresses), vectors
        included (base64 of float32)."""
        return (self._sync({"ids": ",".join(ids), "limit": len(ids)}) or {}).get("tracks") or []

    def similar(self, kind, query):
        """AudioMuse-AI's ``kind`` ("albums" or "artists") like ``query``'s ``track_id`` or ``artist``."""
        if kind not in KINDS:
            raise ValueError(kind)
        if not self.configured:
            raise AudioMuseError("AudioMuse-AI isn't set up for Siansa Beats: add it on its page (Setup › AudioMuse-AI)")
        asked = urllib.parse.urlencode({k: query[k] for k in PASSED_ON if query.get(k)})
        tried = [self.plugin] + [p for p in PLUGINS if p != self.plugin] if self.plugin else list(PLUGINS)
        for prefix in tried:
            try:
                answer = self._get(f"{prefix}api/similar/{kind}?{asked}")
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    continue  # this plugin isn't installed (or doesn't know the song): the other one may
                if e.code in (401, 403):
                    raise AudioMuseError("AudioMuse-AI wants a login: give Siansa Beats one of its API tokens (Setup › AudioMuse-AI)") from None
                raise AudioMuseError(f"AudioMuse-AI answered HTTP {e.code}") from None
            if isinstance(answer, dict) and isinstance(answer.get("similar"), list):
                self.plugin = prefix
                return {"similar": answer["similar"], "source": "audiomuse"}
        raise AudioMuseError("AudioMuse-AI has neither Siansa plugin, or doesn't know the song")
