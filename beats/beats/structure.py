"""Siansa Structure, when set up: All-In-One's sections of a song (intro, verse, chorus, break…), to set against the
phrases Siansa Beats works out from the sound and against the owner's section marks.

It runs in its own container (../structure); Siansa Beats posts it each song's audio file as it analyses it.
"""

import json
import socket
import urllib.error
import urllib.parse
import urllib.request


class StructureError(Exception):
    pass


class Structure:
    def __init__(self, url, token="", timeout=900.0):
        self.url = (url or "").rstrip("/")
        self.token = token or ""
        self.timeout = timeout

    @property
    def configured(self):
        return bool(self.url)

    def _request(self, path, data=None, headers=None):
        parts = urllib.parse.urlsplit(self.url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise StructureError("STRUCTURE_URL needs to start with http:// or https://, like http://siansa-structure:8091")
        request = urllib.request.Request(self.url + path, data=data, headers=dict(headers or {}),
                                         method="POST" if data is not None else "GET")
        if self.token:
            request.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as e:
            try:
                said = json.load(e).get("error") or ""
            except Exception:  # noqa: BLE001
                said = ""
            hint = "set STRUCTURE_TOKEN to Siansa Structure's" if e.code in (401, 403) else "Siansa Structure's log says more"
            raise StructureError(f"HTTP {e.code}: {said or hint}") from None
        except urllib.error.URLError as e:
            if isinstance(e.reason, ConnectionRefusedError):
                raise StructureError(f"Nothing answers at {parts.hostname}:{parts.port or ''}: is Siansa Structure "
                                     "running?") from None
            if isinstance(e.reason, socket.gaierror):
                raise StructureError(f"The name {parts.hostname} isn't known here: use the server's IP address, or "
                                     "the service's name in the same compose file") from None
            raise StructureError(f"Couldn't reach Siansa Structure: {e.reason}") from None
        except (socket.timeout, TimeoutError):
            raise StructureError(f"No answer from Siansa Structure within {self.timeout:.0f} s") from None
        except ValueError:
            raise StructureError("The answer isn't Siansa Structure's: check STRUCTURE_URL") from None

    def health(self):
        return self._request("/health")

    def analyse(self, path, suffix=""):
        """``{"bpm", "beats", "downbeats", "segments": [{"start", "end", "label"}], "seconds", "device"}`` for the
        audio file at ``path`` (of kind ``suffix``, like ``flac``)."""
        with open(path, "rb") as f:
            data = f.read()
        return self._request("/analyse", data, {"Content-Type": "application/octet-stream",
                                                "X-Filename": f"song.{(suffix or 'flac').lstrip('.')}"})


def section_starts(structure):
    """Where All-In-One's sections start (seconds), leaving out its "start" and "end" (the silence around a song)."""
    if not structure or not isinstance(structure, dict):
        return []
    return [s["start"] for s in structure.get("segments") or [] if s.get("label") not in ("start", "end")]
