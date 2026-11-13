"""The checks Start analysis runs first (0.40, the owner, 13 Nov): the server, the libraries, the database, the disk,
the models and the page's own safety, each said in a line, before a single song is read.

A check is ``ok``, ``warn`` (analysis goes ahead; the line says what to look at) or ``fail`` (it doesn't start: the
line says why). The models are fingerprinted: every file in the models folder is hashed the first time a version is
checked and compared each time after, so a damaged or swapped model is caught before it analyses anything.
"""
import glob
import hashlib
import ipaddress
import logging
import os
import shutil
import time
import urllib.parse

from . import __version__
from .subsonic import ServerError

_log = logging.getLogger("beats.checks")

SPACE_FAIL_MB = 500  # a song's file and the database's growth need this much free
SPACE_WARN_MB = 2000
STALE_HOURS = 1  # a song's temporary file older than this is from a run that was cut off
STALE_PREFIXES = ("song-", "style-", "sound-", "loudness-")
PRIVATE_HOSTS = ("localhost", "host.docker.internal")


def _mb(n):
    return f"{n / 1e6:,.0f} MB" if n < 10e9 else f"{n / 1e9:,.1f} GB"


def is_private(url):
    """Whether ``url`` stays in the home: an IP that isn't public, localhost, a bare name or a ``.local``/``.lan``/
    ``.home``/``.internal`` one. A plain-http address to anywhere else sends the password across the internet."""
    host = (urllib.parse.urlsplit(url).hostname or "").lower()
    if not host or host in PRIVATE_HOSTS or "." not in host:
        return True
    try:
        return not ipaddress.ip_address(host).is_global
    except ValueError:
        return host.rsplit(".", 1)[-1] in ("local", "lan", "home", "internal", "localdomain")


def fingerprints(models_dir):
    """``{file: {"size", "sha256"}}`` of every file in the models folder."""
    found = {}
    for name in sorted(os.listdir(models_dir)) if os.path.isdir(models_dir) else []:
        path = os.path.join(models_dir, name)
        if not os.path.isfile(path):
            continue
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                digest.update(chunk)
        found[name] = {"size": os.path.getsize(path), "sha256": digest.hexdigest()}
    return found


class Checks:
    """Runs the checks for one app: ``run(server, ids)`` gives ``{"ok", "checks": [{"id", "name", "state", "detail"}],
    "seconds"}``."""

    def __init__(self, app):
        self.app = app

    def run(self, server, ids):
        started = time.monotonic()
        lines = []
        for check in (self._server, self._libraries, self._database, self._space, self._models, self._access):
            try:
                lines.append(check(server, ids))
            except Exception as e:  # a check that itself breaks is a failed check, said as one
                _log.exception("The check %s went wrong", check.__name__)
                lines.append(self._line(check.__name__.strip("_"), check.__name__.strip("_").title(), "fail",
                                        f"The check itself went wrong: {e}"))
        report = {"ok": all(c["state"] != "fail" for c in lines), "checks": lines,
                  "seconds": round(time.monotonic() - started, 1), "at": time.time(), "version": __version__}
        self.app.store.set_setting("last_checks", report)
        return report

    @staticmethod
    def _line(check_id, name, state, detail):
        return {"id": check_id, "name": name, "state": state, "detail": detail}

    # ---- the checks ----

    def _server(self, server, ids):
        client = self.app.servers.client(server)
        found = self.app.servers.get(server) or {}
        name = found.get("name", server)
        try:
            answer = client.ping()
        except ServerError as e:
            return self._line("server", "Music server", "fail", f"{name} doesn't answer: {e}")
        version = answer.get("version") if isinstance(answer, dict) else None
        detail = f"{name} answers" + (f" ({answer.get('type') or found.get('kind') or ''} {version})".replace("( ", "(") if version else "")
        url = getattr(client, "url", "") or ""
        if url.startswith("http://") and not is_private(url):
            return self._line("server", "Music server", "warn",
                              detail + f". It's reached over plain http at a public address ({urllib.parse.urlsplit(url).hostname}): "
                              "the login goes across the internet unencrypted. Use https there.")
        return self._line("server", "Music server", "ok", detail + ". Nothing is ever written to it.")

    def _libraries(self, server, ids):
        client = self.app.servers.client(server)
        try:
            known = {f["id"]: f["name"] for f in client.libraries()}
        except ServerError as e:
            return self._line("libraries", "Libraries", "fail", f"The libraries can't be listed: {e}")
        ids = [str(i) for i in dict.fromkeys(ids)]
        unknown = [i for i in ids if i not in known]
        if unknown:
            return self._line("libraries", "Libraries", "fail",
                              f"No library {', '.join(unknown)} on the server any more (it has: "
                              f"{', '.join(known.values()) or 'none'}). Untick it.")
        if not ids:
            return self._line("libraries", "Libraries", "warn", "None is ticked: nothing new is analysed. Songs already done stay.")
        counts = self.app.store.folder_counts(server)
        here = sum(counts.get(i, {}).get("songs", 0) for i in ids)
        return self._line("libraries", "Libraries", "ok",
                          f"{', '.join(known[i] for i in ids)}: {here:,} songs known so far; new ones are looked for now.")

    def _database(self, server, ids):
        found = self.app.store.integrity()
        if found["quick_check"] != "ok":
            return self._line("database", "Database", "fail",
                              f"SQLite finds it damaged: {found['quick_check']}. Stop the container and restore beats.db "
                              "from a backup before analysing more.")
        tidied = ", ".join(f"{n} {table}" for table, n in found["orphans"].items() if n)
        size = _mb(found["size"])
        if tidied:
            return self._line("database", "Database", "ok",
                              f"Sound ({size}). Leftovers of removed songs tidied away: {tidied}.")
        return self._line("database", "Database", "ok", f"Sound ({size}); every row belongs to a song.")

    def _space(self, server, ids):
        data_dir = self.app.config.data_dir
        stale, freed = 0, 0
        cutoff = time.time() - STALE_HOURS * 3600
        for prefix in STALE_PREFIXES:
            for path in glob.glob(os.path.join(data_dir, prefix + "*")):
                try:
                    if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                        freed += os.path.getsize(path)
                        os.remove(path)
                        stale += 1
                except OSError:
                    pass
        free = shutil.disk_usage(data_dir).free
        note = f" {stale} temporary files of a run that was cut off removed ({_mb(freed)})." if stale else ""
        if free < SPACE_FAIL_MB * 1e6:
            return self._line("space", "Disk", "fail",
                              f"Only {_mb(free)} free where the data lives: a song's file and the database need room. Make space first.{note}")
        if free < SPACE_WARN_MB * 1e6:
            return self._line("space", "Disk", "warn", f"{_mb(free)} free where the data lives: getting tight.{note}")
        return self._line("space", "Disk", "ok", f"{_mb(free)} free where the data lives.{note}")

    def _models(self, server, ids):
        app = self.app
        models_dir = app.config.models_dir
        missing = []
        tracker = app.tracker
        checkpoint = tracker.checkpoint() if hasattr(tracker, "checkpoint") else None
        downloads = checkpoint is not None and not os.path.exists(checkpoint)
        spec = getattr(app.style, "spec", None)
        if spec is not None and not spec.available(models_dir):
            missing.append(f"the style model {spec.name}")
        if hasattr(app.hearer, "available") and not app.hearer.available:
            missing.append("the sound models (MusiCNN, CLAP)")
        if hasattr(app.words, "available") and not app.words.available:
            missing.append("the text model")
        if hasattr(app.arousal, "available") and not app.arousal.available:
            missing.append("the energy model")
        # Fingerprints: hashed once a version, compared each time after.
        key = f"model_fingerprints:{__version__}"
        now = fingerprints(models_dir)
        kept = app.store.setting(key)
        changed = []
        if kept is None:
            app.store.set_setting(key, now)
            printed = f"{len(now)} model files fingerprinted (SHA-256), to be checked against from now on"
        else:
            changed = [name for name, fp in kept.items() if now.get(name) != fp]
            gone = [name for name in kept if name not in now]
            printed = f"{len(now)} model files unchanged since first checked" if not changed else ""
            if gone:
                missing.extend(f"the file {g}" for g in gone)
        if changed:
            return self._line("models", "Models", "fail",
                              f"Changed since this version was first checked: {', '.join(changed)}. A model file that "
                              "isn't the image's gives wrong answers for every song: pull the image again.")
        if missing:
            return self._line("models", "Models", "warn",
                              f"Not in this image: {', '.join(missing)}. Beats are found all the same; what those models "
                              f"give is left out. {printed}.".replace(". .", "."))
        detail = printed + ("; Beat This! downloads its weights on the first song" if downloads else "") + "."
        return self._line("models", "Models", "ok", detail)

    def _access(self, server, ids):
        app = self.app
        access = app.access()
        notes, state = [], "ok"
        if not access.get("password"):
            state = "warn"
            notes.append("The page has no password: anyone who can reach this port can see and change everything. Set one in Setup")
        else:
            notes.append("The page asks for a password" + (", and the app for its key" if access.get("api_key") else ""))
        if getattr(app.config, "secrets_left", None):
            state = "warn"
            notes.append(f"the container still holds secrets it no longer reads ({', '.join(app.config.secrets_left)}): take them out of the YAML")
        if getattr(app.vault, "from_env", False) is False:
            notes.append("the secret key lives in the data folder")
        return self._line("access", "Safety", state, "; ".join(notes) + ".")
