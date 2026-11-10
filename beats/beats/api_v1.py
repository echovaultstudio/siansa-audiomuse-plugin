"""Siansa Beats' API for the Siansa app, version 1 (``/api/v1/…``): what the app asked the Siansa Beats (dev) plugin for
until 0.20, answered by Siansa Beats itself (the owner, 10 Nov: Beats is its own product, the plugin stays for those who
want it). Its layout only grows: a field is never renamed or dropped within v1, so an older app keeps working. Anything
that would break that is v2, next to v1. The full description is ``openapi.json`` (``GET /api/v1/openapi.json``).

    GET  /api/v1/capabilities          what this Siansa Beats can do now, and how far it is (the app's DJ check)
    GET  /api/v1/profiles?ids=a,b      loudness, fades, key and the beat grid at both ends of up to 500 songs
    GET  /api/v1/genres                every style with songs voted into it, most first
    GET  /api/v1/genre?name=&limit=    the songs of a genre (by vote): {"name", "styles", "genres", "ids"}
    GET  /api/v1/genre/about?name=     a short description of a genre, from Wikipedia: {"name", "about"}
    GET  /api/v1/similar/albums?track_id=          albums that sound like that song's album (asked of AudioMuse-AI)
    GET  /api/v1/similar/artists?track_id=|artist= artists that sound like that one (asked of AudioMuse-AI)
    POST /api/v1/bad-blends            {"from_id", "to_id", "technique", "details"}: a song change marked as bad
    GET  /api/v1/openapi.json          this API described

Song ids are Navidrome's (the app's, when it plays from that Navidrome). Like the rest of the API, it asks for the
password or the API key when one is set (``Authorization: Bearer <key>``, or Basic with the password).
"""

import json
import os
from http import HTTPStatus

from . import __version__
from . import loudness

API = 1
SCHEMA_VERSION = 1  # of the profiles' layout: the plugin's, which the app reads (PluginProfiles.SCHEMA_VERSION)
MAX_IDS = 500
SPEC = os.path.join(os.path.dirname(__file__), "openapi.json")


def capabilities(app):
    """What this Siansa Beats can answer and how far it is: the app decides from it what to offer (a DJ mix needs beat
    grids; the genres need the style model)."""
    counts = app.store.counts()
    styles = app.styles_status()
    measured = app.store.measure_counts(loudness.VERSION)
    return {
        "service": "siansa-beats",
        "version": __version__,
        "api": API,
        "apis": [API],
        "music_server": {"kind": "navidrome", "set_up": bool(app.config.navidrome_url)},
        "songs": {"known": counts.get("songs", 0), "analysed": counts.get("done", 0), "failed": counts.get("failed", 0),
                  "waiting": sum(counts.get(s, 0) for s in ("queued", "downloading", "decoding", "ready", "analysing"))},
        "features": {
            "beat_grids": {"available": True, "songs": counts.get("done", 0), "model": app.config.model},
            "loudness": {"available": True, "songs": measured["measured"], "waiting": measured["waiting"],
                         "version": loudness.VERSION},
            "keys": {"available": True, "songs": measured["measured"]},
            "genres": {"available": bool(styles.get("available")), "styles": len(app.genre_list())},
            "genre_about": {"available": True, "source": "Wikipedia"},
            "bad_blends": {"available": True},
            "similar": {"available": app.audiomuse.configured, "source": "audiomuse"},
        },
        "limits": {"profile_ids": MAX_IDS, "genre_songs": 2000},
    }


def profiles(app, ids):
    """``{"schema_version", "profiles", "pending"}`` in the layout the app reads from the plugin: each song's loudness
    and fades (``ref_lufs``, ``start_ramp``, ``end_ramp``, ``duration_ms``), key and scale, and the beat grid at each
    end (``intro_*``, ``outro_*``, ``beats_until_ms``, ``mix_track``, ``drops_ms``, ``breakdowns_ms``) with
    ``beat_grid`` "siansa_beats". A song with only one of them has it alone; one with neither is ``pending``."""
    measures = app.store.measures(ids)
    grids = app.store.ends(ids, app.config.model)
    out, pending = [], []
    for song_id in ids:
        sound, grid = measures.get(song_id), grids.get(song_id)
        if sound is None and grid is None:
            pending.append(song_id)
            continue
        profile = {"track_id": song_id}
        if sound is not None:
            profile.update({k: sound[k] for k in ("ref_lufs", "start_ramp", "end_ramp", "duration_ms")})
            profile["start_ramp"] = profile["start_ramp"] or ""
            profile["end_ramp"] = profile["end_ramp"] or ""
            profile["duration_ms"] = profile["duration_ms"] or 0
            if sound.get("key"):
                profile.update(key=sound["key"], scale=sound.get("scale"))
        else:
            profile.update(ref_lufs=None, start_ramp="", end_ramp="", duration_ms=0)
        if grid is not None:
            profile.update({k: v for k, v in grid.items() if v is not None})
            profile["beat_grid"] = "siansa_beats"
        out.append(profile)
    return {"schema_version": SCHEMA_VERSION, "profiles": out, "pending": pending}


def get(handler, path, query):
    """Answers a GET of ``/api/v1/…``; False when it isn't one of its paths."""
    app = handler.app

    def arg(name, default=""):
        return (query.get(name) or [default])[0].strip()

    if path == "/api/v1/capabilities":
        return handler._json(capabilities(app)) or True
    if path == "/api/v1/openapi.json":
        with open(SPEC, "rb") as f:
            return handler._send(HTTPStatus.OK, f.read(), "application/json") or True
    if path == "/api/v1/profiles":
        ids = [i for i in dict.fromkeys(p.strip() for p in arg("ids").split(",")) if i]
        if len(ids) > MAX_IDS:
            return handler._error(HTTPStatus.BAD_REQUEST, f"at most {MAX_IDS} ids at a time") or True
        return handler._json(profiles(app, ids)) or True
    if path in ("/api/v1/similar/albums", "/api/v1/similar/artists"):
        from .audiomuse import AudioMuseError, PASSED_ON

        asked = {k: arg(k) for k in PASSED_ON if arg(k)}
        if not asked.get("track_id") and not (path.endswith("artists") and asked.get("artist")):
            return handler._error(HTTPStatus.BAD_REQUEST, "track_id needed (or artist, for artists)") or True
        try:
            return handler._json(app.audiomuse.similar(path.rsplit("/", 1)[1], asked)) or True
        except AudioMuseError as e:
            return handler._error(HTTPStatus.BAD_GATEWAY, str(e)) or True
    if path == "/api/v1/genres":
        return handler._json({"styles": app.styles_status(), "genres": app.genre_list()}) or True
    if path in ("/api/v1/genre", "/api/v1/genre/about"):
        name = arg("name")
        if not name:
            return handler._error(HTTPStatus.BAD_REQUEST, "name needed: the genre, as a tag (Trance)") or True
        if path.endswith("/about"):
            return handler._json(app.genre_about(name)) or True
        try:
            limit = max(1, min(2000, int(arg("limit", "300") or 300)))
        except ValueError:
            return handler._error(HTTPStatus.BAD_REQUEST, "limit is a number") or True
        return handler._json(app.genre_songs(name, limit)) or True
    return False


def post(handler, path, body):
    """Answers a POST of ``/api/v1/…``; False when it isn't one of its paths."""
    if path == "/api/v1/bad-blends":
        answer, problem = add_bad_blend(handler.app, body)
        if problem:
            return handler._error(HTTPStatus.BAD_REQUEST, problem) or True
        return handler._json(answer) or True
    return False


def add_bad_blend(app, body):
    """Keeps a song change marked as bad in the app: ``({"id"}, None)``, or ``(None, why)`` when it isn't one."""
    from_id, to_id = str(body.get("from_id") or "").strip(), str(body.get("to_id") or "").strip()
    if not from_id or not to_id:
        return None, "from_id and to_id needed: the two songs"
    details = body.get("details") if isinstance(body.get("details"), dict) else {}
    technique = str(body.get("technique") or "")[:80]
    blend_id = app.store.add_bad_blend(from_id, to_id, technique, details)
    names = {s: (app.store.song(s) or {}).get("title", s) for s in (from_id, to_id)}
    app.store.event(f"Bad blend marked in the app: {names[from_id]} into {names[to_id]}"
                    + (f" ({technique})" if technique else ""), "warn")
    return {"id": blend_id}, None


def spec():
    """The OpenAPI description, as a dict (for the tests)."""
    with open(SPEC) as f:
        return json.load(f)

