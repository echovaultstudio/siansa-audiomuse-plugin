"""Siansa Beats' API for the Siansa app, version 1 (``/api/v1/…``): what the app asked the Siansa Beats (dev) plugin for
until 0.20, answered by Siansa Beats itself (the owner, 10 Nov: Beats is its own product, the plugin stays for those who
want it). Its layout only grows: a field is never renamed or dropped within v1, so an older app keeps working. Anything
that would break that is v2, next to v1. The full description is ``openapi.json`` (``GET /api/v1/openapi.json``).

    GET  /api/v1/capabilities          what this Siansa Beats can do now, and how far it is (the app's DJ check)
    GET  /api/v1/profiles?ids=a,b      loudness, fades, key and the beat grid at both ends of up to 500 songs
    GET  /api/v1/genres                every style with songs in its genre by their sound, most first
    GET  /api/v1/genre?name=&limit=    the songs of a genre, by their sound (all without limit): {"name", "styles", "genres", "ids", "songs"}
    GET  /api/v1/genre/about?name=     a short description of a genre, from Wikipedia: {"name", "about"}
    POST /api/v1/listens               finished listens of a listener: {"user", "listens": [{"id", "at", "artist", …}]}
    POST /api/v1/listens/now           what's playing now: {"user", "artist", "title", "album", "duration"}
    GET  /api/v1/listens/top?user=&kind=&from=&to=&limit=   a listener's most listened songs, artists or albums
    GET  /api/v1/similar/albums?track_id=          albums that sound like that song's album (asked of AudioMuse-AI)
    GET  /api/v1/similar/artists?track_id=|artist= artists that sound like that one (asked of AudioMuse-AI)
    POST /api/v1/bad-blends            {"from_id", "to_id", "technique", "details"}: a song change marked as bad
    GET  /api/v1/openapi.json          this API described

Song ids are the music server's own (the app's, when it plays from that server). With several servers (0.23.0:
Navidrome, Emby, Jellyfin, several users of one), every request may say which it means with ``?server=``, its id or
name from the capabilities' ``servers``; without it the default server is meant, so an app from before keeps working.
Ids that are asked about are understood from any server; ids that are answered (a genre's songs) are the named
server's, and songs it doesn't have are left out.

Like the rest of the API, it asks for the password or the API key when one is set (``Authorization: Bearer <key>``,
or Basic with the password).
"""

import json
import os
from http import HTTPStatus

from . import listens as listens_module
from . import __version__
from . import loudness

API = 1
SCHEMA_VERSION = 1  # of the profiles' layout: the plugin's, which the app reads (PluginProfiles.SCHEMA_VERSION)
MAX_IDS = 500
SPEC = os.path.join(os.path.dirname(__file__), "openapi.json")


def capabilities(app, server=None):
    """What this Siansa Beats can answer and how far it is (for ``server``'s songs): the app decides from it what to
    offer (a DJ mix needs beat grids; the genres need the style model)."""
    server = server or app.servers.default_id
    only = app.only(server)
    counts = app.store.counts(only)
    styles = app.styles_status(server)
    measured = app.store.measure_counts(loudness.VERSION, only)
    shown = app.servers.get(server) or {}
    sound = app.store.sound_counts(only)
    return {
        "service": "siansa-beats",
        "version": __version__,
        "api": API,
        "apis": [API],
        "music_server": {"kind": shown.get("kind") or "navidrome", "set_up": bool(shown), "id": shown.get("id"),
                         "name": shown.get("name")},
        # Every music server here: the app names the one it plays from (?server=<id>) so the ids are that server's.
        "servers": [{k: s[k] for k in ("id", "kind", "name", "url", "user", "default", "songs")}
                    for s in app.server_list()],
        "songs": {"known": counts.get("songs", 0), "analysed": counts.get("done", 0), "failed": counts.get("failed", 0),
                  "waiting": sum(counts.get(s, 0) for s in ("queued", "downloading", "decoding", "ready", "analysing"))},
        "features": {
            "beat_grids": {"available": True, "songs": counts.get("done", 0), "model": app.config.model},
            "loudness": {"available": True, "songs": measured["measured"], "waiting": measured["waiting"],
                         "version": loudness.VERSION},
            "keys": {"available": True, "songs": measured["measured"]},
            "genres": {"available": bool(styles.get("available")), "styles": len(app.genre_list(server))},
            "genre_about": {"available": True, "source": "Wikipedia"},
            "bad_blends": {"available": True},
            # Similar albums and artists: from the songs' sound here once it's known (0.25), else asked of AudioMuse-AI.
            "similar": {"available": bool(sound["sound"]) or app.audiomuse.configured,
                        "source": "beats" if sound["sound"] else "audiomuse"},
            # What songs sound like, answered as AudioMuse-AI answers it, under /api/v1/audiomuse/… (0.25).
            "sound": {"available": bool(sound["sound"]), "songs": sound["sound"], "of": sound["songs"],
                      "answers": (list(SOUND_ANSWERS) if sound["sound"] else [])
                      + (list(WORD_ANSWERS) if sound["words"] and app.words.available else []),
                      "prefix": AUDIOMUSE},
            # The search by words (0.26): the songs that have the vector it needs, and whether the text model is here.
            "words": {"available": bool(sound["words"]) and app.words.available, "songs": sound["words"],
                      "steering": app.words.steers},
            "servers": {"available": True, "kinds": ["navidrome", "emby", "jellyfin"]},
            # The listening history (0.29): listens taken from the app, and sent on to Last.fm for who connected it.
            "scrobbling": {"available": True, "listens": app.listens.count(), "most_at_once": listens_module.MOST_AT_ONCE},
        },
        "limits": {"profile_ids": MAX_IDS, "genre_songs": None},
    }


def profiles(app, ids, server=None):
    """``{"schema_version", "profiles", "pending"}`` in the layout the app reads from the plugin: each song's loudness
    and fades (``ref_lufs``, ``start_ramp``, ``end_ramp``, ``duration_ms``), key and scale, and the beat grid at each
    end (``intro_*``, ``outro_*``, ``beats_until_ms``, ``mix_track``, ``drops_ms``, ``breakdowns_ms``) with
    ``beat_grid`` "siansa_beats". A song with only one of them has it alone; one with neither is ``pending``. The ids
    are the caller's (any server's): each profile's ``track_id`` is the id it was asked by."""
    here = app.store.resolve(ids, server)
    measures = app.store.measures(here.values())
    grids = app.store.ends(here.values(), app.config.model)
    out, pending = [], []
    for song_id in ids:
        sound, grid = measures.get(here.get(song_id)), grids.get(here.get(song_id))
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


def get(handler, path, query, server=None):
    """Answers a GET of ``/api/v1/…`` about ``server`` (the default without one); False when it isn't one of its
    paths."""
    app = handler.app

    def arg(name, default=""):
        return (query.get(name) or [default])[0].strip()

    if path == "/api/v1/capabilities":
        return handler._json(capabilities(app, server)) or True
    if path == "/api/v1/openapi.json":
        with open(SPEC, "rb") as f:
            return handler._send(HTTPStatus.OK, f.read(), "application/json") or True
    if path == "/api/v1/profiles":
        ids = [i for i in dict.fromkeys(p.strip() for p in arg("ids").split(",")) if i]
        if len(ids) > MAX_IDS:
            return handler._error(HTTPStatus.BAD_REQUEST, f"at most {MAX_IDS} ids at a time") or True
        return handler._json(profiles(app, ids, server)) or True
    if path.startswith(AUDIOMUSE + "/"):
        return sound_get(handler, path[len(AUDIOMUSE):], arg, server)
    if path in ("/api/v1/similar/albums", "/api/v1/similar/artists"):
        from .audiomuse import AudioMuseError, PASSED_ON

        asked = {k: arg(k) for k in PASSED_ON if arg(k)}
        if not asked.get("track_id") and not (path.endswith("artists") and asked.get("artist")):
            return handler._error(HTTPStatus.BAD_REQUEST, "track_id needed (or artist, for artists)") or True
        own = similar_here(app, path.rsplit("/", 1)[1], asked, server)
        if own is not None:
            return handler._json(own) or True
        try:
            return handler._json(app.audiomuse.similar(path.rsplit("/", 1)[1], asked)) or True
        except AudioMuseError as e:
            return handler._error(HTTPStatus.BAD_GATEWAY, str(e)) or True
    if path == "/api/v1/listens/top":  # the most listened songs, artists or albums of a listener in a stretch of time
        kind = arg("kind", "songs")
        if kind not in ("songs", "artists", "albums"):
            return handler._error(HTTPStatus.BAD_REQUEST, "kind is songs, artists or albums") or True
        try:
            start, end = int(arg("from", "0") or 0), int(arg("to", str(2 ** 62)) or 2 ** 62)
            limit = int(arg("limit", "50") or 50)
        except ValueError:
            return handler._error(HTTPStatus.BAD_REQUEST, "from, to (seconds since 1970) and limit are numbers") or True
        user = arg("user")
        return handler._json({"user": user, "kind": kind, "from": start, "to": end, "listens": app.listens.count(user),
                              "top": app.listens.top(user, kind, start, end, limit, server or app.servers.default_id)}) or True
    if path == "/api/v1/genres":
        return handler._json({"styles": app.styles_status(server), "genres": app.genre_list(server)}) or True
    if path in ("/api/v1/genre", "/api/v1/genre/about"):
        name = arg("name")
        if not name:
            return handler._error(HTTPStatus.BAD_REQUEST, "name needed: the genre, as a tag (Trance)") or True
        if path.endswith("/about"):
            return handler._json(app.genre_about(name)) or True
        try:
            limit = max(1, int(arg("limit"))) if arg("limit") else None  # none: every song of the genre
        except ValueError:
            return handler._error(HTTPStatus.BAD_REQUEST, "limit is a number") or True
        return handler._json(app.genre_songs(name, limit, server)) or True
    return False


def post(handler, path, body, server=None):
    """Answers a POST of ``/api/v1/…``; False when it isn't one of its paths."""
    if path.startswith(AUDIOMUSE + "/"):
        return sound_post(handler, path[len(AUDIOMUSE):], body, server)
    if path == "/api/v1/listens":  # finished listens, from the app ("Scrobble to Siansa Beats")
        rows = body.get("listens")
        if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
            return handler._error(HTTPStatus.BAD_REQUEST, 'listens needed: [{"id", "at", "artist", "title", "album", "duration"}]') or True
        if len(rows) > listens_module.MOST_AT_ONCE:
            return handler._error(HTTPStatus.BAD_REQUEST, f"at most {listens_module.MOST_AT_ONCE} listens at a time") or True
        answer = handler.app.listens.add(body.get("user"), rows, "siansa", server or handler.app.servers.default_id)
        return handler._json(answer) or True
    if path == "/api/v1/listens/now":  # what's playing now
        if not handler.app.listens.set_playing(body.get("user"), body):
            return handler._error(HTTPStatus.BAD_REQUEST, "artist and title needed") or True
        return handler._json({"playing": True}) or True
    if path == "/api/v1/bad-blends":
        answer, problem = add_bad_blend(handler.app, body, server)
        if problem:
            return handler._error(HTTPStatus.BAD_REQUEST, problem) or True
        return handler._json(answer) or True
    return False


# ---- what songs sound like, answered as AudioMuse-AI answers it (0.25) ----

AUDIOMUSE = "/api/v1/audiomuse"
# AudioMuse-AI's paths answered here, under AUDIOMUSE: the app asks them of Siansa Beats as it asks AudioMuse-AI.
SOUND_ANSWERS = ("/api/similar_tracks", "/api/mood_centroids", "/api/find_path", "/api/alchemy",
                 "/api/sonic_fingerprint/generate", "/api/sync")
# The search by words (0.26), answered when the text model is in the image and songs have the vector it needs.
WORD_ANSWERS = ("/api/clap/search", "/api/clap/concepts")


def _of_server(app, songs, server):
    """An answer's songs (``item_id``: this service's ids) with ``server``'s own ids, without songs it doesn't have."""
    server = server or app.servers.default_id
    ids = app.store.item_ids([s["item_id"] for s in songs], server)
    return [{**s, "item_id": ids[s["item_id"]]} for s in songs if s["item_id"] in ids]


def _here(app, song_id, server):
    return app.store.resolve([song_id], server).get(song_id)


def similar_here(app, kind, asked, server):
    """Similar albums or artists from the songs' sound here, in the plugin's layout; None when the sound of the song
    (or artist) asked about isn't known here, so AudioMuse-AI is asked instead."""
    from .sound import Unknown

    song_id = _here(app, asked["track_id"], server) if asked.get("track_id") else None
    try:
        if kind == "albums":
            name, artist, found = app.sound.similar_albums(song_id)
            extra = {"name": name, "artist": artist, "track_id": asked.get("track_id")}
        else:
            name, found = app.sound.similar_artists(song_id, asked.get("artist"))
            extra = {"name": name, "artist_asked": name}
    except Unknown:
        return None
    # Nothing found: AudioMuse-AI is asked when it can be, and always while no song's sound is here at all (then
    # the answer is why not, as before 0.25, not an empty list).
    if not found and (app.audiomuse.configured or not app.sound.count()):
        return None
    try:
        limit = max(1, min(50, int(asked.get("limit") or 20)))
    except ValueError:
        limit = 20
    ids = app.store.item_ids([c["track_id"] for c in found], server or app.servers.default_id)
    similar = [{**c, "track_id": ids[c["track_id"]]} for c in found if c["track_id"] in ids][:limit]
    return {"schema_version": SCHEMA_VERSION, "similar": similar, "source": "beats", **extra}


def features(app, ids, server):
    """The analysis of the songs ``ids`` as AudioMuse-AI's ``/api/sync`` gives it, without the vectors.

    ``tempo`` is always Beat This!'s (the owner, 11 Nov: AudioMuse-AI's is never used, and isn't kept), null for a
    song not analysed here yet. ``key`` and ``scale`` are this service's own measure where there is one, AudioMuse-AI's
    otherwise; moods, genres and energy are AudioMuse-AI's."""
    here = app.store.resolve(ids, server)
    rows = app.store.sound_rows(here.values())
    tempos = app.store.tempos(here.values(), app.config.model)
    measured = app.store.measures(here.values())
    tracks = []
    for asked in ids:
        r = rows.get(here.get(asked))
        if r is None:
            continue
        own = measured.get(r["song_id"]) or {}
        tempo = tempos.get(r["song_id"])
        tracks.append({"id": asked, "title": r["title"], "artist": r["artist"], "album_artist": r["album_artist"],
                       "album": r["album"], "year": r["year"], "tempo": round(tempo, 2) if tempo else None,
                       "tempo_source": "beat_this" if tempo else None,
                       "key": own.get("key") or r["key"], "scale": (own.get("scale") if own.get("key") else r["scale"]),
                       "mood_vector": r["moods"], "energy": r["energy"],
                       "other_features": r["other"], "fp": r["fp"]})
    return {"tracks": tracks, "total_tracks": len(tracks), "has_more": False, "next_page": None,
            "provider_type": (app.servers.get(server or app.servers.default_id) or {}).get("kind")}


def sound_get(handler, path, arg, server):
    """AudioMuse-AI's GET ``path`` answered from the songs' sound here."""
    from .sound import Unknown

    app, only = handler.app, handler.app.only(server)

    def number(name, default):
        try:
            return int(arg(name) or default)
        except ValueError:
            return default

    try:
        if path == "/api/mood_centroids":
            return handler._json(app.sound.centre_list()) or True
        if path == "/api/clap/concepts":
            return handler._json(app.words.catalogue()) or True
        if path == "/api/similar_tracks":
            n = number("n", 50)
            if arg("mood") and arg("centroid_index"):
                return handler._json(_of_server(app, app.sound.mood(arg("mood"), number("centroid_index", 0), n, only),
                                                server)) or True
            song_id = _here(app, arg("item_id"), server)
            if not song_id:
                return handler._error(HTTPStatus.NOT_FOUND, "Siansa Beats doesn't know this song") or True
            same_mood = arg("mood_similarity").lower() == "true"
            return handler._json(_of_server(app, app.sound.similar(song_id, n, same_mood, only), server)) or True
        if path == "/api/find_path":
            ends = []
            for song, mood in (("start_song_id", "start_mood"), ("end_song_id", "end_mood")):
                if arg(song):
                    song_id = _here(app, arg(song), server)
                    if not song_id:
                        return handler._error(HTTPStatus.NOT_FOUND, "Siansa Beats doesn't know this song") or True
                    ends.append(song_id)
                elif arg(mood):
                    ends.append(app.sound.centre(arg(mood)))
                else:
                    return handler._error(HTTPStatus.BAD_REQUEST, f"{song} or {mood} needed") or True
            found = _of_server(app, app.sound.path(ends[0], ends[1], number("max_steps", 25), only), server)
            return handler._json({"path": found, "total_distance": None}) or True
        if path == "/api/sync":
            ids = [i for i in dict.fromkeys(p.strip() for p in arg("ids").split(",")) if i]
            if not ids or len(ids) > 500:
                return handler._error(HTTPStatus.BAD_REQUEST, "ids needed: 1 to 500 songs") or True
            return handler._json(features(app, ids, server)) or True
    except Unknown as e:
        return handler._error(HTTPStatus.NOT_FOUND, str(e)) or True
    return False


def sound_post(handler, path, body, server):
    """AudioMuse-AI's POST ``path`` answered from the songs' sound here."""
    from .sound import Unknown
    from .subsonic import ServerError

    app, only = handler.app, handler.app.only(server)
    try:
        n = int(body.get("n") or 50)
    except (TypeError, ValueError):
        n = 50
    try:
        if path == "/api/clap/search":
            from .words import NotThere

            query = body.get("query")
            if not isinstance(query, str) or not query.strip():
                return handler._error(HTTPStatus.BAD_REQUEST, 'Missing "query" in request body') or True
            try:
                limit = max(1, int(body.get("limit") or 50))
            except (TypeError, ValueError):
                return handler._error(HTTPStatus.BAD_REQUEST, '"limit" must be a whole number') or True
            if not app.sound.word_count():
                return handler._error(HTTPStatus.SERVICE_UNAVAILABLE,
                                      "No song has the vector the search by words needs yet: take AudioMuse-AI's "
                                      "analysis over (Setup › AudioMuse-AI)") or True
            steering, warnings = app.words.terms(body.get("steering"))
            if body.get("steering") and not steering and warnings:
                return handler._error(HTTPStatus.BAD_REQUEST, warnings[0]) or True
            try:
                vector = app.words.steer(app.words.embed(query.strip()), steering)
            except NotThere as e:
                return handler._error(HTTPStatus.SERVICE_UNAVAILABLE, str(e)) or True
            found = _of_server(app, app.sound.described(vector, limit, only), server)
            answer = {"query": query.strip(), "results": found, "count": len(found)}
            if steering:
                answer["steering"] = steering
            if warnings:
                answer["steering_warnings"] = warnings
            return handler._json(answer) or True
        if path == "/api/alchemy":
            items = [i for i in body.get("items") or [] if isinstance(i, dict) and i.get("id")
                     and (i.get("type") or "song") == "song"]
            asked = [str(i["id"]) for i in items]
            here = app.store.resolve(asked, server)
            add = [here[str(i["id"])] for i in items if str(i.get("op") or "ADD").upper() == "ADD" and str(i["id"]) in here]
            avoid = [here[str(i["id"])] for i in items if str(i.get("op") or "").upper() == "SUBTRACT" and str(i["id"]) in here]
            if not add:
                return handler._error(HTTPStatus.BAD_REQUEST, "items needed: songs to blend ({id, op: ADD})") or True
            return handler._json({"results": _of_server(app, app.sound.blend(add, avoid, n, only), server)}) or True
        if path == "/api/sonic_fingerprint/generate":
            # The most played songs: the ones the caller names, else the ones this server's user here has played most.
            asked = [str(i) for i in body.get("ids") or []]
            if not asked:
                name = server or app.servers.default_id
                try:
                    asked = app.servers.client(name).most_played(20)
                except ServerError as e:
                    return handler._error(HTTPStatus.BAD_GATEWAY, f"The most played songs couldn't be read: {e}") or True
            here = app.store.resolve(asked, server)
            played = [here[i] for i in asked if i in here]
            if not played:
                return handler._json([]) or True
            return handler._json(_of_server(app, app.sound.yours(played, n, only), server)) or True
    except Unknown as e:
        return handler._error(HTTPStatus.NOT_FOUND, str(e)) or True
    return False


def add_bad_blend(app, body, server=None):
    """Keeps a song change marked as bad in the app: ``({"id"}, None)``, or ``(None, why)`` when it isn't one. The two
    songs are kept by their ids here, whichever server the app plays them from."""
    from_id, to_id = str(body.get("from_id") or "").strip(), str(body.get("to_id") or "").strip()
    if not from_id or not to_id:
        return None, "from_id and to_id needed: the two songs"
    here = app.store.resolve([from_id, to_id], server)
    from_id, to_id = here.get(from_id, from_id), here.get(to_id, to_id)
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

