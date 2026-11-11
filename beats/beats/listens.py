"""What was listened to, and when (0.29, the owner, 11 Nov: Siansa Beats keeps the listening history itself, so no
other history service is needed beside it).

- **Listens** come from the Siansa app as they happen ("Scrobble to Siansa Beats"), and from files: an export of
  Koito, of Last.fm (the JSON or CSV of the usual export tools) or of ListenBrainz (:func:`read_export`). The same
  listen twice (the same listener, second, artist and title) is kept once, so a file can be brought in again and two
  services' exports of the same years don't count double.
- Each listen is one **listener**'s (the music server's user name the app scrobbles as; chosen for a file): two
  people on one server have a history each.
- A listen is linked to the song here when there is one (by the app's id for it, or by artist and title for a file),
  so the app can be given ids to play.
- **The numbers** for a month, a year or all time (:meth:`Listens.summary`): totals, the most played songs, artists
  and albums, when in the day and week, the days, records. Days, months and hours are the listener's clock's (the
  browser says which time zone), not the container's.
- **On to Last.fm** (:class:`LastFm`, :meth:`Listens.relay`): a listener's new listens are sent on to their Last.fm
  account, with their own API account, in the background and again later when Last.fm can't be reached. Listens from
  a file never are.

Plain Python and the database of :mod:`beats.store`.
"""

import csv
import datetime
import hashlib
import json
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter

from . import log

_log = log.get("listens")

SCHEMA = """
CREATE TABLE IF NOT EXISTS listens (
    id INTEGER PRIMARY KEY, user TEXT NOT NULL, at INTEGER NOT NULL, artist TEXT NOT NULL, title TEXT NOT NULL,
    album TEXT, duration INTEGER, song_id TEXT, source TEXT, client TEXT, akey TEXT NOT NULL, tkey TEXT NOT NULL,
    relayed INTEGER,  -- to Last.fm: NULL not to be sent, 0 waiting, 1 sent, 2 turned away, 3 too old to send
    UNIQUE (user, at, akey, tkey)
);
CREATE INDEX IF NOT EXISTS listens_by_artist ON listens (user, akey, at);
CREATE INDEX IF NOT EXISTS listens_waiting ON listens (relayed, user) WHERE relayed = 0;
"""

SOURCES = {"siansa": "the Siansa app", "koito": "Koito", "lastfm": "Last.fm", "listenbrainz": "ListenBrainz"}
MOST_AT_ONCE = 500  # listens in one request from the app
WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December")
FIRST_YEAR = 2002  # Last.fm started counting then: a listen from before is a broken date
TYPICAL_SECONDS = 210  # a listen whose song's length nothing says


def key(name):
    """A name as it's compared: any case, no spaces or punctuation ("AC/DC" and "ac dc" are one)."""
    return re.sub(r"[\W_]+", "", (name or "").casefold())


def zone(name):
    """The time zone ``name`` (IANA, "Europe/Helsinki"), or UTC when there's no such one here."""
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(name) if name else datetime.timezone.utc
    except Exception:  # no such zone, or no zone data in this Python
        return datetime.timezone.utc


# ---- files ----

class CantRead(Exception):
    """A file that isn't an export this can read: says why, for the page."""


def _when(value):
    """Seconds since 1970 from what an export gives: a number (seconds, or milliseconds), or a written date (read as
    UTC when it names no zone). None when it's neither."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)) or re.fullmatch(r"\d{9,13}", str(value).strip()):
        number = float(value)
        return int(number / 1000 if number > 1e11 else number)
    text = str(value).strip()
    try:
        parsed = datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        parsed = None
        for pattern in ("%d %b %Y %H:%M", "%d %b %Y, %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%d/%m/%Y %H:%M"):
            try:
                parsed = datetime.datetime.strptime(text, pattern)
                break
            except ValueError:
                continue
        if parsed is None:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.timezone.utc)
    return int(parsed.timestamp())


def _text(value):
    """What an export says a name is: Last.fm's own shape is ``{"#text": …}``."""
    if isinstance(value, dict):
        value = value.get("#text") or value.get("name") or ""
    return str(value or "").strip()


def _primary(aliases):
    """Koito's name of a thing: its primary alias, else its first."""
    aliases = [a for a in aliases or [] if isinstance(a, dict) and a.get("alias")]
    for a in aliases:
        if a.get("is_primary"):
            return str(a["alias"]).strip()
    return str(aliases[0]["alias"]).strip() if aliases else ""


def _koito(data):
    for item in data.get("listens") or []:
        track, artists = item.get("track") or {}, item.get("artists") or []
        names = [_primary(a.get("aliases")) for a in sorted(artists, key=lambda a: not a.get("is_primary"))]
        yield {"at": _when(item.get("listened_at")), "artist": next((n for n in names if n), ""),
               "title": _primary(track.get("aliases")), "album": _primary((item.get("album") or {}).get("aliases")),
               "duration": track.get("duration") or None, "client": item.get("client")}


def _lastfm_tracks(tracks):
    for t in tracks or []:
        if not isinstance(t, dict):
            continue
        date = t.get("date") if isinstance(t.get("date"), dict) else {}
        if not date and (t.get("@attr") or {}).get("nowplaying"):
            continue  # what was playing while the export ran: not a listen yet
        yield {"at": _when(date.get("uts") or t.get("uts") or t.get("date")), "artist": _text(t.get("artist")),
               "title": _text(t.get("name") or t.get("track")), "album": _text(t.get("album"))}


def _listenbrainz(items):
    for item in items:
        meta = item.get("track_metadata") or {}
        extra = meta.get("additional_info") or {}
        ms = extra.get("duration_ms") or (extra.get("duration") or 0) * 1000
        yield {"at": _when(item.get("listened_at")), "artist": _text(meta.get("artist_name")),
               "title": _text(meta.get("track_name")), "album": _text(meta.get("release_name")),
               "duration": int(ms / 1000) if ms else None, "client": extra.get("media_player")}


def _json_rows(data):
    """``(format, rows)`` of an export read as JSON."""
    if isinstance(data, dict):
        if isinstance(data.get("listens"), list):
            return "koito", _koito(data)
        if isinstance(data.get("scrobbles"), list):  # lastfmstats.com
            return "lastfm", _lastfm_tracks(data["scrobbles"])
        if isinstance(data.get("track"), list):  # one page of Last.fm's own answer
            return "lastfm", _lastfm_tracks(data["track"])
        if isinstance((data.get("recenttracks") or {}).get("track"), list):
            return "lastfm", _lastfm_tracks(data["recenttracks"]["track"])
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict) and "track_metadata" in first:
            return "listenbrainz", _listenbrainz(data)
        if isinstance(first, dict) and isinstance(first.get("track"), list):  # the export tool's pages
            return "lastfm", (row for page in data for row in _lastfm_tracks(page.get("track")))
        if isinstance(first, list):
            return "lastfm", (row for page in data for row in _lastfm_tracks(page))
        if isinstance(first, dict) and ("artist" in first or "name" in first):
            return "lastfm", _lastfm_tracks(data)
    raise CantRead("This JSON isn't an export of Koito, Last.fm or ListenBrainz that Siansa Beats knows")


COLUMNS = {"artist": ("artist", "artist_name", "artistname"), "title": ("track", "title", "name", "track_name", "song"),
           "album": ("album", "album_name", "release_name"), "at": ("uts", "date", "utc_time", "timestamp", "time", "listened_at")}


def _csv_rows(text):
    """An export as a table: with a first line naming the columns, or the four of the oldest export tool (artist,
    album, title, date)."""
    first = text.split("\n", 1)[0]
    delimiter = max(",;\t", key=first.count)
    lines = csv.reader(text.splitlines(), delimiter=delimiter)
    head = next(lines, None)
    if head is None:
        raise CantRead("The file is empty")
    names = [re.sub(r"[^a-z_]", "", h.strip().lower().replace(" ", "_")) for h in head]
    where = {field: next((names.index(n) for n in options if n in names), None) for field, options in COLUMNS.items()}
    if where["artist"] is None or where["title"] is None or where["at"] is None:
        if len(head) >= 4 and _when(head[3]) is not None:  # no first line: artist, album, title, date
            where, lines = {"artist": 0, "album": 1, "title": 2, "at": 3}, _chain([head], lines)
        else:
            raise CantRead("This table has no artist, track and date columns: is it an export of Last.fm?")

    def rows():
        for line in lines:
            if len(line) <= max(v for v in where.values() if v is not None):
                continue
            yield {"at": _when(line[where["at"]]), "artist": line[where["artist"]].strip(),
                   "title": line[where["title"]].strip(),
                   "album": line[where["album"]].strip() if where.get("album") is not None else ""}
    return "lastfm", rows()


def _chain(first, rest):
    yield from first
    yield from rest


def read_export(path):
    """``(format, rows)`` of the export file at ``path``: ``format`` is "koito", "lastfm" or "listenbrainz", ``rows``
    gives ``{"at", "artist", "title", "album", "duration", "client"}`` (a listen without a date, artist or title is
    left out by :meth:`Listens.add`). :class:`CantRead` when it's none of them."""
    with open(path, "rb") as f:
        start = f.read(64).lstrip(b"\xef\xbb\xbf \t\r\n")
    if start[:2] == b"PK":
        raise CantRead("That's a zip: unpack it and choose the file inside")
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        if start[:1] in (b"{", b"["):
            try:
                return _json_rows(json.load(f))
            except ValueError:
                f.seek(0)
                try:  # ListenBrainz's own export: a listen a line
                    return "listenbrainz", _listenbrainz([json.loads(line) for line in f if line.strip()])
                except ValueError:
                    raise CantRead("The file starts like JSON but isn't") from None
        return _csv_rows(f.read())


# ---- Last.fm ----

class LastFmError(Exception):
    """Last.fm said no, or couldn't be reached: ``code`` is its error number (None when it didn't answer)."""

    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code


class LastFm:
    """Last.fm's scrobbling API with the listener's own API account (key and shared secret), as the Siansa app's."""

    URL = "https://ws.audioscrobbler.com/2.0/"
    MOST = 50  # scrobbles in one request
    OLDEST = 14 * 86400  # Last.fm turns away a listen older than two weeks
    LATER = (8, 11, 16, 29)  # its errors that mean "try again later"

    def __init__(self, opener=None):
        self.open = opener or urllib.request.urlopen

    def call(self, params, secret):
        """One signed request; the answer as JSON. Never logs the parameters (a password or a session is in them)."""
        signature = hashlib.md5(("".join(k + str(params[k]) for k in sorted(params)) + secret).encode()).hexdigest()
        body = urllib.parse.urlencode({**params, "api_sig": signature, "format": "json"}).encode()
        request = urllib.request.Request(self.URL, data=body, headers={"User-Agent": "SiansaBeats"})
        try:
            with self.open(request, timeout=20) as response:
                raw = response.read()
        except urllib.error.HTTPError as e:
            raw = e.read()
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise LastFmError(f"Last.fm couldn't be reached: {getattr(e, 'reason', e)}") from None
        try:
            answer = json.loads(raw)
        except ValueError:
            raise LastFmError("Last.fm's answer isn't JSON") from None
        if isinstance(answer, dict) and answer.get("error"):
            raise LastFmError(f"Last.fm: {answer.get('message') or answer['error']}", int(answer["error"]))
        return answer

    def session(self, api_key, secret, name, password):
        """Signs in once and gives the session key to keep; the password isn't kept."""
        answer = self.call({"method": "auth.getMobileSession", "username": name, "password": password,
                            "api_key": api_key}, secret)
        found = (answer.get("session") or {}).get("key")
        if not found:
            raise LastFmError("Last.fm sent no session")
        return found

    def scrobble(self, keys, rows):
        """Sends up to :attr:`MOST` listens (``{"at", "artist", "title", "album", "duration"}``)."""
        params = {"method": "track.scrobble", "api_key": keys["api_key"], "sk": keys["session"]}
        for i, r in enumerate(rows[:self.MOST]):
            params[f"artist[{i}]"], params[f"track[{i}]"], params[f"timestamp[{i}]"] = r["artist"], r["title"], str(r["at"])
            if r.get("album"):
                params[f"album[{i}]"] = r["album"]
            if r.get("duration"):
                params[f"duration[{i}]"] = str(r["duration"])
        return self.call(params, keys["secret"])

    def love(self, keys, artist, title, loved=True):
        """Loves a song on Last.fm (or takes the love back)."""
        return self.call({"method": "track.love" if loved else "track.unlove", "api_key": keys["api_key"],
                          "sk": keys["session"], "artist": artist, "track": title}, keys["secret"])

    def now_playing(self, keys, row):
        params = {"method": "track.updateNowPlaying", "api_key": keys["api_key"], "sk": keys["session"],
                  "artist": row["artist"], "track": row["title"]}
        if row.get("album"):
            params["album"] = row["album"]
        if row.get("duration"):
            params["duration"] = str(row["duration"])
        return self.call(params, keys["secret"])


# ---- the history ----

class Listens:
    def __init__(self, store, vault=None, lastfm=None, now=time.time):
        self.store, self.vault, self.lastfm, self.now = store, vault, lastfm or LastFm(), now
        with store._write():
            store.db.executescript(SCHEMA)
        self.changes = 0  # counts every change, so kept numbers know when they're old
        self.playing = {}  # listener -> {"artist", "title", "album", "at"}: what the app says is playing
        self.importing = {"running": False}
        self.relay_state = {}  # listener -> {"error", "at"}: the last trouble sending on to Last.fm
        self._kept, self._lock = {}, threading.Lock()
        self._relay_lock = threading.Lock()

    # ---- keeping ----

    def add(self, user, rows, source="siansa", server=None):
        """Keeps ``rows`` (``{"at", "artist", "title", "album", "duration", "id" (a server's id for the song),
        "client"}``) as ``user``'s listens: ``{"kept", "known", "skipped"}``; ``known``: already here, ``skipped``:
        without a date, artist or title."""
        user = str(user or "").strip()
        rows = list(rows)
        latest, year = self.now() + 86400, datetime.datetime(FIRST_YEAR, 1, 1, tzinfo=datetime.timezone.utc).timestamp()
        by_id = self.store.resolve([str(r["id"]) for r in rows if r.get("id")], server) if source == "siansa" else {}
        songs = None
        relay = 0 if source == "siansa" and self.relays(user) else None
        values, skipped = [], 0
        for r in rows:
            at, artist, title = r.get("at"), str(r.get("artist") or "").strip(), str(r.get("title") or "").strip()
            if not isinstance(at, (int, float)) or not year <= at <= latest or not artist or not title:
                skipped += 1
                continue
            akey, tkey = key(artist) or artist.casefold(), key(title) or title.casefold()
            song_id = by_id.get(str(r.get("id"))) if r.get("id") else None
            if song_id is None:
                if songs is None:
                    songs = self._songs()
                song_id = songs.get((akey, tkey))
            try:
                duration = int(float(r.get("duration") or 0)) or None
            except (TypeError, ValueError):
                duration = None
            values.append((user, int(at), artist, title, str(r.get("album") or "").strip() or None, duration, song_id,
                           source, str(r.get("client") or "")[:80] or None, akey, tkey, relay))
        before = self.count(user)
        with self.store._write():
            self.store.db.executemany(
                "INSERT OR IGNORE INTO listens (user, at, artist, title, album, duration, song_id, source, client, "
                "akey, tkey, relayed) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", values)
        kept = self.count(user) - before
        if kept:
            self.changes += 1
        return {"kept": kept, "known": len(values) - kept, "skipped": skipped}

    def _songs(self):
        """``{(artist, title) as compared: song id}`` of the songs here, to link a listen that came without an id."""
        return {(key(r["artist"]), key(r["title"])): r["id"]
                for r in self.store._rows("SELECT id, artist, title FROM songs WHERE title IS NOT NULL")}

    def count(self, user=None):
        if user is None:
            return self.store._rows("SELECT COUNT(*) AS n FROM listens")[0]["n"]
        return self.store._rows("SELECT COUNT(*) AS n FROM listens WHERE user=?", (user,))[0]["n"]

    def users(self):
        """The listeners, most listens first: ``[{"user", "listens", "first", "last", "sources", "lastfm"}]``."""
        rows = self.store._rows("SELECT user, COUNT(*) AS listens, MIN(at) AS first, MAX(at) AS last FROM listens "
                                "GROUP BY user ORDER BY listens DESC")
        sources = {}
        for r in self.store._rows("SELECT user, source, COUNT(*) AS n FROM listens GROUP BY user, source"):
            sources.setdefault(r["user"], {})[r["source"] or "siansa"] = r["n"]
        relays = self._relays()
        # Also who's playing now or sends on to Last.fm without a listen kept yet: the app's listener shows at once.
        names = [r["user"] for r in rows]
        names += [u for u in list(self.playing) + list(relays) if u not in names]
        by = {r["user"]: dict(r) for r in rows}
        return [{"user": u, "listens": 0, "first": None, "last": None, **by.get(u, {}), "sources": sources.get(u, {}),
                 "lastfm": self.lastfm_status(u)} for u in names]

    def set_playing(self, user, row, song_id=None):
        """What the app says is playing for ``user`` now (shown on the page, and said to Last.fm); ``song_id``: the
        song here, for its cover."""
        user = str(user or "").strip()
        artist, title = str(row.get("artist") or "").strip(), str(row.get("title") or "").strip()
        if not artist or not title:
            return False
        try:
            duration = int(float(row.get("duration") or 0)) or None
        except (TypeError, ValueError):
            duration = None
        playing = {"artist": artist, "title": title, "album": str(row.get("album") or "").strip() or None,
                   "duration": duration, "at": int(self.now()), "song": song_id}
        self.playing[user] = playing
        keys = self._keys(user)
        if keys:
            threading.Thread(target=self._say_playing, args=(user, keys, playing), name="lastfm-now", daemon=True).start()
        return True

    def _say_playing(self, user, keys, playing):
        try:
            self.lastfm.now_playing(keys, playing)
            if (self.relay_state.get(user) or {}).get("playing"):
                self.relay_state.pop(user, None)
        except LastFmError as e:
            # Said on the page's Last.fm line too, not only in the detailed log (the owner, 11 Nov).
            self.relay_state[user] = {"error": f"Last.fm wasn't told what's playing: {e}", "at": self.now(), "playing": True}
            _log.warning("Last.fm wasn't told what %s is playing: %s", user or "the listener", e)

    def all_playing(self):
        """``[{"user", "artist", "title", …}]``: what's playing now for each listener."""
        return [dict(self.now_playing(u), user=u) for u in list(self.playing) if self.now_playing(u)]

    def rename(self, old, new):
        """Makes ``old``'s listens ``new``'s (a history brought in without a name, then the app's listens under the
        music server's user name: the owner, 11 Nov). A listen both have is kept once; sending on to Last.fm moves
        too unless ``new`` has its own. How many listens moved."""
        old, new = str(old or "").strip(), str(new or "").strip()
        if old == new:
            return 0
        with self.store._write():
            moved = self.store.db.execute("UPDATE OR IGNORE listens SET user=? WHERE user=?", (new, old)).rowcount
            self.store.db.execute("DELETE FROM listens WHERE user=?", (old,))  # the ones new had already
        relays = dict(self._relays())
        if old in relays:
            mine = relays.pop(old)
            relays.setdefault(new, mine)
            self.store.set_setting("lastfm_relay", relays)
        since = (relays.get(new) or {}).get("since")
        if since:  # the app's listens since sending on was connected (under the other name) go on now
            with self.store._write():
                self.store.db.execute("UPDATE listens SET relayed=0 WHERE user=? AND source='siansa' AND relayed IS NULL "
                                      "AND at>=?", (new, since))
        self.changes += 1
        self.store.event(f"The listens of {old or 'no name'} are {new or 'no name'}'s now ({moved} moved)")
        return moved

    def now_playing(self, user):
        """What's playing for ``user``, while it can still be: said within the song's length (10 minutes without)."""
        playing = self.playing.get(user)
        if playing and self.now() - playing["at"] <= (playing.get("duration") or 600) + 30:
            return playing
        return None

    # ---- files ----

    def start_import(self, path, user, name=None, done=None):
        """Brings the export at ``path`` in as ``user``'s, in a thread (a long history takes a while); ``done`` is
        called after it, whatever came of it. False when one runs already."""
        with self._lock:
            if self.importing.get("running"):
                return False
            self.importing = {"running": True, "file": name, "user": user, "started": self.now(), "read": 0}
        threading.Thread(target=self._import, args=(path, user, name, done), name="listens-import", daemon=True).start()
        return True

    def _import(self, path, user, name, done):
        state = {"running": False, "file": name, "user": user, "finished": None}
        try:
            state.update(self.import_file(path, user))
            self.store.event(f"{state['kept']} listens of {SOURCES.get(state['format'], state['format'])} were brought in"
                             + (f" for {user}" if user else "") + (f" ({state['known']} were here already)" if state["known"] else ""))
        except CantRead as e:
            state["error"] = str(e)
        except Exception as e:  # nothing here may take the service down
            state["error"] = f"{type(e).__name__}: {e}"
            _log.exception("Bringing in %s failed", name)
        state["finished"] = self.now()
        self.importing = state
        if done:
            done()

    def import_file(self, path, user):
        """One export, in this thread: ``{"format", "kept", "known", "skipped", "first", "last"}``."""
        kind, rows = read_export(path)
        sums, first, last, chunk = {"kept": 0, "known": 0, "skipped": 0}, None, None, []

        def keep():
            for k, v in self.add(user, chunk, source=kind).items():
                sums[k] += v
            self.importing = dict(self.importing, read=sums["kept"] + sums["known"] + sums["skipped"])
            chunk.clear()

        for row in rows:
            if isinstance(row.get("at"), int):
                first, last = min(first or row["at"], row["at"]), max(last or row["at"], row["at"])
            chunk.append(row)
            if len(chunk) >= 5000:
                keep()
        keep()
        if not sums["kept"] and not sums["known"]:
            raise CantRead("No listens were found in the file")
        _log.info("Brought in an export of %s for %s: %d listens kept, %d here already, %d without a date, artist or "
                  "title", kind, user or "the listener", sums["kept"], sums["known"], sums["skipped"])
        return {"format": kind, **sums, "first": first, "last": last}

    def remove(self, user, source=None):
        """Removes ``user``'s listens (those from ``source`` only, when given). How many went."""
        with self.store._write():
            if source:
                gone = self.store.db.execute("DELETE FROM listens WHERE user=? AND source=?", (user, source)).rowcount
            else:
                gone = self.store.db.execute("DELETE FROM listens WHERE user=?", (user,)).rowcount
        if gone:
            self.changes += 1
        return gone

    # ---- the numbers ----

    def window(self, year=None, month=None, tz=None):
        """``(from, to, label)``: the seconds a year or one of its months covers on the clock of ``tz``; all time
        without a year."""
        if not year:
            return 0, 2 ** 62, "All time"
        start = datetime.datetime(year, month or 1, 1, tzinfo=tz)
        end = datetime.datetime(year + (1 if not month or month == 12 else 0), 1 if not month or month == 12 else month + 1,
                                1, tzinfo=tz)
        return int(start.timestamp()), int(end.timestamp()), f"{MONTHS[month - 1]} {year}" if month else str(year)

    def top(self, user, kind="songs", start=0, end=2 ** 62, limit=50, server=None):
        """The most listened ``kind`` ("songs", "artists" or "albums") of ``user`` from ``start`` to ``end``, most
        first: ``[{"artist", "title"|"album", "listens", "id" (``server``'s for the song, when it has it)}]``."""
        limit = max(1, min(1000, int(limit)))
        if kind == "artists":
            rows = self.store._rows(
                "SELECT artist, COUNT(*) AS listens, MAX(at) AS last, MAX(song_id) AS song_id FROM listens WHERE user=? "
                "AND at>=? AND at<? GROUP BY akey ORDER BY listens DESC, last DESC LIMIT ?", (user, start, end, limit))
            return [{"artist": r["artist"], "listens": r["listens"], "song": r["song_id"]} for r in rows]
        if kind == "albums":
            rows = self.store._rows(
                "SELECT album, artist, COUNT(*) AS listens, MAX(at) AS last, MAX(song_id) AS song_id FROM listens "
                "WHERE user=? AND at>=? AND at<? AND album IS NOT NULL GROUP BY akey, album COLLATE NOCASE "
                "ORDER BY listens DESC, last DESC LIMIT ?", (user, start, end, limit))
            return [{"album": r["album"], "artist": r["artist"], "listens": r["listens"], "song": r["song_id"]} for r in rows]
        rows = self.store._rows(
            "SELECT artist, title, album, COUNT(*) AS listens, MAX(at) AS last, MAX(song_id) AS song_id FROM listens "
            "WHERE user=? AND at>=? AND at<? GROUP BY akey, tkey ORDER BY listens DESC, last DESC LIMIT ?",
            (user, start, end, limit))
        ids = self.store.item_ids([r["song_id"] for r in rows if r["song_id"]], server) if server else {}
        return [{"artist": r["artist"], "title": r["title"], "album": r["album"], "listens": r["listens"],
                 "id": ids.get(r["song_id"]) if server else r["song_id"], "song": r["song_id"]} for r in rows]

    def recent(self, user, limit=30):
        return [dict(r) for r in self.store._rows(
            "SELECT at, artist, title, album, source, song_id AS song FROM listens WHERE user=? ORDER BY at DESC LIMIT ?",
            (user, limit))]

    def summary(self, user, year=None, month=None, tz_name=None, most=25):
        """Everything the page shows of ``user``'s listening in a month, a year or all time; kept until a listen is
        added."""
        kept_as = (user, year, month, tz_name, most)
        with self._lock:
            kept = self._kept.get(kept_as)
            if kept and kept[0] == self.changes:
                return dict(kept[1], now=self.now_playing(user), recent=self.recent(user, 12))
        answer = self._summary(user, year, month, tz_name, most)
        with self._lock:
            if len(self._kept) > 60:
                self._kept.clear()
            self._kept[kept_as] = (self.changes, answer)
        return dict(answer, now=self.now_playing(user), recent=self.recent(user, 12))

    def _summary(self, user, year, month, tz_name, most):
        tz = zone(tz_name)
        start, end, label = self.window(year, month, tz)
        local = lambda at: datetime.datetime.fromtimestamp(at, tz)  # noqa: E731

        # Every listen's second, for the years and months there's something in.
        every = [r["at"] for r in self.store._rows("SELECT at FROM listens WHERE user=? ORDER BY at", (user,))]
        years, months = Counter(), Counter()
        for at in every:
            d = local(at)
            years[d.year] += 1
            if year and d.year == year:
                months[d.month] += 1

        rows = self.store._rows("SELECT at, akey, tkey, album, duration, song_id FROM listens WHERE user=? AND at>=? "
                                "AND at<? ORDER BY at", (user, start, end))
        # A listen's length: its own, else its song's from another listen or from the library, else a typical one.
        lengths = {(r["akey"], r["tkey"]): r["d"] for r in self.store._rows(
            "SELECT akey, tkey, MAX(duration) AS d FROM listens WHERE user=? AND duration IS NOT NULL GROUP BY akey, tkey", (user,))}
        library = {r["id"]: r["duration"] for r in self.store._rows(
            "SELECT id, duration FROM songs WHERE id IN (SELECT DISTINCT song_id FROM listens WHERE user=? AND at>=? "
            "AND at<? AND song_id IS NOT NULL)", (user, start, end))} if rows else {}
        seconds = guessed = 0
        hours, weekdays, days = [0] * 24, [0] * 7, Counter()
        artists, songs, albums = set(), set(), set()
        for r in rows:
            length = r["duration"] or lengths.get((r["akey"], r["tkey"])) or library.get(r["song_id"])
            if not length:
                length, guessed = TYPICAL_SECONDS, guessed + 1
            seconds += length
            d = local(r["at"])
            hours[d.hour] += 1
            weekdays[d.weekday()] += 1
            days[d.date()] += 1
            artists.add(r["akey"])
            songs.add((r["akey"], r["tkey"]))
            if r["album"]:
                albums.add((r["akey"], r["album"].casefold()))

        # Artists heard for the first time in this period.
        firsts = self.store._rows("SELECT artist, MIN(at) AS first, COUNT(*) AS listens FROM listens WHERE user=? "
                                  "GROUP BY akey HAVING first>=? AND first<?", (user, start, end))
        new_artists = sorted(firsts, key=lambda r: -r["listens"])

        if not year:
            unit, points = "year", [{"label": str(y), "listens": years[y], "year": y}
                                    for y in range(min(years), max(years) + 1)] if years else []
        elif not month:
            unit, points = "month", [{"label": MONTHS[m - 1][:3], "listens": months[m], "month": m} for m in range(1, 13)]
        else:
            last = (datetime.date(year + (month == 12), month % 12 + 1, 1) - datetime.timedelta(days=1)).day
            unit, points = "day", [{"label": str(d), "listens": days[datetime.date(year, month, d)]} for d in range(1, last + 1)]

        streak = best = 0
        previous = None
        for day in sorted(days):
            streak = streak + 1 if previous and (day - previous).days == 1 else 1
            best, previous = max(best, streak), day
        busiest = max(days.items(), key=lambda p: p[1]) if days else None
        first = self.store._rows("SELECT at, artist, title FROM listens WHERE user=? AND at>=? AND at<? ORDER BY at LIMIT 1",
                                 (user, start, end))
        return {
            "user": user, "period": {"label": label, "year": year, "month": month, "zone": str(tz_name or "UTC")},
            "years": [{"year": y, "listens": n} for y, n in sorted(years.items())],
            "months": [months[m] for m in range(1, 13)] if year else None,
            "totals": {"listens": len(rows), "seconds": seconds, "guessed": guessed, "artists": len(artists),
                       "songs": len(songs), "albums": len(albums), "days": len(days), "new_artists": len(new_artists)},
            "chart": {"unit": unit, "points": points},
            "top": {kind: self.top(user, kind, start, end, most) for kind in ("songs", "artists", "albums")},
            "new_artists": [{"artist": r["artist"], "listens": r["listens"]} for r in new_artists[:12]],
            "hours": hours, "weekdays": [{"day": WEEKDAYS[i], "listens": n} for i, n in enumerate(weekdays)],
            "records": {"busiest_day": {"date": busiest[0].isoformat(), "listens": busiest[1]} if busiest else None,
                        "streak": best, "first": dict(first[0]) if first else None,
                        "per_day": round(len(rows) / len(days), 1) if days else 0},
            "all": {"listens": len(every), "first": every[0] if every else None, "last": every[-1] if every else None},
        }

    def artist(self, user, name, tz_name=None, most=20):
        """Everything the page shows of one artist in ``user``'s listening, all time: ``None`` when they never
        listened to it. Names are compared as :func:`key` does ("AC/DC" is "ac dc")."""
        akey = key(name)
        if not akey:
            return None
        rows = self.store._rows("SELECT at, artist, tkey, title, album, duration, song_id FROM listens WHERE user=? "
                                "AND akey=? ORDER BY at", (user, akey))
        if not rows:
            return None
        tz = zone(tz_name)
        local = lambda at: datetime.datetime.fromtimestamp(at, tz)  # noqa: E731
        lengths = {r["tkey"]: r["duration"] for r in rows if r["duration"]}
        library = {r["id"]: r["duration"] for r in self.store._rows(
            "SELECT id, duration FROM songs WHERE id IN (SELECT DISTINCT song_id FROM listens WHERE user=? AND akey=? "
            "AND song_id IS NOT NULL)", (user, akey))}
        seconds = guessed = 0
        years, hours, days = Counter(), [0] * 24, set()
        for r in rows:
            length = r["duration"] or lengths.get(r["tkey"]) or library.get(r["song_id"])
            if not length:
                length, guessed = TYPICAL_SECONDS, guessed + 1
            seconds += length
            d = local(r["at"])
            years[d.year] += 1
            hours[d.hour] += 1
            days.add(d.date())
        # Its place among the listener's artists, all time.
        ahead = self.store._rows("SELECT COUNT(*) AS n FROM (SELECT akey, COUNT(*) AS c FROM listens WHERE user=? "
                                 "GROUP BY akey HAVING c > ?)", (user, len(rows)))[0]["n"]
        everything = self.count(user)
        songs = self.store._rows(
            "SELECT title, album, COUNT(*) AS listens, MAX(at) AS last, MAX(song_id) AS song_id FROM listens "
            "WHERE user=? AND akey=? GROUP BY tkey ORDER BY listens DESC, last DESC LIMIT ?", (user, akey, most))
        albums = self.store._rows(
            "SELECT album, COUNT(*) AS listens, MAX(at) AS last, MAX(song_id) AS song_id, COUNT(DISTINCT tkey) AS songs "
            "FROM listens WHERE user=? AND akey=? AND album IS NOT NULL GROUP BY album COLLATE NOCASE "
            "ORDER BY listens DESC, last DESC LIMIT ?", (user, akey, most))
        busiest = max(years.items(), key=lambda p: p[1])
        first, last = rows[0], rows[-1]
        return {
            "user": user, "artist": last["artist"], "song": next((r["song_id"] for r in reversed(rows) if r["song_id"]), None),
            "totals": {"listens": len(rows), "seconds": seconds, "guessed": guessed, "songs": len({r["tkey"] for r in rows}),
                       "albums": len({(r["album"] or "").casefold() for r in rows if r["album"]}), "days": len(days),
                       "share": round(len(rows) / everything * 100, 1) if everything else 0, "rank": ahead + 1},
            "first": {"at": first["at"], "title": first["title"]}, "last": {"at": last["at"], "title": last["title"]},
            "busiest_year": {"year": busiest[0], "listens": busiest[1]},
            "years": [{"year": y, "listens": years[y]} for y in range(min(years), max(years) + 1)],
            "hours": hours,
            "songs": [{"title": r["title"], "album": r["album"], "artist": last["artist"], "listens": r["listens"],
                       "song": r["song_id"]} for r in songs],
            "albums": [{"album": r["album"], "artist": last["artist"], "listens": r["listens"], "songs": r["songs"],
                        "song": r["song_id"]} for r in albums],
            "recent": [{"at": r["at"], "title": r["title"], "album": r["album"], "artist": r["artist"], "song": r["song_id"]}
                       for r in list(reversed(rows))[:10]],
        }

    # ---- on to Last.fm ----

    def _relays(self):
        saved = self.store.setting("lastfm_relay") or {}
        return saved if isinstance(saved, dict) else {}

    def relays(self, user):
        return user in self._relays()

    def _keys(self, user):
        saved = self._relays().get(user)
        if not saved or self.vault is None:
            return None
        return {k: self.vault.open(saved.get(k)) for k in ("api_key", "secret", "session")}

    def lastfm_status(self, user):
        """``{"connected", "name", "waiting", "sent", "error"}`` of ``user``'s sending on to Last.fm; never a key."""
        saved = self._relays().get(user)
        counts = {r["relayed"]: r["n"] for r in self.store._rows(
            "SELECT relayed, COUNT(*) AS n FROM listens WHERE user=? AND relayed IS NOT NULL GROUP BY relayed", (user,))}
        return {"connected": bool(saved), "name": (saved or {}).get("name"), "since": (saved or {}).get("since"),
                "waiting": counts.get(0, 0), "sent": counts.get(1, 0), "turned_away": counts.get(2, 0) + counts.get(3, 0),
                "error": (self.relay_state.get(user) or {}).get("error")}

    def connect_lastfm(self, user, api_key, secret, name, password):
        """Signs ``user`` in to Last.fm with their own API account and keeps the session (encrypted; the password
        isn't kept). Listens from now on are sent on. :class:`LastFmError` says why not."""
        api_key, secret, name = str(api_key or "").strip(), str(secret or "").strip(), str(name or "").strip()
        if not (api_key and secret and name and password):
            raise ValueError("Last.fm's API key, shared secret, user name and password are all needed")
        if self.vault is None:
            raise ValueError("Secrets can't be kept here")
        session = self.lastfm.session(api_key, secret, name, str(password))
        saved = dict(self._relays())
        saved[user] = {"api_key": self.vault.seal(api_key), "secret": self.vault.seal(secret),
                       "session": self.vault.seal(session), "name": name, "since": int(self.now())}
        self.store.set_setting("lastfm_relay", saved)
        self.relay_state.pop(user, None)
        self.store.event(f"Listens{' of ' + user if user else ''} are sent on to Last.fm ({name}) from now on")
        return self.lastfm_status(user)

    def love(self, user, artist, title, loved=True):
        """Passes a heart from the app on to ``user``'s Last.fm, in the background; False when they don't send on."""
        keys = self._keys(user)
        artist, title = str(artist or "").strip(), str(title or "").strip()
        if not keys or not all(keys.values()) or not artist or not title:
            return False

        def send():
            try:
                self.lastfm.love(keys, artist, title, loved)
                _log.info("%s %s – %s on Last.fm for %s", "Loved" if loved else "Unloved", artist, title, user or "the listener")
            except LastFmError as e:
                self.relay_state[user] = {"error": f"A heart wasn't passed on: {e}", "at": self.now()}
                _log.warning("Last.fm didn't take a heart for %s: %s", user or "the listener", e)

        threading.Thread(target=send, name="lastfm-love", daemon=True).start()
        return True

    def disconnect_lastfm(self, user):
        saved = dict(self._relays())
        if saved.pop(user, None) is None:
            return False
        self.store.set_setting("lastfm_relay", saved)
        with self.store._write():  # what was waiting isn't sent later to another account
            self.store.db.execute("UPDATE listens SET relayed=NULL WHERE user=? AND relayed=0", (user,))
        self.relay_state.pop(user, None)
        self.store.event(f"Listens{' of ' + user if user else ''} are no longer sent on to Last.fm")
        return True

    def relay(self):
        """Sends the waiting listens on to Last.fm, oldest first: ``{listener: sent}``. What Last.fm can't take now
        stays waiting; a listen older than two weeks, or one it turns away, is marked and left."""
        sent = {}
        with self._relay_lock:
            for user in list(self._relays()):
                keys = self._keys(user)
                if not keys or not all(keys.values()):
                    continue
                sent[user] = self._relay_user(user, keys)
        return sent

    def _relay_user(self, user, keys):
        sent = 0
        oldest = self.now() - LastFm.OLDEST
        with self.store._write():
            self.store.db.execute("UPDATE listens SET relayed=3 WHERE user=? AND relayed=0 AND at<?", (user, oldest))
        while True:
            rows = [dict(r) for r in self.store._rows(
                "SELECT id, at, artist, title, album, duration FROM listens WHERE user=? AND relayed=0 ORDER BY at LIMIT ?",
                (user, LastFm.MOST))]
            if not rows:
                break
            try:
                self.lastfm.scrobble(keys, rows)
            except LastFmError as e:
                if e.code is None or e.code in LastFm.LATER:
                    self.relay_state[user] = {"error": f"{e} (tried again later)", "at": self.now()}
                elif e.code in (4, 9, 10, 14, 26):  # the session or the API account is no longer taken
                    self.relay_state[user] = {"error": f"{e}: connect Last.fm again", "at": self.now()}
                else:
                    self.relay_state[user] = {"error": str(e), "at": self.now()}
                    self._mark(rows, 2)
                _log.warning("Sending %d listens on to Last.fm for %s: %s", len(rows), user or "the listener", e)
                break
            self._mark(rows, 1)
            self.relay_state.pop(user, None)
            sent += len(rows)
        if sent:
            _log.info("Sent %d listens on to Last.fm for %s", sent, user or "the listener")
        return sent

    def _mark(self, rows, state):
        with self.store._write():
            self.store.db.executemany("UPDATE listens SET relayed=? WHERE id=?", [(state, r["id"]) for r in rows])
