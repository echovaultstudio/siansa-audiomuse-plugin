# Siansa Beats

The companion service for the [Siansa](https://github.com/echovaultstudio/Siansa) music player: a Docker container
next to your music server that listens to every song in your library and tells the app what it heard. The music
server keeps doing what it does (your library, playback); Siansa Beats adds what it can't.

Siansa never needs it: the app works on its own, and with AudioMuse-AI (below). Siansa Beats is for those who want
the most out of the app.

## What it does

- **Beats and bars.** Every beat and downbeat of each song ([Beat This!](https://github.com/CPJKU/beat_this)),
  so the app can mix one song into the next on the beat (DJ mixes), and each song's tempo.
- **Loudness, fades and key.** How loud each song is (EBU R128) for levelling, how it starts and ends for smooth
  crossfades, and its key for mixing.
- **How songs sound.** Similar songs, albums and artists, song paths, blends, mood radios, "your sound", and
  finding music by describing it in words (vibes and their sliders, sound cards). Siansa Beats hears every song
  itself, with the models the [AudioMuse-AI](https://github.com/NeptuneHub/AudioMuse-AI) project publishes (see the
  credits in [beats/README.md](beats/README.md#credits)); it needs no other service.
- **Genres from the sound.** Each song's genres from what it sounds like (400 styles, from Progressive Trance to
  Boom Bap), for the app's genre list, genre pages and genre radios, with a short description of each genre.
- **Your listening history.** Siansa sends it your listens ("Scrobble to Siansa Beats"); its Listening page is a
  time machine of any month or year: top songs, artists and albums, hours, when you listen, records. It takes in
  exports of Koito, Last.fm and ListenBrainz, and can send your listens on to Last.fm.
- **Housekeeping.** Songs removed from the server are forgotten; new ones are analysed as they arrive.

It reads Navidrome, Emby and Jellyfin, several servers and several users of one, without changing anything on them
or in your music files. It runs on a CPU or an NVIDIA GPU, and everything is set up on its own web page (port 8090).

## Quick start

One file on a Docker server, with nothing to fill in:

```
mkdir -p ~/siansa-beats && cd ~/siansa-beats
curl -fsSLO https://raw.githubusercontent.com/echovaultstudio/siansa-beats/main/beats/docker-compose.yml
docker compose up -d
```

Then open `http://<the server>:8090`. The page's **Get started** card walks through the rest: add your music server,
choose the libraries to analyse, and connect the Siansa app (Settings › Connections › Siansa Beats).

- **What you need:** Docker, about 3 GB of disk, 2–3 GB of memory while it works. An NVIDIA GPU makes it many times
  faster; it isn't needed.
- **Images:** `ghcr.io/echovaultstudio/siansa-beats` (`:latest` for the CPU, `:cuda` for an NVIDIA GPU; `:dev` and
  `:dev-cuda` to try what's coming).
- **Everything else** (a GPU, TrueNAS, several servers and users, every page, the settings, the API, a forgotten
  password): [`beats/README.md`](beats/README.md).

## How it works, in short

| What | Worked out by |
|---|---|
| Beats, bars and tempo | [Beat This!](https://github.com/CPJKU/beat_this) |
| Loudness and fades | The broadcast standard (ITU-R BS.1770 / EBU R128); no AI model |
| Key | Krumhansl–Kessler key profiles; no AI model |
| Genres | Discogs-EffNet or MAEST (your choice, under Setup) |
| How songs sound, moods, search by words | MusiCNN and CLAP |

Everything is kept in one SQLite file in the container's volume (`/data/beats.db`): copy it and you have a backup.
Nothing is ever written to your music server or your files.

## Without Siansa Beats: AudioMuse-AI

Siansa talks to [AudioMuse-AI](https://github.com/NeptuneHub/AudioMuse-AI) directly, as it always has: set its
address in the app under **Settings › Connections › AudioMuse-AI**. Similar songs, song paths, mixes, mood radios,
the sonic fingerprint and finding music by describing it then come from AudioMuse-AI. With both set up, the app asks
Siansa Beats first and AudioMuse-AI when Siansa Beats can't answer. How it's set up in the app:
[Siansa's AudioMuse page](https://github.com/echovaultstudio/Siansa/blob/main/docs/audiomuse.md).

What only Siansa Beats does: DJ mixes (they need its beat grids; a tempo is only ever Beat This!'s), genres from the
sound, and the listening history.

### The Siansa Analysis plugin for AudioMuse-AI

An extra, still here and still installable, for those who use AudioMuse-AI without Siansa Beats: each song's loudness and how
it starts and ends (for levelling and crossfades), and similar albums and artists, from inside AudioMuse-AI. It's
no longer worked on: Siansa Beats does all of it and more.

In AudioMuse-AI, open **Plugins › Repositories** and add:

```
https://raw.githubusercontent.com/echovaultstudio/siansa-beats/main/manifest.json
```

Then install **Siansa Analysis** from the **Catalog** tab and click **Apply now (restart)**. Needs
AudioMuse-AI 3.6.4 or later; no extra packages. Added it before this repository was renamed (from
`siansa-audiomuse-plugin`)? The old address keeps working.

Details and its API: [plugins/SiansaAnalysis/README.md](plugins/SiansaAnalysis/README.md).

## License

[GNU Affero General Public License v3.0](LICENSE), like AudioMuse-AI.
