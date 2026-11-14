# Siansa Beats

The companion service for [Siansa](https://github.com/echovaultstudio/Siansa), the music player for your own
Navidrome, Emby or Jellyfin server. It's one Docker container beside your music server: it listens to every song in
your library once and tells the app what it heard. Siansa makes its DJ mixes, even volume, genres, radios and your
listening history from that.

Your music server keeps doing what it does (your library, playback). Nothing leaves your home, and nothing is ever
changed on the server or in your files.

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

## Good to know

- **Siansa never needs it.** The app plays from a music server alone. It also still talks to
  [AudioMuse-AI](https://github.com/NeptuneHub/AudioMuse-AI) for similar music and mixes; with both set up, it asks
  Siansa Beats first. DJ mixes, genres from the sound and the listening history are Siansa Beats' own.
- **In the app:** [Siansa's page about Siansa Beats](https://github.com/echovaultstudio/Siansa/blob/main/docs/siansa-beats.md).
- **On the web:** [siansa.app/beats](https://siansa.app/beats).

## License

[GNU Affero General Public License v3.0](LICENSE), like AudioMuse-AI.

<sub>Also in this repository: the older [Siansa Analysis plugin for AudioMuse-AI](plugins/SiansaAnalysis/README.md), no longer worked on.</sub>
