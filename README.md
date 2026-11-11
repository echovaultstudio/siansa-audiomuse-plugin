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
- **How songs sound.** The same analysis AudioMuse-AI does, with its models: similar songs, albums and artists,
  song paths, blends, mood radios, "your sound", and finding music by describing it in words (vibes and their
  sliders, sound cards). It takes over what AudioMuse-AI has already worked out, from every music server it
  analyses, and hears new songs itself, so AudioMuse-AI can then be switched off.
- **Genres from the sound.** Each song's genres from what it sounds like (400 styles, from Progressive Trance to
  Boom Bap), for the app's genre list, genre pages and genre radios, with a short description of each genre.
- **Your listening history.** Siansa sends it your listens ("Scrobble to Siansa Beats"); its Listening page is a
  time machine of any month or year: top songs, artists and albums, hours, when you listen, records. It takes in
  exports of Koito, Last.fm and ListenBrainz, and can send your listens on to Last.fm.
- **Housekeeping.** Songs removed from the server are forgotten; new ones are analysed as they arrive.

It reads Navidrome, Emby and Jellyfin, several servers and several users of one, without changing anything on them
or in your music files. It runs on a CPU or an NVIDIA GPU, and everything is set up on its own web page (port 8090).

**Set it up:** [`beats/README.md`](beats/README.md). The image is `ghcr.io/echovaultstudio/siansa-beats`
(`:latest` for the CPU, `:cuda` for an NVIDIA GPU; `:dev` and `:dev-cuda` to try what's coming).

## Without Siansa Beats: AudioMuse-AI

Siansa talks to [AudioMuse-AI](https://github.com/NeptuneHub/AudioMuse-AI) directly, as it always has: set its
address in the app under **Settings › Connections › Services › AudioMuse-AI**. Similar songs, song paths, mixes, mood radios,
the sonic fingerprint and finding music by describing it then come from AudioMuse-AI. With both set up, the app asks
Siansa Beats first and AudioMuse-AI when Siansa Beats can't answer. How it's set up in the app:
[Siansa's AudioMuse page](https://github.com/echovaultstudio/Siansa/blob/main/docs/audiomuse.md).

What only Siansa Beats does: DJ mixes (they need its beat grids; a tempo is only ever Beat This!'s), genres from the
sound, and the listening history.

### The Siansa Analysis plugin for AudioMuse-AI

Still here and still installable, for those who use AudioMuse-AI without Siansa Beats: each song's loudness and how
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
