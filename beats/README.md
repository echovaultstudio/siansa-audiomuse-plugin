# Siansa Beats

A test of [Beat This!](https://github.com/CPJKU/beat_this) (CPJKU, 2024), the beat and downbeat tracker, on your own
music. It reads the songs of the Navidrome libraries you choose, finds every beat and downbeat in each, and shows them on a page:
its tempo and bars next to the file's BPM tag, so misread songs stand out. You can play any song with a click on each beat to hear whether they're right.

It runs in its own container, reads Navidrome without changing anything, and keeps what it finds in its own
database. Nothing is written to your music files or Navidrome. The Siansa Beats (dev) plugin on AudioMuse-AI asks it
for each song's beat grid (`GET /api/grids`), for Siansa Dev's DJ blends.

| | |
|---|---|
| **Reads** | The Navidrome libraries chosen on its page (all of them, or leaving some out, as compilations): each song's original file, BPM tag and genre. |
| **Finds** | Every beat and downbeat (Beat This!, the `final0` model, on the CPU or an NVIDIA GPU), and from them the tempo of the whole song and of each end (the first and last 30 seconds), how steady it is, the beats a bar and the first downbeat. |
| **Keeps** | `/data/beats.db` (SQLite) in the container's volume. Each song is downloaded to `/data` while it's read, then deleted. |
| **Shows** | A page at port 8090: what it's doing, a table of every song with what the readings say together, and a page per song with its beats, a player with clicks on the beats, and its tempo along the song. |

## Set it up

You need a Docker server on the same network as Navidrome, with about 3 GB of disk for the image and 2–3 GB of
memory free while it works.

1. **Know your libraries.** Siansa Beats analyses the Navidrome libraries you tick on its page (Navidrome's
   Settings › Libraries lists them), so you can leave some out, such as compilations or audiobooks. A song takes a
   few seconds on a GPU, much longer on the CPU: try one small library first.

2. **Get the two files** into a folder on the server. The image is public, so no login is needed:

   ```
   mkdir -p ~/siansa-beats && cd ~/siansa-beats
   base=https://raw.githubusercontent.com/echovaultstudio/siansa-audiomuse-plugin/main/beats
   curl -fsSLO $base/docker-compose.yml && curl -fsSL $base/.env.example -o .env
   ```

3. **Fill in `.env`:**
   - `NAVIDROME_URL`, `NAVIDROME_USER`, `NAVIDROME_PASSWORD`: as you sign in to Siansa;
   - optionally `BEATS_PASSWORD`, if others on the network shouldn't see the page (its user name is `siansa`).

4. **Start it:**

   ```
   cd ~/siansa-beats
   docker compose pull && docker compose up -d
   docker compose logs -f        # Ctrl+C to stop watching
   ```

5. Open **http://\<the server\>:8090**. The pills at the top say whether Navidrome answers and how the model runs.

To update later: `docker compose pull && docker compose up -d`. What's been analysed is kept.

## On TrueNAS with a GPU

Beat This! runs much faster on an NVIDIA GPU. The **cuda** image (`ghcr.io/echovaultstudio/siansa-beats:cuda`) has
the GPU build of PyTorch; with `BEATS_DEVICE=auto` (the default) it uses the GPU when the container is given one,
and the CPU when it isn't. The page's first pill says which, with the GPU's name.

1. **The NVIDIA driver on TrueNAS** (once): Apps › Configuration › Settings › tick **Install NVIDIA Drivers**,
   Save. `nvidia-smi` in System › Shell should then list the card. (Driver 525 or newer; TrueNAS's is.)
2. **Know your libraries**, as in step 1 above.
3. **Install it:** Apps › Discover Apps › ⋮ › **Install via YAML**. Name it `siansa-beats`, paste
   [`truenas.yml`](truenas.yml) (or [raw](https://raw.githubusercontent.com/echovaultstudio/siansa-audiomuse-plugin/main/beats/truenas.yml)), fill in the values marked `<…>` (Navidrome's address, user and password),
   and Save.
4. Open **http://\<TrueNAS\>:8090**. After the first song, the pill says **Model final0 · GPU** and the card's name.
   If it says CPU and a warning appears at the top, the container didn't get the GPU: check step 1, and that
   `deploy: … driver: nvidia` is in the YAML.

On another Docker server with an NVIDIA GPU and NVIDIA's container toolkit:
`docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d`.

- `BEATS_DEVICE`: `auto` (the default), `cpu`, `cuda`, or `cuda:1` for the second GPU.
- `BEATS_FLOAT16=1`: half precision on the GPU. Faster and lighter on memory, but its readings can differ slightly
  from the full precision the model was tested in; compare a few songs before keeping it on.
- The song is decoded on the CPU either way (ffmpeg); only the model runs on the GPU. The GPU's memory is handed
  back after each song.
- GitHub's runners have no GPU, so the cuda image is tested there on the CPU. Its first run on a GPU is on your
  TrueNAS: the Activity list says where the model loaded and how long each song took.

## Use it

1. Tick the **libraries to analyse** and press **Start analysis**. It reads their songs from Navidrome and works
   through them a song at a time; the top shows the song
   and the step (downloading, decoding, loading the model, analysing), and **Activity** what's been done, with the
   time each step took. The first song takes longer while the model loads.
2. **The table**, a row a song:
   - **Tag**: the file's BPM tag.
   - **Beat This!**: the song's tempo, with intro / outro below (the tempo each end holds for most of its 30
     seconds, or **–** when it has no steady beat, as a beatless intro). It's the tempo the song spends most of its time at,
     so a breakdown, a half-time intro or a tempo change doesn't drag it off (the song's page also shows one
     straight line through every beat, as 0.2 gave it). **Bar**: beats a bar, and the share of
     bars that agree. **Steady**: how evenly the beats fall, and the share on a straight grid (a drum machine is
     near 100 %).
   - **Notes**: what the readings say together: whether they agree, or how they relate (**4/3**, **double**,
     **half**…; a misreading is nearly always one of those). An intro or outro at another tempo than the rest of
     the song is noted, not flagged. Genres are only shown: their tags are partly guessed, so they never change a
     reading or a note.
   - **Worth a listen** shows only the songs where readings really disagree (a warning). **Done** shows the songs analysed, **Failed** those that failed.
3. **Tap a song** to open it:
   - play it: **Clicks on the beats** ticks on every beat, higher on each downbeat. If the clicks sit on the kick
     drum, the beats are right; if they drift or fall between, they're not;
   - the clicks come **from the steady grid** when the song is at one tempo: one even grid at the song's tempo,
     lined up with the model's beats, that carries on through breakdowns and quiet intros (as DJ software does).
     Its bars come from a vote of the model's downbeats and then follow every 4 beats, so the phrase can't wander.
     A track off a mixed CD, where the DJ rode the pitch (its tempo creeps, 123.4 to 124.0 BPM, say), gets a grid
     that **follows the drift** instead: its local tempo read every 16 beats and joined smoothly, up to 4 % apart
     (more is a real tempo change, and gets no grid). Its row says "The grid follows a drifting tempo (0.8 %)", each
     end then carries the tempo it plays at there, and the CSV has `grid_drift`. A song whose even grid holds takes
     the following one only when clearly more of its beats sit on it;
     Pick **the model's beats** to hear what Beat This! heard; the other set shows as small ticks along the bottom
     of the beat view;
   - **bars and phrases come from the sound too.** While a song is analysed, Siansa Beats measures how loud its sub
     bass, bass, mids and highs are along it (kept small in the database, so it's worked out again without
     downloading the song). In dance music a new section (the bass coming in, the drop, the hats stopping) starts
     on beat 1 of a bar and at the start of a phrase. So the big changes check the model's downbeats: when three or
     more agree on another beat, the bars move there (**Bars moved** in the notes, worth a listen). And the bar of
     every 8 they land on is where the **phrases** start: a green mark in the beat view, ticks along the top of the
     chart, and a third, higher click. When fewer than half the big changes fall on one bar of 8, the song says
     **No clear phrases** rather than guessing;
   - a song is counted at the tempo the model heard: drum & bass heard at 87 stays at 87. The Siansa app counts half
     and double as the same tempo and doubles anything under 100 itself, so it still blends at 174;
   - the beat view shows 8 seconds around what's playing, downbeats in amber with their bar number. Tap it to jump;
   - the chart shows the tempo along the song (amber), the BPM tag (green), and the ends (the first and last 30 s,
     shaded). Tap it to jump there;
   - **Analyse again** reads it once more; **Remove** forgets it (nothing changes on Navidrome).
4. **Export CSV** gives the table, for a spreadsheet.

**The log.** Everything Siansa Beats does goes to the container's log (`docker compose logs -f siansa-beats`, or the
app's Logs on TrueNAS) and to the **Log** card at the bottom of the page: starting up, the libraries read, each song
done or failed, errors with their traceback. Tick **Debug details** to add every step of every song with its times and
sizes, what the model and the grid found, and every request to Navidrome with how long it took; it's kept across
restarts, so turn it off when the problem is found (a pill at the top says while it's on). `LOG_LEVEL=debug` turns it
on from the start. Passwords and tokens are never written: Navidrome's login token is taken out of every line.

**When something's wrong or stuck,** **Download debug report** saves one file with the versions (torch, CUDA, the
GPU), the setup (addresses, never passwords), the status, where each part of the service is right now (a stuck one
shows the line it waits on), the latest activity and the log. **Download log** saves the log alone. Send either.

**Back** (the browser's or the phone's) closes a song, and the **Siansa Beats** logo goes back to all the songs.
A song's address (`#song=…`) opens it straight away.

**Section marks.** While a song plays, tap **Section starts here** (or press **M**) on every drop, breakdown or big
change you hear. Each mark shows the bar it's nearest the start of, and whether it falls on a phrase start; tap it to
hear it again, ✕ to take it away. Marks are kept through analysing again, show as ◆ in the table, and go out with
**Download data**: they're what the phrase rules are checked against. About 20 songs with their drops marked is
enough.

**Download data** gives every song's beats and sound levels as one file (a few MB; song names and genres, no
addresses or passwords): what the grid and phrases are worked out from, to tune them on real music.

**Trying the dev version.** New work goes to the `dev` branch first and is published as `:dev` (CPU) and
`:dev-cuda` (GPU): put that tag in the YAML instead of `latest` / `cuda` and restart. It keeps the same data; going
back to `:cuda` keeps it too.

**Libraries.** Each ticked library shows how many of its songs are done. Ticks take effect with **Start analysis**.
Untick one and start to leave it out: its songs not yet analysed leave the queue, the ones done stay, and **Remove
them** next to it forgets them (their beats and marks; nothing changes on Navidrome). The choice is kept across restarts, and the ticked libraries are
read again every 6 hours for new songs (**Look for new songs** does it now). The next song is downloaded and decoded
while the model works on this one, so the model rarely waits for the network; the top says how long each song takes,
to judge how long a library will. The table shows the first 200 songs, the search finding the rest, and **Export
CSV** and **Download data** take the songs it finds (every song when nothing is searched: for a big library the data
is large).

## Test it

Every change is tested by GitHub Actions (`.github/workflows/beats.yml` in this repository):

- **Without the model**: the tempo, bar and comparison maths (`tests/test_grid.py`), and the whole service against
  a fake Navidrome (`tests/test_service.py`): the token login, libraries, the queue, failures, the
  page's API, the audio passing through with seeking, the CSV, the log and debug report (with no secrets in them),
  pause and the password. `tests/test_page_script.py` checks the page's script parses.
- **CPU or GPU** (`tests/test_analyser.py`): which device each `BEATS_DEVICE` gives, and the fallback to the CPU,
  with a note, when a GPU is asked for but missing.
- **With the model**: both images (CPU and cuda) are built, `tests/make_test_songs.py` makes five songs of known tempo and bars (140,
  174, 122, 128 BPM in 4/4, and 96 BPM in 3/4), one a BPM tag at half,
  and `tests/e2e.py` runs the real Beat This! on them through the service. The steady 4/4 dance songs must read
  within 1.5 % with 4 beats a bar; the others are reported. The run's log has the table, and the log and debug report are checked too.

On your own computer, without Docker or the model:

```
cd beats
python3 tests/test_grid.py && python3 tests/test_service.py
python3 tests/make_test_songs.py /tmp/songs        # needs ffmpeg
python3 tests/preview.py /tmp/songs                 # the page with a stand-in for the model: http://127.0.0.1:8090
```

On the server, after setting it up:

1. `docker compose ps` says **healthy**, and http://\<server\>:8090/api/health answers `"status": "ok"`.
2. The page's Navidrome pill is green.
3. **Tick a library** and **Start analysis**: the songs appear, the first one loads the model (the pill then says how long it took),
   and the rest follow. A song that fails says why in its row and in Activity, and the queue goes on.
4. Open Control (or another song you know): play it with clicks on, and check the clicks sit on the beat.

## What it can't do yet

- **Bars in 3/4.** In the first test run, Beat This! read every test song's tempo exactly (140, 174, 122, 128 and
  96 BPM, the 174 not halved), but counted the 96 BPM waltz as 4 beats a bar. Steady 4/4 music, what DJ blends
  need, read right. Watch the **Bar** column on your own 3/4 songs.

- Genres come from the files' tags only (the **tag** source), and are only shown. The database keeps each genre's
  source, so better ones (AudioMuse-AI's labels, say) can be added beside them later.
- `GET /api/grids?ids=a,b` gives each song's grid at both ends as the Siansa app reads it (through the Siansa
  Beats plugin on AudioMuse-AI, for Siansa Dev). `GET /api/beats/<song id>` gives every beat of one song.
- Each end is taken **inside the lock**: from the first 4 bars where the model's beats sit on the grid to the last
  (`intro_lock_ms`, `outro_lock_ms`). A track cut from a mixed CD carries the DJ's own transitions at its ends, the
  next or last record still playing; its blends belong inside the lock. Each end also says how many bars the bass
  stays out (`intro_bass_bars`, `outro_bass_bars`), how loud it is next to the song (`intro_energy`,
  `outro_energy`, dB), and the song where the bass drops out and comes back (`breakdowns_ms`, `drops_ms`), for the
  app to choose how to blend.
- **Bad blends**: a song change marked as bad in Siansa Dev lands in a list on the page, with how the app blended it
  (technique, bars, tempos, confidence at each end), and both songs a tap away. **Download bad blends** saves them all
  with both songs' beats, sound levels and results in one file (`GET /api/bad-blends/export`).
- **Failed songs**: when songs couldn't be analysed, **Download failed songs (N)** beside the exports saves them with
  why (the error), their file type, length and library in one file (`GET /api/failed/export`); the CSV export has the
  error too. Filter **Failed** shows them in the table.
- Phrases are 8 or 16 bars: 16 when three quarters of the big changes (at least 4) fall on every other start of
  8. A song whose sections don't change its sound much (a long, even techno track) may have no clear phrases.
- Songs analysed before 0.5.0 are analysed again once, on start, to measure their sound (a few seconds each).
- 0.15.0 works every song out again once from its saved beats, on start, for the following grid (no download or
  model: about 40 ms a song, so half an hour for 45,000; the top shows how far it is).

## Settings

| Variable | Default | |
|---|---|---|
| `NAVIDROME_URL`, `NAVIDROME_USER`, `NAVIDROME_PASSWORD` | | Navidrome, read only. The password never leaves the container: requests carry a salted token. |
| `LOG_LEVEL` | `info` | `debug`: every step of every song in the log, from the start. The page's **Debug details** switch changes it while running and wins over this. |
| `BEATS_MODEL` | `final0` | `small0` is smaller and faster, a little less exact. Both are in the image. |
| `BEATS_DEVICE` | `auto` | `auto` (the GPU when the container has one), `cpu`, `cuda`, `cuda:N`. Needs the `cuda` image for a GPU. |
| `BEATS_FLOAT16` | `0` | `1`: half precision on a GPU. |
| `BEATS_THREADS` | `0` (all) | The model's CPU threads. |
| `BEATS_CPUS` | `4` | (docker-compose) The cores the container may use. |
| `BEATS_PASSWORD` | empty | The page and API ask for it (user `siansa`); `/api/health` stays open for Docker. |

## API

`GET /api/health`, `/api/status`, `/api/libraries`, `/api/songs`, `/api/songs/<id>` (with every beat),
`/api/beats/<id>`, `/api/audio/<id>`, `/api/grids?ids=`, `/api/export.csv`, `/api/export.json`;
`/api/bad-blends`, `/api/failed/export`, `/api/log?after=&limit=&level=`, `/api/log/download`, `/api/debug`; `POST /api/libraries {"ids"}`, `/api/libraries/read`, `/api/bad-blends`, `/api/songs/<id>/analyse`, `/api/analyse-all`, `/api/log/level {"level"}`, `/api/pause`, `/api/resume`; `DELETE /api/songs/<id>`, `/api/bad-blends/<n>`, `/api/libraries/<id>/songs`. See
`beats/server.py`.

Beat This! is by Francesco Foscarin, Jan Schlüter and Gerhard Widmer (ISMIR 2024), under the MIT licence.
