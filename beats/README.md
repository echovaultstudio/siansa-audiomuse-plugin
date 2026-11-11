# Siansa Beats

A test of [Beat This!](https://github.com/CPJKU/beat_this) (CPJKU, 2024), the beat and downbeat tracker, on your own
music. It reads the songs of the libraries you choose on Navidrome, Emby or Jellyfin, finds every beat and downbeat in each, and shows them on a page:
its tempo and bars next to the file's BPM tag, so misread songs stand out. You can play any song with a click on each beat to hear whether they're right.

It runs in its own container, reads your music servers without changing anything, and keeps what it finds in its own
database. Nothing is written to your music files or the servers. Siansa Dev talks to it directly (Settings › Connections
› Siansa Beats) for each song's loudness, fades, key and beat grid, the genres and their descriptions: the
[app's API](#the-apps-api-v1). The Siansa Beats (dev) plugin on AudioMuse-AI can still pass its beat grids on
(`GET /api/grids`) for those who'd rather keep the plugin.

| | |
|---|---|
| **Reads** | The libraries chosen on its page (all of them, or leaving some out, as compilations) of Navidrome, Emby and Jellyfin, several of them and several users of one: each song's original file, BPM tag and genre. A song two servers have is analysed once. |
| **Finds** | Every beat and downbeat (Beat This!, the `final0` model, on the CPU or an NVIDIA GPU), and from them the tempo of the whole song and of each end (the first and last 30 seconds), how steady it is, the beats a bar and the first downbeat. Each song's loudness (EBU R128) and how it fades in and out, and its key. |
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

3. **`.env` needs nothing filled in.** No password goes in it or in the YAML (0.24.0): once it runs, open the page
   and under **Setup** add your music server (as you sign in to Siansa), and if you like AudioMuse-AI (for similar
   albums and artists), a password for the page and a key for the app.

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

The page shows one page at a time, picked from the menu on the left (on a phone, behind the ☰ button), in three groups.
The button at the top right says whether everything works (a green dot, "All working"); tap it for each connection
(your music servers, AudioMuse-AI), the models and how the app connects.
Each page has its own address (`#songs`, `#loudness`…), so Back and bookmarks work:

| Group | Page | What's on it |
|---|---|---|
| Analysis | **Overview** | What's being done now, how far the beats, loudness and genres are, and the latest activity |
| | **Libraries** | The server's libraries to analyse, Start analysis, Look for new songs, and starting over |
| | **Songs** | The table, and a song's page (its beats, loudness, fades, key and genres) when one is open |
| For the Siansa app | **Loudness & keys** | How many songs are measured (here or taken from the plugin), how loud they are, how they start and end, their keys, and the ones that couldn't be measured |
| | **Genres** | The style model's progress, how often it agrees with the tags, the genre descriptions (with **Fetch all now**), and every genre by songs |
| | **Similar music** | Whether AudioMuse-AI is set up, and a song to try: the albums and artists like it |
| | **Bad blends** | The song changes marked as bad in Siansa Dev |
| Service | **Activity**, **Log** | What was done, and the container's log with Debug details |
| | **Setup** | The music servers (add one, or another user of one), the connections and the model (no passwords), how to connect the app, the debug report |

### Several music servers, and several users of one

Siansa Beats reads **Navidrome, Emby and Jellyfin**, as many as you have (0.23.0). Add each under
**Setup › Music servers**: its kind, a name, its address, a user name and password. **Edit** changes one (its name, or
with its password its address, user or sign-in). It's only kept once it answers and takes the login (**Test** asks
without adding, and says why when it can't). Then tick its libraries under **Libraries**: they're read in the
background, and the page shows which library is being read and how far.

- The address can be pasted as the browser shows it (`http://nas:8096/web/index.html#!/home`): it's cut back to the
  server's own, and an Emby that answers under `/emby` only is found there.
- An Emby or Jellyfin user without a password: leave the password empty.
- When a test fails, the page and the log say why: nothing at that port, not that kind of server (Jellyfin chosen
  for an Emby), an untrusted certificate, or the user name or password refused.

- **A song two servers have is analysed once.** When a server's library is read, each song is looked for among the
  songs already here: by its file (the same path, or paths that end alike, as each server has the music folder under
  its own path: `/music/Artist/Album/01 Song.flac` and `/mnt/media/music/Artist/Album/01 Song.flac`), then by its tags
  (title, artists and album), then by title and album when the lengths agree. Lengths more than 3 seconds apart never
  match. A song that can't be placed for sure is analysed as a new one: never guessed. Activity says how many were
  found and by what ("4,012 already analysed from another server (4,012 by their file)"). So pointing Emby or Jellyfin
  at the folder Navidrome has costs a read of the library, not days of analysis.
- **Navidrome's real paths** come from its own API (the one its web page uses), with the same user and password: the
  path in the Subsonic API is made up from the tags unless the player is set to report real paths.
- **Another user of the same server** is added the same way (same kind and address, another user name): it sees the
  libraries that user may, and its songs are the first user's by their ids.
- **The menu's Music server switch** (shown once there are two) picks whose songs the pages show: the overview's
  numbers, the libraries, the table, the genres and the loudness are that server's alone.
- **The default server** is read first and is the one the Siansa app is answered for when it doesn't name one (an app
  from before 0.23, or the plugin). **Make default** changes it.
- **Removing a server** forgets the songs only it had; songs another server has too stay, analysed.
- No password is kept as it's typed (see "Secrets" below), and none is ever shown, logged or put in the debug
  report. Emby's and Jellyfin's token is sent in a header, never in an address. Nothing is ever written to a server.
- Similar albums and artists are still asked of AudioMuse-AI, by the ids of the server AudioMuse-AI knows.

1. On **Libraries**, tick the libraries to analyse and press **Start analysis**. It reads their songs from Navidrome and works
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
app's Logs on TrueNAS) and to the **Log** page: starting up, the libraries read, each song
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

**Genres by vote.** Each song's genres come from a vote, for Siansa's genre radios:

- **its sound first**: Essentia's Discogs-EffNet (the Music Technology Group, UPF) hears the song and scores each of
  400 Discogs styles, from Progressive Trance and Deep House to Punk and Boom Bap. It runs on the GPU when there is one
  (through the PyTorch Beat This! uses; the model is small, so Beat This! hardly notices), else on the CPU (ONNX
  Runtime, `STYLE_THREADS`, 2 by default); `BEATS_DEVICE=cpu` keeps both on the CPU. New songs are heard on
  their way through the queue; songs analysed before are downloaded once more and heard while the queue is empty;
- **its tags second**: "Electronic / Trance", "Drum And Bass" and "DnB" are matched onto the same styles. A tag
  naming a style adds 0.35 to it, one naming a genre (Electronic, Rock) 0.10 to each of its styles; so a tag tips a
  close call, and a wrong tag (an album tag on a remix) loses to what every bar of the song sounds like. A song
  without a sound score is voted on its tags alone;
- a song keeps up to 5 styles of 0.15 or more, and is in a genre's radio from 0.30.

The **Genres** card says how far it is, where the model runs, how often the sound agrees with the tags (the tagged genre, and the tagged
style in the sound's top 3), and the styles with the most songs. A song's page shows its vote, what its sound said
and its tags; the table's genre column and the CSV (`voted_genres`, `sound_top`) the vote. Last.fm's tags can join
the vote later. The model is under CC BY-NC-ND 4.0 (free for non-commercial use, unchanged), downloaded while the
image is built; how often it's right on your music is what the agreement says, and what you hear.

## Test it

Every change is tested by GitHub Actions (`.github/workflows/beats.yml` in this repository):

- **Without the model**: the tempo, bar and comparison maths (`tests/test_grid.py`), and the whole service against
  a fake Navidrome (`tests/test_service.py`): the token login, libraries, the queue, failures, the
  page's API, the audio passing through with seeking, the CSV, the log and debug report (with no secrets in them),
  pause and the password. `tests/test_page_script.py` checks the page's script parses.
- **Several servers** (`tests/test_match.py`, `tests/test_servers.py`): finding the same song on two servers by its
  file or tags and never by a guess; fake Emby and Jellyfin servers, each taking only its own way of logging in; a
  song another server has not analysed again; a song only the other has, analysed from it; a second user of one
  server; each server's own ids and numbers with `?server=`; and no password in any answer.
- **Real servers** (`tests/real_servers.py`): real Navidrome, Emby and Jellyfin with the same music (the Siansa app's
  test servers, their newest releases and the oldest it supports): each one's login, libraries, songs, file paths and
  downloads, and the songs analysed from Navidrome found again on the other two by their files, none analysed twice.
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
  model: about 40 ms a song, so half an hour for 45,000; the top shows how far it is), and analyses again the songs
  that failed on the two bugs it fixes.

## Settings

| Variable | Default | |
|---|---|---|
| `STYLE_THREADS` | `2` | The style model's CPU threads. |
| `LOG_LEVEL` | `info` | `debug`: every step of every song in the log, from the start. The page's **Debug details** switch changes it while running and wins over this. |
| `BEATS_MODEL` | `final0` | `small0` is smaller and faster, a little less exact. Both are in the image. |
| `BEATS_DEVICE` | `auto` | `auto` (the GPU when the container has one), `cpu`, `cuda`, `cuda:N`. Needs the `cuda` image for a GPU. |
| `BEATS_FLOAT16` | `0` | `1`: half precision on a GPU. |
| `BEATS_THREADS` | `0` (all) | The model's CPU threads. |
| `BEATS_CPUS` | `4` | (docker-compose) The cores the container may use. |
| `BEATS_SECRET_KEY` | empty | The key the kept secrets are encrypted with, to keep it out of the data folder. Without it, a key file (`secret.key`) is made there. Changing it means entering the servers' passwords and AudioMuse-AI's token again. |
| `BEATS_RESET_ACCESS` | empty | `yes`: takes the page's password and the app's key away at the start, when the password is forgotten. Take it off again afterwards. |

The music servers, AudioMuse-AI, the page's password and the app's key are **not** settings of the container: they're
set on the page, under Setup.

### Secrets

No secret is kept as plain text, and none belongs in the YAML (0.24.0).

| Secret | How it's kept |
|---|---|
| The page's password | **Hashed** (scrypt, salted). It can't be read back, only checked. |
| The app's key | **Hashed** (SHA-256 of a random key). It's shown once, when you make it under Setup › Access. |
| Emby's and Jellyfin's password | **Not kept.** It's used once to sign in; the sign-in token the server gives is kept, encrypted. If the server stops taking it (you signed its devices out), Edit the server and enter the password again. |
| Navidrome's password, AudioMuse-AI's token | **Encrypted.** They can't be hashed: Siansa Beats has to say them to those servers each time, and a hash can't be said in their place. |

What's encrypted is locked with a key that isn't in the database: `BEATS_SECRET_KEY` from the container when you set
one, else the file `secret.key` in the data folder, readable by Siansa Beats alone. A copy or backup of `beats.db`
without the key gives nothing away. Someone who has the whole data folder, key file included, can decrypt them, so
set `BEATS_SECRET_KEY` (in the container's settings, or a Docker secret) to keep the key apart from the data. If the
key is lost, the servers show "enter its password again".

**Coming from 0.23 or before:** `NAVIDROME_URL`, `NAVIDROME_USER`, `NAVIDROME_PASSWORD`, `AUDIOMUSE_URL`,
`AUDIOMUSE_TOKEN`, `BEATS_PASSWORD` and `BEATS_API_KEY` are taken over once at the first start of 0.24 (hashed or
encrypted) and never read again. Setup says which are still in your YAML: delete those lines. Nothing is analysed
again, and the app keeps its key.

A key for the app needs the page to have a password (without one the page is open and the app needs no key), and
goes when the password is removed.

## The app's API (v1)

What the Siansa app asks for, under `/api/v1/`, described in full in [`beats/openapi.json`](beats/openapi.json)
(`GET /api/v1/openapi.json`). Within v1, fields are only ever added, so an older app keeps working.

| | |
|---|---|
| `GET /api/v1/capabilities` | What this Siansa Beats can do now and how far it is: songs analysed, measured for loudness, the style model. |
| `GET /api/v1/profiles?ids=a,b` | Up to 500 songs: loudness (`ref_lufs`), fades (`start_ramp`, `end_ramp`), key and scale, and the beat grid at each end. Songs with nothing yet are `pending`. |
| `GET /api/v1/genres` | Every style with songs voted into it. |
| `GET /api/v1/genre?name=&limit=` | The songs of a genre, by vote. |
| `GET /api/v1/genre/about?name=` | A short description of a genre, from Wikipedia. Every genre's (the voted ones and the songs' tags) is fetched in the background every 6 hours and kept; one not fetched yet waits up to 3 s and then answers `"pending": true`, to ask again shortly. |
| `GET /api/v1/similar/albums?track_id=` | Albums that sound like that song's album, asked of AudioMuse-AI's Siansa plugin (until Siansa Beats works them out itself). |
| `GET /api/v1/similar/artists?track_id=` or `?artist=` | Artists that sound like that one, the same way. |
| `POST /api/v1/bad-blends` | A song change marked as bad in the app. |

With several music servers, every one of these takes `?server=`: the id (or name) of the server the app plays from,
from the capabilities' `servers` (each with its `kind`, `url` and `user`, to tell which is the app's). Ids asked about
(`profiles`, `bad-blends`) are understood from any server; ids answered (`genre`) are that server's own, and the counts
(`capabilities`, `genres`) are of its songs. Without `server` the default one is meant. (`similar/…` passes its `server`
on to AudioMuse-AI, whose own name for a server it is.)

Songs analysed before 0.20 get their loudness, fades and key while the queue is empty: with AudioMuse-AI connected
(Setup), whatever AudioMuse-AI's Siansa plugin already measured is taken from it (nothing to download),
and the rest are downloaded once more and measured here. New songs are measured on the way.

## The page's API

`GET /api/health`, `/api/status`, `/api/libraries`, `/api/songs`, `/api/songs/<id>` (with every beat),
`/api/beats/<id>`, `/api/audio/<id>`, `/api/grids?ids=`, `/api/export.csv`, `/api/export.json`;
`/api/bad-blends`, `/api/failed/export`, `/api/genres`, `/api/genre?name=&limit=`, `/api/log?after=&limit=&level=`, `/api/log/download`, `/api/debug`; `POST /api/libraries {"ids"}`, `/api/libraries/read`, `/api/bad-blends`, `/api/songs/<id>/analyse`, `/api/analyse-all`, `/api/log/level {"level"}`, `/api/pause`, `/api/resume`; `DELETE /api/songs/<id>`, `/api/bad-blends/<n>`, `/api/libraries/<id>/songs`. See
`beats/server.py`.

Beat This! is by Francesco Foscarin, Jan Schlüter and Gerhard Widmer (ISMIR 2024), under the MIT licence.
Discogs-EffNet is by Pablo Alonso-Jiménez, Xavier Serra and Dmitry Bogdanov (MTG, Universitat Pompeu Fabra), from
[Essentia's models](https://essentia.upf.edu/models.html), under CC BY-NC-ND 4.0.
