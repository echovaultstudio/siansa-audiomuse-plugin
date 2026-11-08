# Siansa Beats

A test of [Beat This!](https://github.com/CPJKU/beat_this) (CPJKU, 2024), the beat and downbeat tracker, on your own
music. It reads the songs of one Navidrome playlist, finds every beat and downbeat in each, and shows them on a page:
its tempo and bars next to the Siansa plugin's current readings, the file's BPM tag and the genre's usual tempos, so
misread songs stand out. You can play any song with a click on each beat to hear whether they're right.

It's a test bench, on its own: it runs in its own container, reads Navidrome (and AudioMuse-AI) without changing
anything, and keeps what it finds in its own database. Nothing is written to your music files, Navidrome,
AudioMuse-AI or the plugin. If Beat This! proves better, the plugin can later ask it for each song's beats.

| | |
|---|---|
| **Reads** | One Navidrome playlist (default **Beat test**): each song's original file, BPM tag and genre. Optionally the Siansa plugin's readings from AudioMuse-AI. |
| **Finds** | Every beat and downbeat (Beat This!, the `final0` model, on the CPU or an NVIDIA GPU), and from them the tempo of the whole song and of each end (the first and last 30 seconds, as the plugin measures), how steady it is, the beats a bar and the first downbeat. |
| **Keeps** | `/data/beats.db` (SQLite) in the container's volume. Each song is downloaded to `/data` while it's read, then deleted. |
| **Shows** | A page at port 8090: what it's doing, a table of every song with what the readings say together, and a page per song with its beats, a player with clicks on the beats, and its tempo along the song. |

## Set it up

You need a Docker server on the same network as Navidrome, with about 3 GB of disk for the image and 2–3 GB of
memory free while it works.

1. **Make the test library.** In Navidrome, make a playlist called **Beat test** with about 30 songs. A good mix:
   - songs you know are misread today (Control by Ecsape, read at 187 BPM instead of about 140);
   - steady dance music (house, techno, trance), and some drum & bass;
   - a few live or acoustic songs (a drummer, not a drum machine);
   - something in 3/4 (a waltz) or with a tempo change.

2. **Get the two files** into a folder on the server. The image is public, so no login is needed:

   ```
   mkdir -p ~/siansa-beats && cd ~/siansa-beats
   base=https://raw.githubusercontent.com/echovaultstudio/siansa-audiomuse-plugin/main/beats
   curl -fsSLO $base/docker-compose.yml && curl -fsSL $base/.env.example -o .env
   ```

3. **Fill in `.env`:**
   - `NAVIDROME_URL`, `NAVIDROME_USER`, `NAVIDROME_PASSWORD`: as you sign in to Siansa;
   - `AUDIOMUSE_URL` and, if AudioMuse-AI has a login, `AUDIOMUSE_TOKEN` (an API token from AudioMuse-AI): for
     the plugin's column. Leave them empty to skip that comparison;
   - optionally `BEATS_PASSWORD`, if others on the network shouldn't see the page (its user name is `siansa`).

4. **Start it:**

   ```
   cd ~/siansa-beats
   docker compose pull && docker compose up -d
   docker compose logs -f        # Ctrl+C to stop watching
   ```

5. Open **http://\<the server\>:8090**. The pills at the top say whether Navidrome and AudioMuse-AI answer.

To update later: `docker compose pull && docker compose up -d`. What's been analysed is kept.

## On TrueNAS with a GPU

Beat This! runs much faster on an NVIDIA GPU. The **cuda** image (`ghcr.io/echovaultstudio/siansa-beats:cuda`) has
the GPU build of PyTorch; with `BEATS_DEVICE=auto` (the default) it uses the GPU when the container is given one,
and the CPU when it isn't. The page's first pill says which, with the GPU's name.

1. **The NVIDIA driver on TrueNAS** (once): Apps › Configuration › Settings › tick **Install NVIDIA Drivers**,
   Save. `nvidia-smi` in System › Shell should then list the card. (Driver 525 or newer; TrueNAS's is.)
2. **Make the playlist** in Navidrome, as in step 1 above.
3. **Install it:** Apps › Discover Apps › ⋮ › **Install via YAML**. Name it `siansa-beats`, paste
   [`truenas.yml`](truenas.yml) (or [raw](https://raw.githubusercontent.com/echovaultstudio/siansa-audiomuse-plugin/main/beats/truenas.yml)), fill in the values marked `<…>` (Navidrome's address, user and password;
   AudioMuse-AI's address and token, or empty), and Save.
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

1. Press **Analyse playlist**. It reads the playlist and works through it a song at a time; the top shows the song
   and the step (downloading, decoding, loading the model, analysing), and **Activity** what's been done, with the
   time each step took. The first song takes longer while the model loads.
2. **The table**, a row a song:
   - **Tag**: the file's BPM tag. **Plugin**: the Siansa plugin's intro and outro tempo today.
   - **Beat This!**: the song's tempo, with intro / outro below. It's the tempo the song spends most of its time at,
     so a breakdown, a half-time intro or a tempo change doesn't drag it off (the song's page also shows one
     straight line through every beat, as 0.2 gave it). **Bar**: beats a bar, and the share of
     bars that agree. **Steady**: how evenly the beats fall, and the share on a straight grid (a drum machine is
     near 100 %).
   - **Notes**: what the readings say together: whether they agree, or how they relate (**4/3**, **double**,
     **half**…; a misreading is nearly always one of those), and whether the tempo is outside the genre's usual
     range (trance 125–150, drum & bass 160–180…, with 3 % to spare). Half or double the genre's tempo is the
     same beat counted another way (a drum & bass song read at 87 is 174), so it's noted, not flagged; so is an
     intro or outro at another tempo than the rest of the song.
   - **Worth a listen** shows only the songs where readings really disagree (a warning).
3. **Tap a song** to open it:
   - play it: **Clicks on the beats** ticks on every beat, higher on each downbeat. If the clicks sit on the kick
     drum, the beats are right; if they drift or fall between, they're not;
   - the beat view shows 8 seconds around what's playing, downbeats in amber with their bar number. Tap it to jump;
   - the chart shows the tempo along the song (amber), the BPM tag (green), the plugin's intro and outro (red),
     the ends the plugin measures (shaded) and the genre's range (green band). Tap it to jump there;
   - **Analyse again** reads it once more; **Remove** forgets it (nothing changes on Navidrome).
4. **Export CSV** gives the table, for a spreadsheet.

**The plugin's readings.** The pill at the top says whether the Siansa plugin answers and for how many songs it has
readings; when something's wrong, a line under the top says what to check (the address, the token, the plugin not
installed, a name the container can't look up…). **Read the plugin's values again** asks once more.

To add songs, add them to the playlist and press **Analyse playlist** again: only the new ones are read.

## Test it

Every change is tested by GitHub Actions (`.github/workflows/beats.yml` in this repository):

- **Without the model**: the tempo, bar and comparison maths (`tests/test_grid.py`), and the whole service against
  a fake Navidrome and AudioMuse-AI (`tests/test_service.py`): the token login, playlists, the queue, failures, the
  page's API, the audio passing through with seeking, the CSV, pause and the password.
- **CPU or GPU** (`tests/test_analyser.py`): which device each `BEATS_DEVICE` gives, and the fallback to the CPU,
  with a note, when a GPU is asked for but missing.
- **With the model**: both images (CPU and cuda) are built, `tests/make_test_songs.py` makes five songs of known tempo and bars (140,
  174, 122, 128 BPM in 4/4, and 96 BPM in 3/4), one carrying the plugin's 4/3 misreading and one a BPM tag at half,
  and `tests/e2e.py` runs the real Beat This! on them through the service. The steady 4/4 dance songs must read
  within 1.5 % with 4 beats a bar; the others are reported. The run's log has the table.

On your own computer, without Docker or the model:

```
cd beats
python3 tests/test_grid.py && python3 tests/test_service.py
python3 tests/make_test_songs.py /tmp/songs        # needs ffmpeg
python3 tests/preview.py /tmp/songs                 # the page with a stand-in for the model: http://127.0.0.1:8090
```

On the server, after setting it up:

1. `docker compose ps` says **healthy**, and http://\<server\>:8090/api/health answers `"status": "ok"`.
2. The page's pills are green for Navidrome (and AudioMuse-AI if you set it).
3. **Analyse playlist**: the songs appear, the first one loads the model (the pill then says how long it took),
   and the rest follow. A song that fails says why in its row and in Activity, and the queue goes on.
4. Open Control (or another song you know): play it with clicks on, and check the clicks sit on the beat.

## What it can't do yet

- **Bars in 3/4.** In the first test run, Beat This! read every test song's tempo exactly (140, 174, 122, 128 and
  96 BPM, the 174 not halved), but counted the 96 BPM waltz as 4 beats a bar. Steady 4/4 music, what DJ blends
  need, read right. Watch the **Bar** column on your own 3/4 songs.

- Genres come from the files' tags only (the **tag** source). The database keeps each genre's source, so better
  ones (a genre model, AudioMuse-AI's labels) can be added beside them, and the checks use every source.
- It doesn't feed the plugin yet. `GET /api/beats/<song id>` already gives a song's grid in the shape the plugin
  would read.

## Settings

| Variable | Default | |
|---|---|---|
| `NAVIDROME_URL`, `NAVIDROME_USER`, `NAVIDROME_PASSWORD` | | Navidrome, read only. The password never leaves the container: requests carry a salted token. |
| `PLAYLIST` | `Beat test` | The test library's playlist (the page can name another). |
| `AUDIOMUSE_URL`, `AUDIOMUSE_TOKEN` | empty | The Siansa plugin's readings, for the comparison. |
| `BEATS_MODEL` | `final0` | `small0` is smaller and faster, a little less exact. Both are in the image. |
| `BEATS_DEVICE` | `auto` | `auto` (the GPU when the container has one), `cpu`, `cuda`, `cuda:N`. Needs the `cuda` image for a GPU. |
| `BEATS_FLOAT16` | `0` | `1`: half precision on a GPU. |
| `BEATS_THREADS` | `0` (all) | The model's CPU threads. |
| `BEATS_CPUS` | `4` | (docker-compose) The cores the container may use. |
| `BEATS_PASSWORD` | empty | The page and API ask for it (user `siansa`); `/api/health` stays open for Docker. |

## API

`GET /api/health`, `/api/status`, `/api/playlists`, `/api/songs`, `/api/songs/<id>` (with every beat),
`/api/beats/<id>`, `/api/audio/<id>`, `/api/export.csv`; `POST /api/scan {"playlist"}`, `/api/songs/<id>/analyse`,
`/api/analyse-all`, `/api/plugin/refresh`, `/api/pause`, `/api/resume`; `DELETE /api/songs/<id>`. See
`beats/server.py`.

Beat This! is by Francesco Foscarin, Jan Schlüter and Gerhard Widmer (ISMIR 2024), under the MIT licence.
