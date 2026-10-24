# Siansa Analysis for AudioMuse-AI

The [AudioMuse-AI](https://github.com/NeptuneHub/AudioMuse-AI) plugin for the Siansa music player.

- **Volume & fade profiles:** each song's loudness (EBU R128) for levelling, and how it starts and ends, for
  smooth crossfades.
- **Similar albums & artists:** from AudioMuse-AI's sound analysis, rebuilt by itself when the library or the
  analysis changes.

Each part shows its own progress on the plugin's settings page.

## Install

In AudioMuse-AI, open **Plugins › Repositories** and add:

```
https://raw.githubusercontent.com/echovaultstudio/siansa-audiomuse-plugin/main/manifest.json
```

Then install **Siansa Analysis** from the **Catalog** tab and click **Apply now (restart)**. Needs
AudioMuse-AI 3.6.4 or later; no extra packages.

Details, the API and how it works: [plugins/SiansaAnalysis/README.md](plugins/SiansaAnalysis/README.md).

## License

[GNU Affero General Public License v3.0](LICENSE), like AudioMuse-AI.
