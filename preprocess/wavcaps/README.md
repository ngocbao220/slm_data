# WavCaps

**Large-scale weakly-labelled audio captioning** (ChatGPT-assisted) from four sources: FreeSound, BBC Sound Effects, SoundBible, and the AudioSet Strongly-labelled subset.

| Info | Value |
|------|-------|
| **Source** | https://huggingface.co/datasets/cvssp/WavCaps |
| **Paper** | WavCaps: A ChatGPT-Assisted Weakly-Labelled Audio Captioning Dataset for Audio-Language Multimodal Research (arXiv: 2303.17395) |
| **License** | cc-by-4.0 — research/academic use only (see FreeSound / BBC / SoundBible terms) |
| **Language** | Captions in English |

## Statistics

| Source | # audio | avg. duration (s) | avg. text length |
|--------|---------|-------------------|------------------|
| FreeSound | 262,300 | 85.98 | 6.77 |
| BBC Sound Effects | 31,201 | 115.04 | 9.67 |
| SoundBible | 1,232 | 13.12 | 5.87 |
| AudioSet SL subset | 108,317 | 10.00 | 9.79 |
| **WavCaps** | **403,050** | **67.59** | **7.80** |

## Layout (as downloaded from HuggingFace)

```
<dataset_path>/json_files/<Source>/<...>_final.json   metadata (captions + ids)
<dataset_path>/json_files/blacklist/*.json            overlap blacklists
<dataset_path>/Zip_files/<Source>/<Name>.z01 ... .zip audio volumes (flac)
```

Audio is shipped as **multi-volume zip parts** per source (`<Name>.z01 ... <Name>.zNN` + a final `<Name>.zip` holding the end-of-archive). Each part is ~5 GB; the whole repo is ~820 GB. Do **not** concatenate the parts on disk — `preprocess.py --extract` streams the members across volumes directly.

Metadata JSON rows all live under a top-level `data` key and keep a literal `audio: "wav_path"` placeholder. Audio files inside the archives are `<id>.flac` (AudioSet ids carry a leading `Y` and are stored with a `.wav` suffix in the JSON). `json_files/blacklist/` lists ids overlapping external evaluation sets (AudioCaps/Clotho: `all_ac`, `test_ac`; UrbanSound8K/ESC50/VGGSound: `ub8k_esc50_vggsound`) for leakage-free training/evaluation splits.

## Columns (per source, preserved in manifest)

| Source | Main columns |
|--------|--------------|
| FreeSound | `id`, `file_name`, `tags`, `description`, `author`, `duration`, `caption`, ... |
| BBC | `id`, `description`, `category`, `caption`, `duration`, ... |
| SoundBible | `id`, `title`, `description`, `author`, `caption`, `duration`, ... |
| AudioSet SL | `id`, `caption`, `duration` |

Manifest rows additionally get `source` and an `audio` field pointing to the extracted file (the `"wav_path"` placeholder is replaced).

## Scope of this repository

This repository contains the WavCaps audit (`stats.py`) and its tests only.
Download and extraction scripts from the original project are not included.
WavCaps exceeds the 50 GB local-download limit: acquire and extract it on the
server before auditing. Audio must be extracted under `audio/<Source>/`.
The audit never downloads, extracts, resamples, or changes audio.

## Evaluate the downloaded dataset

Install `soundfile` and `numpy` in the active Python environment. Run on the
server containing the full data (no torch, CUDA, model or network required):

```bash
python preprocess/wavcaps/stats.py /home/voice/data/voice/datasets/WavCaps
```

If original JSONs and extracted audio are in different locations:

```bash
python preprocess/wavcaps/stats.py /home/voice/data/voice/WavCaps \
    --audio-root /home/voice/data/voice/datasets/WavCaps/audio \
    --output /home/voice/data/voice/datasets/WavCaps/metadata/audit
```

`--workers` defaults to **64 CPU processes**. Each process decodes one file;
only small statistics return to the parent process, which writes the reports.
At most `2 * workers` tasks are pending. A source with fewer files uses fewer
processes. Worker results are collected in path order so totals and diagnostics
remain identical to serial execution. Spawned workers work on Linux and macOS.

```bash
# Default: 64 workers
python preprocess/wavcaps/stats.py /home/voice/data/voice/datasets/WavCaps
# Reduce CPU / disk pressure, or use 1 for serial execution
python preprocess/wavcaps/stats.py /home/voice/data/voice/datasets/WavCaps --workers 16
```

The worker count is not a promise of a 64x speedup: storage throughput and
available CPU cores limit performance. For network disks or a quick header-only
scan, fewer workers can be faster. `stats.json` records the requested worker count.

`--sources soundbible` audits one source. The default scans all four sources.
`--header-only` reads headers for a faster first inventory; it cannot establish
that the whole audio decodes correctly. Default mode decodes every audio file
in bounded 65,536-frame blocks, reporting progress every 1,000 files.
This can take hours on the full dataset; it does not resample or modify audio.

Outputs (existing reports are replaced on rerun; do not run concurrent audits
into the same output directory):

- `stats.json`: absolute input paths; per-source metadata rows, unique IDs,
  duplicate IDs, physical audio files, readable files, bytes/GB, measured hours,
  sample-rate/channel distributions, and deltas against published counts.
- `issues.jsonl`: every missing/ambiguous audio match, unreadable/empty audio,
  orphan audio, duplicate ID, empty caption, malformed metadata row,
  invalid duration, and duration discrepancy greater than 0.1 seconds.
- `assessment.md`: measured local summary and paper comparison, ready to use
  when updating the dataset README or task. The existing README is preserved.

Hours of matched audio count each unique source/ID once; orphan files have
separate `all_audio_hours` totals. Duplicate IDs and multiple files with the
same normalized stem are reported rather than silently selecting one file.
The audit uses original JSON metadata, not `trainset.jsonl`, so rows skipped by
preprocessing remain visible. It expects `audio/<Source>/`, with already extracted audio; adjust `--audio-root` to the parent of those folders.
It does not apply overlap blacklists: a deliberately filtered dataset will
show missing clips against the full metadata. It does not validate the output
manifest or judge whether captions accurately describe the audio.

Exit code is **0** when no technical/metadata issues are found, **1** when
issues are reported, **2** for CLI usage errors. Published-count deltas are
informational: a technically clean subset can still be incomplete. Read the
count comparison before claiming the complete release was acquired.

### Published comparison

The [official WavCaps repository](https://github.com/XinhaoMei/WavCaps) lists
403,050 clips across the four sources (counts above).
[Audio Flamingo 3, Appendix G, Table 11](https://arxiv.org/html/2507.08128v1)
reports **402.6K samples and 3,793.3 hours** for WavCaps in its training data.
These rounded training-data figures are a separate comparison, not a required
full-release total. The auditor reports raw measured hours without truncation
or resampling and makes no assumption about AF3 filtering.

The statistics table above is published reference information, **not measured
local results**. Fill local sizes/hours only after running the audit.

## Exploitable tasks (QA only)

| Task | How to use |
|------|-----------|
| Audio description QA | Ask “Describe what you hear”; use `caption` as the target |
| Synthetic open-ended sound QA | Generate questions grounded in caption facts and verify answers against audio |
| Yes/no or multiple-choice sound QA | Build questions from caption facts; manually review alternatives and ambiguity |

Caption generation is weakly supervised. This technical audit does not establish
semantic caption quality, license clearance, or absence of overlap with other
benchmarks; apply the official overlap blacklists before training/evaluation.

## Tests

```bash
python -m unittest discover -s preprocess/wavcaps -p test_stats.py
```
