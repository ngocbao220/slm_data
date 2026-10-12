"""Audit extracted WavCaps against original metadata; never download or alter audio."""
import argparse
import ast
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
import json
import math
import multiprocessing
from pathlib import Path
import sys

import numpy as np
import soundfile as sf

# Kept independent of preprocess.py: auditing does not require torch/pyarrow.
SOURCES = {
    'audioset_sl': ('AudioSet_SL', 'as_final.json', 108317),
    'freesound': ('FreeSound', 'fsd_final.json', 262300),
    'bbc_sound_effects': ('BBC_Sound_Effects', 'bbc_final.json', 31201),
    'soundbible': ('SoundBible', 'sb_final.json', 1232),
}
EXTENSIONS = {'.flac', '.wav', '.mp3', '.ogg', '.m4a'}
AF3 = {'url': 'https://arxiv.org/html/2507.08128v1',
       'section': 'Appendix G, Table 11', 'samples_approx': 402600, 'hours': 3793.3}


def key(value):
    value = str(value)
    return (value[:-4] if value.lower().endswith('.wav') else value).lower()


def inspect_audio(path, decode):
    """One independent file per process; return metadata, never audio buffers."""
    result = {'path': path, 'bytes': 0}
    try:
        result['bytes'] = path.stat().st_size
        with sf.SoundFile(path) as audio:
            frames, rate, channels = len(audio), audio.samplerate, audio.channels
            if frames <= 0 or rate <= 0:
                raise ValueError('empty audio or invalid sample rate')
            if decode:
                decoded = 0
                while True:
                    block = audio.read(65536, dtype='float32', always_2d=True)
                    if not len(block):
                        break
                    if not np.isfinite(block).all():
                        raise ValueError('non-finite audio samples')
                    decoded += len(block)
                if decoded != frames:
                    raise ValueError(f'decoded {decoded} frames, header claims {frames}')
            result.update(duration=frames / rate, rate=rate, channels=channels)
    except (OSError, RuntimeError, ValueError) as exc:
        result['error'] = str(exc)
    return result


def inspect_files(paths, decode, workers):
    """Bound pending work to 2 * workers and preserve sorted result order."""
    count = min(workers, len(paths))
    if count <= 1:
        for path in paths:
            yield inspect_audio(path, decode)
        return
    # Spawn avoids inheriting native-library thread state and works on macOS/Linux.
    with ProcessPoolExecutor(max_workers=count, mp_context=multiprocessing.get_context('spawn')) as pool:
        iterator = iter(paths)
        pending = deque(pool.submit(inspect_audio, path, decode)
                        for path in islice(iterator, 2 * count))
        while pending:
            result = pending.popleft().result()
            path = next(iterator, None)
            if path is not None:
                pending.append(pool.submit(inspect_audio, path, decode))
            yield result


def audit(root, audio_root, output, sources, decode=True, workers=64):
    if workers < 1:
        raise ValueError('workers must be a positive integer')
    root, audio_root, output = Path(root).resolve(), Path(audio_root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {'dataset_root': str(root), 'audio_root': str(audio_root),
              'verification': 'full_decode' if decode else 'header_only', 'workers': workers,
              'sources': {}, 'references': {'audio_flamingo_3': AF3,
              'wavcaps_counts': 'https://github.com/XinhaoMei/WavCaps'},
              'coverage_ok': True}
    # Temporary output prevents a failed run from replacing a completed report.
    issues_tmp = output / 'issues.jsonl.part'
    issue_counts = Counter()
    with issues_tmp.open('w', encoding='utf-8') as issues:
        def issue(source, kind, **details):
            issue_counts[kind] += 1
            issues.write(json.dumps({'source': source, 'kind': kind, **details}, ensure_ascii=False) + '\n')

        for source in sources:
            folder, filename, published_count = SOURCES[source]
            stats = Counter(metadata_rows=0, unique_ids=0, duplicate_ids=0,
                            audio_files=0, valid_audio_files=0, valid_matched_audio=0,
                            matched_audio_seconds=0, all_audio_seconds=0, audio_bytes=0)
            report['sources'][source] = stats
            stats['published_samples'] = published_count
            meta = root / 'json_files' / folder / filename
            print(f'[{source}] metadata: {meta}', flush=True)
            try:
                top = json.loads(meta.read_text(encoding='utf-8'))
                rows = top['data']
                if isinstance(rows, str):
                    rows = ast.literal_eval(rows)
                if not isinstance(rows, list) or not rows:
                    raise ValueError('data must be a nonempty list')
            except (OSError, ValueError, SyntaxError, KeyError, TypeError) as exc:
                issue(source, 'metadata_error', path=str(meta), error=str(exc))
                report['coverage_ok'] = False
                rows = []
            index = defaultdict(list)
            durations = {}
            rates, channels = Counter(), Counter()
            source_audio = audio_root / folder
            paths = sorted(p for p in source_audio.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
            stats['audio_files'] = len(paths)
            for path in paths:
                index[path.stem.lower()].append(path)
            print(f'[{source}] checking {len(paths)} audio files with {min(workers, len(paths))} workers', flush=True)
            for n, result in enumerate(inspect_files(paths, decode, workers), 1):
                path = result['path']
                stats['audio_bytes'] += result['bytes']
                if 'error' in result:
                    issue(source, 'unreadable_audio', path=str(path), error=result['error'])
                    report['coverage_ok'] = False
                else:
                    durations[path] = result['duration']
                    rates[str(result['rate'])] += 1
                    channels[str(result['channels'])] += 1
                    stats['valid_audio_files'] += 1
                    stats['all_audio_seconds'] += result['duration']
                if n % 1000 == 0 or n == len(paths):
                    print(f'[{source}] audio {n}/{len(paths)}', flush=True)
            seen, referenced = set(), set()
            stats['metadata_rows'] = len(rows)
            for n, row in enumerate(rows, 1):
                if not isinstance(row, dict) or row.get('id') in (None, ''):
                    issue(source, 'invalid_row', row=n)
                    report['coverage_ok'] = False
                    continue
                rid = key(row['id'])
                caption = row.get('caption')
                if not isinstance(caption, str) or not caption.strip():
                    issue(source, 'empty_caption', id=str(row['id']), row=n)
                    report['coverage_ok'] = False
                if rid in seen:
                    stats['duplicate_ids'] += 1
                    issue(source, 'duplicate_id', id=str(row['id']), row=n)
                    report['coverage_ok'] = False
                    continue
                seen.add(rid)
                matches = index.get(rid, [])
                referenced.update(matches)
                if len(matches) != 1:
                    issue(source, 'missing_audio' if not matches else 'ambiguous_audio',
                          id=str(row['id']), paths=[str(p) for p in matches])
                    report['coverage_ok'] = False
                    continue
                path = matches[0]
                if path not in durations:
                    continue
                duration = durations[path]
                stats['valid_matched_audio'] += 1
                stats['matched_audio_seconds'] += duration
                raw_duration = row.get('duration')
                if raw_duration not in (None, ''):
                    try:
                        declared = float(raw_duration)
                        if not math.isfinite(declared) or declared <= 0:
                            raise ValueError('duration must be finite and positive')
                    except (ValueError, TypeError):
                        issue(source, 'invalid_duration', id=str(row['id']), value=raw_duration)
                    else:
                        if abs(declared - duration) > 0.1:
                            issue(source, 'duration_mismatch', id=str(row['id']),
                                  metadata_seconds=declared, measured_seconds=duration)
            for path in paths:
                if path not in referenced:
                    issue(source, 'orphan_audio', path=str(path))
            stats['unique_ids'] = len(seen)
            stats['sample_rates'] = dict(rates)
            stats['channels'] = dict(channels)
            stats['matched_audio_hours'] = stats['matched_audio_seconds'] / 3600
            stats['all_audio_hours'] = stats['all_audio_seconds'] / 3600
            stats['audio_gb'] = stats['audio_bytes'] / 1e9
            stats['count_delta_vs_full_release'] = stats['unique_ids'] - published_count
    issues_tmp.replace(output / 'issues.jsonl')
    report['issue_counts'] = dict(issue_counts)
    report['total_matched_samples'] = sum(s['valid_matched_audio'] for s in report['sources'].values())
    report['total_matched_hours'] = sum(s['matched_audio_hours'] for s in report['sources'].values())
    report['af3_comparison'] = {
        'samples_delta_approx': report['total_matched_samples'] - AF3['samples_approx'],
        'hours_delta': report['total_matched_hours'] - AF3['hours'],
        'note': 'AF3 reports its training data, rounded counts; this is not a full-release completeness target. Selected sources: ' + ', '.join(sources)}
    tmp = output / 'stats.json.part'
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(output / 'stats.json')
    lines = ['# WavCaps local audit', '', f'Dataset: `{root}`', f'Audio: `{audio_root}`',
             f'Verification: {report["verification"]}; coverage OK: {report["coverage_ok"]}', '',
             '| Source | Metadata rows | Unique IDs | Audio files | Matched readable audio | Measured hours | Full release count |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for source, s in report['sources'].items():
        lines.append(f'| {source} | {s["metadata_rows"]} | {s["unique_ids"]} | {s["audio_files"]} | {s["valid_matched_audio"]} | {s["matched_audio_hours"]:.3f} | {s["published_samples"]} |')
    lines.extend(['', f'Total: {report["total_matched_samples"]:,} matched audio files, {report["total_matched_hours"]:.3f} measured hours.',
                  '', '[Audio Flamingo 3, Appendix G, Table 11](https://arxiv.org/html/2507.08128v1): 402.6K samples, 3793.3 hours.',
                  report['af3_comparison']['note'], '',
                  'Detailed diagnostics: `issues.jsonl`; counters, paths and comparisons: `stats.json`.',
                  'This audit checks coverage and technical integrity, not semantic caption quality or licensing clearance.', '',
                  '## QA uses', '',
                  '- Audio description QA: caption as the target for “Describe what you hear.”',
                  '- Synthetic sound QA: use captions as grounding; verify generated answers against the audio.',
                  '- Yes/no or multiple-choice QA: derive questions from caption facts; review distractors.', '',
                  '## Issues', '', *[f'- {k}: {v}' for k, v in sorted(issue_counts.items())]])
    tmp = output / 'assessment.md.part'
    tmp.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    tmp.replace(output / 'assessment.md')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path, help='Root containing json_files/')
    parser.add_argument('--audio-root', type=Path, help='Directory containing the four source folders (default: PATH/audio)')
    parser.add_argument('--output', type=Path, help='Report directory (default: PATH/metadata/audit)')
    parser.add_argument('--sources', nargs='+', choices=list(SOURCES), default=list(SOURCES))
    parser.add_argument('--workers', type=int, default=64, help='CPU worker processes (default: 64; 1 runs serially)')
    parser.add_argument('--header-only', action='store_true', help='Faster duration scan; does not prove audio decodability')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be a positive integer')
    if not args.path.is_dir():
        parser.error(f'Dataset root does not exist: {args.path}')
    report = audit(args.path, args.audio_root or args.path / 'audio',
                   args.output or args.path / 'metadata/audit', list(dict.fromkeys(args.sources)), not args.header_only, args.workers)
    print(f'{report["total_matched_samples"]:,} matched samples; {report["total_matched_hours"]:.3f} hours; reports: {args.output or args.path / "metadata/audit"}')
    return 0 if report['coverage_ok'] and not report['issue_counts'] else 1


if __name__ == '__main__':
    sys.exit(main())
