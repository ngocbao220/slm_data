import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
from stats import audit


class AuditTest(unittest.TestCase):
    def test_audio_coverage_duration_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            meta = root / 'json_files/SoundBible/sb_final.json'
            meta.parent.mkdir(parents=True)
            meta.write_text(json.dumps({'data': str([
                {'id': 'ok', 'caption': 'A bell.', 'duration': 2},
                {'id': 'missing', 'caption': 'Rain.'},
                {'id': 'bad', 'caption': ''},
                {'id': 'ok', 'caption': 'Duplicate.'},
            ])}))
            audio = root / 'audio/SoundBible'
            audio.mkdir(parents=True)
            sf.write(audio / 'ok.flac', np.zeros(8000), 8000)
            sf.write(audio / 'orphan.flac', np.zeros(4000), 8000)
            (audio / 'bad.flac').write_text('broken')
            out = root / 'report'
            report = audit(root, root / 'audio', out, ['soundbible'], True, workers=1)
            stats = report['sources']['soundbible']
            self.assertEqual(stats['metadata_rows'], 4)
            self.assertEqual(stats['valid_matched_audio'], 1)
            self.assertAlmostEqual(stats['matched_audio_seconds'], 1)
            self.assertEqual(stats['duplicate_ids'], 1)
            issues = [json.loads(line)['kind'] for line in (out / 'issues.jsonl').read_text().splitlines()]
            for kind in ['missing_audio', 'unreadable_audio', 'duplicate_id', 'empty_caption', 'duration_mismatch', 'orphan_audio']:
                self.assertIn(kind, issues)

    def test_parallel_audit_matches_serial_including_errors(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            meta = root / 'json_files/SoundBible/sb_final.json'
            meta.parent.mkdir(parents=True)
            meta.write_text(json.dumps({'data': [
                {'id': 'one', 'caption': 'Bell', 'duration': 1},
                {'id': 'two', 'caption': 'Rain', 'duration': 2},
                {'id': 'bad', 'caption': 'Broken'},
            ]}))
            audio = root / 'audio/SoundBible'
            audio.mkdir(parents=True)
            sf.write(audio / 'one.flac', np.zeros(8000), 8000)
            sf.write(audio / 'two.flac', np.zeros(32000), 16000)
            (audio / 'bad.flac').write_text('broken')
            for decode in [True, False]:
                with self.subTest(decode=decode):
                    serial = audit(root, root / 'audio', root / 'serial', ['soundbible'], decode, workers=1)
                    parallel = audit(root, root / 'audio', root / 'parallel', ['soundbible'], decode, workers=2)
                    self.assertEqual(serial['sources'], parallel['sources'])
                    self.assertEqual(serial['issue_counts'], parallel['issue_counts'])
                    self.assertEqual(serial['total_matched_hours'], parallel['total_matched_hours'])
                    self.assertEqual((root / 'serial/issues.jsonl').read_text(),
                                     (root / 'parallel/issues.jsonl').read_text())

    def test_invalid_worker_count_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaises(ValueError):
                audit(root, root / 'audio', root / 'report', ['soundbible'], workers=0)
            self.assertFalse((root / 'report').exists())
            result = subprocess.run([sys.executable, str(Path(__file__).with_name('stats.py')),
                                     str(root), '--workers', '0'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('workers', result.stderr)

    def test_clean_subset_cli_writes_reports_and_exits_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            meta = root / 'json_files/SoundBible/sb_final.json'
            meta.parent.mkdir(parents=True)
            meta.write_text(json.dumps({'data': [{'id': 'ok.wav', 'caption': 'Bell', 'duration': 1}]}))
            audio = root / 'audio/SoundBible'
            audio.mkdir(parents=True)
            sf.write(audio / 'ok.flac', np.zeros(8000), 8000)
            result = subprocess.run([sys.executable, str(Path(__file__).with_name('stats.py')),
                                     str(root), '--sources', 'soundbible'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            out = root / 'metadata/audit'
            report = json.loads((out / 'stats.json').read_text())
            self.assertAlmostEqual(report['total_matched_hours'], 1 / 3600)
            self.assertTrue(report['coverage_ok'])
            self.assertEqual(report['workers'], 64)
            self.assertEqual((out / 'issues.jsonl').read_text(), '')
            self.assertIn('3793.3', (out / 'assessment.md').read_text())

    def test_missing_metadata_and_ambiguous_ids_are_failures(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            meta = root / 'json_files/SoundBible/sb_final.json'
            meta.parent.mkdir(parents=True)
            meta.write_text(json.dumps({'data': [{'id': 'same.wav', 'caption': 'Sound'}]}))
            audio = root / 'audio/SoundBible'
            audio.mkdir(parents=True)
            for ext in ['wav', 'flac']:
                sf.write(audio / ('same.' + ext), np.zeros(800), 8000)
            report = audit(root, root / 'audio', root / 'report', ['soundbible', 'freesound'], False, workers=1)
            self.assertFalse(report['coverage_ok'])
            self.assertEqual(report['sources']['soundbible']['valid_matched_audio'], 0)
            self.assertEqual(report['sources']['freesound']['metadata_rows'], 0)


if __name__ == '__main__':
    unittest.main()
