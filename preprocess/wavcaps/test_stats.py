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
            report = audit(root, root / 'audio', out, ['soundbible'], True)
            stats = report['sources']['soundbible']
            self.assertEqual(stats['metadata_rows'], 4)
            self.assertEqual(stats['valid_matched_audio'], 1)
            self.assertAlmostEqual(stats['matched_audio_seconds'], 1)
            self.assertEqual(stats['duplicate_ids'], 1)
            issues = [json.loads(line)['kind'] for line in (out / 'issues.jsonl').read_text().splitlines()]
            for kind in ['missing_audio', 'unreadable_audio', 'duplicate_id', 'empty_caption', 'duration_mismatch', 'orphan_audio']:
                self.assertIn(kind, issues)

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
            report = audit(root, root / 'audio', root / 'report', ['soundbible', 'freesound'], False)
            self.assertFalse(report['coverage_ok'])
            self.assertEqual(report['sources']['soundbible']['valid_matched_audio'], 0)
            self.assertEqual(report['sources']['freesound']['metadata_rows'], 0)


if __name__ == '__main__':
    unittest.main()
