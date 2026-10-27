"""Guard the selected dictionary release against a mixed prediction vintage."""

import csv
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TrainOnlyReleaseTest(unittest.TestCase):
    def test_current_dictionary_scores_bind_to_published_predictions(self):
        selection = json.loads((ROOT / 'results/gazetteer_train_only_v1_1/selection.json').read_text())
        with (ROOT / selection['primary_counts']['path']).open(newline='') as f:
            scores = list(csv.DictReader(f))
        with (ROOT / 'predictions/manifest.csv').open(newline='') as f:
            published = {(r['dataset'], r['run']): r for r in csv.DictReader(f) if r['group'] == 'gazetteer'}

        self.assertEqual(len(scores), 14)
        self.assertEqual({r['arm'] for r in scores}, {'train_only'})
        self.assertEqual(len(published), 6)
        for row in scores:
            key = (row['dataset'], row['variant'])
            if key in published:
                self.assertEqual(int(row['prediction_rows']), int(published[key]['rows']))
        for entry in selection['published_predictions']:
            row = published[entry['dataset'], entry['variant']]
            self.assertEqual(entry['path'], row['release_path'])
            self.assertEqual(entry['sha256'], row['release_sha256'])
            self.assertEqual(entry['rows'], int(row['rows']))


if __name__ == '__main__':
    unittest.main()
