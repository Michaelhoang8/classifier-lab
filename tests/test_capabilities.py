import json
import csv
import io
from contextlib import closing
import sqlite3
import tempfile
import unittest
from pathlib import Path
from portfolio import capabilities as c, data_apps as d, ml_apps as ml
from portfolio.db import initialize, connect, SCHEMA
from portfolio.validation import ValidationError, NotFoundError

def enhanced(path, data, function):
    return c.enrich(path, data, function(data))

class AnalysisCapabilities(unittest.TestCase):

    def test_prediction_header_escaping_in_saved_model(self):
        model = {'features': ['quoted,feature'], 'weights': [1], 'means': [0], 'scales': [1], 'intercept': 0}
        r = c.predict({'model': model, 'csv': '"quoted,feature"\n0'})
        self.assertEqual(r['predictions'][0]['score'], 0.5)

    def test_model_import_roundtrip_and_report_wrapper(self):
        model = {'features': ['a'], 'weights': [2], 'means': [3], 'scales': [4], 'intercept': 1, 'threshold': 0.7}
        expected = c.predict({'model': model, 'csv': 'a\n3\n7'})
        for payload in (model, {'model': model, 'metrics': {}}):
            imported = c.load_model({'text': json.dumps(payload)})['model']
            self.assertEqual(c.predict({'model': imported, 'csv': 'a\n3\n7'}), expected)
            self.assertEqual(imported['threshold'], 0.7)

    def test_model_import_rejects_ambiguous_or_unsupported_files(self):
        model = {'features': ['x'], 'weights': [1], 'means': [0], 'scales': [1], 'intercept': 0}
        invalid = ['{}', '[]', 'null', '{bad', '{"model":{},"model":{}}', json.dumps({**model, 'algorithm': 'unknown'}), json.dumps({**model, 'features': ['']}), json.dumps({**model, 'scales': [0]}), json.dumps({**model, 'threshold': 2}), json.dumps({**model, 'weights': [float('inf')]}), ' ' * 200001]
        for raw in invalid:
            with self.subTest(raw=raw[:100]), self.assertRaises(ValidationError):
                c.load_model({'text': raw})

    def test_inference_reuses_scaler_and_aligns_reordered_columns(self):
        model = {'features': ['a', 'b'], 'weights': [1, 2], 'means': [10, 20], 'scales': [2, 4], 'intercept': 0}
        r = c.predict({'model': model, 'csv': 'b,extra,a\n20,ignore,10\n24,ignore,12\n'})
        self.assertEqual(r['predictions'][0]['score'], 0.5)
        self.assertAlmostEqual(r['predictions'][1]['score'], ml.sigmoid(3))
        self.assertEqual(r['predictions'][1]['predicted'], 1)

    def test_invalid_model_vectors_scales_and_features_rejected(self):
        model = {'features': ['x'], 'weights': [1], 'means': [0], 'scales': [1], 'intercept': 0}
        for patch in ({'scales': [0]}, {'weights': []}, {'means': [float('nan')]}, {'features': ['x', 'x']}):
            with self.subTest(patch=patch), self.assertRaises(ValidationError):
                c.predict({'model': {**model, **patch}, 'csv': 'x\n1'})
        with self.assertRaises(ValidationError):
            c.predict({'model': model, 'csv': 'missing\n1'})

    def test_training_model_predicts_heldout_rows_identically(self):
        csv = 'x,label\n' + '\n'.join((f'{i},{int(i >= 0)}' for i in range(-8, 8)))
        fitted = ml.train({'csv': csv})
        r = c.predict({'model': fitted['model'], 'csv': csv})
        by_record = {p['record']: p['score'] for p in r['predictions']}
        for row in fitted['predictions']:
            self.assertAlmostEqual(by_record[row['record']], row['score'])
if __name__ == '__main__':
    unittest.main()
