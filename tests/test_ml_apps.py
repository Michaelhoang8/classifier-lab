import json
import math
import statistics
import unittest
from portfolio import ml_apps as m
from portfolio.validation import ValidationError

def training_csv():
    return 'signal,constant,label\n' + '\n'.join((f'{i},7,{int(i > 0)}' for i in range(-20, 21) if i))

class MLTests(unittest.TestCase):

    def test_auc_known_rankings_and_ties(self):
        self.assertEqual(m.auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]), 0.75)
        self.assertEqual(m.auc([0, 1], [0.5, 0.5]), 0.5)
        self.assertEqual(m.auc([0, 1], [0.9, 0.1]), 0)
        self.assertIsNone(m.auc([1, 1], [0.1, 0.9]))

    def test_confusion_and_probability_losses(self):
        r = m.metrics([0, 1, 1, 0], [0.1, 0.8, 0.4, 0.7])
        self.assertEqual(r['confusion'], {'tp': 1, 'fp': 1, 'tn': 1, 'fn': 1})
        self.assertEqual((r['precision'], r['recall'], r['f1']), (0.5, 0.5, 0.5))
        self.assertAlmostEqual(r['brier'], (0.01 + 0.04 + 0.36 + 0.49) / 4)
        self.assertEqual(m.metrics([0, 1], [0, 1])['log_loss'], 0)

    def test_undefined_metrics_and_threshold_boundary(self):
        r = m.metrics([0, 0], [0, 0.5], 1)
        self.assertIsNone(r['precision'])
        self.assertIsNone(r['recall'])
        self.assertIsNone(r['f1'])
        self.assertEqual(m.metrics([1], [0.5], 0.5)['confusion']['tp'], 1)

    def test_training_improves_loss_and_is_reproducible(self):
        a = m.train({'csv': training_csv()})
        b = m.train({'csv': training_csv()})
        self.assertEqual(a, b)
        self.assertLess(a['loss_history'][-1]['log_loss'], a['loss_history'][0]['log_loss'])
        self.assertGreater(a['metrics']['accuracy'], a['baseline_accuracy'])
        self.assertEqual(a['model']['weights'][1], 0)

    def test_scaling_fits_train_rows_only_and_split_disjoint(self):
        r = m.train({'csv': training_csv(), 'seed': 7})
        self.assertFalse(set(r['train_records']) & set(r['test_records']))
        self.assertEqual(len(r['train_records']) + len(r['test_records']), 40)
        rows = training_csv().splitlines()
        expected = statistics.mean((float(rows[i - 1].split(',')[0]) for i in r['train_records']))
        self.assertEqual(r['features'][0]['mean'], expected)

    def test_exported_model_reproduces_test_predictions(self):
        r = m.train({'csv': training_csv()})
        model = r['model']
        rows = training_csv().splitlines()
        for item in r['predictions']:
            x = [float(v) for v in rows[item['record'] - 1].split(',')[:-1]]
            score = m.sigmoid(model['intercept'] + sum((w * (v - mean) / scale for w, v, mean, scale in zip(model['weights'], x, model['means'], model['scales']))))
            self.assertAlmostEqual(score, item['score'])

    def test_training_rejects_duplicates_and_bad_classes(self):
        for text in [training_csv() + '\n1,7,1', 'x,label\n' + '\n'.join((f'{i},0' for i in range(12)))]:
            with self.assertRaises(ValidationError):
                m.train({'csv': text})

    def test_training_duplicate_errors_identify_records_and_label_conflicts(self):
        for conflicting, duplicate_index in ((False, 1), (True, 7)):
            rows = [[str(index + 10000), str(int(index >= 4))] for index in range(8)]
            rows[duplicate_index][0] = '10000.0'
            source = 'feature,label\n' + '\n'.join((','.join(row) for row in rows))
            with self.subTest(conflicting=conflicting), self.assertRaises(ValidationError) as caught:
                m.train({'csv': source})
            message = str(caught.exception)
            self.assertIn(f'records 2 and {duplicate_index + 2}', message)
            self.assertIn('conflicting target labels' if conflicting else 'matching target labels', message)
            self.assertIn('after numeric parsing', message)
            self.assertNotIn('10000', message)
