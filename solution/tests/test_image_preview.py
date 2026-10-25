"""Regression checks for what the photo viewer reports."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.image_preview import prediction_lines


class PreviewResultsTests(unittest.TestCase):
    def test_only_emitted_final_text_is_displayed(self):
        result = {
            'detections': [{'final_ocr': '', 'postprocessed_ocr': 'A123BC77'}],
            'predictions': [],
        }
        self.assertNotIn('A123BC77', prediction_lines(result))
        self.assertEqual(prediction_lines(result), 'Номер найден, но не распознан')

    def test_multiple_predictions_include_text_and_type(self):
        result = {'predictions': [
            dict(plate_num='A123BC77', plate_type='type1', confidence=.95),
            dict(plate_num='AA12377', plate_type='type1b', confidence=.87),
        ]}
        lines = prediction_lines(result).splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn('A123BC77   type1', lines[0])
        self.assertIn('AA12377   type1b', lines[1])

    def test_no_detections_is_reported(self):
        self.assertEqual(prediction_lines({'detections': [], 'predictions': []}), 'Номер не обнаружен')


if __name__ == '__main__':
    unittest.main()
