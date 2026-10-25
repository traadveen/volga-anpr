"""Check evidence requirements and protection of existing output."""
from copy import deepcopy
from pathlib import Path
import math
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.plate_formats import apply_format_consistent_profile, apply_russian_profile


def fixture(text='A123BC77', plate_type='type1b'):
    return dict(image='fixture.jpg', detections=[dict(
        image='fixture.jpg', detection_index=0, status='ocr',
        detector_class=plate_type, plate_type=plate_type, raw_ocr=text,
        final_ocr=text, char_probs=[.99] * len(text), ocr_confidence=.99,
        observed_quad=[[0, 0], [200, 0], [200, 40], [0, 40]],
    )], predictions=[dict(image='fixture.jpg', detection_index=0,
        plate_num=text, plate_type=plate_type, confidence=.99)])


class FormatConsistencyTests(unittest.TestCase):
    def test_complete_white_formats_are_recovered_without_symbol_changes(self):
        for text in ('A123BC77', 'A123BC777'):
            with self.subTest(text=text):
                output = apply_format_consistent_profile(fixture(text))
                self.assertEqual(output['predictions'][0]['plate_num'], text)
                self.assertEqual(output['predictions'][0]['plate_type'], 'type1')
                self.assertEqual(output['detections'][0]['detector_class'], 'type1b')
                self.assertEqual(output['detections'][0]['raw_ocr'], text)

    def test_correct_yellow_formats_keep_the_same_predictions(self):
        for text in ('AB12377', 'AB123777'):
            data = fixture(text)
            self.assertEqual(apply_format_consistent_profile(data)['predictions'],
                             apply_russian_profile(data)['predictions'])

    def test_no_geometry_or_square_geometry_does_not_override_type(self):
        for quad in (None, [[0, 0], [80, 0], [80, 60], [0, 60]],
                     [[0, 0], [0, 0], [0, 0], [0, 0]]):
            data = fixture()
            data['detections'][0]['observed_quad'] = quad
            self.assertEqual(apply_format_consistent_profile(data)['format_type_corrections'], [])

    def test_low_confidence_character_does_not_override_type(self):
        for position in (0, 1, 4):
            data = fixture()
            data['detections'][0]['char_probs'][position] = .1
            self.assertEqual(apply_format_consistent_profile(data)['format_type_corrections'], [])

    def test_malformed_probabilities_do_not_override_type(self):
        for probabilities in (None, [], [.99], [math.nan] * 8, [True] * 8, [1.1] * 8):
            data = fixture()
            data['detections'][0]['char_probs'] = probabilities
            self.assertEqual(apply_format_consistent_profile(data)['format_type_corrections'], [])

    def test_invalid_sequence_confidence_does_not_override_type(self):
        for value in (None, True, math.nan, -1., 1.1):
            data = fixture()
            data['detections'][0]['ocr_confidence'] = value
            self.assertEqual(apply_format_consistent_profile(data)['format_type_corrections'], [])

    def test_incomplete_ocr_does_not_override_type(self):
        for text in ('A123#C77', 'A12BC77', 'A123BC7', 'A123BC7770'):
            self.assertEqual(apply_format_consistent_profile(fixture(text))['format_type_corrections'], [])

    def test_existing_white_and_other_outputs_are_not_reclassified(self):
        for kind in ('type1', 'type1a', 'other'):
            data = fixture(plate_type=kind)
            self.assertEqual(apply_format_consistent_profile(data)['predictions'], data['predictions'])

    def test_supplemental_partial_text_is_rejected_but_primary_partial_is_kept(self):
        data = fixture('A123BC7#', plate_type='type1')
        primary = apply_format_consistent_profile(data)
        self.assertEqual(primary['predictions'], data['predictions'])
        data['ordinary_reference_detection_indices'] = [0]
        self.assertEqual(apply_format_consistent_profile(data)['predictions'], [])

    def test_complete_supplemental_text_is_kept(self):
        data = fixture(plate_type='type1')
        data['ordinary_reference_detection_indices'] = [0]
        self.assertEqual(apply_format_consistent_profile(data)['predictions'], data['predictions'])

    def test_input_is_not_mutated_and_second_application_is_idempotent(self):
        data = fixture()
        original = deepcopy(data)
        output = apply_format_consistent_profile(data)
        self.assertEqual(data, original)
        self.assertEqual(apply_format_consistent_profile(output)['predictions'], output['predictions'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
