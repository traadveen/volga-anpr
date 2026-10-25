"""Protect primary detections when adding ordinary-plate reference boxes."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.image_anpr import Detection
from src.image_anpr_f import OrdinaryFallbackDetector

def detection(box, kind='type1'):
    a,b,c,d = box
    return Detection(box, .9, class_name=kind,
        quad=((a,b),(c,b),(c,d),(a,d)), quad_source='fixture')

class Fake:
    def __init__(self, rows): self.rows = rows
    def detect(self, frame): return list(self.rows)

class OrdinaryFallbackTests(unittest.TestCase):
    def run_pair(self, primary, reference):
        detector = OrdinaryFallbackDetector(Fake(primary), Fake(reference))
        return detector.detect(np.zeros((300,500,3), np.uint8))

    def test_adds_missing_ordinary_plate(self):
        plate = detection((20,20,220,60))
        self.assertEqual(self.run_pair([], [plate]), [plate])

    def test_keeps_all_primary_detections(self):
        primary = [detection((20,20,220,60), 'type1b'), detection((250,20,350,90),'type1a')]
        self.assertEqual(self.run_pair(primary, []), primary)

    def test_does_not_duplicate_or_override_yellow_or_other(self):
        for kind in ('type1', 'type1a', 'type1b', 'other'):
            primary = detection((20,20,220,60), kind)
            self.assertEqual(self.run_pair([primary], [detection((21,20,221,60))]), [primary])

    def test_does_not_add_a_row_fragment_inside_square_plate(self):
        primary = detection((20,20,220,150), 'type1a')
        row = detection((25,30,215,60))
        self.assertEqual(self.run_pair([primary],[row]), [primary])

    def test_reference_square_and_yellow_are_not_added(self):
        self.assertEqual(self.run_pair([], [
            detection((20,20,100,90)), detection((20,150,220,190),'type1b')]), [])

if __name__ == '__main__':
    unittest.main(verbosity=2)
