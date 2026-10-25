"""Regression contracts with fixture predictions; no held-out data or real models."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get('AUDIT_TEST_PYTHON', sys.executable)
sys.path.insert(0, str(ROOT))
from tools import output_transaction as publication
from tools.validate_output import validate_rows, COLUMNS
from tools import run_final_submission as runner


def row(text='A123BC77'):
    return dict(image='frame.png', plate_num=text, plate_type='type1', confidence=.99)


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name) / 'result.csv'

    def publish(self, text='A123BC77'):
        publication.publish(self.output, [row(text)], [{'image':'frame.png'}], [], {'name':'fixture'})

    def contents(self):
        return {p.name:p.read_bytes() for p in publication.paths(self.output) if p.exists()}

    def test_success_and_tamper_detection(self):
        self.publish()
        self.assertEqual(publication.verify_publication(self.output)['name'], 'fixture')
        self.output.with_suffix('.trace.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            publication.verify_publication(self.output)

    def test_failure_at_every_commit_position_restores_previous_generation(self):
        for failed_target in publication.paths(self.output):
            with self.subTest(target=failed_target.name):
                self.publish()
                before = self.contents()
                real_replace = os.replace
                def failing(source, target):
                    if Path(target) == failed_target and not Path(source).name.startswith('restore-'):
                        raise PermissionError('fixture locked destination')
                    return real_replace(source, target)
                with patch.object(publication.os, 'replace', side_effect=failing):
                    with self.assertRaises(PermissionError):
                        self.publish('B456CT77')
                self.assertEqual(self.contents(), before)
                publication.verify_publication(self.output)
                self.assertFalse(self.output.with_suffix('.publishing.json').exists())

    def test_first_publication_failure_leaves_no_partial_result(self):
        real_replace = os.replace
        def failing(source, target):
            if Path(target) == self.output.with_suffix('.errors.json'):
                raise PermissionError('fixture')
            return real_replace(source, target)
        with patch.object(publication.os, 'replace', side_effect=failing):
            with self.assertRaises(PermissionError):self.publish()
        self.assertEqual(self.contents(), {})

    def test_invalid_predictions_leave_previous_set_unchanged(self):
        self.publish(); before = self.contents()
        with self.assertRaises(ValueError):self.publish('A123GC77')
        self.assertEqual(self.contents(), before)

    def test_hard_crash_is_detected_and_recovery_restores_previous_set(self):
        self.publish(); before = self.contents()
        script = '''
import sys, os
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from tools import output_transaction as p
output = Path(sys.argv[2])
real_replace = os.replace
def crash(source, target):
    real_replace(source, target)
    if Path(target) == output.with_suffix('.trace.json'): os._exit(71)
p.os.replace = crash
p.publish(output, [dict(image='frame.png',plate_num='B456CT77',plate_type='type1',confidence=.9)],
          [dict(image='frame.png')], [], dict(name='new'))
'''
        completed = subprocess.run([PYTHON, '-c', script, str(ROOT), str(self.output)], timeout=30)
        self.assertEqual(completed.returncode, 71)
        with self.assertRaisesRegex(ValueError, 'Interrupted'):
            publication.verify_publication(self.output)
        with self.assertRaisesRegex(ValueError, 'Interrupted'):
            self.publish()
        publication.recover(self.output)
        self.assertEqual(self.contents(), before)
        publication.verify_publication(self.output)

    def test_concurrent_writer_is_refused(self):
        self.publish(); before = self.contents()
        script = '''
import sys
sys.path.insert(0, sys.argv[1])
from tools.output_transaction import publish
try: publish(sys.argv[2], [], [], [], {})
except OSError: sys.exit(23)
sys.exit(99)
'''
        with publication.exclusive(self.output):
            result = subprocess.run([PYTHON, '-c', script, str(ROOT), str(self.output)], timeout=30)
        self.assertEqual(result.returncode, 23)
        self.assertEqual(self.contents(), before)

    def test_all_unreadable_input_preserves_entire_successful_set(self):
        self.publish(); before = self.contents()
        image = self.output.parent/'broken.jpg'; image.write_bytes(b'bad')
        config = self.output.parent/'config.json'; config.write_text('{}')
        errors = [dict(image=image.name, status='input_error')]
        with patch.object(runner,'runtime_config',return_value=config), \
             patch.object(runner,'load_runtime_config',return_value={}), \
             patch.object(runner,'recognize',return_value=([],errors,errors,{})):
            with self.assertRaises(SystemExit):
                runner.main(['--input',str(image),'--output',str(self.output)])
        self.assertEqual(self.contents(), before)
        self.assertTrue(self.output.with_suffix('.failed-run.json').exists())


class InferenceContractTests(unittest.TestCase):
    def test_ocr_six_in_letter_slot_becomes_unknown(self):
        from src.image_anpr import current_postprocess
        self.assertEqual(current_postprocess('A1236C77',[.99]*8), 'A123#C77')

    def test_every_single_character_substitution_obeys_output_contract(self):
        from src.image_anpr import current_postprocess
        for sample in ('A123BC77','A123BC777'):
            for index in range(len(sample)):
                for char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789#\u0661\u00b2':
                    text = sample[:index]+char+sample[index+1:]
                    value = current_postprocess(text,[.99]*len(text))
                    self.assertTrue(validate_rows([row(value)])['valid'], (text,value))

    def test_valid_legacy_texts_are_unchanged(self):
        from src.image_anpr import current_postprocess
        for text in ('A123BC77','B456CT777'):
            self.assertEqual(current_postprocess(text,[.99]*len(text)),text)

    def test_other_remains_other_in_every_router_mode(self):
        import numpy as np
        from src.image_anpr import Detection
        from src.image_anpr_f import RouterF
        for mode in ('geometry','detector','combined'):
            for shape in ((30,100,3),(100,100,3)):
                detection = Detection((0,0,100,30),.99,class_name='other')
                route = RouterF(mode).classify(np.zeros(shape,np.uint8),detection)
                self.assertEqual(route.plate_type,'other')

    def test_g_rejects_other_without_ocr_or_fallback(self):
        import numpy as np
        from src.image_anpr import Detection
        from src.image_anpr_g import ImagePipelineG
        pipeline = ImagePipelineG(None, None)
        with patch.object(pipeline.fallback,'process',side_effect=AssertionError('fallback invoked')):
            result = pipeline.process(np.zeros((60,200,3),np.uint8),image_name='frame.png',
                detections=[Detection((10,10,150,40),.99,class_name='other')])
        self.assertEqual(result['predictions'], [])
        self.assertEqual(result['detections'][0]['reason'],'explicit_detector_class_other')

    def test_g_postprocess_and_csv_contract_together(self):
        import numpy as np
        from src.image_anpr import Detection
        from src.image_anpr_g import ImagePipelineG
        class OCR:
            def recognize_candidates(self, arrays):return [('A1236C77',.99,[.99]*8,-1.)]
            def synchronize(self):pass
        pipe = ImagePipelineG(None,OCR())
        detection = Detection((10,10,150,40),.99,class_name='type1',
            quad=((10,10),(150,10),(150,40),(10,40)),quad_source='fixture')
        prep = (np.zeros((32,160)),None,dict(structural_confidence=1,method='fixture'))
        with patch('src.image_anpr_g.prepare_g',return_value=prep):
            result=pipe.process(np.zeros((60,200,3),np.uint8),image_name='frame.png',detections=[detection])
        rows=[{k:p[k] for k in COLUMNS} for p in result['predictions']]
        self.assertEqual(rows[0]['plate_num'],'A123#C77')
        self.assertTrue(validate_rows(rows)['valid'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
