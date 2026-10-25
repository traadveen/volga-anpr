import json
from types import SimpleNamespace
import pytest
from tools import benchmark


def test_benchmark_counts_images_and_includes_startup(tmp_path,monkeypatch):
    def run(command,**kwargs):
        output=tmp_path/'benchmark'/'round00.csv'
        output.with_suffix('.manifest.json').write_text(json.dumps(dict(skipped_images=0,ground_truth_input=False,
            processed_images=2,initialization_ms=900,performance={},prediction_rows=8)))
        output.with_suffix('.trace.json').write_text(json.dumps(dict(records=[
            dict(cached_detections=False,read_decode_ms=1,submission_pipeline_ms=9),
            dict(cached_detections=False,read_decode_ms=2,submission_pipeline_ms=18)])))
        return SimpleNamespace(returncode=0,stdout='',stderr='')
    times=iter([0.,2.]);monkeypatch.setattr(benchmark.time,'perf_counter',lambda:next(times))
    monkeypatch.setattr(benchmark.subprocess,'run',run)
    out=tmp_path/'benchmark.json'
    assert benchmark.main(['--input',str(tmp_path),'--output',str(out)])==0
    result=json.loads(out.read_text())
    assert result['rounds'][0]['entire_CLI_ms_per_image']==1000
    assert result['individual_read_decode_pipeline_ms']['mean']==15
    assert result['reference_i5_7600_GTX1050Ti_verified'] is False


def test_benchmark_refuses_overwriting_results(tmp_path):
    out=tmp_path/'result.json';out.write_text('preserved')
    with pytest.raises(SystemExit):benchmark.main(['--input',str(tmp_path),'--output',str(out)])
    assert out.read_text()=='preserved'
