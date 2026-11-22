"""Live campaign timing lines from the benchmark artifact layouts in use."""
import json

from campaigns.common.timing_summary import format_rows, summarize
from campaigns.h100_assembly.plan import Task
from campaigns.h100_assembly.run import run_task


def test_results_rows_keep_labels_and_stage_milliseconds(tmp_path):
    path = tmp_path / 'transport.json'
    path.write_text(json.dumps(dict(mesh_size=0.1, results=[
        dict(order=9, config='float64/cutensor/default/magma', elements=9637, device_ms=2.68, solve_ms=1.92,
             build_ms=0.35, samples_seconds=dict(device=[0.0026, 0.0027]), reconstruction_ms=0.07,
             reconstruction_samples_ms=[0.07, 0.071])])))
    [(label, metrics)] = summarize(path)
    assert label == 'order=9 config=float64/cutensor/default/magma elements=9637'
    assert metrics['device_ms'] == 2.68 and metrics['solve_ms'] == 1.92
    assert 'reconstruction_ms(med)' not in metrics


def test_nested_seconds_and_jsonl_samples_become_milliseconds(tmp_path):
    poisson = tmp_path / 'poisson.json'
    poisson.write_text(json.dumps(dict(cases=[dict(order=6, cupy_rhs=dict(median_seconds=0.0007))])))
    assert summarize(poisson) == [('order=6', {'cupy_rhs.median_ms': 0.7})]
    adr = tmp_path / 'adr.jsonl'
    adr.write_text(json.dumps(dict(mode='environment')) + '\n' + json.dumps(dict(
        mode='adr', order=2, samples=[{'raw.kernel.device': 3e-5, 'raw.upload': 5e-4},
                                      {'raw.kernel.device': 5e-5, 'raw.upload': 6e-4}])) + '\n')
    [(label, metrics)] = summarize(adr)
    assert label == 'mode=adr order=2' and abs(metrics['samples.raw.kernel.device(med)_ms'] - 0.04) < 1e-12
    assert format_rows(summarize(adr))[0].startswith('  mode=adr order=2: ')


def test_passed_task_appends_live_timings(tmp_path):
    script = ("import json; from pathlib import Path; "
              "Path('{output}/data.json').write_text(json.dumps(dict(results=[dict(order=3, device_ms=1.5)])))")
    result = run_task(Task('timed', 'test', ('-c', script), 'data.json', 5), tmp_path, 'sources')
    assert result['status'] == 'passed'
    [record] = [json.loads(line) for line in (tmp_path / 'timings.jsonl').read_text().splitlines()]
    assert record['task'] == 'timed' and record['rows'] == [dict(label='order=3', ms={'device_ms': 1.5})]
