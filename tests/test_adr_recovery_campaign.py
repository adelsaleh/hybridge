"""Deadline, memory, qualification and reporting contracts for recovery runs."""
from argparse import Namespace
import json
from pathlib import Path

import pytest

from campaigns.adr_recovery.benchmark import choose_cells, choose_mesh
from campaigns.adr_recovery.run import plan, qualification_passed, report
from hybridge.mixed.postprocess.adr_recovery import recovery_working_bytes
from campaigns.common.slurm_gpu import count_requested_gpus


def test_full_degree_plan_and_hard_budget():
    args=Namespace(seconds=8700,repeats=3,min_nx=192,max_nx=512,chunk_limit=8192)
    result=plan(args)
    assert result['degrees']==list(range(2,10))
    assert result['gpus']==1 and result['verbosity']==3
    assert result['lifecycle']==['reuse','rebuild']
    args.seconds=10801
    with pytest.raises(ValueError): plan(args)


@pytest.mark.parametrize('degree', range(2,10))
def test_chunk_memory_budget(degree):
    free=100*1024**3
    n=choose_cells(degree,free)
    assert 512 <= n <= 8192
    assert recovery_working_bytes(degree,n) <= min(24*1024**3,.35*free)
    assert choose_cells(degree,1024)==0


def test_mesh_adapts_to_timing_budget():
    assert choose_mesh(8192,.05,1000)==512
    assert choose_mesh(8192,100.,1000)==192


@pytest.mark.parametrize('bad', ['failures','errors','skipped'])
def test_qualification_rejects_skips_and_failures(tmp_path,bad):
    p=tmp_path/'test.xml'
    p.write_text(f'<testsuites><testsuite tests="36" {bad}="1"/></testsuites>')
    assert not qualification_passed(p)[0]
    p.write_text('<testsuites><testsuite tests="36" failures="0" errors="0" skipped="0"/></testsuites>')
    assert qualification_passed(p)[0]


def test_report_preserves_incomplete_degrees(tmp_path):
    p=tmp_path/'p9';p.mkdir()
    (p/'result.json').write_text(json.dumps(dict(order=9,status='failed',error='deadline')))
    rows=report(tmp_path)
    assert rows[0]['status']=='failed'
    assert '| 9 | — | failed |' in (tmp_path/'REPORT.md').read_text()
    assert not (tmp_path/'COMPLETE.json').exists()


def test_shared_gpu_guard_counts_gpus_nodes_and_pending_arrays():
    queue='1|a|RUNNING|gpu:h200_nvl:2|2\n2_1|b|PENDING|gpu:1|1\n3|cpu|RUNNING|N/A|1\n'
    assert count_requested_gpus(queue)==5
    assert count_requested_gpus('1|a|RUNNING|gres/gpu:h200_nvl:1|1\n')==1
    assert count_requested_gpus('')==0
    with pytest.raises(ValueError): count_requested_gpus('4|bad|PENDING|gpu:unknown|1\n')
