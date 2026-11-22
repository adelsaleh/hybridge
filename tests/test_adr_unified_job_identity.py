"""run_job's resume identity ignores checkout paths, and accepts records of older copies."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from campaigns.adr_unified.adr_unified_plan import fingerprint  # noqa: E402
from campaigns.adr_unified.run_adr_unified_campaign import job_identity, specification_matches  # noqa: E402

SYSTEM = {'id': 'L7_p6_x', 'mesh': {'generator': 'stress', 'target_triangles': 1904762}, 'problem': {'p': 6}}


def _spec(root):
    return dict(stage='mesh', system=SYSTEM, branch_root=f'{root}/vendor/adr_gmres',
                inventory_directory=f'{root}/run_configs/adr_unified_l5', mesh_directory='/scratch/out/meshes/L7_p6_x')


def test_identity_ignores_checkout_paths_but_not_the_job():
    assert job_identity(_spec('/copies/v3')) == job_identity(_spec('/copies/v5'))
    other = _spec('/copies/v3')
    other['system'] = {**SYSTEM, 'mesh': {**SYSTEM['mesh'], 'target_triangles': 1}}
    assert job_identity(other) != job_identity(_spec('/copies/v3'))


def test_current_and_legacy_records_match(tmp_path):
    spec = _spec('/copies/v5')
    assert specification_matches({'specification_sha256': job_identity(spec)}, spec, tmp_path)
    assert specification_matches({'specification_sha256': fingerprint(spec)}, spec, tmp_path)  # pre-2026-10-28 hash
    assert not specification_matches({'specification_sha256': 'nope'}, spec, tmp_path)


def test_record_built_by_another_checkout_matches_through_its_attempt_spec(tmp_path):
    built = _spec('/copies/v4')
    (tmp_path/'attempt_1').mkdir()
    (tmp_path/'attempt_1'/'spec.json').write_text(json.dumps({**built, 'result': str(tmp_path/'attempt_1/result.json')}))
    prior = {'specification_sha256': fingerprint(built), 'status': 'passed'}
    assert specification_matches(prior, _spec('/copies/v5'), tmp_path)
    changed = _spec('/copies/v5')
    changed['mesh_directory'] = '/scratch/elsewhere'
    assert not specification_matches(prior, changed, tmp_path)
    # A record whose hash does not even match its own attempt spec is not trusted.
    assert not specification_matches({'specification_sha256': 'tampered'}, _spec('/copies/v5'), tmp_path)
