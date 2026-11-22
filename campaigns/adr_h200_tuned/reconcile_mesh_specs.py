"""Reconcile mesh job records produced by another checkout with the copy that resumes them.

The mesh job spec carries two checkout paths (branch_root, inventory_directory); a mesh built
from another checkout has a different specification_sha256 although the mesh file is the same
(deterministic gmsh). This rewrites result.json's specification_sha256 to the hash the target
checkout computes, after verifying the method on a record of the target checkout itself.

usage: python3 -I reconcile_mesh_specs.py OUTPUT_DIR TARGET_COPY [--apply]
"""
import json
import sys
from pathlib import Path

output, target = Path(sys.argv[1]), Path(sys.argv[2]).resolve()
apply = '--apply' in sys.argv
sys.path.insert(0, str(target))
from campaigns.adr_unified.adr_unified_plan import fingerprint  # noqa: E402

target_paths = dict(branch_root=str(target/'vendor/adr_gmres'), inventory_directory=str(target/'run_configs/adr_unified_l5'))
verified = 0
changed = []
for job in sorted(output.glob('jobs/mesh_*')):
    final = job/'result.json'
    if not final.exists():
        continue
    record = json.loads(final.read_text())
    attempts = sorted(job.glob('attempt_*/spec.json'))
    if not attempts:
        continue
    spec = json.loads(attempts[-1].read_text())
    spec.pop('result', None)
    own = fingerprint(spec)
    # Method check: the record's hash must equal the fingerprint of its own attempt's spec.
    if own != record.get('specification_sha256'):
        print(f'{job.name}: record hash does not match its attempt spec; left alone')
        continue
    expected = fingerprint({**spec, **target_paths})
    if expected == record['specification_sha256']:
        verified += 1
        continue
    changed.append(job.name)
    print(f"{job.name}: {spec['branch_root']} -> target; {record['specification_sha256'][:12]} -> {expected[:12]}")
    if apply:
        backup = job/'result.before_spec_reconcile.json'
        if not backup.exists():
            backup.write_text(json.dumps(record, indent=2)+'\n')
        record['specification_sha256'] = expected
        record['specification_note'] = (f"mesh built by {spec['branch_root']}; specification hash reconciled to "
                                        f"{target_paths['branch_root']} (same deterministic mesh, mesh_sha256 unchanged)")
        tmp = job/'result.json.tmp'
        tmp.write_text(json.dumps(record, indent=2)+'\n')
        tmp.replace(final)
print(f'already matching: {verified}; reconciled: {len(changed)}{"" if apply else " (dry run)"}')
