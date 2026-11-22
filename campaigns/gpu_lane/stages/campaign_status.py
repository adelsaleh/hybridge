"""Apply the unified campaign runner's status rule to a rerun worker result:
    campaign_status.py RESULT_JSON SPEC_JSON
Run from the frozen campaign copy. The worker reports any failed verification as
numerical_failure; the runner relabels a finite stall (iteration cap, residual above target) as
not_converged (campaigns.common.reporting.convergence_status). Stores the label as
``campaign_status`` in the result and prints it ("missing" when there is no result).
"""
import json
import sys
from pathlib import Path

from campaigns.common.reporting import convergence_status

path = Path(sys.argv[1])
if not path.exists():
    print('missing')
    raise SystemExit
result = json.loads(path.read_text())
result['campaign_status'] = convergence_status(result, json.loads(Path(sys.argv[2]).read_text()))
path.write_text(json.dumps(result, indent=1))
print(result['campaign_status'])
