#!/usr/bin/env python3
"""Update the article from the accepted sharper family when the campaign ends."""

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from campaigns.common.paths import ARTICLE, ARTICLE_OUTPUTS, EQUILIBRIUM_REPO, FENICS_PYTHON, output_directory, record_file
CODE = Path(__file__).resolve().parent
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import traceback

directory = output_directory('equilibrium_dt080_eps0005_20261016') / 'article_update'
campaign = directory.parent
article = ARTICLE
figure_study = article / 'run_outputs/equilibrium_dt080_20261016'
python = FENICS_PYTHON
geometries = ['star', 'uglypacman', 'iter', 'horseshoe']
env = os.environ.copy()
env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
           NUMBA_NUM_THREADS='1', MPLBACKEND='Agg')
env['BIBINPUTS'] = str(article) + ':' + env.get('BIBINPUTS', '')


def write(name, data):
    path = directory / name
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def section_text(text, sources, comparisons):
    """Update only the figure caption; preserve the compact test description."""
    old = (r'Star and Ugly Pac-Man (top) use the sharper smoothing $\varepsilon/\delta=0.005$; '
           r'ITER and horseshoe (bottom) use $0.08$.')
    new = r'All four domains use the sharper smoothing $\varepsilon/\delta=0.005$.'
    if text.count(old) == 1:
        return text.replace(old, new)
    if text.count(new) == 1 and old not in text:
        return text
    raise ValueError('Equilibrium figure caption changed; review before updating it.')


def execute(label, command, cwd=article):
    print('START', label, flush=True)
    with (directory / (label + '.log')).open('w') as stream:
        worker_env = dict(env)
        if label == 'render':
            worker_env['HYBRIDGE_CAMPAIGN_OUT'] = str(figure_study)
        subprocess.run(command, cwd=cwd, env=worker_env, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)
    print('DONE', label, flush=True)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--check-template', action='store_true')
args = parser.parse_args()
if args.check_template:
    sources = json.loads((article / 'figures/equilibrium_dt080_p6.sources.json').read_text())['sources']
    section_text((article / 'sections/07_equilibria.tex').read_text(), sources, [])
    print('Compact figure-caption update verified; no article files changed.')
    raise SystemExit(0)

lock = (directory / 'launcher.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
write('driver.json', {'pid': os.getpid(), 'pgid': os.getpgrp(), 'started': time.time(),
                     'stop': f'kill -- -{os.getpgrp()}', 'scope': 'Wait for equilibrium campaign, update its figure and caption, preserve the compact section and rebuild article; no simulations'})
if (directory / 'completed.json').exists():
    raise SystemExit(0)
try:
    print('WAIT for equilibrium campaign completion', flush=True)
    while not (campaign / 'completed.json').exists() or (campaign / 'recovery_pending.json').exists():
        if (campaign / 'recovery_pending.json').exists():
            time.sleep(15)
            continue
        driver = json.loads((campaign / 'driver.json').read_text())
        if not (Path('/proc') / str(driver['pid'])).exists():
            raise RuntimeError('Equilibrium driver exited without its completion marker; article retained.')
        if (campaign / 'failed.json').exists():
            raise RuntimeError('Equilibrium campaign failed; article retained.')
        time.sleep(15)
    selected = json.loads((campaign / 'selected.json').read_text())
    missing = [geometry for geometry in geometries if geometry not in selected]
    if missing:
        raise RuntimeError('Article retained: target equilibria unavailable for ' + ', '.join(missing))
    sources, comparisons, selection = [], [], {}
    for geometry in geometries:
        record = selected[geometry]
        case = Path(record['checkpoint']).parent
        done = json.loads((case / 'completed.json').read_text())
        config = json.loads((case / 'config.json').read_text())
        if not (done['target_reached'] and done['archive_id'] == record['archive_id']
                and abs(done['distance'] - .8) <= 2e-7
                and config['degree'] == 6 and config['band']['relative_epsilon'] == .005):
            raise ValueError('Selected state is not an accepted target equilibrium: ' + geometry)
        selection[geometry] = str(case.relative_to(article))
        runtime = json.loads((case / 'manifest.json').read_text())['parallel_runtime']
        sources.append({**done, 'quadrature_degree': config['quadrature_degree'],
                        'mpi_ranks': runtime['mpi_ranks']})
        check = campaign / 'checks' / (geometry + '_quadrature.json')
        if check.exists():
            comparison = json.loads(check.read_text())
            if Path(comparison['fine']).resolve() == Path(record['checkpoint']).resolve():
                comparisons.append(comparison['comparisons'][-1])
    section = article / 'sections/07_equilibria.tex'
    text = section_text(section.read_text(), sources, comparisons)
    reference = directory / 'reference_before_update'
    paths = ['main.pdf', 'sections/07_equilibria.tex',
             'figures/equilibrium_dt080_p6.pdf', 'figures/equilibrium_dt080_p6.png',
             'figures/equilibrium_dt080_p6.sources.json', 'tables/equilibrium_dt080_p6_sources.tex',
             'run_outputs/equilibrium_dt080_20261016/article_selection.json']
    preserved = []
    for relative in paths:
        destination = reference / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(article / relative, destination)
        preserved.append({'path': relative, 'sha256': hashlib.sha256(destination.read_bytes()).hexdigest()})
    write('reference_manifest.json', preserved)
    selection_path = figure_study / 'article_selection.json'
    temporary = selection_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(selection, indent=2) + '\n')
    temporary.replace(selection_path)
    execute('render', [python, str(CODE.parents[1] / 'gc_eq_dt080' / 'prepare_article.py')])
    # Recheck after rendering so a concurrent prose edit is not overwritten.
    latest = section_text(section.read_text(), sources, comparisons)
    temporary = section.with_suffix('.tex.equilibrium.tmp')
    temporary.write_text(latest)
    temporary.replace(section)
    build = directory / 'build'
    build.mkdir(exist_ok=True)
    execute('latex', ['latexmk', '-pdf', '-interaction=nonstopmode', '-halt-on-error',
                     '-file-line-error', '-outdir=' + str(build), str(article / 'main.tex')])
    aux = (build / 'main.aux').read_text()
    label = re.search(r'\\newlabel\{fig:equilibria-dt080-p6\}\{\{([^}]+)\}\{([^}]+)\}', aux)
    if label is None:
        raise ValueError('Equilibrium figure missing from built article')
    temporary = article / 'main.equilibrium.tmp.pdf'
    shutil.copy2(build / 'main.pdf', temporary)
    temporary.replace(article / 'main.pdf')
    write('completed.json', {'finished': time.time(), 'status': 'complete',
          'selection': selection, 'figure': label[1], 'page': label[2],
          'quadrature_comparisons': len(comparisons), 'pdf': str(article / 'main.pdf'),
          'pdf_sha256': hashlib.sha256((article / 'main.pdf').read_bytes()).hexdigest()})
    print('UPDATED article figure', label[1], 'page', label[2], flush=True)
except BaseException as error:
    write('failed.json', {'error': repr(error), 'finished': time.time()})
    traceback.print_exc()
    raise
