#!/usr/bin/env python3
"""Render the native completion census and diagnostic tables from saved data."""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import adr_native_completion as native
from adr_report_labels import PMG, PMG_TEX, polynomial_label, prose

REPORT=native.ROOT/'docs/research/solver_studies/adr_scaling_2026_09_17'
OUT=REPORT/'native_completion'
ROW=r'\\'


def main():
    OUT.mkdir(exist_ok=True)
    for name in ('completion.json','coverage.json','validation.json','all_native_results.json','profiles.json','results.csv'):
        shutil.copy2(native.CAMPAIGN/name,OUT/name)
    coverage=native.read('coverage.json')
    lookup={r['system_id']:r for r in coverage}
    lines=[r'\begin{tabular}{lrr}',r'\toprule',r'Study & Systems attempted & $p$MG--AMG converged \\',r'\midrule']
    for suite,label in (('smooth','Smooth three-regime scaling'),('focused_scaling','Focused mesh/degree scaling'),
                        ('focused_endpoints','Focused million-unknown endpoints'),('oscillatory','Oscillatory three-class scaling'),
                        ('initial','Initial oscillatory stress controls')):
        group=[r for r in coverage if any(o['suite']==suite for o in r['origins'])]
        lines.append(f"{label} & {len(group)} & {sum(r['passed'] for r in group)} "+ROW)
    lines += [r'\midrule',r'Distinct matrix/right-hand-side pairs & 101 & 87 \\',r'\bottomrule',r'\end{tabular}']
    (OUT/'coverage_table.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\begin{tabular}{llrrrr}',r'\toprule',r'Problem & Geometry & $p$ & Apply (ms) & Calls & GPU (\%) \\',r'\midrule']
    names={'advection_dominated':'Smooth transport','anisotropic':'Smooth anisotropy','cellular7_directional':'Oscillatory directional'}
    for profile in native.read('profiles.json'):
        if profile['status']!='passed':continue
        source=lookup[profile['system_id']];d=profile['record']
        pre=next(r for r in d['gmres_profile']['operations'] if r['category']=='preconditioner')
        fraction=100*pre['gpu_time_ms']/d['gmres_profile']['total_gpu_operation_ms']
        lines.append(f"{names[source['case']]} & {source['geometry'].title()} & {source['p']} & {d['complete_preconditioner']['mean_ms']:.3f} & {pre['count']} & {fraction:.1f} "+ROW)
    lines += [r'\bottomrule',r'\end{tabular}'];(OUT/'application_table.tex').write_text('\n'.join(lines)+'\n')
    (OUT/'summary.tex').write_text(r'''\subsection{Native hierarchy: complete coverage and application cost}
\label{sec:adr-native-completion}
Native $hp$-BSR has now been attempted on all 101 distinct matrix/right-hand-side
pairs. The extension adds 126 policy/system measurements: 91 converge and
35 fail numerically, with no execution errors. Including earlier accepted
measurements, native convergence covers 87 systems. Table~\ref{tab:adr-native-coverage}
separates overlapping studies from the distinct-system total.

The standard policy transfers directly from degree $p$ to degree zero, with
order-two Chebyshev smoothing and one pre/post sweep. The robust policy
halves the degree through intermediate levels, uses order-four Chebyshev
smoothing and two pre/post sweeps. Both build a hierarchy for $(A+A^T)/2$
in modal coordinates, use a scalar AMG coarse correction, and apply outer
GMRES to the unchanged full ADR operator. Standard is usually cheaper when
both converge; fewer robust-policy iterations do not imply a shorter solve.

\begin{table}[!htbp]
 \centering\small
 \input{\ADRResultsPath/native_completion/coverage_table.tex}
 \caption{Native coverage. Study rows overlap; the last row counts each
 matrix/right-hand-side pair once. Convergence means at least one native
 policy passes the physical-residual contract.}
 \label{tab:adr-native-coverage}
\end{table}

Native converges on every smooth and focused system and all 16 directional
oscillatory systems. Its 14 unresolved systems comprise 13 weak-isotropic
oscillatory systems and the initial 22,825-triangle weak-anisotropic annulus.
ASM+PP therefore retains a robustness advantage on those difficult controls,
while native provides a faster alternative where the transferred block-AMG
hierarchies fail on directional anisotropy. Numerical failure here means
failure under the tested policies and stopping contract, not mathematical
impossibility of convergence.

\begin{table}[!htbp]
 \centering\small
 \input{\ADRResultsPath/native_completion/application_table.tex}
 \caption{Separate standard-policy native profiles near one million trace
 unknowns. Apply is the mean of ten warmed complete applications; calls and
 GPU share come from an independently instrumented solve. These profiles do
 not enter timing rankings. Failed cases receive no performance profile.}
 \label{tab:adr-native-applications}
\end{table}

\begin{table}[!htbp]
 \centering\small
 \input{\ADRResultsPath/native_completion/directional_tuned_table.tex}
 \caption{Tuned directional endpoints at $p=6$ near one million unknowns.
 Setup/fresh/reused times are seconds; Apply is a separately profiled
 complete application in milliseconds. Both native policies are compared with the retuned ASM+PP
 correction. This supplements the fixed-degree ASM scaling curves.}
 \label{tab:adr-native-tuned}
\end{table}

At the directional endpoints, standard native and tuned ASM+PP use the same
608 preconditioner applications on the square; on the annulus native uses
380 versus 152 for ASM+PP. Native's timing advantage thus comes from its much
cheaper complete correction, not a uniformly smaller outer iteration count.
Its multilevel correction and ASM's polynomial/patch composition perform
different mathematical work. These measurements establish the advantage of
the tested algorithms and implementations; they do not establish global
implementation optimality or isolate a purely mathematical cause.

Fresh time remains the median of paired setup-plus-first-solve measurements;
reused time is the mean second solve from zero across independent setups.
Reusing setup does not mean reusing the previous solution as an initial guess.
Two measured setups are retained for the initial degree-four controls and
three elsewhere. Profiles are separate. The new accepted runs have worst
physical relative residual $9.73\times10^{-12}$ and worst recorded relative
trace difference from the peer solve $8.29\times10^{-9}$; the latter is a
diagnostic rather than an additional convergence criterion.
''')
    (OUT/'summary.tex').write_text(prose((OUT/'summary.tex').read_text()))
    # Compact census of every initial system, including coefficient/source controls.
    names={'advection_dominated':'Smooth transport','oscillatory_rhs':'Oscillatory source',
           'cellular4':'Cellular, four waves','cellular7':'Cellular, seven waves',
           'cellular7_low':'Cellular, reduced diffusion','cellular7_weak':'Oscillatory low isotropic',
           'cellular7_anisotropic':'Oscillatory weak anisotropic'}
    rows=native.rows_for_suite('initial');groups={}
    (REPORT/'oscillatory/initial/native_results.json').write_text(json.dumps(rows,indent=2)+'\n')
    for row in rows:groups.setdefault(row['system_id'],[]).append(row)
    lines=[r'\begin{tabular}{llrrrrl}',r'\toprule',r'Geometry & Problem & $N_T$ & $p$ & Fresh & Reused & Policy / iter. \\',r' & & & & \multicolumn{2}{c}{time (ms)} & \\',r'\midrule']
    for group in sorted(groups.values(),key=lambda g:(g[0]['geometry'],g[0]['triangles'],g[0]['p'],g[0]['case'])):
        r=group[0];fresh=native.best_native(group,'fresh');hot=native.best_native(group,'hot')
        values=[f"{native.metric(v,m):.1f}" if v else 'NC' for v,m in ((fresh,'fresh'),(hot,'hot'))]
        policy=(('S' if hot['policy']=='standard' else 'R')+f" / {native.metric(hot,'iterations'):.0f}") if hot else 'S,R / NC'
        lines.append(f"{r['geometry'].title()} & {names[r['case']]} & {r['triangles']:,} & {r['p']} & "+' & '.join(values)+f' & {policy} '+ROW)
    lines += [r'\bottomrule',r'\end{tabular}'];(REPORT/'oscillatory/initial/native_completion_table.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'native_systems':len(coverage),'native_passed':sum(r['passed'] for r in coverage),'initial_systems':len(groups)}))


if __name__=='__main__':main()
