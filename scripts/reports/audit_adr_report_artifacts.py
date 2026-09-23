#!/usr/bin/env python3
"""Static/data checks for ADR report artifacts; never compile TeX or run solvers."""
from __future__ import annotations
import ast
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import zipfile
import adr_native_completion as native
import make_adr_synthesis_section as synthesis

ROOT=native.ROOT
REPORT=synthesis.REPORT
FIGURES=synthesis.FIGURES.parent


def read(path):return json.loads(path.read_text())


def check_tex(entry, reader, exists, prefixes):
    """Check the reachable input/figure closure, environments, braces and labels."""
    files=[];figures=[];labels=[];references=[]
    def expand(value):
        for _ in range(4):
            for key,replacement in prefixes.items():value=value.replace(key,replacement)
        return value
    def visit(name):
        assert name not in files, f'Repeated/cyclic input: {name}'
        files.append(name);text=reader(name)
        text=re.sub(r'(?<!\\)%[^\n]*','',text)
        braces=0
        for match in re.finditer(r'(?<!\\)[{}]',text):
            braces += 1 if match[0]=='{' else -1
            assert braces>=0,(name,'closing brace')
        assert braces==0,(name,'unbalanced braces',braces)
        stack=[]
        for match in re.finditer(r'\\(begin|end)\{([^}]+)\}',text):
            if match[1]=='begin':stack.append(match[2])
            else:assert stack and stack.pop()==match[2],(name,'environment mismatch',match[2])
        assert not stack,(name,stack)
        labels.extend(re.findall(r'\\label\{([^}]+)\}',text))
        references.extend(re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',text))
        for target in re.findall(r'\\input\{([^}]+)\}',text):
            target=expand(target);assert exists(target),(name,target);visit(target)
        for target in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',text):
            target=expand(target);assert exists(target),(name,target);figures.append(target)
    visit(entry)
    duplicates=[key for key,count in Counter(labels).items() if count>1]
    assert not duplicates,duplicates
    assert set(references)<=set(labels),set(references)-set(labels)
    return {'tex_files':len(files),'figures':len(figures),'references':len(references)}


def main():
    for path in Path(__file__).parent.glob('*.py'):ast.parse(path.read_text(),filename=str(path))
    records=synthesis.normalize()
    groups={}
    for row in records:
        key=(row['suite'],row['geometry'],row['case'],row['n'],row['p'],row['dofs'])
        groups.setdefault(key,[]).append(row)
    with (REPORT/'synthesis/selected_results.csv').open() as stream:selected=list(csv.DictReader(stream))
    for row in selected:
        key=(row['suite'],row['geometry'],row['case'],int(row['n']),int(row['p']),int(row['dofs']))
        predicate={'asm':synthesis.is_asm,'bj':synthesis.is_bj,'local':synthesis.is_local,'amg':synthesis.is_amg,'native':synthesis.is_native,'lu':synthesis.is_lu}[row['group']]
        winner=synthesis.best(groups[key],predicate,row['metric'])
        if winner:
            assert row['status']=='passed' and row['method']==winner['method']
            assert math.isclose(float(row['time_ms']),winner[row['metric']],rel_tol=1e-12)
            if row['group']=='lu':
                assert not row['iterations'] and row['family']==synthesis.lu.alias(winner['threads'])
            else:assert math.isclose(float(row['iterations']),synthesis.iteration_count(winner,row['metric']),rel_tol=1e-12)
        else:assert row['status']=='NC' and not row['time_ms']
    assert len([r for r in selected if r['group']=='native'])==2*len(groups)==204
    assert len(groups)==102  # Suite memberships overlap; not distinct operators.
    # Check every displayed family independently, including NC and untested cells.
    expected=synthesis.overview_cells(records,'h')+synthesis.overview_cells(records,'p')
    with (REPORT/'synthesis/overview_cells.csv').open() as stream:cells=list(csv.DictReader(stream))
    assert len(cells)==len(expected)==180*len(synthesis.OVERVIEW_FAMILIES)
    for row,reference in zip(cells,expected):
        for key in ('status','family','configuration','sweep','metric'):
            assert row[key]==reference[key], (key,row,reference)
        for key in ('triangles','p','n','dofs'):
            assert int(row[key])==reference[key]
        if reference['status']=='passed':
            assert math.isclose(float(row['time_ms']),reference['time_ms'],rel_tol=1e-12)
            if row['family']=='lu':assert not row['iterations'] and reference['iterations'] is None
            else:assert math.isclose(float(row['iterations']),reference['iterations'],rel_tol=1e-12)
            if row['family'] in ('asm','bj'):
                assert int(row['pp_degree'])==reference['pp_degree']>0
        else:assert not row['time_ms'] and not row['iterations']
    c=native.read('coverage.json')
    assert len(c)==101 and sum(r['passed'] for r in c)==87 and all(r['attempted'] for r in c)
    # Preserve original nonnative measurements and baseline campaigns exactly.
    smooth=read(REPORT/'scaling.data.json')
    assert smooth['rows']==read(Path(smooth['local_campaign'])/'summary.json')
    for archive in (REPORT/'oscillatory/scaling.data.json',REPORT/'oscillatory/initial/results.data.json'):
        for campaign in read(archive)['campaigns']:
            if archive.parent.name=='initial':source=ROOT.parent/'hdgfem-gmres/run_logs'/campaign['name']
            else:source=native.CAMPAIGN.parent/'oscillatory/scaling'/campaign['name']
            assert campaign['rows']==read(source/'summary.json')
    for name in ('completion.json','coverage.json','validation.json','all_native_results.json','profiles.json','results.csv'):
        assert (REPORT/'native_completion'/name).read_bytes()==(native.CAMPAIGN/name).read_bytes()
    # Verify each standalone manuscript's dependency closure without running TeX.
    prefixes={r'\ADRResultsPath':str(REPORT),r'\ADROscPath':str(REPORT/'oscillatory'),
              r'\ADRResultsFigurePath':str(FIGURES),r'\ADRStressPath':str(REPORT/'oscillatory/initial'),
              r'\ADRStressFigurePath':str(native.CAMPAIGN.parent/'oscillatory/initial/figures')}
    tex={}
    for entry in ('main.tex','main_synthesis.tex','oscillatory/initial/preview.tex'):
        base=(REPORT/entry).parent
        tex[entry]=check_tex(
            str(REPORT/entry),lambda p:(Path(p) if Path(p).is_absolute() else base/p).read_text(),
            lambda p:(Path(p) if Path(p).is_absolute() else base/p).is_file(),prefixes)
    archives={}
    for path,entry,stress in ((FIGURES.parent/'adr_results_bundle.zip','main.tex',False),
                              (FIGURES.parent/'adr_results_synthesis_bundle.zip','main_synthesis.tex',False),
                              (FIGURES.parent/'oscillatory/initial/adr_oscillatory_bundle.zip','preview.tex',True)):
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None
            names=z.namelist();assert len(names)==len(set(names))
            mapping={r'\ADRResultsPath':'.',r'\ADROscPath':'oscillatory',r'\ADRResultsFigurePath':'figures',
                     r'\ADRStressPath':'.' if stress else 'oscillatory/initial',r'\ADRStressFigurePath':'figures' if stress else 'figures/oscillatory/initial'}
            normal=lambda s:str(Path(s))
            closure=check_tex(entry,lambda s:z.read(normal(s)).decode(),lambda s:normal(s) in names,mapping)
            archives[path.name]={'entries':len(names),**closure,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    result={'status':'passed','native_systems':101,'native_converged_systems':87,'selected_rows':len(selected),
            'lu_selected_rows':sum(r['group']=='lu' for r in selected),
            'lu_overview_cells':sum(r['family']=='lu' for r in cells),
            'native_selected_rows':204,'overview_cells_checked':len(cells),'original_campaign_samples_preserved':True,
            'manuscripts':tex,'portable_archives':archives,'tex_compiled':False,'page_count_verified':False,
            'note':'Static structure, figure dependencies, saved-sample rankings and archive integrity checked; TeX typesetting remains user-run.'}
    (REPORT/'artifact_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
