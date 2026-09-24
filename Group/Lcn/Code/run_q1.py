"""Reproduce or audit all question-one deliverables from the repository root."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

from q3.data import load_data
from q3.physics import Terrain
from q1.solver import solve
from q1.verify import verify


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='Group/Lcn/Results/Q1')
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[3]
    out=Path(args.output)
    if not out.is_absolute():
        out=root/out
    started=datetime.now(timezone.utc).isoformat()
    tick=time.monotonic()
    data=load_data(root)
    terrain=Terrain(data['dem_path'])
    if args.verify_only:
        base=json.loads((out/'solution.json').read_text(encoding='utf-8'))
        sensitivity=json.loads((out/'sensitivity.json').read_text(encoding='utf-8'))
        priorities=json.loads((out/'priority_comparison.json').read_text(encoding='utf-8'))
        manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
        for item in manifest['mathematics']['inputs']+manifest['outputs']:
            if sha(root/item['path'])!=item['sha256']:
                raise AssertionError(f"Recorded hash mismatch: {item['path']}")
    else:
        sensitivity=[solve(data,terrain,rho) for rho in (.10,.15,.20,.25,.30)]
        base=sensitivity[2]
        priorities=[base,solve(data,terrain,.2,'energy_first'),solve(data,terrain,.2,'time_first')]
    validation={}
    for label,result in [('baseline',base)]+[(f"reserve_{r['reserve']:.2f}",r) for r in sensitivity]+[(r['priority'],r) for r in priorities]:
        audit=verify(result,data,terrain)
        if not audit['passed']:
            raise AssertionError(audit)
        validation[label]=audit
    if args.verify_only:
        print(json.dumps({'passed':True,'scenarios_checked':len(validation),'checks':sum(v['checks'] for v in validation.values())}))
        return
    with tempfile.TemporaryDirectory(prefix='q1-matplotlib-') as cache:
        os.environ.setdefault('MPLCONFIGDIR',cache)
        from q1.report import export, write_json
        export(out,base,sensitivity,priorities,validation)
    code=Path(__file__).parent
    inputs=[Path(v) for v in data['source_paths'].values()]
    inputs+=[Path(__file__),code/'q3/data.py',code/'q3/physics.py',code/'requirements.txt']
    inputs+=sorted((code/'q1').glob('*.py'))
    reference=root/'Temp/D题第一问论文正文.md'
    manifest={
        'schema_version':1,'claim_id':'q1-exhaustive-single-destination-batching',
        'repository':{'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
                      'dirty':bool(subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True))},
        'command':'Group/Lcn/.venv/python.exe -B Group/Lcn/Code/run_q1.py'+(' --output '+args.output if args.output!='Group/Lcn/Results/Q1' else ''),
        'environment':{'software':[f'Python {platform.python_version()}']+[f'{p} {importlib.metadata.version(p)}' for p in ['numpy','scipy','rasterio','matplotlib','openpyxl']],
                       'hardware':platform.platform()+'; '+platform.machine()+'; '+platform.processor()},
        'mathematics':{'assertion_tested':'Exhaustive count-state DP agrees with independently enumerated lexicographic MILP at each of 15 destinations and satisfies cargo, capacity, reserve and geometry checks.',
            'coefficient_domain':'Integer cargo counts, grams and cubic centimetres; float64 energy/time/geometry; numerical MILP certificates.',
            'conventions':'Single destination round trip; empty return; WGS84 local affine metres; all touched DEM cells; 50m cruise clearance; 30m service height; g=9.80665; no descent or hover energy.',
            'inputs':[{'path':p.relative_to(root).as_posix(),'sha256':sha(p)} for p in inputs],
            'bounds':{'nodes':15,'boxes':80,'types':3,'reserve_levels':[.1,.15,.2,.25,.3],
                      'priority_orders':['N,E,T','E,N,T','T,N,E'],'candidate_truncation':False,
                      'dp_state_truncation':False,'milp_phase_time_limit_s':30,
                      'baseline_states':sum(a['states'] for a in base['areas']),
                      'baseline_patterns':sum(len(a['patterns']) for a in base['areas'])},
            'non_claims':['No continuous-space real-flight optimality.','No fleet, battery-turnaround or delivery-deadline scheduling.','No independent MATLAB reproduction claimed.']},
        'randomness':{'used':False,'generator':'none','seed':None},
        'reference_documents':([{'path':reference.relative_to(root).as_posix(),
                                 'sha256':sha(reference),'required_for_computation':False}]
                               if reference.is_file() else []),
        'run':{'started_at':started,'runtime_seconds':time.monotonic()-tick,'exit_status':0},
        'outputs':[{'path':p.relative_to(root).as_posix(),'sha256':sha(p)} for p in sorted(out.iterdir()) if p.is_file() and p.name!='manifest.json'],
        'checks':[{'name':k,'passed':v['passed'],'count':v['checks']} for k,v in validation.items()],
        'result':'Implementation and finite assertion verified within numerical tolerances; '+json.dumps(base['metrics']),
        'residual_risks':['DEM resolution and deterministic energy model.','Floating arithmetic and HiGHS feasibility/optimality tolerances.','Legacy v1 manifest records hashes after the run, not a pre/post mutation guard.']}
    write_json(out/'manifest.json',manifest)
    print(json.dumps(base['metrics'],indent=2))
    print('All scenarios independently verified; output:',out)


if __name__=='__main__':
    main()
