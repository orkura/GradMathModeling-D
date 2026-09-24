"""Run, independently verify, and export the problem-3 simulation."""
from __future__ import annotations

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

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'lcn_q3_matplotlib'))
sys.dont_write_bytecode = True
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from q3.data import load_data
from q3.physics import Terrain
from q3.solver import solve
from q3.report import write_report


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--starts',type=int,default=36)
    parser.add_argument('--seed',type=int,default=20260924)
    parser.add_argument('--relay-grid-m',type=float,default=500)
    parser.add_argument('--spatial-step-m',type=float,default=100)
    parser.add_argument('--refine-seconds',type=float,default=90.0)
    parser.add_argument('--verify-only',action='store_true',help='Read and independently verify the saved solution without rewriting outputs')
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent.parent/'Results'/'Q3')
    args=parser.parse_args()
    if args.starts<1 or min(args.relay_grid_m,args.spatial_step_m)<=0 or args.refine_seconds<0:
        parser.error('starts and spatial resolution must be positive')
    started=datetime.now(timezone.utc).isoformat()
    clock=time.perf_counter()
    data=load_data()
    root=Path(data['repo_root'])
    if args.verify_only:
        from q3.verify import verify
        saved=json.loads((args.output/'solution.json').read_text(encoding='utf-8'))
        if saved.get('source_sha256')!=data['source_sha256']:
            raise RuntimeError('Saved solution input hashes differ from the current attachments')
        audit=verify(saved,data,Terrain(data['dem_path']))
        print(json.dumps({k:audit[k] for k in ['passed','error_count','errors','metrics','communication']},ensure_ascii=False,indent=2))
        if not audit['passed']:
            raise SystemExit(1)
        return
    codefiles=sorted(Path(__file__).parent.rglob('*.py'))
    codefiles=[p for p in codefiles if '.venv' not in p.parts and '__pycache__' not in p.parts]
    code_hashes={str(p.relative_to(root)).replace('\\','/'):sha(p) for p in codefiles}
    terrain=Terrain(data['dem_path'])
    print(f'Inputs: {data["totals"]["boxes"]} boxes; {data["totals"]["hard_deadline_boxes"]} hard deadlines; DEM {terrain.dem.shape}',flush=True)
    config=vars(args).copy()
    config.pop('output')
    config.pop('verify_only')
    result=solve(data,terrain,config)
    result['source_sha256']=data['source_sha256']
    result['source_files']={key:str(Path(path).relative_to(root)).replace('\\','/') for key,path in data['source_paths'].items()}
    result['assumptions']=[
        '水平能耗=电池可用能量×水平距离/载荷等效航程；爬升附加能耗=(含电池空机质量+载荷)gΔh/(效率×3.6e6)。这些是附录缺失项的补充假设。',
        '运输交接与下降不额外计能耗；中继建链期间计悬停功率与通信附加功率。',
        '航段巡航海拔=max(经过全部DEM像元的最高高程+50m,两端作业海拔)，兼容高悬停点。',
        '局部WGS84仿射米制坐标；DEM按原始像元分片常值；连续通信结论针对该离散地形模型。',
        '每次访问的全部货箱在交接结束时完成送达；准备开始占用机体和电池；换电时间包含于准备时间。',
        '无额外工位/充电端口数量约束；同一中继可同时服务多架运输机；未添加题目未给出的带宽和干扰约束。',
        '通信表记录全区间保证可用的保障者；若期间直连可用，实际通信优先直连；采样状态统计与保障预约分开。',
        '有限候选位置、多起点贪心联合排程，再对固定结构用整数秒CP-SAT累计资源模型优化时序与实体分配；每运输架次最多3个服务区且至多1个保底中继位置；未证明全局最优。',
    ]
    for name,path in data['source_paths'].items():
        if sha(path)!=data['source_sha256'][name]:
            raise RuntimeError(f'Input changed during execution: {path}')
    for path,digest in code_hashes.items():
        if sha(root/path)!=digest:
            raise RuntimeError(f'Code changed during execution: {path}; rerun with stable inputs')
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'solution.json',result)
    write_json(output/'validation.json',result['validation'])
    write_json(output/'search_history.json',result['search_history'])
    write_report(result,data,terrain,output)
    versions=[f'Python {platform.python_version()}']
    versions += [f'{p} {importlib.metadata.version(p)}' for p in ['numpy','scipy','rasterio','matplotlib','openpyxl','pillow','ortools']]
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,text=True).stdout.strip()
    dirty=bool(subprocess.run(['git','status','--porcelain'],cwd=root,capture_output=True,text=True).stdout.strip())
    files=[p for p in output.iterdir() if p.is_file() and p.name!='manifest.json']
    manifest=dict(schema_version=1,claim_id='Q3_SUPPLIED_INSTANCE_FEASIBLE_SCHEDULE',
        repository=dict(commit=commit,dirty=dirty),command=subprocess.list2cmdline([sys.executable,*sys.argv]),
        environment=dict(software=versions,hardware=platform.platform()),
        mathematics=dict(assertion_tested='All 80 supplied boxes delivered under the stated flight, deadline, inventory, SOC and continuous raster-radio rules.',
            coefficient_domain='IEEE-754 float64; time tolerance 1e-5 s, energy tolerance 1e-6 kWh',
            conventions='m,s,kg,kWh,kW; appendix-2 supplemental energy assumptions; local affine WGS84/piecewise-constant DEM',
            inputs=list(result['source_files'].values()),bounds=result['search_bounds'],
            non_claims=['No global optimality proof','No certification of real subpixel terrain, weather, interference or unspecified bandwidth']),
        randomness=dict(used=True,generator='Python random.Random (MT19937)',seed=args.seed),
        run=dict(started_at=started,runtime_seconds=time.perf_counter()-clock,exit_status=0),
        outputs=[dict(path=str(p.relative_to(root)).replace('\\','/'),sha256=sha(p)) for p in sorted(files)],
        checks=['80 boxes exactly once','all hard deadlines','independent time and energy reconstruction','airframe/battery/component conflicts including recharge','2 s radio sampling plus continuous interval certificates','input and code hashes unchanged'],
        result='Implementation and supplied finite scenario verified under the stated numerical and raster conventions; bounded heuristic schedule.',
        residual_risks=['Finite relay grid and bounded route/search family','Physical energy assumptions supplement incomplete annex equations','Local affine geographic approximation'],
        input_sha256=data['source_sha256'],code_sha256=code_hashes)
    write_json(output/'manifest.json',manifest)
    print(json.dumps(result['metrics'],ensure_ascii=False,indent=2),flush=True)
    print(f'PASS: {output}; elapsed {time.perf_counter()-clock:.1f} s',flush=True)


if __name__=='__main__':
    main()
