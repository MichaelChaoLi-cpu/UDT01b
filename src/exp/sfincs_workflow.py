"""Portable preflight and pinned official SFINCS container launcher.

No flood equations or home-made subgrid files are implemented here. Actual
Tokyo input conversion must use a verified official builder in a later step.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'data/processed/three_ward_sfincs_framework'


def load(p):
    return json.loads(p.read_text())


def audit():
    with (BASE/'input_inventory.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    missing=[r['id'] for r in rows if r['source_path'] and not (ROOT/r['source_path']).exists()]
    readiness=load(BASE/'readiness.json')
    return {'inventory_inputs':len(rows),'missing_registered_files':missing,
            'baseline_run_ready':readiness['baseline_run_ready'],
            'critical_missing_inputs':readiness['critical_missing_inputs'],
            'formal_Tokyo_bundle_created':load(BASE/'model_spec.json')['official_solver_bundle_created'],
            'note':'Source presence does not establish datum alignment or forcing adequacy.'}


def docker_command(folder):
    folder=folder.resolve()
    # Restrict container read/write access to the selected model input folder.
    if not folder.is_relative_to(BASE.resolve()):
        raise ValueError('Model input folder must be inside the project SFINCS framework directory.')
    return ['docker','run','--rm','--platform','linux/amd64','--network','none',
            '--cpus','2','--memory','2g','-e','OMP_NUM_THREADS=2',
            '--mount',f'type=bind,src={folder},dst=/data',
            load(BASE/'model_spec.json')['solver']['docker_image']]


def check_files(folder):
    p=folder/'sfincs.inp'
    if not p.is_file(): raise ValueError('Missing sfincs.inp; Tokyo builder has not yet produced a bundle.')
    params={}
    for line in p.read_text().splitlines():
        line=line.split('#',1)[0].strip()
        if '=' in line:
            k,v=line.split('=',1);params[k.strip()]=v.strip().strip('"\'')
    for key,value in params.items():
        if key.endswith('file'):
            path=(folder/value).resolve()
            if not path.is_relative_to(folder.resolve()) or not path.is_file():
                raise ValueError(f'Missing or external model file: {key}={value}')
    if 'sbgfile' not in params and 'depfile' not in params: raise ValueError('No terrain or subgrid table.')
    if 'mskfile' not in params: raise ValueError('No active-cell mask.')
    return params


def run(folder, smoke):
    params=check_files(folder)
    if not smoke:
        raise ValueError('Formal Tokyo bundles are not prepared. Build/verify official subgrid and forcing inputs first; only installation smoke test is runnable now.')
    if folder.resolve() != (BASE/'engine_smoke_test').resolve():
        raise ValueError('Smoke test can only run the separately labeled synthetic installation case.')
    known_outputs=['sfincs_map.nc','sfincs_his.nc','sfincs_log.txt','execution.json']
    if any((folder/name).exists() for name in known_outputs):
        raise ValueError('Previous outputs exist. Preserve them before explicitly removing/rerunning the smoke case.')
    if load(folder/'test_contract.json').get('synthetic') is not True:
        raise ValueError('Synthetic test contract is missing.')
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.is_file()}
    cmd=docker_command(folder)
    subprocess.run(['docker','info','--format','{{.ServerVersion}}'],check=True,capture_output=True,text=True,timeout=30)
    start=time.monotonic()
    with (folder/'sfincs_log.txt').open('w') as log:
        result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=180)
    text=(folder/'sfincs_log.txt').read_text(errors='replace')
    finished=('Simulation finished' in text or 'Simulation is finished' in text)
    summary={'synthetic':True,'command':cmd,'returncode':result.returncode,
             'runtime_seconds':time.monotonic()-start,'input_hashes':manifest,
             'solver_completion_marker':finished,
             'map_output_present':(folder/'sfincs_map.nc').exists(),
             'history_output_present':(folder/'sfincs_his.nc').exists(),
             'physical_validation':'pending independent comparison to synthetic analytical depth',
             'not_a_Tokyo_validation':True}
    (folder/'execution.json').write_text(json.dumps(summary,indent=2)+'\n')
    if result.returncode or not finished or not summary['map_output_present'] or not summary['history_output_present']:
        raise RuntimeError('Official solver did not complete; inspect sfincs_log.txt and execution.json.')
    return summary


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['audit','command','smoke'])
    ap.add_argument('--model-dir',type=Path)
    args=ap.parse_args()
    folder=args.model_dir or BASE/'engine_smoke_test'
    if args.action=='audit': out=audit()
    elif args.action=='command': out={'command':shlex.join(docker_command(folder)), 'files_checked':check_files(folder),'executes':False}
    else: out=run(folder,smoke=True)
    print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
