"""Preserve a failed diagnostic attempt and repair only a staged runtime key."""
from pathlib import Path
import sys,json,shutil,re,hashlib
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_three_ward_cpu_e2e import OUT,REPORT,BUILD
from prepare_sfincs_cpu_build import sha,dump

def main(root=ROOT):
 root=Path(root);out=root/OUT;report=root/REPORT;manifest=json.loads((out/'run_manifest.json').read_text());first=manifest['runs'][0];folder=root/first['folder'];archive=out/'attempts/v2_ignored_ampr_block'
 assert json.loads((folder/'execution.json').read_text())['returncode']==1
 assert 'requires piecewise linear rain'in(folder/'solver_stdout.txt').read_text()
 shutil.move(folder,archive);folder.mkdir()
 for p in archive.iterdir():
  if p.name in first['input_sha256']:
   if p.name=='sfincs_subgrid.nc':(folder/p.name).hardlink_to(p)
   else:shutil.copyfile(p,folder/p.name)
 source=(root/BUILD/'context/source/src/sfincs_input.f90').read_text();known=set(re.findall(r"call read_\w+_input\(500,\s*'([^']+)'",source,re.I));changes=[]
 for run in manifest['runs']:
  folder=root/run['folder'];cfgpath=folder/'sfincs.inp';before=cfgpath.read_text();assert 'amprblock 'not in before
  cfgpath.write_text(before+'amprblock                = 0\n')
  run['original_source_input_sha256']=dict(run['input_sha256']);run['input_sha256']['sfincs.inp']=sha(cfgpath)
  geometry={k:v for k,v in run['input_sha256'].items()if k not in ['roof_surface_precip.nc','sfincs.bzs']}
  run['identity']['geometry_sha256']=hashlib.sha256(json.dumps(geometry,sort_keys=True,separators=(',',':')).encode()).hexdigest()
  run['runtime_configuration_change']={'added_recognized_key':'amprblock=0','ignored_legacy_key':'ampr_block=0','scientific_curve_changed':False,'source_original_changed':False}
  (folder/'udt_run_identity.txt').write_text('\n'.join(k+'='+v for k,v in run['identity'].items())+'\n');dump(folder/'udt_run_identity.json',run['identity'])
  cfg={k.strip():v.strip()for line in cfgpath.read_text().splitlines()if '='in line for k,v in[line.split('=',1)]}
  changes.append({'case_id':run['case_id'],'original_cfg_sha256':run['original_source_input_sha256']['sfincs.inp'],'staged_cfg_sha256':run['input_sha256']['sfincs.inp'],'added_key':'amprblock','effective_linear_interpolation':True,'remaining_ignored_keys':sorted(set(cfg)-known),'stopdepth_note':'unrecognized legacy entry retained; solver native instability criterion unchanged','rain_and_stage_bytes_unchanged':True})
 manifest['second_failed_attempt']={'folder':str(archive.relative_to(root)),'reason':'ampr_block ignored by official parser; first attempted continuity step rejected before ledger capture','completed_accounted_steps':0,'attempted_steps':1}
 dump(out/'run_manifest.json',manifest);dump(report/'runtime_configuration_audit.json',{'changes':changes,'curve_lineage_split_unchanged':True,'no_frozen_config_or_contract_overwrite':True,'future_input_requirement':'use recognized amprblock=0, dtwnd=1; old ampr_block field alone is ineffective'})
 print('Staged controls now use recognised amprblock=0; prior failed attempt preserved.')
if __name__=='__main__':main()
