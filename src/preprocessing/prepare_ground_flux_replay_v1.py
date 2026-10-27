#!/usr/bin/env python3
"""Stage two isolated observational replays;unchanged physical inputs/image."""
from pathlib import Path
import json,hashlib,shutil
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];TAG='three_ward_ground_flux_replay_v1';P=ROOT/'data/processed'/TAG;R=ROOT/'data/exp/estimation'/TAG;PR=ROOT/'data/processed/three_ward_zone_rain_flux_readiness_v1';RR=ROOT/'data/exp/data-preprocessing/three_ward_zone_rain_flux_readiness_v1'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 assert not (P/'run_manifest.json').exists(),'Preserve staged attempts;no overwrite'
 P.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True);(P/'execution').mkdir(exist_ok=True)
 plan=json.loads((RR/'ground_flux_interface_plan.json').read_text());assert json.loads((RR/'validation.json').read_text())['passed'];cases=plan['proposed_minimum_replay']['cases'];budgets=pd.read_parquet(PR/'ground_request_budget_preprocessed.parquet').set_index('Case');source=json.loads((ROOT/'data/processed/three_ward_five_event_population_validation_v1/run_manifest.json').read_text());runs=[];protected={}
 for case in cases:
  old=next(x for x in source['runs'] if x['case_id']==case);src=ROOT/old['folder'];dst=P/'runs'/case;dst.mkdir(parents=True,exist_ok=False)
  for name,h in old['input_sha256'].items():
   assert sha(src/name)==h,(case,name,'Original input changed');protected[str((src/name).relative_to(ROOT))]=h
   shutil.copy2(src/name,dst/name)
  request=ROOT/budgets.loc[case,'Request Path'];assert sha(request)==budgets.loc[case,'Request SHA256'];base=(src/'sfincs.crs').read_bytes();(dst/'sfincs.crs').write_bytes(base+(b'' if base.endswith(b'\n') else b'\n')+request.read_bytes())
  ident=dict(old['identity']);ident['run_id']=case+'__ground_flux_replay_v1';(dst/'udt_run_identity.txt').write_text('\n'.join(k+'='+v for k,v in ident.items())+'\n');dump(dst/'udt_run_identity.json',ident)
  run=dict(old);run.update(folder=str(dst.relative_to(ROOT)),original_folder=old['folder'],run_id=ident['run_id'],identity=ident,reused=False,large_readonly_source=str((dst/'sfincs_subgrid.nc').resolve()),original_input_sha256=old['input_sha256'],input_sha256={n:sha(dst/n) for n in old['input_sha256']},request_sha256=sha(request),new_ground_sections=int(budgets.loc[case,'New Ground Sections']),maximum_run_active_wall_s=float(budgets.loc[case,'Proposed Wallclock Stop Seconds']))
  assert {n for n in run['input_sha256'] if run['input_sha256'][n]!=old['input_sha256'][n]}=={'sfincs.crs','udt_run_identity.txt'}
  assert (dst/'sfincs.crs').read_bytes().startswith(base);runs.append(run);print('STAGED',case,run['new_ground_sections'],'new sections',flush=True)
  probe=P/'mapping_probe'/case;probe.mkdir(parents=True,exist_ok=False)
  for n in run['input_sha256']:
   # Probe has a separate write directory;large inputs are additionally mounted read-only.
   if n in ['sfincs_subgrid.nc','historical_rain.nc']:(probe/n).touch()
   else:shutil.copy2(dst/n,probe/n)
 dump(P/'run_manifest.json',{'runs':runs,'execution_order':cases,'maximum_new_runs':2,'image_id':runs[0]['image_id'],'budget':{'cpus':6,'memory_gib':7,'minimum_free_gib':40,'maximum_new_output_gib':40,'no_clock_progress_stop_s':900},'original_auditor':'src/analyses/audit_cross_event_protection_run_v1.py','old_6763_sections_audited_without_threshold_changes':True,'observational_replay_only':True,'training':False,'people_counted':False})
 dump(P/'protected_sources.json',protected);dump(R/'input_validation.json',{'passed':True,'cases':cases,'only_changed_inputs':['sfincs.crs','udt_run_identity.txt'],'physical_inputs_exact':True,'image_unchanged':True,'original_B_sections_preserved':True,'source_hashes_valid':True,'new_hydraulic_timesteps_executed':0})
 print('REPLAY_STAGING_COMPLETE',flush=True)
if __name__=='__main__':main()
