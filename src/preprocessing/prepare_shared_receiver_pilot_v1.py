#!/usr/bin/env python3
"""Stage one previously allocated policy at paired July stages;never overwrite runs."""
from pathlib import Path
import sys,json,shutil,subprocess
import pandas as pd,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'analyses'))
from shared_receiver_pilot_common_v1 import ROOT,P,R,sha,dump

def main():
 assert not(P/'run_manifest.json').exists(),'Preserve prepared or launched pilot'
 P.mkdir(exist_ok=True);R.mkdir(parents=True,exist_ok=True);(P/'execution').mkdir(exist_ok=True)
 allocation=ROOT/'data/processed/three_ward_shared_receiver_allocation_v1';ar=ROOT/'data/exp/research-design/three_ward_shared_receiver_allocation_v1'
 assert json.loads((ar/'delivery_qa.json').read_text())['passed'];am=json.loads((allocation/'manifest.json').read_text());assert all(sha(ROOT/f)==h for f,h in am['input_sha256'].items())
 portfolio='R1000_T30_Finite_0p5';source_pump=allocation/'portfolios'/portfolio/'udt_pumps.dat';assert any(sha(source_pump)==a['sha256'] for a in am['pump_inputs']);sites=pd.read_parquet(allocation/'selected_intakes_preprocessed.parquet');sites=sites[sites.Portfolio==portfolio].reset_index(drop=True);assert len(sites)==62
 sites.insert(0,'Pump ID',np.arange(1,len(sites)+1));sites.to_csv(P/'pump_sites.csv',index=False)
 budgets=pd.read_parquet(allocation/'shared_receiver_reservations_preprocessed.parquet');budgets=budgets[budgets.Portfolio==portfolio].copy();budgets.to_csv(P/'pool_allowances.csv',index=False)
 image_binding=ROOT/'data/processed/three_ward_barrier_drainage_v1/solver_binding.json';ib=json.loads(image_binding.read_text());assert ib['unit_qualified'] and ib['hydraulic_prefix_qualified'] and ib['no_op_exact']
 image=ib['image_id'];assert subprocess.run(['docker','image','inspect',image,'--format','{{.Id}}'],check=True,capture_output=True,text=True).stdout.strip()==image
 live=subprocess.run(['docker','ps','--format','{{.Names}}'],check=True,capture_output=True,text=True).stdout.strip();assert not live,'Existing Docker workload;do not oversubscribe'
 assert shutil.disk_usage(P).free>=100*1024**3
 budget={'maximum_new_runs':2,'concurrent_hydraulic_runs':1,'cpus':6,'memory_gib':7,'maximum_run_active_wall_s':7200,'maximum_campaign_active_wall_s':14400,'no_clock_progress_stop_s':900,'maximum_new_output_gib':40,'minimum_free_gib':100,'maximum_audit_wall_s':3600,'maximum_postprocess_wall_s':3600,'automatic_retries':0}
 protocol={'status':'prepared_for_authorized_small_pilot','question':'Do conditional shared-budget static reservations produce positive,conservative actual extraction and stable regional road effects?','portfolio':portfolio,'hypotheses':'1000m straight-line screen,0.5m3/s per named shared pool,0.5m3/s site ceiling,30min local snapshot clearance proxy,5m static lift,.05m trigger','new_hydraulic_runs':2,'baseline_cases':['E20240731__H__U','E20240731__L__U'],'execution':'high then low;advance only after unchanged original numerical gates and independent variable-rate pump/shared-pool checks','reuse_baselines':2,'new_barriers':False,'source_site_count':62,'reserved_total_rate':float(sites['Reserved Rate'].sum()),'count_meaning':'modeled native intake locations,not machine count','scope':'three wards and original hydraulic buffer;no pointwise survey','quality':'original thresholds,masks,clock,rain,closed-face and positivity rules unchanged;independent explicit pump source/sink extension','receiving':'fixed-stage reservoirs,static shared nominal reservations;actual added river storage/acceptance not modeled','comparison':'same event,stage,109times and common admitted roads;new closures and off-target head rises kept separately','not_independent_validation':'same historical parent informed allocation;two stages are scenarios,not two events','budget':budget,'training':False,'sealed_test':False,'new_people_counts':False,'automatic_higher_quota_expansion':False}
 dump(P/'protocol.json',protocol)
 original=json.loads((ROOT/'data/processed/three_ward_five_event_population_validation_v1/run_manifest.json').read_text());runs=[];protected={str(image_binding.relative_to(ROOT)):sha(image_binding),str(source_pump.relative_to(ROOT)):sha(source_pump),str((ROOT/'docs/AnaSOP.md').relative_to(ROOT)):sha(ROOT/'docs/AnaSOP.md')}
 for basecase in protocol['baseline_cases']:
  base=next(r for r in original['runs'] if r['case_id']==basecase);s=ROOT/base['folder'];assert json.loads((ROOT/'data/exp/estimation/three_ward_five_event_population_validation_v1/execution'/basecase/'audit.json').read_text())['all_checks_passed']
  case=basecase+'__shared_r1000_q0p5';folder=P/'runs'/case;folder.mkdir(parents=True,exist_ok=False)
  for name,h in base['input_sha256'].items():
   assert sha(s/name)==h,(basecase,name);protected[str((s/name).relative_to(ROOT))]=h
   if name=='sfincs_subgrid.nc':(folder/name).symlink_to(Path(base['large_readonly_source']).resolve())
   else:shutil.copy2(s/name,folder/name)
  shutil.copy2(source_pump,folder/'udt_pumps.dat');assert (folder/'sfincs.inp').read_bytes()==(s/'sfincs.inp').read_bytes()
  ident=base['identity'].copy();ident.update(run_id=case+'__shared_receiver_10m',split='external_shared_receiver_pilot',solver_digest=image,patch_sha256=ib['patch_sha256'],treatment_sha256=sha(source_pump))
  dump(folder/'udt_run_identity.json',ident);(folder/'udt_run_identity.txt').write_text('\n'.join(k+'='+str(v) for k,v in ident.items())+'\n')
  names=list(base['input_sha256'])+['udt_pumps.dat'];run=base.copy();run.update(case_id=case,baseline_case=basecase,original_folder=base['folder'],measure='shared_static_drainage',portfolio=portfolio,folder=str(folder.relative_to(ROOT)),run_id=ident['run_id'],split=ident['split'],identity=ident,image_id=image,input_sha256={n:sha(folder/n) for n in names},reused=False);runs.append(run)
  for name in ['sfincs_map.nc','execution.json']:protected[str((s/name).relative_to(ROOT))]=sha(s/name)
 dump(P/'run_manifest.json',{'runs':runs,'execution_order':[r['case_id'] for r in runs],'protocol_sha256':sha(P/'protocol.json'),'budget':budget,'baseline_image':base['image_id'],'pump_solver_image':image})
 # Recheck the already-qualified production coupled operator under the chosen6threads.
 fixture=P/'qualification/pump_fixture_6';fixture.mkdir(parents=True,exist_ok=False)
 cmd=['docker','run','--rm','--platform','linux/amd64','--network','none','--cpus','6','--memory','7g','--env','OMP_NUM_THREADS=6','--mount',f'type=bind,source={fixture},target=/data','--entrypoint','/opt/udt-build/pump_fixture',image]
 z=subprocess.run(cmd,capture_output=True,text=True,timeout=120);(fixture/'stdout.txt').write_text(z.stdout);(fixture/'stderr.txt').write_text(z.stderr);assert z.returncode==0 and 'PASS' in z.stdout
 for f in [P/'protocol.json',P/'run_manifest.json',P/'pump_sites.csv',P/'pool_allowances.csv',ROOT/'src/preprocessing/reference_label_policy.py',ROOT/'src/analyses/audit_cross_event_protection_run_v1.py',ROOT/'src/preprocessing/barrier_drainage_v1/pump_routines.inc']:protected[str(f.relative_to(ROOT))]=sha(f)
 dump(P/'protected_sources.json',protected);dump(R/'preflight_validation.json',{'passed':True,'new_runs_staged':2,'existing_baselines':2,'image_pinned_and_present':image,'static_rain_geometry_stage_inputs_exact':True,'pump_fixture_6threads_passed':True,'shared_reservation_checked_by_allocation_delivery_qa':True,'new_full_hydraulic_runs_started':0,'no_new_barrier_or_forcing':True,'minimum_free_gib':100,'budget':budget})
 (R/'README.md').write_text('''# Shared receiver conditional pilot

Two existing July31rain-parent stages,one frozen62-intake portfolio,R1000_T30_Finite_0p5. Serial6CPU/7GiB,maximum7200active seconds per hydraulic run,14400hydraulic seconds for the batch,40GiB new output ceiling,100GiB minimum free disk. Audits and postprocessing each have3600active seconds ceiling. No automatic retries or larger-quota expansion.

Prepare once: `.venv/bin/python src/preprocessing/prepare_shared_receiver_pilot_v1.py`. Run once: `.venv/bin/python src/analyses/run_shared_receiver_pilot_v1.py`. Preparation and execution refuse existing attempts. Review is rebuilt by `report_shared_receiver_pilot_v1.py`. Do not rerun completed cases.

Create PAUSE in the processed pilot directory to pause a running container or audit/postprocess worker;remove PAUSE to resume. Create STOP to stop the current work and hold the queue with all files preserved. The runner never automatically retries held work.

Image unchanged from the qualified pump extension;original geometry,rain,roof assignment,stage and original Bsections remain exact. Only opt-in pump file and derivative run identity differ. Variable per-site rates and pooled Furukawa79/111 limits are independently checked at every actual step;original numerical thresholds are unchanged after explicit independently validated source/sink extension. High stage must pass before low starts. Actual maps are not expected to equal the baseline after treatment.

After both numerical gates pass,automatically compare all109same-time road states on common admitted support and separate restored/newly closed roads and off-target water-level rises/falls. No population or congestion result is computed in this pilot. The parent event informed placement;this is a conditional numerical/mechanism check,not independent real-world validation.62native intake locations do not imply62physical machines. Fixed-stage receivers omit added river storage,permission and real acceptance capacity.
''')
 print('SHARED_PILOT_PREFLIGHT_PASSED',flush=True)
if __name__=='__main__':main()
