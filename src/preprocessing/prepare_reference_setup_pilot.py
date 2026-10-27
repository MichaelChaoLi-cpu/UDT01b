"""Stage only the authorized primary-grid controls, then eight preselected pilot forcings."""
from pathlib import Path
import json,sys,shutil
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_sfincs_cpu_build import sha,dump
from generate_three_ward_minimum_scenarios import semantic,arrays
from load_three_ward_minimum_scenarios import catalog,find_case,load_forcing,export_galibier
OUT='data/processed/three_ward_reference_setup_pilot_v1'
REPORT='data/exp/data-preprocessing/three_ward_reference_setup_pilot_v1'
BASE='data/processed/three_ward_conservative_positivity_v2'
SOLVER='data/processed/three_ward_rain_step_integral_v1'
ADAPTIVE=SOLVER
CONTROLS=['dry_raised','rain60_raised']
PILOT=['dry_low_current_geometry','rain100_raised_current_geometry','train-01','train-02','train-03','train-04','validation-01','validation-02']
HUMAN='MILI-D-20261015-017'

def solver():return json.loads((ROOT/SOLVER/'solver_binding.json').read_text())
def copy_input(source,dest,name):
 if name=='sfincs_subgrid.nc':(dest/name).hardlink_to(source/name)
 else:shutil.copyfile(source/name,dest/name)
def identity(run,ident,dest):
 ident=dict(ident);ident.update(run_id=run['run_id'],solver_digest=solver()['image_id'],patch_sha256=solver()['patch_sha256'])
 (dest/'udt_run_identity.txt').write_text('\n'.join(k+'='+v for k,v in ident.items())+'\n');dump(dest/'udt_run_identity.json',ident);run['identity']=ident

def prepare_controls():
 out=ROOT/OUT;report=ROOT/REPORT;out.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
 p=out/'qualification_manifest.json'
 if p.exists():return
 assert json.loads((ROOT/SOLVER/'qualification.json').read_text())['all_original_event_gates_passed']
 before=json.loads((ROOT/BASE/'run_manifest.json').read_text());runs=[]
 for old in before['runs']:
  case=old['case_id'];src=ROOT/old['folder'];dest=out/'qualification/runs'/(case+'__exact_rain_10m');dest.mkdir(parents=True)
  for name,h in old['input_sha256'].items():assert sha(src/name)==h;copy_input(src,dest,name)
  run={**old,'run_id':dest.name,'folder':str(dest.relative_to(ROOT)),'source_folder':old['folder'],'previous_e2e_folder':old['folder'],'image_id':solver()['image_id'],'image_digest':solver()['image_id'],'input_sha256':dict(old['input_sha256']),'phase':'qualification','grid':'regular_10m','library_descendant':False,'split':'open_control','new_parent_event':False,'forcing_case_id':case}
  identity(run,old['identity'],dest);runs.append(run)
 dump(p,{'runs':runs,'training':False,'stage':'two legacy regular-grid qualification controls','human_record':HUMAN})
 c=catalog();selected=[x for x in c['cases']if x['case_id']in PILOT];assert len(selected)==8 and all(x['adaptive_companion']and not x.get('existing_completed',False)for x in selected)
 dump(out/'authorized_plan.json',{'human_record':HUMAN,'qualification_cases':CONTROLS,'pilot_case_ids':PILOT,'pilot_grid':'regular_10m','pilot_requires_both_legacy_control_gates':True,'new_independent_parent_candidates':6,'new_open_controls':2,'adaptive_pilot_runs':0,'maximum_new_hydraulic_runs':10,'prospective58_subset_authorized':8,'remaining50_not_authorized':True,'training':False,'sealed_test_and_challenge_responses_authorized':False,'selection':'original adaptive-companion selections, frozen before responses'})
 protected={}
 for folder in ['data/processed/three_ward_minimum_scenarios_v1']:
  for f in (ROOT/folder).rglob('*'):
   if f.is_file():protected[str(f.relative_to(ROOT))]=sha(f)
 for f in [ROOT/BASE/'freeze.json',ROOT/SOLVER/'freeze.json',ROOT/'src/preprocessing/reference_label_policy.py',ROOT/'data/processed/three_ward_reference_contract_v1/reference_registry.json']:protected[str(f.relative_to(ROOT))]=sha(f)
 dump(report/'provenance.json',{'protected_sha256':protected,'solver':solver(),'human_record':HUMAN,'baseline58_manifest_is_historical_snapshot':True})
 print('STAGED two primary-grid controls; froze eight-input selection',flush=True)

def prepare_pilot():
 out=ROOT/OUT;report=ROOT/REPORT
 for case in CONTROLS:
  a=json.loads((report/'qualification'/case/'audit.json').read_text());assert a['all_checks_passed']and a['admission']['technical_reference']
 p=out/'pilot_manifest.json'
 if p.exists():return
 base=json.loads((out/'qualification_manifest.json').read_text())['runs'][0];source=ROOT/base['folder'];runs=[]
 for case in PILOT:
  binding=find_case(case);assert binding['split']in ['train','validation','open_control']and not binding.get('existing_completed',False)
  f=ROOT/'data/processed/three_ward_minimum_scenarios_v1'/binding['forcing_file'];assert semantic(arrays(f))==binding['forcing_sha256'];assert sha(f)==json.loads((report/'provenance.json').read_text())['protected_sha256'][str(f.relative_to(ROOT))]
  dest=out/'pilot/runs'/(case+'__regular_10m');dest.mkdir(parents=True)
  export=out/'pilot/forcing_exports'/case;export_galibier(case,export,root=ROOT)
  for name in base['input_sha256']:
   copy_input(export if name in ['roof_surface_precip.nc','sfincs.bzs','sfincs.bnd']else source,dest,name)
  assert sha(dest/'sfincs.bnd')==sha(source/'sfincs.bnd'),'Boundary order changed'
  shutil.copyfile(export/'export_contract.json',dest/'export_contract.json')
  run={**base,'case_id':case,'run_id':dest.name,'folder':str(dest.relative_to(ROOT)),'phase':'pilot','split':binding['split'],'forcing_case_id':case,'library_descendant':True,'new_parent_event':binding['independent_parent_candidate'],'original_forcing_sha256':binding['forcing_sha256'],'parent_id':binding['parent_id'],'family_id':binding['family_id'],'source_forcing_file':str(f.relative_to(ROOT)),'source_forcing_file_sha256':sha(f),'input_sha256':{n:sha(dest/n)for n in base['input_sha256']}}
  for obsolete in ['previous_e2e_folder','original_source_input_sha256','scientific_input_sha256']:
   run.pop(obsolete,None)
  run['geometry_template_folder']=base['folder'];run['forcing_export_folder']=str(export.relative_to(ROOT));run['source_folder']=base['folder']
  ident=dict(base['identity']);ident.update(parent_id=binding['parent_id'],family_id=binding['family_id'],split=binding['split'],forcing_sha256=binding['forcing_sha256']);identity(run,ident,dest);runs.append(run)
 dump(p,{'runs':runs,'training':False,'original58_snapshot_unchanged':True,'executed_subset_overlay':True,'human_record':HUMAN});print('STAGED EIGHT preselected regular-grid pilot inputs',flush=True)
if __name__=='__main__':prepare_controls()
