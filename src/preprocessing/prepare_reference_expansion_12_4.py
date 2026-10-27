"""Select a bounded cumulative 12/4 parent tranche using unchanged forcing inputs only."""
from pathlib import Path
import sys,json,shutil,itertools,csv
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_sfincs_cpu_build import sha,dump
from generate_three_ward_minimum_scenarios import semantic,arrays,normalized,distance
from load_three_ward_minimum_scenarios import catalog,find_case,export_galibier,load_forcing
from prepare_reference_setup_pilot import copy_input
OUT='data/processed/three_ward_reference_expansion_12_4_v1'
REPORT='data/exp/data-preprocessing/three_ward_reference_expansion_12_4_v1'
PRIOR='data/processed/three_ward_reference_setup_pilot_v1'
PRIOR_REPORT='data/exp/data-preprocessing/three_ward_reference_setup_pilot_v1'
LIB='data/processed/three_ward_minimum_scenarios_v1'
HUMAN='MILI-D-20261016-001'
SHAPES=['early','central','late','multiple_peak']
FEATURES=['rain_peak_mm_h','rain_total_mm','rain_duration_min','stage_amplitude_m','meaningful_peak_lag_min']
EXISTING=['train-01','train-02','train-03','train-04','validation-01','validation-02']

def csvwrite(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='')as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def features(c):
 p=c['parameters'];return np.array([p['rain_peak_mm_h'],float(np.trapezoid(p['rain_knot_rate_mm_h'],p['rain_knot_time_s'])/3600),p['rain_duration_s']/60,p['stage_amplitude_m'],p['actual_peak_lag_s']/60 if p['actual_peak_lag_s']is not None else np.nan])

def select():
 out=ROOT/OUT;rep=ROOT/REPORT;out.mkdir(parents=True,exist_ok=True);rep.mkdir(parents=True,exist_ok=True)
 if(out/'selection.json').exists():return json.loads((out/'selection.json').read_text())
 cases=[c for c in catalog()['cases']if c['split']in ['train','validation']];f={c['case_id']:features(c)for c in cases};selected=[];metrics=[];selection_trace=[]
 for split,total,quota in [('train',12,3),('validation',4,1)]:
  pool=[c for c in cases if c['split']==split];lookup={c['case_id']:c for c in pool};initial=[c for c in EXISTING if c in lookup];chosen=initial.copy();values=np.array([f[c['case_id']]for c in pool]);spans=np.nanmax(values,0)-np.nanmin(values,0);assert np.all(spans>0)
  def dist(a,b):
   a,b=f[a],f[b];ok=np.isfinite(a)&np.isfinite(b);return float(np.sqrt(np.mean(((a[ok]-b[ok])/spans[ok])**2)))
  while len(chosen)<total:
   counts={s:sum(lookup[k]['parameters']['shape_class']==s for k in chosen)for s in SHAPES};available=[c['case_id']for c in pool if c['case_id']not in chosen and counts[c['parameters']['shape_class']]<quota];assert available
   k=min(available,key=lambda k:(-min(dist(k,j)for j in chosen),k));selection_trace.append({'split':split,'case_id':k,'minimum_distance_before_selection':min(dist(k,j)for j in chosen)});chosen.append(k)
  assert all(sum(lookup[k]['parameters']['shape_class']==s for k in chosen)==quota for s in SHAPES)
  before_radius=max(min(dist(c['case_id'],k)for k in initial)for c in pool);after_radius=max(min(dist(c['case_id'],k)for k in chosen)for c in pool)
  for stage,ids in [('existing',initial),('selected',chosen),('original_partition_pool',list(lookup))]:
   x=np.array([f[k]for k in ids])
   for j,name in enumerate(FEATURES):
    v=x[:,j];v=v[np.isfinite(v)];metrics.append({'split':split,'scope':stage,'feature':name,'parent_count':len(ids),'meaningful_count':len(v),'minimum':float(v.min()),'median':float(np.median(v)),'maximum':float(v.max()),'max_pool_nearest_distance':before_radius if stage=='existing'else after_radius if stage=='selected'else 0.})
  selected.extend(chosen)
 rows=[]
 for c in cases:
  k=c['case_id'];row={key:c[key]for key in ['case_id','parent_id','family_id','split','forcing_sha256','normalized_shape_sha256']};row.update(shape=c['parameters']['shape_class'],selected=k in selected,reused=k in EXISTING,**{n:float(v)if np.isfinite(v)else None for n,v in zip(FEATURES,f[k])});rows.append(row)
 chosen_cases=[find_case(k)for k in selected];norms={k:normalized(load_forcing(k)['Event Time'],load_forcing(k)['Physical Rainfall Rate'],find_case(k)['parameters']['rain_start_s'],find_case(k)['parameters']['rain_duration_s'])for k in selected};pairs=[]
 for a,b in itertools.combinations(selected,2):
  rmse,corr,near=distance(norms[a],norms[b]);pairs.append({'first':a,'second':b,'normalized_rain_rmse':rmse,'correlation':corr,'near_duplicate':near,'cross_split':find_case(a)['split']!=find_case(b)['split']});assert not near,'Original selected inputs unexpectedly fail the unchanged shape check'
 new=[k for k in selected if k not in EXISTING];assert len(new)==10 and len(selected)==16
 # Pair complementary shapes without changing the input-only selection.
 tr=[k for k in new if find_case(k)['split']=='train'];va=[k for k in new if find_case(k)['split']=='validation'];batches=[tr[i:i+2]for i in range(0,len(tr),2)]+[va];assert len(batches)==5 and all(len(b)==2 for b in batches)
 plan={'human_record':HUMAN,'selection_protocol':'Retain existing parents; within each original partition, greedily maximize minimum forcing distance with fixed shape quotas; lexicographic ties. Five equally weighted selection-only min-max-scaled physical attributes; lag omitted from a pair if either pulse is absent. Scaling is not learned model normalization. No response scores used.','features':FEATURES,'selected_case_ids':selected,'reused_case_ids':EXISTING,'new_case_ids':new,'batches':batches,'cumulative_training_parents':12,'cumulative_validation_parents':4,'training_shape_quota':3,'validation_shape_quota':1,'new_hydraulic_run_limit':10,'grid':'regular_10m','sealed_test_challenge_authorized':False,'training_authorized':False,'selection_trace':selection_trace,'coverage_limitations':['Four validation events span only the selected input ranges, not the entire original pool.','All inputs remain synthetic, spatially uniform physical rain and a coherent seven-group stage pulse under fixed geometry and dry interior initial state.','Sixteen lineage candidates do not establish statistical independence, sample sufficiency, observed-event accuracy or surrogate skill.','No replacement based on resulting depth or admission outcomes; failed responses remain held.']}
 csvwrite(rep/'selection_inventory.csv',rows);csvwrite(rep/'forcing_coverage.csv',metrics);csvwrite(rep/'selected_parent_pair_audit.csv',pairs);dump(out/'selection.json',plan)
 protected=json.loads((ROOT/PRIOR_REPORT/'provenance.json').read_text())['protected_sha256'];protected={**protected,str(Path(PRIOR)/'freeze.json'):sha(ROOT/PRIOR/'freeze.json')}
 dump(rep/'provenance.json',{'protected_sha256':protected,'prior_frozen_delivery':PRIOR,'prior_freeze_sha256':sha(ROOT/PRIOR/'freeze.json'),'selection_sha256':sha(out/'selection.json'),'source_catalog_sha256':sha(ROOT/LIB/'catalog.json'),'original_execution_snapshot_sha256':sha(ROOT/LIB/'run_manifest.json'),'human_record':HUMAN,'model_training':False})
 print(json.dumps({'selected':selected,'new':new,'batches':batches}),flush=True);return plan

def prepare():
 plan=select();out=ROOT/OUT;rep=ROOT/REPORT;p=out/'expansion_manifest.json'
 if p.exists():return json.loads(p.read_text())
 prior=json.loads((ROOT/PRIOR/'pilot_manifest.json').read_text());old={r['case_id']:r for r in prior['runs']};template=old['train-01'];source=ROOT/template['folder'];runs=[];reuse=[]
 for k in EXISTING:
  a=json.loads((ROOT/PRIOR_REPORT/'pilot'/k/'audit.json').read_text());assert a['all_checks_passed']and a['admission']['technical_reference'];reuse.append({'case_id':k,'run_folder':old[k]['folder'],'audit_path':str(Path(PRIOR_REPORT)/'pilot'/k/'audit.json'),'label_path':str(Path(PRIOR)/'pilot'/(k+'_candidate_reference.npz'))})
 for case in plan['new_case_ids']:
  c=find_case(case);f=ROOT/LIB/c['forcing_file'];assert semantic(arrays(f))==c['forcing_sha256'];assert c['split']in ['train','validation'];assert not any((ROOT/PRIOR/'pilot/runs').glob(case+'__*'))
  dest=out/'runs'/(case+'__regular_10m');dest.mkdir(parents=True,exist_ok=True);export=out/'forcing_exports'/case
  if not(export/'export_contract.json').exists():export_galibier(case,export,root=ROOT)
  for name in template['input_sha256']:
   if not(dest/name).exists():copy_input(export if name in ['roof_surface_precip.nc','sfincs.bzs','sfincs.bnd']else source,dest,name)
  assert sha(dest/'sfincs.bnd')==sha(source/'sfincs.bnd');shutil.copyfile(export/'export_contract.json',dest/'export_contract.json')
  run=dict(template);run.update(case_id=case,run_id=dest.name,folder=str(dest.relative_to(ROOT)),phase='expansion',split=c['split'],forcing_case_id=case,parent_id=c['parent_id'],family_id=c['family_id'],source_forcing_file=str(f.relative_to(ROOT)),source_forcing_file_sha256=sha(f),original_forcing_sha256=c['forcing_sha256'],geometry_template_folder=template['folder'],forcing_export_folder=str(export.relative_to(ROOT)),source_folder=template['folder'],input_sha256={n:sha(dest/n)for n in template['input_sha256']},batch=next(i+1 for i,b in enumerate(plan['batches'])if case in b))
  ident=dict(template['identity']);ident.update(run_id=run['run_id'],parent_id=c['parent_id'],family_id=c['family_id'],split=c['split'],forcing_sha256=c['forcing_sha256']);run['identity']=ident;(dest/'udt_run_identity.txt').write_text('\n'.join(k+'='+v for k,v in ident.items())+'\n');dump(dest/'udt_run_identity.json',ident)
  assert all(sha(dest/n)==sha(source/n)for n in template['input_sha256']if n not in ['roof_surface_precip.nc','sfincs.bzs']);runs.append(run)
 dump(p,{'runs':runs,'reused':reuse,'selection_sha256':sha(out/'selection.json'),'human_record':HUMAN,'training':False});return json.loads(p.read_text())
if __name__=='__main__':prepare()
