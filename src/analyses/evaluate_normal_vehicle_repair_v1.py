"""Frozen-screenline development comparison of topology/access repair arms.
The previously inspected holdout is explicitly development data, not a fresh test.
"""
from pathlib import Path
import os,sys,json,datetime as dt
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/analyses'))
from evaluate_normal_vehicle_baseline_v1 import metric,sha
P=ROOT/'data/processed/three_ward_normal_vehicle_repair_v1';R=ROOT/'data/exp/estimation/three_ward_normal_vehicle_repair_v1';OLD=ROOT/'data/processed/three_ward_normal_vehicle_baseline_v1';M=ROOT/'data/processed/three_ward_motor_flood_access_v1'
def load(p):
 with np.load(p,allow_pickle=False)as z:return{k:z[k]for k in z.files}
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item()if isinstance(v,np.generic)else str(v))+'\n')
def main():
 prep=json.loads((R/'preparation_summary.json').read_text());assert all(prep['checks'].values());a=pd.read_csv(OLD/'census_passage_comparison_raw.csv',dtype={'count_authority_code':str,'count_unit_id':str,'observation_section_id':str,'source_road_type':str});a['arm']='A_original_single';a['source_split']=a.pop('split');cross=pd.read_csv(OLD/'census_comparison_eligibility.csv',dtype={'count_authority_code':str,'count_unit_id':str,'observation_section_id':str});pr=load(M/'road_projection.npz');inv=pd.read_csv(M/'road_inventory.csv');parts=[a];runchecks={}
 for arm in ['B_topology_single','C_topology_multi']:
  check=json.loads((R/(arm+'_validation.json')).read_text());assert check['passed'];runchecks[arm]=check;z=load(P/(arm+'_flows.npz'));rf=np.array([[np.bincount(pr['arc_to_road'],weights=z['arc_flow'][b,t],minlength=len(inv))for t in range(5)]for b in range(2)]);rows=[]
  for r in a.itertuples():
   match=cross[cross.count_authority_code.eq(r.count_authority_code)&cross.count_unit_id.eq(r.count_unit_id)];assert len(match)==1;idx=np.array([int(x)for x in str(match.iloc[0].matched_road_indices).split('|')]);b=0 if r.branch=='base_no_remote_remote'else 1;t=['daily','hour8','hour10','hour17','daytime12h'].index(r.period);out=r._asdict();out.pop('Index');out['predicted']=float(rf[b,t,idx].sum());out['arm']=arm;rows.append(out)
  parts.append(pd.DataFrame(rows))
 df=pd.concat(parts,ignore_index=True);df['evaluation_role']=df.source_split.map({'calibration':'development_original_calibration65','validation':'development_previously_inspected20'});df['road_class']=np.where(df.source_road_type.isin(['1','2']),'expressway','ordinary');df.to_csv(P/'passage_comparison.csv',index=False,encoding='utf-8-sig');metrics=[]
 for key,g in df.groupby(['arm','branch','period','comparison_class','evaluation_role'],sort=True):metrics.append(dict(zip(['arm','branch','period','comparison_class','evaluation_role'],key),road_class='all',**metric(g)))
 for key,g in df.groupby(['arm','branch','period','comparison_class','evaluation_role','road_class'],sort=True):metrics.append(dict(zip(['arm','branch','period','comparison_class','evaluation_role','road_class'],key),**metric(g)))
 pd.DataFrame(metrics).to_csv(P/'flow_metrics.csv',index=False,encoding='utf-8-sig');primary=[m for m in metrics if m['branch']=='base_no_remote_remote'and m['period']=='daytime12h'and m['comparison_class']=='small'];primarykey=df.branch.eq('base_no_remote_remote')&df.period.eq('daytime12h')&df.comparison_class.eq('small');paired=df[primarykey].pivot(index=['observation_section_id','count_authority_code','count_unit_id','road_class','evaluation_role','observed'],columns='arm',values='predicted').reset_index();paired.to_csv(P/'paired_screenline_changes.csv',index=False,encoding='utf-8-sig')
 changed=[]
 for arm in ['B_topology_single','C_topology_multi']:
  before=paired.A_original_single;after=paired[arm];changed.append({'arm':arm,'all85_zero_before':int((before<1e-9).sum()),'all85_zero_after':int((after<1e-9).sum()),'zero_to_positive':int(((before<1e-9)&(after>1e-9)).sum()),'positive_to_zero':int(((before>1e-9)&(after<1e-9)).sum()),'absolute_error_improved_screenlines':int((abs(after-paired.observed)<abs(before-paired.observed)-1e-7).sum()),'absolute_error_worsened_screenlines':int((abs(after-paired.observed)>abs(before-paired.observed)+1e-7).sum())})
 bindings=json.loads((P/'input_bindings.json').read_text())['sha256'];hashchecks={p:sha(ROOT/p)==h for p,h in bindings.items()};checks={'all_preparation_checks':all(prep['checks'].values()),'both_assignments_completed':all(r['passed']for r in runchecks.values()),'frozen_input_bindings_unchanged':all(hashchecks.values()),'same85screenlines':len(paired)==85,'same_observation_values_each_arm':df.groupby(['count_authority_code','count_unit_id','branch','period','comparison_class'])["observed"].nunique().eq(1).all().item(),'no_raw_demand_rescaling':True,'no_fresh_holdout_claim':set(df.evaluation_role)=={'development_original_calibration65','development_previously_inspected20'},'previous_baseline_unchanged':hashchecks[str((OLD/'pilot_arc_flows.npz').relative_to(ROOT))]};dump(R/'validation.json',{'passed':all(checks.values()),'checks':checks,'source_hash_checks':hashchecks})
 summary={'completed_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'stage':'Normal-flow development comparison;speed,capacity and congestion uncalibrated','preparation':prep,'primary_small12h_base_metrics':primary,'screenline_changes':changed,'source_groups':'Original65+20groups retained, both development after prior inspection','independent_validation':False,'scalar_demand_scaling':False,'checks_passed':sum(checks.values()),'checks_total':len(checks),'route_summaries':runchecks,'new_hydraulics':False,'people_counts':False};dump(R/'summary.json',summary);print(json.dumps({'primary':[m for m in primary if m['evaluation_role']=='development_previously_inspected20'],'changes':changed,'checks':checks},ensure_ascii=False,indent=2))
 if not all(checks.values()):raise SystemExit('Frozen comparison check failed')
if __name__=='__main__':main()
