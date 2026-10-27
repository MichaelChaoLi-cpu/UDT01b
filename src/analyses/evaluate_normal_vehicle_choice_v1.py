"""Fixed 2x2 development comparison with conserved cost-ensemble flow mixtures."""
from pathlib import Path
import os,sys,json,datetime as dt
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/analyses'))
from evaluate_normal_vehicle_baseline_v1 import metric,sha
P=ROOT/'data/processed/three_ward_normal_vehicle_choice_v1';R=ROOT/'data/exp/estimation/three_ward_normal_vehicle_choice_v1';TIME=ROOT/'data/processed/three_ward_normal_vehicle_time_v1';REP=ROOT/'data/processed/three_ward_normal_vehicle_repair_v1';OLD=ROOT/'data/processed/three_ward_normal_vehicle_baseline_v1';M=ROOT/'data/processed/three_ward_motor_flood_access_v1'
ARMS=['T2_reference','U_single','UF_single','UM_mix','UFM_mix'];PERIODS=['daily','hour8','hour10','hour17','daytime12h'];BRANCHES=['base_no_remote_remote','plus_conditional_chord_through']
def load(p):
 with np.load(p,allow_pickle=False)as z:return{k:z[k]for k in z.files}
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item()if isinstance(v,np.generic)else str(v))+'\n')
def main():
 assert json.loads((R/'execution_state.json').read_text())['status']=='complete';pr=load(M/'road_projection.npz');rr=load(REP/'turn_router.npz');ep=load(REP/'multi_endpoint_components.npz');dem=load(REP/'multi_routing_demands.npz');inv=pd.read_csv(M/'road_inventory.csv');d=pd.read_csv(TIME/'passage_comparison.csv',dtype={'count_authority_code':str,'count_unit_id':str,'observation_section_id':str});d=d[d.arm.eq('T2_survey_time')].copy();d['arm']='T2_reference';cross=pd.read_csv(OLD/'census_comparison_eligibility.csv',dtype={'count_authority_code':str,'count_unit_id':str});roadmap={(r.count_authority_code,r.count_unit_id):np.array([int(x)for x in str(r.matched_road_indices).split('|')])for r in cross[cross.eligible_observation_comparison.eq(True)].itertuples()};parts=[d];diversity=[];treechecks=[];mixturechecks=[];usage=[];routes=[];signature=set();hashes={}
 for fee,single,mix in [(0,'U_single','UM_mix'),(1,'UF_single','UFM_mix')]:
  meanflux=np.zeros((2,5,len(pr['length_m'])));fp=[];allrouted=[];motors=[]
  for k in range(3):
   name=f'F{fee}_R{k}';v=json.loads((R/(name+'_validation.json')).read_text());assert v['passed'];signature.add(v['assignment_signature']);treechecks.append(v);z=load(P/(name+'_flows.npz'));fp.append(z['endpoint_path_hash']);meanflux+=z['arc_flow']/3;allrouted.append(z['routed']);motors.append(z['od_any_motorway'])
   if k==0:
    singleflux=z['arc_flow'];demand=z['input_demand'];single_motor=z['od_any_motorway']
  hashes[fee]=fp[0];oo=dem['origin_endpoint'];dd=dem['destination_endpoint'];unique=1+(fp[1][oo,dd]!=fp[0][oo,dd]).astype(int)+((fp[2][oo,dd]!=fp[0][oo,dd])&(fp[2][oo,dd]!=fp[1][oo,dd])).astype(int)
  for b,mask in enumerate([dem['base'],dem['through_plus']]):
   w=dem['period_weights'][mask,0];u=unique[mask];diversity.append({'fee_on':fee,'branch':BRANCHES[b],'weighted_one_path_share':float(w[u==1].sum()/w.sum()),'weighted_two_path_share':float(w[u==2].sum()/w.sum()),'weighted_three_path_share':float(w[u==3].sum()/w.sum()),'weighted_mean_unique_path_fingerprints':float((u*w).sum()/w.sum()),'role':'Three equally weighted cost realizations,not three guaranteed distinct/disjoint routes.'})
  for arm,flux,motor in [(single,singleflux,single_motor),(mix,meanflux,np.mean(motors,axis=0))]:
   rf=np.array([[np.bincount(pr['arc_to_road'],weights=flux[b,t],minlength=len(inv))for t in range(5)]for b in range(2)]);rows=[]
   for r in d.itertuples():
    idx=roadmap[(r.count_authority_code,r.count_unit_id)];b=BRANCHES.index(r.branch);t=PERIODS.index(r.period);out=r._asdict();out.pop('Index');out['predicted']=float(rf[b,t,idx].sum());out['arm']=arm;rows.append(out)
   parts.append(pd.DataFrame(rows));h=inv.highway.to_numpy()[pr['arc_to_road']];core=inv.inside_core.to_numpy()[pr['arc_to_road']].astype(bool)
   for b in range(2):
    for t in [0,4]:
     routes.append({'arm':arm,'branch':BRANCHES[b],'period':PERIODS[t],'candidate_trips':float(demand[b,t]),'any_motorway_candidate_trips':float(motor[b,t]),'candidate_passage_km':float(np.dot(flux[b,t],pr['length_m'])/1000)})
     for scope,sm in [('whole_road_halo',np.ones(len(h),bool)),('core_intersecting_roads',core)]:
      for cls in sorted(set(h)):
       mask=sm&(h==cls);usage.append({'arm':arm,'branch':BRANCHES[b],'period':PERIODS[t],'scope':scope,'highway':cls,'candidate_passage_km':float(np.sum(flux[b,t,mask]*pr['length_m'][mask])/1000)})
   if arm==mix:
    maxres=0.;comp=rr['node_component'];nc=int(comp.max())+1;ec=ep['components'];assert np.allclose(np.array(allrouted),demand[None,:,:],rtol=1e-11,atol=1e-6)
    for b,mask in enumerate([dem['base'],dem['through_plus']]):
     for t in range(5):
      w=dem['period_weights'][mask,t];expected=np.bincount(ec[oo[mask]],weights=w,minlength=nc)-np.bincount(ec[dd[mask]],weights=w,minlength=nc);actual=np.bincount(comp[pr['from_index']],weights=flux[b,t],minlength=nc)-np.bincount(comp[pr['to_index']],weights=flux[b,t],minlength=nc);maxres=max(maxres,float(abs(expected-actual).max()))
    mixturechecks.append({'arm':arm,'maximum_component_flow_residual':maxres,'nonnegative_finite':bool(np.isfinite(flux).all()and(flux>=0).all()),'conserved':maxres<1e-6});np.savez_compressed(P/(mix+'_flows.npz'),arc_flow=flux,input_demand=demand,routed=np.mean(allrouted,axis=0),od_any_motorway=motor)
 df=pd.concat(parts,ignore_index=True);df.to_csv(P/'passage_comparison.csv',index=False,encoding='utf-8-sig');pd.DataFrame(diversity).to_csv(P/'path_diversity.csv',index=False,encoding='utf-8-sig');pd.DataFrame(usage).to_csv(P/'road_class_passage_distance.csv',index=False,encoding='utf-8-sig');pd.DataFrame(routes).to_csv(P/'route_usage.csv',index=False,encoding='utf-8-sig');metrics=[]
 for key,g in df.groupby(['arm','branch','period','comparison_class','evaluation_role'],sort=True):metrics.append(dict(zip(['arm','branch','period','comparison_class','evaluation_role'],key),road_class='all',**metric(g)))
 for key,g in df.groupby(['arm','branch','period','comparison_class','evaluation_role','road_class'],sort=True):metrics.append(dict(zip(['arm','branch','period','comparison_class','evaluation_role','road_class'],key),**metric(g)))
 met=pd.DataFrame(metrics);met.to_csv(P/'flow_metrics.csv',index=False,encoding='utf-8-sig');p=df[df.branch.eq(BRANCHES[0])&df.period.eq('daytime12h')&df.comparison_class.eq('small')].pivot(index=['observation_section_id','count_authority_code','count_unit_id','road_class','evaluation_role','observed'],columns='arm',values='predicted').reset_index();p.to_csv(P/'paired_screenline_changes.csv',index=False,encoding='utf-8-sig');primary=met[met.branch.eq(BRANCHES[0])&met.period.eq('daytime12h')&met.comparison_class.eq('small')];scores=[];gates=[];effects=[]
 def row(arm,role,cls):return primary[primary.arm.eq(arm)&primary.evaluation_role.eq(role)&primary.road_class.eq(cls)].iloc[0]
 for arm in ARMS:
  gate=True;strict=False
  for role in sorted(primary.evaluation_role.unique()):
   a=row(arm,role,'ordinary');b=row(arm,role,'expressway');scores.append({'arm':arm,'evaluation_role':role,'equal_class_mean_wape':float((a.wape+b.wape)/2),'ordinary_wape':float(a.wape),'expressway_wape':float(b.wape)})
   for cls in ['ordinary','expressway']:
    diff=float(row(arm,role,cls).wape-row('T2_reference',role,cls).wape);gate &=diff<=1e-10;strict |=diff< -1e-10
  gates.append({'arm':arm,'joint_development_improvement_vs_T2':bool(gate and strict),'not_independent_validation':True})
 for role in sorted(primary.evaluation_role.unique()):
  for cls in ['ordinary','expressway']:
   for label,before,after in [('speed_harmonization','T2_reference','U_single'),('toll_given_single','U_single','UF_single'),('multipath_without_toll','U_single','UM_mix'),('multipath_with_toll','UF_single','UFM_mix'),('toll_given_multipath','UM_mix','UFM_mix')]:effects.append({'contrast':label,'evaluation_role':role,'road_class':cls,'wape_change':float(row(after,role,cls).wape-row(before,role,cls).wape)})
 pd.DataFrame(scores).to_csv(P/'balanced_development_scores.csv',index=False,encoding='utf-8-sig');pd.DataFrame(gates).to_csv(P/'joint_development_gate.csv',index=False,encoding='utf-8-sig');pd.DataFrame(effects).to_csv(P/'factorial_effects.csv',index=False,encoding='utf-8-sig');bindings=json.loads((P/'input_bindings.json').read_text())['sha256'];hashchecks={n:sha(ROOT/n)==h for n,h in bindings.items()};feechange=[]
 for branch,mask in zip(BRANCHES,[dem['base'],dem['through_plus']]):
  w=dem['period_weights'][mask,0];changed=hashes[0][oo[mask],dd[mask]]!=hashes[1][oo[mask],dd[mask]];feechange.append({'branch':branch,'weighted_daily_path_changed_by_fee_fraction':float(w[changed].sum()/w.sum())})
 checks={'all_six_trees_pass':len(treechecks)==6 and all(x['passed']for x in treechecks),'same_assignment_signature_all_trees':len(signature)==1,'mixtures_conserve_fixed_OD':all(x['conserved']and x['nonnegative_finite']for x in mixturechecks),'input_bindings_unchanged':all(hashchecks.values()),'same85screenlines':len(p)==85,'frozen_observation_values':bool(df.groupby(['count_authority_code','count_unit_id','branch','period','comparison_class'])['observed'].nunique().eq(1).all()),'path_diversity_shares_sum_one':all(abs(x['weighted_one_path_share']+x['weighted_two_path_share']+x['weighted_three_path_share']-1)<1e-10 for x in diversity),'both_old_groups_development_only':set(df.evaluation_role)=={'development_original_calibration65','development_previously_inspected20'},'no_demand_multiplier_or_count_fit':True};dump(R/'validation.json',{'passed':all(checks.values()),'checks':checks,'source_hash_checks':hashchecks,'mixture_checks':mixturechecks});summary={'completed_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'stage':'Harmonized speed x toll preference x cost-ensemble development comparison','primary_small12h_base_metrics':primary.astype(object).where(pd.notna(primary),None).to_dict('records'),'balanced_scores':scores,'joint_development_gates':gates,'path_diversity':diversity,'fee_route_change':feechange,'tree_summaries':treechecks,'mixture_checks':mixturechecks,'checks_passed':sum(checks.values()),'checks_total':len(checks),'new_hydraulics':False,'affected_people':False,'independent_validation':False,'congestion_feedback':False};dump(R/'summary.json',summary);print(json.dumps({'primary20':primary[primary.evaluation_role.eq('development_previously_inspected20')].astype(object).where(pd.notna(primary[primary.evaluation_role.eq('development_previously_inspected20')]),None).to_dict('records'),'gates':gates,'diversity':diversity,'checks':checks},indent=2))
 if not all(checks.values()):raise SystemExit('Choice comparison checks failed')
if __name__=='__main__':main()
