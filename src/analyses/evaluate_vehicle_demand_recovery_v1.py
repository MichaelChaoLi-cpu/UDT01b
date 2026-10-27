"""Compare separate additive demand arms on frozen observed screenlines.
Both original groups are development; no coefficient or route-cost fitting.
"""
from pathlib import Path
import os,sys,json,datetime as dt
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/analyses'))
from evaluate_normal_vehicle_baseline_v1 import metric
P=ROOT/'data/processed/three_ward_vehicle_demand_recovery_v1';R=ROOT/'data/exp/estimation/three_ward_vehicle_demand_recovery_v1';C=ROOT/'data/processed/three_ward_normal_vehicle_choice_v1';B=ROOT/'data/processed/three_ward_normal_vehicle_baseline_v1';M=ROOT/'data/processed/three_ward_motor_flood_access_v1'
ARMS=['U_reference','C_coarse_only','I_intrazonal_only'];PERIODS=['daily','hour8','hour10','hour17','daytime12h'];BRANCHES=['base_no_remote_remote','plus_conditional_chord_through']
def load(p):
 with np.load(p,allow_pickle=False)as z:return{k:z[k]for k in z.files}
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item()if isinstance(x,np.generic)else str(x))+'\n')
def save(n,x):pd.DataFrame(x).to_csv(P/n,index=False,encoding='utf-8-sig')
def sha(p):
 import hashlib
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def wquant(x,w,q):
 order=np.argsort(x,kind='stable');return float(x[order][np.searchsorted(np.cumsum(w[order]),q*w.sum())])
def main():
 routing=json.loads((R/'routing_validation.json').read_text());assert routing['passed'];prep=json.loads((R/'preparation_summary.json').read_text());assert all(prep['checks'].values());z=load(P/'increment_flows.npz');pr=load(M/'road_projection.npz');inv=pd.read_csv(M/'road_inventory.csv');nr=len(inv);base=pd.read_csv(C/'passage_comparison.csv',dtype={'count_authority_code':str,'count_unit_id':str,'observation_section_id':str});base=base[base.arm.eq('U_single')].copy();base['arm']='U_reference';cross=pd.read_csv(B/'census_comparison_eligibility.csv',dtype={'count_authority_code':str,'count_unit_id':str});cuts={(r.count_authority_code,r.count_unit_id):np.array([int(x)for x in str(r.matched_road_indices).split('|')])for r in cross[cross.eligible_observation_comparison].itertuples()};parts=[base];rf=np.array([[np.bincount(pr['arc_to_road'],weights=z['arc_flow'][b,t],minlength=nr)for t in range(5)]for b in range(2)]);usage=[]
 for b,arm in enumerate(ARMS[1:]):
  rows=[]
  for r in base.itertuples():
   x=r._asdict();x.pop('Index');delta=float(rf[b,PERIODS.index(r.period),cuts[(r.count_authority_code,r.count_unit_id)]].sum());x.update(arm=arm,predicted=r.predicted+delta);rows.append(x)
  parts.append(pd.DataFrame(rows));h=inv.highway.to_numpy()[pr['arc_to_road']];core=inv.inside_core.to_numpy()[pr['arc_to_road']].astype(bool)
  for t in [0,4]:
   for scope,sm in [('whole_road_halo',np.ones(len(h),bool)),('core_intersecting_roads',core)]:
    for cls in sorted(set(h)):
     mask=sm&(h==cls);usage.append({'arm':arm,'period':PERIODS[t],'scope':scope,'highway':cls,'added_candidate_passage_km':float(np.dot(z['arc_flow'][b,t,mask],pr['length_m'][mask])/1000),'added_flow_arcs':int((z['arc_flow'][b,t,mask]>1e-10).sum())})
 df=pd.concat(parts,ignore_index=True);save('passage_comparison.csv',df);save('added_road_class_usage.csv',usage);metrics=[]
 for key,g in df.groupby(['arm','branch','period','comparison_class','road_class'],sort=True):
  fields=dict(zip(['arm','branch','period','comparison_class','road_class'],key));metrics.append(dict(fields,evaluation_role='all85_development',**metric(g)))
  for role,gg in g.groupby('evaluation_role'):metrics.append(dict(fields,evaluation_role=role,**metric(gg)))
 met=pd.DataFrame(metrics);save('flow_metrics.csv',met);primary=met[met.branch.eq(BRANCHES[0])&met.period.eq('daytime12h')&met.comparison_class.eq('small')];save('primary_small12h_metrics.csv',primary);pair=df[df.branch.eq(BRANCHES[0])&df.period.eq('daytime12h')&df.comparison_class.eq('small')].pivot(index=['observation_section_id','count_authority_code','count_unit_id','road_class','evaluation_role','observed'],columns='arm',values='predicted').reset_index()
 for arm in ARMS[1:]:
  pair[arm+'_added_passages']=pair[arm]-pair.U_reference;pair[arm+'_absolute_error_change']=abs(pair[arm]-pair.observed)-abs(pair.U_reference-pair.observed)
 save('paired_screenline_changes.csv',pair);effects=[];gates=[]
 for arm in ARMS[1:]:
  ok=True;strict=False
  for role in sorted(primary.evaluation_role.unique()):
   for cls in ['ordinary','expressway']:
    ref=primary[primary.arm.eq('U_reference')&primary.evaluation_role.eq(role)&primary.road_class.eq(cls)].iloc[0];rr=primary[primary.arm.eq(arm)&primary.evaluation_role.eq(role)&primary.road_class.eq(cls)].iloc[0];pg=pair[pair.road_class.eq(cls)]
    if role!='all85_development':pg=pg[pg.evaluation_role.eq(role)]
    added=float((pg[arm]-pg.U_reference).sum());change=float(rr.wape-ref.wape);gap=float(pg.observed.sum()-pg.U_reference.sum());effects.append({'arm':arm,'evaluation_role':role,'road_class':cls,'screenlines':len(pg),'observed':float(pg.observed.sum()),'baseline_prediction':float(pg.U_reference.sum()),'new_prediction':float(pg[arm].sum()),'added_passages':added,'wape_change':change,'baseline_wape':float(ref.wape),'new_wape':float(rr.wape),'baseline_prediction_observation_ratio':float(ref.prediction_observation_ratio),'new_prediction_observation_ratio':float(rr.prediction_observation_ratio),'baseline_aggregate_shortfall_recovered_fraction':added/gap if gap>0 else None,'paired_error_improved':int(pg[arm+'_absolute_error_change'].lt(-1e-8).sum()),'paired_error_worsened':int(pg[arm+'_absolute_error_change'].gt(1e-8).sum()),'paired_error_unchanged':int(pg[arm+'_absolute_error_change'].abs().le(1e-8).sum())})
    if role!='all85_development':ok&=change<=1e-10;strict|=change< -1e-10
  gates.append({'arm':arm,'joint_relative_improvement_in_both_original_development_groups':bool(ok and strict),'absolute_adequacy_certified':False,'independent_validation':False})
 save('separate_demand_effects.csv',effects);save('joint_development_gate.csv',gates)
 dem=load(P/'increment_demands.npz');routes=[]
 for b,arm in enumerate(ARMS[1:]):
  mask=dem['increment_index']==b;x=z['trip_length_m'][mask]/1000;w=dem['period_weights'][mask,0];t=z['trip_cost_seconds'][mask]/60;motor=z['trip_any_motorway'][mask];
  for period,tidx in [('daily',0),('daytime12h',4)]:
   ww=dem['period_weights'][mask,tidx];routes.append({'arm':arm,'period':period,'added_candidate_trips':float(ww.sum()),'motorway_using_candidate_trips':float(ww[motor].sum()),'motorway_using_candidate_share':float(ww[motor].sum()/ww.sum()),'mean_route_km':float(np.dot(x,ww)/ww.sum()),'weighted_p50_route_km':wquant(x,ww,.5),'weighted_p95_route_km':wquant(x,ww,.95),'maximum_route_km':float(x.max()),'mean_U_cost_minutes':float(np.dot(t,ww)/ww.sum()),'routes_over10km_daily_weight':float(ww[x>10].sum()),'note':'Native access to native access; coarse remote trip only modeled road-halo segment; intrazonal routes can leave the source zone. All are location/clock assumptions.'})
 save('increment_route_diagnostics.csv',routes)
 # Frozen original two external branches differ by exactly their original flow difference.
 invariant=0.
 for arm in ARMS:
  sub=df[df.arm.eq(arm)].set_index(['count_authority_code','count_unit_id','branch','period','comparison_class']);old=base.set_index(['count_authority_code','count_unit_id','branch','period','comparison_class'])
  for branch in BRANCHES:
   changed=sub.xs(branch,level='branch').predicted-old.xs(branch,level='branch').predicted
   if branch==BRANCHES[0]:first=changed
   else:invariant=max(invariant,float(abs(changed-first).max()))
 originalfine=pd.read_csv(B/'conditional_fine_driver_od.csv',dtype={'origin_code':str,'destination_code':str,'o':str,'d':str});originaldem=load(B/'routing_demands.npz');newcoarse=pd.read_csv(P/'coarse_source_controls.csv',dtype={'origin_middle_zone':str,'destination_middle_zone':str});newintra=pd.read_csv(P/'intrazonal_source_controls.csv',dtype={'origin_code':str,'destination_code':str});existingmid=set(zip(originalfine.o,originalfine.d));oldelig=set(zip(originalfine.loc[originaldem['through_plus'],'origin_code'],originalfine.loc[originaldem['through_plus'],'destination_code']));source_disjoint=not bool(set(zip(newcoarse.origin_middle_zone,newcoarse.destination_middle_zone))&existingmid);intra_disjoint=not bool(set(zip(newintra.origin_code,newintra.destination_code))&oldelig)
 bindings=json.loads((P/'input_bindings.json').read_text())['sha256'];checks={'preparation_integrity':all(prep['checks'].values()),'routing_integrity':routing['passed'],'same85_unique_observed_screenlines':len(pair)==85 and not pair.duplicated(['count_authority_code','count_unit_id']).any(),'observed_values_unchanged':bool(df.groupby(['count_authority_code','count_unit_id','branch','period','comparison_class'])['observed'].nunique().eq(1).all()),'coarse_controls_not_previously_fine_allocated':source_disjoint,'intrazonal_controls_not_in_original_routed_set':intra_disjoint,'only_two_separate_additions_no_combination':set(df.arm)==set(ARMS),'added_screenline_passages_nonnegative':bool((pair[[a+'_added_passages'for a in ARMS[1:]]]>=-1e-9).all().all()),'external_through_difference_exactly_preserved':invariant<1e-8,'original_development_roles_unchanged':set(df.evaluation_role)==set(base.evaluation_role),'original_inputs_unchanged':all(sha(ROOT/n)==h for n,h in bindings.items())};summary={'completed_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'date_jst':'2026-10-18','arms':ARMS,'fixed_cost':'Original U single F0_R0,zero toll preference,no perturbation','added_source':prep['by_increment'],'primary_all85_small12h':primary[primary.evaluation_role.eq('all85_development')].astype(object).where(pd.notna(primary[primary.evaluation_role.eq('all85_development')]),None).to_dict('records'),'effects':effects,'gates':gates,'route_diagnostics':routes,'route_seconds':routing['elapsed_s'],'maximum_preserved_external_branch_residual':invariant,'new_hydraulics':False,'affected_people':False,'cost_tuning':False,'independent_validation':False,'checks_passed':sum(checks.values()),'checks_total':len(checks)};dump(R/'validation.json',{'passed':all(checks.values()),'checks':checks});dump(R/'summary.json',summary);print(json.dumps({'all85_effects':[e for e in effects if e['evaluation_role']=='all85_development'],'gates':gates,'route_diagnostics':routes,'checks':checks},indent=2))
 if not all(checks.values()):raise SystemExit('Demand recovery evaluation integrity failure')
if __name__=='__main__':main()
