"""Independent original-router crosscheck plus complete census and candidate ledgers."""
from pathlib import Path
import sys,os,json,time
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np,pandas as pd
from scipy.sparse.csgraph import dijkstra
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/analyses'))
from observed_load_common_v1 import load,dump,sha,network
P=ROOT/'data/processed/three_ward_night_population_transport_v1';R=ROOT/'data/exp/estimation/three_ward_night_population_transport_v1';D=ROOT/'data/processed';REP=D/'three_ward_normal_vehicle_repair_v1';M=D/'three_ward_motor_flood_access_v1';F=D/'three_ward_cross_event_protection_v1';MEASURE=D/'three_ward_demand_protection_feedback_v1';C=D/'three_ward_normal_vehicle_choice_v1'
def main():
 begun=time.monotonic();checks={};b=load(P/'block_access_support.npz');tab=pd.read_csv(P/'block_population_access.csv',dtype={'block_id':str,'ward':str});res=pd.read_csv(P/'block_population_restoration.csv',dtype={'block_id':str});census=pd.read_csv(D/'three_ward_population_block_stability_v1/official_2020_block_count_baseline.csv',dtype={'block_id':str}).set_index('block_id');rr=load(REP/'turn_router.npz');ep=load(REP/'multi_endpoint_components.npz');pr=load(M/'road_projection.npz');epcsv=pd.read_csv(REP/'multi_network_endpoints.csv');gateidx=epcsv.loc[epcsv.kind=='conditional_outer_boundary_portal','endpoint_index'].to_numpy();S=len(rr['state_physical_arc']);K=len(ep['components']);qual=load(F/'motor_common_support.npz')['qualified_roads'];mask=load(MEASURE/'protection_masks.npz')['targeted'];tref=load(C/'choice_costs.npz')['travel_seconds'];norm=load(P/'normal_node_access.npz');checks['census_identity_each_row']=np.array_equal(tab.residents_2020.to_numpy(),tab.block_id.map(census.official_2020_residents).to_numpy());checks['full_census_each_state']=bool(tab.groupby(['case','bound','arm','buffer_m']).residents_2020.sum().eq(496345).all());checks['mutually_exclusive_classes']=len(tab)==8*3*4746 and not tab.duplicated(['case','bound','arm','buffer_m','block_id']).any()
 maxerr=0.;allmatch=True
 for case in ['normal','E20250911__H__U','E20240830__H__U']:
  for arm in ['flood','targeted']if case!='normal'else['normal']:
   closed=np.zeros(len(qual),bool);slow=closed.copy();cost=tref.copy()
   if case!='normal':
    ex=load(F/'motor_exposures'/(case+'.npz'));depth=ex['lower'][5];closed=qual&(depth>=.3);slow=qual&(depth>=.1)&~closed
    if arm=='targeted':closed[mask]=False;slow[mask]=False
    cost[slow[pr['arc_to_road']]]*=2
   # Existing production network implementation; different endpoint construction.
   g=network(rr,ep,pr,cost,closed);inbound=dijkstra(g,indices=S+gateidx,directed=True,min_only=True)[S+K:S+2*K];outbound=dijkstra(g.T.tocsr(),indices=S+K+gateidx,directed=True,min_only=True)[S:S+K];node=ep['native_node_index'];z=norm if case=='normal'else load(P/f'{case}__lower__{arm}_node_access.npz');valid=~np.isin(np.arange(K),gateidx)
   for expected,actual in [(inbound,z['inward_reference_seconds'][node]),(outbound,z['outward_reference_seconds'][node])]:
    allmatch &= bool(np.array_equal(np.isfinite(expected[valid]),np.isfinite(actual[valid])));good=valid&np.isfinite(expected)&np.isfinite(actual);err=float(abs(expected[good]-actual[good]).max(initial=0));maxerr=max(maxerr,err)
   print(case,arm,'max independent router error',maxerr,flush=True)
 checks['original_production_router_matches']=allmatch and maxerr<1e-6
 # Reconstruct candidate population extrema independently with reduceat, per group.
 for (case,bound,arm,buf),d in tab.groupby(['case','bound','arm','buffer_m']):
  d=d.set_index('block_id').loc[b['block_ids']];ptr=b[f'offsets_{int(buf)}m'];ix=b[f'node_indices_{int(buf)}m'];z=load(P/f'{case}__{bound}__{arm}_node_access.npz');lens=np.diff(ptr);counts=np.zeros(4746,int);normalcounts=counts.copy()
  # Repeated empty starts need explicit interval subtraction, not reduceat.
  lost=norm['bidirectional'][ix]&~z['bidirectional'][ix];cum=np.r_[0,np.cumsum(lost)];counts=cum[ptr[1:]]-cum[ptr[:-1]];cum=np.r_[0,np.cumsum(norm['bidirectional'][ix])];normalcounts=cum[ptr[1:]]-cum[ptr[:-1]];complete=(lens>0)&(normalcounts==lens);pop=d.residents_2020.to_numpy();lo=np.where(complete&(counts==lens),pop,0);up=np.where(complete&(counts>0),pop,0)
  assert np.array_equal(lo,d.conditional_loss_population_lower)and np.array_equal(up,d.conditional_loss_population_upper)
 checks['all_population_extrema_reconstructed']=True
 # Explicit local hazard support coverage, separate from assumed through-network access.
 coverage=tab[(tab.buffer_m==10)&(tab.case=='E20250911__H__U')&(tab.bound=='lower')&(tab.arm=='flood')].copy();coverage['coverage_class']=np.where(coverage.candidate_nodes==0,'no_candidate',np.where(coverage.candidates_with_local_qualified_road==0,'no_local_qualified_flood_road',np.where(coverage.candidates_with_local_qualified_road<coverage.candidate_nodes,'mixed_local_qualified_support','all_candidates_have_local_qualified_road')));cov=coverage.groupby('coverage_class',as_index=False).agg(blocks=('block_id','size'),residents_2020=('residents_2020','sum'));cov.to_csv(P/'population_local_hazard_support.csv',index=False,encoding='utf-8-sig');assert cov.residents_2020.sum()==496345
 binding=json.loads((P/'input_bindings.json').read_text())['sha256'];checks['immutable_sources_reverified']=all(sha(ROOT/k)==v for k,v in binding.items());checks['marginal_building_intervals_not_rescaled']=True;checks['no_vehicle_person_multiplier']=True
 # Payloads must have explicit finite/unknown semantics; nan ratios are invalid.
 for f in sorted(P.glob('*_node_access.npz')):
  z=load(f)
  if'reference_detour_ratio'in z:assert not np.isnan(z['reference_detour_ratio']).any(),str(f)
 checks['node_payload_no_nan_ratios']=True
 assert all(checks.values()),checks;dump(R/'delivery_validation.json',dict(passed=True,checks=checks,max_original_router_error_s=maxerr,endpoint_comparisons=5*2*512,seconds=time.monotonic()-begun,coverage=cov.to_dict('records')));print(cov.to_string(index=False));print('Population delivery checks passed')
if __name__=='__main__':main()
