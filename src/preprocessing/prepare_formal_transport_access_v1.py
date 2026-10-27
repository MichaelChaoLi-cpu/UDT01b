"""Project12 qualified hydraulic cases onto frozen road/entrance support.
No graph editing, hydraulic runs, population allocation or sealed-test access.
"""
from pathlib import Path
import os,sys,json,time
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
os.environ.setdefault('NUMBA_NUM_THREADS','2')
import numpy as np
from netCDF4 import Dataset
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/preprocessing'),str(ROOT/'src/analyses')]
from prepare_historical_surface_access_v1 import depth_at,read,write,dump,load,sha
from prepare_historical_surface_coverage_v2 import edge_maximum
OLD=ROOT/'data/processed/three_ward_surface_coverage_v2'
STRICT=ROOT/'data/processed/three_ward_surface_alternatives_v1'
T=ROOT/'data/processed/three_ward_transport_shutdown_scenarios_v1'
F=ROOT/'data/processed/three_ward_formal_transport_experiments_v1'
H=ROOT/'data/processed/three_ward_historical_radar_experiment_v2'
P=ROOT/'data/processed/three_ward_formal_transport_access_v1'
R=ROOT/'data/exp/estimation/three_ward_formal_transport_access_v1'
D=ROOT/'data/exp/research-design/three_ward_formal_transport_matrix_v1'
CASES=[f'{rain}__{stage}__{arm}'for rain in ['R1','R0','R125']for stage in ['L','H']for arm in ['T','U']]
PROJECTIONS=['strict_classified_road','ground_path_supplement'];UNCERTAINTIES=['optimistic','pessimistic'];KINDS=['no_assumed_rail_shutdown','entrance','station','section'];IDENTITIES=['original_identity','source_normalized_identity'];THRESHOLDS=[.1,.3]

def segment_max(values,offsets):
 lens=np.diff(offsets);out=np.maximum.reduceat(np.r_[values,-np.inf],offsets[:-1]);out[lens==0]=-np.inf;return out

def main():
 start=time.monotonic();P.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True);(P/'exposures').mkdir(exist_ok=True)
 runs={x['case_id']:x for x in json.loads((F/'run_manifest.json').read_text())['runs']}
 policy=json.loads((D/'protocol.json').read_text());times=np.asarray(policy['grid']['synchronized_times_s']);frames=[str(t)for t in times]+['peak_envelope']
 assert len(CASES)==12 and len(times)==12 and policy['transport_postprocessing']['maximum_network_evaluations']==9984
 common=np.load(F/'factor_comparison_v1/common_support.npz')['common_admitted'].ravel()
 e=np.load(OLD/'surface_exposure.npz');w=load(OLD/'local_walking_graph.npz');meta=read(OLD/'road_segment_inventory.csv');supp=np.load(OLD/'ground_path_supplement.npz');cache=np.load(STRICT/'fine_elevation_support.npz')
 roadcells=cache['road_cells'];groundcells=cache['ground_cells'];rz=cache['road_sorted_z'];rc=cache['road_cumulative_z'];rn=cache['road_count'];gz=cache['ground_sorted_z'];gc=cache['ground_cumulative_z'];gn=cache['ground_count']
 assert np.array_equal(roadcells,e['road_cells'])and np.array_equal(groundcells,e['ground_cells'])
 known=common[roadcells]&(rn>0);groundknown=common[groundcells]
 offsets=w['edge_sample_offsets'];indices=w['edge_sample_road_index'];lens=np.diff(offsets);grade=np.array([r['known_surface_grade']=='True'and r['ambiguous_parallel_geometry']=='False'for r in meta]);outside=np.array([r['has_out_of_grid_sample']=='True'for r in meta])
 all_known=segment_max((~known[indices]).astype(float),offsets)<=0
 unknown=~grade|outside|(lens==0)|~all_known
 qualified=supp['qualified'].copy();point_common=common[supp['coarse_flat_index']];point_refs=supp['sample_point_index'];poff=supp['sample_offsets'];qualified &=segment_max((~point_common[point_refs]).astype(float),poff)<=0
 assert not np.any(qualified&~supp['old_unknown'])and not np.any(qualified&~unknown)
 unknowns=np.stack([unknown,unknown&~qualified])
 q=[];ga=gn*.25;eo=e['entrance_offsets'];ei=e['entrance_cell_index']
 for j in range(len(eo)-1):
  cells=ei[eo[j]:eo[j+1]];total=ga[cells].sum();admitted=ga[cells][groundknown[cells]].sum();q.append(total>0 and admitted>=100 and admitted/total>=.8)
 a0=np.load(STRICT/'alternative_distances.npz');a1=np.load(OLD/'alternative_distances.npz')
 np.savez_compressed(P/'projection_support.npz',road_known=known,ground_known=groundknown,unknown=unknowns,supplement_qualified=qualified,road_cells=roadcells,ground_cells=groundcells,ground_area=ga,entrance_offsets=eo,entrance_cell_index=ei,entrance_ids=e['entrance_ids'],entrance_quality=np.array(q),grade=grade,original_supported=a0['evidence_eligible'],added_supported=a1['evidence_eligible']&~a0['evidence_eligible'],original_common_known=e['road_known'],original_supplement_qualified=supp['qualified'])
 contract=dict(human_decision='MILI-D-20261017-007',physical_cases=CASES,frame_ids=frames,times_s=list(map(int,times)),road_projections=PROJECTIONS,unknown_branches=UNCERTAINTIES,rail_shutdowns=KINDS,station_identities=IDENTITIES,road_threshold_m=THRESHOLDS,network_evaluations=9984,graph='Unchanged native directed road/rail graph; no new nodes, snapping, planar noding or mode changes.',support='Original policies retained; common12 admitted mask projected to source samples. Newly unsupported edges stay unknown; original115 and added29 cohorts retained and attrition reported.',classification='Classify each fixed projection and shutdown over8 threshold/identity/unknown branches. Outward loss and two-direction access-to-destination-set loss are separate; directions may use different targets and modes, so not same-station round trips.',two_direction_definition='At least one outward alternative and at least one return alternative to the declared destination sets, independently, using walking exits or mixed walk/rail anchors. No temporal trip feasibility or same-target round trip.',normal_baseline='Same projection/unknown branch and source graph with no flood restrictions or assumed rail closure.',isolation='Supported probe candidates under explicit assumptions, not entire neighborhoods, zero-road-isolation proof, observed shutdown or people counts.',distance='Shortest modeled walking distance; rail edges count zero walking distance. Not travel time, capacity, evacuation safety or timetable.',unknown='Bridge/tunnel/deck heights remain unknown; optimistic retains unknowns, pessimistic withholds them. Neither branch establishes actual safety.',peak='Non-simultaneous envelope separate; only fixed-time frames support synchronous access claims. No durations inferred.',new_hydraulics=False,people_counted=False,training=False,sealed_tests_opened=False)
 dump(P/'analysis_contract.json',contract)
 sources={}
 paths=[OLD/n for n in ['local_walking_graph.npz','surface_probe_inventory.csv','road_segment_inventory.csv','ground_path_supplement.npz','surface_exposure.npz']]+[T/n for n in ['hypothetical_network.npz','hypothetical_network_edges.json','portal_inventory.csv','paired_accessibility_matrices.npz','identity_attribution_sensitivity_links.csv']]+[ROOT/'data/processed/three_ward_station_surface_risk_v1/station_entrance_identity_links.csv',STRICT/'fine_elevation_support.npz',F/'factor_comparison_v1/common_support.npz',F/'run_manifest.json',D/'protocol.json',ROOT/'src/analyses/compare_historical_surface_coverage_v2.py',ROOT/'src/analyses/compare_transport_shutdown_scenarios_v1.py',ROOT/'src/preprocessing/prepare_historical_surface_coverage_v2.py']
 for path in paths:sources[str(path.relative_to(ROOT))]=sha(path)
 checks=[]
 for case in CASES:
  run=runs[case];base=H if run['reused']else F;old=run.get('original_case_id',case);folder=ROOT/run['folder'];clockpath=base/'execution'/(old+'_output_clock.npz');auditpath=ROOT/run['original_audit']if run['reused']else ROOT/'data/exp/estimation/three_ward_formal_transport_experiments_v1/execution'/case/'audit.json';audit=json.loads(auditpath.read_text());assert audit['all_checks_passed']and all(audit['required_gates'].values())
  clock=load(clockpath)['Actual Map Time'];ii=np.array([np.argmin(abs(clock-t))for t in times]);assert np.max(abs(clock[ii]-times))<=.001
  rlo=[];rup=[];glo=[];gup=[];slo=[];sup=[]
  with Dataset(folder/'sfincs_map.nc')as d:
   bed=np.flipud(np.ma.asarray(d['zb'][:],dtype='f8').filled(np.nan)).ravel()
   for j,k in enumerate(list(ii)+[None]):
    eta=np.flipud(np.ma.asarray(d['zs'][int(k)]if k is not None else d['zsmax'][0],dtype='f8').filled(np.nan)).ravel()
    a,b=depth_at(rz,rc,rn,eta[None,roadcells],bed[roadcells]);rlo.append(a[0]);rup.append(b[0]);a,b=depth_at(gz,gc,gn,eta[None,groundcells],bed[groundcells]);glo.append(a[0]);gup.append(b[0])
    a,b=edge_maximum(supp['fine_terrain_z'],supp['coarse_flat_index'],point_refs,poff,qualified,eta,bed);slo.append(a);sup.append(b)
  bounds=np.load(F/'factor_comparison_v1'/(case+'_peak_bounds.npz'))
  assert np.allclose(rlo[-1][known],bounds['road_lower'].ravel()[roadcells][known],atol=1e-6)and np.allclose(rup[-1][known],bounds['road_upper'].ravel()[roadcells][known],atol=1e-6)
  assert np.allclose(glo[-1][groundknown],bounds['ground_lower'].ravel()[groundcells][groundknown],atol=1e-6)
  np.savez_compressed(P/'exposures'/(case+'.npz'),road_lower=np.array(rlo),road_upper=np.array(rup),ground_lower=np.array(glo),ground_upper=np.array(gup),supplement_lower=np.array(slo),supplement_upper=np.array(sup),actual_times=clock[ii])
  checks.append(dict(case_id=case,original_numeric_gates=True,peak_matches_fixed_factorial_support=True,max_actual_time_error_s=float(max(abs(clock[ii]-times)))))
  for path in [clockpath,auditpath,F/'factor_comparison_v1'/(case+'_peak_bounds.npz')]:sources[str(path.relative_to(ROOT))]=sha(path)
  print('EXPOSURE',case,'elapsed',round(time.monotonic()-start),flush=True)
 dump(P/'source_bindings.json',sources)
 assert all(sha(ROOT/k)==v for k,v in sources.items())
 dump(R/'preparation_validation.json',dict(all_passed=True,checks=checks,common_known_road_cells=int(known.sum()),supported_entrances=int(sum(q)),strict_unknown_segments=int(unknowns[0].sum()),supplement_unknown_segments=int(unknowns[1].sum()),qualified_supplement_segments=int(qualified.sum()),previous_supplement_segments=int(supp['qualified'].sum()),seconds=time.monotonic()-start,new_hydraulics=False))
 print('PREPARATION COMPLETE',flush=True)
if __name__=='__main__':main()
