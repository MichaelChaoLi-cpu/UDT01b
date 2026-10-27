"""A0: input,clipping and up-to-eight-state old-router equivalence checks."""
from pathlib import Path
import sys,json,time,gzip
import numpy as np,pandas as pd
from shapely.geometry import shape,mapping,box
from shapely.ops import transform,unary_union
from pyproj import Transformer
from netCDF4 import Dataset
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/analyses'))
from temporal_transport_common_v1 import *
from integrate_night_population_transport_v1 import distances
from prepare_historical_surface_coverage_v2 import edge_maximum

def main():
 t0=time.monotonic();init();assert not(R/'A0_validation.json').exists(),'Preserve completed preflight';bindings=json.loads((PLAN/'source_bindings.json').read_text())['sha256'];assert all(sha(ROOT/n)==h for n,h in bindings.items()),'Plan-bound input changed';inventory=pd.read_csv(PLAN/'input_inventory.csv');assert len(inventory)==10;plan=json.loads((PLAN/'protocol.json').read_text());protocol=dict(plan,execution_decision='MILI-D-20261018-015',status='authorized_execution',registered_plan_sha256=sha(PLAN/'protocol.json'));dump(P/'protocol.json',protocol);save(P/'input_inventory.csv',inventory);bindings[str((PLAN/'protocol.json').relative_to(ROOT))]=sha(PLAN/'protocol.json');bindings[str((PLAN/'rain_timing_context.csv').relative_to(ROOT))]=sha(PLAN/'rain_timing_context.csv')
 inv=pd.read_csv(M/'road_inventory.csv');assert inv.source_edge_id.is_unique and np.array_equal(inv.road_index,np.arange(len(inv)));lookup={k:i for i,k in enumerate(inv.source_edge_id)};j=json.loads((D/'plateau_water/central_three_ward_boundaries.geojson').read_text());tr=Transformer.from_crs(6668,6677,always_xy=True);wards={f['properties']['ward'][:5]:transform(tr.transform,shape(f['geometry']))for f in j['features']};assert all(g.is_valid for g in wards.values());union=unary_union(list(wards.values()));bounds=box(*union.bounds);length=np.zeros((3,len(inv)));unionlength=np.zeros(len(inv));seen=np.zeros(len(inv),bool);features=[];geoms={}
 with gzip.open(N/'surface_physical_edges.jsonl.gz','rt')as f:
  for line in f:
   e=json.loads(line);i=lookup.get(e['edge_id'])
   if i is None:continue
   if seen[i]:raise RuntimeError('Duplicate native source geometry:'+e['edge_id'])
   seen[i]=True;g=shape(e['geometry'])
   if not g.intersects(bounds):continue
   clipped=g.intersection(union)
   if clipped.length==0:continue
   unionlength[i]=clipped.length;remaining=clipped
   for wi,w in enumerate(WARD):piece=remaining.intersection(wards[w]);length[wi,i]=piece.length;remaining=remaining.difference(wards[w])
   assert abs(length[:,i].sum()-clipped.length)<.001,(i,length[:,i],clipped.length)
   geoms[i]=clipped;features.append(dict(type='Feature',properties=dict(road_index=i,source_edge_id=e['edge_id']),geometry=mapping(clipped)))
 support=load(PREV/'motor_common_support.npz');q=support['qualified_roads'];protected=support['targeted_effective'];assert q.sum()==28064 and protected.sum()==118;assert seen[q].all(),'Qualified road geometry missing';error=float(abs(length.sum(0)-unionlength).sum());assert error<.001,error;np.savez_compressed(P/'clipped_road_lengths.npz',ward_length_m=length,union_length_m=unionlength,geometry_found=seen,qualified=q,protected=protected,ward_codes=np.array(WARD));dump(P/'clipped_roads_6677.geojson',dict(type='FeatureCollection',crs=dict(type='name',properties=dict(name='EPSG:6677')),features=features));dump(P/'wards_6677.geojson',dict(type='FeatureCollection',crs=dict(type='name',properties=dict(name='EPSG:6677')),features=[dict(type='Feature',properties=dict(ward=w),geometry=mapping(wards[w]))for w in WARD]));save(R/'road_length_accounting.csv',[dict(scope='Three wards',qualified_length_km=float(unionlength[q].sum()/1000),unknown_exposure_length_km=float(unionlength[~q].sum()/1000),unmapped_geometry_context_roads=int((~seen).sum()),unmapped_geometry_core_flag_roads=int((~seen&inv.inside_core.to_numpy()).sum()),ward_sum_error_m=error)]+[dict(scope=w,qualified_length_km=float(length[wi,q].sum()/1000),unknown_exposure_length_km=float(length[wi,~q].sum()/1000))for wi,w in enumerate(WARD)]);print('CLIPPING QUALIFIED',len(features),'source roads,error',error,flush=True)
 engine=Reachability();pr=engine.pr;normal=load(BASE/'normal_node_access.npz')['bidirectional'];assert np.array_equal(engine.get(np.zeros(len(q),bool)),normal);tref=load(C/'choice_costs.npz')['travel_seconds'];rows=[];oldpopulation=pd.read_csv(PREV/'population/block_population_access.csv',dtype={'block_id':str});blocks,pop,normal,infos=population_inputs()
 for case in ['E20250911__H__U','E20240830__H__U','E20250710__H__U','E20240618__H__U']:
  rr=inventory[inventory.case_id.eq(case)].iloc[0];clock=load(ROOT/rr.clock_path)['Actual Map Time'];assert len(clock)==109 and np.max(abs(clock-np.arange(109)*300))<=.001
  with Dataset(ROOT/rr.map_path)as ds:bed=np.flipud(np.ma.asarray(ds['zb'][:],dtype='f8').filled(np.nan)).ravel();eta=np.flipud(np.ma.asarray(ds['zs'][48],dtype='f8').filled(np.nan)).ravel()
  lo,up=edge_maximum(pr['point_z'],pr['point_coarse'],pr['point_refs'],pr['offsets'],q,eta,bed);oldex=load(PREV/'motor_exposures'/(case+'.npz'));assert np.array_equal(lo[q],oldex['lower'][5,q])and np.array_equal(up[q],oldex['upper'][5,q]);closed=q&(lo>=.3);slow=q&(lo>=.1)&~closed
  for arm in ['flood','targeted']:
   gate();c=closed.copy();s=slow.copy()
   if arm=='targeted':c[protected]=False;s[protected]=False
   cost=tref.copy();cost[s[pr['arc_to_road']]]*=2;start=time.monotonic();fast=engine.get(c);elapsed=time.monotonic()-start;oo,ii=distances(engine.rr,pr,cost,c,engine.gates);assert np.array_equal(fast,np.isfinite(oo)&np.isfinite(ii));old=load(PREV/'population/node_access'/(case+'__lower__'+arm+'.npz'));assert np.array_equal(fast,old['bidirectional']);lost=normal&~fast
   for buf,info in infos.items():
    n=counts(lost[info['ix']],info['ptr']);lower=np.where(info['complete']&(n==info['lens']),pop,0);upper=np.where(info['complete']&(n>0),pop,0);before=oldpopulation[(oldpopulation.case==case)&(oldpopulation.bound=='lower')&(oldpopulation.arm==arm)&(oldpopulation.buffer_m==buf)].set_index('block_id').loc[blocks.block_id];assert np.array_equal(lower,before.conditional_loss_population_lower)and np.array_equal(upper,before.conditional_loss_population_upper)
   rows.append(dict(case=case,arm=arm,seconds_reachability=elapsed,old_router_equivalent=True,population_replay_exact=True))
 save(R/'A0_benchmark.csv',rows);assert time.monotonic()-t0<=600,'Registered A0budget';dump(P/'source_bindings.json',dict(sha256=bindings));checks=dict(plan_inputs_exact=True,clipping_not_double_counted=error<.001,qualified_geometries_complete=bool(seen[q].all()),old_five_minute_depths_exact=True,eight_router_states_equivalent=len(rows)==8,old_population_bounds_exact=True,normal_baseline_exact=True,new_hydraulics=False);dump(R/'A0_validation.json',dict(passed=True,checks=checks,seconds=time.monotonic()-t0,benchmark=rows,legacy_length_note='Current km use union-clipped unique source geometry;old whole-intersecting-source lengths are not silently overwritten.',unknown_geometry_pool_count=int((~seen).sum()),known_length_km=float(unionlength[q].sum()/1000)));print('A0 ALL CHECKS PASSED',time.monotonic()-t0,flush=True)
if __name__=='__main__':main()
