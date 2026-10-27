#!/usr/bin/env python3
"""Validate tensor/graph equivalence, physical forcings and supervision isolation."""
from pathlib import Path
import importlib.util,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def main():
 root=ROOT;dest=root/'data/processed/three_ward_surrogate_inputs';out=root/'data/exp/data-preprocessing/three_ward_surrogate_inputs'
 loader=module('input_loader',root/'src/preprocessing/load_three_ward_surrogate_inputs.py')
 build=module('input_builder',root/'src/preprocessing/prepare_three_ward_surrogate_inputs.py')
 s=loader.load_static();g=loader.load_graph();schema=json.loads((dest/'input_schema.json').read_text());v=json.loads((out/'validation.json').read_text())
 checks={}
 checks['static_shape_and_units']=s['Static Tensor'].shape==(28,1012,770) and len(schema['static']['channel_units'])==28
 assert np.isfinite(s['Static Tensor']).all()
 valid=s['Eligible Ground Mask'];index=g['Node Grid Index'];r,c=np.unravel_index(index,valid.shape)
 assert np.array_equal(index,np.flatnonzero(valid))
 assert np.array_equal(g['Node Features'],s['Static Tensor'][:,r,c].T)
 assert np.array_equal(g['Node Feature Valid'],s['Static Feature Valid'][:,r,c].T)
 checks['graph_raster_features_identical']=True
 ei=g['Surface Edge Index'];ef=g['Surface Edge Features'];face=g['Surface Edge Native Face Id'];n=len(index);half=ei.shape[1]//2
 assert np.array_equal(ei[:,:half],ei[::-1,half:]);assert np.array_equal(face[:half],face[half:])
 assert np.all((ei>=0)&(ei<n));assert np.all(ei[0]!=ei[1]);assert np.all(ef[:,0]>0) and np.all(ef[:,1]>=0) and np.all(ef[:,1]<=ef[:,0])
 assert np.allclose(g['Node Position'][ei[1]]-g['Node Position'][ei[0]],ef[:,4:6])
 assert np.all((abs(ef[:,4])+abs(ef[:,5]))==10)
 assert not np.isin(face,g['Blocked Native Face Id']).any()
 checks['surface_edges_are_symmetric_open_neighbours']=True
 re=g['Road Edge Index'];rf=g['Road Edge Native Face Id'];assert np.array_equal(re,ei[:,ef[:,1]>0]) and np.array_equal(rf,face[ef[:,1]>0])
 checks['road_edges_are_verified_surface_subset']=True
 link=g['Receiver Link Index'];assert np.all((link[0]>=0)&(link[0]<n)) and np.all((link[1]>=0)&(link[1]<len(g['Boundary Low Stage'])))
 assert np.all(s['Native Hydraulic Mask'][g['Boundary Point Row'],g['Boundary Point Column']]==2)
 assert np.all(~valid[g['Boundary Point Row'],g['Boundary Point Column']])
 checks['boundary_context_separate_from_surface_targets']=True
 assert len(g['Blocked Native Face Id'])==40310 and len(set(g['Blocked Native Face Id']))==40310
 checks['all_verified_closed_faces_excluded']=True
 # Analytic orientation check independent of the original face generator.
 ar,ac,br,bc=build.endpoints(np.array([20,19]),np.array([40,20]),np.array([0,1]))
 assert np.array_equal(np.c_[ar,ac,br,bc],[[1,1,1,2],[1,1,0,1]])
 checks['east_north_face_orientation_analytic_case']=True
 # Independent direct aggregation over irregular fine ground, with one building.
 z=np.arange(400,dtype=np.float32).reshape(20,20)/100;cls=np.zeros((20,20),np.uint8);cls[0,0]=1;rough=np.full((20,20),.04,np.float32);source=np.ones((20,20),np.uint8);hyd=z.copy();hyd[0,0]+=10
 st,b=build.statistics(z,cls,rough,source,hyd,np.ones((1,1),bool),np.ones((1,1),bool))
 a=z[cls!=1].astype('float64');assert abs(float(st[0,0,0])-a.mean())<1e-6 and st[4,0,0]==99.75 and abs(float(st[3,0,0])-a.std())<1e-6
 checks['ground_features_match_independent_direct_sum']=True
 label_names={'Label Lower','Label Upper','Exact Label Mask','Interval Label Mask','Quarantine Mask','Quality Flags'}
 totals={}
 for case in schema['cases']:
  prescribed,y=loader.load_control(case)
  assert label_names.issubset(y) and not(label_names&prescribed.keys())
  assert np.array_equal(prescribed['Event Time'],schema['dynamic']['event_time_seconds'])
  assert prescribed['Effective Surface Rainfall Rate'].shape==(8,1012,770)
  assert prescribed['River Stage'].shape==(8,10488) and np.isfinite(prescribed['River Stage']).all()
  rr=loader.forcing_rasters(prescribed,g)
  assert np.array_equal(rr['River Stage Raster'][:,g['Boundary Point Row'],g['Boundary Point Column']],prescribed['River Stage'])
  assert not np.any(rr['Stage Known']&valid)
  assert np.all(rr['Physical Rainfall Raster'][:,valid]==prescribed['Physical Rainfall Rate'][:,None])
  reconstructed=g['Boundary Low Stage'][None,:]+prescribed['River Stage Increment By Group'][:,g['Boundary Group Index']]
  assert np.max(abs(reconstructed-prescribed['River Stage']))<1e-5
  exact=y['Exact Label Mask'];iv=y['Interval Label Mask'];q=y['Quarantine Mask'];l=y['Label Lower'];u=y['Label Upper']
  assert not np.any(exact&iv) and not np.any((exact|iv)&q)
  assert not np.any((exact|iv)&~valid)
  assert np.all(u[exact]-l[exact]<=1e-7+1e-7)
  assert np.all((u[iv]-l[iv]>.00000005)&(u[iv]-l[iv]<=.010001))
  assert np.all(q[y['Numerical Volume Flag']&valid]) and np.all(q[y['Assumed Barrier Overtopping Flag']&valid])
  assert np.all((exact|iv|q|y['Wide Interval Mask'])[valid])
  # All masks correspond to original unchanged regular-grid label bounds.
  with np.load(root/'data/processed/three_ward_common_depth_labels'/(case+'_corrected_10m.npz')) as original:
   assert np.array_equal(l,original['Peak Mean Depth Lower'],equal_nan=True) and np.array_equal(u,original['Peak Mean Depth Upper'],equal_nan=True)
  totals[case]=dict(exact=int(exact.sum()),interval=int(iv.sum()),quarantine=int(q.sum()),shared_pool_has_exact_support=bool(exact.any()))
 checks['all_forcing_shapes_and_boundary_roundtrip']=True
 checks['all_cases_preserve_labels_and_exclude_quarantine']=True
 checks['no_response_flags_in_predictor_inputs']=not any('Volume Flag' in x or 'Overtopping' in x or 'Label' in x for x in schema['static']['channels'])
 checks['all_main_comparisons_have_shared_parent_support']=all(x['shared_pool_has_exact_support'] for x in totals.values())
 checks['current_rain_budget_preserved']=all(x['rain_volume_relative_error']<1e-6 for x in v['checks'].values())
 checks['no_training_or_new_scenarios']=not schema['training'] and not schema['new_scenarios']
 assert all(checks.values()),checks
 manifest=[dict(path=str(p.relative_to(root)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(dest.iterdir()) if p.is_file()]
 v.update(valid=True,independent_checks=checks,case_admission_counts=totals,loader_code_sha256=sha(root/'src/preprocessing/load_three_ward_surrogate_inputs.py'),validator_code_sha256=sha(root/'src/analyses/validate_three_ward_surrogate_inputs.py'))
 (out/'validation.json').write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
 (out/'product_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
 print(json.dumps(dict(valid=True,independent_checks=len(checks),cases=totals,output_files=len(manifest),total_bytes=sum(x['bytes'] for x in manifest)),indent=2))
if __name__=='__main__':main()
