"""Lossless forcing access, split custody and Galibier forcing export. No execution API.

This loader is a preparation interface. No normalizer is fitted and no responses
are released. Sealing is an application-level protocol, not filesystem encryption.
"""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
LIB='data/processed/three_ward_minimum_scenarios_v1'

def read_arrays(path):
 with np.load(path,allow_pickle=False) as d:return {k:d[k] for k in d.files}
def catalog(root=ROOT):return json.loads((Path(root)/LIB/'catalog.json').read_text())
def find_case(case_id,root=ROOT):
 matches=[c for c in catalog(root)['cases'] if c['case_id']==case_id]
 if len(matches)!=1:raise KeyError(case_id)
 return matches[0]
def load_forcing(case_id,root=ROOT):
 """Forcing QC may inspect any split, including sealed inputs, without responses."""
 case=find_case(case_id,root)
 d=read_arrays(Path(root)/LIB/case['forcing_file'])
 return {k:d[k] for k in ['Event Time','Physical Rainfall Rate','River Stage Increment By Group']}
def load_geometry(root=ROOT):return read_arrays(Path(root)/LIB/'shared_geometry.npz')
def materialize_at_knots(case_id,grid='10m',root=ROOT):
 """Opt-in full arrays; typically use factorized storage to avoid duplication."""
 if grid not in ('10m','5m'):raise ValueError(grid)
 d=load_forcing(case_id,root);g=load_geometry(root)
 d['Effective Surface Rainfall Rate']=d['Physical Rainfall Rate'][:,None,None]*g['Effective Rainfall Weight '+grid][None]
 d['River Stage']=g['Boundary Low Stage'][None]+d['River Stage Increment By Group'][:,g['Boundary Group Index']]
 return d

def padded_forcing_batch(case_ids,root=ROOT):
 """Keep every native time knot. Time Valid masks padding; no time resampling."""
 items=[load_forcing(c,root) for c in case_ids]
 if not items:raise ValueError('Empty batch')
 n=max(len(d['Event Time']) for d in items)
 t=np.zeros((len(items),n),'f8');rain=t.copy();stage=np.zeros((len(items),n,7),'f8');valid=np.zeros(t.shape,bool)
 for i,d in enumerate(items):
  k=len(d['Event Time']);t[i,:k]=d['Event Time'];rain[i,:k]=d['Physical Rainfall Rate'];stage[i,:k]=d['River Stage Increment By Group'];valid[i,:k]=True
 return {'Event Time':t,'Physical Rainfall Rate':rain,'River Stage Increment By Group':stage,'Time Valid':valid}

def bind_descendant(case_id,grid,requested_split=None,root=ROOT):
 if grid not in ('regular_10m','adaptive_10_5m'):raise ValueError(grid)
 case=find_case(case_id,root)
 if requested_split is not None and requested_split!=case['split']:raise ValueError('Descendants must inherit the parent family partition')
 return {k:case[k] for k in ('case_id','parent_id','family_id','split','forcing_sha256')}|{'grid':grid}

def response_access(case_id,purpose='train',root=ROOT):
 """Never infer release from presence of response files or a user-supplied flag."""
 case=find_case(case_id,root)
 if case['split'] in ('sealed_test','challenge'):raise PermissionError('Sealed responses: explicit future review/release protocol required')
 if purpose not in ('train','validation','open_control'):raise ValueError(purpose)
 expected={'train':'train','validation':'validation','open_control':'open_control'}[purpose]
 if case['split']!=expected:raise PermissionError('Wrong response partition for requested purpose')
 raise PermissionError('Reference response admission and training release are not authorized in this forcing-only library')

def export_galibier(case_id,destination,grid='regular_10m',root=ROOT):
 """Write precipitation NetCDF, boundary stage and original boundary-point order.

 Coordinates deliberately follow the pinned Galibier reader's corner convention:
 it adds a half cell on both axes. They are NOT generic CF centre coordinates.
 One zero halo cell is preserved. Actual zero rain is never a missing value.
 Piecewise-linear interpolation (ampr_block=0) must be retained in any future run.
 No sfincs.inp, executable invocation, or response folder is created here.
 """
 from netCDF4 import Dataset
 if grid not in ('regular_10m','adaptive_10_5m'):raise ValueError(grid)
 d=load_forcing(case_id,root);g=load_geometry(root);dest=Path(destination)
 if dest.exists() and any(dest.iterdir()):raise FileExistsError('Refuse to replace existing forcing export')
 dest.mkdir(parents=True,exist_ok=True)
 fine=grid=='adaptive_10_5m';dx=5 if fine else 10
 w=np.flipud(np.pad(g['Effective Rainfall Weight '+('5m' if fine else '10m')],1))
 ny,nx=w.shape;x=-11360-dx+np.arange(nx)*dx;y=-42440-dx+np.arange(ny)*dx
 with Dataset(dest/'roof_surface_precip.nc','w',format='NETCDF4') as ds:
  ds.createDimension('time',len(d['Event Time']));ds.createDimension('y',ny);ds.createDimension('x',nx)
  tv=ds.createVariable('time','f8',('time',));tv.units='minutes since 2020-01-01 00:00:00';tv[:]=d['Event Time']/60
  for name,v in [('x',x),('y',y)]:
   a=ds.createVariable(name,'f8',(name,));a.units='m';a[:]=v
  rain=ds.createVariable('Precipitation','f4',('time','y','x'),fill_value=-9999.,zlib=True,complevel=2,chunksizes=(1,min(256,ny),min(256,nx)))
  rain.units='mm h-1';rain.unit='mm.hr-1'
  ds.coordinate_convention='Pinned Galibier lower-left corner values; reader adds halfcell; not CF cell centres'
  ds.interpolation_required='ampr_block=0 (piecewise linear); preserve dtwnd=1 of existing configuration'
  ds.forcing_case=case_id;ds.grid=grid;ds.forcing_sha256=find_case(case_id,root)['forcing_sha256']
  for i,r in enumerate(d['Physical Rainfall Rate']):rain[i]=(r*w).astype('f4')
 stage=g['Boundary Low Stage'][None]+d['River Stage Increment By Group'][:,g['Boundary Group Index']]
 np.savetxt(dest/'sfincs.bzs',np.column_stack([d['Event Time'],stage]),fmt='%.10f')
 # Keep the original coordinate order, not rounded/reconstructed graph positions.
 src=Path(root)/'data/processed/three_ward_sfincs_framework/adaptive_connectivity_comparison/rain60_low_corrected_10m/sfincs.bnd'
 (dest/'sfincs.bnd').write_bytes(src.read_bytes())
 (dest/'export_contract.json').write_text(json.dumps({'case_id':case_id,'grid':grid,'forcing_sha256':find_case(case_id,root)['forcing_sha256'],
  'interpolation':'piecewise_linear','corner_convention':True,'zero_is_missing':False,'solver_executed':False,
  'initial_state':'dry interior','measure':'no_action','height_epoch_certified':False},indent=2)+'\n')
 return dest
