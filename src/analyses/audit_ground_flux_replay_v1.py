#!/usr/bin/env python3
"""Original numerical gates on unchanged B view,plus all new G-face diagnostics.
No physical/numerical threshold is relaxed. Local closure remains unattributed where
cell-grouped floating arithmetic is not independently logged.
"""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np,pandas as pd
from netCDF4 import Dataset as NativeDataset,chartostring
from scipy.sparse import coo_matrix
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/analyses'),str(ROOT/'src/preprocessing')]
TAG='three_ward_ground_flux_replay_v1';P=ROOT/'data/processed'/TAG;R=ROOT/'data/exp/estimation'/TAG
NATIVE=ROOT/'data/exp/data-preprocessing/three_ward_protection_requirements_register_v1';RAIN=ROOT/'data/exp/data-preprocessing/three_ward_zone_rain_flux_readiness_v1';RECV=ROOT/'data/exp/data-preprocessing/three_ward_target_boundary_budget_audit_v1'
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
class BVariable:
 def __init__(self,var,name):self.var=var;self.name=name
 def __getattr__(self,n):return getattr(self.var,n)
 def __getitem__(self,key):
  if self.name=='crosssection_name':return self.var[:6763,:][key]
  if isinstance(key,tuple):return self.var[key[0],:6763][(...,)+key[1:]]
  return self.var[key,:6763]
class BHistory:
 def __init__(self,path,*a,**kw):self.d=NativeDataset(path,*a,**kw)
 def __enter__(self):return self
 def __exit__(self,*a):return self.d.__exit__(*a)
 def __getattr__(self,n):return getattr(self.d,n)
 def __getitem__(self,n):return BVariable(self.d[n],n) if n in ['crosssection_name','crosssection_discharge'] else self.d[n]
def unchanged_maps(new,old):
 checks=[];runtime_metadata={}
 with NativeDataset(new/'sfincs_map.nc') as nd,NativeDataset(old/'sfincs_map.nc') as od:
  nd.set_auto_mask(False);od.set_auto_mask(False);assert set(nd.variables)==set(od.variables)
  for name,ov in od.variables.items():
   nv=nd[name];assert ov.shape==nv.shape and ov.dtype==nv.dtype
   if name=='total_runtime':
    before=float(np.asarray(ov[:]).reshape(-1)[0]);after=float(np.asarray(nv[:]).reshape(-1)[0])
    assert np.isfinite(before) and np.isfinite(after) and min(before,after)>=0
    runtime_metadata[name]={'original_seconds':before,'replay_seconds':after,'role':'wallclock cost,not a physical state;source writes tfinish_all-tstart_all'}
    continue
   if ov.ndim and ov.shape[0]==109:
    for i in range(0,109,4):assert np.asarray(ov[i:i+4]).tobytes()==np.asarray(nv[i:i+4]).tobytes(),('Map differs',name,i)
   else:assert np.asarray(ov[:]).tobytes()==np.asarray(nv[:]).tobytes(),('Map differs',name)
   checks.append(name)
 for n in ['udt_output_clock.csv','udt_step_budget.csv','udt_rain_integral.csv','udt_negative_cell_summary.csv']:
  assert sha(new/n)==sha(old/n),('Original diagnostic differs',n)
 # The positivity reporting scalar held is reduced over OpenMP threads.
 # Its final reduction order can differ without changing q,storage or budgets.
 pn=pd.read_csv(new/'udt_positivity_steps.csv',float_precision='round_trip')
 po=pd.read_csv(old/'udt_positivity_steps.csv',float_precision='round_trip')
 assert pn.shape==po.shape and list(pn.columns)==list(po.columns)
 held='held_requested_transport_m3'
 for c in po.columns:
  if c!=held:assert np.array_equal(pn[c],po[c]),('Positivity state differs',c)
 difference=abs(pn[held].to_numpy()-po[held].to_numpy())
 threads=json.loads((P/'run_manifest.json').read_text())['budget']['cpus']
 bound=4*threads*np.finfo('float64').eps*np.maximum(abs(pn[held].to_numpy()),abs(po[held].to_numpy()))
 assert np.all(difference<=bound),'Reporting difference exceeds conservative thread-reduction rounding bound'
 reporting={'held_reporting_rows_different':int((difference>0).sum()),'max_abs_held_reporting_difference_m3':float(difference.max()),'counts_min_factors_and_actual_clocks_exact':True,'comparison':'reporting-only positive-sum reduction bound4*threads*float64_epsilon*max(abs(old),abs(new));original positivity admission unchanged'}
 return checks,reporting,runtime_metadata

def main(case):
 start=time.monotonic();run=next(r for r in json.loads((P/'run_manifest.json').read_text())['runs'] if r['case_id']==case);folder=ROOT/run['folder'];old=ROOT/run['original_folder'];out=R/'execution'/case;out.mkdir(parents=True,exist_ok=True)
 assert all(sha(folder/n)==h for n,h in run['input_sha256'].items())
 oldcrs=(old/'sfincs.crs').read_bytes();newcrs=(folder/'sfincs.crs').read_bytes();assert newcrs.startswith(oldcrs)
 maps,positivity_reporting,runtime_metadata=unchanged_maps(folder,old);print('ALL_PHYSICAL_MAP_VARIABLES_AND_NATIVE_CLOCK_LEDGERS_BITWISE_IDENTICAL',len(maps),flush=True)
 with NativeDataset(folder/'sfincs_his.nc') as nd,NativeDataset(old/'sfincs_his.nc') as od:
  nd.set_auto_mask(False);od.set_auto_mask(False)
  assert np.array_equal(nd['time'][:],od['time'][:]);assert np.array_equal(nd['crosssection_name'][:6763],od['crosssection_name'][:])
  for i in range(0,len(od['time']),512):assert np.asarray(nd['crosssection_discharge'][i:i+512,:6763]).tobytes()==np.asarray(od['crosssection_discharge'][i:i+512]).tobytes(),('Original B q differs',i)
 print('ALL_ORIGINAL_B_DISCHARGES_BITWISE_IDENTICAL',flush=True)
 # Metadata-only adapter:all original gate functions and thresholds run unchanged.
 # Additional G observations are audited separately below;B geometry is identical.
 import audit_cross_event_protection_run_v1 as original
 original.OUT=str(P.relative_to(ROOT));original.REPORT=str(R.relative_to(ROOT));receiver=original.receiver_geometry
 original.receiver_geometry=lambda _:receiver(old)
 original.Dataset=lambda path,*a,**kw:BHistory(path,*a,**kw) if Path(path).name=='sfincs_his.nc' else NativeDataset(path,*a,**kw)
 original.main('execution',case)
 admission=json.loads((out/'audit.json').read_text());assert admission['all_checks_passed'] and admission['admission']['technical_reference']
 clock=pd.read_csv(folder/'udt_output_clock.csv',float_precision='round_trip');led=pd.read_csv(folder/'udt_step_budget.csv',float_precision='round_trip');hist=clock[clock.write_history=='T'];mc=clock[clock.write_map=='T'];dt=led.actual_step_dt.to_numpy();hb=hist.step_index.to_numpy(dtype=int);ms=mc.step_index.to_numpy(dtype=int);nt=len(hist)
 with np.load(ROOT/'data/processed/three_ward_protection_requirements_register_v1/zones'/f'{case}.npz') as z:nodegrid=z['node_grid_index'];labels=z['zone_index'];zids=z['zone_ids']
 mapping=pd.read_csv(P/'mapping_probe'/case/'crosssection_native_mapping.csv');nz=len(zids)
 with np.load(ROOT/'data/processed/three_ward_barrier_drainage_v1/measures/spatial_contract.npz') as z:nativegrid=z['native_grid_north_index'];kcs=z['native_kcs']
 gz=np.zeros(1012*770,dtype=int);gz[nodegrid]=labels;nzlabel=gz[nativegrid]
 fromc=mapping['From Native Cell'].to_numpy(dtype=int);toc=mapping['To Native Cell'].to_numpy(dtype=int);sgn=mapping['Positive Native Sign'].to_numpy(dtype=int)
 def incidence(start,stop):
  rows=[];cols=[];vals=[]
  for si in range(start,stop):
   for cell,v in [(fromc[si],sgn[si]),(toc[si],-sgn[si])]:
    if kcs[cell]==1 and nzlabel[cell]>0:rows.append(si-start);cols.append(nzlabel[cell]-1);vals.append(v)
  return coo_matrix((vals,(rows,cols)),shape=(stop-start,nz)).tocsr()
 nsec=len(mapping);G=incidence(6763,nsec);B=incidence(0,6763);gp=G.maximum(0);gn=(-G).maximum(0)
 assert np.max(abs(np.asarray(G.sum(axis=1)).ravel()))<=1
 groundout=np.zeros((nt,nz));groundin=np.zeros_like(groundout);facearith=np.zeros_like(groundout);maxclosed=0.;closed=mapping['Runtime Closed'].to_numpy(dtype=bool)
 with NativeDataset(folder/'sfincs_his.nc') as d:
  d.set_auto_mask(False);assert [str(v).strip() for v in chartostring(d['crosssection_name'][:])]==mapping.Section.to_list()
  for i in range(0,nt,256):
   q=np.asarray(d['crosssection_discharge'][i:i+256],dtype='float32');assert np.isfinite(q).all() and q.min()>-99990
   maxclosed=max(maxclosed,float(np.max(abs(q[:,closed]),initial=0.)));assert maxclosed==0
   v32=(q*dt[hb[i:i+len(q)],None].astype('float32')).astype('float64');v64=q.astype('float64')*dt[hb[i:i+len(q)],None]
   pos=np.maximum(v32[:,6763:],0);neg=np.maximum(-v32[:,6763:],0)
   groundout[i:i+len(q)]=gp.T.dot(pos.T).T+gn.T.dot(neg.T).T
   groundin[i:i+len(q)]=gp.T.dot(neg.T).T+gn.T.dot(pos.T).T
   diff=v32-v64;facearith[i:i+len(q)]=B.T.dot(diff[:,:6763].T).T+G.T.dot(diff[:,6763:].T).T
  co=np.vstack([np.zeros(nz),np.cumsum(groundout,axis=0)])[ms];ci=np.vstack([np.zeros(nz),np.cumsum(groundin,axis=0)])[ms];ar=np.vstack([np.zeros(nz),np.cumsum(facearith,axis=0)])[ms]
 co-=co[0];ci-=ci[0];ar-=ar[0]
 bounds=np.zeros((109,nz))
 with NativeDataset(folder/'sfincs_map.nc') as d:
  d.set_auto_mask(False)
  for frame in range(109):
   vf=np.flipud(np.asarray(d['subgrid_volume'][frame],dtype='float32')).ravel()[nodegrid];step=np.maximum(abs(np.nextafter(vf,np.float32(np.inf))-vf),abs(vf-np.nextafter(vf,np.float32(-np.inf)))).astype('float64')*.5
   bounds[frame]=np.bincount(labels[labels>0]-1,weights=step[labels>0],minlength=nz)
 bounds+=bounds[0].copy();bounds[0]=0
 records=[]
 for zi,zid in enumerate(zids):
  for f in range(109):records.append({'Case':case,'Zone':str(zid),'Frame':f,'Actual Time':float(mc.actual_state_time.iloc[f]),'Ground Outflow Since First Map':float(co[f,zi]),'Ground Inflow Since First Map':float(ci[f,zi]),'Saved Face Product Arithmetic Difference':float(ar[f,zi]),'Map Storage Change Quantization Bound':float(bounds[f,zi]),'Independent Cell Grouped Roundoff Saved':False})
 tr=pd.DataFrame(records)
 st=pd.read_csv(NATIVE/'zone_storage_native.csv',float_precision='round_trip');st=st[st.Case==case]
 rn=pd.read_csv(RAIN/'zone_rain_trajectories.csv',float_precision='round_trip');rn=rn[rn.Case==case]
 rv=pd.read_csv(RECV/'receiving_exchange_trajectories.csv',float_precision='round_trip');rv=rv[rv.Case==case]
 m=tr.merge(st[['Case','Zone','Frame','Stored Volume']],on=['Case','Zone','Frame'],validate='one_to_one').merge(rn[['Case','Zone','Frame','Reconstructed Rain Since First Map']],on=['Case','Zone','Frame'],validate='one_to_one').merge(rv[['Case','Zone','Frame','Receiving Outflow Since First Map','Receiving Inflow Since First Map']],on=['Case','Zone','Frame'],validate='one_to_one').sort_values(['Zone','Frame'])
 m['Storage Change Since First Map']=m['Stored Volume']-m.groupby('Zone')['Stored Volume'].transform('first')
 m['Local Closure Difference']=m['Storage Change Since First Map']-m['Reconstructed Rain Since First Map']+m['Receiving Outflow Since First Map']-m['Receiving Inflow Since First Map']+m['Ground Outflow Since First Map']-m['Ground Inflow Since First Map']
 m['Closure Difference Meaning']='remaining source reconstruction,grouped continuity roundoff and map write quantization;not independently isolated numerical error'
 m['Required Pump Capacity']=np.nan;m.to_csv(out/'zone_ground_budget_trajectories.csv',index=False)
 result={'passed':True,'case':case,'positivity_reporting_comparison':positivity_reporting,'nonphysical_runtime_metadata':runtime_metadata,'original_numerical_gates_passed':True,'all_physical_map_variables_bitwise_identical':maps,'all_actual_clocks_and_step_budgets_bitwise_identical':True,'all_original_B_q_bitwise_identical':True,'new_ground_face_count':run['new_ground_sections'],'all_new_q_finite':True,'all_runtime_closed_face_q_zero':True,'maximum_closed_face_flow_m3_s':maxclosed,'ground_exchange_collected':True,'zones':nz,'map_rows':len(m),'maximum_absolute_local_closure_difference_m3':float(m['Local Closure Difference'].abs().max()),'maximum_map_storage_change_quantization_bound_m3':float(bounds.max()),'maximum_saved_face_product_arithmetic_difference_m3':float(abs(ar).max()),'independent_cell_grouped_roundoff_saved':False,'local_residual_independently_closed':False,'no_capacity_or_pump_count_inferred':True,'seconds':time.monotonic()-start}
 dump(out/'replay_validation.json',result);print('REPLAY_FULLY_QUALIFIED',case,json.dumps(result),flush=True)
if __name__=='__main__':main(sys.argv[1])
