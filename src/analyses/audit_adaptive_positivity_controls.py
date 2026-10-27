"""Audit adaptive controls on their actual integration clock and physical native face topology."""
import os
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
from pathlib import Path
import sys,json,argparse
import numpy as np
from netCDF4 import Dataset,chartostring
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_adaptive_positivity import OUT,REPORT,CASES
from prepare_sfincs_cpu_build import sha,dump
from reference_label_policy import balance_check,integrate_rate_at

def structured(p):return np.atleast_1d(np.genfromtxt(p,delimiter=',',names=True,dtype=None,encoding='utf8'))
def arr(v):return np.ma.asarray(v,dtype='f8').filled(np.nan)
def topology(p):
 with p.open('rb')as f:
  n,nf,nc,nb=map(int,np.fromfile(f,'<i4',4));d={'np':n,'npuv':nf,'ncuv':nc,'nb':nb}
  for key,typ,count in [('row','<i4',n),('col','<i4',n),('mask','i1',n),('level','i1',n),('a','<i4',nf),('b','<i4',nf),('direction','i1',nf),('face_level','i1',nf),('kcuv','i1',nf),('boundary','<i4',nb),('ibuvdir','i1',nb),('combined','<i4',nc),('child1','<i4',nc),('child2','<i4',nc),('west','<i4',n),('east','<i4',n),('south','<i4',n),('north','<i4',n)]:d[key]=np.fromfile(f,typ,count)
  assert not f.read(), 'unexpected topology bytes'
 return d

def main(only=None):
 out=ROOT/OUT;report=ROOT/REPORT;manifest=json.loads((out/'run_manifest.json').read_text());g=dict(np.load(ROOT/'data/processed/three_ward_adaptive_connectivity/adaptive_cell_geometry.npz'));fg=dict(np.load(ROOT/'data/processed/three_ward_adaptive_connectivity/adaptive_face_geometry.npz'))
 summaries={};checks=[]
 def check(case,name,ok,details=None):checks.append({'case':case,'check':name,'pass':bool(ok),'details':details});print(('PASS 'if ok else'FAIL ')+case+' '+name,flush=True)
 for run in manifest['runs']:
  cid=run['case_id']
  if only and cid!=only:continue
  path=ROOT/run['folder'];ex=json.loads((path/'execution.json').read_text());led=structured(path/'udt_step_budget.csv');clk=structured(path/'udt_output_clock.csv');final=structured(path/'udt_final_clock.csv')[0];lim=structured(path/'udt_positivity_steps.csv');native=structured(path/'udt_adaptive_cell_summary.csv');exchange=structured(path/'udt_adaptive_exchange.csv');top=topology(path/'udt_adaptive_topology.bin')
  t=led['actual_state_time'];dt=led['actual_step_dt'];n=len(t)-1;inside=top['mask']==1
  check(cid,'completed_full_existing_control',ex['returncode']==0 and ex['completion_marker']and final['solver_error']==0 and 14400<=t[-1]<=14400+dt[-1]+1e-8)
  check(cid,'bound_input_hashes_and_identity',all(sha(path/k)==h for k,h in run['input_sha256'].items())and(path/'udt_run_identity.txt').read_bytes()==(path/'udt_diagnostic_identity.txt').read_bytes())
  check(cid,'actual_step_clock_complete',np.array_equal(led['step_index'],np.arange(n+1))and t[0]==dt[0]==0 and np.all(np.diff(t)>0)and max(abs(np.diff(t)-dt[1:]))<1e-8 and final['actual_final_time']==t[-1]and final['step_index']==n)
  check(cid,'all_ledger_quantities_finite',all(np.all(np.isfinite(led[k]))for k in led.dtype.names))
  bal=balance_check(t,led['signed_storage'],led['rain_applied_cumulative'],led['boundary_out_cumulative'],led['boundary_in_cumulative'])
  check(cid,'unchanged_formal_endpoint_and_prefix_balance_gates',bal['endpoint_pass']and bal['prefix_pass'],{'endpoint_m3':float(bal['residual'][-1]),'maximum_absolute_m3':float(max(abs(bal['residual'])))})
  check(cid,'same_step_budget_export_reconstructs',max(abs(bal['residual']-led['budget_residual']))<1e-7 and np.all(led['other_source_cumulative']==0)and np.all(led['other_sink_cumulative']==0))
  check(cid,'physical_topology_matches_frozen_geometry',np.array_equal(top['a']-1,fg['cell_a'])and np.array_equal(top['b']-1,fg['cell_b'])and np.array_equal(top['mask'],g['mask'])and np.array_equal(top['direction'],fg['direction'])and np.array_equal(top['level']-1,g['level'])and np.array_equal(10/2.**(top['face_level']-1),fg['length_pixels']*.5))
  # Validate each coarse aggregate owns two distinct physical child faces and consistent orientation/endpoints.
  f1=top['child1']-1;f2=top['child2']-1;physical1=(f1>=0)&(f1<top['npuv']);physical2=(f2>=0)&(f2<top['npuv']);dummy=top['npuv']+top['ncuv'];paired=physical1&physical2;children=np.r_[f1[physical1],f2[physical2]]
  check(cid,'coarse_fine_aggregates_use_physical_children_or_closed_zero_dummy',top['ncuv']>0 and np.all(physical1|(f1==dummy))and np.all(physical2|(f2==dummy))and np.all(physical1|physical2)and np.all(f1!=f2)and np.all(top['direction'][f1[paired]]==top['direction'][f2[paired]])and np.all(top['face_level'][f1[paired]]==top['face_level'][f2[paired]])and len(np.unique(children))==len(children),{'one_closed_missing_child_aggregates':int((~paired).sum())})
  check(cid,'all_interior_native_minima_and_times_exported',len(native)==top['np']and np.array_equal(native['native_cell_index'],np.arange(1,top['np']+1))and np.array_equal(native['kcs'],top['mask'])and np.all((native['minimum_step']>=0)&(native['minimum_step']<=n))and np.array_equal(native['minimum_time_s'][inside],t[native['minimum_step'][inside]]))
  strict=bool(np.all(native['minimum_volume_m3'][inside]>=0)and np.all(native['strict_negative_steps'][inside]==0)and np.min(led['minimum_native_volume_since_initialization'])>=0 and np.all(led['negative_volume_sum']==0))
  check(cid,'strict_nonnegative_at_every_applied_step',strict)
  check(cid,'native_final_storage_reduces_to_same_step_ledger',float(np.cumsum(native['final_volume_m3'][inside])[-1])==float(led['signed_storage'][-1]))
  check(cid,'no_real_negative_crossings',sum(1 for _ in (path/'udt_adaptive_negative_events.csv').open())==1)
  check(cid,'limiter_diagnostics_bind_every_native_step',len(lim)==n and np.array_equal(lim['actual_end_time_s'],t[1:])and np.array_equal(lim['dt_s'],dt[1:])and np.all((lim['minimum_factor']>=0)&(lim['minimum_factor']<=1))and np.all(lim['held_requested_transport_m3']>=0)and np.any(lim['limited_faces']>0))
  check(cid,'all_closed_receiving_faces_have_zero_actual_flux',len(exchange)==n and np.array_equal(exchange['step'],np.arange(1,n+1))and np.array_equal(exchange['time_s'],t[1:])and np.all(exchange['closed_max_abs_q']==0))
  group_out=sum(exchange[f'group{c}_out']for c in [17,51,71,79,106,110,111]);group_in=sum(exchange[f'group{c}_in']for c in [17,51,71,79,106,110,111]);maxgroup=max(float(np.max(abs(np.r_[0,np.cumsum(group_out)]-led['boundary_out_cumulative']))),float(np.max(abs(np.r_[0,np.cumsum(group_in)]-led['boundary_in_cumulative']))))
  check(cid,'group_split_gross_exchange_matches_native_ledger',maxgroup<1e-5,{'maximum_reduction_difference_m3':maxgroup})
  trace=np.loadtxt(path/'udt_rain_decode.csv',delimiter=',',skiprows=2,ndmin=2);intended=np.array([0,3600,3601,7200,7201,14400],dtype='f8')
  with Dataset(path/'roof_surface_precip.nc')as d:
   d.set_auto_mask(False)
   raw=arr(d['time'][:]);rx=arr(d['x'][:]);ry=arr(d['y'][:]);field=arr(d['Precipitation'][:]);check(cid,'seconds_preserved_from_original_minute_coordinates',np.array_equal(trace[:,0],raw)and np.array_equal(trace[:,1],intended)and d['time'].units=='minutes since 2020-01-01')
  check(cid,'all_original_forcing_values_finite_including_valid_zero_rain',np.all(np.isfinite(field))and np.all(field>=0))
  # Galibier adds half a forcing pixel to supplied coordinates before bilinear interpolation.
  sample_x=g['x']-2.5;sample_y=g['y']-2.5
  ix=(sample_x-rx[0])/5;iy=(sample_y-ry[0])/5;i0=np.floor(ix).astype(int);j0=np.floor(iy).astype(int);fx=ix-i0;fy=iy-j0
  check(cid,'forcing_samples_are_nested_5m_coordinates',np.allclose(fx,np.where(g['size_m']==10,.5,0.),atol=1e-6)and np.allclose(fy,np.where(g['size_m']==10,.5,0.),atol=1e-6))
  rates=[];maxpattern=0.
  for k in range(len(raw)):
   # Source coordinate convention is verified against its contributing-area weights below.
   val=(1-fy)*((1-fx)*field[k,j0,i0]+fx*field[k,j0,i0+1])+fy*((1-fx)*field[k,j0+1,i0]+fx*field[k,j0+1,i0+1])
   rates.append(float(np.sum(val[inside]*g['size_m'][inside]**2)/3.6e6))
   intensity=0. if cid=='dry_raised'or k not in [2,3]else 60.
   expected=intensity*g['contributing_area_m2']/g['size_m']**2
   maxpattern=max(maxpattern,float(np.max(abs(val[inside]-expected[inside]))))
  pattern_bound=float(4*np.finfo('f4').eps*max(1.,float(np.max(60*g['contributing_area_m2']/g['size_m']**2))));check(cid,'roof_donor_assignment_conserved_in_native_sampling',maxpattern<=pattern_bound,{'max_rate_difference_mm_h':maxpattern,'four_float32_epsilon_bound_mm_h':pattern_bound})
  encoded=float(np.trapezoid(rates,intended));continuous=integrate_rate_at(intended,np.asarray(rates),np.minimum(t,14400));rainerror=float(np.max(abs(led['rain_applied_cumulative']-continuous)))
  check(cid,'whole_domain_rain_volume_matches_original_control',abs(encoded-(0. if cid=='dry_raised'else 2338608.))<1 and abs(float(led['rain_applied_cumulative'][-1])-encoded)<=max(1.,encoded)*1e-6 and rainerror<=2*float(max(dt))*max(rates)+max(1.,encoded)*1e-6,{'encoded_m3':encoded,'max_actual_minus_continuous_m3':rainerror})
  check(cid,'no_rain_before_original_second_onset',np.all(led['rain_applied_cumulative'][t<=3600]==0))
  check(cid,'one_second_linear_meteo_brackets',np.all(led['meteo_bracket_observed'][1:]==1)and np.all(led['meteo_update_interval'][1:]==1)and np.all(led['meteo_bracket_end'][1:]-led['meteo_bracket_start'][1:]==1)and np.all(t[1:]>=led['meteo_bracket_start'][1:]-1e-8)and np.all(t[1:]<=led['meteo_bracket_end'][1:]+1e-8))
  steps=clk['step_index'];check(cid,'output_actual_clock_binds_accounted_states',np.array_equal(clk['actual_state_time'],t[steps])and np.array_equal(clk['actual_step_dt'],dt[steps]))
  maps=clk[clk['write_map']=='T'];hist=clk[clk['write_history']=='T'];peaks=clk[clk['write_peak']=='T'];fields_ok=True;storage_ok=True;censor_ok=True;maxdiff=0.;maxbound=0.
  with Dataset(path/'sfincs_map.nc')as d:
   check(cid,'adaptive_map_geometry_and_complete_horizon',np.array_equal(arr(d['msk'][:]),g['mask'])and np.max(abs(arr(d['mesh2d_face_x'][:])-g['x']))<.001 and np.max(abs(arr(d['mesh2d_face_y'][:])-g['y']))<.001 and len(d['time'])==len(maps)==49 and np.array_equal(arr(d['time'][:]),np.arange(0,14401,300)))
   peak=arr(d['zsmax'][0]);zmin=arr(d['zb'][:]);peak_ok=True
   for j,row in enumerate(maps):
    vol=arr(d['subgrid_volume'][j]);v=vol[inside];zs=arr(d['zs'][j]);h=arr(d['h'][j]);fields_ok &=all(np.all(np.isfinite(arr(d[k][j])[inside]))for k in ['subgrid_volume','u','v'])
    delta=abs(float(v.sum())-led['signed_storage'][row['step_index']]);vf=v.astype('f4');bound=float(np.sum(np.maximum(abs(np.nextafter(vf,np.float32(np.inf))-vf),abs(vf-np.nextafter(vf,np.float32(-np.inf)))).astype('f8')*.5))+1e-7;storage_ok &=delta<=bound;maxdiff=max(maxdiff,delta);maxbound=max(maxbound,bound)
    both=inside&np.isfinite(zs)&np.isfinite(h);censor_ok &=np.all(abs(h[both]-(zs[both]-zmin[both]))<1e-5)
    # A censored state has no resolved depth exceeding1cm; online peaks bound all uncensored states.
    peak_ok &=bool(np.all(~np.isfinite(zs[inside])|np.isfinite(peak[inside])|(zs[inside]<=zmin[inside]+.01+1e-5))and np.all(zs[inside&np.isfinite(zs)&np.isfinite(peak)]<=peak[inside&np.isfinite(zs)&np.isfinite(peak)]+1e-5))
   check(cid,'finite_native_state_fields',fields_ok and np.all(np.isfinite(zmin[inside])))
   check(cid,'every_map_storage_matches_same_step_with_float32_write_bound',storage_ok,{'maximum_difference_m3':maxdiff,'maximum_bound_m3':maxbound})
   check(cid,'known_shallow_censoring_and_online_peak_envelope',censor_ok and peak_ok)
   check(cid,'one_complete_online_peak_window',len(peaks)==1 and peaks['actual_peak_start'][0]==0 and peaks['actual_peak_end'][0]>=14400 and len(d['timemax'])==1)
  with Dataset(path/'sfincs_his.nc')as d:
   check(cid,'history_and_solver_status_bind_clock',np.all(arr(d['status'][:])==0)and len(d['time'])==len(hist)and np.array_equal(arr(d['time'][:]),hist['nominal_output_time'].astype('f4').astype('f8')))
  summary={'steps':n,'native_cells':top['np'],'physical_faces':top['npuv'],'combined_faces':top['ncuv'],'closed_receiving_faces':int(np.sum(top['kcuv'][top['boundary']-1]==0)),'final_time_s':float(t[-1]),'minimum_native_volume_m3':float(np.min(native['minimum_volume_m3'][inside])),'negative_native_cells':int(np.sum(native['minimum_volume_m3'][inside]<0)),'max_abs_budget_residual_m3':float(max(abs(bal['residual']))),'final_budget_residual_m3':float(bal['residual'][-1]),'rain_applied_m3':float(led['rain_applied_cumulative'][-1]),'gross_out_m3':float(led['boundary_out_cumulative'][-1]),'gross_in_m3':float(led['boundary_in_cumulative'][-1]),'final_storage_m3':float(led['signed_storage'][-1]),'runtime_s':ex['runtime_seconds'],'all_checks_passed':all(c['pass']for c in checks if c['case']==cid)}
  summaries[cid]=summary
 result={'cases':summaries,'checks':checks,'all_checks_passed':all(c['pass']for c in checks),'training':0,'batch58':0};dump(report/(((only+'_')if only else'')+'audit.json'),result)
 assert result['all_checks_passed'],[c for c in checks if not c['pass']]
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--only',choices=CASES);main(a.parse_args().only)
