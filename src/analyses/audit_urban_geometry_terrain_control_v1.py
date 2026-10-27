"""Terrain-arm adapter of original-rule audit_reference_setup_pilot.py.
Own T topology and whole T storage; common open-ground exposure; unchanged gates.
No hydraulic execution or modifications to frozen experiments.
"""
import os
os.environ['NUMBA_NUM_THREADS']='2'
for key in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(key,None)
from pathlib import Path
import sys,json,csv,argparse
import numpy as np
from netCDF4 import Dataset,chartostring
from scipy.spatial import cKDTree
import rasterio
from rasterio.windows import Window
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'));sys.path.insert(0,str(ROOT/'src/analyses'))
OUT='data/processed/three_ward_urban_geometry_pair_controls_v1'
REPORT='data/exp/estimation/three_ward_urban_geometry_pair_controls_v1'
PREP=ROOT/'data/processed/three_ward_urban_geometry_pair_inputs_v1'
from prepare_sfincs_cpu_build import sha,dump
from load_three_ward_minimum_scenarios import load_forcing,load_geometry
from reference_label_policy import balance_check,integrate_rate_at,classify_labels,event_admission,CONTRACT
from audit_three_ward_negative_volumes import arr,readnp,structured,csvwrite,fine_peaks
from audit_positivity_controls import parse_summary
from audit_three_ward_reference_contract import polyline_sections
NR,NC=1012,770

def receiver_geometry(folder):
 top=readnp(PREP/'terrain_face_topology.npz');a=top['from_native'];b=top['to_native'];sr=top['native_row_south'];cc=top['native_col_east'];mask=np.fromfile(folder/'sfincs.msk',dtype='u1');pos=np.c_[-11360+(cc+.5)*10,-42440+(sr+.5)*10];sel=np.flatnonzero(((mask[a]==1)&(mask[b]==2))|((mask[a]==2)&(mask[b]==1)));mid=(pos[a[sel]]+pos[b[sel]])*.5
 sections=polyline_sections(folder/'sfincs.crs');xy=np.array([x.mean(0)for name,x in sections]);dist,ix=cKDTree(mid).query(xy);ids=sel[ix];normal=np.array([[-(x[1,1]-x[0,1]),x[1,0]-x[0,0]]for name,x in sections]);land=np.where(mask[a[ids]]==1,a[ids],b[ids]);water=np.where(mask[a[ids]]==2,a[ids],b[ids]);ok=len(ids)==len(sel)and len(np.unique(ix))==len(ix)and max(dist)<=CONTRACT['coordinate_tolerance_m']and np.all(np.sum(normal*(pos[water]-pos[land]),axis=1)>0)
 groups=np.zeros(len(mask),int)
 with(folder/'udt_negative_native_map.csv').open()as f:
  for row in csv.DictReader(f):groups[int(row['native_cell_index'])-1]=int(row['water_group_code'])
 # T has no artificial building closure: outer absent faces have no native arc.
 closed=np.zeros(len(ids),bool)
 return bool(ok),closed,groups[water],[name for name,x in sections],xy

def main(phase,case):
 out=ROOT/OUT;report=ROOT/REPORT/case/'T';(out/phase).mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True);manifest=json.loads((out/(phase+'_manifest.json')).read_text());run=next(r for r in manifest['runs']if r['case_id']==case);p=ROOT/run['folder'];ex=json.loads((p/'execution.json').read_text());checks=[]
 def check(name,ok,details=None):checks.append({'check':name,'pass':bool(ok),'details':details});print(('PASS 'if ok else'FAIL ')+case+' '+name,flush=True)
 led=structured(p/'udt_step_budget.csv');clock=structured(p/'udt_output_clock.csv');final=structured(p/'udt_final_clock.csv')[0];lim=structured(p/'udt_positivity_steps.csv');native=parse_summary(p/'udt_negative_cell_summary.csv');t=led['actual_state_time'];dt=led['actual_step_dt'];ns=len(t)-1;maps=clock[clock['write_map']=='T'];hist=clock[clock['write_history']=='T'];peaks=clock[clock['write_peak']=='T']
 check('completion',ex['returncode']==0 and ex['completion_marker']and final['solver_error']==0 and 14400<=t[-1]<=14400+dt[-1]+1e-8)
 check('identity',ex['solver_image_id']==run['image_id']==run['identity']['solver_digest']and(p/'udt_diagnostic_identity.txt').read_bytes()==(p/'udt_run_identity.txt').read_bytes()and all(sha(p/n)==v for n,v in run['input_sha256'].items()))
 check('actual_clock',t[0]==dt[0]==0 and np.array_equal(led['step_index'],np.arange(ns+1))and np.all(np.diff(t)>0)and max(abs(np.diff(t)-dt[1:]))<1e-8 and final['actual_final_time']==t[-1]and final['step_index']==ns and np.array_equal(clock['actual_state_time'],t[clock['step_index']])and np.array_equal(clock['actual_step_dt'],dt[clock['step_index']]))
 check('finite_ledger',all(np.all(np.isfinite(led[k]))for k in led.dtype.names))
 bal=balance_check(t,led['signed_storage'],led['rain_applied_cumulative'],led['boundary_out_cumulative'],led['boundary_in_cumulative'],led['other_source_cumulative'],led['other_sink_cumulative']);check('original_budget',bal['endpoint_pass']and bal['prefix_pass'],{'endpoint_m3':float(bal['residual'][-1]),'max_abs_m3':float(max(abs(bal['residual'])))})
 check('native_budget_reconstruction',max(abs(bal['residual']-led['budget_residual']))<1e-7 and np.all(led['other_source_cumulative']==0)and np.all(led['other_sink_cumulative']==0))
 check('every_step_positivity',np.all(native['minimum_volume_m3']>=0)and np.all(led['minimum_native_volume_since_initialization']>=0)and np.all(led['negative_volume_sum']==0)and np.all(led['negative_native_count']==0))
 check('cell_minimum_clock_and_final_storage',np.array_equal(native['minimum_time_s'],t[native['minimum_step']])and float(np.cumsum(native['final_volume_m3'])[-1])==float(led['signed_storage'][-1]))
 check('limiter_clock',len(lim)==ns and np.array_equal(lim['actual_end_time_s'],t[1:])and np.array_equal(lim['dt_s'],dt[1:])and np.all((lim['minimum_factor']>=0)&(lim['minimum_factor']<=1))and np.all(lim['held_requested_transport_m3']>=0))
 check('meteo_clock',np.all(led['meteo_bracket_observed'][1:]==1)and np.all(led['meteo_update_interval'][1:]==1)and np.all(led['meteo_bracket_end'][1:]-led['meteo_bracket_start'][1:]==1)and np.all(t[1:]>=led['meteo_bracket_start'][1:]-1e-8)and np.all(t[1:]<=led['meteo_bracket_end'][1:]+1e-8))
 forcing=load_forcing(run['case_id']);geo=load_geometry();physical_t=forcing['Event Time'];physical_r=forcing['Physical Rainfall Rate'];allocation_area=float(38976800.)
 with Dataset(p/'terrain_surface_precip.nc')as d:
  d.set_auto_mask(False);raw=np.asarray(d['time'][:]);intended=np.rint(raw*60);fields=np.asarray(d['Precipitation'][:]);rx=np.asarray(d['x'][:]);ry=np.asarray(d['y'][:]);units=d['time'].units
 trace=np.loadtxt(p/'udt_rain_decode.csv',delimiter=',',skiprows=2,ndmin=2);check('rain_seconds',np.array_equal(trace[:,0],raw)and np.array_equal(trace[:,1],intended)and np.max(abs(raw*60-intended))<5e-7 and np.all(np.diff(intended)>0)and units.startswith('minutes since 2020-01-01'))
 support=readnp(ROOT/'data/processed/three_ward_common_depth_labels/common_support_10m.npz');valid=support['Eligible Ground Mask'];area=support['Ground Area'];ward=support['Ward Code'];geometry=readnp(PREP/'paired_support_and_native_mapping.npz');inside=np.flipud(geometry['Terrain Active Mask']==1)
 physical_at=np.interp(intended,physical_t,physical_r);weight=geometry['Terrain Rainfall Weight'];pattern_error=0.
 for i,r in enumerate(physical_at):pattern_error=max(pattern_error,float(np.max(abs(np.flipud(fields[i,1:-1,1:-1])[valid]-(r*weight)[valid]))))
 pattern_tol=float(4*np.finfo('f4').eps*max(1.,np.max(physical_r)*np.max(weight)));check('roof_assignment',np.all(np.isfinite(fields))and np.all(fields>=0)and pattern_error<=pattern_tol,{'max_rate_error_mm_h':pattern_error,'float32_bound':pattern_tol})
 rates=np.sum(fields[:,1:-1,1:-1][:,inside],axis=1,dtype='f8')*100/3.6e6;encoded=float(np.trapezoid(rates,intended));nominal=float(np.trapezoid(physical_r,physical_t)*allocation_area/3.6e6);expected=integrate_rate_at(intended,rates,np.clip(t,intended[0],intended[-1]));difference=led['rain_applied_cumulative']-expected;tol=max(1.,encoded)*1e-6
 check('original_rain_volume',abs(encoded-nominal)<=max(1.,nominal)*1e-6 and abs(difference[-1])<=tol and max(abs(difference))<=2*max(dt)*max(rates)+tol,{'encoded_m3':encoded,'nominal_m3':nominal,'applied_m3':float(led['rain_applied_cumulative'][-1]),'final_difference_m3':float(difference[-1]),'max_prefix_difference_m3':float(max(abs(difference))),'original_final_tolerance_m3':tol})
 integral=structured(p/'udt_rain_integral.csv');exact_sum=np.r_[0,np.cumsum(integral['integrated_native_rate_volume_m3'])];check('exact_step_interface',len(integral)==ns and np.array_equal(integral['actual_end_time_s'],t[1:])and np.array_equal(integral['dt_s'],dt[1:])and max(abs(integral['sum_weight_seconds']-dt[1:]))<1e-9 and max(abs(exact_sum-expected))<=32*np.finfo('f4').eps*max(1.,encoded))
 check('no_early_rain',np.all(led['rain_applied_cumulative'][t<=physical_t[np.flatnonzero(physical_r>0)[0]-1]if np.any(physical_r>0)else np.ones(t.shape,bool)]==0))
 bzs=np.loadtxt(p/'sfincs.bzs');stages=geo['Boundary Low Stage'][None,:]+np.array([np.interp(bzs[:,0],physical_t,forcing['River Stage Increment By Group'][:,j])for j in range(7)]).T[:,geo['Boundary Group Index']];check('stage_curve_and_point_order',bzs.shape==(len(bzs),len(geo['Boundary Low Stage'])+1)and np.max(abs(bzs[:,1:]-stages))<=1e-8)
 ok,closed,groups,names,mid=receiver_geometry(p);check('receiver_geometry',ok)
 hb=hist['step_index'];complete_hist=np.array_equal(hb,np.arange(1,ns+1));check('all_step_boundary_history',complete_hist)
 gross=np.zeros((len(hist),14));maxclosed=0.;finite=True;codes=[17,51,71,79,106,110,111]
 with Dataset(p/'sfincs_his.nc')as d:
  check('history_clock_and_names',len(d['time'])==len(hist)and np.array_equal(arr(d['time'][:]),hist['nominal_output_time'].astype('f4').astype('f8'))and[str(x).strip()for x in chartostring(d['crosssection_name'][:])]==names and np.all(arr(d['status'][:])==0))
  check('closed_faces_match_runtime_constraints',not closed.any() and not (p/'building_closures.thd').exists())
  for i in range(0,len(hist),256):
   q=arr(d['crosssection_discharge'][i:i+256]).astype('f4');finite &=bool(np.all(np.isfinite(q)));maxclosed=max(maxclosed,float(np.max(abs(q[:,closed]),initial=0)));v=(q*dt[hb[i:i+len(q)],None].astype('f4')).astype('f8')
   for j,code in enumerate(codes):part=v[:,groups==code];gross[i:i+len(q),j*2]=np.maximum(part,0).sum(1);gross[i:i+len(q),j*2+1]=np.maximum(-part,0).sum(1)
 check('all_closed_boundary_flux_zero',complete_hist and maxclosed==0 and finite,{'closed_faces':int(closed.sum()),'max_abs_q_m3_s':maxclosed})
 delta=max(abs(float(gross[:,::2].sum())-float(led['boundary_out_cumulative'][-1])),abs(float(gross[:,1::2].sum())-float(led['boundary_in_cumulative'][-1])));check('boundary_budget_independent_reduction',complete_hist and delta<1e-5,{'max_difference_m3':delta})
 np.savez_compressed(out/phase/(case+'_group_exchange.npz'),**{'Actual Time':np.r_[0,hist['actual_state_time']],'Cumulative Directional Volumes':np.vstack([np.zeros(14),np.cumsum(gross,axis=0)]),'Group Codes':np.asarray(codes)})
 with Dataset(p/'sfincs_map.nc')as d:
  native_mask=arr(d['msk'][:]);check('map_geometry',native_mask.shape==(NR,NC)and np.array_equal(native_mask==1,inside)and max(abs(rx[1:-1]+5-arr(d['x'][:])[0]))<1e-7 and max(abs(ry[1:-1]+5-arr(d['y'][:])[:,0]))<1e-7)
  check('map_and_peak_clock',len(maps)==len(d['time'])==49 and np.array_equal(arr(d['time'][:]),np.arange(0,14401,300))and len(peaks)==len(d['timemax'])==1 and peaks['actual_peak_start'][0]==0 and peaks['actual_peak_end'][0]>=14400)
  peak=arr(d['zsmax'][0]);zmin=arr(d['zb'][:]);finite_fields=True;volume_ok=True;censor=True;max_write_error=0.;max_write_bound=0.
  for i,row in enumerate(maps):
   vol=arr(d['subgrid_volume'][i]);vf=vol[inside].astype('f4');delta=abs(float(vf.astype('f8').sum())-led['signed_storage'][row['step_index']]);bound=float(np.sum(np.maximum(abs(np.nextafter(vf,np.float32(np.inf))-vf),abs(vf-np.nextafter(vf,np.float32(-np.inf)))).astype('f8')*.5))+1e-7;volume_ok &=delta<=bound;max_write_error=max(max_write_error,delta);max_write_bound=max(max_write_bound,bound)
   finite_fields &=all(np.all(np.isfinite(arr(d[k][i])[inside]))for k in ['subgrid_volume','u','v']);zs=arr(d['zs'][i]);h=arr(d['h'][i]);both=inside&np.isfinite(zs)&np.isfinite(h);censor &=bool(np.all(abs(h[both]-(zs[both]-zmin[both]))<1e-5)and np.all(~np.isfinite(zs[inside])|np.isfinite(peak[inside])|(zs[inside]<=zmin[inside]+.01+1e-5))and np.all(zs[inside&np.isfinite(zs)&np.isfinite(peak)]<=peak[inside&np.isfinite(zs)&np.isfinite(peak)]+1e-5))
  check('finite_states',finite_fields and np.all(np.isfinite(zmin[inside])))
  check('same_step_map_storage',volume_ok,{'max_difference_m3':max_write_error,'max_float32_write_bound_m3':max_write_bound})
  check('censoring',censor)
 peak=np.flipud(peak);zmin=np.flipud(zmin);lo=np.full((NR,NC),np.nan);up=lo.copy();barrier=np.full((NR,NC),np.inf)
 base=ROOT/'data/processed/three_ward_sfincs_scenarios';con=ROOT/'data/processed/three_ward_urban_constraints'
 with rasterio.open(base/'terrain_sampling_0_50m.tif')as dem,rasterio.open(con/'constraints_0_50m.tif')as cl,rasterio.open(con/'hydraulic_obstacle_0_50m.tif')as hyd:
  for r in range(0,NR,32):
   nr=min(32,NR-r);w=Window(0,r*20,NC*20,nr*20);a,b,ga,ra,ba=fine_peaks(dem.read(1,window=w),cl.read(1,window=w),hyd.read(1,window=w),valid[r:r+nr],peak[None,r:r+nr],zmin[None,r:r+nr]);assert np.array_equal(ga,area[r:r+nr]);lo[r:r+nr]=a[0];up[r:r+nr]=b[0];barrier[r:r+nr]=ba
 native_index=np.fromfile(p/'sfincs.ind',dtype='<i4')[1:]-1;ever=np.zeros((NR,NC),bool);ids=native['native_cell_index']-1;bad=native['minimum_volume_m3']<-.001;ever[NR-1-native_index[ids[bad]]%NR,native_index[ids[bad]]//NR]=True
 contact=np.zeros(valid.shape,bool) # T contains no assumed building-height barrier
 masks=classify_labels(lo,up,valid,ever,contact,interval_source_known=bool(censor));labels={'Peak Mean Depth Lower':lo,'Peak Mean Depth Upper':up,'Online Ever Negative':ever,'Assumed Barrier Contact':contact,**masks};np.savez_compressed(out/phase/(case+'_candidate_reference.npz'),**labels)
 np.savez_compressed(out/phase/(case+'_output_clock.npz'),**{'Actual Map Time':maps['actual_state_time'],'Map Step Index':maps['step_index'],'Actual Peak Start':peaks['actual_peak_start'],'Actual Peak End':peaks['actual_peak_end']})
 check('ground_label_bounds',np.all(np.isfinite(lo[valid]))and np.all(np.isfinite(up[valid]))and np.all(up[valid]>=lo[valid]))
 passed={c['check']:c['pass']for c in checks}
 gatekeys={'completion':['completion'],'identity':['identity','stage_curve_and_point_order'],'geometry':['receiver_geometry','map_geometry','closed_faces_match_runtime_constraints','all_closed_boundary_flux_zero'],'fields':['finite_states','finite_ledger','every_step_positivity','cell_minimum_clock_and_final_storage'],'time':['map_and_peak_clock'],'clock_alignment':['actual_clock','history_clock_and_names'],'boundary':['all_step_boundary_history','boundary_budget_independent_reduction'],'rain':['rain_seconds','roof_assignment','original_rain_volume','meteo_clock','no_early_rain'],'budget':['original_budget','native_budget_reconstruction','same_step_map_storage'],'censoring':['censoring','ground_label_bounds']};gates={g:bool(all(passed[k]for k in keys))for g,keys in gatekeys.items()};admission=event_admission(gates,masks)
 coverage=[]
 for name,sel in [('Whole domain',valid),('Three wards',valid&(ward>0)),('Chiyoda',valid&(ward==1)),('Chuo',valid&(ward==2)),('Minato',valid&(ward==3))]:coverage.append({'case':case,'split':run['split'],'scope':name,**{k.replace(' Mask','').replace(' ','_')+'_km2':float(area[sel&masks[k]].sum()/1e6)for k in ['Exact Label Mask','Interval Label Mask','Wide Interval Mask','Quarantine Mask','Admitted Mask']},'certain_peak_wet_gt0_1m_km2':float(area[sel&(lo>.1)].sum()/1e6),'possible_peak_wet_gt0_1m_km2':float(area[sel&(up>.1)].sum()/1e6),'interval_max_m':float(np.max((up-lo)[sel]))})
 csvwrite(report/'label_support.csv',coverage)
 result={'case_id':case,'phase':phase,'split':run['split'],'all_checks_passed':all(c['pass']for c in checks),'checks':checks,'required_gates':gates,'admission':admission,'summary':{'actual_steps':ns,'actual_final_time_s':float(t[-1]),'runtime_s':ex['runtime_seconds'],'minimum_volume_m3':float(native['minimum_volume_m3'].min()),'negative_cells':int(bad.sum()),'rain_applied_m3':float(led['rain_applied_cumulative'][-1]),'encoded_rain_m3':encoded,'nominal_rain_m3':nominal,'rain_difference_m3':float(difference[-1]),'max_rain_prefix_difference_m3':float(max(abs(difference))),'max_budget_residual_m3':float(max(abs(bal['residual']))),'closed_boundary_faces':int(closed.sum()),'gross_out_m3':float(led['boundary_out_cumulative'][-1]),'gross_in_m3':float(led['boundary_in_cumulative'][-1]),'map_records':len(maps)},'training':False,'formal_reference_release':False,'real_world_accuracy_verified':False};dump(report/'audit.json',result);print(json.dumps({'case':case,'all_checks_passed':result['all_checks_passed'],'admission':admission},indent=2));assert result['all_checks_passed']
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['execution']);p.add_argument('case');a=p.parse_args();main(a.phase,a.case)
