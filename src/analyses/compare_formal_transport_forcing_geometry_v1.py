"""Interpret completed R0/R1/R125 x L/H x T/U results on fixed quality support.
No solver execution, network closures, population joins, training or sealed tests.
Rain-stage contrasts of peak envelopes describe response functionals, not water
source tracing, observed causes, synchronized flood extents or additive attribution.
"""
import os
os.environ.setdefault('NUMBA_NUM_THREADS','2')
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-mpl')
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
from pathlib import Path
import sys,json,csv
import numpy as np
import rasterio
from rasterio.windows import Window
from netCDF4 import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src/analyses'),str(ROOT/'src/preprocessing')]
from prepare_sfincs_cpu_build import sha,dump
from audit_three_ward_negative_volumes import arr,structured
from compare_urban_geometry_pair_controls_v1 import road_depth_bounds,table,csvwrite
from prepare_historical_surface_access_v1 import depth_at
P=ROOT/'data/processed/three_ward_formal_transport_experiments_v1'
H=ROOT/'data/processed/three_ward_historical_radar_experiment_v2'
R=ROOT/'data/exp/estimation/three_ward_formal_transport_experiments_v1'
O=R/'B2_B3_forcing_geometry_comparison'
Q=P/'factor_comparison_v1'
D=ROOT/'data/exp/research-design/three_ward_formal_transport_matrix_v1'
SCOPES={'Three wards':-1,'Chiyoda':1,'Chuo':2,'Minato':3}
RAINS=['R0','R1','R125'];STAGES=['L','H'];ARMS=['T','U']

def cid(r,s,a):return f'{r}__{s}__{a}'
def readcsv(p):
 with p.open()as f:return list(csv.DictReader(f))
def area_sum(weights,mask):return float(weights[mask].sum()/1e6)
def ratio(a,b):return float(a/b)if b else None

def main():
 O.mkdir(parents=True,exist_ok=True);Q.mkdir(parents=True,exist_ok=True)
 manifest=json.loads((P/'run_manifest.json').read_text());runs={x['case_id']:x for x in manifest['runs']}
 protocol=json.loads((D/'protocol.json').read_text());times=np.asarray(protocol['grid']['synchronized_times_s'])
 s=np.load(ROOT/'data/processed/three_ward_common_depth_labels/common_support_10m.npz')
 valid=s['Eligible Ground Mask'];ga=s['Ground Area'];ra=s['Road Ground Area'];ward=s['Ward Code'];common=valid.copy()
 checks=[];bindings={};mapstats={};data={};budgets=[];groups=[];budget_times=[];coverage=[]
 def check(name,v):
  checks.append({'check':name,'pass':bool(v)})
  if not v:raise AssertionError(name)
 def bind(p):bindings[str(p.relative_to(ROOT))]=sha(p)
 for path in [D/'protocol.json',P/'run_manifest.json',P/'execution_code_binding.json',R/'campaign_state.json',ROOT/'data/processed/three_ward_common_depth_labels/common_support_10m.npz',ROOT/'src/analyses/compare_urban_geometry_pair_controls_v1.py',ROOT/'src/preprocessing/prepare_historical_surface_access_v1.py']:
  bind(path)
 for rel,digest in json.loads((P/'execution_code_binding.json').read_text()).items():check('solver_binding_'+rel,sha(ROOT/rel)==digest)
 for case,run in runs.items():
  old=run.get('original_case_id',case);base=H if run['reused']else P
  auditpath=ROOT/run['original_audit']if run['reused']else R/'execution'/case/'audit.json'
  labelpath=base/'execution'/(old+'_candidate_reference.npz');clockpath=base/'execution'/(old+'_output_clock.npz');gpath=base/'execution'/(old+'_group_exchange.npz');folder=ROOT/run['folder']
  audit=json.loads(auditpath.read_text());lab=np.load(labelpath)
  check(case+'_all_original_gates',audit['all_checks_passed']and all(audit['required_gates'].values())and audit['summary']['negative_cells']==0)
  good=valid&lab['Admitted Mask']&~lab['Quarantine Mask']&np.isfinite(lab['Peak Mean Depth Lower'])&np.isfinite(lab['Peak Mean Depth Upper']);common &= good
  clock=np.load(clockpath)['Actual Map Time'];ii=np.asarray([np.argmin(abs(clock-t))for t in times],int)
  check(case+'_original_actual_clock_tolerance',np.max(abs(clock[ii]-times))<=.001)
  with Dataset(folder/'sfincs_map.nc')as ds:peak=np.flipud(arr(ds['zsmax'][0]));bed=np.flipud(arr(ds['zb'][:]))
  stat=(folder/'sfincs_map.nc').stat();mapstats[str((folder/'sfincs_map.nc').relative_to(ROOT))]=dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns)
  data[case]=dict(run=run,folder=folder,audit=audit,good=good,lower=lab['Peak Mean Depth Lower'],upper=lab['Peak Mean Depth Upper'],peak=peak,bed=bed,indices=ii,actual_times=clock[ii])
  led=structured(folder/'udt_step_budget.csv');last=led[-1];initial=led[0];summary=audit['summary']
  budgets.append(dict(case_id=case,rainfall=run['rainfall'],stage=run['stage'],arm=run['arm'],rain_applied_m3=float(last['rain_applied_cumulative']),initial_storage_m3=float(initial['signed_storage']),final_storage_m3=float(last['signed_storage']),storage_change_m3=float(last['signed_storage']-initial['signed_storage']),gross_out_m3=float(last['boundary_out_cumulative']),gross_in_m3=float(last['boundary_in_cumulative']),max_abs_budget_residual_m3=summary['max_budget_residual_m3'],negative_cells=summary['negative_cells'],all_original_checks_passed=True))
  for t in times:
   j=int(np.argmin(abs(led['actual_state_time']-t)));x=led[j]
   budget_times.append(dict(case_id=case,scheduled_time_s=int(t),actual_time_s=float(x['actual_state_time']),storage_change_m3=float(x['signed_storage']-initial['signed_storage']),rain_applied_m3=float(x['rain_applied_cumulative']),gross_in_m3=float(x['boundary_in_cumulative']),gross_out_m3=float(x['boundary_out_cumulative']),budget_residual_m3=float(x['budget_residual'])))
  exchange=np.load(gpath);vol=exchange['Cumulative Directional Volumes'][-1]
  for j,code in enumerate(exchange['Group Codes']):groups.append(dict(case_id=case,water_group=int(code),gross_out_m3=float(vol[2*j]),gross_in_m3=float(vol[2*j+1])))
  for path in [auditpath,labelpath,clockpath,gpath,folder/'sfincs.bzs',folder/'udt_run_identity.txt']:bind(path)
  print('LOADED',case,flush=True)
 for rain in RAINS:
  for stage in STAGES:
   ids=[cid(rain,stage,a)for a in ARMS];pairgood=data[ids[0]]['good']&data[ids[1]]['good']
   check(rain+stage+'_same_stage_bytes',(data[ids[0]]['folder']/'sfincs.bzs').read_bytes()==(data[ids[1]]['folder']/'sfincs.bzs').read_bytes())
   b=[next(x for x in budgets if x['case_id']==case)for case in ids]
   check(rain+stage+'_conserved_rain_across_geometry',abs(b[0]['rain_applied_m3']-b[1]['rain_applied_m3'])<=max(1,b[0]['rain_applied_m3'])*1e-6)
   if rain=='R1':
    pair=np.load(R/'B1_geometry_comparison'/f'{stage}_paired_peak_bounds.npz')
    for a,case in enumerate(ids):data[case]['rlo']=pair['road_lower'][a];data[case]['rup']=pair['road_upper'][a]
   else:
    rlo=np.full((2,1012,770),np.nan);rup=rlo.copy();area=np.zeros((1012,770))
    with rasterio.open(ROOT/'data/processed/three_ward_sfincs_scenarios/terrain_sampling_0_50m.tif')as dem,rasterio.open(ROOT/'data/processed/three_ward_urban_constraints/constraints_0_50m.tif')as cls:
     for row in range(0,1012,32):
      n=min(32,1012-row);win=Window(0,row*20,15400,n*20)
      a,b,c=road_depth_bounds(dem.read(1,window=win),cls.read(1,window=win),valid[row:row+n],np.stack([data[x]['peak']for x in ids])[:,row:row+n],np.stack([data[x]['bed']for x in ids])[:,row:row+n])
      rlo[:,row:row+n]=a;rup[:,row:row+n]=b;area[row:row+n]=c
    check(rain+stage+'_unchanged_fine_road_support',np.array_equal(area,ra))
    for a,case in enumerate(ids):data[case]['rlo']=rlo[a];data[case]['rup']=rup[a]
   for scope,code in SCOPES.items():
    region=valid&(ward>0)if code==-1 else valid&(ward==code)
    coverage.append(dict(rainfall=rain,stage=stage,scope=scope,eligible_ground_km2=area_sum(ga,region),pair_admitted_ground_km2=area_sum(ga,region&pairgood),all_twelve_admitted_ground_km2=area_sum(ga,region&common),excluded_from_all_twelve_km2=area_sum(ga,region&~common)))
   print('ROADS',rain,stage,flush=True)
 np.savez_compressed(Q/'common_support.npz',common_admitted=common,eligible=valid,ground_area=ga,road_area=ra,ward_code=ward)
 # Fixed common support is an intersection of unchanged masks, not a new acceptance rule.
 extents=[];depths=[];geometry=[];stability=[];contrasts=[];effect_depths=[];stationrows=[];station_summary=[]
 def domain(item,name):return (item['lower'],item['upper'],ga)if name=='ground'else(item['rlo'],item['rup'],ra)
 for case,item in data.items():
  run=item['run'];np.savez_compressed(Q/(case+'_peak_bounds.npz'),ground_lower=item['lower'],ground_upper=item['upper'],road_lower=item['rlo'],road_upper=item['rup'])
  for scope,code in SCOPES.items():
   region=common&(ward>0)if code==-1 else common&(ward==code)
   for support in ['ground','road']:
    lo,up,weights=domain(item,support);use=region&(weights>0);mass=weights[use].sum()
    check(case+scope+support+'_ordered_common_bounds',np.isfinite(lo[use]).all()and np.isfinite(up[use]).all()and np.all(up[use]>=lo[use]))
    depths.append(dict(case_id=case,scope=scope,support=support,area_km2=mass/1e6,mean_lower_m=float(np.sum(weights[use]*lo[use])/mass),mean_upper_m=float(np.sum(weights[use]*up[use])/mass)))
    for threshold in [.1,.3,.5]:extents.append(dict(case_id=case,rainfall=run['rainfall'],stage=run['stage'],arm=run['arm'],scope=scope,support=support,threshold_m=threshold,certain_km2=area_sum(weights,use&(lo>threshold)),possible_km2=area_sum(weights,use&(up>threshold))))
 def contrast_terms(rain,arm):
  return {'stage_without_rain':[(1,cid('R0','H',arm)),(-1,cid('R0','L',arm))],
   'rain_at_low_stage':[(1,cid(rain,'L',arm)),(-1,cid('R0','L',arm))],
   'rain_at_raised_stage':[(1,cid(rain,'H',arm)),(-1,cid('R0','H',arm))],
   'stage_with_rain':[(1,cid(rain,'H',arm)),(-1,cid(rain,'L',arm))],
   'rain_stage_interaction':[(1,cid(rain,'H',arm)),(-1,cid(rain,'L',arm)),(-1,cid('R0','H',arm)),(1,cid('R0','L',arm))]}
 for rain in ['R1','R125']:
  for arm in ARMS:
   for effect,terms in contrast_terms(rain,arm).items():
    for scope in SCOPES:
     for support in ['ground','road']:
      for threshold in [.1,.3,.5]:
       low=high=0.
       for sign,case in terms:
        row=next(x for x in extents if x['case_id']==case and x['scope']==scope and x['support']==support and x['threshold_m']==threshold)
        low+=sign*row['certain_km2'if sign>0 else'possible_km2'];high+=sign*row['possible_km2'if sign>0 else'certain_km2']
       contrasts.append(dict(rainfall=rain,arm=arm,effect=effect,scope=scope,support=support,threshold_m=threshold,difference_lower_km2=low,difference_upper_km2=high,conditional_peak_functional=True,not_source_fraction=True))
      # Differences of cell peak depths are separate functionals from peak differences of synchronized fields.
      lo=np.zeros(common.shape);up=lo.copy()
      for sign,case in terms:
       a,b,w=domain(data[case],support);lo+=sign*(a if sign>0 else b);up+=sign*(b if sign>0 else a)
      code=SCOPES[scope];region=common&(ward>0)if code==-1 else common&(ward==code);use=region&(w>0);mass=w[use].sum()
      effect_depths.append(dict(rainfall=rain,arm=arm,effect=effect,scope=scope,support=support,mean_difference_lower_m=float(np.sum(w[use]*lo[use])/mass),mean_difference_upper_m=float(np.sum(w[use]*up[use])/mass)))
 for rain in RAINS:
  for stage in STAGES:
   t=data[cid(rain,stage,'T')];u=data[cid(rain,stage,'U')]
   for scope,code in SCOPES.items():
    region=common&(ward>0)if code==-1 else common&(ward==code)
    for support in ['ground','road']:
     tl,tu,w=domain(t,support);ul,uu,_=domain(u,support);use=region&(w>0);dl=ul-tu;du=uu-tl;mass=w[use].sum()
     for threshold in [.1,.3,.5]:
      added=(ul>threshold)&~(tu>threshold);removed=(tl>threshold)&~(uu>threshold)
      geometry.append(dict(rainfall=rain,stage=stage,scope=scope,support=support,threshold_m=threshold,U_minus_T_extent_lower_km2=area_sum(w,use&(ul>threshold))-area_sum(w,use&(tu>threshold)),U_minus_T_extent_upper_km2=area_sum(w,use&(uu>threshold))-area_sum(w,use&(tl>threshold)),certain_added_km2=area_sum(w,use&added),certain_removed_km2=area_sum(w,use&removed),mean_depth_difference_lower_m=float(np.sum(w[use]*dl[use])/mass),mean_depth_difference_upper_m=float(np.sum(w[use]*du[use])/mass),depth_increase_ge0p1m_km2=area_sum(w,use&(dl>=.1)),depth_decrease_ge0p1m_km2=area_sum(w,use&(du<=-.1))))
 # Stability measures remain descriptive; no data-selected pass threshold.
 for stage in STAGES:
  for scope,code in SCOPES.items():
   region=common&(ward>0)if code==-1 else common&(ward==code)
   for support in ['ground','road']:
    t1,tu1,w=domain(data[cid('R1',stage,'T')],support);u1,uu1,_=domain(data[cid('R1',stage,'U')],support)
    t2,tu2,_=domain(data[cid('R125',stage,'T')],support);u2,uu2,_=domain(data[cid('R125',stage,'U')],support);use=region&(w>0)
    for threshold in [.1,.3,.5]:
     add1=(u1>threshold)&~(tu1>threshold);add2=(u2>threshold)&~(tu2>threshold);rem1=(t1>threshold)&~(uu1>threshold);rem2=(t2>threshold)&~(uu2>threshold)
     union=area_sum(w,use&(add1|add2));intersection=area_sum(w,use&add1&add2)
     a1=area_sum(w,use&add1);a2=area_sum(w,use&add2);r1=area_sum(w,use&rem1);r2=area_sum(w,use&rem2)
     inc1=(u1-tu1)>=.1;inc2=(u2-tu2)>=.1;dec1=(uu1-t1)<=-.1;dec2=(uu2-t2)<=-.1
     stability.append(dict(stage=stage,scope=scope,support=support,threshold_m=threshold,original_added_km2=a1,stronger_added_km2=a2,added_retained_km2=intersection,original_added_retained_fraction=ratio(intersection,a1),added_pattern_jaccard=ratio(intersection,union),original_removed_km2=r1,stronger_removed_km2=r2,removed_retained_fraction=ratio(area_sum(w,use&rem1&rem2),r1),certain_added_to_removed_reversal_km2=area_sum(w,use&add1&rem2),certain_removed_to_added_reversal_km2=area_sum(w,use&rem1&add2),large_increase_retained_fraction=ratio(area_sum(w,use&inc1&inc2),area_sum(w,use&inc1)),large_signed_reversal_km2=area_sum(w,use&((inc1&dec2)|(dec1&inc2)))))
 print('FACTOR AND STABILITY METRICS',flush=True)
 # Station exposure at the predeclared12 synchronous actual times plus a separate peak.
 cachepath=ROOT/'data/processed/three_ward_surface_alternatives_v1/fine_elevation_support.npz';cache=np.load(cachepath)
 gcells=cache['ground_cells'];gz=cache['ground_sorted_z'];gcum=cache['ground_cumulative_z'];gn=cache['ground_count'];garea=gn*.25
 epath=ROOT/'data/processed/three_ward_surface_coverage_v2/surface_exposure.npz';e=np.load(epath);off=e['entrance_offsets'];inds=e['entrance_cell_index'];known=common.ravel()[gcells];frames=[str(t)for t in times]+['peak_envelope']
 for case,item in data.items():
  with Dataset(item['folder']/'sfincs_map.nc')as ds:eta=np.stack([np.flipud(arr(ds['zs'][i])).ravel()[gcells]for i in item['indices']]+[item['peak'].ravel()[gcells]])
  low,up=depth_at(gz,gcum,gn,eta,item['bed'].ravel()[gcells]);check(case+'_station_peak_reconstructed',np.allclose(low[-1],item['lower'].ravel()[gcells],atol=1e-6,equal_nan=True)and np.allclose(up[-1],item['upper'].ravel()[gcells],atol=1e-6,equal_nan=True))
  for ti,frame in enumerate(frames):
   rows=[]
   for j,osm in enumerate(e['entrance_ids']):
    c=inds[off[j]:off[j+1]];total=float(garea[c].sum());c=c[known[c]];admitted=float(garea[c].sum());quality=total>0 and admitted>=100 and admitted/total>=.8
    rows.append(dict(case_id=case,frame_id=frame,simultaneous=frame!='peak_envelope',osm_node_id=str(osm),quality=quality,trigger_lower=bool(quality and garea[c][low[ti,c]>=.3].sum()>=100),trigger_upper=bool(quality and garea[c][up[ti,c]>=.3].sum()>=100)))
   stationrows.extend(rows);station_summary.append(dict(case_id=case,frame_id=frame,simultaneous=frame!='peak_envelope',paired_quality_points=sum(x['quality']for x in rows),trigger_lower_points=sum(x['trigger_lower']for x in rows),trigger_upper_points=sum(x['trigger_upper']for x in rows)))
 print('STATION EXPOSURE COMPLETE',flush=True)
 # Every additional computation remains downstream of the unchanged numerical policy.
 for path in [cachepath,epath,R/'B1_geometry_comparison/L_paired_peak_bounds.npz',R/'B1_geometry_comparison/H_paired_peak_bounds.npz']:bind(path)
 for name,rows in [('common_quality_coverage',coverage),('case_peak_extent',extents),('case_peak_depth',depths),('conditional_forcing_contrasts',contrasts),('conditional_depth_contrasts',effect_depths),('geometry_effects',geometry),('geometry_stability',stability),('full_domain_water_budget',budgets),('full_domain_budget_at_times',budget_times),('water_group_exchange',groups),('station_exposure_summary',station_summary),('station_exposure_detail',stationrows)]:csvwrite(O/(name+'.csv'),rows)
 def primary(rows):return[x for x in rows if x.get('scope')=='Three wards'and ((x.get('support')=='ground'and x.get('threshold_m')==.1)or(x.get('support')=='road'and x.get('threshold_m')==.3))]
 # Save a compact, directly auditable summary. Conclusions are conditional model comparisons.
 summary=dict(common_three_ward_ground_km2=area_sum(ga,common&(ward>0)),common_three_ward_road_km2=area_sum(ra,common&(ward>0)),all_twelve_cases_pass_original_gates=True,case_peak_extent=primary(extents),forcing_contrasts=primary(contrasts),geometry_effects=primary(geometry),geometry_stability=primary(stability),peak_station_exposure=[x for x in station_summary if x['frame_id']=='peak_envelope'],max_budget_residual_m3=max(x['max_abs_budget_residual_m3']for x in budgets),negative_cells=sum(x['negative_cells']for x in budgets),same_historical_parent='HIST-RADAR-TOKYO-20240821',new_hydraulic_simulations=0,people_counted=False,training=False,sealed_tests_opened=False,transport_accessibility_computed=False,real_world_accuracy_verified=False)
 dump(O/'summary.json',summary)
 dump(O/'comparison_contract.json',dict(human_decision='MILI-D-20261017-006',quality='Original numerical gates and masks unchanged. Fixed all12 intersection for cross-factor comparison; per-pair coverage reported separately. No interval midpoint.',forcing_effects='Conditional differences of area/mean-depth peak functionals and separately reported synchronized station/budget diagnostics; no additive source attribution.',interaction='F(rain,H)-F(rain,L)-F(dry,H)+F(dry,L); may be negative because overlap, thresholds and hydraulics are nonlinear. Not the flood volume from a source.',geometry_stability='Persistence and Jaccard of added/removed threshold footprints, depth-change sign reversals and whole-ward differences; no fitted significance or success threshold.',transport='Road depth is classified fine road ground integration; strict roads, path supplementation and unknown bridge/tunnel branches remain separate and unchanged. This task does not rerun traffic graphs or infer isolation from peak envelopes.',limitations='Assumed stages; representative radar intensity and x1.25 sensitivity, not independently observed second storm; terrain/urban bundled2D representations, no observed accuracy, sewers/pumps/infiltration, bathymetry or affected-person estimates.'))
 # Full-domain budget contrasts cannot be recast as ward-specific source volumes.
 check('zero_rain_low_controls_no_native_change',all(x['rain_applied_m3']==x['storage_change_m3']==x['gross_out_m3']==x['gross_in_m3']==0 for x in budgets if x['rainfall']=='R0'and x['stage']=='L'))
 check('source_bindings_unchanged',all(sha(ROOT/k)==v for k,v in bindings.items()))
 check('native_map_metadata_unchanged',all((ROOT/k).stat().st_size==v['size']and(ROOT/k).stat().st_mtime_ns==v['mtime_ns']for k,v in mapstats.items()))
 dump(O/'source_bindings.json',bindings);dump(O/'map_metadata.json',mapstats);dump(O/'validation.json',dict(all_passed=True,checks=checks))
 plots(data,common,ward,ga,budgets,budget_times)
 render(summary,coverage,station_summary,budgets,len(checks))
 interpret_saved_products()
 # Keep B1 and the frozen controller protocol intact; link the new result from the existing matrix page.
 path=D/'review.html';text=path.read_text();marker='<!-- B2_B3_REVIEW_LINK -->'
 if marker not in text:text=text.replace('<h2>首批与完整水动力矩阵</h2>',marker+'<p><a href="'+os.path.relpath(O/'review.html',D)+'">无雨控制与增强降雨：因素分离及城市几何稳定性</a></p><h2>首批与完整水动力矩阵</h2>');path.write_text(text)
 print('COMPLETE',json.dumps({k:summary[k]for k in ['common_three_ward_ground_km2','max_budget_residual_m3','negative_cells']},ensure_ascii=False),flush=True)

def plots(data,common,ward,ga,budgets,budget_times):
 region=common&(ward>0);extent=[-11.36,-3.66,-42.44,-32.32]
 def axes_map(ax,z,title,vmin,vmax,cmap,label):
  im=ax.imshow(np.where(region,z,np.nan),origin='upper',extent=extent,vmin=vmin,vmax=vmax,cmap=cmap);ax.set_title(title,fontsize=11);ax.set_aspect('equal');ax.set_xlabel('Easting (km)');return im
 fig,axs=plt.subplots(1,2,figsize=(9,7))
 for j,arm in enumerate(ARMS):
  im=axes_map(axs[j],data[cid('R0','H',arm)]['lower'],arm+': no rain, raised stage',0,.6,'Blues','m');fig.colorbar(im,ax=axs[j],shrink=.75,label='Peak open-ground mean depth lower bound (m)')
 axs[0].set_ylabel('Northing (km)');fig.suptitle('Stage-only control | common three-ward support');fig.tight_layout(rect=[0,.02,1,.95]);fig.savefig(O/'no_rain_stage_geometry.png',dpi=155);plt.close(fig)
 for stage in STAGES:
  fig,axs=plt.subplots(2,3,figsize=(12,10),layout='constrained')
  for i,rain in enumerate(['R1','R125']):
   t=data[cid(rain,stage,'T')];u=data[cid(rain,stage,'U')]
   for j,item in enumerate([t,u]):
    im=axes_map(axs[i,j],item['lower'],rain+' | '+ARMS[j],0,.75,'Blues','m')
    if j==1:fig.colorbar(im,ax=list(axs[i,:2]),shrink=.75,label='Peak depth lower bound (m)')
   dl=u['lower']-t['upper'];du=u['upper']-t['lower'];signed=np.where(dl>0,dl,np.where(du<0,du,0.))
   im=axes_map(axs[i,2],signed,rain+' | U minus T',-.3,.3,'RdBu_r','m');fig.colorbar(im,ax=axs[i,2],shrink=.75,label='Separated-bound difference (m)')
  axs[0,0].set_ylabel('Northing (km)');axs[1,0].set_ylabel('Northing (km)');fig.suptitle(stage+' stage assumption | original vs1.25 rainfall | peak envelopes')
  fig.savefig(O/(stage+'_rainfall_geometry_stability.png'),dpi=155);plt.close(fig)
 # Actual-time budget samples, not nominal NetCDF labels. Positive is inward flow.
 fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
 for i,arm in enumerate(ARMS):
  for rain,style in [('R0','--'),('R1','-'),('R125',':')]:
   rows=[x for x in budget_times if x['case_id']==cid(rain,'H',arm)]
   axs[i].plot([x['actual_time_s']/3600+15 for x in rows],[x['gross_in_m3']/1e6 for x in rows],style,label=rain)
  axs[i].set_title(arm+': raised-stage boundary inflow');axs[i].set_xlabel('Clock hour JST (24 = midnight)');axs[i].set_ylabel('Gross cumulative inflow (million m3)');axs[i].legend();axs[i].grid(alpha=.2)
 fig.savefig(O/'raised_stage_gross_inflow.png',dpi=155);plt.close(fig)

def render(s,coverage,stations,budgets,nchecks):
 def selected(x):return x.get('scope')=='Three wards'and ((x.get('support')=='ground'and x.get('threshold_m')==.1)or(x.get('support')=='road'and x.get('threshold_m')==.3))
 label=lambda x:'开放地表 >0.1 m'if x['support']=='ground'else'分类道路地表 >0.3 m'
 body='<h1>无雨控制与增强降雨：水位、降雨和城市几何的影响</h1><p>都心三区整体比较，使用已完成的12个物理算例；不追加模拟、不计算受困人数、不训练、不打开封存测试。</p>'
 body+='<aside><b>先区分驱动，再检查几何差异。</b><p>无雨低水位组没有原生储水或边界交换；无雨水位抬升组仍出现积水。降雨与水位的作用会相互影响，不能把两幅峰值图的面积直接当成可相加的水源贡献。</p></aside>'
 body+='<h2>比较口径与质量覆盖</h2><p>原数值闸门、标签隔离和区间规则保持不变。全部12组的共同准入开放地表为'+f"{s['common_three_ward_ground_km2']:.6f}"+' km²，分类道路地表为'+f"{s['common_three_ward_road_km2']:.6f}"+' km²。共同交集用于跨雨量比较，未以删点放宽准入；每组及每对原覆盖单列。与B1表的小幅差异来自新增增强雨组后的共同交集。</p>'
 body+=table(['雨量','水位','范围','原可用 km²','T/U配对 km²','全部12组共同 km²'],[[x['rainfall'],x['stage'],x['scope'],f"{x['eligible_ground_km2']:.6f}",f"{x['pair_admitted_ground_km2']:.6f}",f"{x['all_twelve_admitted_ground_km2']:.6f}"]for x in coverage if x['scope']=='Three wards'])
 body+='<h2>无雨、原降雨与增强降雨的峰值面积</h2>'+table(['降雨','水位','几何','指标','确定面积 km²','可能面积 km²'],[[x['rainfall'],x['stage'],x['arm'],label(x),f"{x['certain_km2']:.4f}",f"{x['possible_km2']:.4f}"]for x in s['case_peak_extent']])
 body+='<p>T=仅地形，U=建筑阻水、道路输水及屋面保守分配的城市信息组合；R125为同一历史雨场乘1.25，仍属于同一父事件。L/H均是河道水位假设。以上是事件峰值包络，不能当作同一时刻积水。</p><img src="no_rain_stage_geometry.png" alt="No-rain raised-stage peak comparison">'
 effects={'stage_without_rain':'无雨条件下抬升水位','rain_at_low_stage':'低水位条件下加雨','rain_at_raised_stage':'抬升水位条件下加雨','stage_with_rain':'有雨条件下抬升水位','rain_stage_interaction':'雨量×水位交互'}
 body+='<h2>区分条件效应与交互</h2><p>令F为同一共同地表上的峰值面积指标。无雨水位效应为F(0,H)−F(0,L)；有雨水位效应为F(R,H)−F(R,L)；交互为二者之差。低水位加雨效应、无雨抬升效应和交互三项可重构复合情景相对干低基线的指标变化。这里是情景响应差值，不是示踪水源、观测因果比例或同一时刻的积水体积。</p>'
 body+=table(['雨量','几何','指标','条件对照','差值下界 km²','上界 km²'],[[x['rainfall'],x['arm'],label(x),effects[x['effect']],f"{x['difference_lower_km2']:+.4f}",f"{x['difference_upper_km2']:+.4f}"]for x in s['forcing_contrasts']if x['rainfall']=='R1'])
 body+='<p>负交互表示复合响应低于对应指标的简单相加；可能包含超阈值足迹重叠、回水及交换变化，不能直接解释为排水改善或水量消失。</p><h2>全域水量与河道交换</h2>'
 body+=table(['算例','降雨 m³','储水变化 m³','累计入流 m³','累计出流 m³','最大预算残差 m³'],[[x['case_id'],f"{x['rain_applied_m3']:.1f}",f"{x['storage_change_m3']:.1f}",f"{x['gross_in_m3']:.1f}",f"{x['gross_out_m3']:.1f}",f"{x['max_abs_budget_residual_m3']:.6f}"]for x in budgets])
 body+='<p>预算覆盖各组全部原生计算域，包含缓冲区和T组恢复的建筑占地，不能冒充三区共同开放地表水量。累计出入流分开积分，不能先相抵或把全期累计入流当作末态残留河水。</p><img src="raised_stage_gross_inflow.png" alt="Actual-clock cumulative river inflow">'
 body+='<h2>增强雨量下城市几何差异是否稳定</h2>'+table(['雨量','水位','指标','U−T面积下界 km²','上界 km²','U明确新增 km²','U明确减少 km²'],[[x['rainfall'],x['stage'],label(x),f"{x['U_minus_T_extent_lower_km2']:+.4f}",f"{x['U_minus_T_extent_upper_km2']:+.4f}",f"{x['certain_added_km2']:.4f}",f"{x['certain_removed_km2']:.4f}"]for x in s['geometry_effects']if x['rainfall']!='R0'])
 def fmt(v):return'无原足迹'if v is None else f'{100*v:.1f}%'
 body+=table(['水位','指标','原U新增足迹保留率','新增足迹Jaccard','原U减少足迹保留率','原增→新减 km²','原减→新增 km²'],[[x['stage'],label(x),fmt(x['original_added_retained_fraction']),fmt(x['added_pattern_jaccard']),fmt(x['removed_retained_fraction']),f"{x['certain_added_to_removed_reversal_km2']:.5f}",f"{x['certain_removed_to_added_reversal_km2']:.5f}"]for x in s['geometry_stability']])
 body+='<p>稳定性同时看净变化方向、原足迹保留与位置翻转，没有事后设置“合格”百分比。保留率的分母是原雨量足迹，Jaccard分母是两组足迹并集；增强雨量跨越阈值可能使差异消失或新出现。整体方向相同不等于每条道路或站口都稳定。</p>'
 for stage in STAGES:body+='<img src="'+stage+'_rainfall_geometry_stability.png" alt="Original and stronger rainfall geometry comparison">'
 body+='<h2>站口周边暴露：同步时刻与峰值分开</h2>'+table(['算例','共同质量达标点','峰值触发下界','上界'],[[x['case_id'],x['paired_quality_points'],x['trigger_lower_points'],x['trigger_upper_points']]for x in stations if x['frame_id']=='peak_envelope'])
 body+='<p>原675点和20 m格心圆、有效面积≥100 m²及覆盖≥80%的质量规则不变。≥0.3 m足迹≥100 m²仍是既有假设参数，不是实测站口关闭标准。CSV另列12个统一时刻；本次不重跑交通网络，不以站口触发点数认定停运、孤岛或受困人数。</p><h2>解释范围与下一步</h2><p>这一步支持把降雨、水位与几何作为独立实验因素保留，并把不同驱动下的暴露变化接入同一交通图继续比较。严格道路、步道补缺及未知桥隧分支保持原样。是否形成交通孤岛仍取决于同步道路限制、站口／车站／区段停运假设及地面替代路径，不能由本页峰值面积直接推出。</p><p>未使用观测河道水位、完整管网泵站或独立灾情验证；几何对照是城市信息组合的模型敏感性，尚未证明三维资料必然提高真实准确性。原人口不确定性、12＋4划分及封存测试保持不变。</p>'
 body+='<p>全部12组原数值门槛通过；本次'+str(nchecks)+'项对照检查通过。独立校验和逐文件版本清单见下。Framelog: MILI-D-20261017-006。</p><h2>复现文件</h2><ul>'
 for name in ['summary.json','comparison_contract.json','case_peak_extent.csv','common_quality_coverage.csv','conditional_forcing_contrasts.csv','conditional_depth_contrasts.csv','geometry_effects.csv','geometry_stability.csv','full_domain_water_budget.csv','full_domain_budget_at_times.csv','water_group_exchange.csv','station_exposure_summary.csv','station_exposure_detail.csv','validation.json','independent_validation.json','source_bindings.json','map_metadata.json','delivery_manifest.json']:
  body+='<li><a href="'+name+'">'+name+'</a></li>'
 body+='</ul>'
 (O/'review.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>无雨与增强降雨：三区城市几何稳定性</title><style>body{font:16px/1.8 system-ui;max-width:1240px;margin:auto;padding:30px;color:#263e4b;background:#f3f6f8}h1,h2{line-height:1.4}h2{margin-top:36px}aside{padding:18px;background:#e5f0ed;border-left:5px solid #3a806e}table{border-collapse:collapse;width:100%;font-size:14px;background:white}td,th{border:1px solid #d7e2e8;padding:9px;text-align:left}th{background:#e5edf1}.scroll{overflow:auto}img{max-width:100%;background:white}a{color:#17648c}</style>'+body+'</html>')


def interpret_saved_products():
 """Explain threshold-pattern turnover using only already exported depth bounds."""
 summary=json.loads((O/'summary.json').read_text());support=np.load(Q/'common_support.npz');mask=support['common_admitted'];ward=support['ward_code'];transitions=[]
 for stage in STAGES:
  for name in ['ground','road']:
   w=support['ground_area'if name=='ground'else'road_area'];sel=mask&(ward>0)&(w>0);threshold=.1 if name=='ground'else .3
   z1=np.load(Q/(cid('R1',stage,'U')+'_peak_bounds.npz'));t1=np.load(Q/(cid('R1',stage,'T')+'_peak_bounds.npz'));z2=np.load(Q/(cid('R125',stage,'U')+'_peak_bounds.npz'));t2=np.load(Q/(cid('R125',stage,'T')+'_peak_bounds.npz'))
   old=(z1[name+'_lower']>threshold)&(t1[name+'_upper']<=threshold)
   retained=(z2[name+'_lower']>threshold)&(t2[name+'_upper']<=threshold)
   bothwet=(z2[name+'_lower']>threshold)&(t2[name+'_lower']>threshold)
   bothdry=(z2[name+'_upper']<=threshold)&(t2[name+'_upper']<=threshold)
   reversed_=(z2[name+'_upper']<=threshold)&(t2[name+'_lower']>threshold)
   states=[retained,bothwet,bothdry,reversed_];count=sum(x.astype('i4')for x in states);assert not np.any(count[sel]>1)
   total=area_sum(w,sel&old);areas=[area_sum(w,sel&old&x)for x in states];uncertain=area_sum(w,sel&old&(count==0));assert abs(total-sum(areas)-uncertain)<1e-10
   transitions.append(dict(stage=stage,support=name,threshold_m=threshold,original_U_only_wet_km2=total,remains_U_only_wet_km2=areas[0],both_arms_wet_under_stronger_rain_km2=areas[1],both_arms_dry_under_stronger_rain_km2=areas[2],reversed_to_T_only_wet_km2=areas[3],uncertain_km2=uncertain,both_arms_wet_fraction=ratio(areas[1],total)))
 csvwrite(O/'threshold_transition_explanation.csv',transitions)
 summary['threshold_transitions']=transitions
 summary['interpretation']={
  'rain_and_stage':'Dry-low native budget is exactly zero; dry-raised flooding is caused by the imposed stage within this model. Rain-stage response is nonlinear and thresholded. No source-fraction estimate.',
  'geometry_direction':'In dry-raised controls U has smaller inundation extent than T; with either positive rainfall U has larger regional threshold extents. This is a forcing-dependent bundled response.',
  'stability':'About97-98% of the original footprint with U-minus-T depth increase>=0.1m retains that increase under1.25 rainfall. U-only road threshold footprint retention is about47-48%; inspect convergence of both arms above the threshold before calling this a spatial reversal.',
  'not_established':'No real-world accuracy advantage, actual rail shutdown, traffic-island count or affected population established.'}
 dump(O/'summary.json',summary)
 def get_extent(rain,stage,arm,support):return next(x['certain_km2']for x in summary['case_peak_extent']if x['case_id']==cid(rain,stage,arm)and x['support']==support)
 dryt=get_extent('R0','H','T','ground');dryu=get_extent('R0','H','U','ground')
 stagewet=next(x['difference_lower_km2']for x in summary['forcing_contrasts']if x['rainfall']=='R1'and x['arm']=='U'and x['support']=='ground'and x['effect']=='stage_with_rain')
 block='<section id="factor-findings"><h2>主要发现</h2><ul>'
 block+=f'<li>水位本身可以造成地表积水：无雨抬升情景中，T与U的&gt;0.1 m峰值湿区分别为{dryt:.3f}与{dryu:.3f} km²；低水位无雨控制为零。</li>'
 block+=f'<li>在U组原降雨下，抬升水位的净增面积为{stagewet:.3f} km²，与无雨时的{dryu:.3f} km²不同。雨量与水位的超阈值足迹有重叠和交互，不能简单相加。</li>'
 block+='<li>几何响应随驱动而变：无雨抬升下U湿区小于T；原雨量与增强雨量下U的三区湿区和道路深水面积均大于T。</li>'
 block+='<li>原有几何加深≥0.1 m的地表／道路足迹约97—98%在增强雨量下仍保持。U独有的道路&gt;0.3 m足迹仅约47—48%继续独有，其余主要转为T/U两组均超阈值；以下另表列出转换，而非把差异消失叫作加深方向翻转。</li></ul>'
 block+=table(['水位','支持','原U独有湿区 km²','仍U独有 km²','增强后两组均湿 km²','两组均干 km²','翻为T独有 km²','区间未定 km²'],[[x['stage'],x['support'],f"{x['original_U_only_wet_km2']:.4f}",f"{x['remains_U_only_wet_km2']:.4f}",f"{x['both_arms_wet_under_stronger_rain_km2']:.4f}",f"{x['both_arms_dry_under_stronger_rain_km2']:.4f}",f"{x['reversed_to_T_only_wet_km2']:.4f}",f"{x['uncertain_km2']:.4f}"]for x in transitions])
 block+='<p>上述稳定性限于本雨场家族与L/H假设。交通判断还需使用同步时刻和同一网络；峰值包络不能直接证明孤岛。</p></section>'
 page=O/'review.html';text=page.read_text();start=text.find('<section id="factor-findings">')
 if start>=0:
  end=text.index('</section>',start)+len('</section>');text=text[:start]+text[end:]
 text=text.replace('<h2>比较口径与质量覆盖</h2>',block+'<h2>比较口径与质量覆盖</h2>')
 if 'href="threshold_transition_explanation.csv"'not in text:text=text.replace('<h2>复现文件</h2><ul>','<h2>复现文件</h2><ul><li><a href="threshold_transition_explanation.csv">threshold_transition_explanation.csv</a></li>')
 page.write_text(text)

if __name__=='__main__':main()
