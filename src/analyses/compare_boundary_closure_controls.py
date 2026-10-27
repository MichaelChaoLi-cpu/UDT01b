"""Whole-domain matched closure-fix changes, interval-aware extent and fixed peak regions."""
from pathlib import Path
import os,sys,json
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-mpl')
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'));sys.path.insert(0,str(ROOT/'src/analyses'))
from prepare_boundary_closure_fix import OUT,REPORT,PRIOR,PRIOR_REPORT,CASES
from prepare_sfincs_cpu_build import dump
from audit_three_ward_negative_volumes import readnp,csvwrite

def main():
 out=ROOT/OUT;report=ROOT/REPORT
 assert not(out/'freeze.json').exists()
 support=readnp(ROOT/'data/processed/three_ward_common_depth_labels/common_support_10m.npz');valid=support['Eligible Ground Mask'];ward=support['Ward Code'];area=support['Ground Area']
 scopes={'Whole domain':valid,'Three wards':valid&(ward>0),'Chiyoda':valid&(ward==1),'Chuo':valid&(ward==2),'Minato':valid&(ward==3),'Contributing area':valid&(ward==0)}
 graph=readnp(ROOT/'data/processed/three_ward_surrogate_inputs/surface_connectivity_graph.npz');flat=graph['Node Grid Index'];edges=graph['Surface Edge Index'];adj=coo_matrix((np.ones(edges.shape[1],np.uint8),(edges[0],edges[1])),shape=(len(flat),len(flat))).tocsr()
 rows=[];regions=[];negative=[];changes={};oldsummary=json.loads((ROOT/PRIOR_REPORT/'summary.json').read_text());newsummary=json.loads((report/'summary.json').read_text());newbudget=json.loads((report/'budget/summary.json').read_text());oldbudget=json.loads((ROOT/PRIOR_REPORT/'budget/summary.json').read_text());checks=[];datasets={}
 for case in CASES:
  a=readnp(ROOT/PRIOR/(case+'_candidate_reference.npz'));b=readnp(out/(case+'_candidate_reference.npz'));al=a['Peak Mean Depth Lower'];au=a['Peak Mean Depth Upper'];bl=b['Peak Mean Depth Lower'];bu=b['Peak Mean Depth Upper'];ae=a['Online Ever Negative'];be=b['Online Ever Negative'];datasets[case]=(al,bl,ae,be)
  for scope,sel in scopes.items():
   negative.append({'case':case,'scope':scope,'old_ever_negative_cells':int((sel&ae).sum()),'new_ever_negative_cells':int((sel&be).sum()),'resolved_negative_cells':int((sel&ae&~be).sum()),'newly_negative_cells':int((sel&~ae&be).sum()),'old_negative_ground_km2':float(area[sel&ae].sum()/1e6),'new_negative_ground_km2':float(area[sel&be].sum()/1e6)})
   for screening,mask in [('all_eligible_ground',sel),('mutually_cell_screened',sel&a['Admitted Mask']&b['Admitted Mask'])]:
    delta=bl-al;minimum_required_change=np.maximum(np.maximum(bl-au,al-bu),0.);finite=mask&np.isfinite(delta);w=area[finite];v=abs(delta[finite]);order=np.argsort(v);p95=float(v[order][np.searchsorted(np.cumsum(w[order]),.95*w.sum())])if w.sum()else None
    for th in [.05,.1,.3,.5]:
     ac=mask&(al>th);bc=mask&(bl>th);ap=mask&(au>th);bp=mask&(bu>th);inter=float(area[ac&bc].sum());union=float(area[ac|bc].sum());possibleunion=float(area[ap|bp].sum());possibleinter=float(area[ap&bp].sum());oldarea=float(area[ac].sum());newarea=float(area[bc].sum())
     rows.append({'case':case,'scope':scope,'support':screening,'threshold_m':th,'old_certain_wet_km2':oldarea/1e6,'new_certain_wet_km2':newarea/1e6,'certain_wet_area_change_pct':(newarea-oldarea)/oldarea*100 if oldarea else None,'certain_extent_iou':inter/union if union else None,'interval_extent_iou_lower':inter/possibleunion if possibleunion else None,'interval_extent_iou_upper':min(1.,possibleinter/union)if union else None,'added_certain_wet_km2':float(area[bc&~ac].sum()/1e6),'removed_certain_wet_km2':float(area[ac&~bc].sum()/1e6),'old_threshold_ambiguous_km2':float(area[ap&~ac].sum()/1e6),'new_threshold_ambiguous_km2':float(area[bp&~bc].sum()/1e6),'max_minimum_required_abs_depth_change_m':float(minimum_required_change[finite].max())if finite.any()else None,'ground_minimum_required_change_gt_0_01m_km2':float(area[finite&(minimum_required_change>.01)].sum()/1e6),'ground_minimum_required_change_gt_0_1m_km2':float(area[finite&(minimum_required_change>.1)].sum()/1e6),'max_abs_lower_bound_depth_change_m':float(v.max())if len(v)else None,'ground_weighted_p95_abs_lower_bound_change_m':p95,'ground_depth_lower_bound_change_gt_0_01m_km2':float(area[finite&(abs(delta)>.01)].sum()/1e6),'mean_depth_change_lower_bound_m':float(np.sum(area[finite]*(bl-au)[finite])/w.sum())if w.sum()else None,'mean_depth_change_upper_bound_m':float(np.sum(area[finite]*(bu-al)[finite])/w.sum())if w.sum()else None})
  # Keep the old connected peak-region footprints fixed, so ranks cannot conceal loss.
  for th in [.1,.3]:
   ids=np.flatnonzero(al.ravel()[flat]>th);nc,labels=connected_components(adj[ids][:,ids],directed=False);sizes=np.bincount(labels,weights=area.ravel()[flat[ids]],minlength=nc)
   for rank,k in enumerate(np.argsort(sizes)[::-1][:10],1):
    oldids=ids[labels==k];ff=flat[oldids];ret=oldids[bl.ravel()[ff]>th];rn,rl=connected_components(adj[ret][:,ret],directed=False);largest=float(np.bincount(rl,weights=area.ravel()[flat[ret]],minlength=rn).max())if len(ret)else 0.
    regions.append({'case':case,'threshold_m':th,'old_region_rank':rank,'old_ground_km2':float(sizes[k]/1e6),'new_certain_wet_retention_pct':float(area.ravel()[ff[bl.ravel()[ff]>th]].sum()/sizes[k]*100),'largest_retained_connected_support_pct':largest/sizes[k]*100,'old_negative_support_pct':float(area.ravel()[ff[ae.ravel()[ff]]].sum()/sizes[k]*100),'new_negative_support_pct':float(area.ravel()[ff[be.ravel()[ff]]].sum()/sizes[k]*100),'meaning':'fixed old peak-support region; not simultaneous pool or dynamic drainage route'})
  x=newsummary['cases'][case];y=oldsummary['cases'][case];oldc=y['boundary_closure_audit'];newc=x['boundary_closure_audit'];comparison={'old_ever_negative_cells':y['ever_negative_cells'],'new_ever_negative_cells':x['ever_negative_cells'],'old_down_crossings':y['down_crossing_events'],'new_down_crossings':x['down_crossing_events'],'old_final_negative_cells':y['final_negative_cells'],'new_final_negative_cells':x['final_negative_cells'],'old_minimum_signed_volume_m3':y['minimum_volume_m3'],'new_minimum_signed_volume_m3':x['minimum_volume_m3'],'old_closed_boundary_faces_with_flux':oldc['faces_with_nonzero_applied_flux'],'new_closed_boundary_faces_with_flux':newc['faces_with_nonzero_applied_flux'],'old_gross_closed_out_m3':oldc['gross_closed_outward_m3'],'new_gross_closed_out_m3':newc['gross_closed_outward_m3'],'old_gross_closed_in_m3':oldc['gross_closed_inward_m3'],'new_gross_closed_in_m3':newc['gross_closed_inward_m3'],'old_applied_rain_m3':oldbudget['case_summaries'][case]['actual_applied_rain_m3'],'new_applied_rain_m3':newbudget['case_summaries'][case]['actual_applied_rain_m3']};changes[case]=comparison
  checks.append({'case':case,'check':'all_declared_closed_boundary_exchanges_exactly_zero','pass':newc['faces_with_nonzero_applied_flux']==0 and newc['gross_closed_outward_m3']==newc['gross_closed_inward_m3']==0})
  checks.append({'case':case,'check':'actual_applied_rain_matches_previous','pass':comparison['old_applied_rain_m3']==comparison['new_applied_rain_m3']})
 csvwrite(report/'matched_extent_changes.csv',rows);csvwrite(report/'fixed_principal_region_changes.csv',regions);csvwrite(report/'matched_negative_changes.csv',negative)
 dump(report/'matched_comparison.json',{'cases':changes,'checks':checks,'all_checks_passed':all(c['pass']for c in checks),'compared_support':'same eligible original ground, with all-ground and mutually-cell-screened views; whole-event admission considered separately','extent_definition':'strict threshold on per-common-cell peak-mean-depth bounds, not instantaneous flood extent','iou_bounds':'conservative bounds use certainly wet intersections and possibly wet unions; neither bound is chosen as evaluation truth','physical_accuracy_verified':False,'corrected_grid_convergence_verified':False,'new_hydrodynamic_runs_here':0})
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.colors import ListedColormap,TwoSlopeNorm
 fig,axs=plt.subplots(3,2,figsize=(12,16),layout='constrained');extent=[-11.36,-3.66,-42.44,-32.32]
 for col,case in enumerate(CASES):
  al,bl,ae,be=datasets[case]
  for row,(depth,title)in enumerate([(al,'Before closure fix'),(bl,'After closure fix')]):
   ax=axs[row,col];ax.imshow(np.where(valid,1,np.nan),extent=extent,cmap=ListedColormap(['#e7ecee']),interpolation='none');m=ax.imshow(np.where(valid&(depth>.01),depth,np.nan),extent=extent,cmap='Blues',vmin=0,vmax=1,interpolation='none');ax.set_title(('Dry'if col==0 else'Rain')+' | '+title)
  delta=bl-al;ax=axs[2,col];ax.imshow(np.where(valid,1,np.nan),extent=extent,cmap=ListedColormap(['#e7ecee']),interpolation='none');d=ax.imshow(np.where(valid&(abs(delta)>1e-6),delta,np.nan),extent=extent,cmap='RdBu',norm=TwoSlopeNorm(vmin=-.1,vcenter=0,vmax=.1),interpolation='none');ax.set_title('After minus before | depth lower bounds')
  for ax in axs[:,col]:
   for code in [1,2,3]:ax.contour(-11.36+(np.arange(770)+.5)*.01,-32.32-(np.arange(1012)+.5)*.01,(ward==code).astype('f4'),levels=[.5],colors='#6b7f87',linewidths=.45)
   ax.set_aspect('equal');ax.set_xlabel('EPSG:6677 easting (km)');ax.set_ylabel('EPSG:6677 northing (km)')
 fig.colorbar(m,ax=list(axs[:2].ravel()),label='Peak mean depth lower bound (m)',shrink=.7,extend='max');fig.colorbar(d,ax=list(axs[2]),label='Lower-bound depth change (m)',shrink=.7,extend='both');fig.savefig(report/'whole_domain_closure_comparison.png',dpi=160);plt.close(fig)
 print(json.dumps(changes,indent=2));assert all(c['pass']for c in checks)
if __name__=='__main__':main()
