"""Whole-three-ward grid sensitivity with peak aggregation and censor intervals retained."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-mpl')
from pathlib import Path
import sys,json,argparse
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'));sys.path.insert(0,str(ROOT/'src/analyses'))
from prepare_adaptive_positivity import OUT,REPORT,PRIOR,CASES
from prepare_sfincs_cpu_build import dump
from audit_three_ward_negative_volumes import readnp,csvwrite

def wquantile(values,weights,q):
 if not len(values)or weights.sum()==0:return None
 s=np.argsort(values);return float(values[s[min(int(np.searchsorted(np.cumsum(weights[s]),q*weights.sum())),len(s)-1)]])
def main(only=None):
 out=ROOT/OUT;report=ROOT/REPORT;support=readnp(ROOT/'data/processed/three_ward_common_depth_labels/common_support_10m.npz');local=readnp(out/'common_local_support_5m.npz');valid=support['Eligible Ground Mask'];ward=support['Ward Code'];area=support['Ground Area'];scopes={'Whole domain':valid,'Three wards':valid&(ward>0),'Chiyoda':valid&(ward==1),'Chuo':valid&(ward==2),'Minato':valid&(ward==3),'Contributing area':valid&(ward==0)}
 graph=readnp(ROOT/'data/processed/three_ward_surrogate_inputs/surface_connectivity_graph.npz');flat=graph['Node Grid Index'];edges=graph['Surface Edge Index'];adj=coo_matrix((np.ones(edges.shape[1],np.uint8),(edges[0],edges[1])),shape=(len(flat),len(flat))).tocsr()
 extents=[];depths=[];coverage=[];regions=[];intervals=[];plots={};checks=[]
 for case in ([only]if only else CASES):
  a=readnp(ROOT/PRIOR/(case+'_candidate_reference.npz'));b=readnp(out/(case+'_adaptive_candidate_reference.npz'));al=a['Peak Mean Depth Lower'];au=a['Peak Mean Depth Upper'];bl=b['Peak Mean Depth Lower'];bu=b['Peak Mean Depth Upper'];q=readnp(out/(case+'_local_depth_bounds_5m.npz'))
  signed=np.where(bl>au,bl-au,np.where(bu<al,bu-al,0));required=abs(signed);plots[case]=(al,bl,signed,b['Peak Interval Width'])
  checks.append({'case':case,'check':'common10m_intervals_finite_ordered_on_identical_support','pass':bool(np.all(np.isfinite(bl[valid]))and np.all(np.isfinite(bu[valid]))and np.all(bu[valid]>=bl[valid]))})
  for name,sel in scopes.items():
   coverage.append({'case':case,'scope':name,**{prefix+'_'+key.lower().replace(' ','_')+'_km2':float(area[sel&d[key]].sum()/1e6)for prefix,d in [('regular',a),('adaptive',b)]for key in ['Exact Label Mask','Interval Label Mask','Wide Interval Mask','Quarantine Mask','Admitted Mask']}})
   w=area[sel];width=(bu-bl)[sel]
   intervals.append({'case':case,'scope':name,'mean_interval_width_m':float(np.average(width,weights=w)),'p95_interval_width_m':wquantile(width,w,.95),'maximum_interval_width_m':float(width.max()),'interval_width_gt_0_01m_km2':float(area[sel&(bu-bl>.010001)].sum()/1e6),'sample_300_minus_600_depth_mean_m':float(np.average((b['Sample Peak Lower 300s']-b['Sample Peak Lower 600s'])[sel],weights=w))})
   for screening,mask in [('all_eligible_ground',sel),('mutually_cell_screened',sel&a['Admitted Mask']&b['Admitted Mask'])]:
    for threshold in [.05,.1,.3,.5]:
     ac=mask&(al>threshold);ap=mask&(au>threshold);bc=mask&(bl>threshold);bp=mask&(bu>threshold);i=float(area[ac&bc].sum());u=float(area[ac|bc].sum());pu=float(area[ap|bp].sum());pi=float(area[ap&bp].sum());aa=float(area[ac].sum());ba=float(area[bc].sum())
     extents.append({'case':case,'scope':name,'support':screening,'threshold_m':threshold,'regular_certain_wet_km2':aa/1e6,'adaptive_certain_wet_km2':ba/1e6,'adaptive_area_change_pct':(ba-aa)/aa*100 if aa else None,'certain_extent_iou':i/u if u else None,'interval_extent_iou_lower':i/pu if pu else None,'interval_extent_iou_upper':min(1.,pi/u)if u else None,'regular_ambiguous_km2':float(area[ap&~ac].sum()/1e6),'adaptive_ambiguous_km2':float(area[bp&~bc].sum()/1e6),'added_certain_km2':float(area[bc&~ac].sum()/1e6),'removed_certain_km2':float(area[ac&~bc].sum()/1e6)})
    for res in [10,5]:
     if res==10:xlo,xup,ylo,yup,ar,ms=al,au,bl,bu,area,mask
     else:xlo,xup,ylo,yup,ar,ms=q['Regular Lower'],q['Regular Upper'],q['Adaptive Lower'],q['Adaptive Upper'],local['Ground Area'],np.repeat(np.repeat(mask,2,0),2,1)&local['Eligible Ground Mask']
     req=np.maximum(np.maximum(ylo-xup,xlo-yup),0.)
     for subset,ss in [('all',ms),('either_possibly_wet_gt_0_1m',ms&((xup>.1)|(yup>.1)))]:
      if not ss.any():continue
      ww=ar[ss];rr=req[ss]
      depths.append({'case':case,'scope':name,'support':screening,'resolution_m':res,'subset':subset,'ground_km2':float(ww.sum()/1e6),'confirmed_abs_change_weighted_mean_m':float(np.average(rr,weights=ww)),'confirmed_abs_change_p50_m':wquantile(rr,ww,.5),'confirmed_abs_change_p95_m':wquantile(rr,ww,.95),'confirmed_abs_change_p99_m':wquantile(rr,ww,.99),'confirmed_abs_change_max_m':float(rr.max()),'confirmed_change_gt_0_01m_km2':float(ar[ss&(req>.01)].sum()/1e6),'confirmed_change_gt_0_1m_km2':float(ar[ss&(req>.1)].sum()/1e6),'confirmed_change_gt_0_3m_km2':float(ar[ss&(req>.3)].sum()/1e6),'mean_signed_change_lower_m':float(np.average((ylo-xup)[ss],weights=ww)),'mean_signed_change_upper_m':float(np.average((yup-xlo)[ss],weights=ww))})
  for threshold in [.1,.3]:
   ids=np.flatnonzero((al.ravel()[flat]>threshold)&(ward.ravel()[flat]>0));nc,labels=connected_components(adj[ids][:,ids],directed=False);sizes=np.bincount(labels,weights=area.ravel()[flat[ids]],minlength=nc)
   for rank,k in enumerate(np.argsort(sizes)[::-1][:10],1):
    idsold=ids[labels==k];ff=flat[idsold];ret=idsold[bl.ravel()[ff]>threshold];rn,rl=connected_components(adj[ret][:,ret],directed=False);largest=float(np.bincount(rl,weights=area.ravel()[flat[ret]],minlength=rn).max())if len(ret)else 0.
    regions.append({'case':case,'threshold_m':threshold,'regular_region_rank':rank,'regular_ground_km2':float(sizes[k]/1e6),'adaptive_certain_retention_pct':float(area.ravel()[ff[bl.ravel()[ff]>threshold]].sum()/sizes[k]*100),'adaptive_possible_retention_pct':float(area.ravel()[ff[bu.ravel()[ff]>threshold]].sum()/sizes[k]*100),'largest_retained_support_on_fixed_10m_graph_pct':largest/sizes[k]*100,'interpretation':'Fixed whole-three-ward peak-support regions on pre-existing10m diagnostic graph; not simultaneous pools or dynamic routes.'})
 prefix=only+'_'if only else''
 for name,rows in [('matched_extent',extents),('matched_local_depth',depths),('matched_label_support',coverage),('fixed_principal_regions',regions),('peak_interval_summary',intervals)]:csvwrite(report/(prefix+name+'.csv'),rows)
 dump(report/(prefix+'spatial_comparison.json'),{'all_checks_passed':all(c['pass']for c in checks),'checks':checks,'extent_definition':'peak over time of common-cell eligible-ground mean depth with censoring and asynchronous native-peak bounds retained','local5m_definition':'common5m original ground, each covered by one native state, no interpolation of depth','differences':'minimum required interval-separated differences are confirmed changes; overlapping intervals yield0 lower bound, not proven equality','new_events':0,'real_event_accuracy_verified':False,'formal_grid_convergence_verified':False})
 if only:return
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.colors import ListedColormap,TwoSlopeNorm
 fig,axs=plt.subplots(3,2,figsize=(12,16),layout='constrained');extent=[-11.36,-3.66,-42.44,-32.32]
 for col,case in enumerate(CASES):
  al,bl,delta,width=plots[case]
  for row,(depth,title)in enumerate([(al,'Uniform10 m'),(bl,'Adaptive10 /5 m')]):
   ax=axs[row,col];ax.imshow(np.where(valid,1,np.nan),extent=extent,cmap=ListedColormap(['#e7ecee']),interpolation='none');m=ax.imshow(np.where(valid&(depth>.01),depth,np.nan),extent=extent,cmap='Blues',vmin=0,vmax=1,interpolation='none');ax.set_title(('Dry control'if col==0 else'Rain control')+' | '+title)
  ax=axs[2,col];ax.imshow(np.where(valid,1,np.nan),extent=extent,cmap=ListedColormap(['#e7ecee']),interpolation='none');d=ax.imshow(np.where(valid&(abs(delta)>1e-6),delta,np.nan),extent=extent,cmap='RdBu',norm=TwoSlopeNorm(vmin=-.1,vcenter=0,vmax=.1),interpolation='none');ax.set_title('Adaptive minus uniform | confirmed depth change')
  for ax in axs[:,col]:
   for code in [1,2,3]:ax.contour(-11.36+(np.arange(770)+.5)*.01,-32.32-(np.arange(1012)+.5)*.01,(ward==code).astype('f4'),levels=[.5],colors='#6b7f87',linewidths=.45)
   ax.set_aspect('equal');ax.set_xlabel('EPSG:6677 easting (km)');ax.set_ylabel('EPSG:6677 northing (km)')
 fig.colorbar(m,ax=list(axs[:2].ravel()),label='Peak mean depth lower bound (m)',shrink=.7,extend='max');fig.colorbar(d,ax=list(axs[2]),label='Interval-separated change (m)',shrink=.7,extend='both');fig.savefig(report/'whole_domain_grid_comparison.png',dpi=150);plt.close(fig)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--only',choices=CASES);main(a.parse_args().only)
