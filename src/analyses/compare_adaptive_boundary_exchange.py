"""Match physical receiving interfaces and compare gross directional exchanges on actual clocks."""
from pathlib import Path
import os,sys,json,csv,argparse,hashlib
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-mpl')
import numpy as np
from netCDF4 import Dataset,chartostring
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'));sys.path.insert(0,str(ROOT/'src/analyses'))
from prepare_adaptive_positivity import OUT,REPORT,PRIOR,CASES
from prepare_sfincs_cpu_build import dump,sha
from reference_label_policy import CONTRACT
from audit_adaptive_positivity_controls import topology,structured,arr
from audit_three_ward_negative_volumes import csvwrite,readnp
from audit_three_ward_reference_contract import polyline_sections
CODES=[17,51,71,79,106,110,111]
GROUP_NAMES=json.loads((ROOT/'data/processed/three_ward_negative_trace_interface_v1/output_schema.json').read_text())['groups']

def geometry():
 out=ROOT/OUT;geo=ROOT/'data/processed/three_ward_adaptive_connectivity';runs=json.loads((out/'run_manifest.json').read_text())['runs'];reg=ROOT/runs[0]['regular_reference_folder'];path=ROOT/'data/processed/three_ward_negative_trace_interface_v1/whole_domain_metadata/whole_native_topology.bin'
 with path.open('rb')as f:
  n,nf=np.fromfile(f,'<i4',2);f.seek(n*4*2,1);mask=np.fromfile(f,'<i4',n);f.seek(n*4*4,1);a=np.fromfile(f,'<i4',nf)-1;b=np.fromfile(f,'<i4',nf)-1
 groups=np.zeros(n,np.int32)
 with(reg/'udt_negative_native_map.csv').open()as f:
  for row in csv.DictReader(f):groups[int(row['native_cell_index'])-1]=int(row['water_group_code'])
 regular={'mask':mask,'a':a,'b':b,'groups':groups};ad=topology(ROOT/runs[0]['folder']/'udt_adaptive_topology.bin');adaptive={'mask':ad['mask'],'a':ad['a']-1,'b':ad['b']-1,'groups':np.loadtxt(ROOT/runs[0]['folder']/'udt_adaptive_groups.txt',dtype=int)[:,1]};cl=np.memmap(geo/'fine_class.bin',mode='r',dtype='uint8',shape=(20240,15400));sets={};receipts=[]
 for kind,d in [('regular',regular),('adaptive',adaptive)]:
  fg=readnp(geo/(kind+'_face_geometry.npz'));sel=np.flatnonzero(((d['mask'][d['a']]==1)&(d['mask'][d['b']]==2))|((d['mask'][d['a']]==2)&(d['mask'][d['b']]==1)));water=np.where(d['mask'][d['a'][sel]]==2,d['a'][sel],d['b'][sel]);codes=d['groups'][water];pixels=set();roads=set();counts={}
  for ix,j in enumerate(sel):
   di=int(fg['direction'][j]);rr=int(fg['row_fine'][j]);cc=int(fg['col_fine'][j]);length=int(fg['length_pixels'][j]);num=0
   for k in range(length):
    r=rr+(k if di==0 else 0);c=cc+(k if di==1 else 0);rb=r+(1 if di==1 else 0);ca=c-(1 if di==0 else 0)
    if cl[r,ca]!=1 and cl[rb,c]!=1:
     key=(int(codes[ix]),di,r,c);assert key not in pixels;pixels.add(key);num+=1
     if cl[r,ca]==2 and cl[rb,c]==2:roads.add(key)
   assert num*.5==fg['native_open_width_m'][j]
  sets[kind]=(pixels,roads)
  for code in CODES:
   rows=codes==code;counts[str(code)]={'physical_faces':int(rows.sum()),'open_face_count':int(np.sum(fg['native_open_width_m'][sel[rows]]>0)),'native_open_width_m':float(np.sum(fg['native_open_width_m'][sel[rows]])),'road_width_m':float(np.sum(fg['native_road_width_m'][sel[rows]]))}
  length=fg['length_pixels'][sel]*.5;di=fg['direction'][sel];xx=-11360+fg['col_fine'][sel]*.5+np.where(di==1,length/2,0);yy=-32320-fg['row_fine'][sel]*.5-np.where(di==1,.5,length/2)
  d.update(face_ids=sel,face_groups=codes,midpoints=np.c_[xx,yy],directions=di);receipts.append({'grid':kind,'boundary_faces':len(sel),'open_half_metre_segments':len(pixels),'road_half_metre_segments':len(roads),'by_water_group':counts})
 equal=sets['regular'][0]==sets['adaptive'][0]and sets['regular'][1]==sets['adaptive'][1]
 check={'all_physical_open_receiving_segments_equal':equal,'regular_only_open_segments':len(sets['regular'][0]-sets['adaptive'][0]),'adaptive_only_open_segments':len(sets['adaptive'][0]-sets['regular'][0]),'grids':receipts,'scope':'Exact paired original0.50m non-building crossing pixels and water-group identity; closed building-only faces may disappear under refinement.'};dump(ROOT/REPORT/'matched_boundary_geometry.json',check);assert equal,check
 return regular,adaptive

def history_groups(path,face_groups,expected_names=None):
 ledger=structured(path/'udt_step_budget.csv');clock=structured(path/'udt_output_clock.csv');hist=clock[clock['write_history']=='T'];steps=hist['step_index'];dt=ledger['actual_step_dt'][steps].astype('f4');increments=np.zeros((len(hist),14),'f8');finite=True
 with Dataset(path/'sfincs_his.nc')as d:
  if expected_names is not None:assert [str(x).strip()for x in chartostring(d['crosssection_name'][:])]==expected_names
  for i in range(0,len(hist),256):
   q=arr(d['crosssection_discharge'][i:i+256]).astype('f4');finite &=bool(np.all(np.isfinite(q)));amount=(q*dt[i:i+len(q),None]).astype('f8')
   for j,code in enumerate(CODES):
    v=amount[:,face_groups==code];increments[i:i+len(q),2*j]=np.maximum(v,0).sum(1);increments[i:i+len(q),2*j+1]=np.maximum(-v,0).sum(1)
 return ledger,hist,increments,finite

def main(only=None,geometry_only=False):
 out=ROOT/OUT;report=ROOT/REPORT;regular,adaptive=geometry()
 if geometry_only:print('PASS exact half-metre receiving support');return
 checks=[];results=[];timeseries=[];summaries={}
 for run in json.loads((out/'run_manifest.json').read_text())['runs']:
  case=run['case_id']
  if only and case!=only:continue
  path=ROOT/run['folder'];reg=ROOT/run['regular_reference_folder']
  def section_mapping(folder,geometry,grid):
   sections=polyline_sections(folder/'sfincs.crs');mid=np.array([xy.mean(0)for name,xy in sections]);distance,index=cKDTree(geometry['midpoints']).query(mid);ok=len(index)==len(geometry['face_ids'])and len(np.unique(index))==len(index)and np.max(distance)<=CONTRACT['coordinate_tolerance_m'];checks.append({'case':case,'check':grid+'_sections_biject_actual_physical_faces_with_frozen_coordinate_tolerance','pass':bool(ok),'maximum_midpoint_error_m':float(np.max(distance)),'frozen_coordinate_tolerance_m':CONTRACT['coordinate_tolerance_m']});assert ok
   faceids=geometry['face_ids'][index];normal=np.array([[-(xy[1,1]-xy[0,1]),xy[1,0]-xy[0,0]]for name,xy in sections]);sign=np.where(geometry['mask'][geometry['a'][faceids]]==1,1,-1);direction=geometry['directions'][index];assert np.all(normal[np.arange(len(index)),direction]*sign>0)
   return geometry['face_groups'][index],[name for name,xy in sections]
  rg,rnames=section_mapping(reg,regular,'regular');ag,anames=section_mapping(path,adaptive,'adaptive')
  rl,rhist,ri,rf=history_groups(reg,rg,rnames);al,ahist,ai,af=history_groups(path,ag,anames);ex=structured(path/'udt_adaptive_exchange.csv');aa=np.array([ex[f'group{code}_{direction}']for code in CODES for direction in ['out','in']]).T
  checks.append({'case':case,'check':'regular_history_covers_every_applied_step','pass':bool(np.array_equal(rhist['step_index'],np.arange(1,len(rl))))})
  checks.append({'case':case,'check':'all_boundary_history_records_finite_and_ordered','pass':rf and af})
  # New histories can skip CFL substeps: compare each written value with its actual last-step diagnostic, never multiply by history spacing.
  delta=float(np.max(abs(ai-aa[ahist['step_index']-1])));scale=float(np.max(aa))
  checks.append({'case':case,'check':'adaptive_output_boundary_flux_matches_actual_step_exchange','pass':delta<=max(1e-7,16*np.finfo('f4').eps*scale),'maximum_group_step_difference_m3':delta})
  rt=np.r_[0,rhist['actual_state_time']];at=np.r_[0,ex['time_s']];rc=np.vstack([np.zeros(14),np.cumsum(ri,axis=0)]);ac=np.vstack([np.zeros(14),np.cumsum(aa,axis=0)])
  checks.append({'case':case,'check':'independent_regular_group_exchange_reproduces_qualified_ledger','pass':abs(float(rc[-1,::2].sum())-float(rl['boundary_out_cumulative'][-1]))<1e-5 and abs(float(rc[-1,1::2].sum())-float(rl['boundary_in_cumulative'][-1]))<1e-5})
  qt=np.arange(0,14401,300);r300=np.array([np.interp(qt,rt,rc[:,j])for j in range(14)]).T;a300=np.array([np.interp(qt,at,ac[:,j])for j in range(14)]).T;rbins=np.diff(r300,axis=0);abins=np.diff(a300,axis=0)
  for j,code in enumerate(CODES):
   ro,rin=r300[-1,2*j:2*j+2];ao,ain=a300[-1,2*j:2*j+2];rn=rbins[:,2*j]-rbins[:,2*j+1];an=abins[:,2*j]-abins[:,2*j+1];valid=(np.maximum(abs(rn),abs(an))>1.)
   results.append({'case':case,'water_group_code':code,'water_group_name':GROUP_NAMES[str(code)],'regular_gross_out_m3':float(ro),'adaptive_gross_out_m3':float(ao),'gross_out_change_pct':float((ao-ro)/ro*100)if ro else None,'regular_gross_in_m3':float(rin),'adaptive_gross_in_m3':float(ain),'gross_in_change_pct':float((ain-rin)/rin*100)if rin else None,'regular_net_out_m3':float(ro-rin),'adaptive_net_out_m3':float(ao-ain),'net_out_difference_m3':float((ao-ain)-(ro-rin)),'bins_with_net_magnitude_gt_1m3':int(valid.sum()),'same_300s_net_direction_fraction':float(np.mean(np.sign(rn[valid])==np.sign(an[valid])))if valid.any()else None,'300s_net_increment_l1_difference_m3':float(np.sum(abs(an-rn))),'300s_net_increment_correlation':float(np.corrcoef(rn,an)[0,1])if np.std(rn)>0 and np.std(an)>0 else None})
   for k in range(1,len(qt)):timeseries.append({'case':case,'water_group_code':code,'from_s':int(qt[k-1]),'to_s':int(qt[k]),'regular_out_increment_m3':float(rbins[k-1,2*j]),'regular_in_increment_m3':float(rbins[k-1,2*j+1]),'adaptive_out_increment_m3':float(abins[k-1,2*j]),'adaptive_in_increment_m3':float(abins[k-1,2*j+1])})
  summaries[case]={'regular_gross_out_m3':float(r300[-1,::2].sum()),'adaptive_gross_out_m3':float(a300[-1,::2].sum()),'regular_gross_in_m3':float(r300[-1,1::2].sum()),'adaptive_gross_in_m3':float(a300[-1,1::2].sum()),'comparison_horizon_s':14400,'regular_actual_final_time_s':float(rt[-1]),'adaptive_actual_final_time_s':float(at[-1]),'regular_native_final_gross_out_m3':float(rc[-1,::2].sum()),'adaptive_native_final_gross_out_m3':float(ac[-1,::2].sum()),'regular_native_final_gross_in_m3':float(rc[-1,1::2].sum()),'adaptive_native_final_gross_in_m3':float(ac[-1,1::2].sum()),'regular_applied_rain_m3':float(rl['rain_applied_cumulative'][-1]),'adaptive_applied_rain_m3':float(al['rain_applied_cumulative'][-1]),'applied_rain_difference_m3':float(al['rain_applied_cumulative'][-1]-rl['rain_applied_cumulative'][-1]),'regular_history_records':len(rhist),'adaptive_history_records':len(ahist),'adaptive_applied_steps':len(ex),'gross_out_change_pct':float((a300[-1,::2].sum()/r300[-1,::2].sum()-1)*100),'gross_in_change_pct':float((a300[-1,1::2].sum()/r300[-1,1::2].sum()-1)*100)}
 for check in checks:check['pass']=bool(check['pass'])
 prefix=only+'_'if only else'';csvwrite(report/(prefix+'boundary_group_comparison.csv'),results);csvwrite(report/(prefix+'boundary_300s_exchange.csv'),timeseries);dump(report/(prefix+'boundary_comparison.json'),{'cases':summaries,'checks':checks,'all_checks_passed':all(c['pass']for c in checks),'time_aggregation':'Linearly resample cumulative actual-step directional transport to common300s edges; do not integrate sparse adaptive history as if it covered CFL substeps. Net-direction fraction excludes bins with both magnitudes<=1m3 as a descriptive denominator only. No hydraulic accuracy or stability pass threshold is introduced.'});assert all(c['pass']for c in checks),checks
 print(json.dumps(summaries,indent=2),flush=True)
 if only:return
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
 for j,case in enumerate(CASES):
  rows=[r for r in timeseries if r['case']==case];tt=np.arange(300,14401,300)/3600
  for i,(direction,label)in enumerate([('out','Gross outward'),('in','Gross inward')]):
   for grid,color in [('regular','#537e9c'),('adaptive','#c76c39')]:
    total=np.zeros(48)
    for code in CODES:total+=np.array([r[f'{grid}_{direction}_increment_m3']for r in rows if r['water_group_code']==code])
    axs[i,j].plot(tt,np.cumsum(total)/1e6,label='Uniform10 m'if grid=='regular'else'Adaptive10 /5 m',color=color)
   axs[i,j].set(title=('Dry'if j==0 else'Rain')+' | '+label,xlabel='Actual event time (hours)',ylabel='Cumulative exchange (million m3)');axs[i,j].grid(alpha=.2);axs[i,j].legend()
 fig.savefig(report/'boundary_directional_exchange.png',dpi=150);plt.close(fig)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--only',choices=CASES);a.add_argument('--geometry-only',action='store_true');args=a.parse_args();main(args.only,args.geometry_only)
