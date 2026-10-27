#!/usr/bin/env python3
"""Reconstruct native-aligned10m peak depths and rigorous adaptive aggregation bounds.

Read-only source data. No solver execution, ML training or network operations.
Uses existing numpy, numba, netCDF4, rasterio, shapely, pyproj. Products are geospatial
NPZ/GeoTIFF rather than tabular parquet; no extra table dependencies are required.
"""
import os
os.environ['NUMBA_NUM_THREADS'] = '4'
for key in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:
    os.environ.pop(key, None)
from pathlib import Path
import argparse, csv, hashlib, html, json, time
from datetime import datetime, timezone
from contextlib import ExitStack
import numpy as np
import netCDF4
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
from rasterio.features import rasterize
from shapely.geometry import shape
from shapely.ops import transform as transform_geometry
from pyproj import Transformer
from numba import njit, prange

CASES = ['rain60_low','rain100_low','rain60_raised','dry_raised']
ROOT = Path(__file__).resolve().parents[2]
NROW, NCOL = 1012, 770

def dump(p, o):
    p.write_text(json.dumps(o, ensure_ascii=False, indent=2, allow_nan=False)+'\n',encoding='utf-8')

def csvwrite(p, rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def readarr(v):return np.ma.filled(np.ma.asarray(v,dtype='float32'),np.nan)

@njit(parallel=True)
def ground_tables(z, cl, inside):
    nr=z.shape[0]//20;nc=z.shape[1]//20
    sorted_z=np.full((nr*nc,4,100),np.inf,np.float32)
    prefix=np.zeros((nr*nc,4,101),np.float64)
    counts=np.zeros((nr*nc,4),np.int32)
    road=np.zeros(nr*nc,np.float64);unknown=np.zeros(nr*nc,np.float64)
    for c in prange(nr*nc):
        r=c//nc;col=c%nc
        if not inside[r,col]:continue
        for q in range(4):
            rr=r*20+(q//2)*10;cc=col*20+(q%2)*10
            n=0
            for dr in range(10):
                for dc in range(10):
                    a=z[rr+dr,cc+dc];k=cl[rr+dr,cc+dc]
                    if k!=1:
                        if np.isfinite(a):
                            sorted_z[c,q,n]=a;n+=1
                            if k==2:road[c]+=.25
                        else:unknown[c]+=.25
            counts[c,q]=n
            sorted_z[c,q]=np.sort(sorted_z[c,q])
            for i in range(n):prefix[c,q,i+1]=prefix[c,q,i]+sorted_z[c,q,i]
    return sorted_z,prefix,counts,road,unknown

@njit(inline='always')
def volume(z,p,n,eta):
    # Exact sum over native0.50m eligible ground pixels, not a fitted storage table.
    if n==0:return 0.
    lo=0;hi=n
    while lo<hi:
        mid=(lo+hi)//2
        if z[mid]<eta:lo=mid+1
        else:hi=mid
    return max(0.,(lo*eta-p[lo])*.25)

@njit(parallel=True)
def bounds(z,pre,count,owners,regular_peak,regular_min,adaptive_peak,adaptive_min,states):
    cells=count.shape[0]
    # Rlower,Rupper,Alower,Aupper,SampleLower300,SampleUpper300,SampleLower600,SampleLower1200
    out=np.full((8,cells),np.nan,np.float64);nowner=np.zeros(cells,np.uint8)
    for c in prange(cells):
        area=np.sum(count[c])*.25
        if area==0:continue
        rlo=0.;rup=0.;up=0.;peaklo=np.zeros(4,np.float64)
        rp=regular_peak[c];rm=regular_min[c]
        for q in range(4):
            n=count[c,q]
            if n==0:continue
            if np.isfinite(rp):
                v=volume(z[c,q],pre[c,q],n,rp);rlo+=v;rup+=v
            else:rup+=volume(z[c,q],pre[c,q],n,rm+.01)
            j=owners[c,q]
            if j<0:continue # caller asserts this can occur only with zero eligible area
            ap=adaptive_peak[j]
            if np.isfinite(ap):
                peaklo[q]=volume(z[c,q],pre[c,q],n,ap);up+=peaklo[q]
            else:up+=volume(z[c,q],pre[c,q],n,adaptive_min[j]+.01)
        # All fine5m quadrants belonging to the same native hydraulic cell peak together.
        native_peak_lower=0.
        for q in range(4):
            if count[c,q]==0:continue
            j=owners[c,q];previous=False
            for p in range(q):
                if count[c,p]>0 and owners[c,p]==j:previous=True
            if previous:continue
            nowner[c]+=1;v=0.
            for p in range(4):
                if count[c,p]>0 and owners[c,p]==j:v+=peaklo[p]
            native_peak_lower=max(native_peak_lower,v)
        sl=0.;su=0.;s600=0.;s1200=0.
        for t in range(states.shape[0]):
            vl=0.;vu=0.
            for q in range(4):
                n=count[c,q]
                if n==0:continue
                j=owners[c,q];eta=states[t,j]
                if np.isfinite(eta):
                    v=volume(z[c,q],pre[c,q],n,eta);vl+=v;vu+=v
                else:
                    cap=adaptive_min[j]+.01
                    if np.isfinite(adaptive_peak[j]):cap=min(cap,adaptive_peak[j])
                    vu+=volume(z[c,q],pre[c,q],n,cap)
            sl=max(sl,vl);su=max(su,vu)
            if t%2==0:s600=max(s600,vl)
            if t%4==0:s1200=max(s1200,vl)
        out[0,c]=rlo/area;out[1,c]=rup/area
        out[2,c]=max(native_peak_lower,sl)/area;out[3,c]=up/area
        out[4,c]=sl/area;out[5,c]=su/area
        out[6,c]=s600/area;out[7,c]=s1200/area
    return out,nowner

def weighted_quantile(v,w,q):
    if len(v)==0 or w.sum()==0:return None
    order=np.argsort(v);v=v[order];w=w[order]
    return float(v[min(np.searchsorted(np.cumsum(w),q*w.sum()),len(v)-1)])

def raster(path, bands, descriptions, tf, dtype='float32'):
    a=np.asarray(bands,dtype=dtype)
    with rasterio.open(path,'w',driver='GTiff',height=NROW,width=NCOL,count=len(a),dtype=dtype,
                       crs='EPSG:6677',transform=tf,nodata=np.nan if dtype=='float32' else None,
                       tiled=True,compress='deflate') as d:
        d.write(a)
        for i,name in enumerate(descriptions,1):d.set_band_description(i,name)
        d.update_tags(target='maximum over time of eligible-ground volume / eligible-ground area',
                      horizontal_crs='JGD2011 / Japan Plane Rectangular CS IX; EPSG:6677',
                      vertical_reference='Source-declared T.P.; epoch alignment uncertified',
                      interpretation='conditional simulation labels, not observed inundation')

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,default=ROOT)
    ap.add_argument('--rows-per-tile',type=int,default=32);args=ap.parse_args()
    root=args.root.resolve();start=time.monotonic()
    p=root/'data/processed/three_ward_common_depth_labels'
    out=root/'data/exp/data-preprocessing/three_ward_common_depth_labels'
    p.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
    base=root/'data/processed/three_ward_sfincs_framework'
    old=root/'data/processed/three_ward_sfincs_scenarios';con=root/'data/processed/three_ward_urban_constraints'
    geo=root/'data/processed/three_ward_adaptive_connectivity'
    ana_before=sha(root/'docs/AnaSOP.md')
    selected=[];fingerprints={};executions={};regpeak=[];apeak=[];states=[];debtreg=[];debtadapt=[]
    def remember(path):
        s=path.stat();fingerprints[str(path.relative_to(root))]=[s.st_size,s.st_mtime_ns]
    def evidence(path):
        selected.append({'source':str(path.relative_to(root)),'sha256':sha(path)})
    with np.load(geo/'adaptive_cell_geometry.npz') as d:g={k:d[k] for k in d.files}
    owner=np.full((NROW*2,NCOL*2),-1,np.int32)
    for size in [5,10]:
        sel=np.flatnonzero(g['size_m']==size);r=g['row_fine'][sel]//10;c=g['col_fine'][sel]//10
        for dr in range(size//5):
            for dc in range(size//5):
                assert np.all(owner[r+dr,c+dc]==-1)
                owner[r+dr,c+dc]=sel
    owner4=owner.reshape(NROW,2,NCOL,2).transpose(0,2,1,3).reshape(NROW,NCOL,4)
    for case in CASES:
        for grid in ['corrected_10m','adaptive_10_5m']:
            folder=base/('adaptive_connectivity_comparison' if case=='rain60_low' else 'adaptive_sensitivity')/(case+'_'+grid)
            ex=json.loads((folder/'execution.json').read_text());assert ex['returncode']==0 and ex['completion_marker']
            evidence(folder/'execution.json');evidence(folder/'scenario_contract.json');remember(folder/'sfincs_map.nc')
            executions[folder.name]=ex
            with netCDF4.Dataset(folder/'sfincs_map.nc') as d:
                times=readarr(d['time'][:]);assert np.array_equal(times,np.arange(0,14401,300));assert float(d['status'][0])==0
                m=readarr(d['msk'][:]).astype('uint8')
                if grid=='corrected_10m':
                    rp=np.flipud(readarr(d['zsmax'][0]));regpeak.append(rp)
                    if len(regpeak)==1:
                        regmin=np.flipud(readarr(d['zb'][:]));regmask=np.flipud(m)
                        # Explicit orientation and physical-coordinate check.
                        assert d['x'].shape==(NROW,NCOL) and d['y'].shape==(NROW,NCOL)
                        assert np.allclose(readarr(d['x'][:]),(-11360+(np.arange(NCOL)+.5)*10)[None,:])
                        assert np.allclose(np.flipud(readarr(d['y'][:])),(-32320-(np.arange(NROW)+.5)*10)[:,None])
                    else:assert np.array_equal(regmask,np.flipud(m))
                else:
                    assert np.array_equal(m,g['mask'])
                    assert np.max(abs(readarr(d['mesh2d_face_x'][:])-g['x']))<.001
                    assert np.max(abs(readarr(d['mesh2d_face_y'][:])-g['y']))<.001
                    apeak.append(readarr(d['zsmax'][0]));states.append(readarr(d['zs'][:]))
                    if len(apeak)==1:amin=readarr(d['zb'][:])
            cache=root/'data/processed/three_ward_adaptive_sensitivity'/(case+'_'+grid+'.npz')
            remember(cache)
            with np.load(cache) as d:
                debt=d['debt5'].reshape(NROW,2,NCOL,2).any(axis=(1,3))
                (debtreg if grid=='corrected_10m' else debtadapt).append(debt)
        print('LOADED',case,flush=True)
    regpeak=np.asarray(regpeak);apeak=np.asarray(apeak);states=np.asarray(states)
    labels=np.full((4,8,NROW,NCOL),np.nan,np.float32)
    area=np.zeros((NROW,NCOL),np.float64);roads=np.zeros_like(area);unknown=np.zeros_like(area)
    nowners=np.zeros((NROW,NCOL),np.uint8)
    fine_sources=[old/'terrain_sampling_0_50m.tif',con/'constraints_0_50m.tif',con/'hydraulic_obstacle_0_50m.tif',old/'active_mask_20m.tif']
    for f in fine_sources:evidence(f);remember(f)
    evidence(geo/'adaptive_cell_geometry.npz')
    with rasterio.open(old/'active_mask_20m.tif') as d:
        mask20=d.read(1);tf=d.transform*Affine.scale(.5,.5)
    interior=np.repeat(np.repeat(mask20==1,2,0),2,1)
    assert interior.shape==(NROW,NCOL)
    with ExitStack() as stack:
        dem=stack.enter_context(rasterio.open(fine_sources[0]));cl=stack.enter_context(rasterio.open(fine_sources[1]));hyd=stack.enter_context(rasterio.open(fine_sources[2]))
        assert dem.shape==(NROW*20,NCOL*20) and dem.transform==cl.transform==hyd.transform
        assert dem.transform==tf*Affine.scale(.05,.05)
        for r0 in range(0,NROW,args.rows_per_tile):
            n=min(args.rows_per_tile,NROW-r0);win=Window(0,r0*20,NCOL*20,n*20)
            z=dem.read(1,window=win);cls=cl.read(1,window=win)
            eligible=np.repeat(np.repeat(interior[r0:r0+n],20,0),20,1)&(cls!=1)&np.isfinite(z)
            assert np.array_equal(hyd.read(1,window=win)[eligible],z[eligible])
            sorted_z,pre,count,road,unk=ground_tables(z,cls,interior[r0:r0+n])
            own=np.ascontiguousarray(owner4[r0:r0+n].reshape(-1,4))
            assert not np.any((count>0)&(own<0))
            ids=own[count>0];assert np.all(g['mask'][ids]==1)
            assert np.all(regmask[r0:r0+n][count.sum(1).reshape(n,NCOL)>0]==1)
            ar=count.sum(1)*.25;area[r0:r0+n]=ar.reshape(n,NCOL)
            roads[r0:r0+n]=road.reshape(n,NCOL);unknown[r0:r0+n]=unk.reshape(n,NCOL)
            for k,case in enumerate(CASES):
                result,owners_n=bounds(sorted_z,pre,count,own,regpeak[k,r0:r0+n].ravel(),regmin[r0:r0+n].ravel(),apeak[k],amin,states[k])
                labels[k,:,r0:r0+n]=result.reshape(8,n,NCOL)
                nowners[r0:r0+n]=owners_n.reshape(n,NCOL)
            print('GROUND ROWS',r0+n,'/',NROW,'elapsed_s',round(time.monotonic()-start),flush=True)
    valid=area>0
    with np.load(root/'data/processed/three_ward_adaptive_sensitivity/common_ground_support.npz') as d:
        assert np.array_equal(area.reshape(NROW//2,2,NCOL//2,2).sum((1,3)),d['area'])
    assert area.sum()==25936197.0
    features=json.loads((root/'data/processed/plateau_water/central_three_ward_boundaries.geojson').read_text())['features']
    project=Transformer.from_crs(6668,6677,always_xy=True).transform
    shapes=[transform_geometry(project,shape(f['geometry'])) for f in features]
    wards=rasterize([(s,i+1) for i,s in enumerate(shapes)],out_shape=(NROW,NCOL),transform=tf)
    scopes={'Whole domain':valid,'Three wards':valid&(wards>0),'Chiyoda':valid&(wards==1),'Chuo':valid&(wards==2),'Minato':valid&(wards==3),'Cross-ward contributing area':valid&(wards==0)}
    support={'Ground Area':area,'Road Ground Area':roads,'Unknown Ground Area':unknown,'Eligible Ground Mask':valid,'Native State Count':nowners,'Ward Code':wards,'Affine Transform':np.array(list(tf)[:6]),'EPSG':np.array(6677)}
    np.savez_compressed(p/'common_support_10m.npz',**support)
    raster(p/'common_support_10m.tif',[np.where(valid,area,np.nan),np.where(valid,roads,np.nan)],['Ground Area (m2)','Road Ground Area (m2)'],tf)
    depth_rows=[];threshold_rows=[];reg_rows=[];products=[];checks={};alias_max=0.
    for k,case in enumerate(CASES):
        rl,ru,lo,up,sl,su,s600,s1200=labels[k].astype('float64')
        violation=max(float(np.nanmax(lo-up)),float(np.nanmax(sl-su)),float(np.nanmax(su-up)),float(np.nanmax(rl-ru)))
        assert violation<1e-5,(case,violation)
        up=np.maximum(up,lo);up=np.maximum(up,su)
        width=up-lo;regwidth=ru-rl;confirmed=np.maximum(lo-su,0)
        alias_max=max(alias_max,violation)
        # Exact-state labels use strict numerical equality; intervals are retained, never midpoint-filled.
        re=np.where(valid&(regwidth<=1e-7),rl,np.nan)
        ae=np.where(valid&(width<=1e-7),lo,np.nan)
        for grid,arrays in [
            ('corrected_10m',{'Peak Mean Depth':re,'Peak Mean Depth Lower':rl,'Peak Mean Depth Upper':ru,'Peak Interval Width':regwidth,'Sampled Negative Volume Flag':debtreg[k],'Censored Peak Flag':valid&~np.isfinite(regpeak[k])}),
            ('adaptive_10_5m',{'Peak Mean Depth':ae,'Peak Mean Depth Lower':lo,'Peak Mean Depth Upper':up,'Peak Interval Width':width,'Sample Peak Lower 300s':sl,'Sample Peak Upper 300s':su,'Sample Peak Lower 600s':s600,'Sample Peak Lower 1200s':s1200,'Confirmed Temporal Miss Lower':confirmed,'Sampled Negative Volume Flag':debtadapt[k],'Multiple Native States Flag':nowners>1})]:
            target=p/(case+'_'+grid+'.npz');np.savez_compressed(target,**{a:np.asarray(v,dtype='float32') if np.issubdtype(np.asarray(v).dtype,np.floating) else v for a,v in arrays.items()});products.append(target)
            raster(p/(case+'_'+grid+'.tif'),[arrays[a] for a in ['Peak Mean Depth Lower','Peak Mean Depth Upper','Peak Interval Width']],['Peak Mean Depth Lower (m)','Peak Mean Depth Upper (m)','Peak Interval Width (m)'],tf)
        for scope,sel in scopes.items():
            for subset,ss in [('all eligible',sel),('multiple native states',sel&(nowners>1)),('possibly wet >0.10m',sel&(up>.1))]:
                if not ss.any():continue
                w=area[ss];v=width[ss];c=confirmed[ss]
                row=dict(case=case,scope=scope,subset=subset,cell_count=int(ss.sum()),open_area_km2=float(w.sum()/1e6),
                         peak_interval_mean_m=float(np.average(v,weights=w)),peak_interval_p95_m=weighted_quantile(v,w,.95),peak_interval_max_m=float(v.max()),
                         confirmed_temporal_miss_mean_m=float(np.average(c,weights=w)),confirmed_temporal_miss_max_m=float(c.max()),
                         sampled_300_minus_600_mean_m=float(np.average((sl-s600)[ss],weights=w)),sampled_600_minus_1200_mean_m=float(np.average((s600-s1200)[ss],weights=w)))
                for tol in [.01,.02,.05]:
                    row[f'area_interval_gt_{tol:g}m_km2']=float(w[v>tol].sum()/1e6)
                    row[f'area_confirmed_temporal_miss_gt_{tol:g}m_km2']=float(w[c>tol].sum()/1e6)
                depth_rows.append(row)
            w=area[sel]
            reg_rows.append(dict(case=case,scope=scope,open_area_km2=float(w.sum()/1e6),
                                 censored_area_km2=float(w[regwidth[sel]>1e-7].sum()/1e6),max_interval_m=float(regwidth[sel].max()),
                                 peak_depth_mean_lower_m=float(np.average(rl[sel],weights=w)),peak_depth_mean_upper_m=float(np.average(ru[sel],weights=w)),
                                 sampled_negative_volume_flag_area_km2=float(area[sel&debtreg[k]].sum()/1e6)))
            for h in [.05,.1,.3,.5]:
                adef=float(area[sel&(lo>h)].sum()/1e6);apos=float(area[sel&(up>h)].sum()/1e6)
                rdef=float(area[sel&(rl>h)].sum()/1e6);rpos=float(area[sel&(ru>h)].sum()/1e6)
                threshold_rows.append(dict(case=case,scope=scope,threshold_m=h,
                    adaptive_definitely_wet_area_km2=adef,adaptive_possibly_wet_area_km2=apos,adaptive_ambiguous_area_km2=apos-adef,
                    regular_definitely_wet_area_km2=rdef,regular_possibly_wet_area_km2=rpos,
                    regular_adaptive_definite_class_disagreement_km2=float(area[sel&(((rl>h)&(up<=h))|((lo>h)&(ru<=h)))].sum()/1e6),
                    interpretation='Area of common10m cells weighted by eligible ground, thresholding cell-mean peak; not fine-ground local-peak footprint'))
        checks[case]={'bound_violation_before_roundoff_repair_m':violation,'regular_interval_max_m':float(np.nanmax(regwidth)),
                      'all_depths_nonnegative':bool(np.all(rl[valid]>=0) and np.all(lo[valid]>=0)),
                      'nested_sample_lower_monotone':bool(np.all(sl[valid]+1e-7>=s600[valid]) and np.all(s600[valid]+1e-7>=s1200[valid]))}
    csvwrite(out/'adaptive_peak_bounds_summary.csv',depth_rows);csvwrite(out/'threshold_area_bounds.csv',threshold_rows);csvwrite(out/'regular_label_summary.csv',reg_rows)
    source_unchanged=all([s==[ (root/f).stat().st_size,(root/f).stat().st_mtime_ns] for f,s in fingerprints.items()])
    assert source_unchanged and sha(root/'docs/AnaSOP.md')==ana_before
    evidence(root/'src/preprocessing/reconstruct_three_ward_common_depth.py')
    contract=dict(version='2026-10-15-r1',created_utc=datetime.now(timezone.utc).isoformat(),cases=CASES,
        source_runs=8,independent_event_count_claimed=0,training_executed=False,new_scenarios_executed=False,
        analysis_grid_m=10,shape=[NROW,NCOL],epsg=6677,transform=list(tf)[:6],
        vertical_reference='Source-declared T.P.; height-epoch alignment uncertified',
        target='max_t sum_j a_j max(eta_native(j,t)-z_j,0) / eligible_ground_area',
        exact_ground_integration='Original0.50m non-building finite ground; boundary rows excluded; no fitted hydraulic storage approximation used in target integration.',
        bound_math={'lower':'max(max_synchronized_written_t sum_j V_j(t), max_native_cell_i sum_{j in i} V_j(native_peak_i)) / A',
                    'upper':'sum_native_cell_i sum_{j in i} V_j(native_peak_i_upper) / A',
                    'censoring':'For known active shallow-suppressed states/peaks, volume lower0; upper from native_zmin+0.01m, capped by known native peak for instantaneous states. Unknown terrain excluded, not filled.',
                    'confirmed_temporal_miss':'max(peak_lower - maximum_written_time_upper,0); nonzero is proof of missed common-cell peak, not merely a loose possible upper bound.',
                    'limitation':'Interval width combines asynchronous-peak uncertainty and shallow censoring; it is not a measured sampling error. Denser states can tighten the interval but may not eliminate an asynchronous upper bound.',
                    'roundoff_m':1e-5},
        point_labels='Peak Mean Depth is NaN where interval width>1e-7; lower/upper remain available. No midpoint target or automatic dry imputation.',
        map_frames=49,map_interval_s=300,event_horizon_s=14400,
        open_ground_area_m2=float(area.sum()),multiple_native_state_area_m2=float(area[nowners>1].sum()),unknown_ground_area_m2=float(unknown.sum()),
        source_fingerprints_unchanged=source_unchanged,source_fingerprints=fingerprints,
        source_evidence=selected,ana_sop_sha256_unchanged=ana_before,
        ward_assignment='Common10m cell-center assignment to administrative polygons; boundary cells approximate, whole-domain support exact.',
        quality='Conditional hydraulic outputs; negative-bookkeeping-volume flags retained from300s audits, not automatically negative physical depth. No real-event validation or certified grid convergence.',
        output_format='NPZ with readable English keys and multiband GeoTIFF. Generic skill tabular parquet/Excel dependencies are inapplicable to these raster labels; no packages installed.',
        variable_status='Constructed reference labels and diagnostic bounds for current controls; training admission remains false, other candidate variables unchanged.',
        processing_runtime_seconds=time.monotonic()-start)
    dump(p/'label_contract.json',contract);dump(out/'decisions.json',contract)
    dump(out/'validation.json',dict(valid=True,checks=checks,open_area_matches_previous_exactly=True,source_fingerprints_unchanged=source_unchanged,
                                  ana_sop_unchanged=True,point_labels_not_imputed=True,all_source_runs_complete=True,
                                  scope='Preprocessing arithmetic and source integrity; does not certify reference physics or approve training.'))
    variables=[
        ('zsmax + Ground Elevation','Peak Mean Depth','Reconstructed primary reference; exact only where interval degenerates'),
        ('censored peak/state intervals + native mapping','Peak Mean Depth Lower','Certified lower bound conditional on native reference states'),
        ('native peak/state intervals + native mapping','Peak Mean Depth Upper','Certified upper bound conditional on native reference states'),
        ('upper minus lower','Peak Interval Width','Reference aggregation ambiguity; not statistical confidence'),
        ('zs at synchronized300s','Sample Peak Lower 300s','Sampled lower bound after fine-ground integration'),
        ('peak lower minus sampled upper','Confirmed Temporal Miss Lower','Minimum proven missed peak at existing output cadence'),
        ('original eligibility mask','Ground Area','Conservative area in square metres'),
        ('multiple native ownership','Native State Count','Distinct active native hydraulic states contributing to analysis cell')]
    csvwrite(out/'variable_list.csv',[dict(original_name=a,readable_name=b,role=c,is_final_variable='no',selection_basis='User-authorized reference-label reconstruction; not a frozen training-variable set') for a,b,c in variables])
    dump(out/'summary.json',dict(cases=CASES,domain_rows=[q for q in depth_rows if q['scope']=='Whole domain'],
        ward_rows=[q for q in depth_rows if q['scope'] in ['Chiyoda','Chuo','Minato'] and q['subset']=='possibly wet >0.10m'],
        domain_thresholds=[q for q in threshold_rows if q['scope']=='Whole domain' and q['threshold_m']==.1],
        regular_domain=[q for q in reg_rows if q['scope']=='Whole domain'],
        open_ground_km2=area.sum()/1e6,multiple_native_state_area_km2=area[nowners>1].sum()/1e6,
        processing_seconds=time.monotonic()-start))
    for q in depth_rows:
        if q['scope']=='Whole domain':print('SUMMARY',json.dumps(q),flush=True)
    print('COMPLETE',out,'elapsed_s',round(time.monotonic()-start),flush=True)

if __name__=='__main__':main()
