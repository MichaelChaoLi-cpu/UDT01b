"""Conservative radar-source to hydraulic-recipient area mapping.

No new solver runs. The NC interval cube is an intermediate forcing contract,
NOT a drop-in replacement for the current linear AMPR reader.
"""
from pathlib import Path
import os
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']: os.environ.pop(k,None)
import hashlib, json
import numpy as np
import rasterio
from pyproj import Transformer
from scipy.ndimage import label, distance_transform_edt
from scipy.sparse import coo_matrix, save_npz, load_npz
from netCDF4 import Dataset

ROOT=Path(__file__).resolve().parents[2]
TAG='tokyo_historical_radar_20240821_v1'
OUT=ROOT/'data/processed'/TAG
P=ROOT/'data/processed/three_ward_urban_constraints'
OLD=ROOT/'data/processed/three_ward_sfincs_scenarios'
FIXED=ROOT/'data/processed/three_ward_fixed_runoff'
R0,R1,C0,C1=1452,1500,1716,1756
NS=(R1-R0)*(C1-C0)


def sha_file(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(8*1024*1024):h.update(b)
    return h.hexdigest()


def aggregate(row,col,rows):
    keys=row.astype('i8')*NS+col
    k,n=np.unique(keys,return_counts=True)
    rows.append((k,n.astype('f8')*.25))


def make_matrix(parts,ncell):
    keys=np.concatenate([k for k,v in parts]);values=np.concatenate([v for k,v in parts])
    a=coo_matrix((values,(keys//NS,keys%NS)),shape=(ncell,NS)).tocsr();a.sum_duplicates();return a


def build():
    paths=[P/'constraints_0_50m.tif',OLD/'active_mask_20m.tif',FIXED/'contributing_area_10m.tif',FIXED/'roof_contributing_area_10m.tif',FIXED/'contract.json']
    hashes={str(p.relative_to(ROOT)):sha_file(p) for p in paths}
    cache=OUT/'source_recipient_map_contract.json'
    if cache.exists():
        c=json.loads(cache.read_text());assert c['source_sha256']==hashes
        for k,v in c['matrix_sha256'].items():assert sha_file(OUT/k)==v
        return c
    old_contract=json.loads(paths[-1].read_text())
    for rel,h in old_contract['source_sha256'].items():assert hashes[rel]==h
    with rasterio.open(paths[0]) as d:
        cl=d.read(1);tf=d.transform;assert d.crs.to_epsg()==6677
    with rasterio.open(paths[1]) as d:m=d.read(1)
    inside=np.repeat(np.repeat(m==1,40,0),40,1)
    opened=inside&(cl!=1)
    print('Reconstruct frozen road-connected roof recipients',cl.shape,flush=True)
    labels=np.empty(cl.shape,'i4');n=label(opened,output=labels)
    road_ids=np.unique(labels[(cl==2)&opened]);road_ids=road_ids[road_ids>0]
    allowed=np.zeros(n+1,bool);allowed[road_ids]=True
    eligible=allowed[labels]&opened
    del labels,allowed,opened
    ny,nx=m.shape[0]*2,m.shape[1]*2;nr,nc=cl.shape
    transform=Transformer.from_crs(6677,6668,always_xy=True)
    terrain=[];urban=[];roof=[];max_dist=0.;block=1024;halo=512;source_count=np.zeros(NS,'i8')
    for r0 in range(0,nr,block):
        for c0 in range(0,nc,block):
            r1=min(nr,r0+block);c1=min(nc,c0+block)
            dr,dc=np.where(inside[r0:r1,c0:c1])
            if dr.size==0:continue
            sr=dr+r0;sc=dc+c0
            lon,lat=transform.transform(tf.c+(sc+.5)*.5,tf.f-(sr+.5)*.5)
            rrad=np.floor((48-lat)*120).astype('i8')-R0
            crad=np.floor((lon-118)*80).astype('i8')-C0
            assert np.all((rrad>=0)&(rrad<R1-R0)&(crad>=0)&(crad<C1-C0))
            source=rrad*(C1-C0)+crad
            source_count+=np.bincount(source,minlength=NS)
            target=(sr//20)*nx+sc//20
            aggregate(target,source,terrain)
            donor=cl[sr,sc]==1
            aggregate(target[~donor],source[~donor],urban)
            if donor.any():
                rs=max(0,r0-halo);re=min(nr,r1+halo);cs=max(0,c0-halo);ce=min(nc,c1+halo)
                e=eligible[rs:re,cs:ce];assert e.any()
                dist,idx=distance_transform_edt(~e,return_indices=True)
                rr=sr[donor]-rs;cc=sc[donor]-cs;ds=dist[rr,cc]
                assert np.all(ds<halo),'Unresolved roof recipient; no truncation allowed'
                tr=idx[0,rr,cc]+rs;tc=idx[1,rr,cc]+cs
                assert np.all(eligible[tr,tc])
                recipient=(tr//20)*nx+tc//20
                aggregate(recipient,source[donor],roof)
                max_dist=max(max_dist,float(ds.max()*.5))
                del dist,idx
        print('fine rows',r0,'/',nr,flush=True)
    del cl,inside,eligible
    t=make_matrix(terrain,ny*nx);del terrain
    u=make_matrix(urban,ny*nx);del urban
    r=make_matrix(roof,ny*nx);del roof
    u=u+r
    with rasterio.open(FIXED/'contributing_area_10m.tif') as f:expected=f.read(1).ravel()
    with rasterio.open(FIXED/'roof_contributing_area_10m.tif') as f:expected_roof=f.read(1).ravel()
    assert np.array_equal(np.asarray(u.sum(1)).ravel(),expected),'Frozen roof recipient allocation changed'
    assert np.array_equal(np.asarray(r.sum(1)).ravel(),expected_roof)
    expected_t=np.repeat(np.repeat(m==1,2,0),2,1).ravel()*100
    assert np.array_equal(np.asarray(t.sum(1)).ravel(),expected_t)
    cols_t=np.asarray(t.sum(0)).ravel();cols_u=np.asarray(u.sum(0)).ravel()
    assert np.array_equal(cols_t,cols_u) and np.array_equal(cols_t,source_count*.25)
    assert cols_t.sum()==old_contract['original_interior_area_m2']
    matrices={'terrain_source_area_matrix.npz':t,'urban_source_area_matrix.npz':u,'roof_source_area_matrix.npz':r}
    for name,a in matrices.items():save_npz(OUT/name,a)
    contract=dict(source_sha256=hashes,matrix_sha256={k:sha_file(OUT/k) for k in matrices},
        shape_recipient_y_x=[ny,nx],source_grid_shape=[R1-R0,C1-C0],source_row0=R0,source_col0=C0,
        entries_unit='m2 (fixed projected 0.50 m source pixel area 0.25 m2)',
        model_crs=6677,source_crs_assignment=6668,
        rainfall_evaluation='Intensity evaluated at each SOURCE radar cell, then transferred using sparse contributing-area matrix; no recipient-sampled roof multiplier.',
        effective_native_rate='matrix_m2 @ representative_mm_h / 100 m2',
        direct_ground_rule='Rain remains in the source hydraulic cell, including isolated courtyards.',
        roof_rule=old_contract['method'],roof_loss_or_delay=0,
        frozen_urban_recipient_marginals_exact=True,frozen_roof_recipient_marginals_exact=True,
        terrain_and_urban_source_area_marginals_exact=True,
        catchment_projected_area_m2=float(cols_t.sum()),roof_projected_area_m2=float(r.sum()),
        roof_nearest_distance_max_m=max_dist,source_cells_with_catchment=int((cols_t>0).sum()),
        nnz={k:int(a.nnz) for k,a in matrices.items()},
        unchanged_population_and_12_4_split=True,no_training_no_sealed_test=True)
    cache.write_text(json.dumps(contract,indent=2,ensure_ascii=False)+'\n');return contract


def export(contract):
    a=np.load(OUT/'radar_intervals.npz')
    rates=a['representative_intensity_mm_h'].reshape(108,NS).astype('f8')
    t=load_npz(OUT/'terrain_source_area_matrix.npz');u=load_npz(OUT/'urban_source_area_matrix.npz')
    areas=np.asarray(t.sum(0)).ravel();used=areas>0
    assert np.isfinite(rates[:,used]).all(),'Missing observation covers catchment; cannot deliver forcing'
    saturated=a['level'].reshape(108,NS)[:,used]==251
    assert not saturated.any(),'Saturated source-cell rainfall requires explicit interval treatment'
    # Missing outside catchment is excluded by zero support, never interpreted as observed zero.
    work=np.zeros_like(rates);work[:,used]=rates[:,used]
    ny,nx=contract['shape_recipient_y_x']
    target=OUT/'forcing_intervals_10m.nc'
    max_abs=0.;max_rel=0.;times=a['interval_end_unix_s'];volume=[]
    with Dataset(target,'w',format='NETCDF4') as nc:
        for k,n in [('time',108),('bnds',2),('y',ny),('x',nx)]:nc.createDimension(k,n)
        v=nc.createVariable('time','f8',('time',));v.units='seconds since 1970-01-01 00:00:00 UTC';v.standard_name='time';v.bounds='time_bounds';v[:]=times-150
        v=nc.createVariable('time_bounds','f8',('time','bnds'));v.units=nc['time'].units;v[:]=np.column_stack([times-300,times])
        for k,data in [('x',-11360+(np.arange(nx)+.5)*10),('y',-32320-(np.arange(ny)+.5)*10)]:
            v=nc.createVariable(k,'f8',(k,));v.units='m';v.standard_name='projection_'+k+'_coordinate';v[:]=data
        crs=nc.createVariable('crs','i4');crs.spatial_ref=rasterio.crs.CRS.from_epsg(6677).to_wkt();crs.epsg_code='EPSG:6677'
        vars={}
        for name in ['terrain_effective_rainfall','urban_effective_rainfall']:
            v=nc.createVariable(name,'f4',('time','y','x'),zlib=True,complevel=4,chunksizes=(1,128,128))
            v.units='mm h-1';v.grid_mapping='crs';v.cell_methods='time: mean (analyst five-minute representative-intensity hold)';vars[name]=v
        nc.title='Conservative historical radar interval forcing for Tokyo three-ward domain'
        nc.warning='INTERMEDIATE CONTRACT ONLY. Current solver supports piecewise linear AMPR and rejects ampr_block. Do not pass this CF interval cube directly to it.'
        nc.native_radar_resolution='Approximately 1 km. 10 m is hydraulic recipient grid, not observed rainfall resolution.'
        nc.temporal_reconstruction='Each radar representative intensity held over preceding 300 seconds; no linear interpolation or invented one-second ramps.'
        nc.roof_rule=contract['roof_rule'];nc.population_split_quality_masks='Unchanged; no hydraulic simulation or ML training.'
        for k in range(108):
            expected=float(np.dot(work[k],areas)/12/1000)
            vols=[]
            for name,mat in [('terrain_effective_rainfall',t),('urban_effective_rainfall',u)]:
                v=(mat@work[k]/100).astype('f4').reshape(ny,nx);vars[name][k]=v
                delivered=float(v.astype('f8').sum()*100/12/1000)
                err=abs(delivered-expected);max_abs=max(max_abs,err)
                if expected>0:max_rel=max(max_rel,err/expected)
                assert err<=max(1e-7,expected*1e-7)
                vols.append(delivered)
            volume.append([int(times[k]),expected,*vols])
    import csv
    with (OUT/'interval_water_volume.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['interval_end_unix_s','source_rain_volume_m3','terrain_nc_volume_m3','urban_nc_volume_m3']);w.writerows(volume)
    volumes=np.asarray(volume)
    uniform=t@np.ones(NS);uniform_u=u@np.ones(NS)
    with rasterio.open(FIXED/'contributing_area_10m.tif') as f:expected_u=f.read(1).ravel()
    assert np.array_equal(uniform_u,expected_u)
    assert float(uniform.sum())==float(uniform_u.sum())
    summary=dict(interval_count=108,temporal_reconstruction='Preceding 300 s representative-intensity hold, NOT directly solver-qualified.',
        original_radar_source_cells_with_catchment=int(used.sum()),catchment_area_m2=float(areas.sum()),
        missing_values_over_catchment=int((~np.isfinite(rates[:,used])).sum()),saturated_values_over_catchment=int(saturated.sum()),
        total_reconstructed_source_rain_volume_m3=float(volumes[:,1].sum()),
        catchment_area_weighted_reconstructed_rain_mm=float(volumes[:,1].sum()/areas.sum()*1000),
        terrain_nc_total_rain_volume_m3=float(volumes[:,2].sum()),urban_nc_total_rain_volume_m3=float(volumes[:,3].sum()),
        max_interval_export_abs_volume_error_m3=max_abs,max_interval_export_relative_volume_error=max_rel,
        uniform_rain_exactly_reproduces_frozen_contributing_weights=True,
        source_recipient_mapping_identity=contract['matrix_sha256'],
        solver_ready=False,solver_blocker='Existing exact step integrator only qualifies ampr_block=0 piecewise linear; a five-minute interval hold needs an explicit, tested solver adapter. Current CF cube is not Galibier corner-coordinate AMPR.',
        historical_river_stage_available=False,solver_runs_started=0,
        protected='No changes to rainfall scenarios, formal 12+4 splits, masks, training, sealed test, population inputs or previous hydraulic outputs.',
        output_file=str(target.relative_to(ROOT)),output_sha256=sha_file(target))
    (OUT/'spatial_forcing_qa.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':export(build())
