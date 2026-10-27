"""Inspect gate profiles and source-supported mixed-level masks; no raw data edits."""
import os
for k in ('PROJ_LIB','PROJ_DATA','GDAL_DATA'):os.environ.pop(k,None)
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-mpl')
from pathlib import Path
import sys,json,hashlib,struct
import numpy as np,rasterio
from rasterio.features import rasterize,shapes
from rasterio.transform import rowcol,xy,from_origin
from scipy.ndimage import label,binary_dilation,binary_closing
from shapely.geometry import Point,Polygon,box,shape,LineString
from shapely import intersects_xy
from shapely.ops import unary_union
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
sys.path.insert(0,str(Path(__file__).resolve().parent))
import trial_hibiya_connectivity as trial

ROOT=trial.ROOT;BASE=trial.BASE;PREV=trial.OUT
OUT=ROOT/'data/exp/data-briefing/hibiya_bottleneck_review'
PROC=ROOT/'data/processed/hibiya_bottleneck_review'
FONT=FontProperties(fname='/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
V=np.array([-.8827467590792446,.4698490814454044])
SITES=[('U1',[-6937.25,-36378.75],V,25),('T1',[-6942.75,-36390.75],V,25),('A10',[-6840.75,-36214.75],V,25),
       ('A2',[-6820.75,-36150.25],np.array([-.0305979117,-.9995317743]),25),('A3',[-6676.75,-36008.75],np.array([-.4002882367,-.9163892882]),25),
       ('P1',[-6911.25,-36328.75],V,16)]

def extract_new_sites():
    path=ROOT/'data/raw/tokyo_lidar_dem/2024/pilot_hibiya/09LD2812.las'
    with path.open('rb') as f:h=f.read(227)
    fmt=h[104];size=struct.unpack_from('<H',h,105)[0];count=struct.unpack_from('<I',h,107)[0];offset=struct.unpack_from('<I',h,96)[0]
    assert fmt==3 and size==34
    scales=np.array(struct.unpack_from('<3d',h,131));origin=np.array(struct.unpack_from('<3d',h,155))
    dtype=np.dtype({'names':['x','y','z','raw_class'],'formats':['<i4','<i4','<i4','u1'],'offsets':[0,4,8,15],'itemsize':34})
    points=np.memmap(path,dtype=dtype,mode='r',offset=offset,shape=(count,))
    parts=[]
    for begin in range(0,count,1000000):
        p=points[begin:begin+1000000];x=p['x']*scales[0]+origin[0];y=p['y']*scales[1]+origin[1]
        m=(x>-6990)&(x<-6885)&(y>-36440)&(y<-36320)
        parts.append(np.c_[x[m],y[m],p['z'][m]*scales[2]+origin[2],p['raw_class'][m]])
    a=np.vstack(parts);np.savez_compressed(OUT/'U1_T1_all_class_points.npz',xyz=a[:,:3],raw_class=a[:,3].astype(np.uint8))
    return a[:,:3],a[:,3].astype(np.uint8)

def draw_profile(sid,center,unit,half,xyz,raw,dem,tf,ortho,waters,paths):
    unit=unit/np.linalg.norm(unit);perp=np.array([-unit[1],unit[0]]);rel=xyz[:,:2]-center
    along=rel@unit;cross=rel@perp;classes=raw&31;withheld=(raw&128)>0
    strip=(abs(cross)<=.5)&(abs(along)<=half)&~withheld
    rows=[];coords=[]
    for d in np.arange(-half+.25,half,.5):
        q=center+d*unit;r,c=rowcol(tf,*q);coords.append(q)
        m=strip&(abs(along-d)<=.25);ground=m&(classes==2);other=m&(classes!=2)
        vals=xyz[ground,2];ovals=xyz[other,2]
        iswater=any(g.covers(Point(*q)) for g in waters)
        dz=float(dem[r,c]) if dem[r,c]!=-32767 and not iswater else None
        rows.append({'offset_m':float(d),'easting_m':float(q[0]),'northing_m':float(q[1]),'dem_source_z_m':dz,
            'class2_n':int(ground.sum()),'class2_median_m':float(np.median(vals)) if len(vals) else None,
            'class2_p10_m':float(np.percentile(vals,10)) if len(vals) else None,'class2_p90_m':float(np.percentile(vals,90)) if len(vals) else None,
            'non_ground_n':int(other.sum()),'non_ground_p10_m':float(np.percentile(ovals,10)) if len(ovals) else None,
            'inside_mapped_water':iswater})
    trial.dump(OUT/(sid+'_profile.json'),rows)
    fig=plt.figure(figsize=(13,5.6));ax0=fig.add_axes([.06,.14,.34,.72]);ax1=fig.add_axes([.48,.14,.49,.72])
    radius=32;b=(center[0]-radius,center[1]-radius,center[0]+radius,center[1]+radius)
    window=rasterio.windows.from_bounds(*b,ortho.transform);rgb=ortho.read(window=window)
    ax0.imshow(np.moveaxis(rgb,0,-1),extent=(b[0],b[2],b[1],b[3]));ax0.plot(*np.array(coords).T,color='#ba2828',lw=1);ax0.plot(*center,'r+')
    for g in waters:
        if g.intersects(box(*b)):
            for part in (list(g.geoms) if hasattr(g,'geoms') else [g]):ax0.plot(*np.array(part.exterior.coords).T,color='#00aaca',lw=1)
    for p in paths:
        if p['route_id'].startswith('R2') and p['route_id'].endswith('reviewed_section'):
            ax0.plot(p['easting_m'],p['northing_m'],color='#c59b20',lw=.8,alpha=.8)
    ax0.set_xlim(b[0],b[2]);ax0.set_ylim(b[1],b[3]);ax0.ticklabel_format(style='plain',useOffset=False);ax0.set_aspect('equal')
    ax0.set_title('源正射；黄线为上一轮候选路径',fontproperties=FONT,fontsize=11)
    water_returns=np.zeros(len(xyz),bool)
    for g in waters:
        if g.intersects(box(*b)):water_returns|=intersects_xy(g,xyz[:,0],xyz[:,1])
    for code,color,lab in [(1,'#969696','Class1 未分类'),(3,'#538441','Class3 低植被'),(2,'#171717','Class2 水域外地面分类')]:
        m=strip&(classes==code)
        if code==2:m&=~water_returns
        ax1.scatter(along[m],xyz[m,2],s=4 if code==2 else 2,color=color,alpha=.65 if code==2 else .25,label=lab)
    m=strip&(classes==2)&water_returns
    if m.any():ax1.scatter(along[m],xyz[m,2],s=4,color='#138ba7',label='图示水域内Class2（地物未定）')
    z=np.array([r['dem_source_z_m'] if r['dem_source_z_m'] is not None else np.nan for r in rows]);ds=np.array([r['offset_m'] for r in rows])
    ax1.plot(ds,z,color='#df8e00',lw=1.5,label='原生0.50m DEM（除图示水域）')
    ax1.axvline(0,color='#ac2424',ls=':');ax1.axhline(2.57,color='#6d6696',ls='--',lw=.8,label='上一轮2.57m模型门槛')
    ax1.set_xlim(-half,half);ax1.set_ylim(-1.5 if sid=='P1' else -.5,7);ax1.grid(alpha=.2)
    ax1.set_xlabel('断面偏移 (m)，正向公园/干侧',fontproperties=FONT);ax1.set_ylabel('源高程 (m)',fontproperties=FONT)
    ax1.legend(prop=FONT,fontsize=8,loc='upper right');ax1.set_title('全部有效类别；非地面回波不能直接当墙/围栏',fontproperties=FONT,fontsize=11)
    fig.suptitle(sid+' | 连续横断面核查：路缘、岸顶或入口仍须辨认',fontproperties=FONT,fontsize=15)
    fig.savefig(OUT/(sid+'_ortho_all_class_section.png'),dpi=155,bbox_inches='tight');plt.close(fig)
    return {'site_id':sid,'easting_m':float(center[0]),'northing_m':float(center[1]),
        'profile_width_m':1.0,'profile_length_m':half*2,'local_class_counts':{str(int(c)):int(np.sum(classes==c)) for c in np.unique(classes)},
        'withheld_count':int(withheld.sum()),'strip_valid_returns':int(strip.sum()),'actual_inlet_confirmed':False}

def point_based_masks(dem,tf):
    a=np.load(ROOT/'data/exp/data-briefing/hibiya_lowpoint_verification/local_points.npz');ground=(a['classification']==2)&((a['raw_class']&128)==0)
    local=box(-6953.25,-36348.25,-6883.25,-36300.25)
    smalltf=from_origin(-6954,-36300,1,1);shape1=(49,72)
    r,c=rowcol(smalltf,a['x'][ground],a['y'][ground]);r=np.array(r);c=np.array(c);z=a['z'][ground]
    keep=(r>=0)&(c>=0)&(r<shape1[0])&(c<shape1[1]);r,c,z=r[keep],c[keep],z[keep]
    low=np.zeros(shape1,np.uint32);high=low.copy();np.add.at(low,(r[z<2],c[z<2]),1);np.add.at(high,(r[z>2.8],c[z>2.8]),1)
    # Separately preserve observed lower returns and direct evidence of mixed levels.
    lower=low>=2;mixed=(low>=2)&(high>=2)
    lab,_=label(binary_closing(lower,iterations=1),np.ones((3,3)))
    seeds=np.unique(lab[(low>0)]);ids=[i for i in seeds if i and np.any((lab==i)&(low>=2))]
    lower=np.isin(lab,ids)
    def vector(mask):
        parts=[shape(g) for g,v in shapes(mask.astype('uint8'),mask=mask,transform=smalltf) if v==1]
        return unary_union(parts).intersection(local)
    lowgeom=vector(lower);mixedgeom=vector(mixed);base=unary_union([lowgeom,mixedgeom])
    masks=[('no_isolation',None),('source_low_return_buffer_0_5m',base.buffer(.5)),('source_low_return_buffer_1m',base.buffer(1)),
           ('source_low_return_buffer_2m',base.buffer(2)),('source_low_return_buffer_3m',base.buffer(3)),('previous_context_rectangle',local)]
    features=[trial.feature(lowgeom,{'mask_id':'lower_class2_return_envelope','role':'provisional source-supported lower-return zone, not a surveyed ramp footprint','rule':'1m cells, at least2 Class2 returns below sourceZ2m, 1cell closing'}),
              trial.feature(mixedgeom,{'mask_id':'direct_mixed_level_cells','role':'direct upper/lower return co-occurrence, not a precise upper-road boundary','rule':'1m cells, >=2 Class2 below2m and >=2 above2.8m'})]
    for name,g in masks:
        if g is not None:features.append(trial.feature(g,{'mask_id':name,'role':'analysis isolation sensitivity, not physical wall or final entrance outline','area_m2':float(g.area)}))
    trial.save_fc(PROC/'parking_layer_isolation_candidates.geojson',features)
    trial.dump(OUT/'parking_layer_evidence.json',{'cell_size_m':1,'lower_return_cut_source_z_m':2.,'upper_return_cut_source_z_m':2.8,
        'low_cells_ge2':int((low>=2).sum()),'mixed_cells_ge2_each_layer':int(mixed.sum()),'lower_envelope_area_m2':lowgeom.area,
        'limitations':'Cuts separate observed low/high source returns at this already verified engineered ramp; not named water datum, final deck reconstruction or proof every return is pavement.'})
    return masks,lowgeom,mixedgeom

def main():
    for d in (OUT,PROC):d.mkdir(parents=True,exist_ok=True)
    xyz,raw=extract_new_sites()
    with rasterio.open(BASE/'official_dem_0_50m_mosaic.tif') as ds:dem=ds.read(1);tf=ds.transform
    allwater=[trial.project(shape(f['geometry'])) for f in json.loads((ROOT/'data/processed/plateau_water/central_three_wards_wtr.geojson').read_text())['features']]
    profiles=json.loads((PREV/'path_profiles.json').read_text());site_rows=[]
    with rasterio.open(BASE/'official_orthophoto_0_20m_mosaic.tif') as ortho:
        for sid,center,u,half in SITES:
            if sid in ('U1','T1'):points,classes=xyz,raw
            elif sid=='P1':
                a=np.load(ROOT/'data/exp/data-briefing/hibiya_lowpoint_verification/local_points.npz');points=np.c_[a['x'],a['y'],a['z']];classes=a['raw_class']
            else:
                a=np.load(ROOT/f'data/exp/data-briefing/hibiya_water_corridor/{sid}_local_points.npz');points=a['xyz'];classes=a['classification'].astype('uint8')
            site_rows.append(draw_profile(sid,np.array(center),u,half,points,classes,dem,tf,ortho,allwater,profiles))
    trial.dump(OUT/'profile_locations.json',site_rows)
    masks,lowgeom,mixedgeom=point_based_masks(dem,tf)
    watermask=rasterize([(g,1) for g in allwater],out_shape=dem.shape,transform=tf,all_touched=True,dtype='uint8').astype(bool)
    buildings=[trial.project(shape(f['geometry'])).buffer(.5) for f in json.loads((BASE/'local_building_exclusion.geojson').read_text())['features']]
    bmask=rasterize([(g,1) for g in buildings],out_shape=dem.shape,transform=tf,all_touched=True,dtype='uint8').astype(bool)
    native=(dem!=-32767)&np.isfinite(dem)&~watermask&~bmask
    source=json.loads((PREV/'source_reference_points.json').read_text())[1];rr,cc=rowcol(tf,source['easting_m'],source['northing_m']);seed=rr*dem.shape[1]+cc
    targets={}
    for f in json.loads((trial.PROC/'target_bands_and_reviewed_sections.geojson').read_text())['features']:
        tid=f['properties']['target_id']
        if tid.endswith('reviewed_section'):
            p=trial.project(shape(f['geometry']));r,c=rowcol(tf,p.x,p.y);m=np.zeros(dem.shape,bool);m[r,c]=True;targets[tid]=m
    results=[];routes=[]
    for name,g in masks:
        m=trial.target_mask(g,tf,dem.shape) if g is not None else np.zeros(dem.shape,bool);allowed=native&~m
        rows,_=trial.thresholds(dem,allowed,{'R2':seed},targets,1,name)
        for row in rows:
            row['isolation_area_m2']=float(g.area) if g is not None else 0
            row['actual_inlet_confirmed']=False;results.append(row)
            ids=trial.shortest_path(allowed&(dem<=row['threshold_source_z_m']),seed,targets[row['target_id']])
            r,c=np.divmod(ids,dem.shape[1]);x,y=xy(tf,r,c);coords=np.c_[x,y]
            assert np.all(allowed.ravel()[ids])
            row['length_m']=float((len(ids)-1)*.5)
            routes.append(trial.feature(LineString(coords),dict(row)))
    trial.dump(OUT/'mask_sensitivity_thresholds.json',results);trial.save_fc(PROC/'mask_sensitivity_routes.geojson',routes)
    trial.save_fc(PROC/'review_cross_sections.geojson',[trial.feature(LineString([np.array(q)-half*u,np.array(q)+half*u]),{'site_id':sid,'role':'source profile strip, not measured bank/kerb breakline'}) for sid,q,u,half in SITES])
    trial.dump(OUT/'provenance.json',{'source_epoch_first':True,'no_original_elevations_modified':True,
        'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [BASE/'official_dem_0_50m_mosaic.tif',ROOT/'data/exp/data-briefing/hibiya_lowpoint_verification/local_points.npz']},
        'U1_T1_source_LAS':'data/raw/tokyo_lidar_dem/2024/pilot_hibiya/09LD2812.las','A10_A2_A3':'cached all-class source-point excerpts; withheld points excluded in original cache',
        'interpretation_status':'Profiles/masks are evidence screening. No actual gate, crest or inlet has been surveyed or assigned a validated hydraulic capacity.'})
    draw_mask_map(masks,lowgeom,mixedgeom,routes)
    print(json.dumps({'profiles':site_rows,'mask_results':results},ensure_ascii=False,indent=2),flush=True)

def draw_mask_map(masks,lowgeom,mixedgeom,routes):
    with rasterio.open(BASE/'official_orthophoto_0_20m_mosaic.tif') as ds:
        b=(-6965,-36365,-6875,-36290);w=rasterio.windows.from_bounds(*b,ds.transform);rgb=ds.read(window=w)
    fig,axs=plt.subplots(1,2,figsize=(12,6),layout='constrained')
    for ax in axs:
        ax.imshow(np.moveaxis(rgb,0,-1),extent=(b[0],b[2],b[1],b[3]));ax.set_xlim(b[0],b[2]);ax.set_ylim(b[1],b[3]);ax.ticklabel_format(style='plain',useOffset=False)
    for name,g in masks:
        if name in ('source_low_return_buffer_1m','previous_context_rectangle'):
            for q in (list(g.geoms) if hasattr(g,'geoms') else [g]):axs[0].plot(*np.array(q.exterior.coords).T,color='#ad2b7f' if name=='previous_context_rectangle' else '#149cb4',lw=1,label='原复核矩形' if name=='previous_context_rectangle' else '下层回波候选+1m')
    for q in (list(mixedgeom.geoms) if hasattr(mixedgeom,'geoms') else [mixedgeom]):axs[0].fill(*np.array(q.exterior.coords).T,color='#da8e32',alpha=.5)
    axs[0].set_title('下层回波带与直接上下层混合像元',fontproperties=FONT,fontsize=12)
    mode_labels={'no_isolation':'未隔离','source_low_return_buffer_1m':'回波候选区+1m','previous_context_rectangle':'原复核矩形'}
    for f in routes:
        r=f['properties']
        if r['target_id']=='心字池|reviewed_section' and r['mode'] in ('no_isolation','source_low_return_buffer_1m','previous_context_rectangle'):
            g=trial.project(shape(f['geometry']));col={'no_isolation':'#b13822','source_low_return_buffer_1m':'#0078ab','previous_context_rectangle':'#ae2b8a'}[r['mode']]
            axs[1].plot(*np.array(g.coords).T,color=col,lw=1.2,label=mode_labels[r['mode']]+f" H={r['threshold_source_z_m']:.2f}m")
    axs[1].set_title('到心字池的路线对隔离范围的敏感性',fontproperties=FONT,fontsize=12)
    for ax in axs:ax.legend(prop=FONT,fontsize=8,loc='lower left');ax.set_xlabel('Easting (m; EPSG:6677)');ax.set_ylabel('Northing (m)')
    from matplotlib.patches import Patch
    handles,labels=axs[0].get_legend_handles_labels();handles.append(Patch(facecolor='#da8e32',alpha=.5));labels.append('同一1m格内上下层Class2并存')
    axs[0].legend(handles,labels,prop=FONT,fontsize=8,loc='lower left')
    fig.suptitle('候选隔离范围有源回波依据，但仍不是实测坡道/上层道路轮廓',fontproperties=FONT,fontsize=14)
    fig.savefig(OUT/'parking_layer_masks_and_routes.png',dpi=160,bbox_inches='tight');plt.close(fig)

def render_saved():
    a=np.load(OUT/'U1_T1_all_class_points.npz');xyz=a['xyz'];raw=a['raw_class']
    with rasterio.open(BASE/'official_dem_0_50m_mosaic.tif') as ds:dem=ds.read(1);tf=ds.transform
    waters=[trial.project(shape(f['geometry'])) for f in json.loads((ROOT/'data/processed/plateau_water/central_three_wards_wtr.geojson').read_text())['features']]
    paths=json.loads((PREV/'path_profiles.json').read_text())
    with rasterio.open(BASE/'official_orthophoto_0_20m_mosaic.tif') as ortho:
        for sid,center,u,half in SITES:
            if sid in ('U1','T1'):p,c=xyz,raw
            elif sid=='P1':
                b=np.load(ROOT/'data/exp/data-briefing/hibiya_lowpoint_verification/local_points.npz');p,c=np.c_[b['x'],b['y'],b['z']],b['raw_class']
            else:
                b=np.load(ROOT/f'data/exp/data-briefing/hibiya_water_corridor/{sid}_local_points.npz');p,c=b['xyz'],b['classification'].astype('uint8')
            draw_profile(sid,np.array(center),u,half,p,c,dem,tf,ortho,waters,paths)
    f=json.loads((PROC/'parking_layer_isolation_candidates.geojson').read_text())['features'];geos={x['properties']['mask_id']:trial.project(shape(x['geometry'])) for x in f}
    masks=[('no_isolation',None)]+[(name,g) for name,g in geos.items() if name.startswith('source_') or name=='previous_context_rectangle']
    draw_mask_map(masks,geos['lower_class2_return_envelope'],geos['direct_mixed_level_cells'],json.loads((PROC/'mask_sensitivity_routes.geojson').read_text())['features'])

if __name__=='__main__':render_saved() if '--render-only' in sys.argv else main()
