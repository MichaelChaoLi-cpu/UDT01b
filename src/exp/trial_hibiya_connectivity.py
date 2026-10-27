"""Source-epoch static terrain connectivity, not rainfall/runoff or confirmed discharge."""
import os
for k in ('PROJ_LIB','PROJ_DATA','GDAL_DATA'):os.environ.pop(k,None)
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-mpl')
from pathlib import Path
from hashlib import sha256
from collections import Counter
import json,heapq,math
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import rowcol,xy
from scipy.ndimage import label,generate_binary_structure,distance_transform_cdt
from pyproj import Transformer
from shapely import transform as st
from shapely.geometry import shape,mapping,Point,LineString,box,Polygon
from shapely.ops import unary_union

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'data/processed/hibiya_water_corridor'
OLD=ROOT/'data/exp/data-briefing/hibiya_water_corridor'
OUT=ROOT/'data/exp/data-briefing/hibiya_connectivity_trial'
PROC=ROOT/'data/processed/hibiya_connectivity_trial'
T=Transformer.from_crs(6668,6677,always_xy=True);LL=Transformer.from_crs(6677,6668,always_xy=True)
NAMES=['心字池','雲形池','日比谷濠']

def project(g):return st(g,T.transform,interleaved=False)
def feature(g,p):return {'type':'Feature','geometry':mapping(st(g,LL.transform,interleaved=False)),'properties':p}
def dump(p,j):p.write_text(json.dumps(j,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def save_fc(p,features):dump(p,{'type':'FeatureCollection','features':features})
def read_raster(name):
    with rasterio.open(BASE/name) as ds:return ds.read(1)

def target_mask(g,tf,dimensions):return rasterize([(g,1)],out_shape=dimensions,transform=tf,dtype='uint8',all_touched=False).astype(bool)

def thresholds(dem,allowed,seeds,targets,connectivity,mode):
    heights=np.unique(dem[allowed]);struct=generate_binary_structure(2,connectivity);target_ids={k:np.flatnonzero(m&allowed) for k,m in targets.items()};cache={}
    def evaluate(i):
        if i in cache:return cache[i]
        components,_=label(allowed&(dem<=heights[i]),struct)
        flat=components.ravel();result={}
        for sid,seed in seeds.items():
            sidlabel=flat[seed]
            result[sid]={name:bool(sidlabel and np.any(flat[ids]==sidlabel)) for name,ids in target_ids.items()}
        cache[i]=result;return result
    result=[]
    for sid,seed in seeds.items():
        for name,ids in target_ids.items():
            if not len(ids) or not evaluate(len(heights)-1)[sid][name]:
                result.append({'source_id':sid,'target_id':name,'mode':mode,'threshold_source_z_m':None});continue
            low=-1;high=len(heights)-1
            while high-low>1:
                middle=(low+high)//2
                if evaluate(middle)[sid][name]:high=middle
                else:low=middle
            h=float(heights[high]);cc,_=label(allowed&(dem<=h),struct);component=cc.ravel()[seed];reach=cc==component
            touches=bool(reach[0,:].any() or reach[-1,:].any() or reach[:,0].any() or reach[:,-1].any())
            result.append({'source_id':sid,'target_id':name,'mode':mode,'threshold_source_z_m':h,
                'rise_above_source_cell_m':max(0.,h-float(dem.ravel()[seed])),
                'connected_area_m2_at_threshold':float(reach.sum()*.25),'component_touches_data_rectangle_boundary':touches,
                'previous_lower_source_height_reaches_target':False if high==0 else evaluate(high-1)[sid][name],
                'target_valid_cells':int(len(ids))})
            print('Threshold',sid,name,mode,round(h,3),flush=True)
    return result,{'distinct_label_evaluations':len(cache),'height_candidates':len(heights)}

def shortest_path(allowed_at_level,seed,target):
    nr,nc=allowed_at_level.shape;target=target&allowed_at_level
    heuristic=distance_transform_cdt(~target,metric='taxicab').ravel()
    valid=allowed_at_level.ravel();goal=target.ravel();dist=np.full(valid.size,np.inf);parents=np.full(valid.size,-2,np.int32)
    dist[seed]=0.;parents[seed]=-1;heap=[(float(heuristic[seed]),0.,int(seed))]
    while heap:
        _,cost,i=heapq.heappop(heap)
        if cost!=dist[i]:continue
        if goal[i]:
            path=[i]
            while parents[path[-1]]>=0:path.append(int(parents[path[-1]]))
            return np.array(path[::-1],dtype=np.int32)
        r,c=divmod(i,nc)
        for j in ((i-nc if r else -1),(i+nc if r<nr-1 else -1),(i-1 if c else -1),(i+1 if c<nc-1 else -1)):
            if j<0 or not valid[j]:continue
            new=cost+1
            if new<dist[j]:dist[j]=new;parents[j]=i;heapq.heappush(heap,(new+float(heuristic[j]),new,j))
    raise ValueError('No path despite connectivity result')

def main():
    for d in (OUT,PROC):d.mkdir(parents=True,exist_ok=True)
    with rasterio.open(BASE/'official_dem_0_50m_mosaic.tif') as ds:dem=ds.read(1);tf=ds.transform;profile=ds.profile;bounds=list(ds.bounds)
    counts=read_raster('ground_class2_count_native_grid.tif');spread=read_raster('ground_class2_range_native_grid.tif')
    means=read_raster('ground_class2_mean_native_grid.tif');supported=(counts>=3)&np.isfinite(spread)&(spread<=.15)
    allwaters=[(project(shape(f['geometry'])),f['properties']) for f in json.loads((ROOT/'data/processed/plateau_water/central_three_wards_wtr.geojson').read_text())['features']]
    waterdict={p['name']:g for g,p in allwaters if p['name'] in NAMES}
    buildings=[project(shape(f['geometry'])).buffer(.5) for f in json.loads((BASE/'local_building_exclusion.geojson').read_text())['features']]
    bmask=target_mask(unary_union(buildings),tf,dem.shape)
    # Match the earlier conservative all-touched exclusions exactly.
    bmask=rasterize([(g,1) for g in buildings],out_shape=dem.shape,transform=tf,all_touched=True,dtype='uint8').astype(bool)
    wmask=rasterize([(g,1) for g,p in allwaters],out_shape=dem.shape,transform=tf,all_touched=True,dtype='uint8').astype(bool)
    native_allowed=(dem!=-32767)&np.isfinite(dem)&~bmask&~wmask
    portal_summary=json.loads((ROOT/'data/exp/data-briefing/hibiya_lowpoint_verification/verification_summary.json').read_text())
    portal_review_area=box(*portal_summary['local_bounds_epsg6677'])
    portal_mask=target_mask(portal_review_area,tf,dem.shape)
    allowed=native_allowed&~portal_mask
    refs={'R1':[-6920.25,-36404.25],'R2':[-6943.835,-36409.17]}
    seeds={};source_rows=[];source_fc=[]
    for sid,q in refs.items():
        r,c=rowcol(tf,*q);xx,yy=xy(tf,r,c);seed=r*dem.shape[1]+c
        assert allowed[r,c] and supported[r,c]
        seeds[sid]=seed
        rec={'source_id':sid,'easting_m':float(xx),'northing_m':float(yy),'dem_z_m':float(dem[r,c]),
             'class2_count':int(counts[r,c]),'class2_mean_z_m':float(means[r,c]),'class2_range_m':float(spread[r,c]),
             'source_role':'earlier traffic-lane reference' if sid=='R1' else 'park-side carriageway, prior section S55/cross+15.5m; grid-center reference',
             'not_event_water_level':True,'vertical_reference':'source elevations; named vertical datum and epoch reconciliation unresolved'}
        source_rows.append(rec);source_fc.append(feature(Point(xx,yy),rec))
    targets={};target_fc=[]
    anchors={'心字池':('A2',[-6820.75,-36150.25]),'雲形池':('A7',[-7069.991166099848,-36227.6171141164]),'日比谷濠':('A3',[-6676.75,-36008.75])}
    for name,g in waterdict.items():
        key=name+'|near_bank';ring=g.buffer(3).difference(g.buffer(.75));targets[key]=target_mask(ring,tf,dem.shape)&allowed
        target_fc.append(feature(ring,{'target_id':key,'role':'approximate outer shore band .75-3m; not crest or inlet'}))
        aid,q=anchors[name];r,c=rowcol(tf,*q);m=np.zeros(dem.shape,bool);m[r,c]=True
        assert allowed[r,c]
        key=name+'|reviewed_section';targets[key]=m
        target_fc.append(feature(Point(*xy(tf,r,c)),{'target_id':key,'review_anchor':aid,'role':'reviewed approach/crest candidate; actual inlet unconfirmed','dem_z_m':float(dem[r,c])}))
    if not (OUT/'unisolated_diagnostic_thresholds.json').exists():
        diagnostic,_=thresholds(dem,native_allowed,seeds,targets,1,'unisolated_diagnostic')
        dump(OUT/'unisolated_diagnostic_thresholds.json',{'role':'Diagnostic only: mixed upper/lower surfaces near known underground portal were not isolated. Not accepted as surface-only route.','main_4_connected':diagnostic})
    base,info4=thresholds(dem,allowed,seeds,targets,1,'native_4_connected')
    loose,info8=thresholds(dem,allowed,seeds,targets,2,'corner_contact_8_connected')
    expanded=rasterize([(g.buffer(1),1) for g,p in allwaters],out_shape=dem.shape,transform=tf,all_touched=True,dtype='uint8').astype(bool)
    guard_allowed=(dem!=-32767)&np.isfinite(dem)&~bmask&~expanded&~portal_mask
    guard_targets={name+'|near_bank':target_mask(g.buffer(4).difference(g.buffer(1.75)),tf,dem.shape)&guard_allowed for name,g in waterdict.items()}
    guard,infog=thresholds(dem,guard_allowed,seeds,guard_targets,1,'outer_water_guard_1m')
    # Border/NoData are search limits, not surveyed impermeable barriers.
    route_fc=[];route_profiles=[];route_checks=[];uncertainties=[]
    reviewed=json.loads((BASE/'reviewed_priority_sites.geojson').read_text())['features']
    for f in reviewed:
        p=f['properties'];uncertainties.append({'type':'Feature','geometry':f['geometry'],'properties':{
            'uncertainty_id':p['candidate_id'],'priority':p['priority'],'target':p['target'],
            'evidence':p['evidence'],'needed_verification':p['needed_verification'],
            'role':'existing reviewed kerb/bank/inlet uncertainty; no control elevation imposed',
            'actual_inlet_confirmed':False,'evidence_image':p['evidence_image']}})
    prior_kerbs=json.loads((ROOT/'data/exp/data-briefing/hibiya_road_verification/correction_and_survey_candidates.geojson').read_text())['features']
    for f in prior_kerbs:
        p=f['properties'];uncertainties.append({'type':'Feature','geometry':f['geometry'],'properties':{
            'uncertainty_id':p['id'],'priority':2,'target':'ordinary_road','evidence':p['evidence'],
            'needed_verification':p['recommended_action'],'role':'prior road section edge/divider review; not measured continuous barrier','actual_inlet_confirmed':False}})
    uncertainties.append(feature(portal_review_area.representative_point(),{'uncertainty_id':'U_portal','priority':1,'target':'surface_route',
        'role':'known mixed-level underground access review area isolated in this surface-only search; not physical wall or surveyed portal footprint',
        'evidence':'先前点云/正射核验的地下停车坡道附近有上下层回波；未隔离路线距最低点约8.3m。',
        'needed_verification':'核对上层道路、坡道、挡墙和入口分别的连接；地下入口另作边界，不以单层DEM贯通。','actual_inlet_confirmed':False}))
    save_fc(PROC/'surface_search_isolation_area.geojson',[feature(portal_review_area,{'role':'conservative known mixed-level review window; isolated only for surface path search','not_surveyed_portal_footprint':True,'not_a_physical_wall':True})])
    for rec in base:
        if rec['threshold_source_z_m'] is None:continue
        seed=seeds[rec['source_id']];h=rec['threshold_source_z_m'];target=targets[rec['target_id']]
        ids=shortest_path(allowed&(dem<=h),seed,target)
        rr,cc=np.divmod(ids,dem.shape[1]);xx,yy=xy(tf,rr,cc);coords=np.c_[xx,yy];z=dem.ravel()[ids]
        sid=rec['source_id'];name,goal=rec['target_id'].split('|');rid=sid+'_'+{'心字池':'shinji','雲形池':'kumogata','日比谷濠':'moat'}[name]+'_'+goal
        rec['route_id']=rid;rec['route_length_m']=float((len(ids)-1)*.5)
        rec['class2_supported_fraction']=float(supported.ravel()[ids].mean())
        rec['route_source_z_max_m']=float(z.max());rec['route_start_cell_z_m']=float(z[0]);rec['route_end_cell_z_m']=float(z[-1])
        rec['minimum_distance_to_known_parking_lowpoint_m']=float(np.linalg.norm(coords-np.array(portal_summary['bottom_xy']),axis=1).min())
        rec['goal_meaning']='only outer dry shore; not a bank crest or inlet' if goal=='near_bank' else 'land approach to reviewed local section; conditional control/inlet still unknown'
        crest={'心字池':[2.25,2.25],'雲形池':[4.20,4.20],'日比谷濠':[2.10,2.20]}[name]
        if goal=='reviewed_section':
            rec['local_reviewed_height_bracket_m']=crest
            rec['conditional_threshold_with_local_review_m']=[max(h,x) for x in crest]
            rec['bracket_note']='Illustrative local source-review values, not a measured global minimum crest, a confidence interval, or verified outlet.'
        route_fc.append(feature(LineString(coords),{**rec,'actual_inlet_confirmed':False}))
        distance=np.arange(len(ids))*.5
        route_profiles.append({'route_id':rid,'distance_m':distance.tolist(),'easting_m':coords[:,0].tolist(),'northing_m':coords[:,1].tolist(),
            'dem_z_m':[float(x) for x in z],'class2_count':[int(x) for x in counts.ravel()[ids]],
            'class2_supported':[bool(x) for x in supported.ravel()[ids]]})
        # Mark where the path first leaves the vector road footprint; this is not a kerb survey.
        if sid=='R2' and goal=='reviewed_section':
            roads=unary_union([project(shape(f['geometry'])) for f in json.loads((BASE/'local_plateau_roads.geojson').read_text())['features']])
            roadmask=target_mask(roads,tf,dem.shape)
            off=np.flatnonzero(~roadmask.ravel()[ids])
            if len(off):
                k=int(off[0]);uncertainties.append(feature(Point(*coords[k]),{'uncertainty_id':rid+'_road_exit','display_id':'U1' if name in ('心字池','日比谷濠') else 'U2','priority':1,'target':name,
                    'role':'computed first exit from LOD1 road footprint, not actual kerb','evidence':f'路径在此离开模型道路面；局部DEM {z[k]:.2f}m。',
                    'needed_verification':'核查实际路缘/人行道/围栏连续性与开口，模型边界不能代替现场断线。','actual_inlet_confirmed':False}))
            # Only long unsupported stretches get new flags; sparse cells are not automatically errors.
            unsupported=~supported.ravel()[ids];starts=np.flatnonzero(unsupported&~np.r_[False,unsupported[:-1]]);ends=np.flatnonzero(unsupported&~np.r_[unsupported[1:],False])+1
            spans=sorted([(b-a,a,b) for a,b in zip(starts,ends) if b-a>=6],reverse=True)[:2]
            for rank,(size,a,b) in enumerate(spans,1):
                k=(a+b-1)//2;uncertainties.append(feature(Point(*coords[k]),{'uncertainty_id':rid+f'_support_gap_{rank}','priority':2,'target':name,
                    'role':'selected route point-cloud support gap, not evidence of a terrain error','gap_length_m':float(size*.5),
                    'evidence':'候选线路出现连续地面点云支撑不足的像元。','needed_verification':'核查树冠/分类缺失、插值及窄坎，保留来源时相；不是自动填平或校正依据。','actual_inlet_confirmed':False}))
        assert ids[0]==seed and target.ravel()[ids[-1]] and np.all(allowed.ravel()[ids]) and not np.any(portal_mask.ravel()[ids])
        assert abs(float(z.max())-h)<1e-6
        assert np.all(np.abs(np.diff(rr))+np.abs(np.diff(cc))==1)
        assert not rec['previous_lower_source_height_reaches_target']
        route_checks.append({'route_id':rid,'source_and_target_and_4_neighbour_edges_valid':True,'no_water_building_or_nodata_crossing':True,
             'maximum_matches_minimum_threshold':True,'next_lower_native_height_not_connected':True})
    index8={(r['source_id'],r['target_id']):r for r in loose}
    for rec in base:
        alt=index8[(rec['source_id'],rec['target_id'])]
        assert alt['threshold_source_z_m']<=rec['threshold_source_z_m']+1e-6
        rec['8_connected_sensitivity_z_m']=alt['threshold_source_z_m']
    levels=[2.45,2.60,4.20];reach_layers=[];reach_rows=[]
    for h in levels:
        # tolerance is only for floating representation of the displayed level.
        cc,_=label(allowed&(dem<=h+1e-5),generate_binary_structure(2,1));component=cc.ravel()[seeds['R2']]
        reach=(cc==component)&allowed if component else np.zeros(dem.shape,bool)
        reach_layers.append(reach.astype(np.uint8));reach_rows.append({'source_id':'R2','water_surface_source_z_m':h,
            'depth_at_R2_m':max(0.,h-source_rows[1]['dem_z_m']),'connected_area_m2':float(reach.sum()*.25),
            'volume_under_horizontal_level_within_connected_cells_m3':float(np.maximum(0,h-dem[reach]).sum()*.25),
            'volume_role':'idealized snapshot storage inside this masked domain; not required rainfall or retained water volume',
            'reached_targets':[k for k,m in targets.items() if np.any(m&reach)]})
    rasterprofile={**profile,'count':len(levels),'dtype':'uint8','nodata':255,'compress':'deflate'}
    with rasterio.open(PROC/'R2_static_connected_areas.tif','w',**rasterprofile) as ds:
        for i,(a,h) in enumerate(zip(reach_layers,levels),1):
            a=a.copy();a[~allowed]=255;ds.write(a,i);ds.set_band_description(i,f'R2-connected land at source Z {h:.2f}m, not flood prediction')
    save_fc(PROC/'source_reference_points.geojson',source_fc)
    save_fc(PROC/'target_bands_and_reviewed_sections.geojson',target_fc)
    save_fc(PROC/'threshold_paths.geojson',route_fc)
    save_fc(PROC/'uncertain_kerbs_banks_and_inlets.geojson',uncertainties)
    dump(OUT/'thresholds.json',{'main_4_connected':base,'corner_contact_8_connected':loose,'outer_water_guard_1m':guard})
    dump(OUT/'source_reference_points.json',source_rows);dump(OUT/'path_profiles.json',route_profiles);dump(OUT/'static_levels.json',reach_rows)
    sourcefiles=[BASE/'official_dem_0_50m_mosaic.tif',BASE/'ground_class2_count_native_grid.tif',BASE/'ground_class2_range_native_grid.tif',BASE/'local_building_exclusion.geojson',ROOT/'data/processed/plateau_water/central_three_wards_wtr.geojson',ROOT/'data/exp/data-briefing/hibiya_lowpoint_verification/verification_summary.json']
    provenance={'date':'2026-10-13','bounds':bounds,'grid_shape':list(dem.shape),'resolution_m':.5,
        'method':'Binary search exact native DEM height values; connected-component labelling; four-neighbour shortest path at first connection level. Eight-neighbour case only tests corner-contact sensitivity.',
        'reference_point_R2':'prior verified ordinary-road section S55, park-side road band c=15.5m; snapped to native 0.5m cell',
        'assumptions':['Hypothetical sustained source head and static horizontal-level connectivity, no finite rainfall supply or time simulation.',
            'Water interiors, NoData and buildings buffered0.5m excluded; ground DEM can omit thin structural barriers.',
            'Known 70x48m mixed-level underground parking review window isolated conservatively; not a surveyed entrance footprint or physical wall.',
            'Outer-band reach is not entry to water. Reviewed approaches plus local proxy heights remain conditional.',
            'Unknown pond/moat stages, controls and vertical datum reconciliation preclude actual hydraulic exchange judgement.',
            'Search limited to available rectangle; boundary contact implies unresolved outside-domain paths or drainage.',
            'Source-era topography retained; later construction not merged. Class2 support is common-source consistency, not independent accuracy.'],
        'source_sha256':{str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in sourcefiles},
        'evaluations':{'native4':info4,'corner8':info8,'water_guard':infog},'uncertainty_record_count':len(uncertainties),'no_original_data_modified':True,
        'uncertainty_count_note':'Records include shared/nearby anchors for multiple routes, not independent physical location count.',
        'diagnostic_comparison':'Unisolated mixed-level access region yields R2 2.45m for Shinji/moat; isolated main case yields 2.57m. Conservative review rectangle is not a surveyed barrier.'}
    dump(OUT/'method_and_provenance.json',provenance);dump(OUT/'validation.json',{'route_checks':route_checks,'8_connected_thresholds_never_exceed_4_connected':True,'no_height_or_stage_assigned_to_water':True})
    draw_results(dem,allowed,tf,bounds,reach_layers,levels,source_rows,route_fc,uncertainties,waterdict,base,route_profiles)
    write_report(source_rows,base,guard,uncertainties,provenance)
    print(json.dumps({'sources':source_rows,'main_results':[{k:r.get(k) for k in ('source_id','target_id','threshold_source_z_m','rise_above_source_cell_m','route_length_m','8_connected_sensitivity_z_m')} for r in base],'uncertainty_flags':len(uncertainties)},ensure_ascii=False,indent=2),flush=True)

def draw_results(dem,allowed,tf,bounds,reach_layers,levels,sources,route_fc,uncertainties,waterdict,base,profiles):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import Rectangle,Patch
    font=FontProperties(fname='/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
    extent=(bounds[0],bounds[2],bounds[1],bounds[3]);colors={'心字池':'#0075b0','雲形池':'#258b46','日比谷濠':'#d97615'}
    lookup={r['source_id']:r for r in sources}
    def decorate(ax):
        for name,g in waterdict.items():
            for part in (list(g.geoms) if hasattr(g,'geoms') else [g]):ax.plot(*np.array(part.exterior.coords).T,color='#0099be',lw=1)
            p=g.representative_point();ax.annotate(name,(p.x,p.y),xytext=(-30,10),textcoords='offset points',fontproperties=font,fontsize=9,
                bbox={'fc':'white','alpha':.8,'ec':'none','pad':1})
        ax.set_xlim(-7180,-6460);ax.set_ylim(-36500,-35700);ax.set_aspect('equal');ax.ticklabel_format(style='plain',useOffset=False)
        ax.set_xlabel('Easting (m; EPSG:6677)');ax.set_ylabel('Northing (m; EPSG:6677)')
        for r in sources:
            ax.plot(r['easting_m'],r['northing_m'],'*',color='#c11d5a',ms=9);ax.annotate(r['source_id'],(r['easting_m'],r['northing_m']),xytext=(7,7) if r['source_id']=='R1' else (-32,-13),textcoords='offset points',fontsize=9,color='#a31649',bbox={'fc':'white','alpha':.8,'ec':'none','pad':1})
        portal=json.loads((ROOT/'data/exp/data-briefing/hibiya_lowpoint_verification/verification_summary.json').read_text())['local_bounds_epsg6677']
        ax.add_patch(Rectangle((portal[0],portal[1]),portal[2]-portal[0],portal[3]-portal[1],fill=False,edgecolor='#9c3880',hatch='///',lw=.8))
    with rasterio.open(BASE/'official_orthophoto_0_20m_mosaic.tif') as ds:
        rgb=ds.read(out_shape=(3,dem.shape[0],dem.shape[1]))
    fig=plt.figure(figsize=(14,8));axs=[fig.add_axes([.055,.12,.385,.78]),fig.add_axes([.51,.12,.385,.78])]
    axs[0].imshow(np.moveaxis(rgb,0,-1),extent=extent,origin='upper')
    im=axs[1].imshow(np.ma.array(dem,mask=~allowed),extent=extent,origin='upper',cmap='terrain',vmin=1,vmax=5)
    for ax in axs:
        decorate(ax)
        for f in route_fc:
            r=f['properties']
            if r['source_id']!='R2' or not r['target_id'].endswith('reviewed_section'):continue
            name=r['target_id'].split('|')[0];g=project(shape(f['geometry']));coords=np.array(g.coords)
            ax.plot(*coords.T,color=colors[name],lw=1.6 if name=='心字池' else 2.4,ls='--' if name=='心字池' else '-',label=name)
        for f in uncertainties:
            p=f['properties'];id=p['uncertainty_id'];g=project(shape(f['geometry']));ax.plot(g.x,g.y,'+',color='#c22323',ms=5,alpha=.8)
            if id in ('A2','A3','A7','A10','U_portal') or id=='R2_shinji_reviewed_section_road_exit':
                txt='U1 新道路过渡' if id.endswith('road_exit') else ('地下混合区' if id=='U_portal' else id)
                offset={'A2':(-70,-8),'A3':(6,-14),'A7':(-20,-20),'A10':(6,6),'U_portal':(8,-5),'R2_shinji_reviewed_section_road_exit':(-110,18)}[id]
                ax.annotate(txt,(g.x,g.y),xytext=offset,textcoords='offset points',fontproperties=font,fontsize=8,color='#a01717',
                    arrowprops={'arrowstyle':'-','color':'#a01717','lw':.5},bbox={'fc':'white','alpha':.85,'ec':'none','pad':1})
        ax.legend(loc='lower left',prop=font)
    axs[0].set_title('正射影像与R2候选路径',fontproperties=font,fontsize=12)
    axs[1].set_title('源DEM与待核查位置；水域内高程未使用',fontproperties=font,fontsize=12)
    fig.canvas.draw();position=axs[1].get_position()
    fig.colorbar(im,cax=fig.add_axes([.912,position.y0,.02,position.height]),label='Source elevation (m; vertical reference unresolved)')
    fig.suptitle('日比谷静态地形连通性 | 岸边可达不等于实际入水',fontproperties=font,fontsize=16)
    fig.savefig(OUT/'connectivity_routes_and_uncertainties.png',dpi=165,bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(15,8),layout='constrained')
    for ax,a,h in zip(axs,reach_layers,levels):
        ax.set_anchor('N')
        ax.imshow(np.moveaxis(rgb,0,-1),extent=extent,origin='upper',alpha=.65)
        overlay=np.zeros((*a.shape,4),np.float32);overlay[a.astype(bool)]=[.05,.4,.78,.65]
        ax.imshow(overlay,extent=extent,origin='upper');decorate(ax)
        ax.set_title(f'假设水面源高程 {h:.2f} m\nR2当地水深约 {max(0.,h-lookup["R2"]["dem_z_m"]):.2f} m',fontproperties=font,fontsize=12)
    fig.suptitle('蓝色为与R2连接的地面像元 | 持续水头假设，非暴雨淹没预测',fontproperties=font,fontsize=15)
    fig.savefig(OUT/'static_level_connectivity.png',dpi=155,bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(3,1,figsize=(11,10),layout='constrained')
    for ax,name in zip(axs,NAMES):
        rec=next(r for r in base if r['source_id']=='R2' and r['target_id']==name+'|reviewed_section')
        p=next(p for p in profiles if p['route_id']==rec['route_id']);s=np.array(p['distance_m']);z=np.array(p['dem_z_m']);supported=np.array(p['class2_supported'])
        ax.plot(s,z,color=colors[name],lw=1,label='原生DEM地形')
        ax.scatter(s[~supported],z[~supported],s=6,color='#8d8d8d',alpha=.55,label='点云支撑不足像元（不是判定错误）')
        ax.axhline(rec['threshold_source_z_m'],color='#9b3535',ls='--',lw=1,label='首次地形连通水头')
        ax.axhline(lookup['R2']['dem_z_m'],color='#9c3880',lw=.8,ls=':',label='R2道路参考高程')
        ax.set_title(f'{name}：到复核岸段约 {rec["threshold_source_z_m"]:.2f}m，较R2高 {rec["rise_above_source_cell_m"]:.2f}m；入口待核查',fontproperties=font,fontsize=11)
        ax.set_xlabel('沿四邻接候选路径距离 (m)',fontproperties=font);ax.set_ylabel('源高程 (m)',fontproperties=font);ax.grid(alpha=.2)
        ax.legend(prop=font,loc='best',fontsize=8)
    fig.suptitle('路线地形与控制门槛 | 四邻接路径长度非实际水流长度',fontproperties=font,fontsize=15)
    fig.savefig(OUT/'threshold_path_profiles.png',dpi=155);plt.close(fig)

def write_report(sources,base,guard,uncertainties,provenance):
    source={r['source_id']:r for r in sources}
    rows=['日比谷测绘时期地形连通性试算','日期2026-10-13；项目根目录 /Volumes/Extension/Research/UDT01b','',
        '结论：按现有0.50m源DEM、隔离已核验地下停车出入口上下层混合区后，从R2道路参考点（源高程2.45m）至心字池南岸复核段和日比谷濠南岸复核段，最低路径水头均约2.57m，即较参考点高约0.12m。至雲形池近岸外侧约4.07m（高1.62m），至A7复核岸顶候选约4.20m（高1.75m）。这是范围内的静态地形筛选门槛，不是已确定的真实排水入口或洪水水深。',
        '三个水体的实际街道入口仍未确认；岸顶断线、窄路缘/围栏、受纳水位、控制和时相都可能改变真实门槛，不能由本轮结果给出可靠的实际阈值上界或排水能力。',
        '', '道路参考点：']
    for r in sources:
        lon,lat=LL.transform(r['easting_m'],r['northing_m'])
        rows.append(f"{r['source_id']}: E={r['easting_m']:.2f}, N={r['northing_m']:.2f}；{lon:.7f}°E, {lat:.7f}°N；DEM {r['dem_z_m']:.2f}m；Class2数{r['class2_count']}、均值{r['class2_mean_z_m']:.3f}m。")
    rows+=['R1是此前交通车道参考；R2来自已核验普通道路S55的公园侧车道带（横向+15.5m），落到最近0.50m像元中心。高程差换算针对该点，不是整条街统一的积水深度。R1高程2.58m，在其处算出的0增加只是源点已经高于部分路径，不能解释为零积水已流入池塘。',
        '', 'R2结果 | 到近岸外带 / 经已复核岸段的源地形高程 | 相对R2增加 | 经复核段路径长度/点云支撑：']
    for name in NAMES:
        near=next(r for r in base if r['source_id']=='R2' and r['target_id']==name+'|near_bank')
        bank=next(r for r in base if r['source_id']=='R2' and r['target_id']==name+'|reviewed_section')
        rows.append(f"{name}: {near['threshold_source_z_m']:.2f} / {bank['threshold_source_z_m']:.2f}m；{near['rise_above_source_cell_m']:.2f} / {bank['rise_above_source_cell_m']:.2f}m；{bank['route_length_m']:.1f}m / {bank['class2_supported_fraction']:.1%}。")
    rows+=['', '目标和计算含义：',
        '近岸外带是图示水域边缘外0.75–3m的地面，可能包括岸脚；不能当作最低岸顶。复核段分别为A2心字池南岸、A3日比谷濠南端及A7雲形池局部岸顶候选。A2/A3只是已复核的陆侧接近点，仍没有完整入口证据。A7为局部约4.20m源地形候选，不是全岸最低岸顶或已测入口。',
        '本轮另列局部源证据的条件门槛：心字池局部隆起约2.25m、日比谷濠约2.1–2.2m、雲形池约4.2m。前两者均低于到岸路径的2.57m门槛，但真正岸顶的连续性和窄结构仍未知。列出的局部数值不是置信区间，也不能证明入口已打开。',
        '算法对所有原生DEM高程值做二分检索，每层用四邻接连通分量判断源点能否到达目标；再在首次连通层寻找最短四邻接路径。没有把二维水体Z设为水面，也没有填洼或修改DEM。只要路径高程不超过假设水头便可连接，隐含持续补水；它不计降雨过程、局部水面坡度、摩阻、入流宽度或到达时间。达到刚好等于门槛只是临界接触，实际越流需水头高于控制点并具备受纳条件。',
        '', '地下停车出入口隔离与对照：',
        '未经隔离的诊断试算把前两个水体的门槛算到2.45m，路线距已核验坡道最低点约8.3m。该处有上层道路/下层坡道同时Class2回波，单值DEM的路径不能直接解释为地表通路。这组结果保留于unisolated_diagnostic_thresholds.json，仅作为敏感性诊断，不采用作主结果。',
        '主试算保守隔离此前核验的70×48m复核窗口：E[-6953.25,-6883.25]、N[-36348.25,-36300.25]。这不是实测入口轮廓，也不是现场不可穿越的墙；只是在尚未分层前防止地表路径借用混合区。未改写坡道的真实低地，地下入流将另作独立边界。隔离使主结果前两条门槛由2.45变为2.57m，说明这项表面选择足以改变结论。',
        '', '需要优先核查的位置：',
        'U1新共同道路过渡：本轮R2至心字池/日比谷濠的线路在E=-6937.25,N=-36378.75附近离开LOD1道路面（模型DEM约2.55m）；路缘、铺装边和围栏是否存在贯通开口未知。该坐标只是算法道路面退出锚点，不是实测路缘或口门。重点取得连续横断面及实际开口证据。',
        'A2心字池南岸：核对完整陆侧走廊、最低连续岸顶、局部隆起、真实入口及池水位。树冠下资料支撑不足，不能把近岸低值作出口。',
        'A3/A8日比谷濠南端：图示水域边缘内有约2.1–2.2m高回波，说明水部边界与真正岸顶不一致；调查路缘、石垣顶、口门及濠水位。',
        'A7/A6雲形池：核对约4.20m局部岸顶候选及路径高地的连续性；A4约3.09m近岸低点不能当作控制门槛。',
        'U_portal混合区：核对上层道路与下层坡道、挡墙、真实入口分别的连通关系；补测后可替换保守隔离窗口。',
        'A10及普通道路K1–K4：保留以前的路缘/中央分隔带不确定点。本轮R2至心字池和日比谷濠的线路仍经过A10附近，其确切路缘、围栏和较低绕行缺口要与U1共同核查。',
        '新增点云支撑空缺标记：候选线路连续≥3m不满足至少3个Class2点且像元内高差≤0.15m的规则时，标出最长两段。它们是资料空缺，不是判定地形错误；优先核对树下插值、窄结构和分类。',
        '', '敏感性与边界：',
        '八邻接角接触试算与四邻接主结果的门槛一致；图示水域向外保守加1m后，到相应外带的门槛也一致。这只说明这些栅格选择未改变结果，不能核验窄路缘、围栏、岸顶或入口。',
        '所有主结果的连通区均触及已下载矩形边缘。门槛只在该范围、掩膜与参考点条件下成立；范围外的分流、绕行和控制未知。0.12m或1.75m的地形差不能换算成需要多大暴雨，也不能断言道路将保持同一水平水面。',
        'DEM与点云共源，原精度没有独立核验。高程保留源值，尚未命名并统一垂直基准及成果改定；局部高差可筛查，实际水位接入仍需控制点和时相核对。测绘时期与后续改造现状分别保存。',
        '', '交付：',
        'connectivity_routes_and_uncertainties.png：源影像、源DEM、R2线路及不确定位置；static_level_connectivity.png：2.45/2.60/4.20m持续水头下的连接地面像元；threshold_path_profiles.png：路线地形和点云支撑。蓝色连接像元不是暴雨淹没预测。',
        'thresholds.json、source_reference_points.json、path_profiles.json、static_levels.json：量化结果；method_and_provenance.json：源文件校验、假设及参数；validation.json：路径和临界高程核验。',
        'data/processed/hibiya_connectivity_trial/：阈值路径、参考点、目标岸段、地下复核隔离区、不确定位置GeoJSON；R2_static_connected_areas.tif与原DEM同网格。',
        '', '复现：.venv/bin/python src/exp/trial_hibiya_connectivity.py；只重绘已有成果可加 --render-only。']
    (OUT/'assessment.txt').write_text('\n'.join(rows)+'\n',encoding='utf-8')

def render_saved():
    with rasterio.open(BASE/'official_dem_0_50m_mosaic.tif') as ds:dem=ds.read(1);tf=ds.transform;bounds=list(ds.bounds)
    with rasterio.open(PROC/'R2_static_connected_areas.tif') as ds:a=ds.read()
    allowed=a[0]!=255;layers=[(x==1).astype(np.uint8) for x in a]
    source=json.loads((OUT/'source_reference_points.json').read_text());data=json.loads((OUT/'thresholds.json').read_text())
    routes=json.loads((PROC/'threshold_paths.geojson').read_text())['features'];unc=json.loads((PROC/'uncertain_kerbs_banks_and_inlets.geojson').read_text())['features']
    waters={f['properties']['name']:project(shape(f['geometry'])) for f in json.loads((ROOT/'data/processed/plateau_water/central_three_wards_wtr.geojson').read_text())['features'] if f['properties']['name'] in NAMES}
    profiles=json.loads((OUT/'path_profiles.json').read_text());provenance=json.loads((OUT/'method_and_provenance.json').read_text())
    draw_results(dem,allowed,tf,bounds,layers,[2.45,2.60,4.20],source,routes,unc,waters,data['main_4_connected'],profiles)
    write_report(source,data['main_4_connected'],data['outer_water_guard_1m'],unc,provenance)

if __name__=='__main__':
    import sys
    render_saved() if '--render-only' in sys.argv else main()
