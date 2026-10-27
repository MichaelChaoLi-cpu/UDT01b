"""Render reviewed positions, validate source grids/routes, preserve decisions."""
import os
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-mpl')
import sys,json,struct,csv
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from analyze_hibiya_water_corridor import ROOT,OUT,PROCESSED,projected,geographic,feature,savegeo,writecsv
import numpy as np,rasterio
from rasterio.windows import from_bounds
from rasterio.features import rasterize
from shapely.geometry import shape,Point,LineString
from shapely.ops import unary_union
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REVIEWS={
 'A1':('路缘/围栏核查','树冠遮挡的公园侧过渡，宽面高差可见，狭窄路缘/围栏是否连续及有无开口未证实。','核查实际铺装边缘、路缘顶、围栏底和公园通道；不要把附近地下出入口当成地表通道。',2),
 'A2':('优先核查心字池南岸','南端从外侧地面约2.6m经约2.25m隆起降向近岸约1.6m；尚未确认跨路缘/岸缘的完整开口。','补测道路—人行道—池岸横断面及最低连续岸顶、开口宽度和池水位；核实是地表越流还是设施连接。',1),
 'A3':('优先核查日比谷濠南岸','道路侧约1.7–1.9m，但岸缘点云有约2.1–2.2m隆起；目前未辨出街道直入开口。','复核路缘与岸顶断线；调查约30m角部是否有口门/管口、控制设施，记录入口阈值和水位。',1),
 'A4':('低岸点不作入口','B1近岸最低地形约3.09m，附近仍有约4.2m及更高地形；树冠遮挡，不能以B1设出口。','如进一步研究雲形池，先沿岸顶实测，不使用近岸最小像元代替控制高程。',3),
 'A5':('剔除西岸低点作为入口','B3约1.28m处为低岸脚候选；横断面外侧上升至约7.6m，存在高陡坎阻隔。','不建街道直接交换口；如需研究该段，仅在有真实通水设施证据后补测。',3),
 'A6':('雲形池路径高点','候选路径最大DEM约4.07m，局部Class2点云中位数约4.13m，源内一致；不是独立精度检验。','保留高点；若需建模，仅按实测断线重建窄坡坎，不按积水洼地自动填平。',3),
 'A7':('雲形池低岸顶候选','短法线筛选约4.20m，点云支持局部高岸缘；最低全岸顶尚未证实，未确认入流口。','优先级低于心字池/日比谷濠；只有核实岸顶及真实入口后才建立交换边界。',2),
 'A8':('修正自动岸顶候选','自动外侧0.75–8m筛选值1.95m漏掉图示水域边缘内的约2.1m隆起；岸线位置/定义需复核。','与A3合并一次核查；用实测/可辨识岸顶断线替代纯水部面边界。',1),
 'A9':('剔除低台地作为街道入口','近岸短条带约1.35m，但延长横断面后外侧升至约5.8–6.0m陡坎；低值不代表道路可直入。','保留为岸脚/低台地候选，不作道路入口；将陡坎/石垣候选纳入阻挡断线。',2),
 'A10':('优先核查共用道路过渡','局部车道/沟槽带约1.76–1.90m，公园侧宽面约2.58–2.62m；存在明显地面抬升，不能把模型道路面边界当成开放入口。','核查车行道—沟槽—路缘—人行道—围栏/绿篱底的连续横断面；记录降缘、开口及能否绕过地形抬升。',1),
}

def main():
    s=json.loads((OUT/'screening_summary.json').read_text());m=json.loads((OUT/'acquisition_manifest.json').read_text())
    loc=json.loads((PROCESSED/'focused_review_locations.geojson').read_text())['features']
    reviews=[];fs=[];sections=[]
    for f in loc:
        p=f['properties'];id=p['candidate_id'];status,evidence,action,priority=REVIEWS[id];q=projected(shape(f['geometry']))
        r={'candidate_id':id,'target':p['target'],'easting_m':p['easting_m'],'northing_m':p['northing_m'],'longitude':shape(f['geometry']).x,'latitude':shape(f['geometry']).y,'priority':priority,'review_status':status,'evidence':evidence,'needed_verification':action,'confirmed_hydraulic_inlet':False,'period':'Official source epoch; exact flight date unresolved','evidence_image':str((OUT/f'{id}_ortho_ground_profile.png').relative_to(ROOT))}
        reviews.append(r);fs.append(feature(q,r))
        u=np.array([p['profile_unit_east'],p['profile_unit_north']]);origin=np.array(q.coords[0]);sections.append(feature(LineString([origin-u*20,origin+u*20]),{'candidate_id':id,'purpose':'核查横断面位置，不是已测岸顶线；原图条带范围以对应图/CSV为准','priority':priority,'required_height_reference':'Tie survey to documented local TP control and record height revision/epoch before matching source elevations'}))
    writecsv(OUT/'reviewed_priority_sites.csv',reviews);savegeo(PROCESSED/'reviewed_priority_sites.geojson',fs);savegeo(PROCESSED/'proposed_survey_cross_sections.geojson',sections)
    validation={};projection=[]
    with rasterio.open(PROCESSED/'official_dem_0_50m_mosaic.tif') as dem,rasterio.open(ROOT/'data/exp/data-briefing/plateau_terrain_pilot/tokyo_dem_native_0_50m.tif') as old:
        a=dem.read(1,window=from_bounds(*old.bounds,dem.transform));validation['original_pilot_values_exact_match']=bool(np.array_equal(a,old.read(1)))
        whole=dem.read(1);bounds=dem.bounds;tf=dem.transform
    for sheet in m['sheets']:
        with rasterio.open(ROOT/sheet['dem_path']) as original:
            r,c=rasterio.transform.rowcol(tf,original.bounds.left+.25,original.bounds.top-.25)
            assert np.array_equal(whole[r:r+600,c:c+800],original.read(1))
    validation['all_eight_dem_sheets_values_exact_match']=True
    profile=list(csv.DictReader((OUT/'screening_path_profiles.csv').open()))
    allwater=json.loads((ROOT/'data/processed/plateau_water/central_three_wards_wtr.geojson').read_text())['features']
    water_geoms=[projected(shape(f['geometry'])) for f in allwater]
    mask=rasterize([(g,1) for g in water_geoms],out_shape=whole.shape,transform=tf,all_touched=True,dtype='uint8')
    building_geoms=[projected(shape(f['geometry'])).buffer(.5) for f in json.loads((PROCESSED/'local_building_exclusion.geojson').read_text())['features']]
    bmask=rasterize([(g,1) for g in building_geoms],out_shape=whole.shape,transform=tf,all_touched=True,dtype='uint8')
    route_checks=[]
    for r in s['routes']:
        rows=[q for q in profile if q['route_id']==r['route_id']];coords=np.array([[float(q['easting_m']),float(q['northing_m'])] for q in rows]);z=np.array([float(q['dem_z_m']) for q in rows]);rr,cc=rasterio.transform.rowcol(tf,coords[:,0],coords[:,1]);rr=np.array(rr);cc=np.array(cc)
        assert np.all(whole[rr,cc]==z) and np.all(mask[rr,cc]==0) and np.all(bmask[rr,cc]==0) and np.all(z!=-32767)
        assert abs(float(z.max())-r['peak_source_dem_m'])<1e-6
        assert abs(np.linalg.norm(np.diff(coords,axis=0),axis=1).sum()-r['length_m'])<1e-6
        delta=np.diff(np.c_[rr,cc],axis=0);assert np.all(np.max(abs(delta),axis=1)==1)
        route_checks.append({'route_id':r['route_id'],'no_water_or_buildings_or_nodata_crossing':True,'grid_neighbours_and_length_and_peak_valid':True,'minimum_distance_to_verified_parking_lowpoint_m':float(np.linalg.norm(coords-[-6918.25,-36324.25],axis=1).min())})
    validation['route_checks']=route_checks
    for e in m['resources']:
        if e.get('kind')!='lp':continue
        path=ROOT/next(p for p in e['extracted'] if p.endswith('.las'))
        with path.open('rb') as f:h=f.read(227);f.seek(0);h=f.read(struct.unpack_from('<I',h,96)[0])
        pos=struct.unpack_from('<H',h,94)[0];vlrs=[];keys=[]
        for i in range(struct.unpack_from('<I',h,100)[0]):
            _,user,rid,size,desc=struct.unpack_from('<H16sHH32s',h,pos);data=h[pos+54:pos+54+size]
            v={'user_id':user.rstrip(b'\0').decode(errors='replace'),'record_id':rid,'description':desc.rstrip(b'\0').decode(errors='replace')}
            if rid==34735:
                values=struct.unpack('<'+'H'*(len(data)//2),data);keys=[list(values[j:j+4]) for j in range(4,len(values),4)];v['geokey_directory']=keys
            if rid in [2112,34737]:v['text']=data.rstrip(b'\0').decode(errors='replace')
            vlrs.append(v);pos+=54+size
        projection.append({'tile':e['tile'],'vlrs':vlrs,'named_vertical_crs_found':any(k[0] in [4096,4098] for k in keys),'horizontal_epsg_6677_key':any(k==[3072,0,1,6677] for k in keys),'note':'No projection VLR in some adjacent LAS files; coordinate agreement with matched official DEM/ortho is checked, but not a named datum declaration.'})
    validation['all_source_zips_crc_verified']=all(e.get('zip_crc_verified') for e in m['resources'] if e.get('kind'))
    validation['named_vertical_reference_still_unresolved']=not any(p['named_vertical_crs_found'] for p in projection)
    (OUT/'validation_and_projection_metadata.json').write_text(json.dumps({'validation':validation,'las_projection':projection},ensure_ascii=False,indent=2)+'\n')
    # Render the reviewed location map; water interior and NoData remain white.
    paths=json.loads((PROCESSED/'screening_paths.geojson').read_text())['features']
    with rasterio.open(PROCESSED/'official_orthophoto_0_20m_mosaic.tif') as ds:rgb=ds.read()
    extent=(bounds.left,bounds.right,bounds.bottom,bounds.top)
    fig,axes=plt.subplots(1,2,figsize=(14,10),layout='constrained')
    axes[0].imshow(np.moveaxis(rgb,0,-1),extent=extent,origin='upper');im=axes[1].imshow(np.ma.array(whole,mask=(mask>0)|(whole==-32767)),extent=extent,origin='upper',cmap='terrain',vmin=0,vmax=7)
    names={'雲形池':'Kumogata','心字池':'Shinji','日比谷濠':'Hibiya moat'}
    for ax in axes:
        for f in allwater:
            if f['properties']['name'] in names:
                g=projected(shape(f['geometry']));ax.plot(*np.array(g.exterior.coords).T,color='#00bfff',lw=.9)
        for f in sorted(paths,key=lambda f: {'P2':0,'P1':1,'P3':2}[f['properties']['route_id']]):
            g=projected(shape(f['geometry']));id=f['properties']['route_id'];col={'P1':'#167ac6','P2':'#ff7600','P3':'#1b9e35'}[id]
            ax.plot(*np.array(g.coords).T,color=col,lw=1.3 if id=='P1' else 2,ls='--' if id=='P1' else '-',label=id+' '+names[f['properties']['target']])
        ax.plot([-6953.35,-6920.35],[-36466.41,-36404.41],color='magenta',lw=2,label='Verified road lane: source area')
        for f in loc:
            id=f['properties']['candidate_id']
            if id not in ['A1','A2','A3','A5','A7','A9','A10']:continue
            p=projected(shape(f['geometry']));label='A3/A8' if id=='A3' else id
            ax.plot(p.x,p.y,'r+',ms=7);ax.annotate(label,(p.x,p.y),xytext=(7,6),textcoords='offset points',color='red',fontsize=9,bbox={'fc':'white','alpha':.8,'ec':'none','pad':1})
        ax.set_xlim(-7200,-6500);ax.set_ylim(-36550,-35630);ax.set_aspect('equal');ax.set_xlabel('Easting (m; JGD2011 IX)');ax.set_ylabel('Northing (m)');ax.grid(alpha=.15)
    axes[0].set_title('Official 0.20 m ortho + reviewed candidate sites');axes[1].set_title('Native 0.50 m DEM; water and NoData excluded')
    axes[0].legend(loc='lower left',fontsize=8);fig.colorbar(im,ax=axes[1],label='Source elevation (m)')
    fig.suptitle('Hibiya source-epoch review | paths are hypotheses; inlets remain unconfirmed',fontsize=13)
    fig.savefig(OUT/'corridor_reviewed_overview.png',dpi=160,bbox_inches='tight');plt.close(fig)
    print(json.dumps(validation,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
