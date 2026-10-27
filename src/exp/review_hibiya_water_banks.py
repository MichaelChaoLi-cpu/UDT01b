"""Focused official orthophoto/LAS review of screened banks and kerbs."""
import os
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-mpl')
import sys,json,csv,struct
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from analyze_hibiya_water_corridor import las_points,projected,geographic,feature,savegeo,writecsv,ROOT,OUT,PROCESSED
import numpy as np,rasterio
from rasterio.windows import from_bounds
from shapely.geometry import shape,Point,LineString
from shapely.ops import nearest_points
from shapely import intersects_xy
from pyproj import Transformer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    targets={f['properties']['name']:projected(shape(f['geometry'])) for f in json.loads((ROOT/'data/processed/plateau_water/central_three_wards_wtr.geojson').read_text())['features'] if f['properties']['name'] in ['雲形池','心字池','日比谷濠']}
    specs=[
      {'id':'A1','target':'road/park','name':'P3 road-to-park transition','xy':[-6898.25,-36299.25],'u':[-1,0],'half':18,'note':'Terrain candidate crossing at park-side edge; exact kerb/opening unverified'},
      {'id':'A2','target':'心字池','name':'Shinji south bank / P1 endpoint','xy':[-6820.75,-36150.25],'half':18,'note':'P1 reaches outer bank; B2 low cell nearby is not crest'},
      {'id':'A3','target':'日比谷濠','name':'Hibiya moat south bank / P2 endpoint','xy':[-6676.75,-36008.75],'half':20,'note':'P2 reaches bank at road junction; inspect top of revetment'},
      {'id':'A4','target':'雲形池','name':'Kumogata east bank / P3 endpoint','xy':[-7065.75,-36212.25],'half':22,'note':'Outer-bank minimum B1; inspect higher crest between road and pond'},
      {'id':'A5','target':'日比谷濠','name':'Hibiya moat west bank low-cell B3','xy':[-6684.75,-35813.25],'half':20,'note':'Lowest supported exterior band cell, not demonstrated inlet'},
      {'id':'A6','target':'雲形池','name':'Kumogata candidate route crest','xy':[-7049.25,-36220.25],'u':[1,-.5],'half':16,'note':'P3 maximum source DEM height 4.07 m'},
    ]
    crest_summary=json.loads((OUT/'bank_crest_summary.json').read_text())
    for i,r in enumerate([r for r in crest_summary['candidates'] if r['rank_within_target']==1],7):
        specs.append({'id':f'A{i}','target':r['target'],'name':{'雲形池':'Kumogata ranked crest','心字池':'Shinji ranked crest','日比谷濠':'Hibiya moat ranked crest'}[r['target']],
                      'xy':[r['crest_easting_m'],r['crest_northing_m']],'u':[r['normal_unit_east'],r['normal_unit_north']],'half':20,
                      'note':f"Lowest ranked 0.75-8 m outer-strip high point {r['screened_outer_strip_high_z_m']:.2f} m; crest/inlet unverified"})
    specs.append({'id':'A10','target':'road/park','name':'Shared P1/P2 road-to-verge transition',
                  'xy':[-6840.75,-36214.75],'u':[-.8827,.4699],'half':22,
                  'note':'Shared path exits model road footprint; physical kerb/fence geometry and opening require verification'})
    manifest=json.loads((OUT/'acquisition_manifest.json').read_text());paths=[ROOT/next(p for p in e['extracted'] if p.endswith('.las')) for e in manifest['resources'] if e.get('kind')=='lp']
    for s in specs:
        p=np.array(s['xy'])
        if 'u' not in s:
            edge=nearest_points(targets[s['target']],Point(*p))[0];u=p-np.array(edge.coords[0]);s['mapped_water_edge_xy']=list(edge.coords[0])
        else:u=np.array(s['u'],dtype=float)
        if np.linalg.norm(u)<.1:raise ValueError('Cannot derive shoreline normal')
        s['u']=u/np.linalg.norm(u);s['v']=np.array([-s['u'][1],s['u'][0]])
        s['points']=[]
    for path in paths:
        a,scale,origin,meta=las_points(path)
        for begin in range(0,len(a),1000000):
            q=a[begin:begin+1000000];x=q['x']*scale[0]+origin[0];y=q['y']*scale[1]+origin[1]
            for s in specs:
                p=s['xy'];m=(abs(x-p[0])<45)&(abs(y-p[1])<45)&((q['class']&128)==0)
                if m.any():s['points'].append(np.c_[x[m],y[m],q['z'][m]*scale[2]+origin[2],q['class'][m]&31])
        print('Local points',path.name,flush=True)
    rows=[];review_features=[];all_profile=[]
    with rasterio.open(PROCESSED/'official_dem_0_50m_mosaic.tif') as dem,rasterio.open(PROCESSED/'official_orthophoto_0_20m_mosaic.tif') as ortho:
        for s in specs:
            p=np.array(s['xy']);a=np.vstack(s['points']);relative=a[:,:2]-p;along=relative@s['u'];cross=relative@s['v'];strip=(abs(cross)<.5)&(abs(along)<=s['half'])
            distance=np.arange(-s['half'],s['half']+.01,.25);coords=p+distance[:,None]*s['u'];zs=np.array([v[0] for v in dem.sample(coords)])
            water=np.array([targets[s['target']].covers(Point(*v)) for v in coords]) if s['target'] in targets else np.zeros(len(coords),bool)
            local=strip&(a[:,3]==2);other=strip&(a[:,3]!=2)
            water_points=intersects_xy(targets[s['target']],a[:,0],a[:,1]) if s['target'] in targets else np.zeros(len(a),bool)
            profile=[]
            for d in np.arange(-s['half']+.5,s['half'],1):
                m=local&(abs(along-d)<=.5);q=p+d*s['u'];dz=float(next(dem.sample([q]))[0]);isin=bool(targets[s['target']].covers(Point(*q))) if s['target'] in targets else False
                r={'candidate_id':s['id'],'profile_offset_m':d,'easting_m':q[0],'northing_m':q[1],'class2_count':int(m.sum()),'class2_median_z_m':float(np.median(a[m,2])) if m.any() else None,'class2_p10_z_m':float(np.percentile(a[m,2],10)) if m.any() else None,'class2_p90_z_m':float(np.percentile(a[m,2],90)) if m.any() else None,'dem_z_m':dz if dz!=-32767 and not isin else None,'inside_mapped_water':isin}
                profile.append(r);all_profile.append(r)
            np.savez_compressed(OUT/f'{s["id"]}_local_points.npz',xyz=a[:,:3],classification=a[:,3],along=along,cross=cross,profile_origin=p,profile_unit=s['u'])
            radius=40 if s['id'] in ['A2','A3'] else 30
            w=from_bounds(p[0]-radius,p[1]-radius,p[0]+radius,p[1]+radius,ortho.transform)
            rgb=ortho.read(window=w)
            fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
            axes[0].imshow(np.moveaxis(rgb,0,-1),extent=(p[0]-radius,p[0]+radius,p[1]-radius,p[1]+radius),origin='upper')
            for g in targets.values():
                if g.intersects(Point(*p).buffer(radius*1.5)):
                    for part in (list(g.geoms) if hasattr(g,'geoms') else [g]):axes[0].plot(*np.array(part.exterior.coords).T,color='#00bfff',lw=1)
            axes[0].plot(*coords[[0,-1]].T,'r-',lw=1);axes[0].plot(*p,'r+',ms=12)
            axes[0].set_xlim(p[0]-radius,p[0]+radius);axes[0].set_ylim(p[1]-radius,p[1]+radius);axes[0].set_aspect('equal');axes[0].ticklabel_format(useOffset=False,style='plain')
            axes[0].set_title('Official 0.20 m ortho; blue = mapped water edge')
            axes[1].scatter(along[other],a[other,2],s=4,alpha=.35,c='grey',label='Other classes (structures / vegetation)')
            axes[1].scatter(along[local&~water_points],a[local&~water_points,2],s=8,c='black',label='Class 2 outside mapped water, strip +/- 0.50 m')
            if np.any(local&water_points):axes[1].scatter(along[local&water_points],a[local&water_points,2],s=8,c='#1287b0',label='Class 2 inside mapped water (ambiguous returns)')
            zs[(zs==-32767)|water]=np.nan
            axes[1].plot(distance,zs,color='#ee8800',label='Native 0.50 m DEM (water excluded)')
            if s['target'] in targets:
                for r in profile:
                    if r['inside_mapped_water']:axes[1].axvspan(r['profile_offset_m']-.5,r['profile_offset_m']+.5,color='#00bfff',alpha=.13)
            upper=max(7,float(np.max(a[local&~water_points,2]))+.5) if np.any(local&~water_points) else 7
            axes[1].axvline(0,color='red',ls='--');axes[1].set_xlim(-s['half'],s['half']);axes[1].set_ylim(-.5,upper);axes[1].set_xlabel('Profile offset (m); + points toward dry exterior for bank normals');axes[1].set_ylabel('Source elevation (m)');axes[1].grid(alpha=.2);axes[1].legend(fontsize=7,loc='upper right')
            axes[1].set_title('Source consistency / barrier review; NOT a water level')
            fig.suptitle(s['id']+' | '+s['name']);fig.savefig(OUT/f'{s["id"]}_ortho_ground_profile.png',dpi=160,bbox_inches='tight');plt.close(fig)
            prop={'candidate_id':s['id'],'target':s['target'],'easting_m':p[0],'northing_m':p[1],'profile_unit_east':s['u'][0],'profile_unit_north':s['u'][1],'strip_class2_count':int(local.sum()),'strip_other_count':int(other.sum()),'local_class_counts':{str(int(c)):int(np.sum(a[:,3]==c)) for c in np.unique(a[:,3])},'interpretation':s['note']}
            rows.append(prop);review_features.append(feature(Point(*p),prop))
    writecsv(OUT/'focused_cross_section_profiles.csv',all_profile)
    savegeo(PROCESSED/'focused_review_locations.geojson',review_features)
    (OUT/'focused_review_summary.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    print('Focused ortho and ground profiles complete',flush=True)

if __name__=='__main__':main()
