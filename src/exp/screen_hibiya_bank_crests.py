"""Screen outer-bank high points across short normals, not shoreline minima."""
import os
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
import sys,json,csv
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from analyze_hibiya_water_corridor import ROOT,OUT,PROCESSED,projected,feature,savegeo,writecsv
import numpy as np,rasterio
from shapely.geometry import shape,Point,LineString
from shapely.ops import unary_union

def main():
    allwater=json.loads((ROOT/'data/processed/plateau_water/central_three_wards_wtr.geojson').read_text())['features']
    waters={f['properties']['name']:projected(shape(f['geometry'])) for f in allwater if f['properties']['name'] in ['雲形池','心字池','日比谷濠']}
    waterunion=unary_union([projected(shape(f['geometry'])) for f in allwater])
    buildings=unary_union([projected(shape(f['geometry'])).buffer(.5) for f in json.loads((PROCESSED/'local_building_exclusion.geojson').read_text())['features']])
    rows=[];features=[];candidate_list=[]
    with rasterio.open(PROCESSED/'official_dem_0_50m_mosaic.tif') as dem,rasterio.open(PROCESSED/'ground_class2_count_native_grid.tif') as count,rasterio.open(PROCESSED/'ground_class2_range_native_grid.tif') as spread:
        for name,g in waters.items():
            shore=LineString(g.exterior.coords)
            for station in np.arange(0,shore.length,3):
                p=np.array(shore.interpolate(station).coords[0]);before=np.array(shore.interpolate((station-.75)%shore.length).coords[0]);after=np.array(shore.interpolate((station+.75)%shore.length).coords[0]);t=after-before
                if np.linalg.norm(t)<.1:continue
                u=np.array([-t[1],t[0]])/np.linalg.norm(t)
                if g.contains(Point(*(p+u*2))):u=-u
                offsets=np.arange(.75,8.001,.25);q=p+offsets[:,None]*u
                z=np.array([v[0] for v in dem.sample(q)]);n=np.array([v[0] for v in count.sample(q)]);width=np.array([v[0] for v in spread.sample(q)])
                outside=np.array([not waterunion.covers(Point(*v)) and not buildings.covers(Point(*v)) for v in q])
                valid=outside&np.isfinite(z)&(z!=-32767)
                support=valid&(n>=3)&(width<=.15)
                # An unobserved wall/kerb can still invalidate this screening.
                accepted=valid.mean()>=.8 and support.sum()>=10 and support.mean()>=.5
                if not valid.any():continue
                j=int(np.flatnonzero(valid)[np.argmax(z[valid])]);crest=q[j]
                props={'target':name,'shore_station_m':float(station),'shore_easting_m':p[0],'shore_northing_m':p[1],'normal_unit_east':u[0],'normal_unit_north':u[1],'crest_easting_m':crest[0],'crest_northing_m':crest[1],'screened_outer_strip_high_z_m':float(z[j]),'crest_offset_from_mapped_shore_m':float(offsets[j]),'crest_class2_count':int(n[j]),'crest_class2_range_m':float(width[j]) if np.isfinite(width[j]) else None,'land_valid_fraction':float(valid.mean()),'ground_supported_fraction':float(support.mean()),'accepted_for_ranking':bool(accepted),'interpretation':'Maximum valid DEM across 0.75-8 m outer normal; screen only, not a surveyed bank crest or verified inlet'}
                rows.append(props);features.append(feature(LineString(np.r_[p[None],q]),props))
            eligible=[r for r in rows if r['target']==name and r['accepted_for_ranking'] and r['crest_class2_count']>=3 and r['crest_class2_range_m'] is not None and r['crest_class2_range_m']<=.15]
            eligible.sort(key=lambda r:r['screened_outer_strip_high_z_m'])
            if not eligible:raise ValueError('No supported bank screening candidate')
            selected=[]
            for r in eligible:
                if all(np.linalg.norm(np.array([r['crest_easting_m'],r['crest_northing_m']])-np.array([prev['crest_easting_m'],prev['crest_northing_m']]))>15 for prev in selected):selected.append(r)
                if len(selected)==3:break
            for rank,r in enumerate(selected,1):candidate_list.append(dict(r,rank_within_target=rank))
    writecsv(OUT/'bank_normal_crest_screening.csv',rows)
    writecsv(OUT/'bank_crest_ranked_candidates.csv',candidate_list)
    savegeo(PROCESSED/'bank_normal_screening.geojson',features)
    savegeo(PROCESSED/'bank_crest_ranked_candidates.geojson',[feature(Point(r['crest_easting_m'],r['crest_northing_m']),r) for r in candidate_list])
    (OUT/'bank_crest_summary.json').write_text(json.dumps({'normal_spacing_m':3,'normal_range_m':[.75,8],'selection_rule':'At least 80% valid exterior; at least 50% of samples supported by >=3 class2 points with range<=0.15 m; highest cell itself supported; retain 3 candidates separated by >15 m','candidate_count':len(candidate_list),'normal_count':len(rows),'candidates':candidate_list,'limits':['Highest ground DEM sample in a short strip is a proxy, not a surveyed bank crest.','Shoreline positional/time error and vegetation/walls can change the result.','Lowest ranked value is not established as the global minimum real bank crest.']},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(candidate_list,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
