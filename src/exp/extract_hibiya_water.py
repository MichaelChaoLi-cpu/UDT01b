"""Extract actual udx/wtr, clip to three ward boundaries and audit Hibiya.

Local reproduction: .venv/bin/python src/exp/extract_hibiya_water.py
Output geometry is 2D; original LOD1 heights are retained as source-vertex
statistics, never treated as riverbed or event water level. fld is not read.
"""
import os
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-mpl')
import csv,hashlib,json,zipfile,xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter
import numpy as np
import pyproj,rasterio
from rasterio.features import rasterize
from shapely import make_valid,normalize,set_precision,intersects_xy
from shapely.geometry import Polygon,MultiPolygon,GeometryCollection,shape,mapping,box
from shapely.ops import unary_union,transform,nearest_points
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PatchPolygon

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/exp/data-briefing/hibiya_water_overlay'
PROCESSED=ROOT/'data/processed/plateau_water'
RAW=ROOT/'data/raw/plateau/2025'
PILOT=ROOT/'data/exp/data-briefing/plateau_terrain_pilot'
G='http://www.opengis.net/gml';W='http://www.opengis.net/citygml/waterbody/2.0';C='http://www.opengis.net/citygml/2.0'
NS={'g':G,'w':W,'c':C}
TO_XY=pyproj.Transformer.from_crs(6668,6677,always_xy=True)
TO_LL=pyproj.Transformer.from_crs(6677,6668,always_xy=True)
CLASSES={'1000':'sea','1010':'tidal water','1020':'watercourse','1030':'river/stream','1060':'lake/pond','1190':'reservoir'}


def only_polygons(g):
    if isinstance(g,(Polygon,MultiPolygon)):return g
    return unary_union([only_polygons(q) for q in g.geoms if isinstance(q,(Polygon,MultiPolygon,GeometryCollection))]) if isinstance(g,GeometryCollection) else Polygon()


def projected_boundary(g):
    typ=g['type'];coords=g['coordinates']
    if typ=='LineString':
        if not np.allclose(coords[0],coords[-1],atol=1e-10,rtol=0):raise ValueError('Ward boundary is not closed')
        geom=Polygon(coords)
    elif typ=='MultiLineString':
        rings=[]
        for ring in coords:
            if not np.allclose(ring[0],ring[-1],atol=1e-10,rtol=0):raise ValueError('Unclosed boundary part')
            rings.append(Polygon(ring))
        geom=unary_union(rings)
    else:geom=shape(g)
    return only_polygons(make_valid(transform(TO_XY.transform,geom)))


def source_xyz(element):
    v=np.fromstring(element.text or '',sep=' ')
    if v.size%3:raise ValueError('Unexpected GML coordinate dimension')
    v=v.reshape(-1,3);x,y=TO_XY.transform(v[:,1],v[:,0])
    return np.c_[x,y,v[:,2]]


def water_geometry(b):
    lod=b.find('w:lod1MultiSurface',NS)
    if lod is None:raise ValueError('Expected LOD1 surface; no silent LOD0 substitution')
    polys=[];vertices=[];invalid=0
    for p in lod.iter(f'{{{G}}}Polygon'):
        rings=[]
        for ring in p.iter(f'{{{G}}}LinearRing'):
            pos=ring.find('g:posList',NS)
            if pos is not None:
                q=source_xyz(pos);rings.append(q[:,:2]);vertices.append(q)
        if rings:
            geom=Polygon(rings[0],rings[1:])
            if not geom.is_valid:invalid+=1;geom=only_polygons(make_valid(geom))
            if not geom.is_empty:polys.append(geom)
    if not polys:raise ValueError('No explicit polygons in water LOD1')
    return only_polygons(unary_union(polys)),np.vstack(vertices),invalid


def zstats(z):
    return {'n':len(z),'min_m':float(z.min()),'median_m':float(np.median(z)),'max_m':float(z.max())} if len(z) else {'n':0}


def save_geojson(path,features):
    path.write_text(json.dumps({'type':'FeatureCollection','features':features},ensure_ascii=False)+'\n')


def write_csv(path,rows):
    if not rows:return
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def draw_geom(ax,g,**kwargs):
    if isinstance(g,Polygon):
        ax.add_patch(PatchPolygon(np.array(g.exterior.coords),**kwargs))
        for ring in g.interiors:ax.plot(*np.array(ring.coords).T,color='white',lw=.5)
    elif hasattr(g,'geoms'):
        for p in g.geoms:draw_geom(ax,p,**kwargs)


def main():
    OUT.mkdir(parents=True,exist_ok=True);PROCESSED.mkdir(parents=True,exist_ok=True)
    boundaries={};boundary_sources=[]
    for related in sorted(RAW.glob('*/*related.zip')):
        ward=related.parent.name
        with zipfile.ZipFile(related) as archive:
            name=next(n for n in archive.namelist() if n.endswith('_border.geojson'))
            original=archive.read(name);document=json.loads(original)
            boundaries[ward]=unary_union([projected_boundary(f['geometry']) for f in document['features']])
            boundary_sources.append({'archive':str(related.relative_to(ROOT)),'member':name,'sha256':hashlib.sha256(original).hexdigest(),'area_m2':boundaries[ward].area})
    domain=unary_union(list(boundaries.values()))
    save_geojson(PROCESSED/'central_three_ward_boundaries.geojson',[{'type':'Feature','properties':{'ward':w},'geometry':mapping(transform(TO_LL.transform,g))} for w,g in boundaries.items()])
    files_by_sha={};all_files=[];object_occurrences=0
    # Preserve file provenance for repeated mesh files across ward packages.
    for archive_path in sorted(RAW.glob('*/*citygml*.zip')):
        with zipfile.ZipFile(archive_path) as archive:
            for name in sorted(n for n in archive.namelist() if n.startswith('udx/wtr/') and n.endswith('.gml')):
                data=archive.read(name);sha=hashlib.sha256(data).hexdigest()
                ref={'archive':str(archive_path.relative_to(ROOT)),'member':name,'sha256':sha,'bytes':len(data)}
                all_files.append(ref)
                if sha not in files_by_sha:
                    dictionaries={}
                    for dname in archive.namelist():
                        if dname.startswith('codelists/') and dname.endswith('.xml'):
                            droot=ET.fromstring(archive.read(dname))
                            dictionaries[dname]={d.findtext('g:name',namespaces=NS):d.findtext('g:description',namespaces=NS) for d in droot.iter(f'{{{G}}}Definition')}
                    files_by_sha[sha]={'data':data,'references':[],'dictionaries':dictionaries}
                files_by_sha[sha]['references'].append(ref)
    objects={};id_conflicts=[];same_id_duplicates=0
    for item in files_by_sha.values():
        root=ET.fromstring(item['data']);bodies=root.findall('.//w:WaterBody',NS)
        envelope=root.find('g:boundedBy/g:Envelope',NS)
        if envelope is None or envelope.get('srsName')!='http://www.opengis.net/def/crs/EPSG/0/6697' or envelope.get('srsDimension')!='3':
            raise ValueError('Unexpected source CRS/dimension; cannot assume latitude-longitude-height')
        for ref in item['references']:ref['envelope_crs']=envelope.get('srsName');ref['coordinate_dimension']=3
        object_occurrences+=len(bodies)*len(item['references'])
        for b in bodies:
            ident=b.get(f'{{{G}}}id');geom,xyz,invalid=water_geometry(b)
            # Only fingerprint rounding (1 mm) is used; retained geometry unchanged.
            fingerprint=hashlib.sha256(normalize(set_precision(geom,.001)).wkb).hexdigest()
            name=b.findtext('g:name',default='',namespaces=NS);cl=b.findtext('w:class',default='',namespaces=NS)
            lod0=b.find('w:lod0MultiCurve',NS)
            z0=np.concatenate([source_xyz(q)[:,2] for q in lod0.findall('.//g:posList',NS)]) if lod0 is not None else np.array([])
            quality={}
            for q in b.iter():
                key=q.tag.split('}')[-1]
                if key in ['geometrySrcDescLod0','geometrySrcDescLod1','srcScaleLod0','srcScaleLod1','publicSurveySrcDescLod0','publicSurveySrcDescLod1']:
                    code_space=q.get('codeSpace','');dictionary=code_space.replace('../../','')
                    quality.setdefault(key,[]).append({'code':q.text,'codeSpace':code_space,'description':item['dictionaries'].get(dictionary,{}).get(q.text)})
            info={'id':ident,'names':{name},'classes':{cl},'geom':geom,'xyz':xyz,'fingerprint':fingerprint,
                  'creation_dates':{b.findtext('c:creationDate',default='',namespaces=NS)},'sources':list(item['references']),
                  'lod0_z_stats':zstats(z0),'invalid_source_polygons':invalid,
                  'quality_attributes':quality,
                  'immediate_property_names':[q.tag.split('}')[-1] for q in b]}
            if ident in objects:
                prev=objects[ident]
                if prev['fingerprint']!=fingerprint or zstats(prev['xyz'][:,2])!=zstats(xyz[:,2]):
                    id_conflicts.append(ident);raise ValueError('Same GML ID differs in geometry/elevation: '+ident)
                prev['sources'].extend(info['sources']);same_id_duplicates+=1
            else:objects[ident]=info
        print('Parsed water file',item['references'][0]['member'],'objects',len(bodies),flush=True)
    geometric={};geom_duplicates=0
    for obj in objects.values():
        key=obj['fingerprint']
        if key in geometric:
            prev=geometric[key];prev['names'].update(obj['names']);prev['classes'].update(obj['classes']);prev['sources'].extend(obj['sources']);prev['creation_dates'].update(obj['creation_dates']);prev['source_ids'].append(obj['id']);geom_duplicates+=1
        else:obj['source_ids']=[obj['id']];geometric[key]=obj
    with rasterio.open(PILOT/'tokyo_dem_native_0_50m.tif') as ds:
        dem=ds.read(1);tf=ds.transform;bounds=ds.bounds;crs=ds.crs;resolution=ds.res
    pilot=box(*bounds);context=box(bounds.left-500,bounds.bottom-500,bounds.right+500,bounds.top+500)
    clipped=[];nearby=[];features=[];clip_rejected=0
    for obj in geometric.values():
        g=only_polygons(obj['geom'].intersection(domain))
        if g.is_empty or g.area<=.01:clip_rejected+=1;continue
        obj['clipped']=g;clipped.append(obj)
        inside=intersects_xy(g,obj['xyz'][:,0],obj['xyz'][:,1]);zs=zstats(obj['xyz'][inside,2])
        name=' / '.join(sorted(n for n in obj['names'] if n and n!='Null')) or '(unnamed water body)'
        wards=[w for w,wardgeom in boundaries.items() if g.intersection(wardgeom).area>.01]
        prop={'water_id':obj['id'],'name':name,'source_names':sorted(obj['names']),'class_codes':sorted(obj['classes']),
              'class_names':[CLASSES.get(v,v) for v in sorted(obj['classes'])],'source_ids':obj['source_ids'],'ward_intersections':wards,
              'area_within_three_wards_m2':g.area,'creation_dates':sorted(obj['creation_dates']),
              'source_lod1_all_vertex_z_stats':zstats(obj['xyz'][:,2]),'original_vertices_within_clipped_domain_z_stats':zs,
              'lod0_vertex_z_stats':obj['lod0_z_stats'],'quality_attributes':obj['quality_attributes'],
              'source_immediate_property_names':obj['immediate_property_names'],
              'height_use_status':'unverified_all_zero_LOD1; do not assign hydraulic stage' if np.all(obj['xyz'][:,2]==0) else 'unverified_source_LOD1_elevation; no event stage or bathymetry',
              'invalid_source_polygons_repaired_for_2d_footprint':obj['invalid_source_polygons'],
              'sources':obj['sources'],'geometry_note':'2D footprint from union of explicit LOD1 polygons; clipped to ward union. No hydraulic stage or bed depth assigned.'}
        obj['props']=prop
        features.append({'type':'Feature','properties':prop,'geometry':mapping(transform(TO_LL.transform,g))})
        if g.intersects(context):
            obj['distance_to_dem_m']=g.distance(pilot)
            obj['overlap_dem_m2']=g.intersection(pilot).area
            nearby.append(obj)
    save_geojson(PROCESSED/'central_three_wards_wtr.geojson',features)
    nearby.sort(key=lambda o:o['distance_to_dem_m'])
    rows=[]
    for i,o in enumerate(nearby,1):
        xyz=o['xyz'];in_context=intersects_xy(context,xyz[:,0],xyz[:,1]);h=zstats(xyz[in_context,2])
        nearest_dem,nearest_water=nearest_points(pilot,o['clipped'])
        rows.append({'label':f'W{i}','water_id':o['id'],'name':o['props']['name'],'class_codes':'|'.join(sorted(o['classes'])),
                     'distance_from_dem_footprint_m':o['distance_to_dem_m'],'water_area_overlap_dem_m2':o['overlap_dem_m2'],
                     'context_original_lod1_vertices':h['n'],'context_source_lod1_min_z_m':h.get('min_m'),
                     'context_source_lod1_median_z_m':h.get('median_m'),'context_source_lod1_max_z_m':h.get('max_m'),
                     'nearest_water_easting_m':nearest_water.x,'nearest_water_northing_m':nearest_water.y,
                     'connectivity_status':'Outside native DEM; complete bank/path elevation comparison unavailable' if o['overlap_dem_m2']<.01 else 'Intersects DEM; terrain and structural verification still required'})
        o['label']=f'W{i}'
    save_geojson(PROCESSED/'hibiya_context_wtr.geojson',[{'type':'Feature','properties':dict(o['props'],context_label=o['label'],distance_from_dem_footprint_m=o['distance_to_dem_m'],water_area_overlap_dem_m2=o['overlap_dem_m2']), 'geometry':mapping(transform(TO_LL.transform,o['clipped'].intersection(context)))} for o in nearby])
    write_csv(OUT/'hibiya_nearby_water_inventory.csv',rows)
    totalmask=rasterize([(o['clipped'],1) for o in nearby],out_shape=dem.shape,transform=tf,fill=0,dtype='uint8')
    profile= {'driver':'GTiff','height':dem.shape[0],'width':dem.shape[1],'count':1,'dtype':'uint8','crs':crs,'transform':tf,'compress':'deflate'}
    with rasterio.open(OUT/'water_footprint_on_native_dem.tif','w',**profile) as ds:ds.write(totalmask,1)
    roads=json.loads((PILOT/'roads_source.geojson').read_text())
    road_geoms=[only_polygons(shape(f['geometry']).intersection(pilot)) for f in roads['features']]
    road_geoms=[g for g in road_geoms if not g.is_empty]
    save_geojson(PROCESSED/'hibiya_pilot_roads.geojson',[{'type':'Feature','properties':{'source':'Previously extracted PLATEAU LOD1 road; extent limited to DEM pilot'},'geometry':mapping(transform(TO_LL.transform,g))} for g in road_geoms])
    # Position map makes the native DEM coverage limitation explicit.
    fig=plt.figure(figsize=(12,9),layout='constrained')
    grid=fig.add_gridspec(1,2,width_ratios=[3.2,1])
    ax=fig.add_subplot(grid[0,0]);keyax=fig.add_subplot(grid[0,1]);keyax.axis('off')
    im=ax.imshow(dem,extent=(bounds.left,bounds.right,bounds.bottom,bounds.top),origin='upper',cmap='terrain',vmin=-1,vmax=7)
    keylines=['WATER INVENTORY','Distance from DEM edge','(straight-line, not flow path)','']
    for g in road_geoms:draw_geom(ax,g,facecolor='none',edgecolor='.25',linewidth=.4,alpha=.7)
    for o in nearby:
        g=o['clipped'].intersection(context);draw_geom(ax,g,facecolor='#5ac8ed',edgecolor='#1464a0',linewidth=1,alpha=.8)
        p=g.representative_point();label=o['props']['name']
        display={'雲形池':'Kumogata Pond','心字池':'Shinji Pond','日比谷濠':'Hibiya Moat','凱旋濠':'Gaisen Moat'}.get(label,'unnamed water' if label=='(unnamed water body)' else label)
        offsets={'W3':(-38,20),'W4':(16,-22),'W10':(8,-18),'W11':(15,0)}
        ax.annotate(o['label'],(p.x,p.y),xytext=offsets.get(o['label'],(7,7)),textcoords='offset points',fontsize=9,
                    bbox={'boxstyle':'round,pad=.15','fc':'white','alpha':.9,'ec':'none'},arrowprops={'arrowstyle':'-','color':'.3','lw':.7})
        keylines.extend([o['label']+'  '+display,f"     {o['distance_to_dem_m']:.0f} m",''])
    keyax.text(0,1,'\n'.join(keylines),va='top',fontsize=9)
    draw_geom(ax,pilot,facecolor='none',edgecolor='red',linewidth=2)
    ax.text(bounds.left,bounds.bottom-45,'Native DEM coverage',color='red',fontsize=9)
    ax.plot(-6918.25,-36324.25,'rx',markersize=8,label='Verified underground parking ramp low point')
    ax.plot([-6956,-6923],[-36465,-36403],color='magenta',lw=3,label='Verified ordinary road segment')
    ax.plot([],[],color='#5ac8ed',lw=6,label='CityGML water footprint (not a flow connection)')
    ax.plot([],[],color='.25',lw=1,label='PLATEAU LOD1 road outline')
    ax.set_xlim(context.bounds[0],context.bounds[2]);ax.set_ylim(context.bounds[1],context.bounds[3]);ax.set_aspect('equal')
    ax.set_xlabel('Easting (m; JGD2011 Plane IX)');ax.set_ylabel('Northing (m)');ax.grid(alpha=.2);ax.legend(loc='lower right',fontsize=8)
    ax.set_title('Hibiya water, roads and 0.50 m DEM\n500 m rectangular expansion; red box = available DEM',fontsize=12)
    fig.colorbar(im,ax=keyax,orientation='horizontal',fraction=.06,pad=.04,label='Native DEM elevation (m)')
    fig.savefig(OUT/'hibiya_water_native_dem_overlay.png',dpi=180,bbox_inches='tight',pad_inches=.2);plt.close(fig)
    counts=Counter(c for o in clipped for c in o['classes'])
    summary={'ward_boundary_sources':boundary_sources,'source_files':all_files,'file_occurrences':len(all_files),'distinct_file_sha256':len(files_by_sha),
             'object_occurrences_in_archives':object_occurrences,'unique_source_ids':len(objects),'same_id_duplicates_after_file_dedup':same_id_duplicates,
             'geometric_duplicates':geom_duplicates,'geometry_fingerprint_rule':'Normalized 2D WKB snapped to 1 mm for duplicate key only; retained geometry not snapped',
             'id_conflicts':id_conflicts,'features_outside_three_wards_rejected':clip_rejected,'clipped_unique_water_features':len(clipped),'class_counts':dict(counts),
             'nearby_context_count':len(nearby),'context_buffer_m':500,'native_dem_water_pixel_count':int(totalmask.sum()),'native_dem_bounds':list(bounds),
             'nearby_water':rows,'limits':['Source objects can span several meshes; object count is not number of named rivers.','LOD0 Z is a planar placeholder; LOD1 surface vertices are source model elevations, not verified bed or event stage.','No observed hydraulic connection is created from proximity or shared names.','Road overlay is available only within original DEM pilot in this first map.']}
    summary['context_shape']='DEM bounds expanded by 500 m in each coordinate direction; rectangle corners exceed a 500 m radial distance'
    summary['all_zero_lod1_feature_count']=sum(bool(np.all(o['xyz'][:,2]==0)) for o in clipped)
    summary['all_zero_lod0_feature_count']=sum(o['lod0_z_stats'].get('min_m')==0 and o['lod0_z_stats'].get('max_m')==0 for o in clipped)
    summary['lod2_or_lod3_or_bottom_feature_count']=sum(any(k in o['immediate_property_names'] for k in ['lod2Solid','lod3Solid','boundedBy']) for o in clipped)
    assert all(o['clipped'].is_valid and o['clipped'].difference(domain).area<1e-6 for o in clipped)
    assert len({o['fingerprint'] for o in clipped})==len(clipped)
    summary['validation']={'all_clipped_geometries_valid':True,'all_clipped_geometries_within_ward_union':True,'unique_geometry_fingerprints':True,'native_water_mask_matches_geometry':int(totalmask.sum())==0 and all(o['overlap_dem_m2']==0 for o in nearby)}
    (OUT/'extraction_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['source_files','ward_boundary_sources']},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
