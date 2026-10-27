"""Extract building bases and classified roads; preserve original terrain, nested domain and rain area."""
import os
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
from pathlib import Path
import json,zipfile,xml.etree.ElementTree as ET,collections,hashlib
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from rasterio.transform import from_origin
from pyproj import Transformer
from shapely.geometry import Polygon,box,mapping
from shapely.ops import unary_union
from shapely import make_valid,STRtree
from scipy.ndimage import distance_transform_edt
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'data/processed/three_ward_urban_constraints';OLD=ROOT/'data/processed/three_ward_sfincs_scenarios'
XY=Transformer.from_crs(6668,6677,always_xy=True)
def tag(e):return e.tag.rsplit('}',1)[-1]
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def polys(node):
 out=[]
 for po in node.iter():
  if tag(po)!='Polygon':continue
  rings=[];zs=[]
  for ring in po.iter():
   if tag(ring)!='LinearRing':continue
   ps=next((q for q in ring.iter() if tag(q)=='posList' and q.text),None)
   if ps is None:continue
   a=np.fromstring(ps.text,sep=' ')
   if len(a)<12 or len(a)%3:continue
   a=a.reshape(-1,3);x,y=XY.transform(a[:,1],a[:,0]);rings.append(np.c_[x,y]);zs.append(a[:,2])
  if not rings:continue
  g=make_valid(Polygon(rings[0],rings[1:]))
  if not g.is_empty and g.area>.05:out.append((g,np.concatenate(zs)))
 return out

def extract():
 records=[];audit=[];seen=set();counts=collections.Counter()
 for archive in sorted((ROOT/'data/raw/plateau/2025').rglob('*citygml*.zip')):
  with zipfile.ZipFile(archive) as z:
   for theme in ['bldg','tran','brid']:
    members=[i for i in z.infolist() if i.filename.startswith('udx/'+theme+'/') and i.filename.endswith('.gml')]
    for j,i in enumerate(members):
     filecounts=collections.Counter()
     with z.open(i) as h:
      context=ET.iterparse(h,events=['start','end']);_,root=next(context)
      for event,ob in context:
       if event!='end' or tag(ob)!='cityObjectMember':continue
       object_=next(iter(ob),None)
       if object_ is None:root.clear();continue
       ident=next((v for k,v in object_.attrib.items() if k.endswith('}id')),'')
       signature=(theme,ident)
       if ident and signature in seen:counts['duplicate_gml_ids']+=1;root.clear();continue
       seen.add(signature);filecounts['objects']+=1
       parts=[];method='';cls=''
       if theme=='bldg':
        ground=[po for q in object_.iter() if tag(q)=='GroundSurface' for po in polys(q)]
        footprint=[po for q in object_.iter() if tag(q)=='lod0FootPrint' for po in polys(q)]
        if ground:parts=ground;method='semantic_GroundSurface'
        elif footprint:parts=footprint;method='lod0FootPrint'
        else:
         solid=[po for q in object_.iter() if tag(q)=='lod1Solid' for po in polys(q)]
         if solid:
          low=min(float(a.min()) for g,a in solid);parts=[(g,a) for g,a in solid if np.ptp(a)<.05 and float(a.max())<=low+.10];method='lod1Solid_lowest_horizontal_base'
        cls='building'
       elif theme=='tran':
        sections=sorted(set(q.text.strip() for q in object_.iter() if tag(q)=='sectionType' and q.text));cls='road_ground_candidate' if sections and set(sections)<=set(['1','4']) else 'road_structure_or_unknown'
        parts=[po for q in object_.iter() if tag(q)=='lod1MultiSurface' for po in polys(q)];method='lod1MultiSurface';filecounts['sections_'+','.join(sections or ['unknown'])]+=1
       else:
        parts=[po for q in object_.iter() if tag(q) in ['lod1Solid','lod2Solid','GroundSurface'] for po in polys(q)];cls='bridge_exclusion';method='solid_projected_extent'
       if parts:
        g=make_valid(unary_union([v[0] for v in parts]));a=np.concatenate([v[1] for v in parts])
        if not g.is_empty and g.area>.1:
         rec={'type':'Feature','geometry':mapping(g),'properties':{'gml_id':ident,'class':cls,'method':method,'z_min':float(a.min()),'z_max':float(a.max()),'source':str(archive.relative_to(ROOT)),'member':i.filename,'section_codes':sections if theme=='tran' else []}}
         records.append(rec);filecounts['usable_'+cls]+=1;filecounts[method]+=1
       else:filecounts['missing_usable_base_or_geometry']+=1
       root.clear()
     counts.update(filecounts);audit.append({'archive':str(archive.relative_to(ROOT)),'member':i.filename,'crc32':f'{i.CRC:08x}',**dict(filecounts)})
     if j%5==0 or j==len(members)-1:print(archive.parent.name,theme,j+1,'/',len(members),'total geometries',len(records),flush=True)
 dump(P/'constraint_vectors_6677.geojson',{'type':'FeatureCollection','crs':{'type':'name','properties':{'name':'EPSG:6677'}},'features':records})
 dump(P/'geometry_audit.json',{'counts':dict(counts),'files':audit,'building_roof_edges_used':False,'road_zero_Z_is_not_elevation':True,'road_ground_codes':['1','4'],'excluded_structure_codes':['2','3','5','6','7','unknown'],'lod1_base_is_geometric_inference':True})

def rasters():
 from shapely.geometry import shape
 features=json.loads((P/'constraint_vectors_6677.geojson').read_text())['features'];groups={}
 for cls in ['building','road_ground_candidate','road_structure_or_unknown','bridge_exclusion']:
  geoms=[shape(f['geometry']) for f in features if f['properties']['class']==cls];groups[cls]=(geoms,STRtree(geoms))
 with rasterio.open(OLD/'terrain_sampling_0_50m.tif') as d:prof=d.profile.copy();h,w=d.shape;tf=d.transform
 prof.update(dtype='uint8',nodata=0)
 with rasterio.open(P/'constraints_0_50m.tif','w',**prof) as out:
  for r in range(0,h,1600):
   for c in range(0,w,1600):
    win=Window(c,r,min(1600,w-c),min(1600,h-r));wt=out.window_transform(win);bb=box(*rasterio.windows.bounds(win,tf));ar=np.zeros((int(win.height),int(win.width)),dtype='uint8');layers={}
    for cls,(gs,tr) in groups.items():
     inds=tr.query(bb,predicate='intersects');layers[cls]=rasterize([(gs[k],1) for k in inds],out_shape=ar.shape,transform=wt,dtype='uint8') if len(inds) else np.zeros_like(ar)
    road=layers['road_ground_candidate'].astype(bool)&~layers['road_structure_or_unknown'].astype(bool)&~layers['bridge_exclusion'].astype(bool)
    ar[road]=2;ar[layers['road_structure_or_unknown'].astype(bool)|layers['bridge_exclusion'].astype(bool)]=3;ar[layers['building'].astype(bool)]=1;out.write(ar,1,window=win)
   print('Rasterized row',r,'/',h,flush=True)
 with rasterio.open(OLD/'active_mask_20m.tif') as d:old=d.read(1);ctf=d.transform
 fractions={dx:np.zeros((h//int(dx/.5),w//int(dx/.5)),dtype='float32') for dx in [20,10]};roadfrac={dx:np.zeros_like(fractions[dx]) for dx in [20,10]};classes=collections.Counter()
 prof.update(dtype='float32',nodata=np.nan)
 with rasterio.open(P/'hydraulic_obstacle_0_50m.tif','w',**prof) as out,rasterio.open(P/'manning_0_50m.tif','w',**prof) as man,rasterio.open(P/'constraints_0_50m.tif') as con,rasterio.open(OLD/'terrain_sampling_0_50m.tif') as dem:
  for r in range(0,h,1600):
   win=Window(0,r,w,min(1600,h-r));a=con.read(1,window=win);z=dem.read(1,window=win);mask=np.repeat(np.repeat(old[r//40:(r+int(win.height))//40],40,axis=0),40,axis=1);building=(a==1)&(mask==1);road=(a==2)&(mask==1)&~building;classes.update({'building_pixels_interior':int(building.sum()),'road_pixels_interior':int(road.sum()),'structure_uncertain_pixels_interior':int(((a==3)&(mask==1)).sum())})
   for dx in [20,10]:
    n=int(dx/.5);rr=r//n;nr=int(win.height)//n;fractions[dx][rr:rr+nr]=building.reshape(nr,n,w//n,n).mean(axis=(1,3));roadfrac[dx][rr:rr+nr]=road.reshape(nr,n,w//n,n).mean(axis=(1,3))
   obstacle=z.copy();obstacle[building]+=10.;out.write(obstacle,1,window=win);rough=np.full(z.shape,.04,dtype='float32');rough[road]=.03;rough[~np.isfinite(z)]=np.nan;man.write(rough,1,window=win)
 model_contracts=[]
 for dx in [20,10]:
  factor=20//dx;mask=np.repeat(np.repeat(old,factor,axis=0),factor,axis=1);covered=(mask==1)&(fractions[dx]>=1);mask[covered]=0
  dist,indices=distance_transform_edt(mask!=1,return_indices=True);donors=collections.Counter()
  for r,c in zip(*np.where(covered)):donors[(int(indices[0,r,c]),int(indices[1,r,c]))]+=dx*dx
  model={'dx':dx,'shape':list(mask.shape),'bounds':list(rasterio.transform.array_bounds(h,w,tf)),'original_interior_area_m2':int((old==1).sum())*400,'active_interior_area_m2':int((mask==1).sum())*dx*dx,'removed_fully_building_cells':int(covered.sum()),'roof_source_area_m2':int(covered.sum())*dx*dx,'roof_max_donor_distance_m':float(dist[covered].max(initial=0)*dx),'roof_donors':[{'row':r,'col':c,'area_m2':a} for (r,c),a in sorted(donors.items())],'boundary_method':'nested expansion of original20m boundary; original20m stage points/fields retained','partial_building_rain':'uniform rain adds full-cell volume to subgrid open area; immediate intra-cell roof runoff assumption','fully_building_rain':'removed from mask; volume-conserving nearest interior open-cell discharge sources; no pipe routing','building_barrier_m':10.,'building_barrier_is_assumed_obstacle_not_roof_height':True,'road_manning_n':.03,'other_manning_n':.04,'terrain_burn_or_lowering':False,'subgrid_nlevels':64}
  model_contracts.append(model);fprof=prof.copy();fprof.update(height=mask.shape[0],width=mask.shape[1],transform=from_origin(tf.c,tf.f,dx,dx),dtype='uint8',nodata=0)
  with rasterio.open(P/f'active_mask_{dx}m.tif','w',**fprof) as d:d.write(mask,1)
  fprof.update(dtype='float32',nodata=None)
  for name,ar in [('building_fraction',fractions[dx]),('road_fraction',roadfrac[dx])]:
   with rasterio.open(P/f'{name}_{dx}m.tif','w',**fprof) as d:d.write(ar,1)
 dump(P/'model_contracts.json',model_contracts);dump(P/'raster_audit.json',{'fine_sampling_m':.5,'interior_areas_km2':{k:v*.25/1e6 for k,v in classes.items()},'raw_dem_modified':False,'fully_building_cells_removed_and_roof_rain_restored':True,'bridge_and_underpass_geometry_not_merged_with_ground':True,'cross_ward_external_buildings_roads_not_acquired':True,'boundary_strip_obstacles_not_imposed_for_fixed_boundary_comparison':True})
 from shapely.ops import transform
 ward_features=json.loads((ROOT/'data/processed/plateau_water/central_three_ward_boundaries.geojson').read_text())['features'];ward_ids=rasterize([(transform(XY.transform,shape(f['geometry'])),i+1) for i,f in enumerate(ward_features)],out_shape=old.shape,transform=ctf);ward_rows=[]
 for i,name in enumerate(['千代田区','中央区','港区','跨区部分'],1):
  select=(old==1)&((ward_ids==i) if i<4 else (ward_ids==0));ward_rows.append({'ward':name,'original_interior_km2':float(select.sum()*400/1e6),'building_footprint_km2':float(fractions[20][select].sum(dtype=np.float64)*400/1e6),'ground_road_km2':float(roadfrac[20][select].sum(dtype=np.float64)*400/1e6)})
 dump(P/'ward_constraint_areas.json',ward_rows)
 print('CONSTRAINTS',dict(classes),flush=True)
if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['extract','rasters','all']);a=ap.parse_args();P.mkdir(exist_ok=True)
 if a.action in ['extract','all']:extract()
 if a.action in ['rasters','all']:rasters()
