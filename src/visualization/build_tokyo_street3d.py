"""Display-only street-level assets: existing PLATEAU mesh + admitted archived heads."""
from pathlib import Path
import os,json,gzip,hashlib
for k in ['PROJ_DATA','PROJ_LIB','GDAL_DATA']:os.environ.pop(k,None)
import numpy as np
from netCDF4 import Dataset
from scipy.spatial import cKDTree
from shapely.geometry import shape,Point,Polygon
from shapely import STRtree,points
from pyproj import Transformer
ROOT=Path(__file__).resolve().parents[2]
SITE=ROOT/'data/exp/dashboards/tokyo-resilience-briefing-v1'
OUT=SITE/'dist/assets/street3d'
P=ROOT/'data/processed'
def dump(p,d):p.write_text(json.dumps(d,separators=(',',':'),allow_nan=False))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 meshfile=P/'tokyo_mesh_53394611/mesh.json'
 metafile=P/'tokyo_mesh_53394611/manifest.json'
 meta=json.loads(metafile.read_text());mesh=json.loads(meshfile.read_text())
 origin=np.array(meta['local_origin_easting_northing'])
 groups={}
 for name in ['terrain','roads','water','buildings']:
  groups[name]=np.array([t['p'] for t in mesh if t['group']==name],dtype=np.float32)
 # Store actual model coordinates; do not apply artificial water or terrain height.
 parts=[];offset=0;groupmeta={}
 for name,a in groups.items():
  raw=a.astype('<f4').tobytes();parts.append(raw);groupmeta[name]={'byteOffset':offset,'floats':a.size,'triangles':len(a)};offset+=len(raw)
 (OUT/'mesh.bin').write_bytes(gzip.compress(b''.join(parts),compresslevel=6))
 tin=groups['terrain'].astype(float);tri2=tin[:,:,[0,2]]
 v1=tri2[:,1]-tri2[:,0];v2=tri2[:,2]-tri2[:,0];valid=abs(v1[:,0]*v2[:,1]-v1[:,1]*v2[:,0])>1e-9
 tri2=tri2[valid];tin=tin[valid];tree=STRtree([Polygon(t) for t in tri2])
 def ground(xz):
  pairs=tree.query(points(xz[:,0],xz[:,1]),predicate='intersects');h=np.full(len(xz),np.nan)
  ids,j=np.unique(pairs[0],return_index=True);t=tin[pairs[1,j]];a=t[:,0][:,[0,2]];b=t[:,1][:,[0,2]];c=t[:,2][:,[0,2]];q=xz[ids]
  den=(b[:,1]-c[:,1])*(a[:,0]-c[:,0])+(c[:,0]-b[:,0])*(a[:,1]-c[:,1])
  w0=((b[:,1]-c[:,1])*(q[:,0]-c[:,0])+(c[:,0]-b[:,0])*(q[:,1]-c[:,1]))/den
  w1=((c[:,1]-a[:,1])*(q[:,0]-c[:,0])+(a[:,0]-c[:,0])*(q[:,1]-c[:,1]))/den
  h[ids]=w0*t[:,0,1]+w1*t[:,1,1]+(1-w0-w1)*t[:,2,1]
  return h
 common=np.load(P/'three_ward_five_event_population_validation_v1/motor_common_support.npz')['common_admitted']
 eligible=np.load(P/'three_ward_surrogate_inputs/static_tensor_10m.npz')['Eligible Ground Mask']
 common=common&eligible
 runs=[]
 for tag in ['three_ward_cross_event_protection_v1','three_ward_five_event_population_validation_v1']:
  m=json.loads((P/tag/'run_manifest.json').read_text())
  for r in m['runs']:
   if r['arm']=='U':
    audit=ROOT/'data/exp/estimation'/tag/'execution'/r['case_id']/'audit.json'
    a=json.loads(audit.read_text());assert a['all_checks_passed'] and a['admission']['technical_reference']
    runs.append((r,tag,audit))
 assert len(runs)==10
 # Native arrays are south-up; admitted masks are north-up. Retain explicit mapping.
 with Dataset(ROOT/runs[0][0]['folder']/'sfincs_map.nc') as ds:
  x=np.asarray(ds['x'][:]);y=np.asarray(ds['y'][:]);shape0=x.shape
 tile=Polygon(meta['projected_cell_corners'])
 ids=np.flatnonzero((x>tile.bounds[0]+7.2)&(x<tile.bounds[2]-7.2)&(y>tile.bounds[1]+7.2)&(y<tile.bounds[3]-7.2)&np.flipud(common))
 xy=np.c_[x.ravel()[ids],y.ravel()[ids]]
 # Full cells must remain within the original mesh; no extrapolated border water.
 keep=[tile.covers(Polygon([(xx-5,yy-5),(xx+5,yy-5),(xx+5,yy+5),(xx-5,yy+5)])) for xx,yy in xy]
 ids=ids[keep];xy=xy[keep];rr,cc=np.unravel_index(ids,shape0)
 cells=np.c_[xy[:,0]-origin[0],-(xy[:,1]-origin[1])]
 # Compact rectangular reads avoid loading full-domain 109-frame maps.
 r0,r1=int(rr.min()),int(rr.max()+1);c0,c1=int(cc.min()),int(cc.max()+1)
 cases={};checks=[];bed_reference=None
 for run,tag,audit in sorted(runs,key=lambda q:q[0]['case_id']):
  case=run['case_id'];path=ROOT/run['folder']/'sfincs_map.nc'
  clockpath=P/tag/'execution'/(case+'_output_clock.npz')
  clock=np.load(clockpath)['Actual Map Time'].astype(float)
  assert len(clock)==109 and np.max(abs(clock-np.arange(109)*300))<=.001
  with Dataset(path) as ds:
   assert ds['zs'].shape==(109,*shape0)
   bed=np.ma.asarray(ds['zb'][r0:r1,c0:c1],dtype=float).filled(np.nan)[rr-r0,cc-c0]
   water=np.ma.asarray(ds['zs'][:,r0:r1,c0:c1],dtype=float).filled(np.nan)[:,rr-r0,cc-c0]
   head=np.where(np.isfinite(water)&(water>bed[None,:]+.001),water,np.nan).astype('<f4')
  if bed_reference is None:bed_reference=bed
  assert np.allclose(bed,bed_reference,equal_nan=True)
  # GPU sentinel is below all visible ground and never rendered.
  encoded=np.nan_to_num(head,nan=-99999.)
  name=case+'.bin';(OUT/name).write_bytes(gzip.compress(encoded.tobytes(),compresslevel=6))
  peak=int(np.nanargmax(np.nansum(np.maximum(head-bed[None,:],0),axis=1)))
  cases[case]={'asset':name,'actualTimes':clock.tolist(),'localVolumePeakFrame':peak}
  checks.append({'case':case,'audit':str(audit.relative_to(ROOT)),'auditSha256':sha(audit),'map':str(path.relative_to(ROOT)),'mapBytes':path.stat().st_size,'clockSha256':sha(clockpath),'finiteHeads':int(np.isfinite(head).sum()),'frames':109})
  print('STREET_HEAD_EXPORT',case,len(ids),flush=True)
 # Camera presets are selected on admitted ground road centerlines, not inside buildings.
 roads=json.loads((P/'three_ward_five_event_temporal_analysis_v1/clipped_roads_6677.geojson').read_text())['features']
 qualified=np.load(P/'three_ward_five_event_temporal_analysis_v1/clipped_road_lengths.npz')['qualified']
 project=Transformer.from_crs(4326,6677,always_xy=True)
 anchors=[('yaesu','Yaesu-side street',139.7691,35.6804),('sotobori','Sotobori-side street',139.7676,35.6787),('kyobashi','Kyobashi-side street',139.7705,35.6767)]
 candidates=[]
 for f in roads:
  if not qualified[int(f['properties']['road_index'])]:continue
  g=shape(f['geometry'])
  for ln in list(g.geoms) if g.geom_type=='MultiLineString' else [g]:
   if ln.length>90 and tile.buffer(-35).covers(ln):candidates.append((ln,f['properties']['road_index']))
 views=[]
 for vid,name,lon,lat in anchors:
  target=Point(*project.transform(lon,lat));ln,rid=min(candidates,key=lambda q:q[0].distance(target))
  middle=ln.project(target);start=max(3,middle-50);end=min(ln.length-3,middle+50)
  distance=np.linspace(start,end,121);world=np.array([[ln.interpolate(t).x,ln.interpolate(t).y] for t in distance])
  local=np.c_[world[:,0]-origin[0],-(world[:,1]-origin[1])];heights=ground(local)
  assert np.isfinite(heights).all()
  # Grid admission at every camera sample is explicit; do not turn unknown cells dry.
  sample_col=np.floor((world[:,0]-float(x[0,0])+5)/10).astype(int)
  sample_row=np.floor((world[:,1]-float(y[0,0])+5)/10).astype(int)
  cell_lookup={int(v):i for i,v in enumerate(ids)}
  lookup=[cell_lookup.get(int(r*shape0[1]+c),-1) for r,c in zip(sample_row,sample_col)]
  views.append({'id':vid,'name':name,'roadIndex':int(rid),'path':np.c_[local[:,0],heights,local[:,1]].round(5).tolist(),'headCell':lookup,'lengthMetres':float(end-start),'initialSample':60,'anchorRequested':[lon,lat],'roadSelection':'Nearest >90m admitted road centerline inside tile;display only'})
 metadata={'meshId':'53394611','crs':'EPSG:6677','origin':origin.tolist(),'groups':groupmeta,'cellCenters':cells.round(5).tolist(),'cellSize':10,'cases':cases,'frames':109,'views':views,'bounds':meta['geographic_bounds'],'palette':json.loads((ROOT/'data/results/tokyo_mesh_53394611/metadata.json').read_text())['palette'],'eyeHeight':1.65,'verticalExaggeration':1,'headConvention':'Absolute archived zs;native 10m cells;dry censored to no surface;quality-rejected cells omitted. Depth-test against original PLATEAU TIN,not terrain+depth. No interpolated detailed flood truth.','scope':'Existing Tokyo Station/Yaesu/Kyobashi mesh only;not all three wards','waterBodyConvention':'Original PLATEAU water geometry is a static contextual layer,not event river level.','nativeCellIds':ids.tolist()}
 dump(OUT/'manifest.json',metadata)
 validation={'passed':True,'displayOnly':True,'newHydraulicRuns':0,'activePilotUntouched':True,'allTenCasesNumericallyAdmitted':True,'commonMaskAndEligibleGround':True,'nativeToNorthUpMaskMappingExplicit':True,'absoluteWaterLevelsNotAddedToTerrain':True,'verticalExaggeration':1,'meshSourceSha256':sha(meshfile),'metadataSourceSha256':sha(metafile),'sourceCases':checks,'nativeCells':len(ids),'cameraPresets':[{'name':v['name'],'roadIndex':v['roadIndex'],'lengthMetres':v['lengthMetres'],'unknownCameraSamples':v['headCell'].count(-1)} for v in views]}
 dump(SITE/'street3d_asset_validation.json',validation)
 print('STREET_3D_ASSETS_PASS',flush=True)
if __name__=='__main__':main()

