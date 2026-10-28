"""Retain original TIN terrain along5m road corridors and build component portals.
No global depression filling, building removal, bridge invention or new bank outlets.
"""

# Resolve original raw names to fixed Moga members; no local raw copies.
import sys as _moga_sys
from pathlib import Path as _MogaPath
_moga_sys.path.insert(0, str(_MogaPath(__file__).resolve().parents[1]))
from moga_raw_inputs import raw_path as moga_raw, source_reference as moga_source, project_path as moga_project_path
import os
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-mpl')
from pathlib import Path
import json,zipfile,xml.etree.ElementTree as ET,heapq
import numpy as np,rasterio
from rasterio.transform import Affine,from_origin
from scipy.ndimage import label
from pyproj import Transformer
ROOT=Path(__file__).resolve().parents[2] if Path(__file__).parent.name=='exp' else Path('/Volumes/Extension/Research/UDT01b');OLD=ROOT/'data/processed/three_ward_surface_trial';PRO=ROOT/'data/processed/three_ward_road_connectivity';OUT=ROOT/'data/exp/data-briefing/three_ward_road_connectivity';TOXY=Transformer.from_crs(6668,6677,always_xy=True)

def write(p,a,tf,nodata=None):
 with rasterio.open(p,'w',driver='GTiff',height=a.shape[0],width=a.shape[1],count=1,dtype=a.dtype,crs=6677,transform=tf,nodata=nodata,compress='deflate',tiled=True) as d:d.write(a,1)

def main():
 PRO.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True);audit=json.loads((ROOT/'data/exp/data-briefing/three_ward_surface_trial/input_audit.json').read_text());a=dict(np.load(OLD/'surface_inputs.npz'));tf=Affine(*audit['transform']);nr,nc=a['land'].shape
 with rasterio.open(OLD/'road_mask_5m.tif') as d:road=d.read(1).astype(bool);ftf=d.transform
 with rasterio.open(OLD/'building_roof_proxy_5m.tif') as d:roof=d.read(1).astype(bool)
 def fine(x):return np.repeat(np.repeat(x,4,axis=0),4,axis=1)
 allowed=road&~roof&fine(a['domain']&(a['water']==0)&np.isfinite(a['z']));ward=fine(a['ward']);dm=allowed;left=ftf.c;top=ftf.f;dx=5.;fnr,fnc=road.shape;cache=PRO/'road_ground_5m.tif';records=[]
 if cache.exists():
  with rasterio.open(cache) as d:z=d.read(1).astype(float)
 else:
  z=np.full(road.shape,np.nan);owner=np.zeros(road.shape,dtype='uint8')
  class Terrain:
   def __init__(self,wi):self.wi=wi;self.capture=False;self.text=[];self.batch=[];self.total=0;self.hits=0
   def start(self,t,attrs):
    if t.endswith('}posList'):self.capture=True;self.text=[]
   def data(self,s):
    if self.capture:self.text.append(s)
   def end(self,t):
    if t.endswith('}posList'):
     self.capture=False;v=np.fromstring(''.join(self.text),sep=' ')
     if len(v)==12:self.batch.append(v.reshape(4,3)[:3]);self.total+=1
     if len(self.batch)>=20000:self.flush()
   def close(self):self.flush()
   def flush(self):
    if not self.batch:return
    p=np.array(self.batch);self.batch=[];x,y=TOXY.transform(p[:,:,1],p[:,:,0]);h=p[:,:,2];c0=np.ceil((x.min(1)-left)/dx-.5).astype(int);c1=np.floor((x.max(1)-left)/dx-.5).astype(int);r0=np.ceil((top-y.max(1))/dx-.5).astype(int);r1=np.floor((top-y.min(1))/dx-.5).astype(int);ids=np.where((c1>=c0)&(r1>=r0)&(c1>=0)&(c0<fnc)&(r1>=0)&(r0<fnr))[0]
    for dc in range(int(np.max(c1[ids]-c0[ids],initial=0))+1):
     for dr in range(int(np.max(r1[ids]-r0[ids],initial=0))+1):
      cc=c0[ids]+dc;rr=r0[ids]+dr;k=(cc<=c1[ids])&(rr<=r1[ids])&(cc>=0)&(cc<fnc)&(rr>=0)&(rr<fnr);ii=ids[k];cc=cc[k];rr=rr[k];k=dm[rr,cc];ii=ii[k];cc=cc[k];rr=rr[k]
      if not len(ii):continue
      xx=left+(cc+.5)*dx;yy=top-(rr+.5)*dx;x0,x1,x2=x[ii].T;y0,y1,y2=y[ii].T;den=(y1-y2)*(x0-x2)+(x2-x1)*(y0-y2);valid=abs(den)>1e-9;den=np.where(valid,den,1);b0=((y1-y2)*(xx-x2)+(x2-x1)*(yy-y2))/den;b1=((y2-y0)*(xx-x2)+(x0-x2)*(yy-y2))/den;b2=1-b0-b1;valid&=(b0>=-1e-7)&(b1>=-1e-7)&(b2>=-1e-7);ii=ii[valid];cc=cc[valid];rr=rr[valid];zz=b0[valid]*h[ii,0]+b1[valid]*h[ii,1]+b2[valid]*h[ii,2];self.hits+=len(zz);prefer=~np.isfinite(z[rr,cc])|((ward[rr,cc]==self.wi)&(owner[rr,cc]!=self.wi));z[rr[prefer],cc[prefer]]=zz[prefer];owner[rr[prefer],cc[prefer]]=self.wi
  for wi,p in enumerate(sorted((moga_raw(ROOT, 'plateau/2025')).rglob('*citygml*.zip')),1):
   with zipfile.ZipFile(p) as archive:
    for f in [i for i in archive.infolist() if i.filename.startswith('udx/dem/') and i.filename.endswith('.gml')]:
     target=Terrain(wi);parser=ET.XMLParser(target=target)
     with archive.open(f) as stream:
      while b:=stream.read(1024*1024):parser.feed(b)
     parser.close();records.append({'archive':moga_source(p, ROOT),'member':f.filename,'crc32':f'{f.CRC:08x}','triangles':target.total,'road_ground_sample_hits':target.hits});print('Road TIN',p.parent.name,f.filename,target.hits,flush=True)
  write(cache,z.astype('float32'),ftf,np.nan)
  (OUT/'road_terrain_sources.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
 eligible=allowed&np.isfinite(z);road_count=eligible.reshape(nr,4,nc,4).sum((1,3));host_area=(1-a['building_fraction'])*400-road_count*25;host_mask=a['land']&(host_area>1);host=np.full((nr,nc),-1,dtype=np.int32);host[host_mask]=np.arange(host_mask.sum());Nhost=int(host_mask.sum());nz=a['z'][host_mask].tolist();area=host_area[host_mask].tolist();parent=np.flatnonzero(host_mask).tolist();ntype=[0]*Nhost;fine_node=np.full(road.shape,-1,dtype=np.int32);gate=np.full(road.shape,np.nan);components=[];bed_bias=[]
 # Keep distinct disconnected road pieces in the same20m square separate.
 for r,c in zip(*np.where(road_count>0)):
  sub=eligible[r*4:r*4+4,c*4:c*4+4];labs,num=label(sub);ground=z[r*4:r*4+4,c*4:c*4+4]
  for k in range(1,num+1):
   coords=list(zip(*np.where(labs==k)));node=len(nz);vals=np.array([ground[q] for q in coords]);bed=float(vals.mean());nz.append(bed);area.append(len(coords)*25);parent.append(int(r*nc+c));ntype.append(1);dist=np.full((4,4),np.inf);start=coords[int(np.argmin(vals))];dist[start]=ground[start];queue=[(dist[start],*start)]
   # Internal portal threshold preserves intervening humps on the original TIN.
   while queue:
    cost,rr,cc=heapq.heappop(queue)
    if cost>dist[rr,cc]+1e-8:continue
    for dr,dc in [(1,0),(-1,0),(0,1),(0,-1)]:
     y,x=rr+dr,cc+dc
     if 0<=y<4 and 0<=x<4 and labs[y,x]==k:
      new=max(cost,ground[y,x])
      if new<dist[y,x]:dist[y,x]=new;heapq.heappush(queue,(new,y,x))
   for rr,cc in coords:fine_node[r*4+rr,c*4+cc]=node;gate[r*4+rr,c*4+cc]=max(bed,dist[rr,cc])
   components.append({'node':node,'parent':int(r*nc+c),'host':int(host[r,c]),'area':len(coords)*25,'bed':bed,'internal_relief':float(vals.max()-vals.min()),'separate_components_in_cell':num});bed_bias.append(float(a['z'][r,c]-bed))
 nz=np.array(nz);area=np.array(area);parent=np.array(parent,dtype=np.int32);ntype=np.array(ntype,dtype='uint8');edges=[];kinds=[];width=[];crest=[];length=[];rough=[]
 def edge(s,d,w,cr,L,n,kind):
  edges.append((s,d));width.append(w);crest.append(max(cr,nz[s],nz[d]));length.append(max(L,5));rough.append(n);kinds.append(kind)
 # Original coarse non-road surface connections remain separate from road portals.
 for axis in [1,0]:
  l=(slice(None),slice(None,-1)) if axis==1 else (slice(None,-1),slice(None));r=(slice(None),slice(1,None)) if axis==1 else (slice(1,None),slice(None));hl=host[l];hr=host[r];k=(hl>=0)&(hr>=0)
  for s,d in zip(hl[k],hr[k]):edge(int(s),int(d),20*min(area[s],area[d])/400,max(nz[s],nz[d]),20,.05,0)
 for comp in components:
  if comp['host']>=0:edge(comp['host'],comp['node'],min(20,np.sqrt(comp['area'])),max(nz[comp['host']],comp['bed']),10,.04,1)
 # True5m four-neighbour corridor contacts across coarse-cell faces, never diagonals.
 contacts={}
 for axis in [1,0]:
  l=(slice(None),slice(None,-1)) if axis==1 else (slice(None,-1),slice(None));r=(slice(None),slice(1,None)) if axis==1 else (slice(1,None),slice(None));u=fine_node[l];v=fine_node[r];k=(u>=0)&(v>=0)&(u!=v);ul=u[k];vl=v[k];gg=np.maximum(gate[l][k],gate[r][k])
  for s,d,cr in zip(ul,vl,gg):
   key=(min(int(s),int(d)),max(int(s),int(d)),float(cr));contacts[key]=contacts.get(key,0)+5
 for (s,d,cr),w in contacts.items():edge(s,d,w,cr,20,.03,2)
 edges=np.array(edges,dtype=np.int32);kinds=np.array(kinds,dtype='uint8');parent=np.array(parent);nward=a['ward'].ravel()[parent];banknodes=[];bankstage=[];bankwidth=[];stages={r['id']:r['synthetic_stage_low_m'] for r in audit['synthetic_water_stages']}
 for r,c in zip(*np.where(host_mask)):
  if not a['land'][r,c]:continue
  for dr,dc in [(1,0),(-1,0),(0,1),(0,-1)]:
   rr,cc=r+dr,c+dc
   if 0<=rr<nr and 0<=cc<nc and a['candidate_water'][rr,cc]:
    h=stages.get(int(a['water'][rr,cc]))
    if h is not None:banknodes.append(int(host[r,c]));bankstage.append(h);bankwidth.append(20*area[host[r,c]]/400)
 # Rain volume assigned by parent-cell open areas; roof/nonroad catchment rain kept.
 node_area_grid=np.bincount(parent,weights=area,minlength=nr*nc).reshape(nr,nc);hasnode=node_area_grid>0
 from scipy.ndimage import distance_transform_edt
 near=distance_transform_edt(~hasnode,return_distances=False,return_indices=True);catch=a['domain']&np.isfinite(a['z'])&(a['water']==0);cw=a['ward'][catch];raw=np.where(cw==3,100.,np.where(cw==1,30.,np.where(cw==2,20.,40.)));norm=60/raw.mean();rain_coarse=np.bincount((near[0][catch]*nc+near[1][catch]).astype(int),weights=raw*norm/1000/3600*400,minlength=nr*nc);rain=rain_coarse[parent]*area/node_area_grid.ravel()[parent]
 np.savez_compressed(PRO/'road_graph.npz',z=nz,area=area,parent=parent,node_type=ntype,ward=nward,host_grid=host,fine_node=fine_node,edges=edges,width=np.array(width),crest=np.array(crest),length=np.array(length),roughness=np.array(rough),edge_kind=kinds,bank_nodes=np.array(banknodes,dtype=np.int32),bank_stage=np.array(bankstage),bank_width=np.array(bankwidth),rain_m3_s=rain)
 restored=(~a['land'])&(road_count>0);face_old_block=0;face_crest_lower=0
 for s,d,cr in zip(edges[kinds==2,0],edges[kinds==2,1],np.array(crest)[kinds==2]):
  ps,pd=parent[s],parent[d]
  if not a['land'].ravel()[ps] or not a['land'].ravel()[pd]:face_old_block+=1
  if max(a['z'].ravel()[ps],a['z'].ravel()[pd])-cr>.5:face_crest_lower+=1
 info={'date':'2026-10-14','method':'5m original-TIN road components nested in20m cells; separate components remain separate; minimax internal terrain humps retained in portal thresholds. No arbitrary channel lowering.','synthetic_rain_and_stage_unchanged':True,'grid_shape':[nr,nc],'transform':list(tf)[:6],'host_nodes':Nhost,'road_component_nodes':int((ntype==1).sum()),'road_faces':int(np.sum(kinds==2)),'host_road_faces':int(np.sum(kinds==1)),'fine_road_pixels':int(road.sum()),'road_roof_overlap_pixels_excluded':int(np.sum(road&roof)),'fine_road_eligible':int(allowed.sum()),'fine_road_missing_TIN':int(np.sum(allowed&~np.isfinite(z))),'reopened_coarse_cells_with_road':int(restored.sum()),'road_faces_previously_blocked_by_coarse_mask':face_old_block,'road_portals_more_than0_5m_below_old_center_threshold':face_crest_lower,'road_bed_old_center_difference_p05_p50_p95_m':np.quantile(bed_bias,[.05,.5,.95]).tolist(),'components_with_internal_relief_over2m':sum(x['internal_relief']>2 for x in components),'new_bank_outlets':0,'bank_note':'Only former host-bank faces retained; width reduced for separate road storage. New road nodes do not discharge directly to water.','node_bed_note':'Mean original5m TIN height within each connected road piece; constant-area storage is still a subgrid approximation, not surveyed road cross section.','grade_separation_note':'No bridge/underpass connections inferred; road-roof conflict and water-covered road pixels excluded. Elevated roads remain unresolved where these masks do not detect them.','terrain_epoch':'Same PLATEAU source, no newer DEM merged, no datum shift.','road_graph_boundaries_not_upstream_complete':True};(OUT/'connectivity_audit.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n');write(PRO/'reopened_cells.tif',restored.astype('uint8'),tf,0);print('GRAPH',json.dumps(info,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
