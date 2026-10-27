"""Batch face geometry and geometry-driven 10/5 m comparison preparation.
Uses unmodified, pinned official SFINCS grid and subgrid routines. No flow solver
is implemented. Original DEM, building obstacle and roughness remain unchanged.
"""
import os
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
os.environ['NUMBA_NUM_THREADS']='4'
from pathlib import Path
import json,hashlib,shutil,argparse,importlib.util
import numpy as np
import rasterio,netCDF4
from rasterio.windows import Window
from numba import njit,prange
from hydromt_sfincs.subgrid import subgrid_q_table,subgrid_v_table
from hydromt_sfincs.utils import write_geoms
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'data/processed/three_ward_adaptive_connectivity';CON=ROOT/'data/processed/three_ward_urban_constraints';OLD=ROOT/'data/processed/three_ward_sfincs_scenarios';FIX=ROOT/'data/processed/three_ward_fixed_runoff';BASE=ROOT/'data/processed/three_ward_sfincs_framework';RUN=BASE/'adaptive_connectivity_comparison';TEMPLATE=BASE/'fixed_runoff_comparison/urban_storage253_10m_template';NR,NC=20240,15400;LEFT,TOP=-11360.,-32320.;LEVELS=253

def dump(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def maps():return tuple(np.memmap(P/f'fine_{k}.bin',dtype=d,mode='r',shape=(NR,NC)) for k,d in [('z','float32'),('class','uint8'),('n','float32')])
def cache():
 P.mkdir(exist_ok=True);RUN.mkdir(exist_ok=True)
 if (P/'fine_cache.json').exists():return
 files=[CON/'hydraulic_obstacle_0_50m.tif',CON/'constraints_0_50m.tif',CON/'manning_0_50m.tif']
 for src,k,dtype in zip(files,['z','class','n'],['float32','uint8','float32']):
  out=np.memmap(P/f'fine_{k}.bin',dtype=dtype,mode='w+',shape=(NR,NC))
  with rasterio.open(src) as d:
   assert d.shape==(NR,NC) and d.transform.a==.5
   for r in range(0,NR,400):out[r:r+400]=d.read(1,window=Window(0,r,NC,min(400,NR-r)))
  out.flush();del out
 dump(P/'fine_cache.json',{'shape':[NR,NC],'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in files},'terrain_or_obstacle_modified':False})
 official=P/'official_source/quadtree_builder.py';dump(P/'official_source/manifest.json',{'repository':'https://github.com/Deltares-research/cht_sfincs','commit':'8d29dcc7d4138e75001ecce1708137791cc3eb6f','quadtree_builder_sha256':sha(official),'code_unmodified':True,'license':'GPL-3.0, copy alongside source','table_functions':'unmodified HydroMT-SFINCS1.2.2 subgrid_v_table and subgrid_q_table'})

@njit(parallel=True)
def calculate_faces(z,cl,n,rr,cc,direction,lengths,za,zb):
 nfaces=len(rr);data=np.empty((nfaces,3,LEVELS),np.float32);scalars=np.empty((nfaces,4),np.float32);width=np.empty(nfaces,np.float32);roads=np.empty(nfaces,np.float32)
 for j in prange(nfaces):
  count=lengths[j];zz=np.empty(2*count,np.float64);nn=np.empty(2*count,np.float64);op=rd=0
  for k in range(count):
   r=rr[j]+(k if direction[j]==0 else 0);c=cc[j]+(k if direction[j]==1 else 0)
   rb=r+(1 if direction[j]==1 else 0);ca=c-(1 if direction[j]==0 else 0)
   a=z[r,ca];b=z[rb,c];rough=max(n[r,ca],n[rb,c]);crest=max(a,b)
   zz[k]=crest;zz[k+count]=crest;nn[k]=rough;nn[k+count]=rough
   if cl[r,ca]!=1 and cl[rb,c]!=1:op+=1
   if cl[r,ca]==2 and cl[rb,c]==2:rd+=1
  a,b,h,nrep,pwet,ffit,navg,_=subgrid_q_table(zz,nn,LEVELS,.01,2,za[j],zb[j],'min')
  scalars[j]=np.array([a,b,ffit,navg]);data[j,0]=h;data[j,1]=nrep;data[j,2]=pwet;width[j]=op*.5;roads[j]=rd*.5
 return scalars,data,width,roads

def face_coordinates(rr,cc,direction,lengths):
 x=LEFT+cc*.5;y=TOP-rr*.5;length=lengths*.5
 # direction0: east face, at column cc; direction1: northward index face
 y=np.where(direction==1,y-.5,y-length/2);x=np.where(direction==1,x+length/2,x)
 return x,y,length

def write_dams(folder,rr,cc,di,ll,width,interior):
 x,y,length=face_coordinates(rr,cc,di,ll);features=[]
 for j in np.flatnonzero((width==0)&interior):
  half=length[j]/2-.001
  features.append({'name':f'BuildingClosure{j:07d}','x':[x[j],x[j]] if di[j]==0 else [x[j]-half,x[j]+half],'y':[y[j]-half,y[j]+half] if di[j]==0 else [y[j],y[j]]})
 write_geoms(folder/'building_closures.thd',features,stype='thd',fmt='%.6f');return len(features)

def base_faces(mask):
 # Fortran active-face ordering: per cell east then north; columns outermost,
 # rows south-to-north. Matches the pinned official regular-grid table order.
 rs=[];cs=[];ds=[];ia=[];ib=[];south=np.flipud(mask);indices=np.full(south.shape,-1,np.int32);ar=np.flatnonzero(south.ravel(order='F')>0);rr,cc=np.unravel_index(ar,south.shape,order='F');indices[rr,cc]=np.arange(len(ar))
 for r,c in zip(rr,cc):
  for dr,dc,di in [(0,1,0),(1,0,1)]:
   rb,cb=r+dr,c+dc
   if rb>=south.shape[0] or cb>=south.shape[1] or south[rb,cb]==0:continue
   rn=south.shape[0]-1-r;rs.append(rn*20 if di==0 else rn*20-1);cs.append((c+1)*20 if di==0 else c*20);ds.append(di);ia.append(indices[r,c]);ib.append(indices[rb,cb])
 return [np.array(v,dtype=np.int32) for v in [rs,cs,ds,ia,ib]]

def regular():
 cache();folder=RUN/'corrected_10m_template';folder.mkdir(exist_ok=True)
 if (folder/'builder_complete.json').exists():print('Reuse corrected10m');return
 for p in TEMPLATE.iterdir():
  if p.suffix in ['.inp','.ind','.msk','.obs','.crs','.dep'] or p.name=='flux_sections.json':shutil.copy2(p,folder/p.name)
 shutil.copyfile(TEMPLATE/'sfincs_subgrid.nc',folder/'sfincs_subgrid.nc')
 with rasterio.open(CON/'active_mask_10m.tif') as d:m=d.read(1);tf=d.transform
 rr,cc,di,ia,ib=base_faces(m);ll=np.full(len(rr),20,np.int32);z,cl,n=maps()
 with netCDF4.Dataset(folder/'sfincs_subgrid.nc','r+') as d:
  assert len(d.dimensions['npuv'])==len(rr),'Official face order/count mismatch'
  zm=np.asarray(d['z_zmin'][:]);za=zm[ia];zb=zm[ib]
  width=np.empty(len(rr),np.float32);roads=np.empty_like(width)
  for j in range(0,len(rr),20000):
   k=min(j+20000,len(rr));s,v,w,road=calculate_faces(z,cl,n,rr[j:k],cc[j:k],di[j:k],ll[j:k],za[j:k],zb[j:k]);assert np.isfinite(v).all() and np.isfinite(s).all();width[j:k]=w;roads[j:k]=road
   for col,name in enumerate(['uv_zmin','uv_zmax','uv_ffit','uv_navg']):d[name][j:k]=s[:,col]
   for col,name in enumerate(['uv_havg','uv_nrep','uv_pwet']):d[name][j:k]=v[:,col,:]
   print('Correct10m faces',k,'/',len(rr),flush=True)
  d.setncattr('face_geometry','Native 0.50m paired face cross-sections; crest=max adjacent original hydraulic elevations; highest paired roughness; official253-level q routine; DEM unchanged')
 interior=(m[rr//20,np.minimum(cc//20,m.shape[1]-1)]==1) # select proper adjacent masks below
 south=np.flipud(m);flat=south.ravel(order='F');cells=flat[flat>0];interior=(cells[ia]==1)|(cells[ib]==1)
 blocked=write_dams(folder,rr,cc,di,ll,width,interior)
 s=(folder/'sfincs.inp').read_text();s+='\nthdfile = building_closures.thd\n';(folder/'sfincs.inp').write_text(s)
 with rasterio.open(CON/'subcell_connectivity_flags_10m.tif') as d:flags=d.read(1)
 refine=(flags>0)&(m==1)
 # Also refine native narrow ground-road crossings (<3m), independent of floods.
 x,y,_=face_coordinates(rr,cc,di,ll);row=((TOP-y)/10).astype(int);col=((x-LEFT)/10).astype(int)
 narrow=(roads>0)&(width<3.)&interior
 for dr,dc in [(0,0),(0,-1),(-1,0)]:
  rn=np.clip(row[narrow]+dr,0,m.shape[0]-1);cn=np.clip(col[narrow]+dc,0,m.shape[1]-1);refine[rn,cn]=True
 refine&=m==1
 with rasterio.open(P/'automatic_refinement_10m.tif','w',driver='GTiff',height=m.shape[0],width=m.shape[1],count=1,dtype='uint8',crs=6677,transform=tf,compress='deflate',tiled=True) as d:d.write(refine.astype('uint8'),1)
 np.savez_compressed(P/'regular_face_geometry.npz',row_fine=rr,col_fine=cc,direction=di,length_pixels=ll,native_open_width_m=width,native_road_width_m=roads)
 rec={'case':'corrected10m','cells':int((m>0).sum()),'faces':len(rr),'fully_building_closed_faces':blocked,'automatic_refinement_parent_cells':int(refine.sum()),'original_storage_tables_preserved':True,'nlevels':LEVELS,'source_sha256':json.loads((P/'fine_cache.json').read_text())['source_sha256'],'method':'Recompute face tables from paired native0.50m cross-sections with unmodified official q routine; fully building-closed faces use official thin dams; geometry-only refinement selection.'}
 dump(folder/'builder_complete.json',rec);print(json.dumps(rec),flush=True)

def official_grid():
 spec=importlib.util.spec_from_file_location('udt_official_quadtree_builder',P/'official_source/quadtree_builder.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

@njit(parallel=True)
def child_storage(z,rr,cc,factors):
 n=len(rr);sc=np.empty((n,3),np.float32);levels=np.empty((n,LEVELS),np.float32)
 for j in prange(n):
  f=factors[j];block=z[rr[j]:rr[j]+f,cc[j]:cc[j]+f].copy().reshape(f*f)
  zz,v,a,b=subgrid_v_table(block,.5,.5,LEVELS,-20.,99999.);sc[j]=np.array([a,b,v[-1]]);levels[j]=zz
 return sc,levels

def adaptive():
 from pyproj import CRS
 cache();folder=RUN/'adaptive_10_5m_template';folder.mkdir(exist_ok=True)
 if (folder/'builder_complete.json').exists():print('Reuse adaptive10/5m');return
 with rasterio.open(CON/'active_mask_10m.tif') as d:m10=d.read(1)
 with rasterio.open(P/'automatic_refinement_10m.tif') as d:refine=d.read(1)>0
 z,cl,n=maps();module=official_grid();g=module.QuadtreeGrid();g.build(LEFT,TOP-NR*.5,NR//20,NC//20,10.,10.,0.,CRS.from_epsg(6677),None,None,None)
 ids=np.flatnonzero(refine[NR//20-1-g.n,g.m]);g.refine_cells(ids,0)
 f=(20//(2**g.level)).astype(np.int32);rn=NR-(g.n+1)*f;cn=g.m*f;parent_r=rn//20;parent_c=cn//20;mask=m10[parent_r,parent_c].copy()
 for lev in [0,1]:
  selected=np.flatnonzero((g.level==lev)&(mask>0));ff=20//(2**lev)
  for j in selected:
   if not np.any(cl[rn[j]:rn[j]+ff,cn[j]:cn[j]+ff]!=1):mask[j]=0
 g.mask=mask;g.get_neighbors();g.get_uv_points();g.to_xugrid();g.cut_inactive_cells();g.compute_cell_center_coordinates()
 f=(20//(2**g.level)).astype(np.int32);rn=NR-(g.n+1)*f;cn=g.m*f;mask=g.mask.astype('uint8')
 # Old regular cell indices in exactly the original Fortran active order.
 lookup=np.full(m10.shape,-1,np.int32);ii=np.flatnonzero(np.flipud(m10).ravel(order='F')>0);rs,cs=np.unravel_index(ii,m10.shape,order='F');lookup[m10.shape[0]-1-rs,cs]=np.arange(len(ii))
 dest=folder/'sfincs_subgrid.nc'
 with netCDF4.Dataset(RUN/'corrected_10m_template/sfincs_subgrid.nc') as old,netCDF4.Dataset(dest,'w') as out:
  out.createDimension('np',g.nr_cells);out.createDimension('npuv',g.nr_uv_points);out.createDimension('levels',LEVELS);out.setncattr('version','1.0');out.setncattr('provenance','Official pinned CHT quadtree geometry; unchanged HydroMT-SFINCS1.2.2 subgrid routines on native0.50m samples. Coarse cell storage copied exactly from matched10m baseline.')
  for key in ['z_zmin','z_zmax','z_volmax']:out.createVariable(key,'f4',('np',))
  out.createVariable('z_level','f4',('np','levels'))
  for lev in [0,1]:
   inds=np.flatnonzero(g.level==lev)
   for j in range(0,len(inds),20000):
    sel=inds[j:j+20000]
    if lev==0:
     ids=lookup[rn[sel]//20,cn[sel]//20];assert np.all(ids>=0)
     for key in ['z_zmin','z_zmax','z_volmax','z_level']:out[key][sel]=old[key][ids]
    else:
     sc,zl=child_storage(z,rn[sel],cn[sel],f[sel]);assert np.isfinite(sc).all() and np.isfinite(zl).all()
     for col,key in enumerate(['z_zmin','z_zmax','z_volmax']):out[key][sel]=sc[:,col]
     out['z_level'][sel]=zl
    print('Adaptive storage level',lev,j+len(sel),'/',len(inds),flush=True)
  zm=np.asarray(out['z_zmin'][:]);g.z=zm.copy()
  a=g.uv_index_z_nm;b=g.uv_index_z_nmu;di=g.uv_dir.astype(np.int32);size=f*.5;length=np.minimum(size[a],size[b]);fineA=size[a]<=size[b];xsmall=np.where(fineA,g.x[a],g.x[b]);ysmall=np.where(fineA,g.y[a],g.y[b]);xf=np.where(di==0,g.x[a]+size[a]/2,xsmall);yf=np.where(di==0,ysmall,g.y[a]+size[a]/2)
  rr=np.rint((TOP-np.where(di==0,yf+length/2,yf))/.5).astype(np.int32)-np.where(di==1,1,0);cc=np.rint((np.where(di==0,xf,xf-length/2)-LEFT)/.5).astype(np.int32);ll=np.rint(length/.5).astype(np.int32)
  for key in ['uv_zmin','uv_zmax','uv_ffit','uv_navg']:out.createVariable(key,'f4',('npuv',))
  for key in ['uv_havg','uv_nrep','uv_pwet']:out.createVariable(key,'f4',('npuv','levels'))
  widths=np.empty(len(a),np.float32);roadwidth=np.empty_like(widths)
  for j in range(0,len(a),20000):
   k=min(j+20000,len(a));sc,v,w,rd=calculate_faces(z,cl,n,rr[j:k],cc[j:k],di[j:k],ll[j:k],zm[a[j:k]],zm[b[j:k]]);assert np.isfinite(sc).all() and np.isfinite(v).all();widths[j:k]=w;roadwidth[j:k]=rd
   for col,key in enumerate(['uv_zmin','uv_zmax','uv_ffit','uv_navg']):out[key][j:k]=sc[:,col]
   for col,key in enumerate(['uv_havg','uv_nrep','uv_pwet']):out[key][j:k]=v[:,col,:]
   print('Adaptive faces',k,'/',len(a),flush=True)
 blocked=write_dams(folder,rr,cc,di,ll,widths,(mask[a]==1)|(mask[b]==1))
 # Assign data on the already verified official topology, without rebuilding it.
 g.data['z'].values[:]=g.z;g.data.ugrid.to_dataset().to_netcdf(folder/'sfincs.nc')
 np.savez_compressed(P/'adaptive_cell_geometry.npz',row_fine=rn,col_fine=cn,size_m=size,mask=mask,level=g.level,x=g.x,y=g.y)
 np.savez_compressed(P/'adaptive_face_geometry.npz',row_fine=rr,col_fine=cc,direction=di,length_pixels=ll,cell_a=a,cell_b=b,native_open_width_m=widths,native_road_width_m=roadwidth)
 # Same numerical settings as the regular10m baseline, with quadtree input.
 shutil.copy2(RUN/'corrected_10m_template/sfincs.inp',folder/'sfincs.inp')
 import sys
 sys.path.insert(0,str(ROOT/'src/exp'));from run_three_ward_adaptive_connectivity import config
 config(folder/'sfincs.inp',{'qtrfile':'sfincs.nc','sbgfile':'sfincs_subgrid.nc','thdfile':'building_closures.thd'},remove=['depfile','mskfile','indexfile'])
 # Geometry-only residual screen: no water-depth-based selection or surveys.
 from scipy.ndimage import label
 flagged_area=0.;open_area=0.;flagged_cells=0
 for j in np.flatnonzero(mask==1):
  ff=f[j];block=cl[rn[j]:rn[j]+ff,cn[j]:cn[j]+ff];op=block!=1;area=float(op.sum()*.25);open_area+=area
  if not np.any(block==1):continue
  lab,num=label(op)
  if num<2:continue
  edges=np.unique(np.r_[lab[0],lab[-1],lab[:,0],lab[:,-1]])
  if np.count_nonzero(edges)>1:flagged_cells+=1;flagged_area+=area
 rec={'case':'adaptive10/5m','cells':g.nr_cells,'faces':g.nr_uv_points,'cells_by_resolution_m':{str(int(v)):int((size==v).sum()) for v in [10.,5.]},'fully_building_closed_faces':blocked,'automatic_refinement_parent_cells':int(refine.sum()),'nlevels':LEVELS,'raw_fine_sampling_m':.5,'residual_multiple_component_cells':flagged_cells,'open_area_inside_residual_flagged_cells_km2':flagged_area/1e6,'common_open_area_km2':open_area/1e6,'raw_terrain_modified':False,'coarse_storage_identical_to_baseline':True,'official_grid_code':json.loads((P/'official_source/manifest.json').read_text())}
 dump(folder/'builder_complete.json',rec);print(json.dumps(rec),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['regular','adaptive','cache']);args=ap.parse_args();{'regular':regular,'adaptive':adaptive,'cache':cache}[args.action]()
