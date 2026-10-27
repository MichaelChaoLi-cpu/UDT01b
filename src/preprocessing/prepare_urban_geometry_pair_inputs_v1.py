"""Prepare matched terrain-only/urban input interfaces without invoking a solver.
Preserves all frozen experiments. No training, sealed responses or new storms.
Usage: project Python ... [--build-tables with the existing pinned builder Python].
"""
from pathlib import Path
import os,sys,json,csv,hashlib,argparse,shutil
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
os.environ.setdefault('NUMBA_NUM_THREADS','4')
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_origin
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'data/processed/three_ward_urban_geometry_pair_inputs_v1';R=ROOT/'data/exp/data-preprocessing/three_ward_urban_geometry_pair_inputs_v1'
BASE=ROOT/'data/processed/three_ward_sfincs_scenarios';CON=ROOT/'data/processed/three_ward_urban_constraints';LIB=ROOT/'data/processed/three_ward_minimum_scenarios_v1';PLAN=ROOT/'data/exp/research-design/three_ward_urban_geometry_value_plan_v1/protocol.json'
U=ROOT/'data/processed/three_ward_reference_setup_pilot_v1/qualification/runs/rain60_raised__exact_rain_10m'
NR,NC=1012,770;NF=20;LEVELS=253
CASES=['rain60_low','rain100_low','rain60_raised','dry_raised']
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from load_three_ward_minimum_scenarios import load_forcing,find_case

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvwrite(p,rows):
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sources():
 return [PLAN,BASE/'terrain_sampling_0_50m.tif',BASE/'terrain_source_id_0_50m.tif',BASE/'active_mask_20m.tif',CON/'constraints_0_50m.tif',CON/'active_mask_10m.tif',CON/'hydraulic_obstacle_0_50m.tif',CON/'manning_0_50m.tif',LIB/'shared_geometry.npz',ROOT/'src/preprocessing/reference_label_policy.py',ROOT/'data/processed/three_ward_rain_step_integral_v1/solver_binding.json',U/'sfincs.inp',U/'sfincs.ind',U/'sfincs.msk',U/'sfincs_subgrid.nc',U/'udt_run_identity.json']+[LIB/find_case(c)['forcing_file'] for c in CASES]
def masks():
 with rasterio.open(BASE/'active_mask_20m.tif') as d:t=np.repeat(np.repeat(d.read(1),2,0),2,1)
 with rasterio.open(CON/'active_mask_10m.tif') as d:u=d.read(1)
 return t,u

def preflight():
 P.mkdir(exist_ok=True);R.mkdir(parents=True,exist_ok=True)
 if (P/'preflight.json').exists():
  a=json.loads((P/'source_bindings.json').read_text());assert all(sha(ROOT/x['file'])==x['sha256'] for x in a);print('REUSE audited geometry support',flush=True);return
 bindings=[{'file':str(x.relative_to(ROOT)),'sha256':sha(x)} for x in sources()];dump(P/'source_bindings.json',bindings)
 t,u=masks();assert t.shape==u.shape==(NR,NC);assert np.array_equal(t==2,u==2);assert np.all((u==0)|(u==t))
 index=np.flatnonzero(np.flipud(t).ravel(order='F')>0);rs,cs=np.unravel_index(index,t.shape,order='F');rn=NR-1-rs;restore=(t==1)&(u==0);lookup=np.full(t.shape,-1,np.int32);lookup[rn,cs]=np.arange(len(index));np.r_[len(index),index+1].astype('<i4').tofile(P/'terrain_native.ind');np.flipud(t).ravel(order='F')[index].astype('u1').tofile(P/'terrain_native.msk')
 ground=np.zeros(t.shape,dtype='f8');roof=np.zeros_like(ground);road=np.zeros_like(ground);missing=np.zeros(t.shape,'i4');minz=np.full(t.shape,np.nan,'f4');maxz=minz.copy();meanz=minz.copy();primary=np.zeros_like(ground);fallback=np.zeros_like(ground);diffmax=0.;obsdiff=0.;roadroughmax=0.;uncertain=np.zeros_like(ground)
 with rasterio.open(BASE/'terrain_sampling_0_50m.tif') as dem,rasterio.open(CON/'constraints_0_50m.tif') as cl,rasterio.open(BASE/'terrain_source_id_0_50m.tif') as src,rasterio.open(CON/'hydraulic_obstacle_0_50m.tif') as ob,rasterio.open(CON/'manning_0_50m.tif') as rough:
  assert dem.shape==(NR*20,NC*20) and dem.transform==from_origin(-11360,-32320,.5,.5)
  for row in range(0,NR,24):
   count=min(24,NR-row);win=Window(0,row*20,NC*20,count*20);z=dem.read(1,window=win);c=cl.read(1,window=win);s=src.read(1,window=win);h=ob.read(1,window=win);nn=rough.read(1,window=win)
   def blocks(a):return a.reshape(count,20,NC,20).transpose(0,2,1,3).reshape(count,NC,400)
   zz=blocks(z);cc=blocks(c);ss=blocks(s);active=t[row:row+count]>0;interior=t[row:row+count]==1;finite=np.isfinite(zz);missing[row:row+count]=(~finite).sum(-1);minz[row:row+count]=np.nanmin(zz,axis=-1);maxz[row:row+count]=np.nanmax(zz,axis=-1);meanz[row:row+count]=np.nanmean(zz,axis=-1)
   ground[row:row+count]=((cc!=1)&finite).sum(-1)*.25*interior;roof[row:row+count]=((cc==1)&finite).sum(-1)*.25*interior;road[row:row+count]=((cc==2)&finite).sum(-1)*.25*interior;uncertain[row:row+count]=((cc==3)&finite).sum(-1)*.25*interior;primary[row:row+count]=(ss==1).sum(-1)*.25*interior;fallback[row:row+count]=(ss==2).sum(-1)*.25*interior
   af=np.repeat(np.repeat(active,20,0),20,1);d=h-z;outside_build=(c!=1)&af&np.isfinite(d);diffmax=max(diffmax,float(np.max(abs(d[outside_build]),initial=0)));bb=(c==1)&np.repeat(np.repeat(interior,20,0),20,1)&np.isfinite(d);obsdiff=max(obsdiff,float(np.max(abs(d[bb]-10),initial=0)));roadroughmax=max(roadroughmax,float(np.max(abs(nn[(c==2)&af]-.03),initial=0)))
   print('AUDIT terrain rows',row+count,'/',NR,flush=True)
 assert int(missing[t>0].sum())==0,'Hold: missing original terrain under active/restored cells; never fill with zero'
 total=float((t==1).sum()*100);g=np.load(LIB/'shared_geometry.npz');assert np.array_equal(ground,g['Ground Area 10m']);assert abs(float(g['Contributing Area 10m'].sum())-total)<1e-6;assert abs(float(roof.sum()+ground.sum())-total)<1e-6
 # Terrain forcing is rain on the original catchment, never U recipient weights.
 terrain_weight=(t==1).astype('f4');urban_weight=g['Effective Rainfall Weight 10m'];assert np.all(urban_weight[u!=1]==0);assert abs(float(urban_weight.sum(dtype='f8')*100)-total)<1e-6
 np.savez_compressed(P/'paired_support_and_native_mapping.npz',**{'Terrain Active Mask':t,'Urban Active Mask':u,'Terrain Native Flat Index Zero Based':index,'Terrain Native Row North':rn,'Terrain Native Column East':cs,'Terrain Native Lookup':lookup,'Terrain Native Minimum Elevation':minz[rn,cs],'Terrain Native Mean Elevation':meanz[rn,cs],'Ground Area':ground,'Roof Area':roof,'Ground Road Area':road,'Uncertain Structure Area':uncertain,'Restored Terrain Cells':restore,'Terrain Rainfall Weight':terrain_weight,'Urban Rainfall Weight':urban_weight,'Original DEM Primary Area':primary,'PLATEAU Fallback Area':fallback})
 from netCDF4 import Dataset
 forcing_rows=[]
 for case in CASES:
  d=load_forcing(case);c=find_case(case);f=P/'terrain_forcing'/case;f.mkdir(parents=True,exist_ok=True);w=np.flipud(np.pad(terrain_weight,1));times=d['Event Time'];rates=d['Physical Rainfall Rate'];encoded_rates=[]
  with Dataset(f/'terrain_surface_precip.nc','w',format='NETCDF4') as ds:
   ny,nx=w.shape
   for k,n in [('time',len(times)),('y',ny),('x',nx)]:ds.createDimension(k,n)
   tv=ds.createVariable('time','f8',('time',));tv.units='minutes since 2020-01-01 00:00:00';tv[:]=times/60
   for key,vals in [('x',-11370+np.arange(nx)*10),('y',-42450+np.arange(ny)*10)]:v=ds.createVariable(key,'f8',(key,));v.units='m';v[:]=vals
   pr=ds.createVariable('Precipitation','f4',('time','y','x'),fill_value=-9999.,zlib=True,complevel=2,chunksizes=(1,256,256));pr.units='mm h-1';pr.unit='mm.hr-1';ds.coordinate_convention='Pinned Galibier corners; reader adds half cell; not generic CF cell centers';ds.forcing_sha256=c['forcing_sha256'];ds.forcing_role='terrain-only uniform physical rain on complete shared catchment; no roof recipients'
   for i,rate in enumerate(rates):
    field=(w*rate).astype('f4');pr[i]=field;encoded_rates.append(float(field.sum(dtype='f8')*100/3.6e6))
  stage=g['Boundary Low Stage'][None]+d['River Stage Increment By Group'][:,g['Boundary Group Index']];np.savetxt(f/'sfincs.bzs',np.column_stack([times,stage]),fmt='%.10f');shutil.copyfile(U/'sfincs.bnd',f/'sfincs.bnd')
  nominal=float(np.trapezoid(rates,times)*total/3.6e6);encoded=float(np.trapezoid(encoded_rates,times));urbanrates=urban_weight.sum(dtype='f8')*100*rates/3.6e6;urbanencoded=float(np.trapezoid(urbanrates,times));assert abs(encoded-nominal)<=max(1,nominal)*1e-6 and abs(urbanencoded-nominal)<=max(1,nominal)*1e-6
  forcing_rows.append({'case_id':case,'parent_id':c['parent_id'],'split':c['split'],'original_semantic_forcing_sha256':c['forcing_sha256'],'catchment_m2':total,'original_physical_depth_mm':float(np.trapezoid(rates,times)/3600),'nominal_rain_volume_m3':nominal,'terrain_encoded_volume_m3':encoded,'urban_ideal_allocation_volume_m3':urbanencoded,'terrain_minus_nominal_m3':encoded-nominal,'time_knots':len(times),'fractional_minute_knots':int(np.count_nonzero(times/60!=np.floor(times/60))),'not_a_new_parent':True,'terrain_forcing_file':str((f/'terrain_surface_precip.nc').relative_to(ROOT))})
 csvwrite(R/'paired_forcing_volume_check.csv',forcing_rows)
 # Fixed grid identities; no copied native U indices for restored T cells.
 faces_from=[];faces_to=[];face_dir=[];nativeS=np.full(t.shape,-1,'i4');nativeS[rs,cs]=np.arange(len(index));south=np.flipud(t)
 for r,c in zip(rs,cs):
  for dr,dc,di in [(0,1,0),(1,0,1)]:
   rr=r+dr;cc=c+dc
   if rr<NR and cc<NC and south[rr,cc]>0:faces_from.append(nativeS[r,c]);faces_to.append(nativeS[rr,cc]);face_dir.append(di)
 a=np.array(faces_from,'i4');b=np.array(faces_to,'i4');di=np.array(face_dir,'i1');m=south[rs,cs];receiver=((m[a]==1)&(m[b]==2))|((m[a]==2)&(m[b]==1));np.savez_compressed(P/'terrain_face_topology.npz',from_native=a,to_native=b,direction=di,boundary_face=receiver,native_row_south=rs,native_col_east=cs)
 summary={'source_design':'MILI-D-20261016-013','status':'interfaces_prepared_tables_pending','scope':'Whole three wards with unchanged provisional contributing domain','grid_shape':[NR,NC],'dx_m':10,'physical_catchment_m2':total,'common_open_ground_m2':float(ground.sum()),'urban_roof_m2':float(roof.sum()),'urban_road_ground_m2':float(road.sum()),'uncertain_structure_ground_m2':float(uncertain.sum()),'terrain_active_cells':len(index),'urban_active_cells':int((u>0).sum()),'terrain_interior_cells':int((t==1).sum()),'urban_interior_cells':int((u==1).sum()),'restored_footprint_cells':int(restore.sum()),'restored_cell_plan_area_m2':int(restore.sum())*100,'terrain_native_faces':len(a),'terrain_receiver_faces':int(receiver.sum()),'finite_missing_active_fine_samples':int(missing[t>0].sum()),'restored_cell_source_zmin_m':float(minz[restore].min()),'restored_cell_source_zmax_m':float(maxz[restore].max()),'unchanged_boundary_cells':int((t==2).sum()),'nonbuilding_hydraulic_terrain_difference_max_m':diffmax,'assumed_building_plus10m_error_max_m':obsdiff,'road_n0p03_error_max':roadroughmax,'forcing_cases':CASES,'actual_solver_runs':0,'tables_built':False,'training':False,'sealed_tests':False,'new_historical_gridded_rain_acquired':False,'not_an_accuracy_comparison':True}
 dump(P/'preflight.json',summary);print(json.dumps(summary,indent=2),flush=True)

def build_tables():
 from numba import njit,prange
 from hydromt_sfincs.subgrid import subgrid_v_table,subgrid_q_table
 from netCDF4 import Dataset
 import hydromt_sfincs,inspect
 assert hydromt_sfincs.__version__=='1.2.2';preflight();folder=P/'terrain_template';folder.mkdir(exist_ok=True)
 if (folder/'table_build_receipt.json').exists():print('REUSE completed T subgrid',flush=True);return
 z=np.load(P/'paired_support_and_native_mapping.npz');m=z['Terrain Active Mask'];index=z['Terrain Native Flat Index Zero Based'];rn=z['Terrain Native Row North'];cs=z['Terrain Native Column East'];lookup=z['Terrain Native Lookup'];top=np.load(P/'terrain_face_topology.npz');a=top['from_native'];b=top['to_native'];di=top['direction'];native_min=z['Terrain Native Minimum Elevation'];nf=len(a)
 @njit(parallel=True)
 def cells(blocks,ids):
  v=np.empty((len(ids),LEVELS),'f8');s=np.empty((len(ids),3),'f4')
  for j in prange(len(ids)):
   zz,vv,mi,ma=subgrid_v_table(blocks[ids[j]],.5,.5,LEVELS,-20.,99999.);v[j]=zz;s[j]=np.array([mi,ma,vv[-1]])
  return s,v
 @njit(parallel=True)
 def faces(dem,row,col,direction,mina,minb):
  s=np.empty((len(row),4),'f8');v=np.empty((len(row),3,LEVELS),'f8')
  for j in prange(len(row)):
   zz=np.empty(40,'f8');nn=np.full(40,.04,'f8')
   for k in range(20):
    r=row[j]+(k if direction[j]==0 else 0);c=col[j]+(k if direction[j]==1 else 0)
    zz[k]=max(dem[r,c-(1 if direction[j]==0 else 0)],dem[r+(1 if direction[j]==1 else 0),c]);zz[k+20]=zz[k]
   mi,ma,h,n,p,ff,navg,_=subgrid_q_table(zz,nn,LEVELS,.01,2,mina[j],minb[j],'min');s[j]=np.array([mi,ma,ff,navg]);v[j,0]=h;v[j,1]=n;v[j,2]=p
  return s,v
 dest=folder/'sfincs_subgrid.nc';assert not dest.exists(),'Retain interrupted table build; do not overwrite'
 # Store as source model schema (float64 table knots, float32 cell scalars).
 with Dataset(dest,'w') as ds:
  ds.createDimension('np',len(index));ds.createDimension('npuv',nf);ds.createDimension('levels',LEVELS)
  for key in ['z_zmin','z_zmax','z_volmax']:ds.createVariable(key,'f4',('np',))
  ds.createVariable('z_level','f8',('np','levels'))
  for key in ['uv_zmin','uv_zmax','uv_ffit','uv_navg']:ds.createVariable(key,'f8',('npuv',))
  for key in ['uv_havg','uv_nrep','uv_pwet']:ds.createVariable(key,'f8',('npuv','levels'))
  ds.provenance='Unmodified pinned HydroMT-SFINCS1.2.2 v/q functions; complete original bare terrain;253 levels;0.5m samples;paired face crest;constant Manning0.04;no building/road/rain-recipient information'
  with rasterio.open(BASE/'terrain_sampling_0_50m.tif') as dem:
   for row in range(0,NR,24):
    n=min(24,NR-row);ar=dem.read(1,window=Window(0,row*20,NC*20,n*20));block=np.ascontiguousarray(ar.reshape(n,20,NC,20).transpose(0,2,1,3).reshape(n*NC,400));ids=np.flatnonzero(m[row:row+n].ravel()>0);target=lookup[row:row+n].ravel()[ids];sc,v=cells(block,ids);assert np.isfinite(v).all() and np.isfinite(sc).all();order=np.argsort(target);target=target[order]
    for j,key in enumerate(['z_zmin','z_zmax','z_volmax']):ds[key][target]=sc[order,j]
    ds['z_level'][target]=v[order];print('T STORAGE',row+n,'/',NR,flush=True)
   # Direct original DEM memmap to avoid repeatedly decompressing face windows.
   cache=P/'terrain_fine_cache.bin';assert not cache.exists(),'Retain incomplete fine cache'
   zz=np.memmap(cache,dtype='f4',mode='w+',shape=(NR*20,NC*20))
   for row in range(0,NR*20,480):zz[row:row+480]=dem.read(1,window=Window(0,row,NC*20,min(480,NR*20-row)))
   zz.flush();fr=np.where(di==0,rn[a]*20,rn[a]*20-1).astype('i4');fc=np.where(di==0,(cs[a]+1)*20,cs[a]*20).astype('i4')
   for start in range(0,nf,12000):
    end=min(nf,start+12000);s,v=faces(zz,fr[start:end],fc[start:end],di[start:end],native_min[a[start:end]],native_min[b[start:end]]);assert np.isfinite(s).all() and np.isfinite(v).all()
    for j,key in enumerate(['uv_zmin','uv_zmax','uv_ffit','uv_navg']):ds[key][start:end]=s[:,j]
    for j,key in enumerate(['uv_havg','uv_nrep','uv_pwet']):ds[key][start:end]=v[:,j]
    print('T FACES',end,'/',nf,flush=True)
   del zz
   cache.unlink() # Disposable derived cache only; original terrain preserved.
 # Generate own native metadata and boundary crosssections; no U-native IDs reused.
 for src,name in [(P/'terrain_native.ind','sfincs.ind'),(P/'terrain_native.msk','sfincs.msk')]:shutil.copyfile(src,folder/name)
 z['Terrain Native Mean Elevation'].astype('<f4').tofile(folder/'sfincs.dep')
 for name in ['sfincs.obs','sfincs.bnd']:shutil.copyfile(U/name,folder/name)
 config=(U/'sfincs.inp').read_text();config='\n'.join(line for line in config.splitlines() if not line.strip().startswith('thdfile'));config=config.replace('roof_surface_precip.nc','terrain_surface_precip.nc');(folder/'sfincs.inp').write_text(config+'\n')
 boundary_groups=np.zeros(m.shape,'i4');g=np.load(LIB/'shared_geometry.npz');points=g['Boundary Point Position'];rr=np.rint((-32320-points[:,1])/10-.5).astype(int);cc=np.rint((points[:,0]+11360)/10-.5).astype(int);codes=g['Boundary Group Codes'][g['Boundary Group Index']];boundary_groups[rr,cc]=codes;assert np.all(boundary_groups[m==2]>0)
 widths=np.full((len(index),4),np.nan);mins=widths.copy();roads=widths.copy()
 with Dataset(dest) as ds:facez=np.asarray(ds['uv_zmin'][:])
 for face,d in enumerate(di):
  eastnorth=1 if d==0 else 3;westsouth=0 if d==0 else 2;aa=a[face];bb=b[face];widths[aa,eastnorth]=widths[bb,westsouth]=10.;roads[aa,eastnorth]=roads[bb,westsouth]=0.;mins[aa,eastnorth]=mins[bb,westsouth]=facez[face]
 header=['native_cell_index','row_south_one_based','col_east_one_based','kcs','water_group_code']+[d+'_'+v for v in ['open_width_m','road_width_m','face_minimum_m'] for d in ['west','east','south','north']]
 with (folder/'udt_negative_native_map.csv').open('w') as f:
  w=csv.writer(f);w.writerow(header)
  for i in range(len(index)):w.writerow([i+1,NR-rn[i],cs[i]+1,int(m[rn[i],cs[i]]),int(boundary_groups[rn[i],cs[i]]),*widths[i],*roads[i],*mins[i]])
 (folder/'udt_negative_trace.request').write_text('schema=negative-cell-face-v1\n')
 lines=[];sections=[]
 for k in np.flatnonzero(top['boundary_face']):
  land=a[k] if m[rn[a[k]],cs[a[k]]]==1 else b[k];water=b[k] if land==a[k] else a[k];lx=-11360+(cs[land]+.5)*10;ly=-32320-(rn[land]+.5)*10;wx=-11360+(cs[water]+.5)*10;wy=-32320-(rn[water]+.5)*10;dr=(rn[water]-rn[land]);dc=(cs[water]-cs[land]);x=(lx+wx)/2;y=(ly+wy)/2;tx=-dr;ty=-dc;name=f'B{len(sections)+1:05d}';lines.extend([name,'2 2',f'{x-tx*4.9:.4f} {y-ty*4.9:.4f}',f'{x+tx*4.9:.4f} {y+ty*4.9:.4f}']);sections.append({'name':name,'native_face_zero_based':int(k),'land_native_zero_based':int(land),'water_native_zero_based':int(water),'water_group_code':int(boundary_groups[rn[water],cs[water]])})
 (folder/'sfincs.crs').write_text('\n'.join(lines)+'\n');dump(folder/'receiver_sections.json',sections)
 # Bind algorithm and input files; no executable/response launch in this script.
 receipt={'status':'built_not_runtime_qualified','native_cells':len(index),'native_faces':nf,'levels':253,'builder_version':hydromt_sfincs.__version__,'subgrid_module_sha256':sha(Path(inspect.getfile(subgrid_v_table.py_func))),'subgrid_sha256':sha(dest),'solver_runs':0,'terrain_only':True,'original_dem_unchanged':True,'no_building_or_road_constraints':True,'boundary_stage_cells_unchanged':True,'new_native_map':True,'actual_step_diagnostics_pending_runtime':True}
 dump(folder/'table_build_receipt.json',receipt);print('T BUILD COMPLETE',json.dumps(receipt),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--build-tables',action='store_true');args=ap.parse_args();preflight()
 if args.build_tables:build_tables()
