"""Freeze eight matched input bundles and validate interfaces, without solver runs."""
from pathlib import Path
import sys,os,json,csv,hashlib,shutil
import numpy as np
from netCDF4 import Dataset
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_urban_geometry_pair_inputs_v1 import P,R,BASE,CON,LIB,U,CASES,sha,dump,csvwrite,masks
from load_three_ward_minimum_scenarios import export_galibier,load_forcing,find_case
from reference_label_policy import CONTRACT,POLICY

def finish():
 receipt=json.loads((P/'terrain_template/table_build_receipt.json').read_text());assert receipt['status']=='built_not_runtime_qualified';z=np.load(P/'paired_support_and_native_mapping.npz');t=z['Terrain Active Mask'];u=z['Urban Active Mask'];ground=z['Ground Area'];roof=z['Roof Area'];road=z['Ground Road Area'];total=float((t==1).sum()*100);checks=[]
 def check(name,value,details=None):checks.append({'check':name,'pass':bool(value),'details':details})
 protected=json.loads((P/'source_bindings.json').read_text());check('all_bound_originals_unchanged',all(sha(ROOT/x['file'])==x['sha256'] for x in protected))
 native_u=np.flatnonzero(np.flipud(u).ravel(order='F')>0);check('native_order_matches_existing_U_convention',np.array_equal(np.fromfile(U/'sfincs.ind','<i4'),np.r_[len(native_u),native_u+1]) and np.array_equal(np.fromfile(U/'sfincs.msk','u1'),np.flipud(u).ravel(order='F')[native_u]))
 check('same_lattice_and_boundary_cells',t.shape==u.shape==(1012,770) and np.array_equal(t==2,u==2))
 check('common_rain_catchment_and_open_ground_accounting',abs(total-ground.sum()-roof.sum())<1e-8 and total==38976800.)
 # Validate table payloads in chunks: own cell/face counts, finite arrays,
 # monotone storage knots, positive roughness and bounded wet fractions.
 table_rows=[]
 for arm,f,nn in [('T',P/'terrain_template',int((t>0).sum())),('U',U,int((u>0).sum()))]:
  finite=True;mono=True;positive=True;lower=True;max_friction_difference=0.
  with Dataset(f/'sfincs_subgrid.nc') as d:
   check(arm+'_native_cells_and_253_levels',len(d.dimensions['np'])==nn and len(d.dimensions['levels'])==253)
   for key,v in d.variables.items():
    for i in range(0,v.shape[0],6000):
     a=np.asarray(v[i:i+6000]);finite &= bool(np.isfinite(a).all())
     if key=='z_level':mono &=bool(np.all(np.diff(a,axis=1)>=-1e-10))
     if key=='uv_navg':positive &=bool(np.all(a>0));max_friction_difference=max(max_friction_difference,float(np.max(abs(a-.04))))
     if key=='uv_nrep':positive &=bool(np.all(a>0))
     if key=='uv_havg':positive &=bool(np.all(a>=0))
     if key=='uv_pwet':lower &=bool(np.all((a>=-1e-8)&(a<=1+1e-8)))
   check(arm+'_finite_monotone_and_physical_table_arrays',finite and mono and positive and lower)
   if arm=='T':check('T_uniform_background_manning_no_road_leak',max_friction_difference<1e-7)
   table_rows.append({'arm':arm,'native_cells':nn,'native_faces':len(d.dimensions['npuv']),'levels':253,'finite_tables':finite,'monotone_storage':mono,'max_face_average_n_minus0p04':max_friction_difference})
 # Independently verify new topology, metadata fields and outward boundary normals.
 top=np.load(P/'terrain_face_topology.npz');native=np.flipud(t).ravel(order='F')[z['Terrain Native Flat Index Zero Based']];a=top['from_native'];b=top['to_native'];di=top['direction'];rows=top['native_row_south'];cols=top['native_col_east'];check('T_face_neighbor_geometry',np.all((di==0)&(rows[a]==rows[b])&(cols[b]==cols[a]+1)|(di==1)&(cols[a]==cols[b])&(rows[b]==rows[a]+1)))
 sel=((native[a]==1)&(native[b]==2))|((native[a]==2)&(native[b]==1));sections=json.loads((P/'terrain_template/receiver_sections.json').read_text());check('all_T_receiving_faces_exactly_once',set(x['native_face_zero_based'] for x in sections)==set(np.flatnonzero(sel)) and len(sections)==int(sel.sum()))
 with (P/'terrain_template/udt_negative_native_map.csv').open() as f:
  read=csv.DictReader(f);correct=True;nread=0;groupok=True
  for j,row in enumerate(read):
   correct &= int(row['native_cell_index'])==j+1 and int(row['row_south_one_based'])==int(rows[j])+1 and int(row['col_east_one_based'])==int(cols[j])+1 and int(row['kcs'])==int(native[j]);groupok &= int(row['water_group_code']) in [17,51,71,79,106,110,111] if native[j]==2 else int(row['water_group_code'])==0;nread+=1
  check('T_diagnostic_native_map_order_and_groups',correct and groupok and nread==len(native))
 # Freeze original reference gates verbatim as a contract copy, no relaxation.
 dump(P/'reference_policy_binding.json',{'policy':POLICY,'output_contract':CONTRACT,'source_sha256':sha(ROOT/'src/preprocessing/reference_label_policy.py'),'note':'Historical authority fields retained verbatim. Current bundle is preparation only; executable evaluation and admission are pending.'})
 filesT=['sfincs.ind','sfincs.msk','sfincs.dep','sfincs.obs','sfincs.bnd','sfincs.crs','sfincs.inp','sfincs_subgrid.nc','udt_negative_native_map.csv','udt_negative_trace.request'];filesU=filesT+['building_closures.thd'];filesT=[n for n in filesT if (P/'terrain_template'/n).exists()]
 commonnames=['alpha','theta','huthresh','manning','manning_land','manning_sea','dtmax','hmin_cfl','advection','viscosity','zsini','qinf','dtout','dthisout','dtmaxout','dtwnd','ampr_block','amprblock','tref','tstart','tstop','x0','y0','mmax','nmax','dx','dy']
 def config(p):return {a.strip():b.strip() for line in p.read_text().splitlines() if '=' in line for a,b in [line.split('=',1)]}
 tc=config(P/'terrain_template/sfincs.inp');uc=config(U/'sfincs.inp');check('same_solver_grid_numerics_and_output_schedules',all(tc.get(k)==uc.get(k) for k in commonnames),{k:[tc.get(k),uc.get(k)] for k in commonnames if tc.get(k)!=uc.get(k)})
 check('T_removes_building_dam_U_retains_it','thdfile' not in tc and uc.get('thdfile')=='building_closures.thd')
 manifest=[];volume=[];solver=json.loads((ROOT/'data/processed/three_ward_rain_step_integral_v1/solver_binding.json').read_text())
 for case in CASES:
  forc=load_forcing(case);binding=find_case(case);uf=P/'urban_forcing'/case
  if not (uf/'export_contract.json').exists():export_galibier(case,uf)
  pdirs={'T':P/'terrain_forcing'/case,'U':uf};encoded={};curve=None
  for arm,pdir in pdirs.items():
   filename='terrain_surface_precip.nc' if arm=='T' else 'roof_surface_precip.nc';mask=t==1 if arm=='T' else u==1
   with Dataset(pdir/filename) as d:
    tt=np.asarray(d['time'][:])*60;check(case+'_'+arm+'_seconds_and_halo',np.max(abs(tt-forc['Event Time']))<1e-6 and d['Precipitation'].shape[1:]==(1014,772))
    rates=[];pattern_ok=True
    weight=z['Terrain Rainfall Weight'] if arm=='T' else z['Urban Rainfall Weight']
    for j,rate in enumerate(forc['Physical Rainfall Rate']):
     field=np.flipud(np.asarray(d['Precipitation'][j,1:-1,1:-1]));expected=(weight*rate).astype('f4');pattern_ok &=np.array_equal(field,expected);rates.append(float(field[mask].sum(dtype='f8')*100/3.6e6))
    check(case+'_'+arm+'_native_precip_pattern_and_nonnegative',pattern_ok and min(rates)>=0)
    encoded[arm]=float(np.trapezoid(rates,tt))
   dest=P/'prepared_runs'/(case+'__'+arm+'_10m');dest.mkdir(parents=True,exist_ok=True);src=P/'terrain_template' if arm=='T' else U;names=filesT if arm=='T' else filesU
   for name in names:
    target=dest/name
    if target.exists():continue
    if name=='sfincs_subgrid.nc':target.hardlink_to(src/name)
    else:shutil.copyfile(src/name,target)
   for name in ['sfincs.bzs','sfincs.bnd',filename]:
    if not (dest/name).exists():shutil.copyfile(pdir/name,dest/name)
   geomfields=['sfincs.ind','sfincs.msk','sfincs_subgrid.nc','sfincs.crs']+(['building_closures.thd'] if arm=='U' else [])
   geomhash=hashlib.sha256(json.dumps({k:sha(dest/k) for k in geomfields},sort_keys=True).encode()).hexdigest();oldid=json.loads((U/'udt_run_identity.json').read_text());identity={**oldid,'run_id':dest.name,'geometry_sha256':geomhash,'forcing_sha256':binding['forcing_sha256']};dump(dest/'udt_run_identity.json',identity);(dest/'udt_run_identity.txt').write_text('\n'.join(k+'='+str(v) for k,v in identity.items())+'\n')
   inputs={n:sha(dest/n) for n in names+['sfincs.bzs','sfincs.bnd',filename,'udt_run_identity.txt']};manifest.append({'case_id':case,'arm':arm,'folder':str(dest.relative_to(ROOT)),'image_id':solver['image_id'],'physical_forcing_sha256':binding['forcing_sha256'],'geometry_sha256':geomhash,'input_sha256':inputs,'executed':False,'runtime_qualification':'pending','parent_id':binding['parent_id'],'split':binding['split']})
  nominal=float(np.trapezoid(forc['Physical Rainfall Rate'],forc['Event Time'])*total/3.6e6);tol=max(1.,nominal)*1e-6
  check(case+'_actual_encoded_pair_rain_volumes_match',abs(encoded['T']-nominal)<=tol and abs(encoded['U']-nominal)<=tol and abs(encoded['U']-encoded['T'])<=2*tol)
  check(case+'_boundary_curves_identical',(pdirs['T']/'sfincs.bnd').read_bytes()==(pdirs['U']/'sfincs.bnd').read_bytes() and np.max(abs(np.loadtxt(pdirs['T']/'sfincs.bzs')-np.loadtxt(pdirs['U']/'sfincs.bzs')))<1e-8)
  volume.append({'case_id':case,'physical_rain_depth_mm':float(np.trapezoid(forc['Physical Rainfall Rate'],forc['Event Time'])/3600),'physical_catchment_m2':total,'nominal_m3':nominal,'terrain_encoded_m3':encoded['T'],'urban_encoded_m3':encoded['U'],'urban_minus_terrain_m3':encoded['U']-encoded['T'],'individual_gate_m3':tol,'status':'encoded_only_actual_step_application_pending'})
 csvwrite(R/'actual_encoded_pair_volumes.csv',volume);dump(P/'prepared_run_manifest.json',{'runs':manifest,'new_hydraulic_runs':0,'maximum_candidate_runs':8,'original_case_and_parent_counts_unchanged':True,'no_historical_raster_forcing_yet':True,'scope':'Prepared only; no run invoked. No old response is certified as paired truth.'});csvwrite(R/'table_interface_summary.csv',table_rows)
 check('eight_bundles_no_solver_response_generated',len(manifest)==8 and all(not (ROOT/r['folder']/'sfincs.log').exists() and not (ROOT/r['folder']/'sfincs_map.nc').exists() for r in manifest))
 result={'all_passed':all(x['pass'] for x in checks),'checks':checks,'input_pair_readiness':'prepared_and_static_interfaces_validated','runtime_qualified':False,'actual_solver_runs':0,'actual_training':0,'sealed_tests':False,'needed_next':'Run dry and rain pairs under unchanged numerical gates, then common-ground labels and transport exposure; historical rainfall branch still awaits numeric raster.'};dump(R/'validation.json',result);assert result['all_passed'],[x for x in checks if not x['pass']];print('STATIC INPUT VALIDATION PASSED',len(checks),'checks;8 bundles;0 solver runs',flush=True)
if __name__=='__main__':finish()
