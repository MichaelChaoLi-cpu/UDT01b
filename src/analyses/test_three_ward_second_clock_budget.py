"""Compiled precipitation I/O and bookkeeping tests. No hydraulic update API."""
from pathlib import Path
import os,sys,json,csv,subprocess,tempfile,shutil,hashlib,re
import numpy as np
from netCDF4 import Dataset
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_sfincs_second_clock_budget import OUT,REPORT,COMMIT,sha,dump
from generate_three_ward_minimum_scenarios import semantic
from stage_second_clock_budget_identity import expected_bindings,stage_identity,LEGACY_DIGEST
LIB='data/processed/three_ward_minimum_scenarios_v1'
HARNESS='src/analyses/second_clock_budget_tests'

def write_csv(p,rows):
 with Path(p).open('w',newline='')as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def read_csv(p):return list(csv.DictReader(Path(p).open()))
def main(root=ROOT):
 root=Path(root);out=root/OUT;report=root/REPORT;src=out/'patched';test=root/HARNESS
 if(out/'freeze.json').exists():raise FileExistsError('Frozen interface receipt; use independent validator')
 checks=[];commands=[]
 def check(name,value,evidence=None):
  checks.append({'check':name,'pass':bool(value),'evidence':evidence});print(('PASS 'if value else'FAIL ')+name,flush=True)
 def command(argv,cwd,allow_fail=False,env=None):
  r=subprocess.run([str(x) for x in argv],cwd=cwd,env=env,capture_output=True,text=True)
  commands.append({'argv':[str(x) for x in argv],'returncode':r.returncode,'stderr':r.stderr[-6000:]})
  if r.returncode and not allow_fail:raise RuntimeError(r.stderr+'\n'+r.stdout)
  return r
 env=os.environ.copy();env['PATH']='/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin';env['OMP_NUM_THREADS']='2'
 with tempfile.TemporaryDirectory(prefix='udt01b-second-interface-',dir='/private/tmp')as tmp:
  build=Path(tmp);fc='/opt/homebrew/bin/gfortran'
  flags=['-cpp','-O0','-fopenmp','-ffree-line-length-none','-fcheck=all','-ffp-contract=off','-J',build,'-I',build]
  command([fc,*flags,'-c',src/'sfincs_data.f90',src/'sfincs_date.f90',src/'udt_rain_clock.f90',src/'udt_clock_budget.f90',src/'sfincs_continuity.f90'],build,env=env)
  check('actual_patched_continuity_and_new_modules_compile_against_unmodified_real_data_module',True)
  # Compile real library/output hooks with dependency shells. Compile only: no linking or execution.
  bridge=(test/'netcdf_bridge.f90').read_text().replace('module sfincs_log\ncontains','module sfincs_log\n character(2048)::logstr\ncontains')
  (build/'netcdf_bridge.f90').write_text(bridge)
  command([fc,*flags,'-c',build/'netcdf_bridge.f90'],build,env=env)
  modules=re.findall(r'^\s*use (sfincs_\w+)',(src/'sfincs_lib.f90').read_text(),flags=re.M)
  keep={'sfincs_data','sfincs_date','sfincs_output','sfincs_continuity','sfincs_log'}
  (build/'dependency_shells.f90').write_text('\n'.join('module '+m+'\nend module'for m in dict.fromkeys(modules)if m not in keep))
  command([fc,*flags,'-c',build/'dependency_shells.f90',src/'sfincs_output.f90',src/'sfincs_lib.f90'],build,env=env)
  check('actual_library_and_final_output_hooks_compile_with_dependency_shells',True,{'complete_solver_link':False})
  source=(src/'sfincs_ncinput.F90').read_text();a=source.index('   subroutine read_netcdf_ampr_data()');b=source.index('   end subroutine',a)+len('   end subroutine')
  typ=re.search(r'   type net_type_ampr\n.*?   end type',source,re.S).group()
  reader='#define NF90(nf90call) call handle_err(nf90call,__FILE__,__LINE__)\nmodule extracted_rain_reader\n use sfincs_log\n'+typ+'\n type(net_type_ampr)::net_file_ampr\ncontains\n'+source[a:b]+'''
 subroutine handle_err(status,path,line)
 integer::status,line
 character(*)::path
 if(status/=0)then
 write(*,*)'NetCDF error',status,path,line
 error stop 2
 endif
 end subroutine
end module
'''
  (build/'extracted_reader.F90').write_text(reader)
  (out/'extracted_reader_test.F90').write_text(reader)
  command(['/usr/bin/clang','-I/opt/homebrew/include','-c',test/'netcdf_bridge.c','-o',build/'netcdf_bridge_c.o'],build,env=env)
  command([fc,*flags,'-c',build/'extracted_reader.F90'],build,env=env)
  objects=[build/(x+'.o') for x in ['sfincs_data','sfincs_date','udt_rain_clock','netcdf_bridge','extracted_reader','netcdf_bridge_c']]
  command([fc,*flags,*objects,test/'rain_reader_driver.f90','-L/opt/homebrew/lib','-lnetcdf','-o',build/'rain_reader'],build,env=env)
  command([fc,*flags,build/'sfincs_data.o',build/'udt_clock_budget.o',test/'budget_driver.f90','-o',build/'budget_fixture'],build,env=env)
  check('real_patched_reader_body_compiles_with_actual_netcdf_c_io_bridge',True,{'production_fortran_netcdf_link':False,'rain_reader_routine_sha256':hashlib.sha256(source[a:b].encode()).hexdigest()})
  catalog=json.loads((root/LIB/'catalog.json').read_text());cases=[c for c in catalog['cases']if not c.get('existing_completed')]
  assert len(cases)==50
  receipt=[];geometry=np.load(root/LIB/'shared_geometry.npz',allow_pickle=False)
  areas={grid:float(np.sum(geometry['Effective Rainfall Weight '+grid],dtype='f8')*(100 if grid=='10m' else 25))for grid in ['10m','5m']}
  allocation_ok=all(np.max(abs(geometry['Ground Area '+g]+geometry['Roof Assigned Area '+g]-geometry['Contributing Area '+g]))<1e-7 for g in ['10m','5m'])
  check('fixed_roof_allocation_preserves_ground_plus_roof_area',allocation_ok)
  ground_area=float(np.sum(geometry['Ground Area 10m'],dtype='f8'));roof_area=float(np.sum(geometry['Roof Assigned Area 10m'],dtype='f8'))
  # A nonuniform tile tests source reshape, row flip, zeros and float32 rain values.
  weights=np.array([[0.,.25,1.5],[1.,2.,0.]],dtype='f8');forcing_dir=out/'transport_test_inputs';forcing_dir.mkdir(exist_ok=True)
  def make_nc(path,t,rates,units='minutes since 2020-01-01 00:00:00',dx=10):
   with Dataset(path,'w')as ds:
    for name,n in [('time',len(t)),('y',2),('x',3)]:ds.createDimension(name,n)
    tv=ds.createVariable('time','f8',('time',));tv.units=units;tv[:]=t
    ds.createVariable('x','f8',('x',))[:]=-11360-dx+np.arange(3)*dx
    ds.createVariable('y','f8',('y',))[:]=-42440-dx+np.arange(2)*dx
    p=ds.createVariable('Precipitation','f4',('time','y','x'),fill_value=-9999.)
    p[:]=(np.array(rates)[:,None,None]*weights).astype('f4')
    ds.test_fixture='factorized spatial interface tile, not a full-domain forcing export or response'
  for case in cases:
   cid=case['case_id']
   with np.load(root/LIB/case['forcing_file'],allow_pickle=False)as z:
    t=z['Event Time'];r=z['Physical Rainfall Rate'];semantic_sha=semantic({k:z[k]for k in z.files})
   path=forcing_dir/(cid+'.nc');make_nc(path,t/60,r)
   folder=build/cid;folder.mkdir();result=command([build/'rain_reader',path],folder,env=env)
   data=np.fromstring(result.stdout,sep=' ');nt,ny,nx=data[:3].astype(int);decoded=data[3:3+nt]
   rain=data[3+nt:3+nt+nt*ny*nx].reshape(nt,ny,nx)
   expected=(r[:,None,None]*weights[None]).astype('f4')[:,::-1,:].astype('f8')
   source_depth=float(np.trapezoid(r,t)/3600);decoded_depth=float(np.trapezoid(r,decoded)/3600)
   trace=folder/'udt_rain_decode.csv';shutil.copyfile(trace,report/(cid+'__decode.csv'))
   receipt.append({'case_id':cid,'parent_id':case['parent_id'],'family_id':case['family_id'],'split':case['split'],
    'knots':nt,'time_error_s':float(np.max(abs(t-decoded))),'collisions':int(np.sum(np.diff(decoded)<=0)),
    'rain_values_match':bool(np.array_equal(rain,expected)),'source_depth_mm':source_depth,'decoded_depth_mm':decoded_depth,
    'time_integral_error_mm':abs(source_depth-decoded_depth),'original_file_sha256':sha(root/LIB/case['forcing_file']),'original_forcing_sha256':semantic_sha,
    'allocated_input_volume_10m_m3':decoded_depth/1000*areas['10m'],'allocated_input_volume_5m_m3':decoded_depth/1000*areas['5m'],
    'ground_rain_volume_m3':decoded_depth/1000*ground_area,'roof_assigned_volume_m3':decoded_depth/1000*roof_area,
    'x_corner':float(data[-4]),'y_corner':float(data[-3]),'dx':float(data[-2]),'dy':float(data[-1])})
  write_csv(report/'50_case_transport_receipt.csv',receipt)
  check('50_original_forcings_through_actual_patched_reader',len(receipt)==50)
  check('second_knots_exact_and_strict_in_all_50',all(x['time_error_s']==0 and x['collisions']==0 for x in receipt))
  check('all_50_original_piecewise_linear_integrals_preserved',all(x['time_integral_error_mm']==0 for x in receipt))
  check('original_float32_rain_values_row_flip_and_valid_zeros_unchanged',all(x['rain_values_match']for x in receipt))
  check('pinned_coordinate_corner_convention_unchanged',all(x['x_corner']==-11370 and x['y_corner']==-42450 and x['dx']==x['dy']==10 for x in receipt))
  check('forcing_file_bytes_and_parent_split_bindings_preserved',all(x['original_forcing_sha256']==next(c['forcing_sha256']for c in cases if c['case_id']==x['case_id'])for x in receipt))
  check('full_domain_roof_allocated_volume_equal_across_grid_supports',max(abs(x['allocated_input_volume_10m_m3']-x['allocated_input_volume_5m_m3'])for x in receipt)<.001,areas)
  check('ground_rain_and_roof_transfer_are_not_double_counted', all(abs(x['allocated_input_volume_10m_m3']-x['ground_rain_volume_m3']-x['roof_assigned_volume_m3'])<.001 for x in receipt))
  fault_inputs={
   'seconds_header':([0,60], 'seconds since 2020-01-01 00:00:00'),
   'duplicate':([0,0], 'minutes since 2020-01-01 00:00:00'),
   'reverse':([1,0], 'minutes since 2020-01-01 00:00:00'),
   'nonfinite':([0,np.nan], 'minutes since 2020-01-01 00:00:00'),
   'subsecond':([0,1.001], 'minutes since 2020-01-01 00:00:00'),
   'float32_range':([0,16777217/60], 'minutes since 2020-01-01 00:00:00'),
   'timezone':([0,1], 'minutes since 2020-01-01 00:00:00 +0900'),
   'invalid_date':([0,1], 'minutes since 2020-02-30 00:00:00'),
  }
  faults=[]
  for name,(t,units)in fault_inputs.items():
   folder=build/('fault_'+name);folder.mkdir();path=folder/'input.nc';make_nc(path,t,[0,1],units)
   r=command([build/'rain_reader',path],folder,allow_fail=True,env=env);faults.append({'test':name,'rejected':r.returncode!=0})
  write_csv(report/'rain_fault_tests.csv',faults)
  check('unsupported_precision_units_dates_nonfinite_and_collisions_fail_closed',all(x['rejected']for x in faults))
  # Native 5 m transport for the eight preselected adaptive descendants, without new curves.
  bindings=expected_bindings(root);adaptive=[]
  for item in bindings:
   if item['grid']!='adaptive_10_5m':continue
   cid=item['case_id']
   with np.load(root/LIB/next(c['forcing_file']for c in cases if c['case_id']==cid),allow_pickle=False)as z:
    t=z['Event Time'];rates=z['Physical Rainfall Rate']
   folder=build/(cid+'__5m');folder.mkdir();path=folder/'input.nc';make_nc(path,t/60,rates,dx=5)
   r=command([build/'rain_reader',path],folder,env=env);v=np.fromstring(r.stdout,sep=' ');n=int(v[0])
   adaptive.append({'run_id':item['run_id'],'parent_id':item['parent_id'],'family_id':item['family_id'],'split':item['split'],'knots_equal':bool(np.array_equal(v[3:3+n],t)),'coordinates_equal':bool(np.array_equal(v[-4:],[-11365,-42445,5,5]))})
  write_csv(report/'adaptive_transport_receipt.csv',adaptive)
  check('eight_adaptive_descendant_inputs_keep_seconds_coordinates_and_partitions',len(adaptive)==8 and all(x['knots_equal']and x['coordinates_equal']for x in adaptive))
  dump(out/'pending_run_bindings.json',{'bindings':bindings,'execution_release':False,'reference_release':False})
  demo=stage_identity(bindings[0]['run_id'],build/'identity_staging','sha256:'+'a'*64,root)
  check('all_58_manifest_descendants_bound_without_split_or_lineage_change',len(bindings)==58 and demo['parent_id']==bindings[0]['parent_id']and demo['split']==bindings[0]['split'])
  denied=False
  try:stage_identity(bindings[0]['run_id'],build/'wrong_digest',LEGACY_DIGEST,root)
  except ValueError:denied=True
  check('old_unpatched_image_digest_cannot_be_staged_as_new_build',denied)
  # Epoch offset and complete legacy UTC header; full and short forms both supported.
  offsets=[]
  for name,units,t,expected in [('epoch','minutes since 2019-12-31 23:59:00',[0,1,2],[-60,0,60]),('full','minutes since 2020-01-01 00:00:00.0 +0000',[0,3601/60,14400/60],[0,3601,14400])]:
   folder=build/name;folder.mkdir();path=folder/'input.nc';make_nc(path,t,[0,1,0],units)
   r=command([build/'rain_reader',path],folder,env=env);v=np.fromstring(r.stdout,sep=' ')
   offsets.append(np.array_equal(v[3:6],expected))
  check('utc_epoch_offset_and_full_header_preserve_seconds',all(offsets))
  # The same real decoder rejects an existing trace rather than overwriting it.
  first=cases[0]['case_id'];rr=command([build/'rain_reader',forcing_dir/(first+'.nc')],build/first,allow_fail=True,env=env)
  check('existing_decode_trace_cannot_be_overwritten',rr.returncode!=0)
  fixtures=[]
  for mode in ['normal','dry','negative','unsupported','infiltration','spinup','storage','waves','block','clock_mismatch','nonfinite','missing_cell','missing_identity','wrong_identity']:
   folder=build/('budget_'+mode);folder.mkdir()
   if mode!='missing_identity':
    identity={'schema_version':'second-clock-budget-interface-v1','run_id':'fixture','parent_id':'fixture','family_id':'fixture','split':'fixture','forcing_sha256':'fixture','geometry_sha256':'fixture','source_commit':COMMIT,'patch_sha256':'fixture','solver_digest':'fixture'}
    if mode=='wrong_identity':identity['source_commit']='wrong'
    (folder/'udt_run_identity.txt').write_text('\n'.join(k+'='+v for k,v in identity.items())+'\n')
   r=command([build/'budget_fixture',mode],folder,allow_fail=True,env=env)
   expected=mode in ['normal','dry','negative'];ok=(r.returncode==0)==expected
   fixtures.append({'test':mode,'expected_success':expected,'returncode':r.returncode,'pass':ok})
   if expected:
    dest=report/('fixture_'+mode);shutil.copytree(folder,dest,dirs_exist_ok=True)
  write_csv(report/'budget_fixture_tests.csv',fixtures)
  check('bookkeeping_fixtures_and_fail_closed_profiles',all(x['pass']for x in fixtures))
  ledger=read_csv(report/'fixture_normal/udt_step_budget.csv');f=lambda k:np.array([float(x[k])for x in ledger])
  check('true_initial_row_and_all_actual_steps_exported',len(ledger)==5 and float(ledger[0]['step_index'])==0 and f('actual_state_time')[0]==0 and f('actual_step_dt')[0]==0 and np.allclose(np.diff(f('actual_state_time')),f('actual_step_dt')[1:],atol=1e-12,rtol=0))
  check('gross_inflow_and_outflow_do_not_cancel',f('boundary_out_cumulative')[-1]>.99 and f('boundary_in_cumulative')[-1]>.99)
  check('boundary_native_10m_and_5m_widths_use_same_step',abs(f('boundary_out_cumulative')[-1]-1)<1e-6 and abs(f('boundary_in_cumulative')[-1]-1)<1e-6)
  check('signed_storage_and_roundoff_residual_are_measured',max(abs(f('budget_residual')))<5e-5 and max(abs(f('internal_flux_roundoff_residual')))<1e-6,{'maximum_storage_roundoff_m3':float(max(abs(f('budget_residual'))))})
  check('observed_meteo_brackets_separate_from_pre_step_initial_state', f('meteo_bracket_observed')[0]==0 and np.all(f('meteo_bracket_observed')[1:]==1) and np.all(f('meteo_update_interval')[1:]==1))
  check('actual_rain_cumulative_positive_and_monotone',f('rain_applied_cumulative')[-1]>0 and np.all(np.diff(f('rain_applied_cumulative'))>=0))
  dry=read_csv(report/'fixture_dry/udt_step_budget.csv');check('no_rain_control_exports_explicit_zero_rain',all(float(x['rain_applied_cumulative'])==0 for x in dry))
  negative=read_csv(report/'fixture_negative/udt_step_budget.csv');mask=read_csv(report/'fixture_negative/udt_native_ever_negative.csv')
  check('negative_native_storage_not_clipped_and_ever_mask_exported',any(float(x['minimum_native_volume_since_initialization'])<=-.25 for x in negative) and mask[1]['ever_volume_below_minus_0_001_m3']=='T')
  mappings=read_csv(report/'fixture_normal/udt_output_clock.csv');index={int(x['step_index']):x for x in ledger}
  check('each_output_mapping_binds_exactly_same_accounted_clock',all(float(x['actual_state_time'])==float(index[int(x['step_index'])]['actual_state_time'])for x in mappings))
  peaks=[x for x in mappings if x['write_peak']=='T'];check('peak_windows_include_finalizer_and_actual_endpoints',len(peaks)==2 and float(peaks[0]['actual_peak_start'])==0 and float(peaks[1]['actual_peak_start'])==float(peaks[0]['actual_peak_end']))
  final=read_csv(report/'fixture_normal/udt_final_clock.csv')[0];check('final_state_receipt_uses_live_time_and_solver_error',float(final['actual_final_time'])==f('actual_state_time')[-1]and final['solver_error']=='0')
  check('only_bookkeeping_fixture_and_rain_io_executed_not_continuity',all('sfincs'not in Path(x['argv'][0]).name for x in commands if x['argv'][0]not in [fc,'/usr/bin/clang']))
  compiler=command([fc,'--version'],build,env=env).stdout.splitlines()[0]
  dump(report/'compile_receipt.json',{'compiler':compiler,'commands':commands,'production_solver_linked':False,'production_fortran_netcdf_linked':False,'hydrodynamic_solver_executed':False,'continuity_compiled_not_executed':True,'test_bridge':'real NetCDF C library; limited test-only nf90 signatures'})
 prov=json.loads((report/'provenance.json').read_text())
 check('protected_old_sources_curves_policies_and_reference_labels_unchanged',all(sha(root/p)==h for p,h in prov['protected_sha256'].items()))
 check('existing_native_results_unchanged',all([(root/p).stat().st_size,(root/p).stat().st_mtime_ns]==fp for p,fp in prov['existing_result_stat_fingerprints'].items()))
 registry=json.loads((root/'data/processed/three_ward_reference_contract_v1/reference_registry.json').read_text())
 check('original_strict_reference_holds_unchanged',all(c['state']=='hold_event'and not c['technical_reference']for c in registry['controls'].values()))
 summary={'version':'second-clock-budget-interface-v1','case_count':50,'max_time_error_s':max(x['time_error_s']for x in receipt),'max_time_integral_error_mm':max(x['time_integral_error_mm']for x in receipt),'all_checks_passed':all(x['pass']for x in checks),'checks':len(checks),'new_hydrodynamic_runs':0,'training':0,'sealed_responses_read':0,'production_image_qualified':False,'reference_labels_released':False,'all_58_descendants_held':True,'forcing_curves_lineage_splits_unchanged':True}
 dump(report/'validation.json',{'all_checks_passed':summary['all_checks_passed'],'checks':checks})
 dump(report/'summary.json',summary)
 dump(out/'readiness.json',{'transport_component':'implemented_compiled_and_50_inputs_verified','step_diagnostics':'source_hooks_compiled_and_bookkeeping_fixtures_verified','production_image':'not_built_or_end_to_end_qualified','pending_runs':58,'execution_release':False,'reference_release':False,'remaining':['complete production solver and Fortran NetCDF build/link receipt','seven-group optional budget mapping','operational end-to-end clock/ledger and geometry validation; requires separate simulation authorization']})
 print(json.dumps(summary,indent=2))
 if not summary['all_checks_passed']:raise SystemExit(1)
if __name__=='__main__':main()
