"""Production-linked diagnostic fixtures and reader regressions; zero solver calls."""
from pathlib import Path
import sys,json,csv,subprocess,shutil,argparse
import numpy as np
from netCDF4 import Dataset
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_sfincs_negative_trace import OUT,REPORT,TAG
from prepare_sfincs_cpu_build import sha,dump,COMMIT
sys.path.insert(0,str(ROOT/'src/analyses'))
from summarize_sfincs_negative_trace import summarize,read

def identity(folder,image,wrong=False):
 values={'schema_version':'second-clock-budget-interface-v1','run_id':'isolated-prescribed-fixture','parent_id':'fixture','family_id':'fixture','split':'fixture','forcing_sha256':'fixture','geometry_sha256':'fixture','source_commit':'wrong'if wrong else COMMIT,'patch_sha256':sha(ROOT/OUT/'negative_trace_full.patch'),'solver_digest':image}
 (folder/'udt_run_identity.txt').write_text('\n'.join(k+'='+v for k,v in values.items())+'\n')
def metadata(folder,mode):
 nm=[2,2,3,2,1,4];mm=[2,3,2,1,2,2];masks=[1,1,1,2,2,2];groups=[0,0,0,79,111,17]
 pointers=[[1,2,3,4],[2,0,0,0],[0,0,4,5],[0,1,0,0],[0,0,0,3],[0,0,5,0]]
 minimum=np.array([0,.1,.2,.3,.4,.5],dtype='f4')
 columns=['native_cell_index','row_south_one_based','col_east_one_based','kcs','water_group_code']+[d+'_'+k for k in ['open_width_m','road_width_m','face_minimum_m']for d in ['west','east','south','north']]
 rows=[]
 for i in range(6):
  widths=[3.5 if ip==2 else 8. if ip else np.nan for ip in pointers[i]]
  roads=[1.25 if ip else np.nan for ip in pointers[i]]
  crests=[float(minimum[ip])if ip else np.nan for ip in pointers[i]]
  rows.append([i+1,nm[i],mm[i],masks[i],groups[i],*widths,*roads,*crests])
 if mode=='unknown_group':rows[3][4]=42
 if mode=='badcrest':rows[0][13]+=.5
 if mode=='badwidth':rows[0][5]=-1
 if mode=='shuffled':rows[0],rows[1]=rows[1],rows[0]
 if mode=='extra_row':rows.append(rows[-1])
 if mode!='missingmap':
  with(folder/'udt_negative_native_map.csv').open('w',newline='')as f:
   w=csv.writer(f);w.writerow(columns);w.writerows(rows)
 if mode!='no_trace':(folder/'udt_negative_trace.request').write_text('schema=wrong\n'if mode=='badrequest'else'schema=negative-cell-face-v1\n')

def reference(mode):
 # Scalar single-precision arithmetic, using the same grouping as production continuity.
 f=np.float32;v=np.array([.1,.01,.2],dtype='f8');rate=np.array([.001,0,.002],dtype='f4')
 if mode=='initial_negative':v[1]=-.2
 if mode=='threshold_equal':v[0]=-.001;rate[:]=0
 result=[v.copy()];records=[];t=0.;qs=[[0,.01,.2,-.02,.005,0],[0,.1,.01,0,0,.2],[0,0,.3,0,0,-.5],[0,0,0,0,0,0]]
 indices=[[1,2,3,4],[2,0,0,0],[0,0,4,5]]
 for n,dt0 in enumerate([.125,.375,.2,.3],1):
  dt=f(dt0);q=np.array(qs[n-1],dtype='f4');old=v.copy()
  if mode=='threshold_equal':q[:]=0
  t0=t;t+=float(dt)
  for cell,ids in enumerate(indices):
   west,east,south,north=q[ids]
   horizontal=f(f(f(west-east)*f(10))+f(f(south-north)*f(10)))*dt
   applied=f(f(rate[cell]*f(100))*dt);dvol=f(horizontal+applied);v[cell]+=float(dvol)
   face=[float(f(f(q[ip]*f(10))*dt))*(1 if i in [0,2]else -1)for i,ip in enumerate(ids)]
   records.append({'step':n,'cell':cell+1,'before':t0,'after':t,'previous':old[cell],'new':v[cell],'horizontal':float(horizontal),'rain':float(applied),'rain_increment':float(dvol)-float(horizontal),'faces':face})
  result.append(v.copy())
 return np.array(result),records

def main(validate_only=False):
 out=ROOT/OUT;report=ROOT/REPORT
 if(out/'freeze.json').exists():raise FileExistsError('Frozen interface')
 meta=json.loads((out/'build-metadata.json').read_text());image=meta['containerimage.digest'];outputs=out/'interface_tests';outputs.mkdir(exist_ok=True)
 invalid={'dummy_nonzero':'nonzero native dummy-face flux','raw_zero_pointer':'invalid crossing face pointer','adaptive':'regular one-level grid','missingmap':'missing negative native map','unknown_group':'unknown boundary water group','badcrest':'mapped face minimum differs','badwidth':'invalid open face width','shuffled':'negative native map identity mismatch','extra_row':'unexpected extra negative native map row','badrequest':'invalid negative trace request','missing_cell':'missing active-cell continuity contribution','clock_mismatch':'actual clock/dt disagreement'}
 modes=['normal','normal_parallel','initial_negative','threshold_equal','no_trace','whole_metadata',*invalid]
 script=['set -eu','/opt/udt-build/loader_probe > /audit/loader.txt','/opt/udt-build/netcdf_link_probe > /audit/netcdf_probe.txt','cp /opt/udt-build/NEGATIVE_BINDING.json /audit/embedded_binding.json','cp /opt/udt-build/receipts/negative-* /audit/','ldd /usr/local/lib/libsfincs.so > /audit/library_ldd.txt','nm -D --defined-only /usr/local/lib/libsfincs.so > /audit/symbols.txt','sha256sum /usr/local/bin/sfincs /usr/local/lib/libsfincs.so.0.0.0 /usr/local/lib/libnetcdff.so.7.2.0 > /audit/binaries.sha256']
 for mode in modes:
  folder=outputs/mode;folder.mkdir(exist_ok=True);identity(folder,image)
  if mode=='whole_metadata':
   for name in ['udt_negative_native_map.csv','udt_negative_trace.request','whole_native_topology.bin']:shutil.copyfile(out/'whole_domain_metadata'/name,folder/name)
  else:metadata(folder,mode)
  actual='normal'if mode=='normal_parallel'else mode
  script+=['set +e',f'(cd /audit/{mode} && OMP_NUM_THREADS={4 if mode=="normal_parallel"else 1} /opt/udt-build/negative_trace_fixture {actual} >stdout.txt 2>stderr.txt)','code=$?','set -e',f'printf "%s\\n" "$code" > /audit/{mode}/exit_code.txt']
 old_modes=['normal','dry','negative','unsupported','infiltration','spinup','storage','waves','block','clock_mismatch','nonfinite','missing_cell','missing_identity','wrong_identity']
 for mode in old_modes:
  folder=outputs/'legacy'/mode;folder.mkdir(parents=True,exist_ok=True)
  if mode!='missing_identity':identity(folder,image,mode=='wrong_identity')
  script+=['set +e',f'(cd /audit/legacy/{mode} && /opt/udt-build/production_budget_fixture {mode} >stdout.txt 2>stderr.txt)','code=$?','set -e',f'printf "%s\\n" "$code" > /audit/legacy/{mode}/exit_code.txt']
 inputs=ROOT/'data/processed/three_ward_cpu_e2e_controls_v1/compatibility_v2/test_inputs'
 for p in sorted(inputs.glob('*.nc')):
  name=p.stem;(outputs/'rain'/name).mkdir(parents=True,exist_ok=True)
  script+=['set +e',f'(cd /audit/rain/{name} && /opt/udt-build/production_rain_reader /inputs/{name}.nc >stdout.txt 2>stderr.txt)','code=$?','set -e',f'printf "%s\\n" "$code" > /audit/rain/{name}/exit_code.txt']
 path=out/'run-interface-tests.sh';path.write_text('\n'.join(script)+'\n')
 if not validate_only:subprocess.run(['docker','run','--rm','--platform','linux/amd64','--network','none','--mount','type=bind,source='+str(outputs)+',target=/audit','--mount','type=bind,source='+str(inputs)+',target=/inputs,readonly','--mount','type=bind,source='+str(path)+',target=/probe.sh,readonly','--entrypoint','/bin/sh',image,'/probe.sh'],check=True)
 checks=[]
 def check(n,ok,details=None):checks.append({'check':n,'pass':bool(ok),'details':details});print(('PASS 'if ok else'FAIL ')+n,flush=True)
 check('embedded_source_binding_matches_host',sha(outputs/'embedded_binding.json')==sha(out/'source_binding.json'))
 check('real_Fortran_NetCDF_and_library_loader_pass','PASS:'in(outputs/'netcdf_probe.txt').read_text()and'zero solver entry-point calls'in(outputs/'loader.txt').read_text()and'not found'not in(outputs/'library_ldd.txt').read_text())
 check('public_budget_entry_points_and_enabled_trace_receipt',all(name in(outputs/'symbols.txt').read_text()for name in ['udt_budget_initialize','udt_budget_begin','udt_budget_finish','udt_budget_close'])and 'schema=negative-cell-face-v1'in(outputs/'normal/udt_negative_trace_receipt.txt').read_text())
 for mode in modes:
  code=int((outputs/mode/'exit_code.txt').read_text());ok=(code!=0 and invalid[mode]in(outputs/mode/'stdout.txt').read_text())if mode in invalid else code==0
  check('fixture_'+mode,ok,{'exit_code':code,'expected_rejection':invalid.get(mode)})
 whole=outputs/'whole_metadata'
 if int((whole/'exit_code.txt').read_text())==0:
  fullcells=read(whole/'udt_negative_cell_summary.csv');links=read(whole/'udt_negative_cell_group_links.csv')
  with np.load(ROOT/'data/processed/three_ward_surrogate_inputs/surface_connectivity_graph.npz')as g:
   nodeflat=g['Node Grid Index'];codes=g['Boundary Group Codes'];group_index=g['Boundary Group Index'];receiver_links=g['Receiver Link Index']
   expected={(int(nodeflat[a]),int(codes[group_index[b]]))for a,b in receiver_links.T}
  idx=np.fromfile(ROOT/'data/processed/three_ward_cpu_e2e_controls_v1/runs/dry_raised__e2e_regular_10m/sfincs.ind',dtype='<i4')[1:]-1
  flat=(1011-idx%1012)*770+idx//1012
  actual={(int(flat[int(r['native_cell_index'])-1]),int(r['water_group_code']))for r in links if int(r['water_group_code'])!=0}
  check('whole_domain_all_358384_interior_cells_native_metadata_accepted',len(fullcells)==358384 and all(r['first_below_endpoint_s']==''for r in fullcells))
  check('whole_domain_open_receiver_group_memberships_match_existing_graph',expected==actual)
  check('whole_domain_fixture_has_zero_steps_and_no_fictitious_crossing',len(read(whole/'udt_step_budget.csv'))==1 and len(read(whole/'udt_negative_crossings.csv'))==len(read(whole/'udt_negative_crossing_faces.csv'))==0)
 for mode in ['normal','initial_negative','threshold_equal']:
  folder=outputs/mode
  if int((folder/'exit_code.txt').read_text())!=0:continue
  states,expected=reference(mode);cells=read(folder/'udt_negative_cell_summary.csv');events=read(folder/'udt_negative_crossings.csv');faces=read(folder/'udt_negative_crossing_faces.csv')
  mapping={(int(r['step_index']),int(r['native_cell_index'])):r for r in events}
  crosses=[r for r in expected if (r['previous']<-.001)!=(r['new']<-.001)]
  check(mode+'_all_down_and_recovery_crossings_and_four_faces',set(mapping)=={(r['step'],r['cell'])for r in crosses}and len(faces)==4*len(events))
  check(mode+'_initial_flags_minima_and_counts',all((r['initial_below_threshold']=='T')==(states[0,i]<-.001)and int(r['negative_poststep_count'])==int(np.sum(states[1:,i]<-.001))and abs(float(r['minimum_volume_m3'])-states[:,i].min())<1e-12 for i,r in enumerate(cells)))
  detailed=True
  for e in crosses:
   row=mapping[(e['step'],e['cell'])];ff=[r for r in faces if r['event_id']==row['event_id']]
   detailed &=row['kind']==('down'if e['new']<-.001 else'up')
   for field,key in [('before_time_s','before'),('after_time_s','after'),('previous_volume_m3','previous'),('new_volume_m3','new'),('horizontal_applied_m3','horizontal'),('rain_applied_m3','rain'),('rain_postadd_increment_m3','rain_increment')]:detailed &=abs(float(row[field])-e[key])<1e-12
   detailed &=np.allclose([float(r['signed_inward_volume_m3'])for r in ff],e['faces'],rtol=0,atol=1e-12)
   detailed &=abs(float(row['horizontal_minus_four_faces_m3'])-(e['horizontal']-sum(e['faces'])))<1e-12
  check(mode+'_exact_step_flux_signs_rain_and_roundoff_match_scalar_reference',detailed)
  # No undocumented fractional-step crossing or negative-duration interpolation.
  timing=True
  times=np.r_[0,np.cumsum(np.array([.125,.375,.2,.3],dtype='f4').astype('f8'))]
  for i,row in enumerate(cells):
   seen=np.flatnonzero(states[:,i]<-.001);minidx=int(np.argmin(states[:,i]))
   timing &=int(row['minimum_step'])==minidx and abs(float(row['minimum_time_s'])-times[minidx])<1e-12
   timing &=float(row['first_below_endpoint_s'])==times[seen[0]]and float(row['last_below_endpoint_s'])==times[seen[-1]]if len(seen)else row['first_below_endpoint_s']==row['last_below_endpoint_s']==''
  check(mode+'_first_last_and_earliest_minimum_endpoint_times',timing)
  result=summarize(folder,outputs/(mode+'_grouped'),json.loads((out/'output_schema.json').read_text())['groups'])
  check(mode+'_water_group_aggregation_unique_overall',result['unique_ever_negative_cells']==int(np.sum((states<-.001).any(0))))
 if int((outputs/'normal/exit_code.txt').read_text())==0:
  columns=['udt_step_budget.csv','udt_negative_crossings.csv','udt_negative_crossing_faces.csv','udt_negative_cell_summary.csv','udt_negative_cell_group_links.csv']
  check('serial_parallel_diagnostic_records_identical',all(sha(outputs/'normal'/p)==sha(outputs/'normal_parallel'/p)for p in columns))
  check('diagnostic_opt_in_does_not_change_any_budget_or_state',sha(outputs/'normal/udt_step_budget.csv')==sha(outputs/'no_trace/udt_step_budget.csv')and not(outputs/'no_trace/udt_negative_cell_summary.csv').exists())
 for mode in old_modes:
  code=int((outputs/'legacy'/mode/'exit_code.txt').read_text());check('legacy_budget_'+mode,(code==0)==(mode in ['normal','dry','negative']))
 for p in sorted(inputs.glob('*.nc')):
  code=int((outputs/'rain'/p.stem/'exit_code.txt').read_text());success=p.stem not in ['bad_date','partial_datetime','wrong_unit','non_utc'];ok=(code==0)==success
  if success and code==0:
   with Dataset(p)as d:expected=np.rint(np.array(d['time'][:])*60)
   v=np.fromstring((outputs/'rain'/p.stem/'stdout.txt').read_text(),sep=' ');ok &=np.array_equal(v[3:3+int(v[0])],expected)
  check('rain_reader_'+p.stem,ok)
 binding=json.loads((out/'source_binding.json').read_text());materials={'sha256:'+x['digest']['sha256']for x in meta['buildx.build.provenance']['materials']}
 check('exact_local_builder_and_runtime_base_ids',materials=={binding['base_builder_image_id'],binding['base_runtime_image_id']})
 prov=json.loads((report/'provenance.json').read_text());check('original_forcings_topology_policies_and_bindings_unchanged',all(sha(ROOT/p)==h for p,h in prov['protected_sha256'].items()))
 summary={'all_checks_passed':all(c['pass']for c in checks),'checks':checks,'new_image_id':image,'new_patch_sha256':sha(out/'negative_trace_full.patch'),'source_commit':COMMIT,'production_linked_prescribed_fixtures':len(modes),'legacy_budget_regressions':len(old_modes),'rain_reader_regressions':len(list(inputs.glob('*.nc'))),'whole_domain_hydrodynamic_runs':0,'solver_initialize_calls':0,'solver_update_calls':0,'old_control_times_backfilled':False,'real_negative_causes_resolved':False,'training':0,'batch_58':0,'formal_reference_release':False}
 dump(report/'validation.json',summary);dump(out/'solver_binding.json',{'image_tag':TAG,'image_id':image,'source_binding_sha256':sha(out/'source_binding.json'),'patch_sha256':sha(out/'negative_trace_full.patch'),'runtime_binaries':(outputs/'binaries.sha256').read_text(),'production_interface_qualified':summary['all_checks_passed'],'hydraulic_end_to_end_qualified':False,'trace_activation_requires_manifest_metadata':True})
 print(json.dumps({'checks':len(checks),'all_passed':summary['all_checks_passed'],'image_id':image}))
 if not summary['all_checks_passed']:raise SystemExit(1)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--validate-only',action='store_true');main(a.parse_args().validate_only)
