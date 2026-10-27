"""Existing-result timing forensics and isolated Fortran date-interface tests.
No SFINCS executable/library update, hydraulic run, ML training or label edits.
"""
from pathlib import Path
import os,sys,json,csv,hashlib,shutil,subprocess,tempfile,re,argparse
import numpy as np
from netCDF4 import Dataset
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from sfincs_clock_budget_interface import decode_legacy_fews_minutes,candidate_integer_second_decoder,imported_rate,scheduler_curve,forcing_only_capped_replay,LEDGER_SCHEMA
from reference_label_policy import integrate_rate_at
OUT='data/processed/three_ward_clock_budget_audit'
REPORT='data/exp/data-preprocessing/three_ward_clock_budget_audit'
OLD='data/exp/data-preprocessing/three_ward_reference_contract'
LIB='data/processed/three_ward_minimum_scenarios_v1'
COMMIT='1f1d8286520a8709b2e513854dff18a3616583db'


def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def write_csv(p,rows):
 if not rows:return
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with open(p,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def npz(p):
 with np.load(p,allow_pickle=False) as d:return {k:d[k] for k in d.files}
def source_excerpt(path,first,last):
 lines=Path(path).read_text().splitlines();return [{'line':i,'text':lines[i-1]} for i in range(first,min(last,len(lines))+1)]

DRIVER='''program date_probe
 use sfincs_date
 implicit none
 real*4,allocatable :: x(:),y(:)
 character*41 :: units
 character*15 :: ref
 integer :: i,n
 read(*,*)n
 allocate(x(n))
 read(*,*)x
 read(*,'(A)')units
 ref='20200101 000000'
 y=convert_fewsdate(x,n,units,ref)
 do i=1,n
  write(*,'(F18.7,1X,F18.7,1X,I10)')x(i),y(i),nint(x(i)*60.0)
 enddo
end program
'''


def build_probe(sources,driver):
 compiler=shutil.which('gfortran',path='/opt/homebrew/bin:/usr/bin:/bin')
 if compiler is None:raise RuntimeError('No local compiler for isolated converter test; do not infer a pass')
 env=os.environ.copy();env['PATH']='/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin'
 temp=tempfile.TemporaryDirectory(prefix='udt01b-date-unit-',dir='/private/tmp');folder=Path(temp.name);binary=folder/'date_probe'
 result=subprocess.run([compiler,'-O0','-ffree-line-length-none','-J',str(folder),str(sources/'sfincs_date.f90'),str(driver),'-o',str(binary)],env=env,capture_output=True,text=True)
 if result.returncode:temp.cleanup();raise RuntimeError(result.stderr)
 return temp,binary,{'compiler':subprocess.run([compiler,'--version'],capture_output=True,text=True).stdout.splitlines()[0],
  'build_warnings':result.stderr,'source_unmodified':True,'hydrodynamic_executable_built':False,'hydrodynamic_executable_run':False}


def probe(binary,seconds,units='minutes since 2020-01-01 00:00:00.0 +0000'):
 minutes=np.asarray(seconds,dtype='f8')/60
 text=str(len(minutes))+'\n'+' '.join(format(x,'.17g') for x in minutes)+'\n'+units+'\n'
 r=subprocess.run([str(binary)],input=text,capture_output=True,text=True)
 if r.returncode:raise RuntimeError(r.stderr)
 values=np.array([[float(x) for x in line.split()] for line in r.stdout.splitlines()])
 assert len(values)==len(seconds)
 return values[:,1],values[:,2]


def main(root=ROOT,date_source=Path('/private/tmp/udt01b-sfincs_date-pinned.f90'),tree_source=Path('/private/tmp/sfincs_source_tree.json')):
 root=Path(root);out=root/OUT;report=root/REPORT;out.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
 if (out/'freeze.json').exists():raise FileExistsError('Existing diagnostic package frozen; validate or create a new version')
 tree=json.loads(tree_source.read_text());assert tree['sha']==COMMIT
 blobs={e['path']:e['sha'] for e in tree['tree'] if e['type']=='blob'}
 src=out/'source_snapshots';src.mkdir(exist_ok=True)
 local=root/'data/processed/three_ward_sfincs_scenarios/solver_provenance'
 mapping={f:local/f for f in ['sfincs_lib.f90','sfincs_meteo.f90','sfincs_ncoutput.F90','sfincs_continuity.f90','sfincs_crosssections.f90','sfincs_boundaries.f90']}
 mapping|={'sfincs_ncinput.F90':root/'data/processed/three_ward_fixed_runoff/sfincs_ncinput.F90','sfincs_date.f90':date_source,
  'sfincs_output.f90':Path('/private/tmp/sfincs_output.f90'),'sfincs_timestep_analysis.f90':Path('/private/tmp/sfincs_timestep_analysis.f90')}
 evidence=[]
 for name,p in mapping.items():
  b=p.read_bytes();git=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest();path='source/src/'+name
  assert blobs[path]==git,(name,git,blobs[path]);(src/name).write_bytes(b)
  evidence.append({'file':name,'official_path':path,'commit':COMMIT,'git_blob_sha1':git,'sha256':sha(p),'official_url':f'https://raw.githubusercontent.com/Deltares/SFINCS/{COMMIT}/{path}'})
 dump(out/'source_manifest.json',evidence)
 dump(out/'git_tree_subset.json',{'commit':COMMIT,'entries':[{k:e[k] for k in ['path','sha']} for e in tree['tree'] if e.get('path') in [x['official_path'] for x in evidence]]})
 driver=out/'date_probe.f90';driver.write_text(DRIVER)
 temp,binary,compile_info=build_probe(src,driver)
 protected={str(p.relative_to(root)):sha(p) for folder in [root/'data/processed/three_ward_reference_contract_v1',root/LIB,root/'data/processed/three_ward_surrogate_inputs',root/'data/processed/three_ward_common_depth_labels'] for p in folder.iterdir() if p.is_file()}
 olddiag=json.loads((root/OLD/'run_diagnostics.json').read_text());clockrows=[];onset=[];pointrows=[];sensitivity=[];source_stat={}
 for tag,d in olddiag.items():
  case=tag.removesuffix('_corrected_10m').removesuffix('_adaptive_10_5m');grid='corrected_10m' if tag.endswith('_corrected_10m') else 'adaptive_10_5m'
  folder=root/'data/processed/three_ward_sfincs_framework'/('adaptive_connectivity_comparison' if case=='rain60_low' else 'adaptive_sensitivity')/tag
  for file in ['roof_surface_precip.nc','sfincs_his.nc','sfincs_map.nc','sfincs.log','execution.json']:
   p=folder/file;st=p.stat();source_stat[str(p.relative_to(root))]=[st.st_size,st.st_mtime_ns]
  with Dataset(folder/'roof_surface_precip.nc') as nc:raw_sec=np.asarray(nc['time'][:],dtype='f8')*60;units=nc['time'].units
  decoded,rounded=probe(binary,raw_sec,units);assert np.array_equal(decoded,decode_legacy_fews_minutes(raw_sec/60))
  peak=100. if case=='rain100_low' else 0. if case=='dry_raised' else 60.
  rates=np.array([0,0,peak,peak,0,0],dtype='f8');ft,fr=scheduler_curve(decoded,rates)
  checkpoints=list(csv.DictReader((root/OLD/(tag+'_budget_checkpoints.csv')).open()));ct=np.array([float(x['time_s']) for x in checkpoints]);mass_factor=38976800/3.6e6
  pr=integrate_rate_at(ft,fr,ct)*mass_factor;nom=integrate_rate_at(raw_sec,rates,ct)*mass_factor
  row=checkpoints[list(ct).index(3600.)];observed=float(row['residual_m3'])
  replay=forcing_only_capped_replay(rate_scale=peak*mass_factor)
  prediction=replay['scaled_applied_rain'];delta=observed-prediction
  onset.append({'run':tag,'rain_peak_mm_h':peak,'nominal_onset_residual_m3':observed,'continuous_imported_meteo_rain_at_nominal3600_m3':float(pr[list(ct).index(3600.)]),
   'conditional_capped_step_rain_m3':prediction,'onset_residual_minus_conditional_rain_m3':delta,
   'assumed_actual_onset_state_time_s':replay['assumed_actual_time_s'],'step_trace_recovered':False,
   'interpretation':'forcing-only kernel reproduction, not fitted time shift or hydrodynamic rerun'})
  for i,x in enumerate(raw_sec):clockrows.append({'run':tag,'index':i,'file_seconds':float(x),'official_decoded_seconds':float(decoded[i]),'undeployed_integer_second_candidate':float(rounded[i]),'collision_with_previous':bool(i>0 and decoded[i]==decoded[i-1])})
  for i,t in enumerate(ct):sensitivity.append({'run':tag,'nominal_time_s':float(t),'original_nominal_residual_m3':float(checkpoints[i]['residual_m3']),
   'imported_continuous_rain_minus_file_rain_m3':float(pr[i]-nom[i]),'unresolved_after_input_semantics_difference_m3':float(checkpoints[i]['residual_m3'])-float(pr[i]-nom[i]),
   'actual_applied_rain_known':False,'reference_admission_changed':False})
  with Dataset(folder/'sfincs_his.nc') as nc:
   ht=np.asarray(nc['time'][:],dtype='f8')
   n_onset=int(np.searchsorted(ht,3600.,side='right'));qnet=np.zeros(n_onset)
   for start in range(0,n_onset,1000):
    q=np.ma.asarray(nc['crosssection_discharge'][start:min(start+1000,n_onset)],dtype='f8').filled(np.nan)
    assert np.all(np.isfinite(q));qnet[start:start+len(q)]=q.sum(1)
   delta_t=np.diff(ht[:n_onset]);right_net=float(qnet[1:]@delta_t);trap_net=float(((qnet[:-1]+qnet[1:])*.5)@delta_t)
   onset[-1]['boundary_right_minus_trapezoid_at_onset_m3']=right_net-trap_net
   onset[-1]['remaining_after_forcing_and_boundary_rule_m3']=delta+right_net-trap_net
   onset[-1]['note']='Right-rectangle comparison at onset only, conditional capped-step probe; no change to frozen full-event trapezoid policy.'
   ix=np.flatnonzero((ht>=3597.5)&(ht<=3602.5))
   h=np.ma.asarray(nc['point_h'][:]).filled(np.nan)[:,0]
   for j in ix:pointrows.append({'run':tag,'nominal_time_s':float(ht[j]),'existing_single_station_h_m':float(h[j])})
  with Dataset(folder/'sfincs_map.nc') as nc:avg=float(nc['average_dt'][0]);map_vars=list(nc.variables)
  log=(folder/'sfincs.log').read_text();m=re.search(r'Average time step \(s\)\s*:\s*([0-9.]+)',log);logged=float(m.group(1))
  d['clock_summary']={'netcdf_average_dt_value':avg,'log_average_dt_s':logged,'actual_state_time_variable_present':'actual_state_time' in map_vars,
    'note':'Pinned finalize writes accumulated real4 dtavg before dividing for log; neither the mislabeled scalar nor averaged per-cell timestep maps recover a step trace.'}
  print('EXISTING ONSET',tag,'observed',round(observed,6),'forcing_only',round(prediction,6),'unresolved',round(delta,6),flush=True)
 write_csv(report/'existing_rain_time_decode.csv',clockrows);write_csv(report/'onset_residual_explanation.csv',onset);write_csv(report/'nominal_prefix_sensitivity.csv',sensitivity);write_csv(report/'existing_station_onset.csv',pointrows)
 dump(report/'existing_clock_inventory.json',{k:v['clock_summary'] for k,v in olddiag.items()})
 # The same integer-minute bug affects prospective exports. Inputs stay frozen;
 # this is a superseding readiness finding, not a rewrite of previous receipts.
 cat=json.loads((root/LIB/'catalog.json').read_text());newrows=[];newclocks=[];probe_count=8
 for c in cat['cases']:
  if c.get('existing_completed'):continue
  data=npz(root/LIB/c['forcing_file']);t=data['Event Time'].astype('f8');r=data['Physical Rainfall Rate'].astype('f8')
  decoded,candidate=probe(binary,t);probe_count+=1
  assert np.array_equal(decoded,decode_legacy_fews_minutes(t/60))
  ft,fr=scheduler_curve(decoded,r);intended=float(integrate_rate_at(t,r,[14400])[0]/3600);native=float(integrate_rate_at(ft,fr,[14400])[0]/3600)
  relative=abs(native-intended)/max(intended,1e-30) if intended else 0.
  collisions=int(np.sum(np.diff(decoded)==0));advance=float(np.max(t-decoded));restored=bool(np.array_equal(candidate,t))
  newrows.append({'case_id':c['case_id'],'parent_id':c['parent_id'],'split':c['split'],'knots':len(t),'decoded_collisions':collisions,
   'max_knot_advance_s':advance,'intended_depth_mm':intended,'legacy_resampled_depth_mm':native,'relative_depth_change':relative,
   'original_peak_mm_h':float(r.max()),'legacy_resampled_peak_mm_h':float(fr.max()),'integer_second_candidate_restores_knots':restored,
   'transport_status':'hold_legacy_reader_not_qualified','hydrodynamic_response_exists':False})
  newclocks.extend({'case_id':c['case_id'],'index':i,'intended_second':float(x),'decoded_second':float(decoded[i]),'candidate_second':float(candidate[i])} for i,x in enumerate(t))
 # Direct header/unit contrast: changing label to seconds does NOT change a hard-coded*60 reader.
 legacy_header,_=probe(binary,[0,3600,3601,7200,7201,14400],'minutes since 2020-01-01')
 assert np.array_equal(legacy_header,[0,3600,3600,7200,7200,14400])
 wrong_units,_=probe(binary,np.array([0.,3601.])*60,'seconds since 2020-01-01 00:00:00.0 +0000')
 assert np.array_equal(wrong_units,[0,216060])
 write_csv(report/'prospective_export_impact.csv',newrows);write_csv(report/'prospective_knot_decode.csv',newclocks)
 dump(out/'diagnostic_export_schema.json',LEDGER_SCHEMA)
 dump(out/'transport_readiness.json',{'version':'clock-interface-findings-v1','supersedes_scope':'Only transport-readiness inference from former host-roundtrip file preflight; scientific input arrays, family partitions and previous receipts remain unchanged',
  'source_forcing_freeze_sha256':sha(root/LIB/'freeze.json'),'pending_hydraulic_runs_held':58,'new_forcings_audited':50,
  'all_new_exports_need_requalification':True,'official_reader_preserves_fractional_minutes':False,
  'integer_second_candidate':'Source-level arithmetic candidate only; uninstalled and not tested in the official hydrodynamic image',
  'safe_alternatives':['qualify a versioned reader that preserves integer-second or finer times, with exact converter and forcing integration tests','or design a new whole-minute forcing library with explicitly recomputed curves/volumes and new version; cannot silently resample current library'],
  'unsafe_shortcuts':['rename time units to seconds without changing decoder','shift timestamps to fit observed residuals','change only output cadence or relax conservation gates'],
  'policy_or_reference_release_changed':False,'training':False,'new_hydrodynamic_runs':0})
 dump(out/'interface_findings.json',{'source_commit':COMMIT,'date_converter':'real(int(real4_minutes)*60+epoch_offset) truncates minute fractions',
  'meteo_scheduler':'endpoint samples on one-second brackets turn duplicate3600 knots into a3599..3600 ramp',
  'clock':'public sfincs_lib.t/dt exist; write_output forwards scheduled tout only; actual clocks not in existing NetCDF',
  'step_api':'sfincs_update(0.d0) single-step branch exists in pinned library; it was not called here; standalone existing image is not a verified instrumentation interface',
  'budget':'No matched every-step applied-source/boundary ledger found in existing files; signed map volume and instantaneous boundary history remain',
  'average_dt':'NetCDF finalization occurs before dtavg division; scalar stores accumulated float32 dtavg, log reports divided mean',
  'stage_path':'ASCII bzs times read directly into t_bnd; this NetCDF minute-truncation path does not process existing stage files',
  'admission':'All existing strict admission holds and all source masks remain unchanged'})
 snippets={}
 for name,a,b in [('sfincs_date.f90',230,261),('sfincs_lib.f90',410,430),('sfincs_lib.f90',460,480),('sfincs_lib.f90',618,630),('sfincs_lib.f90',696,708),('sfincs_meteo.f90',1128,1170),('sfincs_meteo.f90',1240,1256),('sfincs_meteo.f90',1292,1326),('sfincs_ncoutput.F90',3909,3921),('sfincs_continuity.f90',455,496),('sfincs_boundaries.f90',78,97)]:snippets[name+':'+str(a)]=source_excerpt(src/name,a,b)
 dump(report/'source_evidence.json',snippets);dump(report/'fortran_probe_receipt.json',compile_info|{'official_function_invocations':probe_count+2,'all_python_legacy_decodes_equal_compiled_function':True,'legacy_original_ramp_seconds':[0,3600,3601,7200,7201,14400],'legacy_decoded_seconds':legacy_header.tolist(),'seconds_unit_rename_probe_decoded_seconds':wrong_units.tolist(),'no_hydrodynamic_model_called':True})
 summary={'existing_runs_inspected':8,'isolated_official_function_calls':probe_count+2,'new_forcings_audited':len(newrows),
  'onset_kernel_max_absolute_remaining_m3':max(abs(x['onset_residual_minus_conditional_rain_m3']) for x in onset),
  'onset_kernel_plus_boundary_rule_max_remaining_m3':max(abs(x['remaining_after_forcing_and_boundary_rule_m3']) for x in onset),
  'rainy_onset_min_explained_fraction':min(1-abs(x['onset_residual_minus_conditional_rain_m3'])/abs(x['nominal_onset_residual_m3']) for x in onset if x['rain_peak_mm_h']>0),
  'prospective_cases_with_decoded_collisions':sum(x['decoded_collisions']>0 for x in newrows),
  'prospective_cases_with_relative_depth_change_gt_1e_6':sum(x['relative_depth_change']>1e-6 for x in newrows),
  'prospective_max_relative_depth_change':max(x['relative_depth_change'] for x in newrows),
  'prospective_max_knot_advance_s':max(x['max_knot_advance_s'] for x in newrows),
  'candidate_integer_second_knots_all_restored':all(x['integer_second_candidate_restores_knots'] for x in newrows),
  'new_hydrodynamic_runs':0,'training':0,'sealed_responses_read':0,'strict_reference_holds_preserved':8,'pending_runs_held_for_transport_qualification':58,
  'new_forcing_curves_or_partitions_modified':False,'actual_step_trace_recovered':False,'kernel_replay_is_conditional':True,
  'old_packages_unchanged':all(sha(root/p)==h for p,h in protected.items())}
 dump(report/'summary.json',summary);dump(report/'provenance.json',{'source_manifest_sha256':sha(out/'source_manifest.json'),'protected_sha256':protected,'existing_result_stat_fingerprints':source_stat,
  'new_code_sha256':{str(p.relative_to(root)):sha(p) for p in [Path(__file__),root/'src/preprocessing/sfincs_clock_budget_interface.py']},
  'supplemental_source_retrieved':'Only exact pinned official sfincs_date.f90, Git-blob-verified; no literature search or model-data acquisition'})
 dump(report/'decisions.json',{'human_decision':'MILI-D-20261015-007','scope':'read-only existing-result and isolated timing-interface audit','variables':['Raw Rain Time','Imported Rain Time','Meteo Resampled Rainfall','Nominal Budget Residual','Conditional Forcing Integral','Actual Clock Availability','Applied Step Ledger Requirements'],'is_final_variable':'no','no_training':True,'no_new_hydrodynamic_runs':True})
 temp.cleanup();print(json.dumps(summary),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--date-source',type=Path,default=Path('/private/tmp/udt01b-sfincs_date-pinned.f90'));p.add_argument('--tree-source',type=Path,default=Path('/private/tmp/sfincs_source_tree.json'));a=p.parse_args();main(a.root,a.date_source,a.tree_source)
