"""Build a reviewable, pinned source patch; never invoke a hydraulic solver."""
from pathlib import Path
import json, hashlib, difflib, shutil
ROOT=Path(__file__).resolve().parents[2]
BASE='data/processed/three_ward_clock_budget_audit'
OUT='data/processed/three_ward_second_clock_budget_v1'
REPORT='data/exp/data-preprocessing/three_ward_second_clock_budget'
COMMIT='1f1d8286520a8709b2e513854dff18a3616583db'
MODULES=Path('src/preprocessing/sfincs_second_clock_budget_v1')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def once(s,old,new):
 if s.count(old)!=1:raise ValueError('Nonunique patch anchor: '+old[:100])
 return s.replace(old,new,1)
def main(root=ROOT):
 root=Path(root);out=root/OUT;report=root/REPORT
 if (out/'freeze.json').exists():raise FileExistsError('Frozen: create another version')
 out.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
 before=out/'base';after=out/'patched';before.mkdir(exist_ok=True);after.mkdir(exist_ok=True)
 manifest=json.loads((root/BASE/'source_manifest.json').read_text())
 extra_blobs={'sfincs_data.f90':'c5ada99dd20cd10c1435a3888275b41084060f6d','sfincs_domain.f90':'d754dee73c41cf68653037fe6900fea3c1e989e1','sfincs_momentum.f90':'66f9a219297a8c21d20fb2effb2f924452acded4'}
 for entry in manifest:
  p=root/BASE/'source_snapshots'/entry['file'];assert sha(p)==entry['sha256'];shutil.copyfile(p,before/p.name)
 for name in ['sfincs_data.f90','sfincs_domain.f90','sfincs_momentum.f90']:
  p=before/name
  if not p.exists():p=Path('/private/tmp/udt01b-pinned-'+name)
  b=p.read_bytes();entry={'path':'source/src/'+name,'sha':extra_blobs[name]}
  assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==entry['sha']
  if p.resolve()!=(before/name).resolve():shutil.copyfile(p,before/name)
  manifest.append({'file':name,'commit':COMMIT,'git_blob_sha1':entry['sha'],'sha256':sha(p),'official_path':entry['path']})
 for p in before.iterdir():shutil.copyfile(p,after/p.name)
 name='sfincs_ncinput.F90';s=(after/name).read_text();a=s.index('   subroutine read_netcdf_ampr_data()');b=s.index('   end subroutine',a)
 block=s[a:b]
 block=once(block,'   use sfincs_date','   use udt_rain_clock\n   use sfincs_date')
 block=once(block,'   integer nt','   integer nt\n   real*8, allocatable :: udt_raw_rain_times(:)')
 block=once(block,'   allocate(ampr_times(ampr_nt))','   allocate(ampr_times(ampr_nt))\n   allocate(udt_raw_rain_times(ampr_nt))')
 block=once(block,'net_file_ampr%time_varid, ampr_times(:)','net_file_ampr%time_varid, udt_raw_rain_times(:)')
 block=once(block,'   ampr_times = convert_fewsdate(ampr_times, ampr_nt, treftimefews, trefstr)',
 '''   call udt_decode_rain_seconds(udt_raw_rain_times, treftimefews, trefstr, ampr_times)
   call udt_write_rain_trace(udt_raw_rain_times, treftimefews, ampr_times)
   deallocate(udt_raw_rain_times)''')
 (after/name).write_text(s[:a]+block+s[b:])
 name='sfincs_continuity.f90';s=(after/name).read_text()
 s=once(s,'module sfincs_continuity','module sfincs_continuity\n use udt_clock_budget, only: udt_budget_begin,udt_budget_cell,udt_budget_finish,udt_budget_source_added')
 a=s.index('   subroutine compute_water_levels_subgrid');b=s.index('   end subroutine',a);block=s[a:b]
 block=once(block,'   if (wavemaker) then','   call udt_budget_begin()\n   if (wavemaker) then')
 block=once(block,'         if (precip .or. use_qext) then','         call udt_budget_cell(nm,dvol,a,dt)\n         if (precip .or. use_qext) then')
 block=once(block,'            dvol = dvol + dzsdt * a * dt', '            dvol = dvol + dzsdt * a * dt\n            call udt_budget_source_added(nm,dvol)')
 block+='   call udt_budget_finish(t,dt)\n'
 (after/name).write_text(s[:a]+block+s[b:])
 name='sfincs_lib.f90';s=(after/name).read_text()
 s=once(s,'   use sfincs_continuity','   use udt_clock_budget, only: udt_budget_initialize,udt_budget_capture,udt_budget_close\n   use sfincs_continuity')
 s=once(s,'   call initialize_output(tmapout, tmaxout, thisout, trstout)',
  '   call udt_budget_initialize(t)\n   call initialize_output(tmapout, tmaxout, thisout, trstout)')
 old='         call write_output(tout, write_map, write_his, write_max, write_rst, ntmapout, ntmaxout, nthisout, tloopoutput)'
 new='''         call udt_budget_capture(tout,t,dt,int(nt,8),write_map,write_his,write_max,write_rst,ntmapout,nthisout,ntmaxout)
'''+old
 s=once(s,old,new)
 old='         call write_output(t, .true., .true., .true., .false., ntmapout + 1, ntmaxout, nthisout + 1, tloopoutput)'
 s=once(s,old,'         call udt_budget_capture(t,t,dt,int(nt,8),.true.,.true.,.true.,.false.,ntmapout+1,nthisout+1,ntmaxout)\n'+old)
 # Capture implicit final peak writes through the finalization boundary as well.
 s=once(s,'   call finalize_output(t, ntmaxout, tloopoutput, tmaxout)',
 '''   call finalize_output(t, ntmaxout, tloopoutput, tmaxout)
   call udt_budget_close()''')
 (after/name).write_text(s)
 name='sfincs_output.f90';s=(after/name).read_text()
 s=once(s,'module sfincs_output','module sfincs_output\n   use udt_clock_budget, only: udt_budget_capture_final')
 s=once(s,'      call write_output(t,.false.,.false.,.true.,.false.,0,ntmaxout,0,tloopoutput)',
  '      call udt_budget_capture_final(t,ntmaxout)\n      call write_output(t,.false.,.false.,.true.,.false.,0,ntmaxout,0,tloopoutput)')
 s=once(s,'      call write_output(t2,.false.,.false.,.true.,.false.,0,ntmaxout,0,tloopoutput)',
  '      call udt_budget_capture_final(t2,ntmaxout)\n      call write_output(t2,.false.,.false.,.true.,.false.,0,ntmaxout,0,tloopoutput)')
 (after/name).write_text(s)
 for p in (root/MODULES).glob('*.f90'):shutil.copyfile(p,after/p.name)
 diff=[]
 for p in sorted(after.iterdir()):
  original=before/p.name
  diff.extend(difflib.unified_diff(original.read_text().splitlines(True) if original.exists() else [],p.read_text().splitlines(True),fromfile='a/source/src/'+p.name,tofile='b/source/src/'+p.name))
 (out/'sfincs_second_clock_budget.patch').write_text(''.join(diff))
 dump(out/'source_manifest.json',{'base_commit':COMMIT,'base_sources':manifest,'patched_sha256':{p.name:sha(p) for p in sorted(after.iterdir())},'original_sources_immutable':True})
 # Original source/library preservation and native files: compare again after all tests.
 protected={}
 for folder in [root/'data/processed/three_ward_minimum_scenarios_v1',root/'data/processed/three_ward_reference_contract_v1',root/BASE]:
  for p in folder.rglob('*'):
   if p.is_file():protected[str(p.relative_to(root))]=sha(p)
 for path in ['src/preprocessing/load_three_ward_minimum_scenarios.py','src/preprocessing/sfincs_clock_budget_interface.py']:
  protected[path]=sha(root/path)
 old=json.loads((root/'data/exp/data-preprocessing/three_ward_clock_budget_audit/provenance.json').read_text())
 dump(report/'provenance.json',{'protected_sha256':protected,'existing_result_stat_fingerprints':old['existing_result_stat_fingerprints']})
 dump(report/'decisions.json',{'human_record':'MILI-D-20261015-008','scope':'50 frozen inputs; no training, no new hydrodynamic responses','new_variable_selection':False,'candidate_status_unchanged':True,'implementation_choices_are_agent_choices':True})
 (out/'BUILD.md').write_text('''# CPU source integration, not an operational image\n\nBase: SFINCS Galibier commit `'''+COMMIT+'''`. Apply the unified patch to that exact complete source tree. Add the two new modules to the build source list, after the original data/date modules and before continuity/NetCDF input/library. Enable Fortran preprocessing for the diagnostic module. CPU/OpenMP only; compiling with OpenACC is rejected at initialization. The complete upstream tree, Fortran NetCDF dependency and image build are not supplied here.\n\nThe existing NetCDF exporter and curves remain unchanged. Its time variable is read directly into real64 before conversion; this scoped transport accepts strictly increasing integer seconds representable exactly by the downstream real32 clock. Minutes units, an explicit UTC reference, and at least two finite knots are required. Fractional seconds and other units fail closed. Existing precipitation values, coordinate convention, halo, row flip and stage transport are unchanged.\n\nEach future run requires a manifest-bound identity file generated by the staging helper. Diagnostics are additive CSV sidecars; original solver time labels remain unchanged. The per-step ledger has a true initial row, actual time/dt/index, signed storage, gross native boundary exchange, applied net rainfall, horizontal increments, source-addition rounding, residual and online negativity. Output rows bind map/history/peak record numbers to exactly the accounted step. The final-status file reports the last real state, independently of an instability sentinel. Grouped river budgets and a complete production image receipt remain pending.\n\nNo new water-flow simulation, BMI update or training is part of this package. Isolated fixtures exercise transport and bookkeeping only; the patched continuity routine is compiled but never executed.\n''')
 print(out)
if __name__=='__main__':main()
