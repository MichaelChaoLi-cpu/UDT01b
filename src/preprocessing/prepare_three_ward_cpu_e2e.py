"""Stage exactly two existing open controls; no event-library edits or solver calls."""
from pathlib import Path
import sys,json,hashlib,shutil,os
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_sfincs_cpu_build import sha,dump,COMMIT
OUT='data/processed/three_ward_cpu_e2e_controls_v1'
REPORT='data/exp/data-preprocessing/three_ward_cpu_e2e_controls'
BUILD='data/processed/three_ward_cpu_solver_build_v1'
CASES=['dry_raised','rain60_raised']

def main(root=ROOT):
 root=Path(root);out=root/OUT;report=root/REPORT
 if(out/'freeze.json').exists():raise FileExistsError('Frozen: use new version')
 out.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
 binding=json.loads((root/BUILD/'solver_image_binding.json').read_text());image=binding['local_image_id']
 catalog=json.loads((root/'data/processed/three_ward_minimum_scenarios_v1/catalog.json').read_text())
 astro_source=root/BUILD/'context/source/src/sfincs_boundaries.f90';text=astro_source.read_text()
 assert "nr_tidal_components = 0"in text and "if (bcafile(1:4) /= 'none' .and. nbnd > 0 .and. use_bcafile) then"in text
 assert text.index('nr_tidal_components = 0')<text.index('call update_nodal_factors')<text.index('   end subroutine',text.index('call update_nodal_factors'))
 inp_source=(root/BUILD/'context/source/src/sfincs_input.f90').read_text();assert "'bcafile',bcafile,'none'"in inp_source
 oldprov=json.loads((root/'data/exp/data-preprocessing/three_ward_cpu_solver_build/provenance.json').read_text())
 protected=dict(oldprov['protected_sha256']);fingerprints=dict(oldprov['existing_result_stat_fingerprints']);records=[];astro=[];source_hash_cache={}
 for cid in CASES:
  folder=root/'data/processed/three_ward_sfincs_framework/adaptive_sensitivity'/(cid+'_corrected_10m')
  ex=json.loads((folder/'execution.json').read_text());cfg={k.strip():v.strip()for line in(folder/'sfincs.inp').read_text().splitlines()if '='in line for k,v in[line.split('=',1)]}
  assert cfg.get('bcafile','none')=='none'and cfg.get('use_bcafile','1')in ['1','true','True']
  assert cfg['dx']==cfg['dy']=='10'and cfg['ampr_block']=='0'and cfg['dtwnd']=='1.0'
  run_id=cid+'__e2e_regular_10m';dest=out/'runs'/run_id
  if dest.exists():raise FileExistsError('Refuse to overwrite control staging '+run_id)
  dest.mkdir(parents=True)
  hashes={}
  for name,expected in ex['input_sha256'].items():
   p=folder/name;key=(p.stat().st_dev,p.stat().st_ino,p.stat().st_size,p.stat().st_mtime_ns)
   if key not in source_hash_cache:source_hash_cache[key]=sha(p)
   assert source_hash_cache[key]==expected,name
   if name=='sfincs_subgrid.nc':os.link(p,dest/name)
   else:shutil.copyfile(p,dest/name)
   hashes[name]=expected;protected[str(p.relative_to(root))]=expected
  geometry={k:v for k,v in hashes.items()if k not in ['roof_surface_precip.nc','sfincs.bzs']}
  gh=hashlib.sha256(json.dumps(geometry,sort_keys=True,separators=(',',':')).encode()).hexdigest()
  case=next(c for c in catalog['cases']if c['case_id']==cid)
  identity={'schema_version':'second-clock-budget-interface-v1','run_id':run_id,'parent_id':case['parent_id'],'family_id':case['family_id'],'split':case['split'],'forcing_sha256':case['forcing_sha256'],'geometry_sha256':gh,'source_commit':COMMIT,'patch_sha256':binding['interface_patch_sha256'],'solver_digest':binding['oci_index_digest']}
  (dest/'udt_run_identity.txt').write_text('\n'.join(k+'='+v for k,v in identity.items())+'\n')
  dump(dest/'udt_run_identity.json',identity)
  record={'case_id':cid,'run_id':run_id,'folder':str(dest.relative_to(root)),'source_folder':str(folder.relative_to(root)),'image_id':image,'image_digest':binding['oci_index_digest'],'identity':identity,'input_sha256':hashes,'large_readonly_source':str(folder/'sfincs_subgrid.nc'),'status':'staged_user_authorized_two_control_e2e_only','expected_horizon_s':14400,'library_descendant':False}
  records.append(record)
  astro.append({'case_id':cid,'bcafile_effective':'none','use_bcafile_default':True,'tidal_component_count_from_initialization_branch':0,'nodal_table_call_reachable':False,'evidence_type':'fixed-source guard plus unchanged input; verify absence of tidal loading in runtime log','warning_retained_in_source':True})
 dump(out/'run_manifest.json',{'runs':records,'case_count':2,'control_family':'CONTROL-LEGACY','independent_new_parent_count':0,'pending_58_scenarios_authorized':False,'training':False})
 dump(report/'astronomical_tide_audit.json',{'source_commit':COMMIT,'boundary_source_sha256':sha(astro_source),'cases':astro,'warning':'Delft3D astro.f90 l(235) with declared size234; unmodified; current no-bca branch does not call table','future_astronomical_tides_qualified':False})
 dump(report/'provenance.json',{'protected_sha256':protected,'existing_result_stat_fingerprints':fingerprints})
 dump(report/'decisions.json',{'human_record':'MILI-D-20261015-009','user_scope':'two existing controls, full10m, actual clocks/applied budgets/outputs; no training, no58-case batch','agent_case_selection':CASES,'selection_reason':'matched raised boundary with dry and rain inputs; same static geometry','new_input_curves':False,'candidate_variable_status_unchanged':True})
 print(json.dumps({'cases':CASES,'image':image,'astro_branch_called':False}))
if __name__=='__main__':main()
