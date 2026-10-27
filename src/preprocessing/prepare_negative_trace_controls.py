"""Stage exactly two authorized legacy-control replays with frozen trace metadata."""
from pathlib import Path
import sys,json,shutil
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_sfincs_cpu_build import sha,dump
OUT='data/processed/three_ward_negative_trace_controls_v1'
REPORT='data/exp/data-preprocessing/three_ward_negative_trace_controls'
CASES=['dry_raised','rain60_raised']
PRIOR='data/processed/three_ward_cpu_e2e_controls_v1'
INTERFACE='data/processed/three_ward_negative_trace_interface_v1'

def main():
 out=ROOT/OUT;report=ROOT/REPORT
 if(out/'run_manifest.json').exists():raise FileExistsError('Existing staging retained')
 out.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
 interface=ROOT/INTERFACE;binding=json.loads((interface/'solver_binding.json').read_text());meta=json.loads((interface/'metadata_binding.json').read_text())
 assert binding['production_interface_qualified']and not binding['hydraulic_end_to_end_qualified']
 manifest=json.loads((ROOT/PRIOR/'run_manifest.json').read_text());protected={};runs=[];cache={}
 def hashed(p):
  st=p.stat();key=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
  if key not in cache:cache[key]=sha(p)
  protected[str(p.relative_to(ROOT))]=cache[key];return cache[key]
 for old in manifest['runs']:
  assert old['case_id']in CASES
  source=ROOT/old['folder'];dest=out/'runs'/(old['case_id']+'__negative_trace_10m');dest.mkdir(parents=True)
  inputs={}
  for name,expected in old['input_sha256'].items():
   p=source/name;assert hashed(p)==expected,name
   if name=='sfincs_subgrid.nc':(dest/name).hardlink_to(p)
   else:shutil.copyfile(p,dest/name)
   inputs[name]=expected
  assert inputs['sfincs.ind']==meta['source_index_sha256']and inputs['sfincs.msk']==meta['source_mask_sha256']
  for name,key in [('udt_negative_native_map.csv','map_sha256'),('udt_negative_trace.request','request_sha256')]:
   p=interface/'whole_domain_metadata'/name;assert hashed(p)==meta[key];shutil.copyfile(p,dest/name);inputs[name]=meta[key]
  ident=dict(old['identity']);ident['run_id']=dest.name;ident['solver_digest']=binding['image_id'];ident['patch_sha256']=binding['patch_sha256']
  (dest/'udt_run_identity.txt').write_text('\n'.join(k+'='+v for k,v in ident.items())+'\n');dump(dest/'udt_run_identity.json',ident)
  runs.append({'case_id':old['case_id'],'run_id':dest.name,'folder':str(dest.relative_to(ROOT)),'source_folder':old['folder'],'previous_e2e_folder':old['folder'],'image_id':binding['image_id'],'image_digest':binding['image_id'],'identity':ident,'input_sha256':inputs,'original_source_input_sha256':old['original_source_input_sha256'],'scientific_input_sha256':old['input_sha256'],'large_readonly_source':str(source/'sfincs_subgrid.nc'),'library_descendant':False,'trace_required':True,'new_parent_event':False})
 for p in [interface/'solver_binding.json',interface/'freeze.json',ROOT/PRIOR/'freeze.json',ROOT/'data/processed/three_ward_negative_volume_audit_v1/freeze.json',ROOT/'data/processed/three_ward_reference_contract_v1/reference_registry.json',ROOT/'data/processed/three_ward_minimum_scenarios_v1/run_manifest.json',ROOT/'src/preprocessing/reference_label_policy.py']:hashed(p)
 dump(out/'run_manifest.json',{'runs':runs,'case_count':2,'pending_58_scenarios_authorized':False,'training':False,'control_family':'CONTROL-LEGACY','required_trace_schema':'negative-cell-face-v1','reference_release':False})
 dump(report/'provenance.json',{'protected_sha256':protected,'scientific_inputs_unchanged':True,'new_parent_events':0,'human_record':'MILI-D-20261015-012'})
 print(json.dumps({'staged_cases':CASES,'image':binding['image_id']}))
if __name__=='__main__':main()
