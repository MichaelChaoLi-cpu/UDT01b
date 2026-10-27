"""Prepare pinned full CPU solver build context; never initialize a solver."""
from pathlib import Path
import json,hashlib,tarfile,shutil,subprocess,difflib
ROOT=Path(__file__).resolve().parents[2]
OUT='data/processed/three_ward_cpu_solver_build_v1'
REPORT='data/exp/data-preprocessing/three_ward_cpu_solver_build'
PREVIOUS='data/processed/three_ward_second_clock_budget_v1'
COMMIT='1f1d8286520a8709b2e513854dff18a3616583db'
BASE_IMAGE='deltares/sfincs-cpu@sha256:4301bffc74f3cb3f334be5d21aa75dfe5742a16c9838942cdfc2f870859fb404'
TAG='udt01b/sfincs-cpu:galibier-second-clock-budget-v1'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def main(root=ROOT):
 root=Path(root);out=root/OUT;report=root/REPORT;context=out/'context'
 if(out/'freeze.json').exists():raise FileExistsError('Frozen build: create a new version')
 out.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True);context.mkdir(exist_ok=True)
 archive=out/'upstream_source.tar.gz';treefile=out/'upstream_git_tree.json'
 if not archive.exists():shutil.copyfile('/private/tmp/udt01b-sfincs-full-pinned.tar.gz',archive)
 if not treefile.exists():shutil.copyfile('/private/tmp/sfincs_source_tree.json',treefile)
 tree=json.loads(treefile.read_text());assert tree['sha']==COMMIT and not tree['truncated']
 expected={e['path']:e['sha']for e in tree['tree']if e['type']=='blob'}
 old=json.loads((root/PREVIOUS/'source_manifest.json').read_text())
 for e in old['base_sources']:assert expected['source/src/'+e['file']]==e['git_blob_sha1']
 verified=[]
 with tarfile.open(archive)as tar:
  for e in tar.getmembers():
   rel='/'.join(e.name.split('/')[1:])
   if not e.isfile()or rel not in expected:continue
   b=tar.extractfile(e).read();git=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
   assert git==expected[rel],rel
   if rel.startswith('source/'):
    verified.append({'path':rel,'git_blob_sha1':git,'sha256':hashlib.sha256(b).hexdigest(),'size':len(b)})
    p=context/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);p.chmod(e.mode&0o777)
  assert len(verified)>200
 # Preserve source-level patch and its frozen source identity; change only build integration.
 patch=root/PREVIOUS/'sfincs_second_clock_budget.patch'
 (context/'source_patch.diff').write_bytes(patch.read_bytes())
 r=subprocess.run(['/usr/bin/patch','--batch','-p1','-i',str(patch)],cwd=context,capture_output=True,text=True)
 assert r.returncode==0,r.stdout+r.stderr
 for name,h in old['patched_sha256'].items():assert sha(context/'source/src'/name)==h,name
 make=context/'source/src/Makefile.am';before=make.read_text()
 anchor='\tsfincs_data.f90 \\\n';assert before.count(anchor)==1
 after=before.replace(anchor,anchor+'\tudt_rain_clock.f90 \\\n\tudt_clock_budget.f90 \\\n')
 make.write_text(after)
 (out/'build_integration.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/source/src/Makefile.am',tofile='b/source/src/Makefile.am')))
 for p in (root/'src/preprocessing/sfincs_cpu_build_v1').iterdir():
  if p.is_file():shutil.copyfile(p,context/p.name)
 dump(out/'source_binding.json',{'commit':COMMIT,'archive_sha256':sha(archive),'git_tree_sha256':sha(treefile),'verified_source_blobs':len(verified),'source_files':verified,'frozen_interface_patch_sha256':sha(patch),'build_integration_patch_sha256':sha(out/'build_integration.patch'),'final_source_sha256':{str(p.relative_to(context)):sha(p)for p in sorted((context/'source').rglob('*'))if p.is_file()},'base_image':BASE_IMAGE,'target_platform':'linux/amd64','tag':TAG,'original_inputs_modified':False})
 (context/'SOURCE_BINDING.json').write_bytes((out/'source_binding.json').read_bytes())
 protected={}
 for folder in [root/PREVIOUS,root/'data/processed/three_ward_minimum_scenarios_v1',root/'data/processed/three_ward_reference_contract_v1']:
  for p in folder.rglob('*'):
   if p.is_file():protected[str(p.relative_to(root))]=sha(p)
 oldprov=json.loads((root/'data/exp/data-preprocessing/three_ward_second_clock_budget/provenance.json').read_text())
 dump(report/'provenance.json',{'protected_sha256':protected,'existing_result_stat_fingerprints':oldprov['existing_result_stat_fingerprints']})
 print(json.dumps({'context':str(context),'verified_blobs':len(verified),'platform':'linux/amd64','image_tag':TAG}))
if __name__=='__main__':main()
