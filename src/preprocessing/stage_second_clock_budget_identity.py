"""Stage manifest-bound diagnostic identity only. No solver invocation or release."""
from pathlib import Path
import hashlib,json,re,argparse
ROOT=Path(__file__).resolve().parents[2]
PACKAGE='data/processed/three_ward_second_clock_budget_v1'
LIB='data/processed/three_ward_minimum_scenarios_v1'
VERSION='second-clock-budget-interface-v1'
LEGACY_DIGEST='sha256:4301bffc74f3cb3f334be5d21aa75dfe5742a16c9838942cdfc2f870859fb404'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def expected_bindings(root=ROOT):
 root=Path(root);package=root/PACKAGE;source=json.loads((package/'source_manifest.json').read_text())
 if any(sha(package/'patched'/f)!=h for f,h in source['patched_sha256'].items()):raise ValueError('Patched source manifest mismatch')
 original=json.loads((root/LIB/'run_manifest.json').read_text())['pending_runs']
 shared=sha(root/LIB/'shared_geometry.npz');patch=sha(package/'sfincs_second_clock_budget.patch')
 return [dict(r,schema_version=VERSION,geometry_sha256=shared,source_commit=source['base_commit'],patch_sha256=patch,execution_release=False,reference_release=False)for r in original]
def stage_identity(run_id,destination,solver_digest,root=ROOT):
 if not re.fullmatch(r'sha256:[0-9a-f]{64}',solver_digest)or solver_digest==LEGACY_DIGEST:raise ValueError('A distinct rebuilt-image digest is required; a digest alone does not qualify the image')
 matches=[r for r in expected_bindings(root)if r['run_id']==run_id]
 if len(matches)!=1:raise KeyError(run_id)
 binding=matches[0];folder=Path(destination)
 if folder.exists()and any(folder.iterdir()):raise FileExistsError('Refuse to replace an existing run/identity directory')
 folder.mkdir(parents=True,exist_ok=True)
 keys=['schema_version','run_id','parent_id','family_id','split','forcing_sha256','geometry_sha256','source_commit','patch_sha256','solver_digest']
 identity={k:(solver_digest if k=='solver_digest'else binding[k])for k in keys}
 (folder/'udt_run_identity.txt').write_text('\n'.join(k+'='+v for k,v in identity.items())+'\n')
 (folder/'udt_run_identity.json').write_text(json.dumps(identity|{'grid':binding['grid'],'execution_release':False,'reference_release':False,'status':'identity_staged_not_operationally_qualified'},indent=2)+'\n')
 return identity
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('run_id');p.add_argument('destination',type=Path);p.add_argument('--solver-digest',required=True);a=p.parse_args()
 stage_identity(a.run_id,a.destination,a.solver_digest)
