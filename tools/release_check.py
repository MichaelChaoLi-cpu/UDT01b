#!/usr/bin/env python3
"""Check published source integrity without importing or running research code."""
from pathlib import Path
import argparse,ast,hashlib,json,re,sys,tomllib
ROOT=Path(__file__).resolve().parents[1]
PRIVATE_HELPERS={'archive_large_diagnostics.py','verify_diagnostic_archive_cleanup.py','rollback_nas_diagnostic_archive.py'}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--json',action='store_true');args=parser.parse_args()
 errors=[];python_count=0
 for base in ['src','tools','solver']:
  for p in (ROOT/base).rglob('*.py'):
   if '__pycache__' in p.parts or 'build-context' in p.parts or p.name in PRIVATE_HELPERS:continue
   try:ast.parse(p.read_text(),filename=str(p));python_count+=1
   except (SyntaxError,UnicodeError) as e:errors.append(str(e))
 manifest_path=ROOT/'docs/code_release_manifest.json'
 m=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
 if not m:errors.append('Missing code release manifest')
 for rel,h in m.get('files_sha256',{}).items():
  p=ROOT/rel
  if not p.is_file() or sha(p)!=h:errors.append('Release hash mismatch: '+rel)
 project=tomllib.loads((ROOT/'pyproject.toml').read_text());lock=tomllib.loads((ROOT/'uv.lock').read_text())
 package=next((p for p in lock['package'] if p['name'].lower()==project['project']['name'].lower()),None)
 declared={re.split(r'[<>=!~;\[]',d)[0].lower().replace('_','-') for d in project['project']['dependencies']}
 locked={p['name'].lower().replace('_','-') for p in package.get('dependencies',[])} if package else set()
 if declared!=locked:errors.append('Project and lock direct dependencies disagree')
 for name in ['README.md','docs/DATA_SOURCES.md','docs/REPRODUCIBILITY.md','solver/README.md','THIRD_PARTY_NOTICES.md']:
  text=(ROOT/name).read_text()
  for target in re.findall(r'\]\(([^)]+)\)',text):
   if not target.startswith(('http://','https://','#')) and not (ROOT/name).parent.joinpath(target.split('#')[0]).exists():errors.append('Broken local link: '+name+' -> '+target)
 sm=json.loads((ROOT/'solver/manifest.json').read_text())
 for rel,h in sm['base_overlay_and_build_support_sha256'].items():
  if sha(ROOT/'solver'/rel)!=h:errors.append('Solver common source mismatch: '+rel)
 for variant,row in sm['variants'].items():
  for name,h in row['source_files_sha256'].items():
   if sha(ROOT/'solver/variants'/variant/name)!=h:errors.append('Solver variant mismatch: '+variant+'/'+name)
 result={'passed':not errors,'python_sources_parsed_without_import':python_count,'release_files_hash_checked':len(m.get('files_sha256',{})),'solver_variants_checked':len(sm['variants']),'hydraulic_runs_started':0,'training_runs_started':0,'network_or_nas_access':False,'errors':errors}
 print(json.dumps(result,indent=2) if args.json else ('PASS' if result['passed'] else 'FAIL')+' '+str(result))
 return 0 if result['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
