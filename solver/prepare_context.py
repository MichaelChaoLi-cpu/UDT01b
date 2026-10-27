#!/usr/bin/env python3
"""Prepare pinned source-equivalent SFINCS contexts; never build or run a solver."""
from pathlib import Path,PurePosixPath
import argparse,hashlib,json,os,shutil,tarfile,urllib.request
HERE=Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--archive',required=True,type=Path,help='Local pinned upstream .tar.gz')
 parser.add_argument('--download',action='store_true',help='Explicitly download a missing archive from the recorded GitHub URL')
 parser.add_argument('--variant',choices=['historical','static_pumps','weighted_pools'],required=True)
 parser.add_argument('--output',required=True,type=Path,help='New build context directory; refuses an existing directory')
 a=parser.parse_args();m=json.loads((HERE/'manifest.json').read_text())
 if not a.archive.exists():
  if not a.download:parser.error('Archive missing; obtain the pinned archive or explicitly supply --download')
  a.archive.parent.mkdir(parents=True,exist_ok=True)
  partial=a.archive.with_name(a.archive.name+'.partial')
  with urllib.request.urlopen(m['upstream_archive_url'],timeout=120) as r,partial.open('xb') as f:shutil.copyfileobj(r,f)
  if sha(partial)!=m['upstream_archive_sha256']:raise RuntimeError('Upstream archive checksum mismatch; partial retained for inspection')
  partial.rename(a.archive)
 if sha(a.archive)!=m['upstream_archive_sha256']:raise RuntimeError('Pinned archive checksum mismatch')
 for rel,h in m['base_overlay_and_build_support_sha256'].items():
  if sha(HERE/rel)!=h:raise RuntimeError('Release source checksum mismatch: '+rel)
 for name,h in m['variants'][a.variant]['source_files_sha256'].items():
  if sha(HERE/'variants'/a.variant/name)!=h:raise RuntimeError('Variant source checksum mismatch: '+name)
 if a.output.exists():raise FileExistsError('Preserve existing context; choose a new output directory')
 a.output.mkdir(parents=True)
 with tarfile.open(a.archive) as tar:
  for member in tar.getmembers():
   if not member.isfile():continue
   rel=PurePosixPath(*PurePosixPath(member.name).parts[1:])
   if not rel.parts or rel.parts[0]!='source':continue
   if rel.is_absolute() or '..' in rel.parts:raise RuntimeError('Unsafe upstream archive path')
   p=a.output/str(rel);p.parent.mkdir(parents=True,exist_ok=True)
   with tar.extractfile(member) as src,p.open('xb') as dst:shutil.copyfileobj(src,dst)
   os.chmod(p,member.mode&0o777)
 for p in (HERE/'base_overlay').rglob('*'):
  if p.is_file():shutil.copy2(p,a.output/p.relative_to(HERE/'base_overlay'))
 for p in (HERE/'variants'/a.variant).glob('*.f90'):
  target=a.output/'source/src'/p.name if p.name.startswith(('sfincs_','udt_')) else a.output/p.name
  shutil.copy2(p,target)
 for p in (HERE/'build_support').iterdir():shutil.copy2(p,a.output/p.name)
 shutil.copy2(HERE/'Dockerfile',a.output/'Dockerfile')
 source_hashes={str(p.relative_to(a.output)):sha(p) for p in (a.output/'source/src').iterdir() if p.is_file()}
 binding={'upstream_commit':m['upstream_commit'],'upstream_archive_sha256':m['upstream_archive_sha256'],'variant':a.variant,'original_image_id':m['variants'][a.variant]['original_image_id'],'base_image':m['base_image'],'source_files_sha256':source_hashes,'scope':'Source reconstruction only; no build, simulation or inherited numerical qualification.'}
 (a.output/'SOURCE_BINDING.json').write_text(json.dumps(binding,indent=2)+'\n')
 print(json.dumps({'status':'SOURCE_CONTEXT_READY','variant':a.variant,'output':str(a.output),'core_source_files':len(source_hashes),'hydraulic_simulations':0},indent=2))
if __name__=='__main__':main()
