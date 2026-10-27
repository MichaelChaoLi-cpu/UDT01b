"""Bind saved local OCI index, AMD64 manifest, config and all archive blobs."""
from pathlib import Path
import sys,json,tarfile,hashlib
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_sfincs_cpu_build import OUT,REPORT,sha,dump

def main(root=ROOT):
 root=Path(root);out=root/OUT;report=root/REPORT
 if(out/'freeze.json').exists():raise FileExistsError('Frozen artifact')
 binding=json.loads((out/'solver_image_binding.json').read_text());small={};blobs={}
 with tarfile.open(out/'solver_image.tar')as archive:
  for member in archive.getmembers():
   if not member.isfile():continue
   if member.name.startswith('blobs/sha256/'):
    h=hashlib.sha256();data=bytearray()
    with archive.extractfile(member)as f:
     for block in iter(lambda:f.read(2**20),b''):
      h.update(block)
      if member.size<2**20:data.extend(block)
    assert h.hexdigest()==member.name.split('/')[-1],member.name
    digest='sha256:'+h.hexdigest();blobs[digest]=member.size
    if data:small[digest]=bytes(data)
   elif member.name=='index.json':small['archive_index']=archive.extractfile(member).read()
 index_digest=binding['oci_manifest_digest'];index=json.loads(small[index_digest])
 matches=[x for x in index['manifests']if x.get('platform',{}).get('os')=='linux'and x.get('platform',{}).get('architecture')=='amd64']
 assert len(matches)==1
 manifest_digest=matches[0]['digest'];manifest=json.loads(small[manifest_digest]);config_digest=manifest['config']['digest'];config=json.loads(small[config_digest])
 assert config['architecture']=='amd64'and config['os']=='linux'
 assert all(layer['digest']in blobs for layer in manifest['layers'])
 assert config['config']['Labels']['org.opencontainers.image.revision']==binding['source_commit']
 record={'archive_sha256':sha(out/'solver_image.tar'),'archive_size_bytes':(out/'solver_image.tar').stat().st_size,'oci_index_digest':index_digest,'platform_manifest_digest':manifest_digest,'config_digest':config_digest,'verified_blob_count':len(blobs),'all_blob_hashes_verified':True,'all_runtime_layers_present':True,'platform':'linux/amd64','no_registry_push':True}
 dump(out/'oci_archive_receipt.json',record)
 binding.update({'oci_index_digest':index_digest,'oci_platform_manifest_digest':manifest_digest,'oci_config_digest':config_digest,'offline_archive_sha256':record['archive_sha256']})
 dump(out/'solver_image_binding.json',binding)
 print(json.dumps(record,indent=2))
if __name__=='__main__':main()
