
# Resolve original raw names to fixed Moga members; no local raw copies.
import sys as _moga_sys
from pathlib import Path as _MogaPath
_moga_sys.path.insert(0, str(_MogaPath(__file__).resolve().parents[1]))
from moga_raw_inputs import raw_path as moga_raw, source_reference as moga_source, project_path as moga_project_path
from pathlib import Path
import zipfile,xml.etree.ElementTree as ET,json,collections
root=Path(__file__).resolve().parents[2];seen=set();themes=[];counts=collections.Counter();lod=collections.Counter();files=[]
class Scan:
 def __init__(self):self.inside=False;self.capture=None;self.text=[];self.depth=0;self.n=0
 def start(self,tag,attrs):
  local=tag.split('}')[-1]
  if local=='CityFurniture':self.inside=True;self.depth=1;self.n+=1
  elif self.inside:
   self.depth+=1
   if self.depth==2 and local in ['function','class']:self.capture=local;self.text=[]
   if self.depth==2 and local.startswith('lod'):lod[local]+=1
 def data(self,s):
  if self.capture:self.text.append(s)
 def end(self,tag):
  if not self.inside:return
  if self.capture and tag.split('}')[-1]==self.capture:
   if self.capture=='function':counts[''.join(self.text).strip()]+=1
   self.capture=None
  self.depth-=1
  if self.depth==0:self.inside=False
 def close(self):pass
for p in sorted((moga_raw(root, 'plateau/2025')).rglob('*citygml*.zip')):
 with zipfile.ZipFile(p) as z:
  members=[i for i in z.infolist() if i.filename.startswith('udx/') and i.filename.endswith('.gml')];themes.append({'ward':p.parent.name,'themes':dict(collections.Counter(i.filename.split('/')[1] for i in members))})
  for i in members:
   if not i.filename.startswith('udx/frn/'):continue
   key=(i.filename,i.file_size,i.CRC)
   if key in seen:continue
   seen.add(key);scan=Scan();parser=ET.XMLParser(target=scan)
   with z.open(i) as f:
    while b:=f.read(1024*1024):parser.feed(b)
   parser.close();files.append({'archive':moga_source(p, root),'member':i.filename,'crc32':f'{i.CRC:08x}','objects':scan.n})
   print(i.filename,scan.n,flush=True)
result={'archive_theme_counts':themes,'explicit_cons_theme_present':False,'unique_frn_files':files,'frn_function_counts_package_extent_not_ward_clipped':dict(counts),'frn_lod_properties_package_extent':dict(lod),'drainage_functions_actually_present':{k:v for k,v in counts.items() if k in ['7000','7100','7200','7300','7400','7500']},'riverbed_or_complete_channel_in_frn':'not established;furniture and bridge models are separate structures,not channel bathymetry'}
p=root/'data/exp/data-briefing/three_ward_sfincs_scenarios/river_3d_availability.json';old=json.loads(p.read_text());old['other_theme_audit']=result;p.write_text(json.dumps(old,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
