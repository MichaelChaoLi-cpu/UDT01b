"""Extract bounded OSM primitives while preserving native node/way/relation IDs.
A small strict wire decoder implements the primary OSM-binary schema. It ignores
personal metadata, supports raw/zlib blobs, and fails on truncated/unknown encoding.
No inferred intersections or nearest-station links are constructed here.
"""
from pathlib import Path
import struct,zlib,json,collections,hashlib,math,time
import numpy as np
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'data/processed/three_ward_normal_transport_network_v1';S=OUT/'sources'

def varint(b,p=0):
 n=0;s=0
 while True:
  v=b[p];p+=1;n|=(v&127)<<s
  if v<128:return n,p
  s+=7
  if s>70:raise ValueError('Malformed varint')
def packed(b):
 p=0;out=[]
 while p<len(b):v,p=varint(b,p);out.append(v)
 return out
def zig(v):return (v>>1)^-(v&1)
def fields(b):
 p=0
 while p<len(b):
  key,p=varint(b,p);tag=key>>3;wire=key&7
  if wire==0:v,p=varint(b,p)
  elif wire==2:
   n,p=varint(b,p);v=b[p:p+n];p+=n
   if len(v)!=n:raise ValueError('Truncated wire payload')
  elif wire==1:v=b[p:p+8];p+=8
  elif wire==5:v=b[p:p+4];p+=4
  else:raise ValueError(('Unsupported protobuf wire',wire))
  yield tag,v
 assert p==len(b)
def msg(b):return dict(fields(b))
def unpack_delta(b):
 x=0;out=[]
 for v in packed(b):x+=zig(v);out.append(x)
 return out
def blobs(path):
 with path.open('rb') as f:
  while True:
   n=f.read(4)
   if not n:return
   assert len(n)==4;h=f.read(struct.unpack('>I',n)[0]);d=msg(h);blob=f.read(d[3]);assert len(blob)==d[3];bb=msg(blob)
   if 1 in bb:raw=bb[1]
   elif 3 in bb:raw=zlib.decompress(bb[3]);assert len(raw)==bb[2]
   else:raise ValueError('Only raw and zlib OSM blobs are supported')
   yield d[1].decode(),raw

def main():
 # Technical acquisition extents, not final isolation boundaries.
 xy=Transformer.from_crs(6668,6677,always_xy=True).transform;ll=Transformer.from_crs(6677,4326,always_xy=True).transform
 fs=json.loads((ROOT/'data/processed/three_ward_transport_framework_v1/inventory_domains_6677.geojson').read_text())['features'];core=shape(fs[0]['geometry']);bounds=transform(ll,core.buffer(21000)).bounds
 xmin,ymin,xmax,ymax=bounds;roadbounds=transform(ll,core.buffer(11000)).bounds;rx0,ry0,rx1,ry1=roadbounds
 nodes={};node_tags={};ways={};relations={};c=collections.Counter();header={};start=time.time()
 for block,(kind,raw) in enumerate(blobs(S/'kanto-261015.osm.pbf')):
  if kind=='OSMHeader':
   h=msg(raw);header={'required_features':[v.decode() for k,v in fields(raw) if k==4],'writing_program':h.get(16,b'').decode(),'replication_timestamp':h.get(32),'replication_sequence':h.get(33)};continue
  assert kind=='OSMData';parts=list(fields(raw));st=[v.decode('utf8') for k,b in parts if k==1 for kk,v in fields(b) if kk==1];attrs=dict((k,v) for k,v in parts if k!=2);gran=attrs.get(17,100);lat0=attrs.get(19,0);lon0=attrs.get(20,0)
  def tags(d):return {st[k]:st[v] for k,v in zip(packed(d.get(2,b'')),packed(d.get(3,b'')))}
  for _,pg in (x for x in parts if x[0]==2):
   for ptag,b in fields(pg):
    d=msg(b)
    if ptag==2:
     ids=unpack_delta(d[1]);lat=unpack_delta(d[8]);lon=unpack_delta(d[9]);kv=packed(d.get(10,b''));kp=0;c['national_nodes']+=len(ids)
     for id,la,lo in zip(ids,lat,lon):
      y=(lat0+gran*la)*1e-9;x=(lon0+gran*lo)*1e-9;keep=xmin<=x<=xmax and ymin<=y<=ymax
      if keep:nodes[id]=(x,y)
      if kv:
       begin=kp
       while kv[kp]!=0:kp+=2
       if keep and kp>begin:node_tags[id]={st[kv[j]]:st[kv[j+1]] for j in range(begin,kp,2)}
       kp+=1
     assert not kv or kp==len(kv)
    elif ptag==1:
     id=zig(d[1]);y=(lat0+gran*zig(d[8]))*1e-9;x=(lon0+gran*zig(d[9]))*1e-9;c['national_nodes']+=1
     if xmin<=x<=xmax and ymin<=y<=ymax:nodes[id]=(x,y);node_tags[id]=tags(d)
    elif ptag==3:
     c['national_ways']+=1;t=tags(d)
     if not any(k in t for k in ['highway','railway','public_transport']):continue
     refs=unpack_delta(d.get(8,b''));present=[nodes[id] for id in refs if id in nodes]
     if not present:continue
     isroad='highway' in t
     if isroad and not any(rx0<=x<=rx1 and ry0<=y<=ry1 for x,y in present):continue
     ways[d[1]]={'id':d[1],'refs':refs,'tags':t}
    elif ptag==4:
     c['national_relations']+=1;t=tags(d)
     wanted=t.get('type')=='restriction' or t.get('type') in ['public_transport','route','route_master'] and (t.get('public_transport') in ['stop_area','stop_area_group'] or t.get('route') in ['train','subway','monorail','light_rail','tram'] or t.get('route_master') in ['train','subway','monorail','light_rail','tram'])
     if not wanted:continue
     mids=unpack_delta(d.get(9,b''));roles=packed(d.get(8,b''));types=packed(d.get(10,b''));members=[{'type':['node','way','relation'][typ],'id':id,'role':st[role]} for id,role,typ in zip(mids,roles,types)]
     if any(id in nodes if ty==0 else id in ways if ty==1 else id in relations for id,ty in zip(mids,types)):relations[d[1]]={'id':d[1],'tags':t,'members':members}
  if block%1000==0:print('PBF',block,dict(c),'kept',len(nodes),len(ways),len(relations),'seconds',round(time.time()-start,1),flush=True)
 # Recover coordinates referenced by selected ways or relation members outside
 # the broad node window; keep the source member order and do not skip stations.
 need={id for w in ways.values() for id in w['refs'] if id not in nodes}|{m['id'] for r in relations.values() for m in r['members'] if m['type']=='node' and m['id'] not in nodes}
 if need:
  print('RECOVER REFERENCED NODES',len(need),flush=True)
  for kind,raw in blobs(S/'kanto-261015.osm.pbf'):
   if kind!='OSMData':continue
   parts=list(fields(raw));attrs=dict((k,v) for k,v in parts if k!=2);gran=attrs.get(17,100);lat0=attrs.get(19,0);lon0=attrs.get(20,0);st=None
   for _,pg in (x for x in parts if x[0]==2):
    for tag,b in fields(pg):
     if tag not in [1,2]:continue
     d=msg(b)
     if tag==2:
      ids=unpack_delta(d[1]);hit=[i for i,id in enumerate(ids) if id in need]
      if not hit:continue
      lat=unpack_delta(d[8]);lon=unpack_delta(d[9]);kv=packed(d.get(10,b''));kp=0;hitset=set(hit)
      if st is None:st=[v.decode('utf8') for k,bb in parts if k==1 for kk,v in fields(bb) if kk==1]
      for i,id in enumerate(ids):
       if i in hitset:nodes[id]=((lon0+gran*lon[i])*1e-9,(lat0+gran*lat[i])*1e-9);need.discard(id)
       if kv:
        begin=kp
        while kv[kp]!=0:kp+=2
        if i in hitset and kp>begin:node_tags[id]={st[kv[j]]:st[kv[j+1]] for j in range(begin,kp,2)}
        kp+=1
     else:
      id=zig(d[1])
      if id in need:nodes[id]=((lon0+gran*zig(d[9]))*1e-9,(lat0+gran*zig(d[8]))*1e-9);need.discard(id)
   if not need:break
 # Save only nodes required by relevant primitives, plus tagged stations/portals.
 used={id for w in ways.values() for id in w['refs']}|{m['id'] for r in relations.values() for m in r['members'] if m['type']=='node'}|{id for id,t in node_tags.items() if any(k in t for k in ['railway','public_transport','entrance','barrier'])}
 outnodes=[{'id':id,'lon':nodes[id][0],'lat':nodes[id][1],'tags':node_tags.get(id,{})} for id in sorted(used) if id in nodes]
 for name,rr in [('nodes',outnodes),('ways',list(ways.values())),('relations',list(relations.values()))]:
  with(OUT/('osm_'+name+'.jsonl')).open('w') as f:
   for row in rr:f.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
 receipt={'header':header,'source_bytes':(S/'kanto-261015.osm.pbf').stat().st_size,'node_inventory_bounds_lonlat':list(bounds),'road_selection_bounds_lonlat':list(roadbounds),'counts':dict(c),'retained_nodes':len(outnodes),'retained_ways':len(ways),'retained_relations':len(relations),'unresolved_member_node_ids':len(need),'all_selected_way_nodes_present':all(id in nodes for w in ways.values() for id in w['refs']),'seconds':time.time()-start,'personal_metadata_preserved':False,'license':'ODbL 1.0; OpenStreetMap contributors; distribution extract by Geofabrik'}
 assert receipt['all_selected_way_nodes_present'];(OUT/'osm_extraction_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2),flush=True)
if __name__=='__main__':main()
