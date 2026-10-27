"""Aggregate actual diagnostic tables by direct-open water group; never rerun a model."""
from pathlib import Path
import json,csv,argparse,math

def read(p):
 with Path(p).open()as f:return list(csv.DictReader(f))
def write(p,rows):
 with Path(p).open('w',newline='')as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def summarize(folder,dest,groups):
 folder=Path(folder);dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
 cells=read(folder/'udt_negative_cell_summary.csv');events=read(folder/'udt_negative_crossings.csv');faces=read(folder/'udt_negative_crossing_faces.csv');links=read(folder/'udt_negative_cell_group_links.csv')
 bycell={int(r['native_cell_index']):r for r in cells};byevent={int(r['event_id']):r for r in events}
 if len(bycell)!=len(cells)or len(byevent)!=len(events):raise ValueError('Duplicate identities')
 valid_codes={int(k)for k in groups}
 per_event={i:[]for i in byevent}
 for row in faces:
  event=int(row['event_id']);code=int(row['boundary_group_code'])
  if event not in byevent or int(row['native_cell_index'])!=int(byevent[event]['native_cell_index']):raise ValueError('Unbound crossing face')
  if code not in valid_codes:raise ValueError('Unknown face water group')
  per_event[event].append(row)
 for event,items in per_event.items():
  if len(items)!=4 or {r['direction']for r in items}!={'west','east','south','north'}:raise ValueError('Missing or duplicate crossing directions')
  row=byevent[event];old=float(row['previous_volume_m3']);new=float(row['new_volume_m3']);kind=row['kind']
  if not ((kind=='down'and old>=-.001 and new<-.001)or(kind=='up'and old<-.001 and new>=-.001)):raise ValueError('Invalid threshold direction')
  before=float(row['before_time_s']);after=float(row['after_time_s']);dt=float(row['dt_s'])
  if not all(math.isfinite(x)for x in [old,new,before,after,dt])or dt<=0 or abs(after-before-dt)>1e-8:raise ValueError('Invalid crossing clock')
  total=sum(float(r['signed_inward_volume_m3'])for r in items)
  if not math.isfinite(total)or abs(total-float(row['four_face_sum_m3']))>1e-10*max(1,abs(total)):raise ValueError('Inconsistent face sum')

 members={};seen=set()
 for r in links:
  key=(int(r['native_cell_index']),int(r['water_group_code']))
  if key[1]not in valid_codes:raise ValueError('Unknown cell water group')
  if key in seen or key[0]not in bycell:raise ValueError('Invalid direct group links')
  seen.add(key);members.setdefault(key[1],set()).add(key[0])
 if set.union(*members.values())!=set(bycell):raise ValueError('Incomplete native group coverage')
 for ids in members.values():
  if any(cid not in bycell for cid in ids):raise ValueError('Unknown native cell')
 flagged={cid for cid,r in bycell.items()if r['first_below_endpoint_s']!=''}
 cellrows=[];facerows=[]
 codes=sorted({int(k)for k in groups}|set(members))
 for code in codes:
  name=groups.get(str(code),'Unspecified group');ids=members.get(code,set());negative=ids&flagged
  first=[float(bycell[c]['first_below_endpoint_s'])for c in negative]
  down=sum(int(bycell[c]['down_crossings'])for c in ids);up=sum(int(bycell[c]['up_crossings'])for c in ids)
  cellrows.append({'water_group_code':code,'water_group_name':name,'linked_interior_cells':len(ids),'ever_negative_unique_cells':len(negative),'initial_below_cells':sum(bycell[c]['initial_below_threshold']=='T'for c in ids),'earliest_below_endpoint_s':min(first)if first else None,'minimum_flagged_volume_m3':min(float(bycell[c]['minimum_volume_m3'])for c in negative)if negative else None,'down_crossings':down,'up_crossings':up,'negative_poststep_observations':sum(int(bycell[c]['negative_poststep_count'])for c in ids),'counting':'nonexclusive group membership; do not sum as whole-domain unique cells'})
  for kind in ['down','up']:
   selected=[r for r in faces if int(r['boundary_group_code'])==code and byevent[int(r['event_id'])]['kind']==kind]
   amounts=[float(r['signed_inward_volume_m3'])for r in selected]
   facerows.append({'boundary_group_code':code,'water_group_name':name if code else'Interior or absent face; not a receiving-water attribution','crossing_kind':kind,'face_records':len(selected),'gross_inward_at_crossing_steps_m3':sum(max(v,0)for v in amounts),'gross_outward_at_crossing_steps_m3':sum(max(-v,0)for v in amounts),'signed_inward_at_crossing_steps_m3':sum(amounts),'scope':'crossing-face observations only; repeated faces/events possible, not an event water budget'})
 write(dest/'water_group_cell_summary.csv',cellrows);write(dest/'water_group_crossing_face_summary.csv',facerows)
 result={'origin':str(folder),'unique_interior_cells':len(cells),'unique_ever_negative_cells':len(flagged),'down_crossing_records':sum(r['kind']=='down'for r in events),'up_crossing_records':sum(r['kind']=='up'for r in events),'face_records':len(faces),'group_cell_memberships_nonexclusive':True,'crossing_face_sums_are_event_budget':False,'local_time_meaning':'actual step-end observation; continuous crossing is only bracketed','continuous_duration_claimed':False}
 (dest/'aggregation.json').write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('folder',type=Path);a.add_argument('destination',type=Path);a.add_argument('--schema',type=Path,required=True);args=a.parse_args();print(json.dumps(summarize(args.folder,args.destination,json.loads(args.schema.read_text())['groups'])))
