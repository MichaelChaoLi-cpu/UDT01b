"""Fixed diagnostic gates; no optimizer, fitting loop or validation access."""
import math
GATE_SCOPES=('three_wards','Chiyoda','Chuo','Minato')
GATES={'iou_0.1':('min',.85),'precision_0.1':('min',.90),'recall_0.1':('min',.90),'wet_mae_m':('max',.05),'deep_mae_m':('max',.20)}
def evaluate_fit_gates(events,case_ids,expected_masks):
 rows=[r for r in events if r['scope'] in GATE_SCOPES]
 required={(c,s) for c in case_ids for s in GATE_SCOPES}
 if len(rows)!=len(required) or {(r['case_id'],r['scope']) for r in rows}!=required:raise ValueError('Incomplete or duplicate parent/ward gate support')
 details=[]
 for r in rows:
  if r['split']!='train' or r['support']!='own_exact' or r['support_sha256']!=expected_masks[r['case_id']][r['scope']]:raise ValueError('Changed split or original support')
  checks={};not_applicable=[]
  for key,(direction,limit) in GATES.items():
   if key=='deep_mae_m' and r.get('deep_area_km2',0)==0:not_applicable.append(key);continue
   v=r.get(key);checks[key]=v is not None and math.isfinite(v) and (v>=limit if direction=='min' else v<=limit)
  details.append({'case_id':r['case_id'],'scope':r['scope'],'pass':all(checks.values()),'checks':checks,'not_applicable':not_applicable})
 return {'all_parent_ward_gates_pass':all(r['pass'] for r in details),'rows':details,'gate_units':'depth errors in m; dimensionless extent rates','not_formal_admission':True}
def diagnostic_key(events,step):
 rows=[r for r in events if r['scope'] in GATE_SCOPES]
 if not rows:raise ValueError('No gate rows')
 return (-round(min(r['iou_0.1'] for r in rows),6),round(max(r['wet_mae_m'] for r in rows),6),round(max(r['mae_m'] for r in rows),6),int(step))
def update_success_streak(old,gate,steps):return old+1 if steps>0 and gate['all_parent_ward_gates_pass'] else 0
