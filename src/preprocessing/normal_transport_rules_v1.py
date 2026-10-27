"""Explicit public-through access and native-node level rules for the dry graph.
These are research parsing assumptions, not a certification of current legal access.
"""
import heapq,collections
YES={'yes','designated','official','permissive'}
NO={'no','private','customers','delivery','agricultural','forestry','permit','destination','use_sidepath'}
DRIVE={'motorway','motorway_link','trunk','trunk_link','primary','primary_link','secondary','secondary_link','tertiary','tertiary_link','unclassified','residential','living_street','service','road'}
WALK=DRIVE|{'pedestrian','footway','path','steps','corridor','elevator','track','cycleway'}
def access(t,mode):
 h=t.get('highway','');keys=['foot','access'] if mode=='walk' else ['motorcar','motor_vehicle','vehicle','access']
 if h in ['construction','proposed','abandoned','raceway'] or t.get('area')=='yes':return 'excluded','non_linear_or_not_open'
 for k in keys:
  if k+':conditional' in t or k+':forward:conditional' in t or k+':backward:conditional' in t:return 'unknown','conditional_access'
 for k in keys:
  if k in t:
   v=t[k].lower()
   if v in NO:return 'excluded',k+'='+v
   if v in YES:return 'allowed',k+'='+v
   return 'unknown',k+'='+v
 if mode=='walk':
  if h not in WALK:return 'excluded','not_a_walk_highway'
  if h.startswith('motorway') or t.get('motorroad')=='yes':return 'excluded','motor_only_default'
  if h in ['cycleway','road']:return 'unknown','unspecified_walk_permission'
  return 'allowed','documented_highway_class_default'
 return ('allowed','documented_highway_class_default') if h in DRIVE else ('excluded','not_a_public_motorcar_highway')
def directions(t,mode):
 if mode=='walk':v=t.get('oneway:foot',t.get('foot:oneway','no'))
 else:v=t.get('oneway:motorcar',t.get('oneway:motor_vehicle',t.get('oneway','yes' if t.get('junction')=='roundabout' or t.get('highway') in ['motorway','motorway_link'] else 'no')))
 if mode=='walk' and t.get('conveying') in ['forward','backward']:v='yes' if t['conveying']=='forward' else '-1'
 conditional_keys=['oneway:foot:conditional','foot:oneway:conditional'] if mode=='walk' else ['oneway:motorcar:conditional','oneway:motor_vehicle:conditional','oneway:conditional']
 if any(k in t for k in conditional_keys):return False,False,'unknown_conditional_direction'
 if v in ['yes','1','true']:return True,False,'forward'
 if v in ['-1','reverse']:return False,True,'reverse'
 if v in ['no','0','false']:return True,True,'both'
 return False,False,'unknown_direction'
def grade(t):
 level=t.get('level','').strip();layer=t.get('layer','0').strip();structure='tunnel' if t.get('tunnel') not in [None,'','no','false','0'] else 'bridge' if t.get('bridge') not in [None,'','no','false','0'] else 'surface'
 def clean(v):
  try:return format(float(v),'g')
  except ValueError:return v
 return (clean(level) if level else '0' if structure=='surface' and t.get('indoor')!='yes' and t.get('highway')!='corridor' else '?',clean(layer or '0'),structure)
def grade_key(t):return '|'.join(grade(t))
def transitions_allowed(a,b,at_a_endpoint,at_b_endpoint,vertical=False):
 ga,gb=grade(a),grade(b)
 if ga==gb:return True,'same_grade'
 if ga[1:]==gb[1:] and '?' in [ga[0],gb[0]]:return True,'source_shared_node_unspecified_floor_compatible'
 if vertical:return True,'explicit_vertical_way'
 # Level tags are relative floors; do not merge different known floors by XY.
 if ga[0]!='?' and gb[0]!='?' and ga[0]!=gb[0]:return False,'different_explicit_floors'
 # A declared bridge/tunnel boundary on the same native node can connect to
 # an at-grade approach. Interior crossings do not get this exception.
 if ga[2] in ['bridge','tunnel'] and gb[2]=='surface' and at_a_endpoint:return True,'source_structure_portal'
 if gb[2] in ['bridge','tunnel'] and ga[2]=='surface' and at_b_endpoint:return True,'source_structure_portal'
 # Same shared node but unspecified floor is insufficient for an interior
 # surface/indoor bridge level mismatch. Keep uncertainty instead of snapping.
 return False,'shared_node_grade_conflict'
def node_access(t,mode):
 status,why='allowed','no_node_block'
 for k in (['foot','access'] if mode=='walk' else ['motorcar','motor_vehicle','vehicle','access']):
  if k+':conditional' in t:return 'unknown','node_conditional_access'
  if k in t:
   v=t[k].lower()
   if v in YES:return 'allowed','node_'+k+'='+v
   if v in NO:return 'excluded','node_'+k+'='+v
   return 'unknown','node_'+k+'='+v
 barrier=t.get('barrier','')
 if barrier in ['wall','fence','block','retaining_wall']:return 'excluded','node_physical_barrier'
 if mode=='drive' and barrier in ['bollard','cycle_barrier','stile','turnstile','bus_trap','sump_buster']:return 'excluded','vehicle_barrier'
 if barrier in ['gate','lift_gate','swing_gate','kissing_gate','turnstile'] and not t.get('access') and not t.get('foot' if mode=='walk' else 'motor_vehicle'):return 'unknown','unresolved_gate_access'
 return status,why

def restriction_allows(history,new_way,via_node,previous_from,next_to,restrictions):
 """Turn-memory check for unconditional motorcar via-node/via-way relations."""
 for r in restrictions:
  pre=r['prefix'];kind=r['kind'];target=r['to_way']
  if r['via_node'] is not None:
   if via_node!=r['via_node'] or not history or history[-1]!=pre[0]:continue
   if r['restriction']=='no_u_turn' and target==pre[0]:
    if new_way==target and previous_from==next_to:return False
    continue
   if kind=='no' and new_way==target:return False
   if kind=='only' and new_way!=target:return False
  elif kind=='no':
   if len(history)>=len(pre) and tuple(history[-len(pre):])==tuple(pre) and new_way==target:return False
  else:
   # For only_* via-way restrictions each transition after an active prefix
   # must follow its declared chain; remaining on the same way is permitted.
   full=pre+[target]
   for n in range(1,len(full)):
    if len(history)>=n and tuple(history[-n:])==tuple(full[:n]) and new_way not in [history[-1],full[n]]:
     transition_nodes=r.get('transition_nodes')
     if transition_nodes is None or via_node in transition_nodes[n-1]:return False
 return True

def shortest_paths(adj,start,targets,restrictions=(),max_states=1500000):
 """Distance-only Dijkstra with turn history; no people, flood or travel times."""
 targets=set(targets);found={};heap=[(0.,start,(),None)];best={(start,(),None):0.};byfrom=collections.defaultdict(list);maxhist=1;prefixes=set()
 for r in restrictions:
  index_ways=[r['prefix'][0]] if r['via_node'] is not None else [r['prefix'][-1]] if r['kind']=='no' else r['prefix']
  for wid in set(index_ways):byfrom[wid].append(r)
  maxhist=max(maxhist,len(r['prefix']))
  for i in range(1,len(r['prefix'])+1):prefixes.add(tuple(r['prefix'][:i]))
 while heap and len(found)<len(targets):
  d,u,hist,prev=heapq.heappop(heap);key=(u,hist,prev)
  if d!=best.get(key):continue
  if u in targets and u not in found:found[u]=d
  if len(best)>max_states:break
  for v,length,way in adj.get(u,[]):
   candidates=byfrom.get(hist[-1],[]) if hist else []
   native=int(u.split(':')[1]) if u.startswith('osm:') else None
   if way!=0 and not restriction_allows(hist,way,native,prev,v,candidates):continue
   nh=hist if restrictions else ()
   if restrictions and way!=0 and (not hist or hist[-1]!=way):
    candidate=(hist+(way,))[-maxhist:];nh=(way,)
    for size in range(len(candidate),0,-1):
     if candidate[-size:] in prefixes:nh=candidate[-size:];break
   # For no_u_turn retain incoming vertex; pedestrian search needs no memory.
   np=(prev if way==0 else u) if restrictions else None;nk=(v,nh,np);nd=d+length
   if nd<best.get(nk,float('inf')):best[nk]=nd;heapq.heappush(heap,(nd,v,nh,np))
 return found,{'states':len(best),'state_limit_hit':len(best)>max_states}

def selfcheck():
 cases=[]
 def check(n,x):cases.append({'check':n,'passed':bool(x)})
 check('mode_specific_access_overrides_general',access({'highway':'footway','access':'private','foot':'yes'},'walk')[0]=='allowed')
 check('private_through_access_excluded',access({'highway':'residential','access':'private'},'drive')[0]=='excluded')
 check('destination_not_public_through_route',access({'highway':'service','access':'destination'},'drive')[0]=='excluded')
 check('motorway_default_no_walk',access({'highway':'motorway'},'walk')[0]=='excluded')
 check('footway_default_no_motorcar',access({'highway':'footway'},'drive')[0]=='excluded')
 check('car_oneway_does_not_limit_walking',directions({'oneway':'yes'},'walk')[:2]==(True,True))
 check('reverse_motorcar_oneway',directions({'oneway':'-1'},'drive')[:2]==(False,True))
 check('conditional_not_silently_open',access({'highway':'residential','access:conditional':'no @ (Mo-Fr)'},'walk')[0]=='unknown')
 check('interior_bridge_does_not_merge_ground',not transitions_allowed({'bridge':'yes','layer':'1'},{},False,False)[0])
 check('source_bridge_endpoint_connects_approach',transitions_allowed({'bridge':'yes','layer':'1'},{},True,False)[0])
 check('different_floors_do_not_snap',not transitions_allowed({'level':'-1'},{'level':'0'},True,True)[0])
 check('explicit_stairs_connect_levels',transitions_allowed({'level':'-1'},{'level':'0'},False,False,True)[0])
 r={'prefix':[10],'to_way':20,'via_node':5,'kind':'no','restriction':'no_left_turn'}
 check('no_turn_respected',not restriction_allows((10,),20,5,'x','y',[r]))
 check('allowed_other_turn_retained',restriction_allows((10,),30,5,'x','y',[r]))
 r={'prefix':[10,11],'to_way':20,'via_node':None,'kind':'no','restriction':'no_straight_on'}
 check('via_way_history_respected',not restriction_allows((10,11),20,8,'x','y',[r]))
 check('unrelated_history_retained',restriction_allows((12,11),20,8,'x','y',[r]))
 # Turn-aware paths must preserve their incoming way across a grade portal.
 adj={'osm:1:g':[('osm:2:g',1.,10)],'osm:2:g':[('osm:2:b',0.,0),('osm:3:g',2.,30)],'osm:2:b':[('osm:4:g',1.,20)],'osm:3:g':[('osm:4:g',2.,40)]}
 rr={'prefix':[10],'to_way':20,'via_node':2,'kind':'no','restriction':'no_left_turn'}
 out,meta=shortest_paths(adj,'osm:1:g',['osm:4:g'],[rr]);check('routing_preserves_turn_rule_across_grade_portal',out.get('osm:4:g')==5.)
 check('equivalent_floor_tokens_normalized',grade_key({'level':'0.0'})==grade_key({'level':'0'}))
 check('specific_conditional_oneway_held',directions({'oneway:motorcar:conditional':'yes @ (Mo-Fr)'},'drive')[2]=='unknown_conditional_direction')
 rr={'prefix':[10,11],'to_way':20,'via_node':None,'kind':'only','restriction':'only_straight_on','transition_nodes':[[5],[8]]}
 check('via_way_only_applies_at_source_transition',restriction_allows((10,),30,4,'x','y',[rr]) and not restriction_allows((10,),30,5,'x','y',[rr]))
 aa={'osm:1:g':[('osm:5:g',1.,10)],'osm:5:g':[('osm:8:g',1.,11)],'osm:8:g':[('osm:9:g',1.,20),('osm:10:g',1.,30)]}
 oo,_=shortest_paths(aa,'osm:1:g',['osm:9:g','osm:10:g'],[rr]);check('pruned_turn_history_retains_full_via_way_constraint','osm:9:g' in oo and 'osm:10:g' not in oo)
 assert all(r['passed'] for r in cases)
 return cases
