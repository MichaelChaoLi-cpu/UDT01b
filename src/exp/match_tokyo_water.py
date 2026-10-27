"""Match source water footprints and river lines; keep hydraulic uncertainty explicit.

Run .venv/bin/python src/exp/match_tokyo_water.py from the project root.
No source geometries, names or elevations are overwritten.
"""
import os
for k in ('PROJ_LIB','PROJ_DATA','GDAL_DATA'): os.environ.pop(k, None)
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-mpl')
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations, product
from hashlib import sha256
from zipfile import ZipFile
from xml.etree import ElementTree as ET
import json, unicodedata, html
import numpy as np
from pyproj import Transformer
from shapely import transform as st
from shapely.geometry import shape, mapping, Point, LineString
from shapely.ops import unary_union, nearest_points
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/exp/data-briefing/central_three_water_network'
PROC=ROOT/'data/processed/central_three_water_network'
T=Transformer.from_crs(6668,6677,always_xy=True)
OLD=Transformer.from_crs(4612,6677,always_xy=True)
LL=Transformer.from_crs(6677,6668,always_xy=True)
PARAM={'near_search_m':25,'direct_min_length_m':5,'fragment_length_m':1,
       'significant_cross_name_fraction':0.10,'gap_review_min_m':25,'low_coverage_fraction':0.60}
G='{http://www.opengis.net/gml/3.2}';K='{http://nlftp.mlit.go.jp/ksj/schemas/ksj-app}';X='{http://www.w3.org/1999/xlink}'
WARD={'13101-chiyoda-ku':'千代田区','13102-chuo-ku':'中央区','13103-minato-ku':'港区'}
KNOWN_MOATS=set('桜田濠 蛤濠 桔梗濠 凱旋濠 半蔵濠 千鳥ヶ淵 牛ヶ淵 清水濠 大手濠 和田倉濠 馬場先濠 日比谷濠'.split())
MOE_URL='https://www.env.go.jp/garden/content/000230549.pdf'

def project(g, t=T):return st(g,t.transform,interleaved=False)
def read_fc(p):
    return [(project(shape(f['geometry'])),f['properties']) for f in json.loads(p.read_text())['features']]
def norm(name):
    if name in ('Null','名称不明','(unnamed water body)','',None):return ''
    return unicodedata.normalize('NFKC',name).replace(' ','').replace('　','')
def dump(p,j):p.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def fc(p,records):
    dump(p,{'type':'FeatureCollection','features':[{'type':'Feature','geometry':mapping(project(g,LL)),'properties':r} for g,r in records]})
def geom_parts(g):
    return list(g.geoms) if hasattr(g,'geoms') else [g]
def anchor(g):return g.representative_point()
def link_geometry(a,b):
    p,q=nearest_points(a,b)
    return p if p.distance(q)<1e-6 else LineString([p,q])
def source_topology():
    with ZipFile(ROOT/'data/raw/ksj/W05-08_13_GML.zip') as z:r=ET.fromstring(z.read('W05-08_13-g.xml'))
    byid={e.get(G+'id'):e for e in r if e.get(G+'id')}
    nodes={}
    for e in r.findall(K+'RiverNode'):
        p=byid[e.find(K+'position').get(X+'href').lstrip('#')]
        lat,lon=map(float,p.find(G+'pos').text.split())
        nodes[e.get(G+'id')]=Point(*OLD.transform(lon,lat))
    curves={}
    for e in r.findall(K+'Stream'):
        c=byid[e.find(K+'location').get(X+'href').lstrip('#')]
        v=np.fromstring(c.find('.//'+G+'posList').text,sep=' ').reshape(-1,2)
        x,y=OLD.transform(v[:,1],v[:,0]);curves[e.get(G+'id')]=LineString(np.c_[x,y])
    return nodes,curves

def main():
    for d in (OUT,PROC):d.mkdir(parents=True,exist_ok=True)
    water_path=ROOT/'data/processed/plateau_water/central_three_wards_wtr.geojson'
    line_path=ROOT/'data/processed/ksj_tokyo/W05_central_three_wards.geojson'
    boundary_path=ROOT/'data/processed/plateau_water/central_three_ward_boundaries.geojson'
    waters=sorted(read_fc(water_path),key=lambda q:q[1]['water_id'])
    lines=sorted(read_fc(line_path),key=lambda q:int(q[1]['source_id'][2:]))
    wards=read_fc(boundary_path);domain=unary_union([g for g,p in wards])
    all_water=unary_union([g for g,p in waters]);nodes,full_curves=source_topology()
    wmap={p['water_id']:(g,p) for g,p in waters}
    lmap={p['source_id']:(g,p) for g,p in lines}
    short={p['water_id']:f'W{i:03}' for i,(g,p) in enumerate(waters,1)}
    matches=[]; wm=defaultdict(list); lm=defaultdict(list)
    for lg,lp in lines:
        for wg,wp in waters:
            distance=lg.distance(wg); pn,ln=norm(wp['name']),norm(lp['riverName'])
            if distance>PARAM['near_search_m'] and not (pn and pn==ln):continue
            length=lg.intersection(wg).length
            fraction=length/lg.length if lg.length else 0
            near_length=lg.intersection(wg.buffer(PARAM['near_search_m'])).length
            fragment=lg.length<PARAM['fragment_length_m']
            if fragment:status='boundary_fragment';association=False
            elif length>=PARAM['direct_min_length_m']:
                if pn and ln and pn==ln:status='name_and_overlap';association=True
                elif not pn or not ln:status='overlap_name_unresolved';association=True
                else:status='cross_named_footprints';association=False
            elif pn and pn==ln:
                status='same_name_offset' if distance<=PARAM['near_search_m'] else 'same_name_distant'
                association=False
            else:status='near_only';association=False
            rec={'match_id':f'M{len(matches)+1:03}','water_id':wp['water_id'],'short_id':short[wp['water_id']],
                 'w05_id':lp['source_id'],'plateau_name':wp['name'],'w05_name':lp['riverName'],
                 'status':status,'provisional_geometric_association':association,
                 'distance_m':round(distance,3),'inside_water_length_m':round(length,3),
                 'fraction_of_clipped_line_inside_water':round(fraction,6),
                 'length_within_25m_search_m':round(near_length,3),'confirmed_hydraulic_connection':False}
            matches.append(rec);wm[wp['water_id']].append(rec);lm[lp['source_id']].append(rec)

    # Undirected footprint relationships. Metadata endpoints are retained separately;
    # no river-code equality or near-distance is promoted to a hydraulic edge.
    edges={}
    def add_edge(a,b,evidence):
        if a==b:return
        key=tuple(sorted((a,b)))
        if key not in edges:
            ga,pa=wmap[key[0]];gb,pb=wmap[key[1]]
            edges[key]={'water_a':key[0],'water_b':key[1],'short_a':short[key[0]],'short_b':short[key[1]],
                'name_a':pa['name'],'name_b':pb['name'],'footprint_distance_m':round(ga.distance(gb),3),
                'evidence':[],'hydraulic_status':'unverified','flow_direction':'unassigned',
                'geometry_role':'nearest footprint gap/contact locator, not a surveyed conduit'}
        if evidence not in edges[key]['evidence']:edges[key]['evidence'].append(evidence)
    for (ga,pa),(gb,pb) in combinations(waters,2):
        d=ga.distance(gb)
        if d<=PARAM['near_search_m']:
            add_edge(pa['water_id'],pb['water_id'],{'kind':'footprints_touch_or_overlap' if d<1e-6 else 'footprints_near_25m'})
    for lg,lp in lines:
        hit=[m for m in lm[lp['source_id']] if m['inside_water_length_m']>=5 and m['fraction_of_clipped_line_inside_water']>=.02 and m['status']!='boundary_fragment']
        # Consecutive intersections along the original curve, not all-to-all cliques.
        hit.sort(key=lambda m:full_curves[lp['source_id']].project(anchor(lg.intersection(wmap[m['water_id']][0]))))
        for a,b in zip(hit,hit[1:]):add_edge(a['water_id'],b['water_id'],{'kind':'consecutive_footprints_on_same_W05_line','w05_id':lp['source_id']})
    memberships=defaultdict(list)
    for lg,lp in lines:
        for end in ('startStreamNode','endStreamNode'):
            memberships[lp[end].lstrip('#')].append(lp['source_id'])
    node_relations=[]
    for node,line_ids in sorted(memberships.items()):
        line_ids=sorted(set(line_ids));pos=nodes.get(node)
        if len(line_ids)<2:continue
        inside=pos is not None and domain.buffer(1).covers(pos)
        rec={'node_id':node,'w05_ids':line_ids,'node_in_three_ward_domain':inside,
             'hydraulic_status':'source_topology_only','confirmed_hydraulic_connection':False,
             'endpoint_fields_used':['startStreamNode','endStreamNode']}
        node_relations.append((pos,rec))
        if inside:
            for la,lb in combinations(line_ids,2):
                aa=[m['water_id'] for m in lm[la] if m['provisional_geometric_association'] and wmap[m['water_id']][0].distance(pos)<=PARAM['near_search_m']]
                bb=[m['water_id'] for m in lm[lb] if m['provisional_geometric_association'] and wmap[m['water_id']][0].distance(pos)<=PARAM['near_search_m']]
                for a,b in product(aa,bb):add_edge(a,b,{'kind':'W05_shared_stream_endpoint','node_id':node,'w05_ids':[la,lb]})

    # Located checks: anchors are clearly distinguished from measured inlets/outfalls.
    checks=[]
    def check(geo,priority,kind,subject,detail,**kwargs):
        checks.append((geo,{'check_id':f'V{len(checks)+1:03}','priority':priority,'kind':kind,
            'subject':subject,'detail':detail,'anchor_role':'review locator; not a measured inlet or hydraulic control',
            'confirmed_hydraulic_connection':False,**kwargs}))
    line_audit=[]
    for lg,lp in lines:
        sid=lp['source_id'];provisional=[m for m in lm[sid] if m['provisional_geometric_association']]
        gaps=lg.difference(all_water);gap_parts=[g for g in geom_parts(gaps) if g.length>=PARAM['gap_review_min_m']]
        coverage=lg.intersection(all_water).length/lg.length if lg.length else 0
        cross=[m for m in lm[sid] if m['status']=='cross_named_footprints' and m['fraction_of_clipped_line_inside_water']>=.10]
        same_geoms=[wmap[m['water_id']][0] for m in lm[sid] if m['status']=='name_and_overlap']
        same_coverage=lg.intersection(unary_union(same_geoms)).length/lg.length if same_geoms else None
        row={**lp,'length_in_three_wards_m':round(lg.length,3),'original_full_length_m':round(full_curves[sid].length,3),
             'fraction_inside_any_plateau_water':round(coverage,6),'geometric_associated_water_ids':[m['water_id'] for m in provisional],
             'fraction_inside_same_named_plateau_water':round(same_coverage,6) if same_coverage is not None else None,
             'significant_cross_named_match_ids':[m['match_id'] for m in cross],
             'original_endpoint_locations_inside_domain':{key:domain.buffer(1).covers(nodes[lp[key].lstrip('#')]) for key in ('startStreamNode','endStreamNode')},
             'fragment':lg.length<PARAM['fragment_length_m'],'uncovered_gaps_ge_25m':len(gap_parts),
             'height_use':'W05 node elevation is 50m-DEM-derived; not riverbed, crest or event stage'}
        line_audit.append((lg,row))
        if row['fragment']:
            check(anchor(lg),3,'ward_clip_fragment',sid+' '+lp['riverName'],'区界裁剪后不足1m；保留来源，但不单独认作一条水体或以其名称改写邻接水域。',w05_ids=[sid])
        elif cross:
            g=lg.intersection(unary_union([wmap[m['water_id']][0] for m in cross]))
            check(anchor(g),2,'cross_named_footprints',sid+' '+lp['riverName'],
                  '同一W05线跨入不同名称水域。可能是汇合口、命名分段或旧位置差异；核对后再划分连接。',
                  w05_ids=[sid],water_ids=sorted(set(m['water_id'] for m in cross+provisional)))
        named_gap=same_coverage is not None and same_coverage<PARAM['low_coverage_fraction'] and not cross
        if not row['fragment'] and (coverage<PARAM['low_coverage_fraction'] or not provisional or named_gap):
            g=max(gap_parts,key=lambda g:g.length) if gap_parts else lg
            check(anchor(g),2,'line_water_coverage_gap',sid+' '+lp['riverName'],
                  f'线在任何PLATEAU水域内的比例为{coverage:.1%}'+(f'，在同名水域内为{same_coverage:.1%}' if same_coverage is not None else '')+'。检查桥下遗漏、暗渠、边界位置或时期差异；不能视为断流。',w05_ids=[sid])
    inventory=[]
    name_counts=Counter(norm(p['name']) for g,p in waters if norm(p['name']))
    for wg,wp in waters:
        wid=wp['water_id'];mm=wm[wid];assoc=[m for m in mm if m['provisional_geometric_association']]
        reasons=[];name=wp['name'];nn=norm(name)
        if not nn:reasons.append('原包无有效名称；未由邻近河川线自动命名')
        if not assoc:reasons.append('未找到直接几何对应的W05线；可能是覆盖范围或分类差异')
        if name_counts[nn]>1 and nn:reasons.append('同名多个源对象，保持分开；是否同一水体待核查')
        cross=[m for m in mm if m['status']=='cross_named_footprints' and m['fraction_of_clipped_line_inside_water']>=.10]
        if cross:reasons.append('存在跨不同名称水域的W05线；检查汇合或分段边界')
        if any(m['status'] in ('same_name_offset','same_name_distant') for m in mm):reasons.append('同名线位置未直接对应')
        classes=wp['class_names']
        if name in KNOWN_MOATS: category='官方外苑濠；受控系统，逐口控制待核查'
        elif '濠' in name:category='名称含濠；独立控制资料待核查'
        elif classes==['river/stream']:category='源分类河川；实际控制和通水条件待核查'
        elif classes==['lake/pond']:category='源分类湖沼/池；用途与出口待核查'
        elif classes==['sea']:category='源分类海域；潮位边界待核查'
        else:category='源分类蓄水；用途与出口待核查'
        related=[e for key,e in edges.items() if wid in key]
        priority=1 if name in ('心字池','雲形池','日比谷濠') else (2 if assoc or cross or (nn and name_counts[nn]>1) or wp['area_within_three_wards_m2']>=5000 else 3)
        point=anchor(wg);lon,lat=LL.transform(point.x,point.y)
        row={'water_id':wid,'short_id':short[wid],'name':name,'source_names':wp['source_names'],
             'wards':[WARD[x] for x in wp['ward_intersections']],'source_class_codes':wp['class_codes'],
             'source_class_names':classes,'interpretation':category,'area_within_three_wards_m2':round(wg.area,3),
             'anchor_longitude':lon,'anchor_latitude':lat,'anchor_role':'footprint representative point, not outlet',
             'provisional_w05_ids':sorted(set(m['w05_id'] for m in assoc)),
             'candidate_match_ids':[m['match_id'] for m in mm],
             'match_status':'name_and_overlap' if any(m['status']=='name_and_overlap' for m in assoc) else ('overlap_name_unresolved' if assoc else 'no_direct_association'),
             'connections':[e['water_b'] if e['water_a']==wid else e['water_a'] for e in related],
             'all_connections_hydraulically_unverified':True,'review_priority':priority,'review_reasons':reasons,
             'inlet_crest_stage_status':'not verified; no street-to-water exchange assigned',
             'source_lod1_z_stats':wp['source_lod1_all_vertex_z_stats'],'height_use_status':wp['height_use_status'],
             'creation_dates':wp['creation_dates'],'source_ids':wp['source_ids'],'sources':wp['sources']}
        inventory.append((wg,row))
        if not nn and assoc:
            check(point,2,'unnamed_network_water',short[wid], '水域与W05线直接相交，但源名称不明；确认实际地物、名称和是否存在闸/管。',water_ids=[wid])
        elif nn and not assoc:
            check(point,max(2,priority),'named_water_without_W05',short[wid]+' '+name,'W05没有直接对应线；保留独立水体，核查入口、出口和实际用途。',water_ids=[wid])
        if nn and name_counts[nn]>1:
            check(point,2,'same_name_multiple_objects',short[wid]+' '+name,'同名对象未合并。核对是否同一水体的不同片段、被堤岸分隔，或名称标注范围不同。',water_ids=[wid])
    edge_records=[]
    for key,e in sorted(edges.items()):
        e['connection_id']=f'C{len(edge_records)+1:03}'
        e['review_priority']=2 if any('W05' in x['kind'] for x in e['evidence']) else 3
        ga,pa=wmap[key[0]];gb,pb=wmap[key[1]];geo=link_geometry(ga,gb)
        edge_records.append((geo,e))
        if e['review_priority']==2 and norm(e['name_a'])!=norm(e['name_b']):
            check(anchor(geo),2,'water_to_water_control_unknown',e['connection_id']+' '+e['name_a']+' / '+e['name_b'],
                  '几何或源线网提示相邻/连接；需核实明渠、闸门、通水管、堰或实体隔断及阈值，未给流向或流量。',
                  water_ids=list(key),connection_id=e['connection_id'])
    for entry in json.loads((ROOT/'data/exp/data-briefing/hibiya_water_corridor/priority_survey_list.json').read_text()):
        check(Point(*T.transform(entry['anchor_longitude'],entry['anchor_latitude'])),entry['priority'],'hibiya_existing_priority_site',
              entry['work_package_id']+' '+entry['location'],entry['measure_or_verify'],
              evidence=entry['existing_evidence'],historical_epoch_requirement=entry['historical_epoch_requirement'])
    # Official plan names Wadakura separately. Do not silently rename either source Baba-saki footprint.
    baba=[(g,p) for g,p in waters if p['name']=='馬場先濠']
    if baba and not any(p['name']=='和田倉濠' for g,p in waters):
        for g,p in baba:
            check(anchor(g),2,'official_name_not_separately_represented','馬場先濠 / 和田倉濠名称范围',
                  '官方2024外苑濠计划单列和田倉濠；两套源数据未以该名单列，PLATEAU有两个馬場先濠对象。核对官方范围图，未自动更名。',
                  water_ids=[p['water_id']],reference_url=MOE_URL)

    system_relations=[
        {'relation':'皇居外苑12濠受控连接系统','member_names':sorted(KNOWN_MOATS),'evidence_level':'official system-level description',
         'reference_url':MOE_URL,'reference':'2024-06 plan, printed page2/PDF page6; individual connection controls not acquired',
         'individual_gate_pipe_crests_capacities':'unknown','surface_free_flow_assumed':False},
        {'from':'日比谷濠','to':'公共下水道','relation':'overflow discharge described by official plan','evidence_level':'official system-level description',
         'reference_url':MOE_URL,'outfall_location':'not acquired','threshold_capacity_and_event_conditions':'unknown'},
        {'from':'清水濠','to':'日本橋川','relation':'partial discharge described by official plan','evidence_level':'official system-level description',
         'reference_url':MOE_URL,'outfall_location':'not acquired','threshold_capacity_and_event_conditions':'unknown'}]
    summary={'source_water_objects':len(waters),'named_objects':sum(bool(norm(p['name'])) for g,p in waters),
        'distinct_source_name_labels':len(name_counts),'unnamed_objects':sum(not norm(p['name']) for g,p in waters),
        'source_W05_segments':len(lines),'boundary_fragments_lt_1m':sum(r['fragment'] for g,r in line_audit),
        'water_match_status_counts':dict(Counter(r['match_status'] for g,r in inventory)),
        'W05_segments_with_direct_association':sum(bool(r['geometric_associated_water_ids']) for g,r in line_audit),
        'W05_low_any_water_coverage_lt_60pct':sum(not r['fragment'] and r['fraction_inside_any_plateau_water']<.60 for g,r in line_audit),
        'geometric_connection_candidates':len(edge_records),'W05_shared_endpoint_relations':len(node_relations),
        'located_checks':len(checks),'check_priority_counts':dict(Counter(r['priority'] for g,r in checks)),
        'hydraulically_verified_individual_connections':0,
        'count_note':'Objects, line segments and source names are not independent physical waterbody counts.'}
    for path,records in [('water_inventory.geojson',inventory),('w05_line_audit.geojson',line_audit),('connection_candidates.geojson',edge_records),('verification_locations.geojson',checks),('w05_shared_nodes.geojson',[(g,r) for g,r in node_relations if g is not None])]:fc(PROC/path,records)
    dump(OUT/'water_inventory.json',[r for g,r in inventory]);dump(OUT/'matches.json',matches)
    dump(OUT/'connection_candidates.json',[r for g,r in edge_records]);dump(OUT/'official_system_relations.json',system_relations)
    dump(OUT/'verification_list.json',[{'longitude':LL.transform(g.x,g.y)[0],'latitude':LL.transform(g.x,g.y)[1],**r} for g,r in checks])
    dump(OUT/'summary.json',summary)
    manifest={'parameters':PARAM,'source_sha256':{str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in (water_path,line_path,boundary_path,ROOT/'data/raw/ksj/W05-08_13_GML.zip')},
        'metric_crs':'EPSG:6677','output_horizontal_crs':'EPSG:6668','height_transformation':'none',
        'W05_source_epoch':'2008 FY','plateau_product':'2025 package; creationDate not measurement date',
        'topology_fields':'startStreamNode/endStreamNode on original un-clipped XML; shared nodes outside domain not lifted',
        'threshold_note':'25m is an exploratory search distance, not a positional accuracy claim; direct matches remain provisional',
        'official_sources':['https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-W05.html',MOE_URL]}
    dump(OUT/'method_and_provenance.json',manifest)
    # Consistency checks address meaningful risks: lost objects, orphan references, false directed edges.
    assert len(inventory)==126 and len(lines)==50
    assert len(set(r['water_id'] for g,r in inventory))==len(inventory)
    assert all(m['water_id'] in wmap and m['w05_id'] in lmap for m in matches)
    assert all(e['water_a'] in wmap and e['water_b'] in wmap and e['flow_direction']=='unassigned' for g,e in edge_records)
    assert all(g.is_valid and not g.is_empty for g,r in inventory+edge_records+checks)
    assert all(not r['confirmed_hydraulic_connection'] for g,r in checks)
    dump(OUT/'validation.json',{'objects_preserved':True,'all_foreign_keys_valid':True,'geometries_valid':True,
        'no_unverified_edges_given_flow_direction':True,'no_height_values_assigned':True,'parameters':PARAM})
    build_map(waters,lines,wards,checks,inventory)
    build_html(inventory,line_audit,matches,edge_records,checks,summary,wards)
    write_assessment(summary,inventory,line_audit,checks)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

def draw(ax,g,**kwargs):
    if g.geom_type=='Polygon':
        xy=np.asarray(g.exterior.coords);ax.fill(xy[:,0],xy[:,1],**kwargs)
        for ring in g.interiors:
            xy=np.asarray(ring.coords);ax.fill(xy[:,0],xy[:,1],color='white',lw=0)
    elif g.geom_type in ('LineString','LinearRing'):
        xy=np.asarray(g.coords);ax.plot(xy[:,0],xy[:,1],**kwargs)
    elif hasattr(g,'geoms'):
        for part in g.geoms:draw(ax,part,**kwargs)

def build_map(waters,lines,wards,checks,inventory):
    font=FontProperties(fname='/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
    fig,axs=plt.subplots(1,2,figsize=(15,10),gridspec_kw={'width_ratios':[1.2,1]},layout='constrained')
    for ax in axs:
        for g,p in wards:draw(ax,g.boundary,color='#8a929a',lw=.7)
        for g,r in inventory:draw(ax,g,color='#83b9ce' if r['match_status']!='no_direct_association' else '#bdc6cf',lw=.15,edgecolor='#52788a')
        for g,p in lines:draw(ax,g,color='#1e5478',lw=.8)
        for g,p in checks:
            ax.plot(g.x,g.y,'^' if p['priority']==1 else 'o',color='#b33425' if p['priority']==1 else '#bc7f22',markersize=5 if p['priority']==1 else 2.5,alpha=.85)
        ax.set_aspect('equal');ax.ticklabel_format(style='plain',useOffset=False)
        ax.set_xlabel('Easting (m; EPSG:6677)');ax.set_ylabel('Northing (m; EPSG:6677)')
    axs[0].set_title('都心三区：水体、W05线及核查锚点',fontproperties=font,fontsize=14)
    bounds=unary_union([g for g,p in wards]).bounds;axs[0].set_xlim(bounds[0]-100,bounds[2]+100);axs[0].set_ylim(bounds[1]-100,bounds[3]+100)
    axs[1].set_xlim(-7900,-5700);axs[1].set_ylim(-36900,-34200)
    axs[1].set_title('日比谷—外苑局部：相邻不代表自由通水',fontproperties=font,fontsize=14)
    for g,r in inventory:
        if r['name'] in ('心字池','雲形池','日比谷濠','凱旋濠','馬場先濠','桔梗濠','二重橋濠'):
            a=anchor(g)
            label=r['name']+' '+r['short_id']
            axs[1].annotate(label,(a.x,a.y),xytext=(5,5),textcoords='offset points',fontproperties=font,fontsize=9,
                            bbox={'facecolor':'white','alpha':.85,'edgecolor':'none','pad':1})
    axs[0].legend(handles=[Patch(facecolor='#83b9ce',label='有直接W05几何对应的水域'),Patch(facecolor='#bdc6cf',label='无直接W05对应的水域'),
        plt.Line2D([],[],color='#1e5478',label='W05 河川中心线'),plt.Line2D([],[],marker='^',color='#b33425',ls='',label='第一优先核查'),
        plt.Line2D([],[],marker='o',color='#bc7f22',ls='',label='其他核查锚点')],prop=font,loc='lower left')
    fig.suptitle('地表受纳水体清单与待核查连接 | 126个源水域对象，50条W05线段',fontproperties=font,fontsize=17)
    fig.savefig(OUT/'water_network_overview.png',dpi=170);plt.close(fig)

def svg_path(g):
    if g.geom_type=='Polygon':
        return ' '.join('M'+' L'.join(f'{x:.1f},{-y:.1f}' for x,y in ring.coords)+' Z' for ring in (g.exterior,*g.interiors))
    if g.geom_type=='LineString':return 'M'+' L'.join(f'{x:.1f},{-y:.1f}' for x,y in g.coords)
    if hasattr(g,'geoms'):return ' '.join(svg_path(p) for p in g.geoms)
    return ''

def build_html(inventory,line_audit,matches,edges,checks,summary,wards):
    esc=html.escape
    paths=[]
    for g,p in wards:paths.append(f'<path d="{svg_path(g)}" fill="none" stroke="#859199" stroke-width="1" vector-effect="non-scaling-stroke"/>')
    for g,r in inventory:
        color='#72afca' if r['match_status']!='no_direct_association' else '#b9c1ca'
        paths.append(f'<path data-water="{r["water_id"]}" d="{svg_path(g.simplify(.75,preserve_topology=True))}" fill="{color}" fill-rule="evenodd" stroke="#436d80" stroke-width=".5" vector-effect="non-scaling-stroke"><title>{esc(r["short_id"]+" "+r["name"])}</title></path>')
    for g,r in line_audit:paths.append(f'<path class="w05-line" d="{svg_path(g)}" fill="none" stroke="#164864" stroke-width="1" vector-effect="non-scaling-stroke" pointer-events="none"/>')
    for g,r in checks:paths.append(f'<circle data-check="{r["check_id"]}" cx="{g.x:.1f}" cy="{-g.y:.1f}" r="18" fill="{"#b33226" if r["priority"]==1 else "#b87913"}" stroke="white" stroke-width=".6" vector-effect="non-scaling-stroke"><title>{esc(r["check_id"]+" "+r["subject"])}</title></circle>')
    b=unary_union([g for g,p in wards]).bounds
    vb=f'{b[0]-120:.1f} {-b[3]-120:.1f} {b[2]-b[0]+240:.1f} {b[3]-b[1]+240:.1f}'
    status_text={'name_and_overlap':'名称与空间对应','overlap_name_unresolved':'空间对应，名称待核查','no_direct_association':'无直接W05对应'}
    rows=[]
    for g,r in sorted(inventory,key=lambda q:(q[1]['review_priority'],not norm(q[1]['name']),q[1]['name'],q[1]['short_id'])):
        rows.append(f'<tr data-row-water="{r["water_id"]}"><td><button class="pick" data-pick="{r["water_id"]}">{r["short_id"]}</button></td><td>{esc(r["name"])}</td><td>{"、".join(r["wards"])}</td><td>{esc(r["interpretation"])}</td><td>{status_text[r["match_status"]]}</td><td>{", ".join(r["provisional_w05_ids"]) or "—"}</td><td>{len(r["connections"])}</td><td>{esc("；".join(r["review_reasons"]) or "逐口控制、水位和入口仍待核查")}</td></tr>')
    check_rows=[]
    for g,r in sorted(checks,key=lambda q:(q[1]['priority'],q[1]['check_id'])):
        lon,lat=LL.transform(g.x,g.y)
        check_rows.append(f'<tr><td><button class="pick" data-check-pick="{r["check_id"]}">{r["check_id"]}</button></td><td>{r["priority"]}</td><td>{esc(r["subject"])}</td><td>{lon:.6f}, {lat:.6f}</td><td>{esc(r["detail"])}</td></tr>')
    payload=json.dumps({'waters':[r for g,r in inventory],'checks':[r for g,r in checks],
                        'edges':[r for g,r in edges],'matches':matches},ensure_ascii=False).replace('<','\\u003c')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>都心三区水体清单</title>
<style>body{margin:0;font:15px/1.65 system-ui,sans-serif;color:#18313e;background:#fafbfc}main{max-width:1280px;margin:auto;padding:24px}h1{font-size:26px}h2{font-size:20px;margin-top:32px}p{max-width:1000px}button,input,select{font:inherit}button{cursor:pointer;padding:5px 10px;border:1px solid #9cabb5;background:white;border-radius:4px}input,select{padding:6px 8px;margin:4px 10px 4px 0}svg{width:100%;height:580px;background:#edf1f3;touch-action:none}svg [data-water],svg [data-check]{cursor:pointer}svg .selected{stroke:#cb2d24;stroke-width:3;fill:#ffd785}.map-layout{display:grid;grid-template-columns:minmax(0,2fr) minmax(260px,1fr);gap:24px}.detail{overflow-wrap:anywhere}table{border-collapse:collapse;width:100%;background:white}th,td{padding:9px;text-align:left;border-bottom:1px solid #d5dde2;vertical-align:top}th{background:#e9eff3}td:first-child{white-space:nowrap}.table-wrap{overflow-x:auto}.small{font-size:13px;color:#526572}.legend{display:flex;gap:20px;flex-wrap:wrap}.matched{color:#176080}.unmatched{color:#697987}.check{color:#a95f10}.priority{color:#b33226}.controls{display:flex;flex-wrap:wrap;align-items:center;gap:8px}@media(max-width:800px){.map-layout{grid-template-columns:1fr}svg{height:420px}main{padding:16px}}@media(prefers-color-scheme:dark){body{background:#14222c;color:#e0eaf0}button,input,select,table{background:#223844;color:#e0eaf0}th{background:#29434f}svg{background:#1c303b}.small{color:#b4c6d0}}
</style><main><h1>都心三区水体清单与待核查连接</h1>
<p>126 个 PLATEAU 源水域对象 · 50 条 W05 源线段。对象、名称和线段数量不等于独立水体数量。蓝色表示直接几何对应，灰色表示没有直接对应；所有逐水体连接仍待水力核验。</p>
<p class="small">W05：2008 年、JGD2000，原始流路端点保留；PLATEAU：2025 包、JGD2011，创建日期不是测量日。没有使用源 Z 指定水位。25 m 只是检索范围。地图点为核查锚点，不是测量确认的入口。</p>
<div class="legend"><span class="matched">■ 有直接几何对应</span><span class="unmatched">■ 无直接对应</span><span class="priority">● 第一优先核查</span><span class="check">● 其他核查</span></div>
<div class="controls"><button id="all-map">三区全图</button><button id="hibiya-map">日比谷局部</button><button id="zoom-in">放大</button><button id="zoom-out">缩小</button></div>
<div class="map-layout"><svg id="map" viewBox="__VIEWBOX__" role="img" aria-label="都心三区水体与核查位置，支持拖动地图、点击水域和核查点"><title>三区水体与核查锚点</title>__PATHS__</svg><section class="detail"><h2>选中位置</h2><div id="detail" aria-live="polite">点击地图中的水域或核查点，或点击下方编号查看来源和待核查事项。</div></section></div>
<h2>水体清单</h2><div class="controls"><label>检索名称、区、W05编号 <input id="search" placeholder="例如 日比谷、中央区、r-750"></label><label>对应情况 <select id="status"><option value="">全部</option><option value="name_and_overlap">名称与空间对应</option><option value="overlap_name_unresolved">空间对应，名称待核查</option><option value="no_direct_association">无直接W05对应</option></select></label><span id="count"></span></div>
<div class="table-wrap"><table><thead><tr><th>编号</th><th>源名称</th><th>区</th><th>分类与控制</th><th>对应情况</th><th>直接对应W05</th><th>候选连接数</th><th>核查事项</th></tr></thead><tbody id="waters">__ROWS__</tbody></table></div>
<h2>定位核查清单</h2><p class="small">第一优先为已有日比谷工作包；第二优先检查线面不一致、名称范围和源线网提示的控制连接；第三优先为区界碎段等资料检查。未证实项不是已判定的数据错误。</p>
<div class="table-wrap"><table><thead><tr><th>编号</th><th>优先级</th><th>位置/对象</th><th>经度、纬度</th><th>需要核查</th></tr></thead><tbody>__CHECKS__</tbody></table></div>
<h2>已有官方系统关系</h2><p>环境省2024计划说明外苑12濠以水门和通水管组成受控系统，日比谷濠溢流水进入公共下水道，部分水由清水濠排入日本橋川。这是系统级描述；本成果未取得逐口位置、控制阈值、容量或事件水位，也未确认街道至池塘或濠的入口。</p>
<p>官方计划单列和田倉濠，两套源数据未以该名单列；PLATEAU有两个馬場先濠对象，保留原名并标记核查，没有自动更名。</p>
<p><a href="https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-W05.html">W05 官方说明</a> · <a href="https://www.env.go.jp/garden/content/000230549.pdf">环境省2024外苑濠计划</a> · <a href="assessment.txt">核查说明</a> · <a href="water_inventory.json">完整清单</a> · <a href="verification_list.json">定位核查记录</a></p>
<script id="data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('data').textContent),map=document.getElementById('map'),detail=document.getElementById('detail');
const waterById=new Map(data.waters.map(w=>[w.water_id,w])),checkById=new Map(data.checks.map(c=>[c.check_id,c]));
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const full=map.getAttribute('viewBox').split(' ').map(Number);let view=[...full];
function setView(v){view=v;map.setAttribute('viewBox',view.join(' '));}
function pickWater(id){const w=waterById.get(id);if(!w)return;map.querySelectorAll('.selected').forEach(x=>x.classList.remove('selected'));map.querySelector('[data-water="'+id+'"]').classList.add('selected');
const evidenceLabel={footprints_touch_or_overlap:'水域几何接触或重叠',footprints_near_25m:'水域相邻（25m检索范围）',consecutive_footprints_on_same_W05_line:'同一W05线依序经过',W05_shared_stream_endpoint:'W05流路共享端点'};
const links=data.edges.filter(e=>e.water_a===id||e.water_b===id).map(e=>{const o=waterById.get(e.water_a===id?e.water_b:e.water_a);return esc(e.connection_id+' '+o.short_id+' '+o.name+'：'+e.evidence.map(v=>evidenceLabel[v.kind]||v.kind).join('、'));});
const z=w.source_lod1_z_stats;
detail.innerHTML='<h3>'+esc(w.short_id+' '+w.name)+'</h3><p>'+esc(w.interpretation)+'</p><p>区：'+esc(w.wards.join('、'))+'<br>W05：'+esc(w.provisional_w05_ids.join(', ')||'无直接对应')+'</p><p>'+esc(w.review_reasons.join('；')||'入口、控制阈值和水位仍待核查')+'</p><p class="small">源LOD1顶点高程范围：'+Number(z.min_m).toFixed(2)+'–'+Number(z.max_m).toFixed(2)+' m；未作水位或河床。</p><p>候选关系（均未核实水力连通）：</p><ul>'+links.map(s=>'<li>'+s+'</li>').join('')+'</ul>';
}
function pickCheck(id){const c=checkById.get(id);if(!c)return;detail.innerHTML='<h3>'+esc(c.check_id+' '+c.subject)+'</h3><p>优先级 '+c.priority+'</p><p>'+esc(c.detail)+'</p><p class="small">核查锚点；不是已测入口。'+esc(c.evidence||'')+'</p>';map.querySelectorAll('.selected').forEach(x=>x.classList.remove('selected'));map.querySelector('[data-check="'+id+'"]').classList.add('selected');}
document.querySelectorAll('[data-pick]').forEach(b=>b.onclick=()=>pickWater(b.dataset.pick));document.querySelectorAll('[data-check-pick]').forEach(b=>b.onclick=()=>pickCheck(b.dataset.checkPick));
map.querySelectorAll('[data-water]').forEach(p=>p.onclick=()=>pickWater(p.dataset.water));map.querySelectorAll('[data-check]').forEach(p=>p.onclick=()=>pickCheck(p.dataset.check));
function filter(){const q=document.getElementById('search').value.trim().toLowerCase(),s=document.getElementById('status').value;let n=0;document.querySelectorAll('[data-row-water]').forEach(r=>{const w=waterById.get(r.dataset.rowWater),show=(!s||w.match_status===s)&&(!q||JSON.stringify(w).toLowerCase().includes(q));r.hidden=!show;if(show)n++;});document.getElementById('count').textContent=n+' 个源对象';}
document.getElementById('search').oninput=filter;document.getElementById('status').onchange=filter;filter();
document.getElementById('all-map').onclick=()=>setView([...full]);document.getElementById('hibiya-map').onclick=()=>setView([-7900,34200,2200,2700]);
function zoom(f){setView([view[0]+view[2]*(1-f)/2,view[1]+view[3]*(1-f)/2,view[2]*f,view[3]*f]);}
document.getElementById('zoom-in').onclick=()=>zoom(.65);document.getElementById('zoom-out').onclick=()=>zoom(1.5);
let drag=null;map.addEventListener('pointerdown',e=>{if(e.target.hasAttribute('data-water')||e.target.hasAttribute('data-check'))return;drag={x:e.clientX,y:e.clientY,v:[...view]};map.setPointerCapture(e.pointerId);});map.addEventListener('pointermove',e=>{if(!drag)return;const rect=map.getBoundingClientRect(),scale=Math.max(drag.v[2]/rect.width,drag.v[3]/rect.height);setView([drag.v[0]-(e.clientX-drag.x)*scale,drag.v[1]-(e.clientY-drag.y)*scale,drag.v[2],drag.v[3]]);});map.addEventListener('pointerup',()=>drag=null);map.addEventListener('pointercancel',()=>drag=null);
</script></main></html>'''
    page=page.replace('__VIEWBOX__',vb).replace('__PATHS__',''.join(paths)).replace('__ROWS__',''.join(rows)).replace('__CHECKS__',''.join(check_rows)).replace('__DATA__',payload)
    page=page.replace('h1{font-size:26px}', 'h1{font-size:26px;text-wrap:balance}').replace('svg{background:#1c303b}', 'svg{background:#1c303b}.w05-line{stroke:#86bfd6}')
    (OUT/'water_inventory.html').write_text(page,encoding='utf-8')

def write_assessment(summary,inventory,line_audit,checks):
    rows=['都心三区水体清单与连接关系核查','2026-10-13；项目根目录 /Volumes/Extension/Research/UDT01b','',
          '本轮完成：W05河川线与已有PLATEAU水域多对多对应，126个源对象全部保留；原名、原分类、原高程统计和文件来源不覆盖。所有逐对象水力连接仍未确认。',
          json.dumps(summary,ensure_ascii=False,indent=2),'',
          '方法：JGD2011经纬度转EPSG:6677，以米计算线在水域内的长度及最近距离。超过5m直接相交的同名或至少一方未命名对象登记为暂定几何对应。已知名称不同的相交记为跨命名面，不自动更名或合并。25m仅作为候选检索范围，不是位置精度。源W05已从JGD2000转换；垂直高程未转换。',
          '原始W05 XML的startStreamNode/endStreamNode用来检查源线网共享节点；没有使用覆盖整条河川的startRiverNode/endRiverNode来虚构连接。区界之外共享节点不投影为本区水体间关系，裁剪端点不当作实际河流端点。',
          '连接分三种证据：水域接触/25m内相邻、同一W05线依序跨水域、原始W05共享流路节点。全部无方向、无流量、无控制阈值。最近间隙线只定位核查，不代表真实明渠或管线。',
          'W05基准2008年，PLATEAU为2025包（创建日期不等于测量时间）。粗尺度线跨到另一具名水域可能反映汇合、命名分段或时期差异，不直接判为错误。W05缺线不能解释为无水体；PLATEAU线覆盖空缺可能是桥下、暗渠或采集范围问题，不能解释为断流。',
          '', '具名水体（按原始标签分组仅供查阅，不合并几何）：']
    grouped=defaultdict(list)
    for g,r in inventory:
        if norm(r['name']):grouped[r['name']].append(r)
    for name,items in sorted(grouped.items()):
        rows.append(name+' | '+','.join(r['short_id'] for r in items)+' | '+'、'.join(sorted(set(w for r in items for w in r['wards'])))+' | W05 '+','.join(sorted(set(s for r in items for s in r['provisional_w05_ids']))))
    rows+=['','重点核查：',
        '1. 日比谷S1共用道路—公园过渡、S2心字池南岸、S3日比谷濠南端：沿用已有定位与点云证据，核实真实控制岸顶、开口与受纳水位；尚未确认道路入水入口。S4–S6仍按之前优先级记录。',
        '2. 外苑各濠：几何接触和W05共享节点不构成自由通水证据；需要逐口闸/管/堰、底顶标高和操作条件。官方2024计划说明12濠构成受控系统；日比谷濠溢流入公共下水道，部分由清水濠入日本橋川，只登记系统级关系，未编造逐口位置。',
        '3. 和田倉濠名称范围：官方12濠清单包含该名，两套源数据未以该名单列。PLATEAU的两个馬場先濠对象保留原名并各设锚点，未将任一自动改名为和田倉濠。',
        '4. 神田川、日本橋川、古川等线面覆盖不足的位置：重点检查桥下缺面、暗渠或不同时期几何。显著跨入其他具名水域的线，核对汇合口或命名边界。',
        '5. 三个不足1m的区界线段：单列资料碎段，不能拿它们的河名覆盖隅田川水域，也不能计作三条独立河流。',
        '6. 未命名水域与同名多对象：保留对象和来源；优先给与源线网有关的未名对象补名称和用途，其他小面先作地图核验，是否需要实地测量由其对排水路径的影响决定。',
        '', '每个连接进入模拟前需记录：真实连接形式、方向/可逆性、入口及岸顶高程、受纳水位时间/基准、开口宽度或控制能力、数据时期和维护/操作条件。未知位置不赋零水位、零岸顶或无限排水。三区边界是数据裁剪范围，不是水力封闭边界。',
        '', '成果：water_inventory.html 可检索水体并查看核查锚点；JSON保留完整清单和多对多关系；data/processed/central_three_water_network/的GeoJSON可导入GIS。water_network_overview.png为概览图，精确定位使用GeoJSON/核查记录。',
        '', '来源： https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-W05.html',MOE_URL,
        '官方系统关系依据2024-06计划正文印刷页2（PDF第6页）；只有系统级事实，不从示意图推算具体水口的标高或能力。']
    (OUT/'assessment.txt').write_text('\n'.join(rows)+'\n',encoding='utf-8')

if __name__=='__main__':main()
