"""Conserve source driver-candidate OD,declare fine allocation andnative road access.
This is conditional model input preparation,not an observed fine-scale vehicle OD.
"""
from pathlib import Path
import os,json,csv,hashlib,gzip,time,datetime as dt
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np,pandas as pd,geopandas as gpd
from shapely.geometry import shape,Point,LineString
from shapely import STRtree,points,distance
from pyproj import Transformer
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'data/processed/three_ward_normal_vehicle_baseline_v1';R=ROOT/'data/exp/estimation/three_ward_normal_vehicle_baseline_v1';F=ROOT/'data/processed/three_ward_full_pt_od_analysis_v1';B=ROOT/'data/processed/three_ward_traffic_baseline_inputs_v1';N=ROOT/'data/processed/three_ward_normal_transport_network_v1';M=ROOT/'data/processed/three_ward_motor_flood_access_v1'
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda x:x.item()if isinstance(x,np.generic)else str(x))+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def readnp(p):
 with np.load(p,allow_pickle=False)as z:return {k:z[k]for k in z.files}
def main():
 start=time.monotonic();R.mkdir(parents=True,exist_ok=True)
 lookup=pd.read_csv(B/'pt_2018_zone_code_crosswalk.csv',dtype=str);fine_to_mid=lookup.drop_duplicates('planning_zone').set_index('planning_zone').middle_zone.to_dict();knownmid=set(fine_to_mid.values())
 src=pd.read_csv(F/'decoded_tables/H30_5_3.csv',skiprows=4,header=None,dtype=str,keep_default_na=False)
 for c in [1,2]:src['o'if c==1 else'd']=src[c].str.extract(r':([^:\s]+)',expand=False)
 src['q']=pd.to_numeric(src[11]);src['source_row']=np.arange(len(src))+5;allq=float(src.loc[src[0].eq('合計')&src.o.eq('888')&src.d.eq('888'),'q'].iloc[0])
 controls=src[src[0].eq('合計')&src.o.isin(knownmid)&src.d.isin(knownmid)].copy();assert not controls.duplicated(['o','d']).any()
 seeds=pd.read_csv(F/'all_partition_car_person_od.csv',dtype={'origin_code':str,'destination_code':str});seeds=seeds[seeds.origin_code.isin(fine_to_mid)&seeds.destination_code.isin(fine_to_mid)&seeds.car_person_trips.gt(0)].copy();seeds['o']=seeds.origin_code.map(fine_to_mid);seeds['d']=seeds.destination_code.map(fine_to_mid);sums=seeds.groupby(['o','d']).car_person_trips.sum();qlookup=controls.set_index(['o','d']).q.to_dict();seeds['middle_q']=np.array([qlookup.get((o,d),0)for o,d in zip(seeds.o,seeds.d)],float);seeds['seed_denominator']=np.array([sums[o,d]for o,d in zip(seeds.o,seeds.d)],float);seeds['driver_candidate_daily']=seeds.middle_q*seeds.car_person_trips/seeds.seed_denominator;seeds=seeds[seeds.driver_candidate_daily.gt(0)].copy()
 allocated=seeds.groupby(['o','d']).driver_candidate_daily.sum();ledger=[]
 for r in controls.itertuples():
  got=float(allocated.get((r.o,r.d),0));has=(r.o,r.d)in sums;ledger.append({'origin_middle_zone':r.o,'destination_middle_zone':r.d,'native_driver_candidate_daily':int(r.q),'allocated_fine_daily':got,'held_no_positive_person_seed':float(r.q)if not has else 0.,'difference':got+(float(r.q)if not has else 0.)-float(r.q),'source_row_1_based':int(r.source_row)})
 pd.DataFrame(ledger).to_csv(P/'middle_control_conservation.csv',index=False,encoding='utf-8-sig');seeds.to_csv(P/'conditional_fine_driver_od.csv',index=False,encoding='utf-8-sig')
 # Existing directed graph andturn-state mapping are retained unchanged.
 z=readnp(M/'source_motor_graph.npz');router=readnp(M/'turn_router.npz');proj=readnp(M/'road_projection.npz');nodeids=z['node_ids'];nd=pd.read_csv(N/'surface_nodes.csv',usecols=['node_id','x_m','y_m'],dtype={'node_id':str}).set_index('node_id');xy=nd.loc[nodeids,['x_m','y_m']].to_numpy();del nd
 roads=pd.read_csv(M/'road_inventory.csv',dtype={'source_edge_id':str});ground=roads.grade_key.eq('0|0|surface').to_numpy();nonmotorway=~roads.highway.isin(['motorway','motorway_link','trunk','trunk_link']).to_numpy();goodarc=ground[proj['arc_to_road']]&nonmotorway[proj['arc_to_road']];candidate=np.unique(np.r_[proj['from_index'][goodarc],proj['to_index'][goodarc]]);largest=int(np.argmax(np.bincount(z['strong_component'])));candidate=candidate[z['strong_component'][candidate]==largest]
 archive=(B/'sources/H30_gis.zip').resolve();zones=gpd.read_file(f'/vsizip/{archive}/H30_gis/H30_kzone.shp',encoding='cp932');zones['code']=zones.kzone.map(lambda x:f'{int(x):04d}');zones=zones.sort_values('code');geoms=zones.geometry.to_numpy();codes=zones.code.to_numpy();tree=STRtree(geoms);pairs=tree.query(points(xy[candidate]),predicate='covered_by');byzone={}
 for a,b in pairs.T:byzone.setdefault(str(codes[b]),[]).append(int(candidate[a]))
 features=json.loads((ROOT/'data/processed/three_ward_transport_framework_v1/inventory_domains_6677.geojson').read_text())['features'];core=shape(features[0]['geometry']);domain=core.buffer(10000);center=np.array([core.centroid.x,core.centroid.y]);component=router['node_component'];used=set();endpoints=[];mapping=[];local_by={}
 for code_,g in zip(codes,geoms):
  choices=np.array(byzone.get(str(code_),[]),dtype=int)
  if not len(choices):continue
  rp=g.representative_point();dist2=np.sum((xy[choices]-[rp.x,rp.y])**2,axis=1);order=np.argsort(dist2,kind='stable');chosen=next((int(choices[j])for j in order if int(component[choices[j]])not in used),None)
  if chosen is None:continue
  i=len(endpoints);local_by[str(code_)]=i;used.add(int(component[chosen]));endpoints.append({'endpoint_index':i,'kind':'local_zone_representative','planning_zone':str(code_),'native_node_index':chosen,'native_node_id':str(nodeids[chosen]),'component':int(component[chosen]),'x_m':xy[chosen,0],'y_m':xy[chosen,1],'distance_to_zone_reference_m':float(np.sqrt(np.min(dist2))),'zone_area_inside_road_domain_share':float(g.intersection(domain).area/g.area),'assumption':'One model representative,not observed address/parking access'})
 # Native boundary portals; no spatial intersections orconnectingroads arecreated.
 major=roads.highway.isin(['motorway','motorway_link','trunk','trunk_link','primary','primary_link','secondary','secondary_link','tertiary','tertiary_link']).to_numpy();pa=np.unique(np.r_[proj['from_index'][major[proj['arc_to_road']]],proj['to_index'][major[proj['arc_to_road']]]]);pa=pa[z['strong_component'][pa]==largest];bd=distance(points(xy[pa]),domain.boundary);pa=pa[bd<=500];bd=bd[bd<=500];angles=np.arctan2(xy[pa,1]-center[1],xy[pa,0]-center[0]);portals={}
 for sector in range(16):
  angle=2*np.pi*sector/16;diff=np.abs(np.angle(np.exp(1j*(angles-angle))));order=np.argsort(diff*15000+bd,kind='stable');chosen=next((int(pa[j])for j in order if int(component[pa[j]])not in used),None)
  if chosen is None:raise ValueError('No distinctnativeportal')
  j=int(np.flatnonzero(pa==chosen)[0]);i=len(endpoints);portals[sector]=i;used.add(int(component[chosen]));endpoints.append({'endpoint_index':i,'kind':'conditional_outer_boundary_portal','planning_zone':'','native_node_index':chosen,'native_node_id':str(nodeids[chosen]),'component':int(component[chosen]),'x_m':xy[chosen,0],'y_m':xy[chosen,1],'boundary_distance_m':float(bd[j]),'sector':sector,'assumption':'Bearing-basedboundaryassignment,not observed entrypoint'})
 if len(endpoints)>180:raise ValueError('Registeredoriginbudgetexceeded')
 zone_ep={};zone_local={};zone_geom={str(c):g for c,g in zip(codes,geoms)}
 for code_,g in zip(codes,geoms):
  key=str(code_);rp=g.representative_point();angle=np.arctan2(rp.y-center[1],rp.x-center[0]);sector=int(np.round(angle/(2*np.pi)*16))%16;islocal=key in local_by;ep=local_by[key]if islocal else portals[sector];zone_ep[key]=ep;zone_local[key]=islocal;mapping.append({'planning_zone':key,'endpoint_index':ep,'is_local_zone_representative':islocal,'representative_x_m':rp.x,'representative_y_m':rp.y,'native_access_assumption':'within_zone_modelpoint'if islocal else'conditional_remote_boundary_portal'})
 pd.DataFrame(endpoints).to_csv(P/'network_endpoints.csv',index=False,encoding='utf-8-sig');pd.DataFrame(mapping).to_csv(P/'zone_to_network_mapping.csv',index=False,encoding='utf-8-sig')
 # Native clockproxy: daily,8am,10am,5pm,and7–19h. Unknown clock isnotfilled.
 clock=pd.read_csv(F/'decoded_tables/H30_c3.csv',skiprows=4,header=None,dtype=str,keep_default_na=False);clock['zone']=clock[2].str.extract(r':([^:\s]+)',expand=False);clock['car']=pd.to_numeric(clock[6]);clock['plain']=clock[3].map(lambda x:__import__('unicodedata').normalize('NFKC',x));ratios={};clockledger=[]
 for code_,group in clock[clock.zone.isin(fine_to_mid)].groupby('zone'):
  total=float(group.loc[group[3].isin(['計','合計']),'car'].sum());exact={int(m.group(1)):float(v)for label,v in zip(group.plain,group.car)if(m:=__import__('re').fullmatch(r'(\d+)時台',label))};unknown=float(group.loc[group[3].eq('不明'),'car'].sum());weights=[1.]+[exact.get(h,0.)/total if total else 0. for h in [8,10,17]]+[sum(exact.get(h,0.)for h in range(7,19))/total if total else 0.];ratios[code_]=weights;clockledger.append({'planning_zone':code_,'source_car_person_daily':total,'unknown_clock':unknown,'known_exact_hour_sum':sum(exact.values()),'weights_daily_hour8_hour10_hour17_12h':json.dumps(weights)})
 pd.DataFrame(clockledger).to_csv(P/'clock_proxy_ledger.csv',index=False,encoding='utf-8-sig')
 epindex=np.array([zone_ep[c]for c in seeds.origin_code],int);dpindex=np.array([zone_ep[c]for c in seeds.destination_code],int);weights=seeds.driver_candidate_daily.to_numpy()[:,None]*np.array([ratios[c]for c in seeds.origin_code]);base=np.array([zone_local[o]or zone_local[d]for o,d in zip(seeds.origin_code,seeds.destination_code)]);through=np.zeros(len(seeds),bool)
 for j,(o,d)in enumerate(zip(seeds.origin_code,seeds.destination_code)):
  if not base[j]and epindex[j]!=dpindex[j]:
   a=zone_geom[o].representative_point();b=zone_geom[d].representative_point();through[j]=LineString([(a.x,a.y),(b.x,b.y)]).intersects(domain)
 same=epindex==dpindex;eligible_base=base&~same;eligible_plus=(base|through)&~same
 np.savez_compressed(P/'routing_demands.npz',origin_endpoint=epindex,destination_endpoint=dpindex,period_weights=weights,base=eligible_base,through_plus=eligible_plus,same_representative=same,local_base_candidate=base,chord_through_candidate=through)
 np.savez_compressed(P/'endpoint_components.npz',components=np.array([x['component']for x in endpoints]),native_node_index=np.array([x['native_node_index']for x in endpoints]),coordinates=np.array([[x['x_m'],x['y_m']]for x in endpoints]),node_xy=xy)
 checks={'middle_controls_conserved':max(abs(r['difference'])for r in ledger)<1e-7,'native_graph_not_changed':True,'representatives_are_native_graph_nodes':all(x['native_node_id']==str(nodeids[x['native_node_index']])for x in endpoints),'distinct_endpoint_components':len(used)==len(endpoints),'origin_budget':len(endpoints)<=180,'remote_source_quantities_retained':len(epindex)==len(seeds),'nonnegative_driver_weights':bool((weights>=0).all()),'registered_protocol_exists':(P/'protocol.json').exists()}
 summary={'prepared_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'whole_region_native_driver_candidate_daily':allq,'known_middle_driver_candidate_daily':float(controls.q.sum()),'allocated_fine_driver_candidate_daily':float(seeds.driver_candidate_daily.sum()),'held_no_fine_seed_daily':sum(r['held_no_positive_person_seed']for r in ledger),'unlocated_or_other_source_geography_daily':allq-float(controls.q.sum()),'fine_positive_od_pairs':len(seeds),'network_endpoints':len(endpoints),'local_fine_zone_representatives':len(local_by),'outer_native_portals':16,'main_strong_component':largest,'base_route_candidate_daily':float(weights[eligible_base,0].sum()),'plus_chord_through_candidate_daily':float(weights[eligible_plus,0].sum()),'same_representative_held_daily':float(weights[same,0].sum()),'bothremote_noteligible_base_daily':float(weights[~base,0].sum()),'periods':['daily','hour8','hour10','hour17','daytime12h'],'elapsed_s':time.monotonic()-start,'checks':checks}
 dump(R/'preparation_summary.json',summary);dump(P/'input_bindings.json',{'sha256':{str(p.relative_to(ROOT)):sha(p)for p in [F/'sources/H30_5_3.xlsx',F/'decoded_tables/H30_5_3.csv',F/'all_partition_car_person_od.csv',F/'decoded_tables/H30_c3.csv',B/'pt_2018_zone_code_crosswalk.csv',B/'sources/H30_gis.zip',M/'source_motor_graph.npz',M/'turn_router.npz',M/'road_projection.npz',M/'road_inventory.csv',N/'surface_nodes.csv',N/'surface_physical_edges.jsonl.gz']}});print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
 if not all(checks.values()):raise SystemExit('Candidatepreparationchecksfailed')
if __name__=='__main__':main()
