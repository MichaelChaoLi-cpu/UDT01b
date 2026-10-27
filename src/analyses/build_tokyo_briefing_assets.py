#!/usr/bin/env python3
"""Export display-only dashboard assets from existing approved research outputs."""
from pathlib import Path
import sys,json,gzip,hashlib,datetime
import numpy as np,pandas as pd
from shapely.geometry import shape,mapping
from shapely.ops import transform
from pyproj import Transformer
ROOT=Path(__file__).resolve().parents[2];SITE=ROOT/'data/exp/dashboards/tokyo-resilience-briefing-v1';OUT=SITE/'dist/assets'
BASE=ROOT/'data/processed/three_ward_five_event_temporal_analysis_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=True,separators=(',',':'),allow_nan=False))
def coords(geom):
 def rec(x):
  if isinstance(x,(tuple,list)) and len(x)>=2 and isinstance(x[0],(int,float)):return [round(x[0],1),round(x[1],1)]
  return [rec(y) for y in x]
 return {'type':geom.geom_type,'coordinates':rec(mapping(geom)['coordinates'])}
def main():
 OUT.mkdir(parents=True,exist_ok=True);inputfiles=[]
 with np.load(BASE/'clipped_road_lengths.npz') as z:weights=np.vstack([z['union_length_m'],z['ward_length_m']]);qualified=z['qualified']
 geodata=json.loads((BASE/'clipped_roads_6677.geojson').read_text());ids=np.array([f['properties']['road_index'] for f in geodata['features']],dtype=int);assert len(np.unique(ids))==len(ids)
 original=[shape(f['geometry']) for f in geodata['features']];assert np.max(abs(np.array([g.length for g in original])-weights[0,ids]))<1e-6
 roads=[coords(g.simplify(2.,preserve_topology=True))['coordinates'] if g.geom_type=='MultiLineString' else [coords(g.simplify(2.,preserve_topology=True))['coordinates']] for g in original]
 wards=json.loads((BASE/'wards_6677.geojson').read_text());wn={'13101':'Chiyoda','13102':'Chuo','13103':'Minato'};warddata=[]
 for f in wards['features']:
  g=shape(f['geometry']);point=g.representative_point();warddata.append({'name':wn[str(f['properties']['ward'])],'shape':coords(g.simplify(3.,preserve_topology=True)),'label':[point.x,point.y]})
 rawwater=ROOT/'data/processed/plateau_water/central_three_wards_wtr.geojson';waterjson=json.loads(rawwater.read_text());project=Transformer.from_crs(4326,6677,always_xy=True).transform;waters=[coords(transform(project,shape(f['geometry'])).simplify(2.,preserve_topology=True)) for f in waterjson['features']]
 geometry={'roads':roads,'roadWeights':weights[:,ids].T.tolist(),'wards':warddata,'water':waters,'bounds':[-11360,-42440,-3660,-32320],'crs':'EPSG:6677','cartographicSimplificationMetres':2}
 (OUT/'geometry.bin').write_bytes(gzip.compress(json.dumps(geometry,separators=(',',':')).encode(),compresslevel=6));print('GEOMETRY',len(ids),flush=True)
 traces=pd.read_csv(BASE/'road_trajectories.csv');traces=traces[traces.arm=='flood'].copy();rain=pd.read_csv(BASE/'rainfall_trajectories.csv');events=sorted(rain.event_id.unique());sourcepos=np.full(len(qualified),-1,int);sourcepos[ids]=np.arange(len(ids));series={};validations=[];caseassets={}
 for event in events:
  for stage in ['L','H']:
   key=event+'__'+stage+'__U';f=BASE/'exposure'/(key+'.npz');inputfiles.append(f)
   with np.load(f) as z:
    q=z['qualified_road_indices'];where=sourcepos[q];keep=where>=0;assert np.all(weights[:,q[~keep]]==0.);where=where[keep];stack=[]
    for bound in ['lower','upper']:
     states=np.full((109,len(ids)),3,dtype='u1');depth=z[bound][:,keep];states[:,where]=np.where(depth>=.3,2,np.where(depth>=.1,1,0));stack.append(states)
     for scope,col in [('Three wards',0),('13101',1),('13102',2),('13103',3)]:
      g=traces[(traces.event_id==event)&(traces.stage==stage)&(traces.bound==bound)&(traces.scope==scope)].sort_values('frame');assert len(g)==109
      closed=(states==2)@(weights[col,ids]/1000);slow=(states==1)@(weights[col,ids]/1000);opened=(states==0)@(weights[col,ids]/1000);assert np.max(abs(closed-g.qualified_closed_km.to_numpy()))<1e-7;assert np.max(abs(slow-g.qualified_slow_km.to_numpy()))<1e-7
      series[event+'_'+stage+'_'+bound+'_'+str(col)]={'closed':g.qualified_closed_km.tolist(),'slow':g.qualified_slow_km.tolist(),'open':g.qualified_open_km.tolist(),'unknown':float(g.unknown_exposure_km.iloc[0]),'exposure':float(np.trapezoid(g.qualified_closed_km,g.analysis_elapsed_s)/3600)}
    raw=np.stack(stack).tobytes();name=key+'.bin';(OUT/name).write_bytes(gzip.compress(raw,compresslevel=6));caseassets[event+'_'+stage]=name;validations.append({'case':key,'all_109_states_both_bounds_match_source_lengths':True})
   print('EXPORTED',key,flush=True)
 intake=pd.read_csv(ROOT/'data/processed/three_ward_shared_receiver_pilot_v1/pump_sites.csv');source=pd.read_parquet(ROOT/'data/processed/three_ward_intake_receiver_candidates_v1/source_candidates_preprocessed.parquet').set_index('Source Candidate');receiver=pd.read_parquet(ROOT/'data/processed/three_ward_intake_receiver_candidates_v1/receiver_candidates_preprocessed.parquet').set_index('Receiver Candidate')
 intakes=[]
 for row in intake.to_dict('records'):
  s=source.loc[row['Source Candidate']];r=receiver.loc[row['Receiver Candidate']];intakes.append({'id':int(row['Pump ID']),'source':[float(s['Source Easting']),float(s['Source Northing'])],'receiver':[float(r['Receiver Easting']),float(r['Receiver Northing'])],'pool':row['Receiving Pool'],'rate':float(row['Reserved Rate']),'length':float(row['Straight Distance Lower Bound'])})
 portfolios=pd.read_parquet(ROOT/'data/processed/three_ward_shared_receiver_allocation_v1/portfolio_summary_preprocessed.parquet');portfolio_data=[]
 for row in portfolios.to_dict('records'):portfolio_data.append({'id':row['Portfolio'],'reach':int(row['Straight Distance Screen']),'horizon':int(row['Reference Clearance Horizon']/60),'receiver':row['Receiving Scenario'],'intakes':row['Activated Model Intakes'],'rate':row['Reserved Rate']})
 metadata={'title':'Tokyo | Storm to Street','createdAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),'events':events,'eventLabels':{e:e[1:5]+'-'+e[5:7]+'-'+e[7:9] for e in events},'caseAssets':caseassets,'roadCount':len(ids),'frames':109,'series':series,'rain':{e:rain[rain.event_id==e].source_area_mean_mm_h.tolist()+[0] for e in events},'qualifiedRoadKm':float(weights[0,qualified].sum()/1000),'unknownRoadKm':float(weights[0,~qualified].sum()/1000),'residentStock':496345,'residentYear':2020,'intakes':intakes,'portfolios':portfolio_data,'liveEndpoint':'/api/live','thresholds':{'restricted':.1,'closed':.3},'evidence':['Five historical rainfall parents; prescribed water levels are scenario assumptions.','Map and road curves show archived no-action results, not live treatment outcomes.','Road closure at depth >=0.30m; restriction from0.10m. Thresholds do not certify safety.','Unknown bridge/tunnel exposure remains separate. Resident stock is not an affected-person count.','Pilot locations and receiver allowances are hypothetical. Real transfer routes and acceptance remain unverified.','3D city geometry constrains a2D surface-water solver; this is not full3D fluid simulation.']}
 dump(OUT/'briefing.json',metadata)
 inputfiles +=[BASE/n for n in ['clipped_road_lengths.npz','clipped_roads_6677.geojson','wards_6677.geojson','road_trajectories.csv','rainfall_trajectories.csv']]+[rawwater]
 dump(SITE/'asset_validation.json',{'passed':True,'cartographic_display_only':True,'road_geometry_source_lengths_exact':True,'cases':validations,'states_exported':len(events)*2*2*109,'all_scopes_both_bounds_match_original_road_totals':True,'no_new_hydraulic_runs':True,'no_source_modifications':True,'inputs':{str(f.relative_to(ROOT)):sha(f) for f in inputfiles}})
 print('BRIEFING_ASSETS_PASSED',flush=True)
if __name__=='__main__':main()
