"""Qualify fixed rainfall/geometry/epoch contracts before starting any new solver run."""
import os
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
from pathlib import Path
import sys,json,datetime as dt
import numpy as np
from netCDF4 import Dataset
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/preprocessing')]
from prepare_cross_event_protection_v1 import P,R,D,OLD,F,MEASURE,PAIR,sha,dump,load

def main():
 checks=[]
 def check(name,v):checks.append(dict(check=name,passed=bool(v)));assert v,name
 manifest=json.loads((P/'run_manifest.json').read_text());protocol=json.loads((D/'protocol.json').read_text());selection=json.loads((P/'event_selection.json').read_text());runs=manifest['runs'];old=json.loads((F/'run_manifest.json').read_text());pair=load(PAIR/'paired_support_and_native_mapping.npz')
 check('bounded8cases2newparents',len(runs)==8 and len({r['parent_id']for r in runs})==2 and len({(r['event_id'],r['arm'],r['stage'])for r in runs})==8 and manifest['maximum_new_runs']==8)
 check('frozen_protocol',sha(D/'protocol.json')==manifest['protocol_sha256'])
 check('fixed_original_measures',all(sha(ROOT/f)==h for f,h in json.loads((MEASURE/'frozen_selection_binding.json').read_text()).items()))
 check('both_profiles_complete',all(e['missing_supported']==e['saturated_supported']==0 for e in selection['events']))
 check('window_selection_no_outcome_dependence',selection['selection_uses_rainfall_only']and selection['measures_not_reselected'])
 check('contrasting_rainfall_concentration',selection['events'][1]['peak60_fraction_of9h']>selection['events'][0]['peak60_fraction_of9h']*2)
 adapter=json.loads((P/'audit_metadata_adapter.json').read_text());source=ROOT/adapter['original_path'];derived=ROOT/adapter['derived_path'];text=derived.read_text()
 for edit in reversed(adapter['reversible_exact_metadata_edits']):assert text.count(edit['derived'])==1;text=text.replace(edit['derived'],edit['original'])
 check('audit_numeric_gates_exactly_unchanged',text==source.read_text()and sha(source)==adapter['original_sha256']and sha(derived)==adapter['derived_sha256'])
 for r in runs:
  folder=ROOT/r['folder'];check(r['case_id']+'_input_binding',all(sha(folder/f)==h for f,h in r['input_sha256'].items()))
  template=next(x for x in old['runs']if x['case_id']=='R1__L__'+r['arm']);src=ROOT/template['folder'];kv=lambda p:{line.partition('=')[0].strip():line.partition('=')[2].strip()for line in p.read_text().splitlines()if'='in line};new=kv(folder/'sfincs.inp');before=kv(src/'sfincs.inp');check(r['case_id']+'_only_epoch_inp_changes',set(new)==set(before)and all(new[k]==before[k]for k in new if k not in ['tref','tstart','tstop']))
  begin=dt.datetime.fromisoformat(r['window_start_utc']);check(r['case_id']+'_true_utc_epoch',new['tref']==new['tstart']==begin.strftime('%Y%m%d %H%M%S')and new['tstop']==(begin+dt.timedelta(hours=9)).strftime('%Y%m%d %H%M%S')and r['rain_time_units']=='minutes since '+begin.strftime('%Y-%m-%d %H:%M:%S'))
  check(r['case_id']+'_no_previous_attempt',not(folder/'launch.json').exists()and not(folder/'sfincs_map.nc').exists())
 # Distinct forcing fields: exact spatial export,109 START-stamped knots including terminalzero.
 for event in selection['events']:
  folder=P/'rainfall'/event['event_id'];rad=load(folder/'radar_intervals.npz');check(event['event_id']+'_contiguous_original5min',len(rad['interval_end_unix_s'])==108 and np.all(np.diff(rad['interval_end_unix_s'])==300)and rad['interval_start_unix_s'][0]==dt.datetime.fromisoformat(event['window_start_utc']).timestamp())
  with Dataset(folder/'forcing_intervals_10m.nc')as ref:
   for g in ['T','U']:
    with Dataset(P/'forcing'/(event['event_id']+'__'+g)/'historical_rain.nc')as nc:
     field=nc['Precipitation'];ok=True;rates=[];inside=pair['Terrain Active Mask']==1 if g=='T'else pair['Urban Active Mask']==1
     for k in range(108):
      a=np.asarray(field[k]);expected=np.asarray(ref['terrain_effective_rainfall'if g=='T'else'urban_effective_rainfall'][k]);ok &=np.array_equal(np.flipud(a[1:-1,1:-1]),expected)and np.isfinite(a).all()and(a>=0).all();rates.append(float(np.flipud(a[1:-1,1:-1])[inside].sum(dtype='f8')*100/12/1000))
     nominal=json.loads((folder/'spatial_forcing_qa.json').read_text())['total_reconstructed_source_rain_volume_m3'];check(event['event_id']+'_'+g+'_exact_export_clock_volume',ok and np.all(field[108]==0)and np.array_equal(nc['time'][:]*60,np.arange(109)*300)and abs(sum(rates)-nominal)<=max(1.,nominal)*1e-6)
 check('protected_sources_unchanged',all(sha(ROOT/f)==h for f,h in json.loads((P/'protected_sources.json').read_text()).items()))
 check('no_training_no_population_no_sealed_tests',all(protocol[k]is False for k in ['training','sealed_tests_opened','people_counted','traffic_flow_or_queue']))
 dump(R/'plan_validation.json',dict(all_passed=True,checks=checks,count=len(checks),max_new_hydraulic_runs=8,new_hydraulic_runs_started=0));print('CROSS-EVENT INPUT CHECKS',len(checks),'PASSED')
if __name__=='__main__':main()
