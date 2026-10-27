#!/usr/bin/env python3
"""Audit isolated pump files,shared caps and reference arithmetic without solver runs."""
from pathlib import Path
from html.parser import HTMLParser
import hashlib,json,sys,importlib.util
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];TAG='three_ward_shared_receiver_allocation_v1';P=ROOT/'data/processed'/TAG;R=ROOT/'data/exp/research-design'/TAG

def sha(f):
 h=hashlib.sha256()
 with Path(f).open('rb') as s:
  for b in iter(lambda:s.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 manifest=json.loads((P/'manifest.json').read_text());assert all(sha(ROOT/f)==h for f,h in manifest['input_sha256'].items())
 assert sha(ROOT/'src/analyses/allocate_shared_receiver_portfolios_v1.py')==manifest['script_sha256']
 assert all(sha(ROOT/f)==h for f,h in manifest['upstream_physical_source_bindings'].items())
 for e in manifest['outputs']:
  f=ROOT/e['path'];assert sha(f)==e['sha256'];assert pd.read_parquet(f).shape==(e['rows'],e['columns'])
 summary=pd.read_parquet(P/'portfolio_summary_preprocessed.parquet').set_index('Portfolio');selected=pd.read_parquet(P/'selected_intakes_preprocessed.parquet');pools=pd.read_parquet(P/'shared_receiver_reservations_preprocessed.parquet')
 with np.load(ROOT/'data/processed/three_ward_barrier_drainage_v1/measures/spatial_contract.npz') as z:kcs=z['native_kcs']
 files=0
 for e in manifest['pump_inputs']:
  f=ROOT/e['path'];assert sha(f)==e['sha256'];ident=f.parent.name;a=np.loadtxt(f,skiprows=1,ndmin=2);count=int(f.read_text().splitlines()[0]);assert len(a)==count==int(summary.loc[ident,'Activated Model Intakes'])
  assert (a[:,:2]>=1).all() and (a[:,:2]<=len(kcs)).all();assert np.equal(a[:,:2],np.floor(a[:,:2])).all();assert (kcs[a[:,0].astype(int)-1]==1).all() and (kcs[a[:,1].astype(int)-1]==2).all();assert len(np.unique(a[:,0]))==count;assert (a[:,2]>0).all() and (a[:,2]<=.5).all();assert np.all(a[:,3]==.05) and np.all(a[:,4]==5.)
  s=selected[selected.Portfolio==ident].sort_values('Grid Index');assert np.array_equal(a[:,0],s['Source Native Cell One Based']);assert np.array_equal(a[:,1],s['Receiver Native Cell One Based']);assert np.array_equal(a[:,2],s['Reserved Rate'])
  for pool,g in s.groupby('Receiving Pool'):
   cap=pools[(pools.Portfolio==ident)&(pools['Receiving Pool']==pool)]['Scenario Shared Allowance'].iloc[0]
   if pd.notna(cap):assert g['Reserved Rate'].sum()<=cap+1e-11
  files+=1
 frame=pd.read_parquet(P/'snapshot_allocation_trajectories_preprocessed.parquet');assert len(frame)==2180
 assert np.allclose(frame['Snapshot ReferenceRequested Rate'] if 'Snapshot ReferenceRequested Rate' in frame else frame['Snapshot Reference Requested Rate'],frame['Snapshot Reference Assigned Rate']+frame['Snapshot Reference Unassigned Rate'],rtol=0,atol=1e-10)
 assert np.allclose(frame['Snapshot Reference Unassigned Rate'],frame['No Robust Named Link Request']+frame['Unit Ceiling Gap']+frame['Reservation Gap'],rtol=0,atol=1e-10)
 prior=pd.read_parquet(ROOT/'data/processed/three_ward_protection_service_demand_plan_v1/policy_control_storage_trajectories_preprocessed.parquet');prior=prior[prior.Policy=='Combined'][['Case','Frame','Three Ward Control Excess Storage']]
 check=frame.merge(prior,on=['Case','Frame'],validate='many_to_one');times=check.Portfolio.map(summary['Reference Clearance Horizon']);assert np.allclose(check['Snapshot Reference Requested Rate']*times,check['Three Ward Control Excess Storage'],rtol=0,atol=1e-6)
 assert frame['Actual Pumped Volume'].isna().all() and frame['Actual Unmet Drainage Requirement'].isna().all();assert summary['Machine Count'].isna().all()
 # A real regression guard:two distinct model groups must not receive two budgets.
 sys.dont_write_bytecode=True;spec=importlib.util.spec_from_file_location('allocation',ROOT/'src/analyses/allocate_shared_receiver_portfolios_v1.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 q,_,_=mod.allocate(np.array([.4,.4]),np.eye(2,dtype=bool),np.ones((2,2)),np.array([79,111]),np.array([1,2]),.5);assert np.isclose(q.sum(),.5)
 q,_,_=mod.allocate(np.array([1.,1.]),np.eye(2,dtype=bool),np.ones((2,2)),np.array([79,111]),np.array([1,2]),0.);assert np.all(q==0)
 class Links(HTMLParser):
  def __init__(self):super().__init__();self.count=0
  def handle_starttag(self,t,attrs):
   if t=='a':
    for k,v in attrs:
     if k=='href':assert (R/v).resolve().is_file(),v;self.count+=1
 parser=Links();parser.feed((R/'review.html').read_text());assert json.loads((R/'validation.json').read_text())['passed']
 result={'passed':True,'pump_files_checked':files,'native_endpoint_masks_and_one_based_ids':True,'fixed_receiver_sums_within_scenario_caps':True,'Furukawa_split_group_budget_regression_passed':True,'blocked_allocator_zero':True,'same_frame_reference_volume_matches_prior_screen':len(check),'all_upstream_physical_inputs_unchanged':True,'actual_unknowns_and_machine_count_not_imputed':True,'valid_html_links':parser.count,'AnaSOP_sha256':sha(ROOT/'docs/AnaSOP.md'),'new_hydraulic_runs':0,'audit_script_sha256':sha(Path(__file__))}
 (R/'delivery_qa.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
