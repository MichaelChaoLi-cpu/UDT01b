"""Delivery checks of diagnosis provenance, arithmetic and document boundaries."""
from pathlib import Path
from html.parser import HTMLParser
import sys,json,csv,re
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_three_ward_unet_exact_pilot import read,write,sha
NAME='three_ward_d_false_positive_diagnosis_v1';OUT=ROOT/'data/processed'/NAME;REPORT=ROOT/'data/exp/estimation'/NAME

def section(text,n):return re.search(r'^## '+str(n)+r'\..*?(?=^## \d+\.|\Z)',text,re.M|re.S).group(0)
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):
  for k,v in attrs:
   if k in ('src','href'):self.links.append(v)
def main():
 if (OUT/'freeze.json').exists():raise FileExistsError('Already frozen')
 s=read(REPORT/'summary.json');f=read(REPORT/'findings.json');p=read(OUT/'diagnostic_protocol.json');checks=[]
 def check(n,v):
  if not bool(v):raise AssertionError(n)
  checks.append({'check':n,'pass':True})
 check('analysis388checks_pass',read(REPORT/'validation.json')['all_checks_passed'] and len(read(REPORT/'validation.json')['checks'])==388)
 check('no_fits_inference_updates_or_sealed_access',s['new_fit_count']==s['optimizer_updates']==s['inference_runs']==0 and not s['sealed_opened'])
 check('original12_4_partitions',sum(v=='train' for v in p['splits'].values())==12 and sum(v=='validation' for v in p['splits'].values())==4)
 check('451protected_sources_still_unchanged',len(p['source_files_sha256'])==451 and all(sha(ROOT/k)==v for k,v in p['source_files_sha256'].items()))
 check('diagnostic_implementation_bound',p['new_script_sha256']==sha(ROOT/'src/analyses/diagnose_three_ward_d_false_positives.py'))
 rr=list(csv.DictReader((REPORT/'selected_event_diagnostics.csv').open()));st=list(csv.DictReader((REPORT/'spatial_strata_by_event.csv').open()))
 for split in ['train','validation']:
  ev=[r for r in rr if r['split']==split];macro=s['primary_metrics'][split]
  for k in ['false_km2_0.1','balanced_extent_error','wet_nonwet_rank_auc','precision_0.1']:
   check(split+'_'+k+'_macro_matches_event_csv',abs(np.mean([float(r[k]) for r in ev])-macro[k])<1e-12)
  for fam in {r['family'] for r in st}:
   # Every physical-family FP total must retain all original FP mass.
   total=sum(float(r['false_km2']) for r in st if r['split']==split and r['family']==fam)/len(ev)
   check(split+'_'+fam+'_area_partition',abs(total-macro['false_km2_0.1'])<1e-10)
 val=[r for r in st if r['split']=='validation'];n=4
 near=sum(float(r['fraction_of_event_false_area']) for r in val if r['family']=='distance_to_reference_wet_m' and r['stratum'] in ['(0,20]','(20,50]'])/n
 dry=sum(float(r['fraction_of_event_false_area']) for r in val if r['family']=='reference_nonwet_depth_m' and r['stratum']=='(-inf,0.01]')/n
 check('reported_proximity_share_recomputed',abs(near-f['validation_false_share_within50m_of_reference_wet'])<1e-12)
 check('reported_near_dry_share_recomputed',abs(dry-f['validation_false_share_reference_at_most1cm'])<1e-12)
 for epoch in range(1,41):
  old=read(ROOT/f'data/exp/estimation/three_ward_unet_controls_v2/D_huber_lower_lr/epoch_{epoch:03}.jsonl')
  check(f'epoch{epoch}_original_guard_still_failed',not old['baseline_readiness']['all_parents_pass'])
 before=(REPORT/'previous-AnaSOP.md').read_text();after=(ROOT/'docs/AnaSOP.md').read_text()
 check('anasop_1_4_8_preserved',all(section(before,n)==section(after,n) for n in [1,2,3,4,8]))
 check('anasop_only5_7_updated',all(section(before,n)!=section(after,n) for n in [5,6,7]))
 check('anasop_english_ascii_and_snapshot',after.isascii() and after==(REPORT/'updated-AnaSOP.md').read_text())
 check('published_plans_remain_pending',section(after,8).count('| pending |')==8)
 parser=Links();parser.feed((REPORT/'review.html').read_text())
 check('all_report_links_local_and_valid',all('://' not in link and (REPORT/link).is_file() for link in parser.links))
 for name in ['fit_and_selection_trajectories.png','whole_ward_recurrence.png','false_area_strata.png']:
  check(name+'_valid_png',(REPORT/name).read_bytes()[:8]==b'\x89PNG\r\n\x1a\n')
 write(REPORT/'delivery_validation.json',{'all_checks_passed':True,'checks':checks,'analysis_checks':388,'delivery_checks':len(checks),'source_files_checked':451,'new_fit_count':0,'optimizer_updates':0,'sealed_opened':False,'figures_visually_reviewed':True})
 scripts=[ROOT/'src/analyses'/name for name in ['diagnose_three_ward_d_false_positives.py','review_three_ward_d_false_positives.py','document_three_ward_d_false_positives.py','validate_three_ward_d_false_positives.py']]
 files=scripts+[x for folder in [OUT,REPORT] for x in folder.rglob('*') if x.is_file()]
 write(OUT/'freeze.json',{'status':'complete','decision':'MILI-D-20261016-006','files_sha256':{str(x.relative_to(ROOT)):sha(x) for x in files},'analysis_checks':388,'delivery_checks':len(checks),'new_fit_count':0,'optimizer_updates':0,'sealed_opened':False,'original_selected_checkpoint':36,'no_new_checkpoint_promoted':True,'figures_visually_reviewed':True,'publication_outputs_completed':False})
 print(json.dumps({'all_checks_passed':True,'analysis_checks':388,'delivery_checks':len(checks),'files_frozen':len(files),'new_fit_count':0,'sealed_opened':False}))
if __name__=='__main__':main()
