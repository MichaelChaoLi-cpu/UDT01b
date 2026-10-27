from pathlib import Path
import sys,json,hashlib
from html.parser import HTMLParser
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/analyses'))
import demand_protection_feedback_v1 as m
m.check_sources();R=m.R;P=m.P
vals=[json.loads(f.read_text())for f in R.glob('*_fit_validation.json')];paths=[json.loads(f.read_text())for f in R.glob('*_path_validation.json')]
assert len(vals)==24 and all(v['passed']for v in vals)
assert len(paths)==10 and all(v['passed']for v in paths)
assert sum(v['seconds']for v in vals)<=2400 and sum(v['seconds']for v in paths)<=1800
assert max(v['seconds']for v in vals)<=120 and max(v['seconds']for v in paths)<=180
assert len(set(v['code_sha256']for v in vals+paths))==1
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
assert vals[0]['code_sha256']==sha(ROOT/'src/analyses/demand_protection_feedback_v1.py')
class Parser(HTMLParser):
 def __init__(self):super().__init__();self.refs=[]
 def handle_starttag(self,tag,attrs):
  for k,v in attrs:
   if k in ['src','href']and not v.startswith(('http:','https:','data:','#')):self.refs.append(v)
x=Parser();x.feed((R/'review.html').read_text());assert all((R/v).exists()for v in x.refs)
assert json.loads((R/'browser_validation.json').read_text())['passed']
assert json.loads((R/'continuity_validation.json').read_text())['passed']
checks={'24scenario_gates':True,'10path_sweeps':True,'same_solver_source_version':True,'preregistered_compute_budgets':True,'local_links_images_mobile':True,'original_A_flow_continuity':True}
manifest={'passed':True,'checks':checks,'actual_path_active_seconds':sum(v['seconds']for v in paths),'actual_fit_active_seconds':sum(v['seconds']for v in vals),'maximum_restricted_gap':max(v['gap']for v in vals),'source_code_sha256':{str(f.relative_to(ROOT)):sha(f)for f in [ROOT/'src/analyses/demand_protection_feedback_v1.py',ROOT/'src/analyses/finalize_demand_feedback_v1.py',ROOT/'src/analyses/validate_demand_feedback_browser_v1.cjs',Path(__file__)]},'reproduction':['demand_protection_feedback_v1.py prepare','demand_protection_feedback_v1.py paths','demand_protection_feedback_v1.py fit','demand_protection_feedback_v1.py report','finalize_demand_feedback_v1.py','validate_demand_feedback_browser_v1.cjs','validate_feedback_delivery_v1.py'],'interpretation':'No observed traffic validation,no dynamic queues,no new hydraulics,no people impacts.'}
(R/'delivery_validation.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in manifest.items()if k not in ['source_code_sha256','reproduction']},indent=2))
