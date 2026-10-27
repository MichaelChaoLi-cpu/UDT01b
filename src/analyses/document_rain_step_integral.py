"""Record qualified evidence through the read data-preprocessing and framework responsibilities."""
from pathlib import Path
import sys,json,re
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_rain_step_integral import REPORT,CASES
from prepare_sfincs_cpu_build import sha,dump

def main():
 report=ROOT/REPORT;s=json.loads((report/'summary.json').read_text());audit=json.loads((report/'audit.json').read_text());rain=json.loads((report/'rain_integral_audit.json').read_text());ad=json.loads((report/'reference_admission_assessment.json').read_text())
 assert audit['all_checks_passed']and rain['all_checks_passed']and all(all(ad[c]['required_gates'].values())and ad[c]['preview']['technical_reference']for c in CASES),'Document the actual hold instead of running the passing-evidence update'
 p=ROOT/'docs/AnaSOP.md';backup=report/'previous-AnaSOP.md'
 if not backup.exists():backup.write_bytes(p.read_bytes())
 before=backup.read_text();parts=re.split(r'(?=^## [1-8]\. )',before,flags=re.M)
 assert all(any(x.startswith('## '+str(k)+'.')for x in parts)for k in [1,3,4,8])
 r=s['cases']['rain60_raised']['rain'];states='dry control: partial numerical reference; rainfall control: partial numerical reference'
 additions={4:'For the rainfall integration correction, preserve the original Rainfall Rate curves, roof-transfer weights, Ground Area, parent lineage and split. Integrate the original piecewise-linear curve over each actual hydraulic step; use its mean rate consistently in donor availability, continuity, applied-step accounting and cumulative precipitation. Retain instantaneous-rate output semantics and separate native floating-point source rounding from time-quadrature error. Reconstruct the same common-ground peak intervals and apply the unchanged barrier, negative-volume and interval rules. No diagnostic becomes a predictor or a final variable.',6:f'The two existing adaptive open controls now pass the unchanged numerical event gates with actual-step rainfall integration. In the rainy control the original allocation gives {r["original_curve_integral_m3"]:.6f} cubic metres, and the native final applied difference is {r["final_error_m3"]:.6f} cubic metres, compared with {r["previous_endpoint_final_error_m3"]:.6f} before correction. Preserve the original final rainfall tolerance, endpoint and prefix balance gates, strict positivity checks and zero exchange on closed receiving faces. The dry control retains identical actual-step accounting and common-ground labels. Technical previews are {states}; wider intervals and original quarantines remain excluded from primary point regression. Refined responses remain diagnostic and do not replace the regular-grid primary-label policy. These are model-conditional numerical references, without observational accuracy, dynamic-path validity or formal grid-convergence certification.',7:'Complete only the two existing adaptive dry and rainfall control replays for the rain-integral repair, preserving the original scientific input bytes, source versions, family and partitions. Keep the original uniform-grid and previous adaptive results frozen. Recheck actual clocks, original-curve rainfall totals, same-step budgets, strict positivity, native boundary exchange and common-ground label bounds before the unchanged event-policy preview. The default reference registry and loader remain closed, and the prospective batch and training remain unexecuted. Use the Reference uncertainty and Scenario coverage and split audit outputs to distinguish successful numerical control qualification from response-library construction and later observational validation. Active measures remain the fifth milestone.'}
 for i,part in enumerate(parts):
  if part.startswith('## '):
   k=int(re.match(r'## (\d)',part).group(1))
   if k in additions:parts[i]=part.rstrip()+'\n\n'+additions[k]+'\n\n'
 after=''.join(parts);assert after.isascii()
 oldparts=re.split(r'(?=^## [1-8]\. )',before,flags=re.M);changed={int(re.match(r'## (\d)',a).group(1))for a,b in zip(oldparts,parts)if a!=b};assert changed=={4,6,7}
 p.write_text(after);dump(report/'anasop-update.json',{'sections_changed':[4,6,7],'skills':['data-preprocessing','estimation-framework-planning'],'human_record':'MILI-D-20261015-016','new_final_variables':0,'new_formulas':0,'prior_sha256':sha(backup),'current_sha256':sha(p),'other_sections_unchanged':True,'Section8_output_mapping':'Reference uncertainty; Scenario coverage and split audit','Section3_refinement_needed':False,'Section4_candidate_status_preserved':True})
 print('Updated AnaSOP Sections4,6,7 through responsible skills; English/ASCII verified; no final variables or formulas added.')
if __name__=='__main__':main()
