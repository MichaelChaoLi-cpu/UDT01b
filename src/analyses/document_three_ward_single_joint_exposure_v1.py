"""Update AnaSOP5-7 from completed, independently validated actual fitting results."""
from pathlib import Path
import sys,re
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_three_ward_unet_exact_pilot import read,write,sha
NAME='three_ward_single_joint_exposure_fit_v1';REPORT=ROOT/'data/exp/estimation'/NAME

def sections(t):return {int(n):v for n,v in re.findall(r'^## ([1-8])\. (.*?)(?=^## [1-8]\. |\Z)',t,re.M|re.S)}
def main():
 if (REPORT/'anasop_update.json').exists():raise FileExistsError('Completed delivery is immutable')
 assert read(REPORT/'completion.json')['status']=='complete' and read(REPORT/'delivery_validation.json')['all_checks_passed']
 s=read(REPORT/'summary.json');assert s['actual_new_fits']==2 and s['actual_new_updates']==480
 sop=ROOT/'docs/AnaSOP.md';before=sop.read_text();old=sections(before);base=sections((REPORT/'previous-AnaSOP.md').read_text())
 s5='''### Completed exposure-matched fitting controls

The separately authorized two individual-parent fits are complete, each using 240 actual optimizer updates from the same original untrained parameters. The existing joint fit is reused at 480 updates, matching 240 exposures of each parent. Original twelve-training/four-validation roles, exact quality masks, normalization, curves and earlier deliveries are preserved. No hydraulic response, new parent, validation scoring or sealed/challenge access is added. This comparison provides known-field recipe diagnostics rather than independent emulation or physical validation.

'''
 s6='''### Implemented individual-versus-joint comparison

The individual arms retain the original full-field conditional architecture, fixed training normalization, robust stratified loss, learning rate, seed and exact support. Each fresh optimizer completes the fixed 240-update budget, without another seed, scheduler, changed loss or favorable-checkpoint selection. Fifty new fixed-exposure evaluations retain initialization and every ten exposures for each arm. All primary comparisons use the actual fixed endpoints and identical target-specific original exact eligible Ground Area; secondary curves match own-parent exposure rather than optimizer index. Depth-stratum errors and false/missed Peak Wet Area remain separate, with actual gradients, parameter changes and optimizer clocks retained. Frozen endpoint inference is independently replayed without parameter updates.

'''
 rows=[]
 for r in s['whole_three_ward_endpoints']:
  rows.append('| '+r['case_id']+' | '+('Individual' if r['method']=='single' else 'Joint reference')+' | '+f"{r['iou_0.1']:.6f} | {r['recall_0.1']:.6f} | {r['precision_0.1']:.6f} | {r['wet_mae_m']:.6f} | {r['false_km2_0.1']:.6f} | {r['missed_km2_0.1']:.6f} |")
 s7='''### Completed individual-parent fitting results

Exactly two new fits complete 240 updates each, adding 480 updates and no new parents. The joint reference is not retrained. Every event has 240 own-parent exposures at the compared endpoints. The following results use the whole-three-ward original own-exact support and Ground Area weighting.

| known training parent | fitting regime | wet overlap | recall | precision | wet mean absolute error in m | false area in square kilometres | missed area in square kilometres |
|---|---|---|---|---|---|---|---|
'''+ '\n'.join(rows)+'\n\n'
 passn=s['singles_with_demonstrated_endpoint_fit']
 s7+=f'The number of individual arms demonstrating consecutive full parent/ward fitting-gate passage at the endpoint is {passn} out of two. First consecutive confirmations, endpoint gate failures and all twenty-five fixed evaluations per arm are reported separately; the endpoint is not replaced by an earlier favorable score.\n\n'
 if passn==0:
  s7+='Neither individual arm demonstrates adequate known-field fit within this recipe and budget. Modest metric improvements cannot support an explanation based solely on interference between the two parents or a lack of training-event diversity. Individual-field spatial placement and depth fitting remain inadequate. This is not proof of intrinsic network incapacity, eventual nonconvergence or a particular representation/loss defect. The prespecified development exit is to stop automatic tweaks to this recipe and make an explicit representation, supervision or optimization design decision before further library investment; no next fit is authorized by this result.\n\n'
 elif passn==1:
  s7+='Only one individual arm meets the complete endpoint criterion. The limitation is parent-dependent under the current recipe and budget. The failing known field requires whole-area representation/supervision assessment before attributing all joint error to conditional discrimination or update interference. No automatic further fit or expansion follows.\n\n'
 else:
  s7+='Both known fields meet the individual endpoint criteria, whereas the original joint fit does not. This supports a difficulty associated with the shared/interleaved regime for this pair at equal target-parent exposure. It does not isolate pure gradient conflict or condition-encoder failure. A subsequent shared-conditioning/optimization comparison remains a separate research decision.\n\n'
 s7+='''Matching parent exposures does not match global optimizer steps, adaptive moments, weight-decay history or intervening updates. The two individual networks also have twice the aggregate parameter count of one joint network and each receives a constant event condition. They are diagnostic specialists, not evidence of a superior general-purpose deployment alternative. A single initialization and two known synthetic events cannot establish independent generalization, event sufficiency, terrain accuracy or observed inundation skill.

| completed step | existing variables | evidence connection | claim status |
|---|---|---|---|
| Verify shared untrained state and original support | Ground Elevation, Building Occupancy, Road Connectivity, Rainfall Rate, River Stage, Ground Area | Reference uncertainty; Scenario coverage and split audit | Source and comparison invariants verified; no new sample. |
| Complete two finite individual fits | Peak Mean Depth, Ground Area | Stress and ablation response | Two actual fits, 480 new updates and fixed stopping completed. |
| Compare at equal parent exposure | Peak Mean Depth, Peak Wet Area, Ground Area | Held-out surrogate performance | Known-training evidence only; independent-performance status remains pending. |
| Audit update records and replay endpoint outputs | Peak Mean Depth, Ground Area | Held-out metric and runtime summary | Actual update and frozen-output bindings checked; no hydraulic speedup claim. |

The underlying reference uses ground geometry derived from bare-earth elevation and city-model terrain where needed, building/road surface constraints and prescribed receiving-water levels. This is depth-averaged surface-flow modeling informed by three-dimensional urban geometry, not a full three-dimensional flow or resolved pipe/river-bed simulation. The learned model approximates the resulting four-hour Peak Mean Depth field; evolving drainage paths still require the physical time-series and flux diagnostics.

Sections 3-4 require no sample, input or variable changes for this comparison. Section 8 remains a coherent planned evidence map for Section 1, but independent unseen-event, observational and active-measure claims remain unsupported by these known-field diagnostics. All publication statuses remain pending. No new equation symbols, hidden test use or automatic sample expansion are introduced.

'''
 assert (s5+s6+s7).isascii()
 after=before.replace('## 6. Main Estimation Framework',s5+'## 6. Main Estimation Framework',1).replace('## 7. Analytical Workflow',s6+'## 7. Analytical Workflow',1).replace('## 8. Figure and Table Plan',s7+'## 8. Figure and Table Plan',1);new=sections(after)
 assert all(old[k]==new[k]==base[k] for k in [1,2,3,4,8])
 (REPORT/'pre-completion-AnaSOP.md').write_text(before);sop.write_text(after);(REPORT/'updated-AnaSOP.md').write_text(after)
 write(REPORT/'anasop_update.json',{'skill':'.codex/skills/estimation-framework-planning/SKILL.md','decision':'MILI-D-20261016-012','sections_changed':[5,6,7],'before_registration_sha256':sha(REPORT/'previous-AnaSOP.md'),'registered_sha256':sha(REPORT/'registered-AnaSOP.md'),'before_completion_sha256':sha(REPORT/'pre-completion-AnaSOP.md'),'after_sha256':sha(sop),'sections1to4_and8_preserved':True,'new_text_english_ascii':True,'data_and_variables_unchanged':True,'no_new_formula_symbols':True,'publication_statuses_unchanged':True,'based_on_actual_validated_completion':True})
 print('AnaSOP5-7 updated from completed actual results only.')
if __name__=='__main__':main()
