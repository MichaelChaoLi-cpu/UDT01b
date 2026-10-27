"""Estimation-framework-planning responsibility: completed Sections5-7 evidence."""
from pathlib import Path
import re,sys,json,csv
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_three_ward_unet_exact_pilot import read,write,sha
REPORT=ROOT/'data/exp/estimation/three_ward_unet_controls_v2'
def sec(t,n):return re.search(r'^## '+str(n)+r'\..*?(?=^## \d+\.|\Z)',t,re.M|re.S).group(0)
def append(t,n,a):
 old=sec(t,n);return t.replace(old,old.rstrip()+'\n\n'+a.strip()+'\n\n',1)
def main():
 assert read(REPORT/'delivery_validation.json')['all_checks_passed'];s=read(REPORT/'summary.json');op=list(csv.DictReader((REPORT/'optimization_summary.csv').open()));contrasts=list(csv.DictReader((REPORT/'paired_contrast_macro.csv').open()))
 path=ROOT/'docs/AnaSOP.md';before=path.read_text();text=before
 if '### Completed four-arm loss and learning-rate comparison' in text:raise FileExistsError('Comparison already recorded')
 (REPORT/'previous-AnaSOP.md').write_text(before)
 text=append(text,5,'''### Completed bounded comparison and interpretation

The user subsequently authorized execution of the prepared two-by-two comparison. All four arms are complete under the accepted fixed settings: original MSE or the stratified robust-loss package, crossed with constant learning rates 0.001 and 0.0003. Each uses forty epochs and 480 real optimizer updates, for 1,920 updates in total. Preserve the same stored untrained initialization, original twelve training and four validation parent families, forty-epoch exposure order, original quality masks and training-fitted normalization. No extra fit, adaptive restart, hydraulic simulation or sealed-response access is added.

The treatment comparison applies to the whole development pipeline, including the frozen common checkpoint ranking. Selected checkpoints can have different epoch counts; therefore also report matched final-epoch contrasts at the identical forty-epoch budget. Each contrast pairs the same validation parent and support. One initialization and four already exposed validation events do not support population inference or real-world flood accuracy. The loss treatment combines depth stratification with Huber residuals; their separate effects remain unidentified. These four loss-by-rate arms are distinct from the still-pending raster/graph by exact/interval architecture-supervision comparisons.''')
 text=append(text,6,'''### Executed settings and optimization evidence contract

The four executed arms retain the prepared exact-only definitions and selection rules. Nonwet, ordinary-wet and deep strata use mixture weights 0.50, 0.40 and 0.10, with twice-Huber transition at residual 0.10 m for the candidate loss. Original MSE is the reference loss. Reference depth and mask boundaries are unchanged. A lower learning-rate arm changes only its constant rate, not the optimizer, weight decay, clipping, architecture, initialization or input encoding. No early stopping is applied within the fixed forty-epoch budget.

Checkpoint ranking uses the original paired three-ward support and parent-macro balanced extent error, followed by wet MAE, overall MAE and earlier epoch with the declared rounding. Readiness remains the separate all-parent comparison to training-only retrieval and always-dry. Report a failed guard explicitly; a best diagnostic ranking is not automatic model admission. Full target-exact and geographic strata are retained alongside the primary paired comparison.

Real per-step logs now include actual optimizer increments, parameter changes and pre/post-clipping gradients. Every completed epoch retains the fixed-model training objective and all training/validation score frames, distinct from the online mean loss during sequential updates. Module update magnitudes, clipping counts, fixed-loss trajectories and last-ten-epoch variation are optimization diagnostics. Raw objective or gradient magnitudes from different losses are not interchangeable quality scores. A forty-epoch endpoint or a small gradient does not establish convergence. Instrumented training, fixed-field evaluation and checkpoint I/O costs are recorded separately and do not establish hydraulic or end-to-end acceleration.''')
 header='''### Completed four-arm loss and learning-rate comparison

The frozen exact-label comparison is complete. The table reports four-parent macro metrics on identical paired three-ward support at the unchanged strict 0.10 m wet threshold. Depth MAE is in metres and false/missed areas are in square kilometres. The checkpoint epoch is chosen separately for each arm using the predeclared development ranking.

| arm | supervision | learning rate | selected epoch | balanced extent error | overall MAE | wet MAE | deep MAE | wet IoU | wet recall | false wet area | missed wet area | all-parent baseline guard |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
'''
 for r in s['arms']:
  def v(k):return f"{r[k]:.8f}" if r.get(k) is not None else 'undefined'
  header+='| '+r['arm'][0]+' | '+('Original MSE' if r['loss']=='mse' else 'Stratified robust')+' | '+str(r['learning_rate'])+' | '+str(r['selected_epoch'])+' | '+' | '.join(v(k) for k in ['balanced_extent_error','mae_m','wet_mae_m','deep_mae_m','iou_0.1','recall_0.1','false_km2_0.1','missed_km2_0.1'])+' | '+str(r['all_four_parents_pass_baseline_guard']).lower()+' |\n'
 b=s['baselines_on_same_primary_support'];header+=f"\nThe original training-event retrieval baseline on the same primary support gives balanced extent error {b['retrieval']['balanced_extent_error']:.8f}, overall MAE {b['retrieval']['mae_m']:.8f} m and wet IoU {b['retrieval']['iou_0.1']:.8f}. Always-dry gives balanced extent error {b['dry']['balanced_extent_error']:.8f} and overall MAE {b['dry']['mae_m']:.8f} m.\n"
 header+='\nPaired left-minus-right differences use each arm\'s selected checkpoint. Negative depth/extent error and false area are improvements; positive IoU and recall are improvements.\n\n| comparison | contrast | delta balanced error | delta wet MAE | delta deep MAE | delta IoU | delta recall | delta false area |\n|---|---|---|---|---|---|---|---|\n'
 for r in contrasts:
  if r['scope']=='three_wards' and r['support']=='paired_exact':
   header+='| '+{'loss_original_lr':'Loss at original learning rate','loss_lower_lr':'Loss at lower learning rate','lr_mse':'Learning rate with MSE','lr_stratified':'Learning rate with stratified loss'}[r['contrast']]+' | '+r['left']+' minus '+r['right']+' | '+' | '.join(f"{float(r[k]):.8f}" if r[k] else 'undefined' for k in ['delta_balanced_extent_error','delta_wet_mae_m','delta_deep_mae_m','delta_iou_0.1','delta_recall_0.1','delta_false_km2_0.1'])+' |\n'
 f40={r['contrast']:r for r in s['same_final_epoch40_contrasts']}
 header+=f"\nAt the same forty-epoch budget, the stratified package increases validation wet recall relative to MSE by {f40['loss_original_lr']['delta_recall_0.1']*100:.4f} percentage points at the original learning rate and {f40['loss_lower_lr']['delta_recall_0.1']*100:.4f} points at the lower rate, while false area increases by {f40['loss_original_lr']['delta_false_km2_0.1']:.6f} and {f40['loss_lower_lr']['delta_false_km2_0.1']:.6f} square kilometres. Selected-checkpoint contrasts additionally include selection timing: the original-rate robust arm selects epoch one and the lower-rate robust arm selects epoch thirty-six. At matched epoch forty, their false-area difference is only {f40['lr_stratified']['delta_false_km2_0.1']:.6f} square kilometres. Do not attribute the much larger selected-checkpoint difference solely to update size.\n"
 contribution=s['realized_training_loss_contributions'];deep={r['arm'][0]:r for r in contribution if r['point']=='selected_checkpoint' and r['stratum']=='deep'}
 header+=f"\nThe deep mixture coefficient of 0.10 does not cap deep loss contribution at ten percent: deep support contributes {deep['B']['fraction_of_mean_objective']*100:.4f}% and {deep['D']['fraction_of_mean_objective']*100:.4f}% of the selected robust arms' mean training objectives. These are loss-value shares, not gradient attribution. Preserve the separate deep error and ordinary-wet/false-area metrics when interpreting the package.\n"
 header+='\nOptimization summaries retain the distinction between relative changes within each loss and quality comparisons under common evaluation metrics.\n\n| arm | clipping count out of 480 | gradient p95 | gradient maximum | final/first fixed objective | final/minimum fixed objective | last-ten fixed-objective CV | fixed-objective increases out of 39 |\n|---|---|---|---|---|---|---|---|\n'
 for r in op:header+='| '+r['arm'][0]+' | '+r['clipping_count']+' | '+' | '.join(f"{float(r[k]):.6f}" for k in ['gradient_p95','gradient_max','fixed_objective_final_over_epoch1','fixed_objective_final_over_minimum','fixed_objective_last10_cv'])+' | '+r['fixed_objective_increase_count']+' |\n'
 passing=[r['arm'][0] for r in s['arms'] if r['all_four_parents_pass_baseline_guard']]
 header+='\n'+('Arms '+', '.join(passing)+' satisfy the prepared all-parent development baseline guard. This remains model-conditional development evidence, not an independent or observational certification.' if passing else 'No arm satisfies the complete all-parent development baseline guard. Retain the results as diagnostic models and do not describe the selected development best as a reliable surrogate.')+'\n'
 header+='''

| completed analytical step | variables used | specification | planned evidence connection | support status |
|---|---|---|---|---|
| Execute matched loss and learning-rate arms | Ground Elevation, Ground Area, Building Occupancy, Road Connectivity, Rainfall Rate, River Stage, Peak Mean Depth | Frozen four-arm exact-only fit with shared initialization and exposure | Stress and ablation response; Held-out metric and runtime summary | Completed development comparison; sealed and architecture-supervision evidence remain pending. |
| Compare wet misses, false area and depth | Peak Mean Depth, Peak Wet Area, Ground Area | Same-event paired support and fixed checkpoint ranking; matched final-epoch companion | Held-out surrogate performance | Quantified conditional agreement and tradeoffs; not an independent test or dynamic-route validation. |
| Audit optimization and computation | Peak Mean Depth, Ground Area, End-to-End Runtime | Real optimizer clocks, gradients, parameter changes and fixed-model scores | Held-out metric and runtime summary | Actual bounded-run diagnostics available; convergence and end-to-end speedup unestablished. |
| Preserve subsequent stages | Measure Configuration, Mitigation Benefit, Boundary Exchange | Pending interval/graph study and physically verified measure comparisons | Physically checked measure comparison | No action library, ranking or intervention claim; measures remain the fifth milestone. |

Source data, quality masks, partitions, normalization and the preparation and historical deliveries remain frozen. The four validation parents were used for prior diagnosis, rule design and current selection. Report every event and each ward rather than treating pixels or epochs as independent evidence. Do not use these results to open sealed tests, alter reference admission, add another seed or enlarge the scenario library automatically. The prepared full-study figures and tables in Section 8 remain pending; current exploratory outputs do not complete their sealed evaluation.
'''
 text=append(text,7,header);assert all(sec(before,n)==sec(text,n) for n in [1,2,3,4,8]);assert text.isascii();path.write_text(text)
 write(REPORT/'anasop_update.json',{'decision':'MILI-D-20261016-005','responsible_skill':'estimation-framework-planning','changed_sections':[5,6,7],'unchanged_sections':[1,2,3,4,8],'before_sha256':sha(REPORT/'previous-AnaSOP.md'),'after_sha256':sha(path),'english_ascii':True,'actual_fits':4,'optimizer_updates':1920,'sealed_opened':False,'publication_outputs_completed':False});print(json.dumps({'updated':[5,6,7],'ascii':True,'actual_fits':4}))
if __name__=='__main__':main()
