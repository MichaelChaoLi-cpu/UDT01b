"""Skill-routed estimation-framework-planning update of AnaSOP Sections 5-7."""
from pathlib import Path
import re,sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_three_ward_unet_exact_pilot import read,write,sha
REPORT=ROOT/'data/exp/estimation/three_ward_unet_control_plan_v2'
def section(t,n):return re.search(r'^## '+str(n)+r'\..*?(?=^## \d+\.|\Z)',t,re.M|re.S).group(0)
def append(t,n,a):
 old=section(t,n);return t.replace(old,old.rstrip()+'\n\n'+a.strip()+'\n\n',1)
def main():
 path=ROOT/'docs/AnaSOP.md';before=path.read_text();text=before
 if '### Prepared loss and optimization comparison' in text:raise FileExistsError('Already documented')
 assert read(REPORT/'validation.json')['all_checks_passed'];(REPORT/'previous-AnaSOP.md').write_text(before)
 text=append(text,5,'''### Prepared bounded loss and learning-rate comparison

The next preparation task implements, but does not execute, a finite two-by-two comparison on the same twelve training and four validation parent events. Cross the original MSE with a stratified robust-loss package, and the original learning rate 0.001 with a lower fixed rate 0.0003. All four proposed arms share one saved untrained initialization, the original conditional U-Net, training-fitted normalization, exact-label support, a fixed forty-epoch parent order and a maximum of 480 updates each. No new training, hydraulic execution, model promotion or sealed-response access occurs during preparation. The total proposed future budget is four fits and 1,920 updates, without a seed or architecture search.

The loss contrast measures the combined change in stratification and robust residual treatment; it does not separately identify their individual effects. The learning-rate contrast holds loss, optimizer and all remaining settings fixed, with no simultaneous scheduler or output-head change. All arms use the same new checkpoint-evaluation rule, so comparison with the historical early-stopped pilot is contextual rather than an additional equal-budget control. Effects are descriptive for this initialization and development sample, not population inference or causal effects of urban interventions. Numeric weights and settings are a concrete, reviewable engineering proposal under the user's preparation request, not a claim of proven efficacy or separate human selection of each value.''')
 text=append(text,6,r'''### Prepared loss and optimization comparison

Retain original point labels and Ground Area weights. Proposed candidate supervision separates reference Peak Mean Depth into nonwet support at most 0.10 m, ordinary wet support above 0.10 through 1.00 m, and deep support above 1.00 m. The nonwet label remains its original depth, not a dry-zero replacement. Define a robust residual penalty in square metres:

\[
\psi_\delta(d)=\begin{cases}d^2,& |d|\leq\delta,\\2\delta |d|-\delta^2,& |d|>\delta.\end{cases}
\]

Here \(d\) is predicted minus reference Peak Mean Depth in metres, \(\delta\) is the fixed proposed residual transition of 0.10 m, and \(\psi_\delta\) is twice the usual Huber penalty. It coincides with MSE for small residuals and has linear tails. The residual transition is distinct from the reference-depth thresholds defining strata.

\[
L_e^{\mathrm{strat}}=\frac{1}{\sum_{k\in\mathcal{K}_e}w_k}\sum_{k\in\mathcal{K}_e}w_k\frac{\sum_{c\in S_{e,k}}A_c\psi_\delta(d_{e,c})}{\sum_{c\in S_{e,k}}A_c}.
\]

Here \(k\) indexes the three reference-depth strata, \(S_{e,k}\) is original eligible exact support for event \(e\) in stratum \(k\), \(\mathcal{K}_e\) contains its nonempty strata, \(w_k\) is the proposed fixed mixture weight 0.50, 0.40 and 0.10 for nonwet, ordinary wet and deep support, and \(d_{e,c}\) is the corresponding residual in cell \(c\). Ground Area \(A_c\) and event/cell notation retain their existing definitions. Renormalize only over present strata, record absent-stratum metrics as undefined and reject an event with no exact support. Deep labels are neither capped nor removed, and deep MAE remains a separate evaluation. The reference arm retains the original area-weighted MSE. These settings are not validation-fitted loss weights.

Define an area-based extent score at the unchanged strict 0.10 m threshold:

\[
B_e=\frac{1}{2}\left(\frac{FN_e}{TP_e+FN_e}+\frac{FP_e}{FP_e+TN_e}\right).
\]

Here \(B_e\) is balanced extent error for parent \(e\); \(TP_e\), \(FN_e\), \(FP_e\) and \(TN_e\) are Ground Area classified as true wet, missed wet, false wet and true nonwet respectively. The first ratio is the missed fraction of reference-wet area and the second is the false-wet fraction of reference-nonwet area. False discovery instead uses predicted-wet area and is reported separately. Empty-class rates remain undefined; an undefined primary validation rate cannot rank a checkpoint. Report original IoU, precision, recall, absolute missed/false area and all depth metrics, with nonwet, ordinary-wet, deep and wet strata and defined-parent counts.

Every proposed arm uses one parent update per training event per epoch, AdamW weight decay 0.00001 and gradient norm limit 10. Keep a constant learning rate within an arm, a full forty-epoch budget without early stopping, the same seed and epoch order, and unchanged architecture, nonnegative output and target units. No augmentation, target rescaling, input changes or additional scenarios enter the contrast. Save each epoch's model and the latest optimizer/random states. Forty epochs are a finite comparison budget rather than a convergence guarantee.

After each epoch, evaluate one fixed model over training and validation separately. Primary selection uses the three-ward intersection of target exact support, original training-only retrieval source exact support and eligible ground. Bind support masks to their original fingerprints and retain all target-exact scores separately. Rank checkpoints by parent-macro balanced extent error, then reference-wet MAE, overall MAE and earliest epoch. Round the metric entries to six decimal places before this total ordering. This is a proposed development selection rule, not a rule retroactively applied to promote an existing model.

A separate proposed development guard requires every validation parent to have balanced extent error no worse than retrieval, overall MAE no worse than both retrieval and always-dry, recall no worse than retrieval and false wet area no larger than retrieval. Comparisons use a declared numerical tolerance of 0.000001 in the reported dimensionless, metre or square-kilometre units. A diagnostic best checkpoint may be retained while marked not ready; a failed guard never triggers a looser limit, another fit or test access. These guards do not modify numerical reference admission or establish physical accuracy.

Per-update logs bind parent/family, forcing, mask, seed and protocol, and record online loss by stratum, gradient norms before/after clipping, finite/missing-gradient flags, actual module parameter changes, optimizer clock, learning rate and declared timing scope. Whole-epoch summaries require all twelve training parents and contiguous optimizer steps. Fixed-model loss and per-parent metrics are recorded separately from the mean loss observed during sequential updates. Epoch frames cover all sixteen parents, six geographic scopes and original/paired exact support, retaining undefined-metric counts. Check that model fingerprints and update counts do not change during evaluation; bind the checkpoint version and separate training, evaluation and checkpoint I/O clocks. Reject nonfinite active predictions, gradients or parameters instead of improving coverage by omission. Interface probes are explicitly segregated from true updates and cannot stand in for historical telemetry.''')
 text=append(text,7,'''### Prepared comparison readiness and execution boundary

Loss, metric, checkpoint and optimization-observer interfaces are now implemented and checked without a fitting loop. Tests cover manually derived robust-loss values and gradients, excluded NaN support, missing strata, strict wet thresholds, area-weighted confusion denominators, undefined classes, parent-macro weighting, mask-bound ranking, baseline failure reporting, actual frozen-data support, zero-update backward probes and complete fixed-model logging. Shared initialization is stored as untrained, with zero optimizer steps. Known synthetic clock and clipping fixtures test the logger only and do not constitute real optimizer updates. Original forcing, normalization, parent partitions, labels and historical frozen deliveries are preserved.

| prepared analytical step | variables used | specification | planned evidence connection | support status |
|---|---|---|---|---|
| Preserve exact reference support | Ground Area, Peak Mean Depth, Rainfall Rate, River Stage | Original admission, unchanged exact masks, frozen normalization and training-only retrieval | Reference uncertainty; Scenario coverage and split audit | Interface custody verified; no new response or training release. |
| Compare loss package and learning rate | Ground Elevation, Ground Area, Building Occupancy, Road Connectivity, Rainfall Rate, River Stage, Peak Mean Depth | Proposed two-by-two MSE/stratified robust loss and constant-rate contrast | Stress and ablation response; Held-out metric and runtime summary | Prepared only; four fits remain unexecuted. |
| Select and diagnose fixed checkpoints | Peak Mean Depth, Peak Wet Area, Ground Area | Parent-macro balanced extent error and depth tie-breaks with separate baseline guards | Held-out surrogate performance | Interfaces verified on synthetic and frozen predictions; no model promoted. |
| Observe optimization faithfully | Peak Mean Depth, Ground Area, End-to-End Runtime | True step clocks, stratified online loss, fixed checkpoint scores and module gradient/update summaries | Held-out metric and runtime summary | Logging contract verified; real future training telemetry and end-to-end timing remain absent. |
| Preserve subsequent measures | Measure Configuration, Mitigation Benefit, Boundary Exchange | Later physically checked no-action/action pairing | Physically checked measure comparison | Pending after the first four milestones; no measures simulated. |

The development-rule probe recognizes the saved late checkpoint's extent improvement but retains its false-positive and depth failures; neither existing checkpoint passes the proposed complete baseline guard. This is an interface check on already exposed validation data, not new model performance or independent verification. The next small trial can begin only under a subsequent scoped training request. Do not extend the four-arm budget, alter data or weights after seeing its outcomes, or open sealed tests automatically. The current data and variables support this limited preparation; final-variable status, real-event validation and the full Section 8 sealed-comparison evidence remain unresolved. Section 8 remains pending, without a new publication result or claim that the prepared loss works.''')
 assert all(section(before,n)==section(text,n) for n in [1,2,3,4,8]);assert text.isascii();path.write_text(text)
 write(REPORT/'anasop_update.json',{'decision':'MILI-D-20261016-004','responsible_skill':'estimation-framework-planning','changed_sections':[5,6,7],'unchanged_sections':[1,2,3,4,8],'before_sha256':sha(REPORT/'previous-AnaSOP.md'),'after_sha256':sha(path),'english_ascii':True,'new_training_executed':False,'numeric_settings_are_proposals':True,'publication_outputs_completed':False});print(json.dumps({'updated':[5,6,7],'unchanged':[1,2,3,4,8],'ascii':True}))
if __name__=='__main__':main()
