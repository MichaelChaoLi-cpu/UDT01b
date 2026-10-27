"""Estimation-framework-planning: record the executed bounded comparisons."""
from pathlib import Path
import re,sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_three_ward_unet_exact_pilot import read,write,sha
REPORT=ROOT/'data/exp/estimation/three_ward_spatial_fp_controls_v1'
def sec(t,n):return re.search(r'^## '+str(n)+r'\..*?(?=^## \d+\.|\Z)',t,re.M|re.S).group(0)
def append(t,n,a):
 old=sec(t,n);return t.replace(old,old.rstrip()+'\n\n'+a.strip()+'\n\n',1)
def main():
 assert read(REPORT/'delivery_validation.json')['all_checks_passed'];s=read(REPORT/'summary.json');p=ROOT/'docs/AnaSOP.md';before=p.read_text();text=before
 if '### Executed separated spatial and false-area controls' in before:raise FileExistsError('Already documented')
 text=append(text,5,'''### Executed separated spatial and false-area controls

The user subsequently authorizes implementation and execution of the prepared fixed-data comparison. The three new configurations each complete forty epochs and 480 actual updates, adding 1,440 updates. The original lower-rate stratified configuration is reused as the reference without a new fit. All four share the original stored untrained parameters, original twelve training and four validation parent families, exact quality masks, training-fitted normalization and forty-epoch exposure order. Spatial detail changes only the last pooling operation; false-area supervision adds only the previously proposed nonwet exceedance term with multiplier one. No sample expansion, hydraulic rerun, sealed-test or challenge-response access is included.

The primary treatment comparison uses the same final update count. Also report the common-checkpoint-policy comparisons and same-pool ranking contrasts, which can select different epochs and therefore do not isolate a fixed-budget training effect alone. This single-initialization comparison is conditional on the existing synthetic geometry and already exposed validation parents. Spatial pooling changes computational work and receptive context as well as retained detail; it does not identify a universal representation-capacity effect.''')
 text=append(text,6,'''### Executed interface, objective and selection safeguards

Before fitting, zero-update full-field checks confirm identical complete initial parameter fingerprints and original mask bindings, exact reproduction of the original spatial operator when the switch is off, and the declared one-operation spatial change when on. Independent added-loss and derivative checks cover threshold ties, excluded cells, absent nonwet support and rejection of empty exact support. Synthetic selection fixtures test the guard, overlap key, same-pool balanced-error key and mask/split failures; these fixtures are interface evidence rather than model performance.

Every actual update retains original family, forcing, mask and optimizer clocks, separate base and added penalties, gradients before and after clipping, and module parameter changes. Each epoch retains a checkpoint and fixed-model score frame for all parents and geographic supports. Report common base objective separately from the optimized objective; totals from different supervision policies are not interchangeable quality scores. The latest optimizer and random states are retained. Reference responses missing only a diagnostic export are recovered by frozen-checkpoint inference, never by refitting the reference or creating another parent event.

The accepted common selector preserves the original all-parent baseline guard, ranks passing checkpoints by overlap then wet error, total error and earlier epoch, and retains only a not-ready diagnostic best when no checkpoint passes. Its balanced-error shadow uses the identical candidate pool. Original historical reference selection remains archived, and its later prospective rescoring remains retrospective development evidence. No failure automatically changes weights, budgets, thresholds, label admission or test access.''')
 header='''### Executed separated spatial and false-area controls

The table reports four-parent macro validation values at the common forty-epoch endpoint and unchanged strict 0.10 m threshold on original paired three-ward support. Area is in square kilometres and depth errors in metres. Spatial ranking is an area-weighted wet/nonwet discrimination diagnostic, not another fitted classifier.

| spatial / supervision | wet IoU | spatial ranking | precision | recall | false area | missed area | overall MAE | wet MAE | endpoint admission |
|---|---|---|---|---|---|---|---|---|---|
'''
 for a in ['S0F0','S1F0','S0F1','S1F1']:
  r=s['primary_fixed_epoch40'][a];label={'S0F0':'Original pooling / original loss (reused)','S1F0':'Finer deepest lattice / original loss','S0F1':'Original pooling / added nonwet penalty','S1F1':'Finer deepest lattice / added nonwet penalty'}[a]
  header+='| '+label+' | '+' | '.join(f"{r[k]:.8f}" for k in ['iou_0.1','AUC','precision_0.1','recall_0.1','false_km2_0.1','missed_km2_0.1','mae_m','wet_mae_m'])+' | '+r['status']+' |\n'
 header+='\nPaired factor differences are left minus right at the same endpoint. Positive overlap/ranking/recall and negative false-area/depth differences are ordinarily favorable but must be interpreted jointly and per parent.\n\n| contrast | delta IoU | delta spatial ranking | delta false area | delta recall | delta wet MAE |\n|---|---|---|---|---|---|\n'
 for r in s['matched_epoch40_contrasts']:
  header+='| '+{'spatial_F0':'Finer lattice under original loss','spatial_F1':'Finer lattice under added penalty','FP_S0':'Added penalty under original pooling','FP_S1':'Added penalty under finer lattice'}[r['contrast']]+' | '+' | '.join(f"{r[k]:.8f}" for k in ['delta_iou_0.1','delta_AUC','delta_false_km2_0.1','delta_recall_0.1','delta_wet_mae_m'])+' |\n'
 header+='\nThe common overlap-first selected epochs are '+', '.join(a+'='+str(s['selection'][a]['selected_epoch']) for a in ['S0F0','S1F0','S0F1','S1F1'])+'. The same-pool balanced-error shadow epochs are '+', '.join(a+'='+str(s['selection'][a]['shadow_BER_epoch']) for a in ['S0F0','S1F0','S0F1','S1F1'])+'. These policy contrasts change which retained weights are chosen without adding parameter updates. The original reference selection at epoch thirty-six remains unchanged in its historical record.\n'
 header+=('\nAt least one new arm satisfies all original per-parent development baseline guards. This remains conditional development evidence and does not authorize independent-test access.\n' if s['any_new_arm_passes_original_guard'] else '\nNo new arm satisfies the complete original per-parent development baseline guard. The selected models remain diagnostic and are not promoted as reliable surrogates.\n')
 header+='''

| completed analytical step | variables used | specification | planned evidence connection | support status |
|---|---|---|---|---|
| Verify accepted interfaces | Ground Area, Road Connectivity, Peak Mean Depth | Original initialization and masks; zero-update loss/operator/selection checks | Reference uncertainty; Stress and ablation response | Implementation contract verified before real updates; no physical accuracy conclusion. |
| Execute three new controls and reuse the reference | Ground Elevation, Building Occupancy, Road Connectivity, Rainfall Rate, River Stage, Peak Mean Depth | Two spatial settings crossed with two supervision policies at common exposure budget | Stress and ablation response | Bounded conditional numerical comparison completed, not a population or intrinsic-capacity claim. |
| Compare same-budget footprint and depth | Peak Mean Depth, Peak Wet Area, Ground Area | Paired parent differences, ranking, depth strata and existing geographic descriptors | Held-out surrogate performance | Development errors and tradeoffs quantified; four exposed parents are not independent final evidence. |
| Separate selection and optimization | Peak Mean Depth, Peak Wet Area, End-to-End Runtime | Same candidate pool under two ranking keys, retained true update logs and computation | Held-out metric and runtime summary | Checkpoint timing and optimization diagnostics retained; no end-to-end hydraulic speedup established. |

Training and validation ranking, false-area recurrence and road/building/fallback descriptors remain descriptive. Reduced false area with lower recall and unchanged spatial ranking does not establish better placement. Do not infer true drainage paths, affected road length or active-measure performance from these peak-depth outputs. Source data, geometry, initial parameters, original inputs, normalization and earlier deliveries remain immutable. Sections 3-4 remain sufficient for this bounded comparison without new predictors; later sample expansion still requires its own response qualification and versioned preparation. Additional validation parents remain outside this run. All Section 8 publication statuses remain pending because sealed, observational and intervention evidence have not been supplied.
'''
 text=append(text,7,header);assert all(sec(before,n)==sec(text,n) for n in [1,2,3,4,8]);assert text.isascii();(REPORT/'previous-AnaSOP.md').write_text(before);p.write_text(text);(REPORT/'updated-AnaSOP.md').write_text(text)
 write(REPORT/'anasop_update.json',{'responsible_skill':'estimation-framework-planning','decision':'MILI-D-20261016-008','changed_sections':[5,6,7],'unchanged_sections':[1,2,3,4,8],'english_ascii':True,'before_sha256':sha(REPORT/'previous-AnaSOP.md'),'after_sha256':sha(p),'new_fits':3,'new_updates':1440,'sealed_opened':False,'sample_expansion':False,'publication_outputs_completed':False});print(json.dumps({'updated':[5,6,7],'english_ascii':True}))
if __name__=='__main__':main()
