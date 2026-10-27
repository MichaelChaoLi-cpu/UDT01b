"""Estimation-framework-planning: narrowly record the authorized diagnosis."""
from pathlib import Path
import sys,re,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_three_ward_unet_exact_pilot import read,write,sha
REPORT=ROOT/'data/exp/estimation/three_ward_d_false_positive_diagnosis_v1'
def sec(text,n):return re.search(r'^## '+str(n)+r'\..*?(?=^## \d+\.|\Z)',text,re.M|re.S).group(0)
def append(text,n,new):
 old=sec(text,n);return text.replace(old,old.rstrip()+'\n\n'+new.strip()+'\n\n',1)
def main():
 s=read(REPORT/'summary.json');f=read(REPORT/'findings.json');assert read(REPORT/'validation.json')['all_checks_passed'];path=ROOT/'docs/AnaSOP.md';before=path.read_text();text=before
 if '### Frozen D-arm false-area diagnosis' in text:raise FileExistsError('Diagnosis already documented')
 (REPORT/'previous-AnaSOP.md').write_text(before)
 text=append(text,5,'''### Frozen D-arm false-area diagnosis

The user authorized a whole-three-ward diagnosis of the lower-rate stratified arm using only existing predictions and forty saved score frames. Preserve the original twelve training and four validation parent families, quality masks, normalization and selected epoch. No additional fitting, inference, optimizer update, hydraulic response or sealed-test access is undertaken. Compare training and validation error, geographic recurrence, loss allocation and checkpoint tradeoffs as descriptive development evidence. Poor training agreement can establish a failure of the current fitting recipe, but does not identify intrinsic spatial capacity, input sufficiency or eventual convergence separately. Static-feature associations and label-side proximity do not establish causal feature importance or hydraulic connectivity.''')
 text=append(text,6,'''### Descriptive false-area diagnostic specification

Retain the original strict 0.10 m Peak Mean Depth threshold, Ground Area weighting, parent-first macro averaging and original paired three-ward support. Contrast false Peak Wet Area, recall, precision, overlap and area-weighted depth-ranking discrimination between training and validation. Summarize false area by existing Ground Elevation, Building Occupancy and Road Connectivity descriptors with explicit within-stratum nonwet denominators. Label-side distance and recurrence summaries describe where errors occur; they are not predictors, replacement quality masks or new operating thresholds.

Decompose the unchanged robust objective on its original full-domain exact support. Keep loss-value shares distinct from mixture coefficients and gradient attribution. An analytic derivative along the existing shared output-head bias, independently checked by central finite differences, indicates only a local direction for that one shared parameter; no parameter is changed. Inspect all existing checkpoint scores under the original balanced-error-first ranking and separate baseline admission. Retrospective minimum-false-area, minimum-depth-error and final-budget comparisons explain tradeoffs without promoting a new checkpoint or tuning on validation.''')
 a=s['primary_metrics']['train'];b=s['primary_metrics']['validation'];late=next(r for r in s['retrospective_checkpoints_not_promoted'] if r['reason']=='final_budget')
 text=append(text,7,f'''### Frozen D-arm false-area diagnosis

The selected lower-rate stratified arm has mean false area {a['false_km2_0.1']:.3f} square kilometres on training and {b['false_km2_0.1']:.3f} on validation, with wet overlap {a['iou_0.1']:.3f} and {b['iou_0.1']:.3f}. Area-weighted wet/nonwet depth-ranking discrimination is approximately 0.75 in both splits. These measurements show inadequate spatial fitting under the current recipe, rather than only a new-event generalization failure; event intensities differ, so the area difference itself is not a controlled generalization contrast.

On validation, {f['validation_false_share_within50m_of_reference_wet']*100:.1f}% of false area lies within 50 m straight-line distance of certified reference wet cells, and {f['validation_false_share_reference_at_most1cm']*100:.1f}% has reference depth at most 0.01 m. On common support, approximately 1.723 square kilometres stays nonwet in all four validation references but is falsely wet in at least three. This supports recurrent broad false footprints around reference wet areas, not proof of a terrain-source error or a capacity limit. Road-rich nonwet cells have higher descriptive false rates; explicit subcell ambiguity and predominantly city-model-filled terrain account for only small shares of false area.

At the selected epoch, deep support covers approximately 0.30% of training area yet contributes {f['training_deep_loss_value_share']*100:.1f}% of mean objective value; nonwet support covers 91.5% of area and contributes {f['training_nonwet_loss_value_share']*100:.1f}%. The false-area contribution is {f['training_false_wet_loss_value_share']*100:.1f}%. The instantaneous shared-bias derivative penalizes false wet predictions and favors a uniform logit decrease for eleven of twelve training parents. Thus false predictions are not ignored by the objective, and deep loss-value dominance does not prove that all outputs must increase. Spatially selective fitting remains necessary.

Historical gradients are finite with no clipping among 480 updates, and the fixed training objective decreases by approximately half from first to final epoch. Late oscillation and continuing improvement prevent a convergence claim. Epoch forty has false area {late['false_km2_0.1']:.3f} square kilometres, {f['retrospective_epoch40_false_area_reduction_fraction']*100:.1f}% below selected epoch thirty-six, but recall is {-f['retrospective_epoch40_recall_change']*100:.2f} percentage points lower. The original balanced-error rule therefore retains epoch thirty-six; every saved D checkpoint still fails complete baseline admission. Changing checkpoint alone cannot make the retained candidates admissible under the original rules.

| diagnostic step | existing variables used | evidence connection | support status |
|---|---|---|---|
| Compare fitted and development events | Peak Mean Depth, Peak Wet Area, Ground Area | Held-out surrogate performance | Current recipe has a training-side spatial fit failure; no independent test conclusion. |
| Locate recurring whole-area errors | Ground Elevation, Building Occupancy, Road Connectivity, Peak Mean Depth | Domain and data readiness; Stress and ablation response | Descriptive associations only; no causal input attribution or verified flow-route inference. |
| Inspect objective and retained checkpoints | Peak Mean Depth, Peak Wet Area, Ground Area | Held-out metric and runtime summary | Loss and ranking tradeoffs quantified; no new supervision rule, threshold or promoted model. |

A later separately scoped comparison should isolate spatial discrimination from supervision and optimization changes under matched data and budgets. That is a recommendation, not authorization or a claim that additional capacity, a graph model, more events or longer training will fix the problem. Sections 3-4 support this limited diagnosis without new variables or admission changes. Section 8 remains conditional on independent sealed and observational evidence, and its publication outputs remain pending.''')
 assert all(sec(before,n)==sec(text,n) for n in [1,2,3,4,8]);assert text.isascii();path.write_text(text);(REPORT/'updated-AnaSOP.md').write_text(text)
 write(REPORT/'anasop_update.json',{'responsible_skill':'estimation-framework-planning','decision':'MILI-D-20261016-006','changed_sections':[5,6,7],'unchanged_sections':[1,2,3,4,8],'english_ascii':True,'before_sha256':sha(REPORT/'previous-AnaSOP.md'),'after_sha256':sha(path),'section8_status':'pending; conditionally coherent but not yet supported by sealed/observational/action evidence','section3_refinement_required_for_this_diagnosis':False,'section4_refinement_required_for_this_diagnosis':False,'new_fit_count':0,'optimizer_updates':0,'sealed_opened':False})
 print(json.dumps({'updated_sections':[5,6,7],'preserved_sections':[1,2,3,4,8],'ascii':True}))
if __name__=='__main__':main()
