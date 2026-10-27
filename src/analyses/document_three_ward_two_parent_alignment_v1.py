"""Record read-only diagnostic evidence in AnaSOP 5-7 under planning skill."""
from pathlib import Path
import sys,re
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_three_ward_unet_exact_pilot import read,write,sha
NAME='three_ward_two_parent_alignment_condition_audit_v1';REPORT=ROOT/'data/exp/estimation'/NAME

def sections(t):return {int(n):v for n,v in re.findall(r'^## ([1-8])\. (.*?)(?=^## [1-8]\. |\Z)',t,re.M|re.S)}
def main():
 receipt=REPORT/'anasop_update.json'
 if receipt.exists():raise FileExistsError('Preserve completed document delivery')
 assert read(REPORT/'delivery_validation.json')['all_checks_passed']
 sop=ROOT/'docs/AnaSOP.md';before=sop.read_text();old=sections(before);s=read(REPORT/'summary.json');decomp=next(r for r in s['error_decomposition'] if r['scope']=='three_wards');a=s['endpoint_field_difference']
 s5='''### Read-only two-parent alignment and conditional-response audit

The user authorizes a read-only investigation of the two already fitted training parents, without another optimizer update, response simulation, sample expansion or sealed-test access. Preserve all original twelve-training/four-validation family roles, original quality masks, original input normalization and earlier fitted states. Use only the two training responses for new diagnostics. Whole-domain source reconstruction and whole-three-ward summaries assess program-level spatial alignment, condition identity and frozen conditional response; they do not independently validate the source terrain or physical reference.

A reproducible input/target mapping reduces support for an interface-error explanation. Nonzero conditional response rules out complete condition collapse for these two inputs, but does not establish correct physical sensitivity. Frozen rain/stage hybrids lack matched reference responses and remain out-of-training-point sensitivity probes. Endpoint gradient compatibility is local and parameterization-dependent; neither a negative component nor a positive full-network cosine identifies the historical cause of poor joint fitting.

'''
 s6='''### Read-only spatial and conditional-response assessment

Independently reconstruct the common-grid affine coordinates, native sparse-cell mapping, surface-node coordinates, boundary point/group order, known-value static input encoding, eligible Ground Area and Peak Mean Depth bounds. Integrate the original fine Ground Elevation under native online Reference Water Level peaks after excluding the original Building Occupancy footprint. Compare the full reconstructed bounds and original quality masks, rather than shifting or relabeling cells to improve learned scores. Preserve the source terrain and targets even when arithmetic-order roundoff is observed.

Independently reconstruct minute-average Rainfall Rate and River Stage from the original piecewise-linear second-level curves using cumulative antiderivatives, then apply the unchanged training-fitted normalization and original minute/channel order. The minute encoding preserves integrals while compressing within-minute shape. Repeated coherent stage groups reflect the original forcing generator and do not establish independent river-group identification.

Replay only the stored initialization, diagnostic update 440 and endpoint update 480 with frozen parameters. Compare both original condition vectors against both original Peak Mean Depth targets on their common original exact support, with Ground Area weighting and separate ward summaries. Report condition-embedding and feature-modulation responses, event-difference error and reference-defined wetness cohorts. Separately retain original own-exact metrics. Algebraically partition paired squared errors into shared and event-contrast components; the resulting shares describe errors, not causal shares of architecture or input effects. Use zero-update endpoint gradients and the historical real-update logs to assess active learning paths without constructing an optimizer or selecting a new checkpoint.

'''
 s7=f'''### Completed read-only alignment and conditional-response audit

The full 1012 by 770 projected 10 m grid, native index reversal, surface-node coordinates, boundary point/group order, all fifty-eight prepared static channels and original quality masks reproduce. Independent integration of the original 0.50 m ground and building classification reconstructs both full-domain Peak Mean Depth bounds within {s['spatial_reconstruction_max_bound_error_m']:.8g} m. The two original condition vectors reproduce exactly after independent minute integration and fixed scaling. No program-level flip, translation, parent swap or condition-layout defect is found within these checks; source geographic and hydraulic accuracy remain separate unresolved questions.

The frozen model does respond to the different conditions. Over {a['common_support_km2']:.6f} square kilometres of common original exact three-ward support, the endpoint event-difference correlation is {a['difference_correlation']:.6f}, and its mean absolute error is {a['difference_MAE_m']:.6f} m. The zero event-difference comparator has error {a['reference_mean_abs_difference_m']:.6f} m. Thus a noncollapsed and partly correlated response still has inadequate placement and amplitude; correlation alone is not successful Peak Wet Area fitting.

On 21.338451 square kilometres classified nonwet in both references at the strict 0.10 m threshold, the two reference mean depths are 0.003639 and 0.012712 m, whereas predictions average 0.068006 and 0.085781 m. This shared reference-nonwet support contributes false predicted areas of 2.008268 and 5.381434 square kilometres. Meanwhile the shared reference-wet support is underpredicted. The algebraic shared-error component accounts for {decomp['shared_fraction']*100:.2f}% of mean paired squared error; this does not assign that percentage to a static-network mechanism.

The complete endpoint gradient cosine is positive at 0.130791; the output-head component is negative at -0.677006, while condition-encoder and feature-modulation components are nearly orthogonal. Historical condition gradients and parameter changes are active after zero-initialized modulation opens the path. These observations do not establish severe whole-network conflict, optimizer failure or convergence. Current evidence prioritizes unresolved spatial/conditional fitting and objective/optimization compatibility over a demonstrated input-index defect or complete conditional collapse. Single-parent fit capability and the dominant mechanism remain unisolated.

| completed diagnostic | existing variables | evidence connection | conclusion boundary |
|---|---|---|---|
| Full-domain coordinate and source-label reconstruction | Ground Elevation, Building Occupancy, Ground Area, Reference Water Level, Peak Mean Depth, Road Connectivity | Reference uncertainty; Domain and data readiness | Program-level alignment verified; physical accuracy unverified. |
| Original condition reconstruction and frozen response | Rainfall Rate, River Stage, Peak Mean Depth | Stress and ablation response | Conditions distinct and used, but response amplitude and placement remain inadequate. |
| Shared and event-dependent error assessment | Peak Mean Depth, Peak Wet Area, Ground Area | Held-out surrogate performance | Known-training-field diagnostic only; no independent performance claim. |
| Endpoint local gradient and historical update audit | Peak Mean Depth, Ground Area | Held-out metric and runtime summary | Active updates verified; dominant fitting mechanism and convergence unisolated. |

No new fit, optimizer update, hydraulic run or parent is added. Original quality masks, normalization and formal family roles remain unchanged. Sections 3-4 need no new data or variables for this diagnostic. The Section 8 outputs remain capable of addressing Section 1 only after independent emulation, observational and active-measure evidence is supplied; no publication-output status is promoted. A matched-exposure single-parent versus joint-parent fitting comparison is an agent recommendation for a later separately authorized design, not an executed or user-finalized experiment.

'''
 assert all(ord(c)<128 for c in s5+s6+s7)
 after=before.replace('## 6. Main Estimation Framework',s5+'## 6. Main Estimation Framework',1).replace('## 7. Analytical Workflow',s6+'## 7. Analytical Workflow',1).replace('## 8. Figure and Table Plan',s7+'## 8. Figure and Table Plan',1)
 new=sections(after);assert set(new)==set(old)==set(range(1,9))
 for i in [1,2,3,4,8]:assert new[i]==old[i]
 assert before[:before.index('## 1.')]==after[:after.index('## 1.')]
 (REPORT/'previous-AnaSOP.md').write_text(before);sop.write_text(after);(REPORT/'updated-AnaSOP.md').write_text(after)
 write(receipt,{'skill':'.codex/skills/estimation-framework-planning/SKILL.md','decision':'MILI-D-20261016-010','sections_changed':[5,6,7],'sections_1_4_and_8_unchanged':True,'before_sha256':sha(REPORT/'previous-AnaSOP.md'),'after_sha256':sha(sop),'new_text_ascii_english':True,'section8_statuses_unchanged':True,'no_new_variables_or_sample_status_change':True,'no_new_formula_symbols':True})
 print('Updated AnaSOP Sections 5-7; Sections 1-4 and 8 preserved.')
if __name__=='__main__':main()
