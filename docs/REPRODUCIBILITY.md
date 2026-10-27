# Reproduction guide

This guide distinguishes code inspection, data preparation, simulation and
post-processing. No stage should be inferred from a successful syntax check.
All commands are run at the repository root. Existing frozen campaigns and
numerical tolerances must not be overwritten or relaxed to obtain acceptance.

## Stage map

| Stage | Main code | Required inputs or prior state |
|---|---|---|
| PLATEAU acquisition | `src/preprocessing/download_plateau_models.py` | Three 2025 ward catalogs |
| Terrain preparation | `src/preprocessing/acquire_three_ward_dem.py`, `prepare_plateau_dem_fallback.py` | Tokyo survey sheets, original ward/water geometry |
| Historical forcing | `src/preprocessing/acquire_tokyo_historical_radar_v1.py`, `acquire_population_additional_radar_v1.py` | Original RISH GRIB2 records and source clocks |
| Normal road network | `src/preprocessing/build_normal_surface_network_v1.py` | Retained Kanto OSM snapshot and access/structure rules |
| Census baseline | `src/preprocessing/build_three_ward_population_block_stability_v1.py` | Official 2020 basic-unit boundaries and population fields |
| Traffic sources | `src/preprocessing/inspect_tokyo_traffic_data_sources_v1.py`, `decode_full_pt_tables_v1.py` | Census tables and applicant-authorized PT files |
| Source-equivalent solver context | `solver/prepare_context.py` | Checksum-matched upstream archive and selected overlay |
| Five-parent baseline | `src/preprocessing/prepare_five_event_population_validation_v1.py`, `src/analyses/run_five_event_population_validation_v1.py` | Prepared geometry, radar source/recipient mapping and qualified solver |
| Shared-receiver comparisons | `src/analyses/shared_receiver_controls_v1.py`, `shared_receiver_controls_transfer_v1.py`, `shared_receiver_controls_august_v1.py` | Frozen intakes, receivers, baselines and source bindings |
| Fixed-layout allowance comparison | `src/analyses/s1r_cross_event_validation_v1.py` | Three parent-specific baselines, fixed S1R layout and weighted allocator |
| September vehicle comparison | `src/analyses/s1r_september_traffic_phases_v1.py` | Admitted simultaneous states and original fixed-cohort demand definition |
| Final evidence | `src/preprocessing/consolidate_three_ward_final_evidence_v1.py` and `preprocess_final_*.py` | Admitted source tables, direct paired contrasts and source receipts |

Acquisition and some historical preparation scripts also create reports or
working directories when invoked. Inspect a script before importing or running
it. The code inspection tool parses sources without importing them.

## Data directories

- `data/raw/`: provider originals and acquisition metadata.
- `data/processed/`: terrain/forcing/network inputs, campaigns and derived arrays.
- `data/exp/`: numerical and analytical checks, provenance and review products.
- `data/results/`: approved figure and table exports.

The directory contracts and individual scripts specify exact filenames.
The released code does not bundle the entire original data tree. A fresh clone
therefore supports source checks immediately, but requires source acquisition,
preparation and qualification before reproducing scientific outputs. Do not
substitute empty placeholders for masks, clocks, data or qualification receipts.

## Qualification rules

Keep the originally registered tests for source/solver identity, true output
clock, conservative rain and roof-volume integration, closed-face exchange,
nonnegative native volume, endpoint/prefix water budgets and bounded finite
pump transfer. Shared-pool runs also retain the original allocation checks.
The final study uses the original 28 numerical and 8 pump checks where
applicable. Admission describes numerical accounting, not agreement with
observed road flooding. No new scenarios or simulations are run for publication
of this code snapshot.

## Approved principal figure builders

| Figure | Script |
|---|---|
| Study area and urban flood computation | `src/analyses/figure_study_area_and_urban_flood_computation.py` |
| Urban geometry and road disruption | `src/analyses/figure_urban_geometry_and_road_disruption.py` |
| Receiving allowance, roads and residents | `src/analyses/figure_receiving_allowance_road_and_resident_effects.py` |
| Transport benefits and local adverse effects | `src/analyses/figure_transport_benefits_and_local_adverse_effects.py` |

The manuscript appendix also uses
`src/analyses/figure_physical_measure_deployment_and_road_effects.py` and
`src/analyses/figure_cross_event_allocation_and_placement_effects.py`. Other
historical builders remain for traceability; they are not all approved
standalone figures.
The current `docs/AnaSOP.md` and `docs/actionbrief.yaml` identify the approved set.

## Table builders

`src/analyses/table_experimental_design.py`,
`table_deployment_assumptions_and_receiving_limits.py`,
`table_physical_road_and_resident_comparison.py`,
`table_adverse_transfer_and_coverage.py`,
`table_vehicle_request_and_common_cohort_cost.py`, and
`table_scenario_coverage_and_evidence_limits.py` generate the numerical
workbooks and review products from qualified evidence. Their prior layouts and
source guards remain part of the record. Browser PNG exports are optional
visual products; they do not replace the numeric tables.

## Rebuilding and compute

`solver/README.md` describes source reconstruction and an explicit container
build. Old image hashes are recorded to identify original runs; they are not
credentials or public image download addresses. Exact source hashes are checked
against the retained experimental contexts during release preparation.

Historical production workers generally use six CPU threads, a 7 GiB container
limit and one hydraulic case at a time. Original campaign manifests define the
actual case-specific time/storage budgets, including explicitly extended limits.
Do not infer a fixed runtime on another computer. Multi-hour nine-hour-domain
runs generate large diagnostics; budget disk space before an explicit rerun.

No default NAS destination, automatic data upload or background simulation is
configured by this release. Local maintenance scripts and their cancelled
archive records are excluded from the publication commits.
