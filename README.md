# Urban flood disruption and resident accessibility in central Tokyo

Research code for **Receiving constraints shape modeled road and resident
access benefits during urban flooding**, covering Chiyoda, Chuo and Minato.
The workflow connects PLATEAU city geometry and terrain to two-dimensional
surface-water simulation, road passability, directed vehicle access and
census-based resident exposure. It compares drainage deployment, allocation
and shared receiving allowances under matched scenarios.

Repository: https://github.com/MichaelChaoLi-cpu/UDT01b

## Scope

- Five historical JMA radar rainfall windows, each lasting nine hours.
- A separate rainfall family for the terrain-only and urban-geometry contrast.
- A common 10 m hydraulic grid, 109 saved states and preserved quality masks.
- No-action, local barrier/drainage and later shared-receiver comparisons.
- Official 2020 nighttime population, with candidate-location uncertainty.
- A September vehicle-demand comparison on a fixed comparison cohort.

These are rainfall-driven scenario experiments. Receiving stages and
interventions are prescribed assumptions. Numerical admission checks mass
accounting and interfaces; they do not establish observed flood accuracy.
The three-dimensional city information constrains a two-dimensional hydraulic
calculation; this is not a three-dimensional fluid solver.

## Repository contents

| Location | Contents |
|---|---|
| `src/preprocessing/` | Acquisition, geometry, forcing, masks and evidence preparation |
| `src/analyses/` | Hydraulic/traffic orchestration, audits, paired effects and display builders |
| `src/visualization/` | Browser export helpers and optional city-viewer code |
| `src/exp/` | Earlier exploratory workflows retained for provenance |
| `solver/` | Pinned upstream identity, modified solver sources and context reconstruction |
| `docs/DATA_SOURCES.md` | Provider URLs, versions, exact census queries and access conditions |
| `docs/REPRODUCIBILITY.md` | Workflow order, required inputs and verification boundaries |
| `docs/AnaSOP.md` | Approved research and figure/table plan |
| `docs/actionbrief.yaml` | Variables, evidence resources and output contracts |
| `tools/release_check.py` | Code, dependency-lock and publication-scope checks |

## Environment

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
git clone https://github.com/MichaelChaoLi-cpu/UDT01b.git
cd UDT01b
uv sync --frozen
uv run --no-sync python tools/release_check.py
```

The principal environment is recorded in `pyproject.toml` and `uv.lock`.
The release check uses the Python standard library and does not download data,
start Docker, train a model or run a scientific workflow. `uv sync` installs the
pinned environment and may download substantial geospatial/PyTorch packages.

Hydraulic runs require Docker with Linux AMD64 support and the separately
prepared solver image. Build instructions and exact source bindings are in
[solver/README.md](solver/README.md). Earlier exploratory scripts that import
`hydromt_sfincs` require that additional legacy package; they are not part of
the principal final-analysis environment.

Browser acquisition and table-export helpers also require Node.js, Playwright
and a Chromium/Chrome installation. Some historical helpers preserve the
original author's absolute Playwright/Chrome locations. Adapt those browser
paths to the local installation before using those optional helpers. No
browser session or applicant credentials are supplied.

## Obtain the data

Start with [the complete source list](docs/DATA_SOURCES.md). It gives the three
PLATEAU 2025 ward catalogs, the Tokyo 0.50 m DEM resource, JMA/RISH radar archive,
e-Stat census queries, dated OSM extract, MLIT river/boundary and road-traffic
data, and 2018 Tokyo Person Trip Survey tables.

The PT downloads may require the provider's application process. Obtain them
under your own authorization; the original applicant session is not included.
The source manifests, original field definitions and provider terms determine
how acquired files can be reused. Do not replace historical snapshots silently
with a provider's current dataset.

Large raw files, simulation output, browser sessions, manuscripts, PDF papers,
local storage-maintenance utilities and internal review records are excluded
from the code commits. The tracked legacy DVC pointers are not a complete
public data archive. This release does not invoke DVC or copy data to network
storage. External data locations must be configured explicitly by the user.

## Reproduce the analysis

The scripts use repository-relative data paths. Create/populate `data/raw`,
`data/processed`, `data/exp` and `data/results` according to the documented
acquisition and preprocessing stages. The code check can run on a fresh clone;
scientific scripts require their declared data and prior qualification records.

1. Acquire and inspect sources, then construct the common terrain and city
   constraints. Preserve the source CRS, epochs and unsupported-area masks.
2. Generate forcing with the original rainfall clocks and roof-volume mapping.
3. Reconstruct/build the selected solver, prepare the scenario manifest and
   pass its interface qualification before explicitly starting hydraulic runs.
4. Apply the original numerical admission rules to each completed run.
5. Map the same admitted depth branches and output times to the qualified road
   support; compute directed access and paired resident effects.
6. Evaluate vehicle demand and costs for the declared September states and
   fixed cohort, without treating resident stocks as trips.
7. Consolidate the final evidence and generate the approved figures/tables.

[The reproduction guide](docs/REPRODUCIBILITY.md) lists the concrete scripts.
Campaign workers contain frozen input hashes, completion guards and explicit
run budgets. They refuse to overwrite prior campaigns. Inspect their phase
arguments before running them in a newly prepared workspace. The repository
is a research workflow collection, not a single-command simulator.

## Figures and tables

With the prepared, qualified data and evidence contracts present:

```bash
uv run --no-sync python src/preprocessing/consolidate_three_ward_final_evidence_v1.py
uv run --no-sync python src/analyses/figure_receiving_allowance_road_and_resident_effects.py
```

The approved figures use English labels and 450 dpi. The example commands
perform consolidation and plotting, not hydraulic simulation. Other figure
and table builders are indexed in the reproduction guide. Retain their
validation and provenance outputs when rebuilding displays.

## Interpretation

- Q2.5 and Q5 mean assumed total receiving allowances of 2.5 and 5.0 m³/s,
  shared across five receiving pools; they are not official installed capacities.
- The 319 intake locations in the later comparison are a frozen hypothetical
  layout, not a surveyed pump inventory.
- Road closure uses the stated 0.30 m scenario threshold. Unknown bridge,
  tunnel and exposure branches remain distinct from verified passability.
- Resident-hours measure stock-time exposure, not unique affected people.
- Five rainfall parents, their stage variants and output frames are different
  units. They do not establish a large independent-event validation sample.

## Attribution and versioning

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for solver and data
attributions. Cite the manuscript and the exact repository commit used.
The solver manifests preserve the upstream source commit and original image
identities; a fresh build must be qualified under its own identity.
