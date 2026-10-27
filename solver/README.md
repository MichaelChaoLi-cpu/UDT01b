# Research solver sources

The study uses modified CPU SFINCS sources from Deltares, pinned to commit
`1f1d8286520a8709b2e513854dff18a3616583db`. The upstream repository is
https://github.com/Deltares/SFINCS. Its GPL licence is retained in
[UPSTREAM_LICENSE.txt](UPSTREAM_LICENSE.txt). Upstream notices remain in the
copied files. The pinned upstream archive supplies the remaining source and
third-party notices; this repository does not redistribute the full archive.

`manifest.json` records exact source hashes, the upstream archive hash, the base
container digest and the original experimental image identities. `base_overlay`
contains the modified common sources. The three complete final overlays are:

| Variant | Study use |
|---|---|
| `historical` | Rainfall forcing, actual clocks, closed-face rules and conservative positivity |
| `static_pumps` | Opt-in finite drainage transfer and the original physical pilot |
| `weighted_pools` | Shared receiving allowances and weighted allocation, including S1R |

No hydrodynamic simulation is needed to reconstruct a context:

```bash
python solver/prepare_context.py \
  --archive /path/to/SFINCS-1f1d8286520a8709b2e513854dff18a3616583db.tar.gz \
  --variant weighted_pools \
  --output /tmp/udt01b-weighted-source
```

To explicitly download a missing archive, add `--download`. The script checks
the complete archive and every overlay hash, refuses an existing output
folder, and never starts Docker or a simulation.

A Linux AMD64 CPU image can then be built explicitly:

```bash
docker build --platform linux/amd64 \
  -t udt01b/research-weighted:local /tmp/udt01b-weighted-source
```

The Docker recipe retains the pinned Deltares base image, Fortran NetCDF build
and link probes. This publication pass verifies context reconstruction and
source equality; it does not rebuild images or rerun the simulations. A rebuilt
image has its own identity and must pass the original numerical admission
checks before generating replacement reference outputs. The original image
digests identify local experimental images; they are not downloadable image
URLs or a promise of a public container registry.

The independent fixture drivers remain beside their original source modules
in `src/preprocessing/` and `src/analyses/second_clock_budget_tests/`. These test
interfaces and conservation rules; they do not validate field inundation.
