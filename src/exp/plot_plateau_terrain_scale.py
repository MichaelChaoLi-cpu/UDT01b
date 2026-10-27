"""Plot the saved triangle-edge distributions with locally installed matplotlib.

Run after inspect_plateau_terrain.py. The bands describe packaged geometry,
not acquisition density, elevation accuracy, or hydrodynamic model resolution.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/exp/data-briefing/plateau_terrain"
data = json.loads((OUT / "terrain_audit.json").read_text())
labels, near5, near7 = [], [], []
for ward in data["wards"]:
    for terrain in ward["terrain_files"]:
        labels.append(ward["ward"].split("-", 1)[1].replace("-ku", "").title() + " / " + Path(terrain["gml_member"]).name.split("_")[0])
        near5.append(terrain["sample_edge_fraction_4_8_to_5_2_m"] * 100)
        near7.append(terrain["sample_edge_fraction_6_8_to_7_3_m"] * 100)
other = [100 - a - b for a, b in zip(near5, near7)]
fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
ax.barh(labels, near5, label="4.8-5.2 m (nominal grid side)", color="#287b9c")
ax.barh(labels, near7, left=near5, label="6.8-7.3 m (nominal diagonal)", color="#73c1bc")
ax.barh(labels, other, left=[a+b for a,b in zip(near5,near7)], label="Other edge lengths", color="#d4d6db")
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xlabel("Share of sampled triangle edges (%)")
ax.set_title("Packaged PLATEAU terrain is dominated by a nominal 5 m mesh", loc="left", fontsize=13, pad=15)
ax.legend(loc="lower left", bbox_to_anchor=(0, -0.25), frameon=False, fontsize=9)
fig.text(.03, -.13, "10,000 uniformly sampled triangles per GML; seed 13101. Edge lengths use a local ellipsoidal approximation.\nBoundary meshes are included; these proportions are not administrative-area or accuracy statistics.", fontsize=8, color="#555555")
fig.savefig(OUT / "terrain_edge_distribution.png", dpi=170, bbox_inches="tight")
plt.close(fig)
print(OUT / "terrain_edge_distribution.png")
