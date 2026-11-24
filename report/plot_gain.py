"""Render the report's paired improvement plot from the Tufts bootstrap results."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

HERE = Path(__file__).resolve().parent
with (HERE.parent / "wind-6afo1x/primary_gain.csv").open() as stream:
    rows = list(csv.DictReader(stream))

fig, ax = plt.subplots(figsize=(6.42, 2.05))
for row in rows:
    lead = int(row["lead_h"])
    gain, lower, upper = (float(row[k]) for k in ("wind_rmse_gain_ms", "lower95", "upper95"))
    color = "#11586a" if lead in (24, 48) else "#77868b"
    ax.errorbar(gain, lead, xerr=[[gain - lower], [upper - gain]], fmt="o",
                color=color, markersize=5.8, elinewidth=1.8, capsize=3)
ax.axvline(0, lw=1.1, color="#343d40", zorder=0)
ax.set(yticks=[12, 24, 36, 48], yticklabels=["12 h", "24 h", "36 h", "48 h"],
       ylim=(55, 5), xlim=(-.015, .255), xticks=[0, .05, .10, .15, .20, .25],
       xlabel="Reduction in vector wind RMSE (m/s)")
ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
ax.set_axisbelow(True)
ax.grid(axis="x", lw=.65, color="#d9dfe1")
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(length=0, labelsize=9, pad=4)
ax.xaxis.label.set_size(9.5)
fig.subplots_adjust(left=.105, right=.965, bottom=.27, top=.975)
fig.savefig(HERE / "physics-gain.png", dpi=250, facecolor="white")
