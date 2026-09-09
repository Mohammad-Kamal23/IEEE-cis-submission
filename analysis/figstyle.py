import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl

# IEEE two-column: text width 7.16 in, column width 3.5 in
COL, WIDE = 3.5, 7.16

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["TeX Gyre Termes", "Liberation Serif", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7.4,
    "axes.titlesize": 7.8,
    "axes.labelsize": 7.6,
    "xtick.labelsize": 6.9,
    "ytick.labelsize": 6.9,
    "legend.fontsize": 6.7,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.35,
    "lines.linewidth": 1.1,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.4,  "ytick.major.size": 2.4,
    "legend.frameon": False,
    "legend.handlelength": 1.9,
    "legend.columnspacing": 1.1,
    "legend.labelspacing": 0.32,
    "figure.dpi": 200,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.012,
    "pdf.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# CVD-validated categorical order (validate_palette.js: all checks PASS,
# worst adjacent CVD dE 10.7). Secondary encoding: line style + marker + label.
INK   = "#1a1a1a"
MUTED = "#6b6b6b"
GRID  = "#d8d8d4"
APEXC = "#0072B2"
STYLE = {
    "BASE":       dict(color="#4D4D4D", ls=(0,(4,2)),      marker="o", label="Uncalibrated"),
    "MC_DROPOUT": dict(color="#7B5AA6", ls=(0,(1,1.4)),    marker="v", label="MC dropout"),
    "TEMP_SCALE": dict(color="#E69F00", ls=(0,(3,1.4,1,1.4)), marker="s", label="Temperature"),
    "VEC_SCALE":  dict(color="#009E73", ls=(0,(5,1.6)),    marker="^", label="Vector"),
    "ISOTONIC":   dict(color="#CC79A7", ls=(0,(1.6,1.4)),  marker="D", label="Isotonic"),
    "DIRICHLET":  dict(color="#D55E00", ls=(0,(2.4,1.2)),  marker="P", label="Dirichlet"),
    "APEX":       dict(color=APEXC,     ls="-",            marker="",  label="APEX"),
}
def grid(ax, axis="both"):
    ax.grid(True, axis=axis, color=GRID, lw=0.35, zorder=0)
    ax.set_axisbelow(True)
    for s in ("left","bottom"):
        ax.spines[s].set_color("#9a9a9a")
def tag(ax, t, dx=-0.20, dy=1.02):
    ax.text(dx, dy, t, transform=ax.transAxes, fontsize=7.9,
            fontweight="bold", va="bottom", ha="left", color=INK)


def declutter(vals, gap):
    """Nudge label positions apart, preserving order. vals: list of floats."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    out = list(vals)
    for k in range(1, len(order)):
        i, j = order[k-1], order[k]
        if out[j] - out[i] < gap:
            out[j] = out[i] + gap
    for k in range(len(order)-1, 0, -1):
        i, j = order[k-1], order[k]
        if out[j] - out[i] < gap:
            out[i] = out[j] - gap
    return out
