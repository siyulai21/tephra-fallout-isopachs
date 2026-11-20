"""Shared plot colours and typography for eruption reports."""
from matplotlib.colors import LinearSegmentedColormap


INK = "#183843"
MUTED = "#5C7075"
TEAL = "#087E8B"
AMBER = "#AF6C25"
PAPER = "#FAFBF8"
CMAP = LinearSegmentedColormap.from_list("tephra", ["#E7F2EA", "#B0D9CF", "#5EB3AE", "#257F8E", "#183B57"])
STYLE = {"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
         "axes.labelcolor": MUTED, "axes.edgecolor": "#B8C8CA", "xtick.color": MUTED,
         "ytick.color": MUTED, "axes.titleweight": "bold", "svg.fonttype": "none",
         "savefig.facecolor": PAPER, "figure.facecolor": PAPER, "axes.facecolor": PAPER}
