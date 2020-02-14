""" Constants used by various modules """

# Cell Parameters
NEIGHBOR_RADIUS = 20  # um - radius for a cell to be a "neighbor"
SAME_CELL_RADIUS = 5   # um - radius for a cell to be "the same"
AGGREGATE_RADIUS = 75.3  # um - radius for the aggregate as a whole

NUM_RED = 400  # number of mKate+ cells to simulate
NUM_GREEN = 127  # number of GFP+ cells to simulate

# Plotting
PALETTE = 'deep'

MATHTEXT = 'Arial'

RC_PARAMS_FONTSIZE_NORMAL = {
    'figure.titlesize': '32',
    'axes.titlesize': '24',
    'axes.labelsize': '20',
    'xtick.labelsize': '20',
    'ytick.labelsize': '20',
    'legend.fontsize': '20',
}
RC_PARAMS_LINEWIDTH_NORMAL = {
    'axes.linewidth': '2',
    'lines.linewidth': '5',
    'lines.markersize': '10',
    'xtick.major.size': '5',
    'ytick.major.size': '5',
    'xtick.major.width': '1.5',
    'ytick.major.width': '1.5',
    'xtick.minor.size': '3',
    'ytick.minor.size': '3',
    'xtick.minor.width': '1.5',
    'xtick.minor.width': '1.5',
}
RC_PARAMS_FONT = {
    'font.family': 'sans-serif',
    'font.sans-serif': f'{MATHTEXT}, Arial, Liberation Sans, Bitstream Vera Sans, DejaVu Sans, sans-serif',
    'text.usetex': 'False',
    'mathtext.fontset': 'custom',
    'mathtext.it': f'{MATHTEXT}:italic',
    'mathtext.rm': f'{MATHTEXT}',
    'mathtext.tt': f'{MATHTEXT}',
    'mathtext.bf': f'{MATHTEXT}:bold',
    'mathtext.cal': f'{MATHTEXT}',
    'mathtext.sf': f'{MATHTEXT}',
    'mathtext.fallback_to_cm': 'True',
}
RC_PARAMS_LINE = {
    'grid.linestyle': '-',
    'lines.dash_capstyle': 'butt',
    'lines.dash_joinstyle': 'miter',
    'lines.solid_capstyle': 'projecting',
    'lines.solid_joinstyle': 'miter',
}
RC_PARAMS_DARK = {
    'figure.facecolor': 'black',
    'figure.edgecolor': 'white',
    'text.color': 'white',
    'axes.labelcolor': 'white',
    'axes.facecolor': 'black',
    'axes.edgecolor': 'white',
    'axes.axisbelow': 'True',
    'axes.grid': 'False',
    'axes.spines.left': 'True',
    'axes.spines.bottom': 'True',
    'axes.spines.top': 'False',
    'axes.spines.right': 'False',
    'axes.axisbelow': 'True',
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'legend.frameon': 'False',
    'legend.numpoints': '1',
    'legend.scatterpoints': '1',
    'image.cmap': 'Greys',
    'hatch.color': 'white',
    'grid.color': 'white',
    **RC_PARAMS_FONT,
    **RC_PARAMS_FONTSIZE_NORMAL,
    **RC_PARAMS_LINE,
    **RC_PARAMS_LINEWIDTH_NORMAL,
}
RC_PARAMS_LIGHT = {
    'figure.facecolor': 'white',
    'figure.edgecolor': 'black',
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.spines.left': 'True',
    'axes.spines.bottom': 'True',
    'axes.spines.top': 'False',
    'axes.spines.right': 'False',
    'axes.axisbelow': 'True',
    'axes.grid': 'False',
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'xtick.top': 'False',
    'xtick.bottom': 'True',
    'ytick.left': 'True',
    'ytick.right': 'False',
    'legend.frameon': 'False',
    'legend.numpoints': '1',
    'legend.scatterpoints': '1',
    'image.cmap': 'Greys',
    'grid.linestyle': '-',
    'hatch.color': 'black',
    'grid.color': 'black',
    **RC_PARAMS_FONT,
    **RC_PARAMS_FONTSIZE_NORMAL,
    **RC_PARAMS_LINE,
    **RC_PARAMS_LINEWIDTH_NORMAL,
}
