""" Plotting tools for the simulation framework

Styling tools:

* :py:class:`set_plot_style`: Plot style context manager
* :py:class:`colorwheel`: Custom color palettes

Plotting Functions:

* :py:func:`plot_3d_sphere_cloud`: Plot a sphere cloud in 3D

Axis element functions:

* :py:func:`add_lineplot`: Add lineplots to an axis
* :py:func:`add_histogram`: Add a histogram to an axis

Utilities:

* :py:func:`bootstrap_ci`: Bootstrap estimate of confidence intervals
* :py:func:`get_histogram`: Get a kernel smoothed histogram from binned data

"""

# Imports
import itertools
from contextlib import ContextDecorator
from typing import List, Tuple, Optional, Dict, Callable
import pathlib

# 3rd party imports
import numpy as np

from scipy.stats import gamma, gaussian_kde
from scipy.integrate import simps

import pandas as pd

import seaborn as sns

import matplotlib.cm as mplcm
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
from mpl_toolkits.mplot3d import Axes3D

# Our own imports
from .consts import (
    PALETTE, RC_PARAMS_DARK, RC_PARAMS_LIGHT
)

# Styling


class set_plot_style(ContextDecorator):
    """ Context manager for styling matplotlib plots

    Basic usage as a context manager

    .. code-block:: python

        with set_plot_style('dark') as style:
            # In here, plots are 'dark' styled
            fig, ax = plt.subplots(1, 1)
            ax.plot([1, 2, 3], [1, 2, 3])
            # Save the plot with correct background colors
            style.savefig('some_fig.png')

    Can also be used as a decorator

    .. code-block:: python

        @set_plot_style('dark')
        def plot_something():
            # In here, plots are 'dark' styled
            fig, ax = plt.subplots(1, 1)
            ax.plot([1, 2, 3], [1, 2, 3])
            plt.show()

    For more complex use, see the
    `Matplotlib rcParam <http://matplotlib.org/users/customizing.html>`_
    docs which list all the parameters that can be tweaked.

    :param str style:
        One of 'dark', 'minimal', 'poster', 'dark_poster', 'default'
    """

    _active_styles = []

    def __init__(self, style: str = 'dark'):
        style = style.lower().strip()
        self.stylename = style
        if style == 'dark':
            self.params = RC_PARAMS_DARK
            self.savefig_params = {'frameon': False,
                                   'facecolor': 'k',
                                   'edgecolor': 'k'}
        elif style == 'light':
            self.params = RC_PARAMS_LIGHT
            self.savefig_params = {'frameon': False,
                                   'facecolor': 'w',
                                   'edgecolor': 'w'}
        elif style == 'default':
            self.params = {}
            self.savefig_params = {}
        else:
            raise KeyError(f'Unknown plot style: "{style}"')

    @property
    def axis_color(self):
        if self.stylename.startswith('dark'):
            default = 'white'
        else:
            default = 'black'
        return self.params.get('axes.edgecolor', default)

    @classmethod
    def get_active_style(cls) -> Optional[str]:
        """ Get the currently active style, or None if nothing is active """
        if cls._active_styles:
            return cls._active_styles[-1]
        return None

    def twinx(self, ax: Optional = None):
        """ Create a second axis sharing the x axis

        :param Axes ax:
            The axis instance to set to off
        """
        if ax is None:
            ax = plt.gca()
        ax2 = ax.twinx()

        # Fix up the defaults to make sense
        ax2.spines['right'].set_visible(True)
        ax2.tick_params(axis='y',
                        labelcolor=self.axis_color,
                        color=self.axis_color,
                        left=True)
        return ax2

    def set_axis_off(self, ax: Optional = None):
        """ Remove labels and ticks from the axis

        :param Axes ax:
            The axis instance to set to off
        """
        if ax is None:
            ax = plt.gca()

        # Blank all the things
        ax.set_xticks([])
        ax.set_yticks([])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_axis_off()

    def rotate_xticklabels(self, ax,
                           rotation: float,
                           horizontalalignment: str = 'center',
                           verticalalignment: str = 'center',
                           rotation_mode: str = 'default'):
        """ Rotate the x ticklabels

        :param float rotation:
            Rotation of the text (in degrees)
        :param str rotation_mode:
            Either "default" or "anchor"
        """
        for tick in ax.get_xticklabels():
            plt.setp(tick,
                     rotation=rotation,
                     horizontalalignment=horizontalalignment,
                     verticalalignment=verticalalignment,
                     rotation_mode=rotation_mode)

    def rotate_yticklabels(self, ax,
                           rotation: float,
                           horizontalalignment: str = 'center',
                           verticalalignment: str = 'center',
                           rotation_mode: str = 'default'):
        """ Rotate the y ticklabels

        :param float rotation:
            Rotation of the text (in degrees)
        :param str rotation_mode:
            Either "default" or "anchor"
        """
        for tick in ax.get_yticklabels():
            plt.setp(tick,
                     rotation=rotation,
                     horizontalalignment=horizontalalignment,
                     verticalalignment=verticalalignment,
                     rotation_mode=rotation_mode)

    def show(self,
             outfile: Optional[pathlib.Path] = None,
             transparent: bool = True,
             tight_layout: bool = False,
             close: bool = True,
             fig: Optional = None):
        """ Act like matplotlib's show, but also save the file if passed

        :param Path outfile:
            If not None, save to this file instead of plotting
        :param bool transparent:
            If True, save with a transparent background if possible
        :param bool tight_layout:
            If True, try and squish the layout before saving
        """
        if tight_layout:
            plt.tight_layout()

        if outfile is None:
            plt.show()
        else:
            print('Writing {}'.format(outfile))
            self.savefig(outfile, transparent=transparent, fig=fig)
            if close:
                plt.close()

    def update(self, params: Dict):
        """ Update the matplotlib rc.params

        :param dict params:
            rcparams to fiddle with
        """
        self.params.update(params)

    def savefig(self,
                savefile: pathlib.Path,
                fig: Optional = None,
                **kwargs):
        """ Save the figure, with proper background colors

        :param Path savefile:
            The file to save
        :param fig:
            The figure or plt.gcf()
        :param \\*\\*kwargs:
            The keyword arguments to pass to fig.savefig
        """
        if fig is None:
            fig = plt.gcf()

        savefile = pathlib.Path(savefile)
        savefile.parent.mkdir(exist_ok=True, parents=True)

        savefig_params = dict(self.savefig_params)
        savefig_params.update(kwargs)
        fig.savefig(str(savefile), **kwargs)

    def __enter__(self):
        self._style = plt.rc_context(self.params)
        self._style.__enter__()
        self._active_styles.append(self.stylename)
        return self

    def __exit__(self, *args, **kwargs):
        self._style.__exit__(*args, **kwargs)
        self._active_styles.pop()


class colorwheel(object):
    """ Generate colors like a matplotlib color cycle

    .. code-block:: python

        palette = colorwheel(palette='some seaborn palette', n_colors=5)
        for item, color in zip(items, colors):
            # In here, the colors will cycle over and over for each item

        # Access by index
        color = palette[10]

    :param str palette:
        A palette that can be recognized by seaborn
    :param int n_colors:
        The number of colors to generate
    """

    def __init__(self,
                 palette: str = PALETTE,
                 n_colors: int = 10):
        if isinstance(palette, colorwheel):
            palette = palette.palette
        self.palette = palette
        self.n_colors = n_colors

        self._idx = 0
        self._color_table = None

    @classmethod
    def from_colors(cls,
                    colors: List[str],
                    n_colors: Optional[int] = None):
        """ Make a palette from a list of colors

        :param str colors:
            A list of matplotlib colors to use
        """
        if n_colors is None:
            n_colors = len(colors)
        palette = []
        for _, color in zip(range(n_colors, itertools.cycle)):
            palette.append(mplcolors.to_rgba(color))
        return cls(palette, n_colors=n_colors)

    @classmethod
    def from_color_range(cls,
                         color_start: str,
                         color_end: str,
                         n_colors: int):
        """ Make a color range """
        palette = []
        color_start = mplcolors.to_rgba(color_start)
        color_end = mplcolors.to_rgba(color_end)

        red_color = np.linspace(color_start[0], color_end[0], n_colors)
        green_color = np.linspace(color_start[1], color_end[1], n_colors)
        blue_color = np.linspace(color_start[2], color_end[2], n_colors)

        for r, g, b in zip(red_color, green_color, blue_color):
            palette.append((r, g, b, 1.0))
        return cls(palette, n_colors=n_colors)

    # Dynamic color palettes
    # These aren't as good as the ones that come with matplotlib
    def wheel_bluegrey3(self):
        return [
            (0x04/255, 0x04/255, 0x07/255, 1.0),
            (0xb0/255, 0xb0/255, 0xb3/255, 1.0),
            (0x00/255, 0x00/255, 0xff/255, 1.0),
        ]

    def wheel_bluegrey4(self):
        return [
            (0xa2/255, 0xa5/255, 0xa7/255, 1.0),
            (0x5c/255, 0xca/255, 0xe7/255, 1.0),
            (0x04/255, 0x07/255, 0x07/255, 1.0),
            (0x3e/255, 0x5b/255, 0xa9/255, 1.0),
        ]

    def wheel_blackwhite(self) -> List[Tuple]:
        """ Colors from black to white in a linear ramp """
        colors = np.linspace(0, 1, self.n_colors)
        return [(c, c, c, 1.0) for c in colors]

    def wheel_greyblack(self) -> List[Tuple]:
        """ Colors from grey to black in a linear ramp """
        colors = np.linspace(0.75, 0, self.n_colors)
        return [(c, c, c, 1.0) for c in colors]

    def wheel_greywhite(self) -> List[Tuple]:
        """ Colors from grey to white in a linear ramp """
        colors = np.linspace(0.25, 1, self.n_colors)
        return [(c, c, c, 1.0) for c in colors]

    def wheel_lightgreywhite(self) -> List[Tuple]:
        """ Colors from grey to white in a linear ramp """
        colors = np.linspace(0.608, 1, self.n_colors)
        return [(c, c, c, 1.0) for c in colors]

    def wheel_redgrey(self) -> List[Tuple]:
        """ Grey to red color space """
        red = np.linspace(155/255, 228/255, self.n_colors)
        green = np.linspace(155/255, 26/255, self.n_colors)
        blue = np.linspace(155/255, 28/255, self.n_colors)
        return [(r, g, b, 1.0) for r, g, b in zip(red, green, blue)]

    def wheel_bluegrey(self) -> List[Tuple]:
        """ Grey to blue color space """
        red = np.linspace(155/255, 70/255, self.n_colors)
        green = np.linspace(155/255, 130/255, self.n_colors)
        blue = np.linspace(155/255, 180/255, self.n_colors)
        return [(r, g, b, 1.0) for r, g, b in zip(red, green, blue)]

    @property
    def color_table(self):
        if self._color_table is not None:
            return self._color_table

        # Magic color palettes
        palette = self.palette
        if isinstance(palette, str):
            if palette.startswith('wheel_'):
                palette = getattr(self, palette)()
            elif palette.startswith('color_'):
                color = palette.split('_', 1)[1]
                color = mplcolors.to_rgba(color)
                palette = [color for _ in range(self.n_colors)]
            else:
                palette = palette
        else:
            palette = self.palette

        # Memorize the color table then output it
        self._color_table = sns.color_palette(palette=palette, n_colors=self.n_colors)
        return self._color_table

    def __len__(self):
        return len(self.color_table)

    def __getitem__(self, idx):
        return self.color_table[idx % len(self.color_table)]

    def __iter__(self):
        self._idx = 0
        return self

    def __next__(self):
        color = self.color_table[self._idx]
        self._idx = (self._idx + 1) % len(self.color_table)
        return color

    next = __next__

# Helper Functions


def bootstrap_ci(data: np.ndarray,
                 n_boot: int = 1000,
                 random_seed: Optional[int] = None,
                 ci: float = 95,
                 func: Callable = np.mean,
                 axis: int = 0) -> Tuple[np.ndarray]:
    """ Calculate a confidence interval from the input data using bootstrapping

    :param ndarray data:
        The data to bootstrap sample
    :param int n_boot:
        Number of times to sample the frame
    :param int random_seed:
        Seed for the random number generator
    :param float ci:
        Confidence interval to calculate (mean +/- ci/2.0)
    :param Callable func:
        Function to calculate the ci around (default: np.mean)
    :param int axis:
        Which axis to sample over
    :returns:
        The upper and lower bounds on the CI
    """
    n = data.shape[axis]
    rs = np.random.RandomState(random_seed)
    boot_dist = []
    for i in range(n_boot):
        resampler = rs.randint(0, n, n)
        sample = data.take(resampler, axis=axis)
        boot_dist.append(func(sample, axis=axis))
    boot_dist = np.array(boot_dist)
    return np.percentile(boot_dist, [50 - ci/2, 50 + ci/2], axis=0)


def get_histogram(data: np.ndarray,
                  bins: int,
                  range: Optional[Tuple[int]] = None,
                  kernel_smoothing: bool = True,
                  kernel_bandwidth: Optional[str] = None,
                  kernel_samples: int = 100) -> Tuple[np.ndarray]:
    """ Get a histogram and a kernel fit for some data

    :param ndarray data:
        The data to fit
    :param int bins:
        The number of bins to generate
    :param tuple[float] range:
        The range to fit bins to (argument to np.histogram)
    :param bool kernel_smoothing:
        If True, also generate a kernel-smoothed fit. If False, xkernel, ykernel are None
    :param str kernel_bandwidth:
        If not None, the method to use to estimate the kernel smoothed fit
    :param int kernel_samples:
        The number of samples to draw for the kernel fit
    :returns:
        xbins, ybins, xkernel, ykernel
    """
    bins_y, bins_x = np.histogram(data, bins=bins, range=range)

    # Estimate the kernel smoothed fit
    if kernel_smoothing:
        kernel = gaussian_kde(data, bw_method=kernel_bandwidth)
        kernel_x = np.linspace(bins_x[0], bins_x[-1], kernel_samples)
        kernel_y = kernel(kernel_x)

        # Rescale for equal areas
        bin_width = bins_x[1:] - bins_x[:-1]
        hist_area = np.sum(bin_width * bins_y)
        kernel_area = simps(kernel_y, kernel_x)
        kernel_y = kernel_y * hist_area / kernel_area
    else:
        kernel_x = kernel_y = None
    return bins_x, bins_y, kernel_x, kernel_y


# Plot functions


def add_lineplot(ax,
                 data: pd.DataFrame,
                 x: str, y: str,
                 hue: Optional[str] = None,
                 order: Optional[List[str]] = None,
                 hue_order: Optional[List[str]] = None,
                 palette: str = PALETTE,
                 savefile: Optional[pathlib.Path] = None,
                 label: Optional[str] = None,
                 err_style: str = 'band'):
    """ Add a seaborn-style lineplot with extra decorations

    :param Axes ax:
        The matplotlib axis to add the barplot for
    :param DataFrame data:
        The data to add a barplot for
    :param str x:
        The column to use for the categorical values
    :param str y:
        The column to use for the real values
    :param str palette:
        The palette to use
    :param Path savefile:
        If not None, save the figure data to this path
    """
    bins = {}

    data = data.dropna()

    if order is None:
        order = np.sort(np.unique(data[x]))
    if hue is None:
        hue_order = [None]
    elif hue_order is None:
        hue_order = np.sort(np.unique(data[hue]))

    for cat in order:
        for hue_cat in hue_order:
            if hue_cat is None:
                mask = data[x] == cat
            else:
                mask = np.logical_and(data[x] == cat, data[hue] == hue_cat)

            # Handle missing categories
            n_samples = np.sum(mask)
            if n_samples >= 3:
                catdata = data[mask]
                ydata = catdata[y].values

                ymean = np.mean(ydata)
                ylow, yhigh = bootstrap_ci(ydata)
            else:
                ymean = ylow = yhigh = np.nan

            if hue is None:
                bins.setdefault(x, []).append(cat)
                bins.setdefault(f'{y} Mean', []).append(ymean)
                bins.setdefault(f'{y} CI Low', []).append(ylow)
                bins.setdefault(f'{y} CI High', []).append(yhigh)
                bins.setdefault('Samples', []).append(n_samples)
            else:
                bins.setdefault(x, []).append(cat)
                bins.setdefault(hue, []).append(hue_cat)
                bins.setdefault(f'{y} Mean', []).append(ymean)
                bins.setdefault(f'{y} CI Low', []).append(ylow)
                bins.setdefault(f'{y} CI High', []).append(yhigh)
                bins.setdefault('Samples', []).append(n_samples)

    # Save the background data
    bins = pd.DataFrame(bins)
    if savefile is not None:
        if savefile.suffix != '.xlsx':
            savefile = savefile.parent / (savefile.stem + '.xlsx')
        bins.to_excel(str(savefile))

    # Now draw the plots
    palette = colorwheel(palette, len(hue_order))

    for i, hue_cat in enumerate(hue_order):
        if hue_cat is None:
            xcoords = bins[x].values
            ymean = bins[f'{y} Mean'].values
            ylow = bins[f'{y} CI Low'].values
            yhigh = bins[f'{y} CI High'].values
            hue_label = label
        else:
            hue_bins = bins[bins[hue] == hue_cat]

            xcoords = hue_bins[x].values
            ymean = hue_bins[f'{y} Mean'].values
            ylow = hue_bins[f'{y} CI Low'].values
            yhigh = hue_bins[f'{y} CI High'].values
            if label is None:
                hue_label = hue_cat
            else:
                hue_label = f'{hue_cat} {label}'
        color = palette[i]

        if err_style in ('band', 'bands'):
            ax.fill_between(xcoords, ylow, yhigh, facecolor=color, alpha=0.5)
            ax.plot(xcoords, ymean, '-', color=color, label=hue_label)
        elif err_style in ('bar', 'bars'):
            ax.errorbar(xcoords, ymean, np.stack([ymean-ylow, yhigh-ymean], axis=0),
                        capsize=15, linewidth=3, color=color, label=hue_label)
        else:
            raise ValueError(f'Unknown error style: "{err_style}"')
    return ax


def add_histogram(ax,
                  data: np.ndarray,
                  xlabel: Optional[str] = None,
                  ylabel: str = 'Counts',
                  title: Optional[str] = None,
                  bins: int = 10,
                  draw_bars: bool = True,
                  bar_width: float = 0.7,
                  range: Optional[Tuple[float]] = None,
                  fit_dist: Optional[str] = None,
                  fit_dist_color: str = 'r',
                  kernel_smoothing: bool = True,
                  label_kernel_peaks: Optional[str] = None,
                  kernel_smoothing_color: str = 'c',
                  kernel_bandwidth: Optional[str] = None,
                  vlines: Optional[List[np.ndarray]] = None,
                  vline_colors: str = 'b'):
    """ Add a histogram plot

    Basic Usage:

    .. code-block:: python

        fig, ax = plt.subplots(1, 1)
        histogram(ax, np.random.rand(64, 64),
                  draw_bars=True,
                  kernel_smoothing=True,
                  fit_dist='poisson',
                  vlines=[0.25, 0.75])

    This will draw the histogram with a kernel smoothed fit, a poisson fit,
    and vertical lines at x coordinates 0.25 and 0.75.

    :param Axis ax:
        The axis to add the histogram to
    :param ndarray data:
        The data to make the histogram for
    :param str xlabel:
        Label for the x axis
    :param str ylabel:
        Label for the y axis
    :param str title:
        Title for the axis
    :param int bins:
        Number of bins in the histogram
    :param bool draw_bars:
        If True, draw the histogram bars
    :param float bar_width:
        The width of the bars to plot
    :param tuple[float] range:
        The range to fit bins to (argument to np.histogram)
    :param str fit_dist:
        The name of a distribution to fit to the data
    :param str fit_dist_color:
        The color of the fit dist line
    :param bool kernel_smoothing:
        If True, plot the kernel smoothed line over the bars
    :param str label_kernel_peaks:
        Any of min, max, both to label extrema in the kernel
    :param str kernel_smoothing_color:
        The color of the kernel smoothed fit line
    :param str kernel_bandwidth:
        The method to calculate the kernel width with
    :param list vlines:
        x coords to draw vertical lines at
    :param list vline_colors:
        The color or list of colors for the spectra
    """

    # Estimate the histogram
    data = data[np.isfinite(data)]

    xbins, hist, kernel_x, kernel_y = get_histogram(
        data, bins=bins, range=range,
        kernel_smoothing=kernel_smoothing,
        kernel_bandwidth=kernel_bandwidth)

    width = bar_width * (xbins[1] - xbins[0])
    center = (xbins[:-1] + xbins[1:])/2

    # Add bars for the histogram
    if draw_bars:
        ax.bar(center, hist, align='center', width=width)

    # Estimate the kernel smoothed fit
    if kernel_smoothing:
        # Add a kernel smoothed fit
        ax.plot(kernel_x, kernel_y, color=kernel_smoothing_color)

        if label_kernel_peaks in ('max', 'both', True):
            maxima = (np.diff(np.sign(np.diff(kernel_y))) < 0).nonzero()[0] + 1
            kx_maxima = kernel_x[maxima]
            ky_maxima = kernel_y[maxima]

            ax.plot(kx_maxima, ky_maxima, 'oc')
            for kx, ky in zip(kx_maxima, ky_maxima):
                ax.text(kx, ky*1.05, "{}".format(float("{:.2g}".format(kx))),
                        color="c", fontsize=12)

        if label_kernel_peaks in ('min', 'both', True):
            minima = (np.diff(np.sign(np.diff(kernel_y))) > 0).nonzero()[0] + 1
            kx_minima = kernel_x[minima]
            ky_minima = kernel_y[minima]

            ax.plot(kx_minima, ky_minima, 'oy')
            for kx, ky in zip(kx_minima, ky_minima):
                ax.text(kx, ky*0.88, "{}".format(float("{:.2g}".format(kx))),
                        color="y", fontsize=12)

    # Fit an model distribution to the data
    if fit_dist is not None:
        opt_x = np.linspace(xbins[0], xbins[-1], 100)

        if fit_dist == 'gamma':
            fit_alpha, fit_loc, fit_beta = gamma.fit(data + 1e-5)
            # print(fit_alpha, fit_loc, fit_beta)
            opt_y = data = gamma.pdf(opt_x, fit_alpha, loc=fit_loc, scale=fit_beta) * data.shape[0]
        else:
            raise KeyError(f'Unknown fit distribution: {fit_dist}')

        ax.plot(opt_x, opt_y, fit_dist_color)

    # Add spectral lines
    if vlines is None:
        vlines = []
    if isinstance(vline_colors, (str, tuple)):
        vline_colors = [vline_colors for _ in vlines]

    if len(vlines) != len(vline_colors):
        raise ValueError(f'Number of colors and lines needs to match: {vlines} vs {vline_colors}')

    ymin, ymax = ax.get_ylim()
    for vline, vline_color in zip(vlines, vline_colors):
        ax.vlines(vline, ymin, ymax, colors=vline_color)

    # Label the axes
    if xlabel not in (None, ''):
        ax.set_xlabel(xlabel)
    if ylabel not in (None, ''):
        ax.set_ylabel(ylabel)
    if title not in (None, ''):
        ax.set_title(f'{title} (n={data.shape[0]})')
    else:
        ax.set_title(f'n = {data.shape[0]}')


# Complete Plots


def plot_3d_sphere_cloud(centers: List[Tuple[np.ndarray]],
                         colors: List[str] = None,
                         cmap: str = 'inferno',
                         cvalues: Optional[List[np.ndarray]] = None,
                         vmin: Optional[float] = None,
                         vmax: Optional[float] = None,
                         radii: List[float] = 1.0,
                         title: Optional[str] = None,
                         marker: str = 'o',
                         markersize: float = 10,
                         figsize: Tuple[int] = (16, 16),
                         outfile: Optional[pathlib.Path] = None,
                         add_colorbar: bool = False):
    """ Plot the raw points we sampled

    :param list[tuple[ndarray]] points:
        A list of x, y, z tuples for each population
    :param list[str] colors:
        A list of colors for each population
    :param str title:
        The title for the plot
    :param Path outfile:
        The path to write the output file to
    :param str marker:
        Matplotlib marker shape to plot
    :param int markersize:
        Size for the markers to draw
    """
    if isinstance(radii, (int, float)):
        radii = [radii for _ in centers]

    if colors is None and cvalues is None:
        raise ValueError('Pass one of "colors" or "cvalues" to plot_3d_sphere_cloud')

    # Convert the color values into a heatmap
    if colors is None:
        if vmin is None:
            vmin = np.nanmin(cvalues)
        if vmax is None:
            vmax = np.nanmax(cvalues)
        norm = mplcolors.Normalize(vmin=vmin, vmax=vmax)
        cmapper = mplcm.get_cmap(cmap)
        colors = []
        for cvalue in cvalues:
            colors.append(cmapper(norm(cvalue)))
        mappable = mplcm.ScalarMappable(norm=norm, cmap=cmap)
    else:
        mappable = None

    # Check that the shapes make sense
    assert Axes3D is not None
    if len(centers) != len(colors):
        raise ValueError('Got {} centers but {} colors'.format(len(centers), len(colors)))
    if len(centers) != len(radii):
        raise ValueError('Got {} centers but {} radii'.format(len(centers), len(radii)))

    # Plot everything
    all_x = []
    all_y = []
    all_z = []

    if add_colorbar:
        figsize = (figsize[0]*1.4, figsize[1])
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    for center, color, radius in zip(centers, colors, radii):
        px, py, pz = center
        ax.scatter(px, py, pz,
                   marker=marker,
                   c=color,
                   s=radius*50,  # Convert radius from um to dpi
                   depthshade=False,
                   cmap=cmap)
        all_x.append(px)
        all_y.append(py)
        all_z.append(pz)

    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    all_z = np.concatenate(all_z)

    # Work out the bounding box
    min_x = np.min(all_x)
    max_x = np.max(all_x)

    min_y = np.min(all_y)
    max_y = np.max(all_y)

    min_z = np.min(all_z)
    max_z = np.max(all_z)

    range_x = max_x - min_x
    range_y = max_y - min_y
    range_z = max_z - min_z

    range_max = max([range_x, range_y, range_z])
    center_x = (min_x + max_x)/2
    center_y = (min_y + max_y)/2
    center_z = (min_z + max_z)/2

    ax.set_xlim([center_x - range_max/2, center_x+range_max/2])
    ax.set_ylim([center_y - range_max/2, center_y+range_max/2])
    ax.set_zlim([center_z - range_max/2, center_z+range_max/2])

    if title is not None:
        fig.suptitle(title)

    if add_colorbar and mappable is not None:
        plt.colorbar(mappable, ax=ax, fraction=0.15, pad=0.05)

    if outfile is None:
        plt.show()
    else:
        outfile.parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(str(outfile), transparent=True)
        plt.close()
