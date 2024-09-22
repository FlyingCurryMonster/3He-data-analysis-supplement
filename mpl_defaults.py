import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pandas as pd

mpl.rcdefaults()

mpl.rcParams['font.size'] = 24  # change the size of the font in every figure
mpl.rcParams['font.family'] = 'Liberation Sans'  # font Arial in every figure

mpl.rcParams['axes.labelsize'] = 24.
mpl.rcParams['axes.labelpad'] = 8

mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 2

mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24
mpl.rcParams['xtick.direction'] = "in"
mpl.rcParams['ytick.direction'] = "in"
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['xtick.major.width'] = 0.6
mpl.rcParams['ytick.major.width'] = 0.6

mpl.rcParams['xtick.major.pad'] = 7
mpl.rcParams['xtick.major.size'] = 8
mpl.rcParams['xtick.minor.pad'] = 7
mpl.rcParams['xtick.minor.size'] = 4

mpl.rcParams['ytick.major.pad'] = 8
mpl.rcParams['ytick.major.size'] = 8
mpl.rcParams['ytick.minor.pad'] = 8
mpl.rcParams['ytick.minor.size'] = 4


mpl.rcParams['axes.linewidth'] = 0.6  # thickness of the axes lines


mpl.rcParams['figure.figsize'] = 9.2, 5.6
mpl.rcParams['figure.subplot.left'] = 0.18
mpl.rcParams['figure.subplot.right'] = 0.82
mpl.rcParams['figure.subplot.bottom'] = 0.18
mpl.rcParams['figure.subplot.top'] = 0.95

mpl.rcParams['legend.frameon'] = False

mpl.rcParams['pdf.fonttype'] = 42
# Output Type 3 (Type3) or Type 42 (TrueType),
# TrueType allows editing the text in illustrator

# Example on gradient color schemes
# cmap = mpl.cm.get_cmap("plasma", int(len(fig1_pressures)*1.2))
# colors = cmap(np.arange(int(1.2*len(fig1_pressures))))


def autoscale_y(ax, margin=0.1):
    """This function rescales the y-axis based on the data that is visible
    given the current xlim of the axis.
    ax -- a matplotlib axes object
    margin -- the fraction of the total height of the y-data to pad the upper
    and lower ylims"""

    def get_bottom_top(line):
        xd = line.get_xdata()
        yd = line.get_ydata()
        lo, hi = ax.get_xlim()
        # y_displayed = yd[((xd>lo) & (xd<hi))]
        if len(xd) == 2 and xd[0] == 0.0 and xd[1] == 1.0:
            y_displayed = yd  # special case to handle axhline
        else:
            y_displayed = yd[((xd >= lo) & (xd <= hi))]
        h = np.max(y_displayed) - np.min(y_displayed)
        bot = np.min(y_displayed)-margin*h
        top = np.max(y_displayed)+margin*h
        return bot, top

    lines = ax.get_lines()
    bot, top = np.inf, -np.inf

    for line in lines:
        new_bot, new_top = get_bottom_top(line)
        if new_bot < bot:
            bot = new_bot
        if new_top > top:
            top = new_top

    ax.set_ylim(bot, top)


def lorentzian_plot(df: pd.DataFrame, fcol='f', Xcol='X', Ycol='Y', Acol='A',
                    Xfit_col='X_fit', Yfit_col='Y_fit'):
    fig = plt.figure(figsize=np.array([14, 12]))

    outer = gridspec.GridSpec(2, 1)
    gs1 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[0])
    gs2 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[1],
                                           hspace=0, wspace=0.4)
    fig.subplots_adjust(hspace=0.3)

    # gs = gridspec.GridSpec(3,3)
    axb = plt.subplot(gs2[0, 0])
    axc = plt.subplot(gs2[1, 0])
    axd = plt.subplot(gs2[:, 1])
    axa = plt.subplot(gs1[0])
    # axd = plt.subplot(gs[0, :])

    data_plot_param = {'marker': 'o', 'markersize': 5, 'linewidth': 0}
    axa.plot(*df[[fcol, Acol]].values.T, **data_plot_param, label='data')
    axb.plot(*df[[fcol, Xcol]].values.T, **data_plot_param)
    axc.plot(*df[[fcol, Ycol]].values.T, **data_plot_param)
    axd.plot(*df[[Xcol, Ycol]].values.T, **data_plot_param)

    fit_plot_param = {'linestyle': '--', 'linewidth': 3}
    axa.plot(df[fcol],
             np.sqrt(np.diag(df[[Xfit_col, Yfit_col]].dot(
                 df[[Xfit_col, Yfit_col]].T))),
             **fit_plot_param, label='Lorentz fit')
    axb.plot(*df[[fcol, Xfit_col]].values.T, **fit_plot_param)
    axc.plot(*df[[fcol, Yfit_col]].values.T, **fit_plot_param)
    axd.plot(*df[[Xfit_col, Yfit_col]].values.T, **fit_plot_param)
    # axd.plot(*df[[fcol, 'R_fit']].values.T, **fit_plot_param,
    #          label='Lorentz fit')

    axd.set_aspect('equal')

    # figures settings
    axb.set_xticklabels([])
    axc.set_xlabel('frequency (Hz)')
    axb.set_ylabel(r'$X$ (mV)')
    axc.set_ylabel(r'$Y$ (mV)')

    axa.set_xlabel('frequency (Hz)')
    axa.set_ylabel(r'$R$ (mV)')
    # axc.set_yticklabels(axc.get_xticklabels())

    axd.set_xlabel(axb.get_ylabel())
    axd.set_ylabel(axc.get_ylabel(), labelpad=1)
    axa.legend()

    return fig, [axa, axb, axc, axd]
