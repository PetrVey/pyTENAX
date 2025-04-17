import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Union

from pyTENAX import tenax


def TNX_FIG_magn_model(
    P: np.ndarray,
    T: np.ndarray,
    F_phat: np.ndarray,
    thr: float,
    eT: np.ndarray,
    qs: list,
    obscol="r",
    valcol="b",
    xlimits: list = [-12, 30],
    ylimits: list = [0.1, 1000],
) -> None:
    """Plots Figure 2a of Marra et al. (2024), the observed T-P pairs and the W model percentiles.

    Args:
        P (np.ndarray): Precipitation values.
        T (np.ndarray): Temperature values.
        F_phat (np.ndarray): Distribution values. F_phat = [kappa_0,b,lambda_0,a]..
        thr (float): Precipitation threshold for left-censorig.
        eT (np.ndarray): x values to plot W model.
        qs (list): Percentiles to calculate W.
        obscol (str, optional): Color code to plot observations. Defaults to "r".
        valcol (str, optional): Color code to plot model. Defaults to "b".
        xlimits (list, optional): x limits of plot. Defaults to [-12, 30].
        ylimits (list, optional): y limits of plot. Defaults to [0.1, 1000].
    """
    percentile_lines = tenax.inverse_magnitude_model(F_phat, eT, qs)
    plt.scatter(T, P, s=1, color=obscol, label="observations")
    plt.plot(
        eT,
        [thr] * np.size(eT),
        "--",
        alpha=0.5,
        color="k",
        label="Left censoring threshold",
    )  # plot threshold

    # first one outside loop so can be in legend
    n = 0
    plt.plot(eT, percentile_lines[n], label="Magnitude model W(x,T)", color=valcol)
    plt.text(
        eT[-1], percentile_lines[n][-1], str(qs[n] * 100) + "th", ha="left", va="center"
    )
    n = 1
    while n < np.size(qs):
        plt.plot(eT, percentile_lines[n], color=valcol)  # ,label = str(qs[n]),
        plt.text(
            eT[-1],
            percentile_lines[n][-1],
            str(qs[n] * 100) + "th",
            ha="left",
            va="center",
        )
        n = n + 1

    plt.legend()
    plt.yscale("log")
    plt.ylim(ylimits[0], ylimits[1])
    plt.xlim(xlimits[0], xlimits[1])
    plt.xlabel("T [°C]")


def TNX_FIG_valid(
    AMS: pd.DataFrame,
    RP: list,
    RL: np.ndarray,
    smev_RL: Union[np.ndarray, list] = [],
    RL_unc=0,
    smev_RL_unc=0,
    TENAXcol="b",
    obscol_shape="g+",
    smev_colshape="--r",
    TENAXlabel="The TENAX model",
    obslabel="Observed annual maxima",
    smevlabel="The SMEV model",
    alpha=0.2,
    xlimits: list = [1, 200],
    ylimits: list = [0, 50],
) -> None:
    """Plots figure 4 of Marra et al. (2024).

    Args:
        AMS (pd.DataFrame): Dataframe containing annual maxima.
        RP (list): Return periods to plot.
        RL (np.ndarray): Return levels calculated by TENAX.
        smev_RL (Union[np.ndarray, list], optional): Return levels calculated by SMEV. Defaults to [].
        RL_unc (int, optional): Uncertainty of return levels calculated by TENAX. Only relevant if `smev_RL` is provided. Defaults to 0.
        smev_RL_unc (int, optional): Uncertainty of return levels calculated by SMEV. Only relevant if `smev_RL` is provided. Defaults to 0.
        TENAXcol (str, optional): Linestyle for TENAX data to use in plot. Defaults to "b".
        obscol_shape (str, optional): Linestyle for annual maxima data to use in plot. Defaults to "g+".
        smev_colshape (str, optional): Linestyle for SMEV data to use in plot. Defaults to "--r".
        TENAXlabel (str, optional): Label for TENAX data to use in plot. Defaults to "The TENAX model".
        obslabel (str, optional): Label for annual maxima observation data to use in plot. Defaults to "Observed annual maxima".
        smevlabel (str, optional): Label for SMEV data to use in plot. Defaults to "The SMEV model".
        alpha (float, optional): Transparency to use in plot. Defaults to 0.2.
        xlimits (list, optional): x limits of plot. Defaults to [1, 200].
        ylimits (list, optional): y limits of plot. Defaults to [0, 50].
    """
    AMS_sort = AMS.sort_values(by=["AMS"])["AMS"]
    plot_pos = np.arange(1, np.size(AMS_sort) + 1) / (1 + np.size(AMS_sort))
    eRP = 1 / (1 - plot_pos)
    if np.size(smev_RL) != 0:
        # calculate uncertainty bounds. between 5% and 95%
        smev_RL_up = np.quantile(smev_RL_unc, 0.95, axis=0)
        smev_RL_low = np.quantile(smev_RL_unc, 0.05, axis=0)
        # plot uncertainties
        plt.fill_between(
            RP, smev_RL_low, smev_RL_up, color=smev_colshape[-1], alpha=alpha
        )  # SMEV

    if RL_unc.size > 0:
        RL_up = np.quantile(RL_unc, 0.95, axis=0)
        RL_low = np.quantile(RL_unc, 0.05, axis=0)
        plt.fill_between(RP, RL_low, RL_up, color=TENAXcol, alpha=alpha)  # TENAX
    
    plt.plot(RP, RL, TENAXcol, label=TENAXlabel)  # plot TENAX return levels
    plt.plot(eRP, AMS_sort, obscol_shape, label=obslabel)  # plot observed return levels
    if np.size(smev_RL) != 0:
        plt.plot(RP, smev_RL, smev_colshape, label=smevlabel)  # plot SMEV return lvls

    plt.xscale("log")
    plt.xlabel("return period (years)")
    plt.legend()
    plt.xlim(xlimits[0], xlimits[1])
    plt.ylim(ylimits[0], ylimits[1])
