import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
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
    b_exp = False
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
    percentile_lines = tenax.inverse_magnitude_model(F_phat, eT, qs, b_exp = b_exp)
    plt.scatter(T, P, s=1, color=obscol, label="Observations")
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

def TNX_FIG_temp_model(
    T,
    g_phat,
    beta,
    eT,
    obscol="r",
    valcol="b",
    obslabel="Observations",
    vallabel="Temperature model g(T)",
    xlimits=[-15, 30],
    ylimits=[0, 0.06],
    method="norm",
):
    """
    Plots the observational and model temperature pdf

    Parameters
    ----------
    T : numpy.ndarray
        Array of observed temperatures.
    g_phat : numpy.ndarray
        [mu, sigma] of temperature distribution.
    beta : float
        value of beta in generalised normal distribution.
    eT : numpy.ndarray
        x (temperature) values to produce distribution.
    obscol : string, optional
        color to plot observations. The default is 'r'.
    valcol : string, optional
        color to plot magnitude model. The default is 'b'.
    obslabel : string, optional
        Label for observations. The default is 'observations'.
    vallabel : string, optional
        Label for model plot. The default is 'temperature model g(T)'.
    xlimits : list, optional
        limits for the x axis [lower_x_limit, upper_x_limit]. The default is [-15,30].
    ylimits : list, optional
        limits for the y axis [lower_y_limit, upper_y_limit]. The default is [0,0.06].

    Returns
    -------
    hist : numpy.ndarray
        pdf values of observed distribution.
    pdf_values : numpy.ndarray
        pdf values of fitted model.

    """

    # Plot empirical PDF of T
    eT_edges = np.concatenate(
        [np.array([eT[0] - (eT[1] - eT[0]) / 2]), (eT + (eT[1] - eT[0]) / 2)]
    )  # convert bin centres into bin edges
    hist, bin_edges = np.histogram(T, bins=eT_edges, density=True)
    plt.plot(eT, hist, "--", color=obscol, label=obslabel)

    # Plot analytical PDF of T (validation)
    if method == "skewnorm":
        pdf_values = tenax.skewnorm.pdf(eT, *g_phat)
    elif method == "norm":
        pdf_values = tenax.gen_norm_pdf(eT, g_phat[0], g_phat[1], beta)

    plt.plot(eT, pdf_values, "-", color=valcol, label=vallabel)

    # Set plot parameters
    # ax.set_xlim(Tlims)
    plt.xlabel("T [°C]", fontsize=14)
    plt.ylabel("pdf", fontsize=14)
    plt.ylim(ylimits[0], ylimits[1])
    plt.xlim(xlimits[0], xlimits[1])
    plt.legend(
        fontsize=8
    )  # NEED TO SET LOCATION OF THIS, maybe fontsize is too small as well
    plt.tick_params(axis="both", which="major", labelsize=14)

    return hist, pdf_values

def TNX_FIG_valid(
    AMS: pd.DataFrame,
    RP: list,
    RL: np.ndarray,
    smev_RL: Union[np.ndarray, list] = [],
    RL_unc: Union[np.ndarray, list] = [],
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

    if np.size(RL_unc) != 0:
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


def TNX_FIG_scaling(
    P,
    T,
    P_mc,
    T_mc,
    F_phat,
    niter_smev,
    eT,
    iTs,
    qs=[0.99],
    obscol="r",
    valcol="b",
    xlimits=[-15, 30],
    ylimits=[0.4, 1000],
):
    """
    Plots figure 5.

    Parameters
    ----------
    P : numpy.ndarray
        precipitation values
    T : numpy.ndarray
        temperature values
    P_mc : numpy.ndarray
        Monte Carlo generated precipitation values.
    T_mc : numpy.ndarray
        Monte Carlo generated temperature values.
    F_phat : numpy.ndarray
        distribution values. F_phat = [kappa_0,b,lambda_0,a].
    niter_smev : int
        Number of iterations for uncertainty for the SMEV model .
    eT : numpy.ndarray
        x (temperature) values to produce distribution for magnitude model.
    iTs : numpy.ndarray
        x (temperature) values to produce distribution for quantile regression, binning, and TENAX.
    qs : list
        percentiles to calculate W.
    obscol : string, optional
        color code to plot observations. The default is 'r'.
    valcol : string, optional
        color code to plot model. The default is 'b'.
    xlimits : list, optional
        [min_x,max_x]. x limits to plot. The default is [-15,30].
    ylimits : list, optional
        [min_y,max_y]. y limits to plot. The default is [0.4,1000].

    Returns
    -------
    scaling_rate : float
        scaling rate of

    """
    percentile_lines = tenax.inverse_magnitude_model(F_phat, eT, qs)
    scaling_rate_W = (np.exp(F_phat[3]) - 1) * 100

    # TODO: this doesn't seem quite right ... uncertainty is way off compared to paper
    # This is due to number of samples in bootstrapping, somehow this influenced unc of qs much more than in MATLAB
    qhat, qhat_unc = TNX_obs_scaling_rate(P, T, qs[0], 1000)
    scaling_rate_q = (np.exp(qhat[1]) - 1) * 100

    # quantile regression uncertainties
    q_reg_full_unc = np.zeros([len(iTs), 1000])
    for i in np.arange(0, len(iTs)):
        q_reg_full_unc[i, :] = np.exp(qhat_unc[0, :]) * np.exp(iTs[i] * qhat_unc[1, :])

    q_up = np.quantile(q_reg_full_unc, 0.95, axis=1)
    q_low = np.quantile(q_reg_full_unc, 0.05, axis=1)

    plt.figure(figsize=(5, 5))
    plt.scatter(T, P, s=1.5, color=obscol, alpha=0.3, label="Observations")
    plt.plot(
        iTs[0:-7],
        np.exp(qhat[0]) * np.exp(iTs[0:-7] * qhat[1]),
        "--k",
        label="Quantile regression method",
    )  # need uncertainty on this too...
    plt.fill_between(
        iTs[0:-7], q_low[0:-7], q_up[0:-7], color="k", alpha=0.2
    )  # quantile regression uncertainty

    ############################################################### PUT THIS ELSEWHERE
    T_mc_bins = np.reshape(T_mc, [np.size(T), niter_smev])
    P_mc_bins = np.reshape(P_mc, [np.size(P), niter_smev])

    qperc_model = np.zeros([np.size(iTs), 1000])
    qperc_obs = np.zeros([np.size(iTs), 1000])

    for nit in range(niter_smev):
        for i in range(np.size(iTs) - 1):
            tmpP = P_mc_bins[:, nit]
            mask_model = (T_mc_bins[:, nit] > iTs[i]) & (
                T_mc_bins[:, nit] <= iTs[i + 1]
            )
            if np.any(mask_model):
                qperc_model[i, nit] = np.quantile(
                    tmpP[mask_model], qs[0]
                )  # binning monte carlos to get TENAX model

            mask_obs = (T > iTs[i]) & (T <= iTs[i + 1])
            if np.any(mask_obs):
                qperc_obs[i] = np.quantile(P[mask_obs], qs[0])  # binning observations

    qperc_obs_med = np.median(qperc_obs, axis=1)
    qperc_model_med = np.median(qperc_model, axis=1)

    qperc_model_up = np.quantile(qperc_model, 0.95, axis=1)
    qperc_model_low = np.quantile(qperc_model, 0.05, axis=1)

    #####################################################################################

    # plot uncertainty
    plt.fill_between(
        iTs[1:-6] + (iTs[2] - iTs[1]) / 2,
        qperc_model_low[1:-6],
        qperc_model_up[1:-6],
        color="m",
        alpha=0.2,
    )  # TENAX

    plt.plot(
        iTs[1:-7] + (iTs[2] - iTs[1]) / 2,
        qperc_obs_med[1:-7],
        "-xr",
        label="Binning method",
    )  # don't really know why we cut off at the end like this
    plt.plot(
        iTs[1:-6] + (iTs[2] - iTs[1]) / 2,
        qperc_model_med[1:-6],
        "-om",
        label="The TENAX model",
    )

    n = 0
    while n < np.size(qs):
        plt.plot(eT, percentile_lines[n], color=valcol, label="Magnitude model W(x,T)")
        n = n + 1

    plt.yscale("log")
    plt.ylim(ylimits[0], ylimits[1])
    plt.xlim(xlimits[0], xlimits[1])
    plt.legend(title=str(qs[0] * 100) + "th percentile lines computed by:")

    return scaling_rate_W, scaling_rate_q


def TNX_obs_scaling_rate(P, T, qs, niter):
    """
    Calculate quantile regression parameters.

    Parameters
    ----------
    P : numpy.ndarray
        precipitation values
    T : numpy.ndarray
        temperature values
    qs : float
        percentile.

    Returns
    -------
    qhat : numpy.ndarray
        [intercept, scaling rate].

    """
    T = sm.add_constant(T)  # Add a constant (intercept) term
    model = sm.QuantReg(np.log(P), T)
    qhat = model.fit(q=qs, max_iter=10000).params

    qhat_unc = np.zeros([2, niter])
    for iter in np.arange(0, niter):
        rr = np.random.randint(0, len(T), size=(niter))
        model = sm.QuantReg(np.log(P[rr]), T[rr])
        qhat_unc[:, iter] = model.fit(q=qs, max_iter=10000).params

    return qhat, qhat_unc


