import alerce
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# ---------------------------------------------------------
# 1. Data Fetching & Preparation
# ---------------------------------------------------------
alerce_client = alerce.core.Alerce()
oid = "ZTF18accedau"
col_names = [
    "mjd",
    "flux_2_20",
    "err_2_20",
    "flux_2_4",
    "err_2_4",
    "flux_4_10",
    "err_4_10",
    "flux_10_20",
    "err_10_20",
]


def mag_to_flux(mag, mag_err):
    flux = 3631 * 10 ** (-0.4 * mag) * 1000  # mJy
    flux_err = flux * (mag_err / 1.0857)
    return flux, flux_err


def fred(t, A1, A2, tau_rise, tau_decay, t_peak):
    """Fast Rise Exponential Decay (FRED) profile."""
    return np.where(
        t < t_peak,
        A1 * np.exp((t - t_peak) / tau_rise),
        A2 * np.exp(-(t - t_peak) / tau_decay),
    )


def reprocessing_model(x_flux, C, beta):
    """Reprocessed optical flux as a power law of the X-ray flux: F_opt = C * F_X^beta."""
    return C * np.power(x_flux, beta)


maxi_conv = 2.4e-8 / 3.3  # erg/cm^2/s per (ph/s/cm^2)

# Load MAXI X-ray data
df_xray = pd.read_csv(
    "GX339_4/gx339_4_maxi_data.csv", sep=" ", header=None, names=col_names
)
df_xray["X_FLUX"] = df_xray["flux_2_20"] * maxi_conv
df_xray["X_FLUX_ERR"] = df_xray["err_2_20"] * maxi_conv

# Fetch ZTF Optical data
"""
response_data = alerce_client.query_detections(oid, format="json")
df = pd.DataFrame(response_data)

df_g = df[df["fid"] == 1][["mjd", "magpsf", "sigmapsf"]].reset_index(drop=True)
df_r = df[df["fid"] == 2][["mjd", "magpsf", "sigmapsf"]].reset_index(drop=True)

df_g["flux"], df_g["flux_err"] = mag_to_flux(df_g["magpsf"], df_g["sigmapsf"])
df_r["flux"], df_r["flux_err"] = mag_to_flux(df_r["magpsf"], df_r["sigmapsf"])"""
df_optical = pd.read_csv(
    "GX339_4/gx339_4_yale_data.csv", sep=r"\s+", header=None, names=["mjd", "V_mag", "V_mag_err", "V_flux", "V_flux_err", "I_mag", "I_mag_err", "I_flux", "I_flux_err", "J_mag", "J_mag_err", "J_flux", "J_flux_err", "H_mag", "H_mag_err", "H_flux", "H_flux_err"]
)
df_optical = df_optical[(df_optical != 999.0).all(axis=1)]
df_g = pd.DataFrame({
    "mjd": df_optical["mjd"]+ 2450000 - 2400000.5,
    "flux": df_optical["V_flux"],
    "flux_err": df_optical["V_flux_err"]
})
df_r = pd.DataFrame({
    "mjd": df_optical["mjd"]+ 2450000 - 2400000.5,
    "flux": df_optical["I_flux"],
    "flux_err": df_optical["I_flux_err"]
})

# Filter out zero or negative fluxes to prevent fitting errors
df_xray = (
    df_xray.dropna(subset=["mjd", "X_FLUX"])
    .query("X_FLUX > 0")
    .sort_values("mjd")
)

# ---------------------------------------------------------
# 2. Plotting Setup (Dual Axes)
# ---------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(11, 6))

# Left Axis: X-ray Data
(line_xray_data,) = ax1.plot(
    df_xray["mjd"],
    df_xray["X_FLUX"],
    color="#6baed6",
    marker="o",
    linestyle="none",
    label="MAXI (X-ray data)",
    zorder=2,
)
ax1.set_xlabel("MJD", fontsize=11)
ax1.set_ylabel("X-ray Flux ($erg/cm^2/s$)", color="#08306b", fontsize=11)
ax1.tick_params(axis="y", labelcolor="#08306b")

# Right Axis: ZTF Optical Data
ax2 = ax1.twinx()
line_g = ax2.errorbar(
    df_g["mjd"],
    df_g["flux"],
    yerr=df_g["flux_err"],
    fmt="s",
    color="#15a508",
    ecolor="#15a508",
    capsize=2,
    markersize=4,
    linestyle="none",
    label="V-band",
    zorder=3,
)
line_r = ax2.errorbar(
    df_r["mjd"],
    df_r["flux"],
    yerr=df_r["flux_err"],
    fmt="s",
    color="#ca0020",
    ecolor="#ca0020",
    capsize=2,
    markersize=4,
    linestyle="none",
    label="I-band",
    zorder=3,
)
ax2.set_ylabel("Optical Flux (mJy)", color="#e66101", fontsize=11)
ax2.tick_params(axis="y", labelcolor="#e66101")

# Empty line objects for the fitted curves (updated dynamically)
(line_xray_model,) = ax1.plot(
    [],
    [],
    color="#08306b",
    linewidth=2,
    label="FRED X-ray model",
    zorder=4,
)
(line_opt_model,) = ax2.plot(
    [],
    [],
    color="#800000",
    linewidth=2,
    linestyle="--",
    label=r"Reprocessed Optical model ($F_{opt} = C \cdot F_{X}^{\beta}$)",
    zorder=5,
)

# Setup initial combined legend
handles = [line_xray_data, line_g, line_r, line_xray_model, line_opt_model]
labels = [h.get_label() for h in handles]
ax1.legend(handles, labels, loc="upper right", fontsize=9)
base_title = "Drag mouse horizontally over an outburst region to fit FRED model"
ax1.set_title(base_title, fontsize=12, fontweight="bold")


# ---------------------------------------------------------
# 3. SpanSelector Callback for Interactive Curve Fitting
# ---------------------------------------------------------
def onselect(xmin, xmax):
    # Slice X-ray data within the selected interval
    mask = (df_xray["mjd"] >= xmin) & (df_xray["mjd"] <= xmax)
    x_sub = df_xray.loc[mask, "mjd"].values
    y_sub = df_xray.loc[mask, "X_FLUX"].values

    if len(x_sub) < 5:
        print("Selection too small! Please select a wider region.")
        return

    # Initial parameter guess [A1, A2, tau_rise, tau_decay, t_peak]
    t_peak_guess = x_sub[np.argmax(y_sub)]
    max_flux_guess = np.max(y_sub)
    p0 = [max_flux_guess, max_flux_guess, 5.0, 15.0, t_peak_guess]
    bounds = (
        [0, 0, 0.1, 0.1, xmin],
        [np.inf, np.inf, 100, 200, xmax],
    )

    try:
        popt, _ = curve_fit(fred, x_sub, y_sub, p0=p0, bounds=bounds)

        # Dense array for smooth plotting across the selected span
        t_dense = np.linspace(xmin, xmax, 500)
        xray_fitted = fred(t_dense, *popt)

        # -------------------------------------------------------------
        # Fit the reprocessing power law F_opt = C * F_X^beta
        # beta (and C) are now free parameters determined by curve_fit,
        # using the X-ray model evaluated AT the optical observation
        # times as the predictor, and the observed optical flux as the
        # target. Prefer g-band; fall back to r-band if g has no
        # coverage in this window; combine both if both are present.
        # -------------------------------------------------------------
        g_mask = (df_g["mjd"] >= xmin) & (df_g["mjd"] <= xmax)
        r_mask = (df_r["mjd"] >= xmin) & (df_r["mjd"] <= xmax)

        if len(g_mask) > len(r_mask):
            opt_mjd = df_g.loc[g_mask, "mjd"].values
            opt_flux = df_g.loc[g_mask, "flux"].values
        elif len(r_mask) >= len(g_mask):
            opt_mjd = df_r.loc[r_mask, "mjd"].values
            opt_flux = df_r.loc[r_mask, "flux"].values

        if len(opt_mjd) < 3:
            print("Not enough optical points in this window to fit beta;"
                  " showing X-ray fit only.")
            line_xray_model.set_data(t_dense, xray_fitted)
            line_opt_model.set_data([], [])
            ax1.set_ylim(-0.05 * np.max(xray_fitted), np.max(xray_fitted) * 1.1)
            fig.canvas.draw_idle()
            return

        # X-ray model flux evaluated at the same times as the optical data
        xray_at_opt_times = fred(opt_mjd, *popt)

        # Power law needs a strictly positive base
        valid = xray_at_opt_times > 0
        opt_mjd = opt_mjd[valid]
        opt_flux = opt_flux[valid]
        xray_at_opt_times = xray_at_opt_times[valid]

        if len(xray_at_opt_times) < 3:
            print("Not enough valid overlapping points to fit beta;"
                  " showing X-ray fit only.")
            line_xray_model.set_data(t_dense, xray_fitted)
            line_opt_model.set_data([], [])
            ax1.set_ylim(-0.05 * np.max(xray_fitted), np.max(xray_fitted) * 1.1)
            fig.canvas.draw_idle()
            return

        # Initial guess: beta=0.5, C chosen so the model roughly matches
        # the median observed optical flux at the median X-ray flux.
        beta0 = 0.5
        C0 = np.median(opt_flux) / (np.median(xray_at_opt_times) ** beta0 + 1e-30)
        C0 = max(C0, 1e-10)

        try:
            reproc_popt, _ = curve_fit(
                reprocessing_model,
                xray_at_opt_times,
                opt_flux,
                p0=[C0, beta0],
                bounds=([0, 0], [np.inf, 5]),
                maxfev=10000,
            )
        except RuntimeError as e:
            print(f"Beta fit failed, falling back to beta=0.5: {e}")
            reproc_popt = [C0, beta0]

        C_fit, beta_fit = reproc_popt

        opt_fitted = reprocessing_model(xray_fitted, C_fit, beta_fit)

        line_xray_model.set_data(t_dense, xray_fitted)
        line_opt_model.set_data(t_dense, opt_fitted)
        line_opt_model.set_label(
            rf"Reprocessed Optical model ($C$={C_fit:.3g}, $\beta$={beta_fit:.3f})"
        )

        handles = [line_xray_data, line_g, line_r, line_xray_model, line_opt_model]
        labels = [h.get_label() for h in handles]
        ax1.legend(handles, labels, loc="upper right", fontsize=9)

        ax1.set_title(
            f"{base_title}\nFitted beta = {beta_fit:.3f}, C = {C_fit:.3g}",
            fontsize=12,
            fontweight="bold",
        )

        ax1.set_ylim(-0.05 * np.max(xray_fitted), np.max(xray_fitted) * 1.1)
        opt_peak_display = max(np.max(opt_fitted), np.max(opt_flux))
        ax2.set_ylim(-0.05 * opt_peak_display, opt_peak_display * 1.1)

        fig.canvas.draw_idle()

    except Exception as e:
        print(f"Fit failed: {e}")


# Initialize SpanSelector
span = SpanSelector(
    ax1,
    onselect,
    direction="horizontal",
    useblit=True,
    props=dict(alpha=0.25, facecolor="gray"),
    interactive=True,
    drag_from_anywhere=True,
)

plt.tight_layout()
plt.show()