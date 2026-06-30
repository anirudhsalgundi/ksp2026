import os
import re
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import datetime

warnings.filterwarnings("ignore")

SHORTLISTED_DIR = r"C:\Users\abhin\Documents\ksp2026\Shortlisted_Data"
FRED_DIR        = r"C:\Users\abhin\Documents\ksp2026\FRED_Data"

BANDS = [
    ("flux_2_20",  "2-20 keV",  "steelblue"),
    ("flux_2_4",   "2-4 keV",   "darkorange"),
    ("flux_4_10",  "4-10 keV",  "forestgreen"),
    ("flux_10_20", "10-20 keV", "crimson"),
]

def fred_model(t, A, t0, tau_r, tau_d):
    """
    FRED: Fast Rise Exponential Decay
        F(t) = A * exp(-(t-t0)/tau_d) * (1 - exp(-(t-t0)/tau_r))
    """
    dt = t - t0
    return np.where(
        dt > 0,
        A * np.exp(-dt / tau_d) * (1.0 - np.exp(-dt / tau_r)),
        0.0,
    )


def norris_model(t, A, t0, tau1, tau2):
    dt = t - t0
    safe = dt > 0
    result = np.zeros_like(t, dtype=float)
    result[safe] = A * np.exp(-tau1 / dt[safe] - dt[safe] / tau2)
    return result


def aic(n_params, n_data, chi2):
    return chi2 + 2 * n_params + (2 * n_params * (n_params + 1)) / max(n_data - n_params - 1, 1)


def _fit_single(model_fn, p0, bounds, t, f, sig):
    try:
        popt, pcov = curve_fit(
            model_fn, t, f,
            p0=p0, sigma=sig, absolute_sigma=True,
            bounds=bounds, maxfev=30000,
        )
        perr     = np.sqrt(np.diag(pcov))
        resid    = f - model_fn(t, *popt)
        chi2     = float(np.sum((resid / sig) ** 2))
        dof      = len(f) - len(popt)
        chi2_red = chi2 / dof if dof > 0 else np.inf
        aic_val  = aic(len(popt), len(f), chi2)
        return dict(popt=popt, perr=perr,
                    chi2=chi2, chi2_red=chi2_red, dof=dof, aic=aic_val)
    except Exception as exc:
        return {"error": str(exc)}


def fit_band(mjd, flux, err):

    mask = np.isfinite(flux) & np.isfinite(err) & (err > 0)
    if mask.sum() < 12:
        return None

    t   = mjd[mask]
    f   = flux[mask]
    sig = err[mask]

    peak_idx = int(np.argmax(f))
    A0       = max(f[peak_idx], 1e-6)
    t0_0     = t[peak_idx] - 5.0      # onset before peak
    t0_0     = np.clip(t0_0, t.min() - 50, t[peak_idx] - 0.5)

    # FRED: [A, t0, tau_r, tau_d]
    fred_res = _fit_single(
        fred_model,
        p0=[A0, t0_0, 3.0, 20.0],
        bounds=([0, t.min()-100, 0.1,  0.1 ],
                [np.inf, t[peak_idx], 300., 1000.]),
        t=t, f=f, sig=sig,
    )

    # Norris: [A, t0, tau1, tau2]
    # tau1 ~ rise^2 / decay, tau2 ~ decay; start with rough estimates
    norris_res = _fit_single(
        norris_model,
        p0=[A0 * 1.5, t0_0, 2.0, 20.0],
        bounds=([0, t.min()-100, 0.01, 0.1 ],
                [np.inf, t[peak_idx], 500., 1000.]),
        t=t, f=f, sig=sig,
    )

    # store raw data for plotting
    for r in (fred_res, norris_res):
        if r and "popt" in r:
            r["t_data"] = t
            r["f_data"] = f
            r["sig_data"] = sig

    # pick best by AIC (lower = better)
    fred_aic   = fred_res.get("aic",   np.inf) if fred_res   else np.inf
    norris_aic = norris_res.get("aic", np.inf) if norris_res else np.inf
    best = "fred" if fred_aic <= norris_aic else "norris"

    return {"fred": fred_res, "norris": norris_res, "best": best}


def find_maxi_files(directory):
    patterns = [
        os.path.join(directory, "*_maxi_lc.csv"),
        os.path.join(directory, "*_maxi.csv"),
        os.path.join(directory, "*_maxi_lc"),
        os.path.join(directory, "*_maxi"),
    ]
    found = {}
    for pat in patterns:
        for fp in glob.glob(pat):
            base = os.path.splitext(os.path.basename(fp))[0]
            key  = re.sub(r"(_maxi_lc|_maxi)$", "", base)
            if key not in found:
                found[key] = fp
    return found

def make_plot(source_key, df, all_band_results, out_path):
    mjd = df["mjd"].values

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    src_display = source_key.replace("_", " ")
    fig.suptitle(f"FRED & Norris Model Fits  --  {src_display}",
                 fontsize=13, fontweight="bold")

    for ax, (flux_col, band_label, colour) in zip(axes, BANDS):
        err_col = flux_col.replace("flux_", "err_")

        if flux_col not in df.columns:
            ax.set_title(f"{band_label}\n(no data)")
            ax.axis("off")
            continue

        flux = df[flux_col].values
        err  = df[err_col].values if err_col in df.columns else np.ones_like(flux) * 0.01

        mask  = np.isfinite(flux) & np.isfinite(err) & (err > 0)
        t_plt = mjd[mask]; f_plt = flux[mask]; e_plt = err[mask]

        ax.errorbar(t_plt, f_plt, yerr=e_plt,
                    fmt="o", ms=2.5, lw=0.6, alpha=0.5,
                    color=colour, label="Data", zorder=2)

        res = all_band_results.get(flux_col)

        if res is None:
            ax.text(0.5, 0.5, "Insufficient data",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color="gray")
        else:
            t_fine = np.linspace(t_plt.min(), t_plt.max(), 3000)
            best   = res.get("best", "fred")

            for model_key, fn, ls, lcolor, lbl in [
                ("fred",   fred_model,   "--", "black",  "FRED"),
                ("norris", norris_model, "-",  "purple", "Norris"),
            ]:
                r = res.get(model_key)
                if r and "popt" in r:
                    f_model = fn(t_fine, *r["popt"])
                    lw = 2.0 if model_key == best else 1.0
                    alpha = 1.0 if model_key == best else 0.55
                    label = f"{lbl} (best)" if model_key == best else lbl
                    ax.plot(t_fine, f_model, ls=ls, lw=lw, color=lcolor,
                            alpha=alpha, label=label, zorder=3)
                    # onset marker for best model
                    if model_key == best:
                        ax.axvline(r["popt"][1], color="gray",
                                   ls=":", lw=0.8, alpha=0.6)

            # annotation box with both model params
            info_lines = []
            fred_r   = res.get("fred")
            norris_r = res.get("norris")

            if fred_r and "popt" in fred_r:
                p = fred_r["popt"]; pe = fred_r["perr"]
                info_lines += [
                    "-- FRED --",
                    f"A     = {p[0]:.4f}+/-{pe[0]:.4f}",
                    f"t0    = {p[1]:.2f}+/-{pe[1]:.2f} MJD",
                    f"tau_r = {p[2]:.2f}+/-{pe[2]:.2f} d",
                    f"tau_d = {p[3]:.2f}+/-{pe[3]:.2f} d",
                    f"chi2/dof={fred_r['chi2_red']:.2f}  AIC={fred_r['aic']:.1f}",
                ]
            else:
                info_lines += ["-- FRED --", "fit failed"]

            info_lines.append("")

            if norris_r and "popt" in norris_r:
                p = norris_r["popt"]; pe = norris_r["perr"]
                info_lines += [
                    "-- Norris --",
                    f"A     = {p[0]:.4f}+/-{pe[0]:.4f}",
                    f"t0    = {p[1]:.2f}+/-{pe[1]:.2f} MJD",
                    f"tau1  = {p[2]:.2f}+/-{pe[2]:.2f} d",
                    f"tau2  = {p[3]:.2f}+/-{pe[3]:.2f} d",
                    f"chi2/dof={norris_r['chi2_red']:.2f}  AIC={norris_r['aic']:.1f}",
                ]
            else:
                info_lines += ["-- Norris --", "fit failed"]

            info_lines.append(f"Best: {best.upper()}")

            ax.text(0.97, 0.97, "\n".join(info_lines),
                    transform=ax.transAxes, fontsize=6.8,
                    va="top", ha="right", family="monospace",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

        ax.set_title(band_label, fontsize=11)
        ax.set_xlabel("MJD", fontsize=9)
        ax.set_ylabel("Flux (ph/cm2/s)", fontsize=9)
        ax.legend(fontsize=7.5, loc="upper left")
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.2, lw=0.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot]  saved -> {out_path}")


def _fmt_model_block(label, r):
    lines = [f"  [{label}]"]
    if r is None:
        lines.append("    SKIPPED (insufficient data)")
    elif "error" in r:
        lines.append(f"    FAILED: {r['error']}")
    else:
        p  = r["popt"]
        pe = r["perr"]
        if label == "FRED":
            lines += [
                f"    A      = {p[0]:.6e}  +/-  {pe[0]:.6e}  [ph/cm2/s]",
                f"    t0     = {p[1]:.4f}   +/-  {pe[1]:.4f}   [MJD]",
                f"    tau_r  = {p[2]:.4f}   +/-  {pe[2]:.4f}   [days]  (rise e-fold)",
                f"    tau_d  = {p[3]:.4f}   +/-  {pe[3]:.4f}   [days]  (decay e-fold)",
            ]
        else:  # Norris
            t_peak_rel = np.sqrt(p[2] * p[3]) if p[2] > 0 and p[3] > 0 else float("nan")
            lines += [
                f"    A      = {p[0]:.6e}  +/-  {pe[0]:.6e}  [ph/cm2/s]",
                f"    t0     = {p[1]:.4f}   +/-  {pe[1]:.4f}   [MJD]",
                f"    tau1   = {p[2]:.4f}   +/-  {pe[2]:.4f}   [days]  (rise sharpness)",
                f"    tau2   = {p[3]:.4f}   +/-  {pe[3]:.4f}   [days]  (decay timescale)",
                f"    t_peak = t0 + sqrt(tau1*tau2) = t0 + {t_peak_rel:.3f} days",
            ]
        lines += [
            f"    chi2   = {r['chi2']:.4f}",
            f"    dof    = {r['dof']}",
            f"    chi2/dof = {r['chi2_red']:.4f}",
            f"    AIC    = {r['aic']:.4f}",
        ]
    return lines


def write_log(source_key, all_band_results, log_path, mode="a"):
    sep  = "=" * 72
    sep2 = "-" * 50

    lines = [
        sep,
        f"SOURCE : {source_key}",
        f"TIME   : {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
        sep,
        "",
        "Models",
        "  FRED  : F(t) = A*exp(-(t-t0)/tau_d)*(1-exp(-(t-t0)/tau_r))  [t>t0]",
        "  Norris: F(t) = A*exp(-tau1/(t-t0) - (t-t0)/tau2)             [t>t0]",
        "  Both  = 0 for t <= t0",
        "  Best model chosen by AIC (corrected, lower = better)",
        "",
    ]

    for flux_col, band_label, _ in BANDS:
        lines.append(sep2)
        lines.append(f"  Band: {band_label}  ({flux_col})")
        lines.append(sep2)
        res = all_band_results.get(flux_col)
        if res is None:
            lines.append("  SKIPPED (column missing or insufficient data)")
        else:
            lines += _fmt_model_block("FRED",   res.get("fred"))
            lines.append("")
            lines += _fmt_model_block("Norris", res.get("norris"))
            lines.append(f"  >>> Best fit: {res.get('best','?').upper()}")
        lines.append("")

    lines += [sep, ""]

    with open(log_path, mode, encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"  [log]   written -> {log_path}")
def main():
    os.makedirs(FRED_DIR, exist_ok=True)
    master_log = os.path.join(FRED_DIR, "all_sources_summary.log")

    # initialise master log
    with open(master_log, "w", encoding="utf-8") as fh:
        fh.write(
            f"FRED + Norris Batch Log\n"
            f"Started: {datetime.utcnow():%Y-%m-%d %H:%M:%S} UTC\n"
            + "=" * 72 + "\n\n"
        )

    files = find_maxi_files(SHORTLISTED_DIR)
    if not files:
        print(f"[ERROR] No MAXI CSV files found in:\n  {SHORTLISTED_DIR}")
        return

    print(f"Found {len(files)} source(s): {', '.join(sorted(files))}\n")

    for source_key in sorted(files):
        fp = files[source_key]
        print(f"\n{'='*60}")
        print(f"Source : {source_key}")
        print(f"File   : {fp}")

        try:
            df = pd.read_csv(fp)
        except Exception as exc:
            print(f"  [SKIP] Read error: {exc}")
            continue

        # Normalize: strip whitespace, lowercase everything
        df.columns = [c.strip().lower() for c in df.columns]
        if "mjd" not in df.columns:
            print(f"  [SKIP] No MJD column. Got: {df.columns.tolist()}")
            continue

        mjd = df["mjd"].values
        all_band_results = {}

        for flux_col, band_label, _ in BANDS:
            if flux_col not in df.columns:
                all_band_results[flux_col] = None
                continue
            err_col = flux_col.replace("flux_", "err_")
            err = df[err_col].values if err_col in df.columns else np.ones(len(mjd)) * 0.01

            print(f"  {band_label:12s} ...", end=" ", flush=True)
            res = fit_band(mjd, df[flux_col].values, err)
            all_band_results[flux_col] = res

            if res:
                fred_ok   = "popt" in (res.get("fred")   or {})
                norris_ok = "popt" in (res.get("norris") or {})
                best = res.get("best", "?")
                fred_chi  = f"chi2/dof={res['fred']['chi2_red']:.2f}"   if fred_ok   else "FAIL"
                norris_chi= f"chi2/dof={res['norris']['chi2_red']:.2f}" if norris_ok else "FAIL"
                print(f"FRED:{fred_chi}  Norris:{norris_chi}  best={best.upper()}")
            else:
                print("insufficient data")

        # plot
        plot_path = os.path.join(FRED_DIR, f"{source_key}_plot.png")
        make_plot(source_key, df, all_band_results, plot_path)

        # per-source log
        src_log = os.path.join(FRED_DIR, f"{source_key}_log.txt")
        write_log(source_key, all_band_results, src_log, mode="w")

        # append to master
        write_log(source_key, all_band_results, master_log, mode="a")

    print(f"\nDone. All outputs in: {FRED_DIR}")


if __name__ == "__main__":
    main()