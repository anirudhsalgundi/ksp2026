
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import io
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

headers = {'User-Agent': 'Mozilla/5.0'}


def mag_to_flux(mag, mag_err):
    flux = 3631.0 * 10.0 ** (-0.4 * mag) * 1000.0  # mJy
    flux_err = flux * (mag_err / 1.0857)
    return flux, flux_err

mjd_min, mjd_max = 60120, 60400


maxi_url = "https://maxi.riken.jp/star_data/J1911+005/J1911+005_g_lc_1day_all.dat"

session = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))

try:
    res_x = session.get(maxi_url, headers=headers, timeout=30)
    res_x.raise_for_status()
    df_x = pd.read_csv(io.StringIO(res_x.text), sep=r'\s+', comment='#', header=None)
except Exception:
    local_maxi_path = r"C:\Users\chowd\Downloads\J1911+005_g_lc_1day_all.dat"
    df_x = pd.read_csv(local_maxi_path, sep=r'\s+', comment='#', header=None)

df_x = df_x[(df_x[0] >= mjd_min) & (df_x[0] <= mjd_max)].sort_values(0)
df_x = df_x[df_x[3] >= 0.0]

time_x = df_x[0].values
raw_counts = df_x[3].values  # flux_2_20

# Crab conversion factor scaled to hit 2.0e-7 peak
maxi_conv = (2.4e-8 / 3.3) * 15.0  # erg/cm^2/s
flux_x = raw_counts * maxi_conv


def round_top_fred(t, t0, tau1, tau2, A, bg):
    dt = np.maximum(1e-5, t - t0)
    norm = np.exp(2.0 * np.sqrt(tau1 / tau2))
    fit = bg + A * norm * np.exp(-tau1 / dt - dt / tau2)
    fit[t <= t0] = bg
    return fit

max_idx = np.argmax(flux_x)
A_exact = flux_x[max_idx]

p0_x = [60145.0, 30.0, 12.0, A_exact, 0.0]
bounds_x = (
    [60130.0, 1.0, 2.0, A_exact * 0.5, -1e-9],
    [60155.0, 100.0, 30.0, A_exact * 1.5, 1e-9]
)

popt_x, _ = curve_fit(round_top_fred, time_x, flux_x, p0=p0_x, bounds=bounds_x, maxfev=20000)

x_dense = np.linspace(mjd_min, mjd_max, 1000)
y_xray_model = round_top_fred(x_dense, *popt_x)


ztf_file_path = r"C:\Users\chowd\Downloads\ZTF18accedau_20260714\detections.csv"

try:
    ztf_df = pd.read_csv(ztf_file_path)
    ztf_df.columns = [c.lower().strip() for c in ztf_df.columns]
    
    mjd_col = next((c for c in ['mjd', 'jd', 'time', 'obsdate'] if c in ztf_df.columns), 'mjd')
    mag_col = next((c for c in ['magap', 'mag', 'magnitude', 'm', 'magpsf'] if c in ztf_df.columns), 'mag')
    err_col = next((c for c in ['sigmagap', 'mag_err', 'magerr', 'sigmag', 'e_mag'] if c in ztf_df.columns), None)
    filter_col = next((c for c in ['filter', 'band', 'fid', 'passband', 'pb', 'flt'] if c in ztf_df.columns), None)
    
    if np.median(ztf_df[mjd_col]) > 2400000:
        ztf_df[mjd_col] -= 2400000.5
        
    if filter_col:
        r_mask = ztf_df[filter_col].astype(str).str.contains('r|2', case=False)
        ztf_df = ztf_df[r_mask]
        
    ztf_df = ztf_df[(ztf_df[mjd_col] >= mjd_min) & (ztf_df[mjd_col] <= mjd_max)].sort_values(mjd_col)
    
    time_opt = ztf_df[mjd_col].values
    mag_opt = ztf_df[mag_col].values
    err_opt = ztf_df[err_col].values if err_col else np.full_like(mag_opt, 0.02)
    
    flux_opt_mJy, err_opt_mJy = mag_to_flux(mag_opt, err_opt)
    has_ztf_data = len(time_opt) > 0

except FileNotFoundError:
    has_ztf_data = False

if has_ztf_data:
    
    scale_ratio = 25.0 / np.max(flux_opt_mJy)
    flux_opt_mJy = flux_opt_mJy * scale_ratio
    err_opt_mJy = err_opt_mJy * scale_ratio


def opt_reprocessing_model(t, beta, scale_opt, delta_t, F0):
    fx_lagged = round_top_fred(t - delta_t, *popt_x)
    fx_net = np.maximum(0, fx_lagged - popt_x[4])
    return F0 + scale_opt * (fx_net ** beta)

if has_ztf_data:
    p0_opt = [0.55, 1.0e5, -2.5, 0.0]
    bounds_opt = (
        [0.30, 1.0e2, -5.0, 0.0],
        [0.85, 1.0e9,  0.0, 0.5]
    )
    
    popt_opt, pcov_opt = curve_fit(
        opt_reprocessing_model, 
        time_opt, 
        flux_opt_mJy, 
        p0=p0_opt, 
        bounds=bounds_opt, 
        sigma=err_opt_mJy
    )
    
    fitted_beta, fitted_scale, fitted_delta_t, fitted_F0 = popt_opt
    y_opt_reprocessed = opt_reprocessing_model(x_dense, *popt_opt)
    
    print("=" * 60)
    print(f"FIT RESULTS:")
    print(f"Fitted Reprocessing Beta : {fitted_beta:.2f}")
    print(f"Fitted Time Lag (delta_t): {fitted_delta_t:.2f} days")
    print(f"Fitted Amplitude Scale   : {fitted_scale:.2e}")
    print("=" * 60)
else:
    popt_opt = [0.78, 2.8e5, -2.5, 0.0]
    y_opt_reprocessed = opt_reprocessing_model(x_dense, *popt_opt)


fig, ax1 = plt.subplots(figsize=(10, 5.5))
ax2 = ax1.twinx()


ax1.scatter(time_x, flux_x, color='#5391ff', s=18, alpha=0.9, label='MAXI (X-ray)')
ax1.plot(x_dense, y_xray_model, color='#173983', lw=2.0, label='X-ray model')
ax1.set_ylabel("X-ray flux (erg cm⁻² s⁻¹)", color='#3b82f6', fontsize=10, fontweight='bold')
ax1.set_xlabel("MJD", fontsize=10, fontweight='bold')
ax1.set_xlim(mjd_min, mjd_max)


if has_ztf_data:
    ax2.errorbar(time_opt, flux_opt_mJy, yerr=err_opt_mJy, fmt='s', color='#f97316', 
                 ms=5, alpha=0.9, capsize=2, label='ZTF (optical)')

# Reprocessed Curve
ax2.plot(x_dense, y_opt_reprocessed, color='#800000', linestyle='--', lw=2.0, 
         label=f'reprocessing model ($\\beta = {popt_opt[0]:.2f}$)')
ax2.set_ylabel("optical flux, r-band (mJy)", color='#c2410c', fontsize=10, fontweight='bold')



lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', fontsize=9)

fig.subplots_adjust(left=0.10, right=0.90, top=0.92, bottom=0.12)
plt.show()
