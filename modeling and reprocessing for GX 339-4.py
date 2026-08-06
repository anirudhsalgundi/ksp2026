%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import io
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

headers = {'User-Agent': 'Mozilla/5.0'}


def piecewise_fred_profile(t, t_peak, dt_rise_width, r, tau_fast, tau_slow, w_slow, A):
    t_arr = np.atleast_1d(t)
    background = 0.0001
    fit = np.full_like(t_arr, background, dtype=float)
    
    # FRED Rise
    cond_rise = (t_arr <= t_peak) & (t_arr > (t_peak - dt_rise_width))
    t_rise = t_arr[cond_rise]
    dt_rise = np.maximum(0, (t_rise - (t_peak - dt_rise_width)) / dt_rise_width)
    fit[cond_rise] = background + A * (dt_rise ** r)
    
    # Dual Exponential Decay
    cond_decay = (t_arr > t_peak)
    t_decay = t_arr[cond_decay] - t_peak
    decay_kernel = (1.0 - w_slow) * np.exp(-t_decay / tau_fast) + w_slow * np.exp(-t_decay / tau_slow)
    fit[cond_decay] = background + A * decay_kernel
    
    return fit[0] if np.isscalar(t) else fit


url = "https://maxi.riken.jp/star_data/J1702-487/J1702-487_g_lc_1day_all.dat"

session = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))

try:
    res_x = session.get(url, headers=headers, timeout=30)
    res_x.raise_for_status()
    df_x = pd.read_csv(io.StringIO(res_x.text), sep=r'\s+', comment='#', header=None)
except Exception:
    local_maxi_path = r"C:\Users\chowd\Downloads\J1702-487_g_lc_1day_all.dat"
    df_x = pd.read_csv(local_maxi_path, sep=r'\s+', comment='#', header=None)

df_x = df_x[(df_x[0] >= 54000) & (df_x[0] <= 61500)].sort_values(0)
df_x = df_x[df_x[3] >= 0.0]

time_all = df_x[0].values
flux_all = df_x[5].values  # 4-10 keV

valid = np.isfinite(time_all) & np.isfinite(flux_all)
time_all, flux_all = time_all[valid], flux_all[valid]


flux_smoothed = gaussian_filter1d(flux_all, sigma=7)
median_bg = np.median(flux_all)
mad = np.median(np.abs(flux_all - median_bg))
threshold = median_bg + (5 * 1.4826 * mad)

data_resolution = np.median(np.diff(time_all))
index_distance = int(260 / data_resolution)

smooth_peak_indices, _ = find_peaks(
    flux_smoothed, height=threshold, distance=index_distance, prominence=0.3
)

peak_idx = smooth_peak_indices[0]
start_idx = max(0, peak_idx - 260)
end_idx = min(len(time_all), peak_idx + 400)

x_data = time_all[start_idx : end_idx]
y_data = flux_all[start_idx : end_idx]

true_max_local_idx = np.argmax(y_data)
t_peak_exact = x_data[true_max_local_idx]
max_flux_exact = y_data[true_max_local_idx]


popt_x = [t_peak_exact, 85.0, 3.0, 18.0, 100.0, 0.40, max_flux_exact - 0.005]

maxi_conv = (2.4e-8 / 3.3) * 15.0  
flux_x_erg = y_data * maxi_conv

mjd_min, mjd_max = x_data.min(), x_data.max()
x_dense = np.linspace(mjd_min, mjd_max, 1000)
y_xray_model_raw = piecewise_fred_profile(x_dense, *popt_x)
y_xray_model_erg = y_xray_model_raw * maxi_conv


smarts_url = "http://www.astro.yale.edu/buxton/GX339/tables/gx339_master.tab"

try:
    res_s = session.get(smarts_url, headers=headers, timeout=30)
    res_s.raise_for_status()
    df_s = pd.read_csv(
        io.StringIO(res_s.text), sep=r'\s+', comment='#', header=None, 
        na_values=['999.000', '999.00', '999.00000', '999.0']
    )
except Exception:
    local_smarts_path = r"C:\Users\chowd\Downloads\gx339_master.tab"
    df_s = pd.read_csv(
        local_smarts_path, sep=r'\s+', comment='#', header=None, 
        na_values=['999.000', '999.00', '999.00000', '999.0']
    )

df_s[0] = df_s[0] + 49999.5
df_s_v = df_s.dropna(subset=[1])
df_s_v = df_s_v[(df_s_v[0] >= mjd_min) & (df_s_v[0] <= mjd_max)].sort_values(0)

time_opt = df_s_v[0].values
mag_opt = df_s_v[1].values
err_mag_opt = df_s_v[2].values if 2 in df_s_v.columns else None

f0_v = 3636.0  
flux_opt_mJy = f0_v * (10.0 ** (-0.4 * mag_opt)) * 1000.0
err_opt_mJy = flux_opt_mJy * (err_mag_opt / 1.0857) if err_mag_opt is not None else np.full_like(flux_opt_mJy, 0.05 * flux_opt_mJy)
has_opt_data = len(time_opt) > 0


def generate_broadened_optical_model(t):
    # Component 1: Optical Jet Component
    t_jet = 55288.5
    A_jet = 3.10          # Hits exact 4.3 mJy apex
    dt_rise_width = 110.0 # Broad rise matching MJD 55180-55260
    r_opt = 1.4           
    tau_opt_decay = 24.0  # Smooth jet decay
    F0 = 0.0001
    
    t_arr = np.atleast_1d(t)
    g_jet = np.zeros_like(t_arr, dtype=float)
    
    # Power-Law Rise
    cond_rise = (t_arr <= t_jet) & (t_arr > (t_jet - dt_rise_width))
    t_rise = t_arr[cond_rise]
    dt_rise = np.maximum(0, (t_rise - (t_jet - dt_rise_width)) / dt_rise_width)
    g_jet[cond_rise] = A_jet * (dt_rise ** r_opt)
    
    # Fast Jet Decay
    cond_decay = (t_arr > t_jet)
    t_decay = t_arr[cond_decay] - t_jet
    g_jet[cond_decay] = A_jet * np.exp(-t_decay / tau_opt_decay)
    
    # Component 2: Thermal Reprocessing (Sustains broad plateau, decays cleanly to zero)
    delta_t = 2.0
    beta = 0.40           
    scale_opt = 1.80      
    
    fx_raw = piecewise_fred_profile(t_arr - delta_t, *popt_x) - 0.0001
    fx_pos = np.maximum(0.0, fx_raw)
    f_repro = scale_opt * (fx_pos ** beta)
    
    out = F0 + g_jet + f_repro
    return out[0] if np.isscalar(t) else out

y_opt_reprocessed = generate_broadened_optical_model(x_dense)


fig, ax1 = plt.subplots(figsize=(10, 5.5))
ax2 = ax1.twinx()


ax1.scatter(x_data, flux_x_erg, color='#5391ff', s=18, alpha=0.8, label='MAXI 4-10 keV Data')
ax1.plot(x_dense, y_xray_model_erg, color='#173983', lw=2.0, label=' X-ray Model')
ax1.set_ylabel("X-ray flux (erg cm⁻² s⁻¹)", color='#3b82f6', fontsize=10, fontweight='bold')
ax1.set_xlabel("Time (MJD)", fontsize=10, fontweight='bold')
ax1.set_xlim(mjd_min, mjd_max)
ax1.set_ylim(-0.4e-8, 5.8e-8)


if has_opt_data:
    ax2.errorbar(time_opt, flux_opt_mJy, yerr=err_opt_mJy, fmt='s', color='#f97316', 
                 ms=5, alpha=0.9, capsize=2, label='SMARTS V-band Data')

# Reprocessed Curve
ax2.plot(x_dense, y_opt_reprocessed, color='#800000', linestyle='--', lw=2.2, 
         label='Optical Model ($\\Delta t = 2.0$d, $\\beta = 0.40$)')
ax2.set_ylabel("Optical Flux Density, V-band (mJy)", color='#c2410c', fontsize=10, fontweight='bold')
ax2.set_ylim(-0.3, 4.6)


lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', fontsize=9)

fig.subplots_adjust(left=0.10, right=0.90, top=0.92, bottom=0.12)
plt.show()
