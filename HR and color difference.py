
# V0332+53 

import io
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit


source_name = "V0332+53"
maxi_url = "https://maxi.riken.jp/star_data/J0334+531/J0334+531_g_lc_1day_all.dat"
ztf_csv_path =  r"C:\Users\chowd\Downloads\ZTF18acbvefj_20260606\detections.csv"



headers = {'User-Agent': 'Mozilla/5.0'}
res = requests.get(maxi_url, headers=headers, timeout=15)
df_x = pd.read_csv(io.StringIO(res.text), sep=r'\s+', comment='#', header=None)


df_x = df_x[(df_x[0] >= 56000) & (df_x[0] <= 61500)].sort_values(0)
t_x_full  = df_x[0].values
soft_2_4  = df_x[3].values  # 2-4 keV band
med_4_10  = df_x[5].values  # 4-10 keV band


ztf_df = pd.read_csv(ztf_csv_path, comment='#')
ztf_df.columns = ztf_df.columns.str.lower().str.strip()

time_col = next((c for c in ['mjd', 'jd', 'time'] if c in ztf_df.columns), ztf_df.columns[0])
mag_col  = next((c for c in ['magpsf', 'mag', 'magnitude', 'brightness'] if c in ztf_df.columns), ztf_df.columns[1])
err_col  = next((c for c in ['e_mag', 'sigmagpsf', 'magerr'] if c in ztf_df.columns), 'e_mag')
filt_col = next((c for c in ['fid', 'filtercode', 'filter', 'band'] if c in ztf_df.columns), None)

if np.nanmax(ztf_df[time_col].values) > 2400000:
    ztf_df['mjd_calc'] = ztf_df[time_col] - 2400000.5
    time_col = 'mjd_calc'

g_data_full = ztf_df[(ztf_df[filt_col] == 1) | (ztf_df[filt_col] == '1') | (ztf_df[filt_col] == 'zg')].sort_values(time_col)
r_data_full = ztf_df[(ztf_df[filt_col] == 2) | (ztf_df[filt_col] == '2') | (ztf_df[filt_col] == 'zr')].sort_values(time_col)


optical_merged = pd.merge_asof(
    g_data_full, r_data_full, on=time_col, direction='nearest', 
    tolerance=2.5, suffixes=('_g', '_r')
).dropna(subset=[f"{mag_col}_g", f"{mag_col}_r"])

t_opt_full = optical_merged[time_col].values
col_diff_full = (optical_merged[f"{mag_col}_g"] - optical_merged[f"{mag_col}_r"]).values
err_col_diff_full = np.sqrt(optical_merged[f"{err_col}_g"]**2 + optical_merged[f"{err_col}_r"]**2).values


def piecewise_fred_profile_standard(t, t_peak, dt_rise_width, r, tau_decay, A, background):
    fit = np.full_like(t, background, dtype=float)
    cond_rise = (t <= t_peak) & (t > (t_peak - dt_rise_width))
    t_rise = t[cond_rise]
    dt_rise = (t_rise - (t_peak - dt_rise_width)) / dt_rise_width
    fit[cond_rise] = background + A * (dt_rise ** r)
    
    cond_decay = (t > t_peak)
    t_decay = t[cond_decay]
    fit[cond_decay] = background + A * np.exp(-(t_decay - t_peak) / tau_decay)
    return fit


outburst_windows = [
   
    (57130, 57480, "Outburst #1: Major X-Ray Outburst (Apex MJD ~57230)", True),
    (58650, 58950, "Outburst #2: First Optical Outburst (Apex MJD ~58800)", False),
    (59600, 60000, "Outburst #3: Second Optical Outburst (Apex MJD ~59800)", False)
]

print("=" * 65)
print(f" V0332+53 OUTBURST PIPELINE: PROCESSING {len(outburst_windows)} WINDOWS")
print("=" * 65 + "\n")


for burst_num, (w_start, w_end, win_label, is_xray_driven) in enumerate(outburst_windows, start=1):
    
    # Slice X-Ray Data
    x_mask = (t_x_full >= w_start) & (t_x_full <= w_end)
    x_sub  = t_x_full[x_mask]
    y_sub  = med_4_10[x_mask]
    s_sub  = soft_2_4[x_mask]

    # Slice Optical Light Curves
    g_sub = g_data_full[(g_data_full[time_col] >= w_start) & (g_data_full[time_col] <= w_end)] if not g_data_full.empty else pd.DataFrame()
    r_sub = r_data_full[(r_data_full[time_col] >= w_start) & (r_data_full[time_col] <= w_end)] if not r_data_full.empty else pd.DataFrame()

    # Slice Optical Color Index
    o_mask    = (t_opt_full >= w_start) & (t_opt_full <= w_end)
    t_o_sub   = t_opt_full[o_mask]
    c_o_sub   = col_diff_full[o_mask]
    err_c_sub = err_col_diff_full[o_mask]

    # --- FRED FIT (X-RAY ONLY) ---
    fit_params, reduced_chi_square = None, None
    if is_xray_driven and len(y_sub) > 5:
        peak_idx_loc = np.argmax(y_sub)
        t_peak_g = x_sub[peak_idx_loc]
        max_g = y_sub[peak_idx_loc]

        initial_guess = [t_peak_g, 35.0, 1.8, 25.0, max_g, 0.0]
        bounds = (
            (t_peak_g - 2.0, 15.0, 1.1, 5.0, max_g * 0.99, -0.01),
            (t_peak_g + 2.0, 65.0, 6.0, 120.0, max_g * 1.15, 0.01)
        )
        y_err = np.ones_like(y_sub) * 0.03
        y_err[(x_sub >= t_peak_g - 20) & (x_sub <= t_peak_g + 20)] = 0.018

        try:
            fit_params, _ = curve_fit(piecewise_fred_profile_standard, x_sub, y_sub, p0=initial_guess, bounds=bounds, sigma=y_err, maxfev=25000)
            y_model = piecewise_fred_profile_standard(x_sub, *fit_params)
            residuals = y_sub - y_model
            dof = max(1, len(x_sub) - len(fit_params))
            reduced_chi_square = np.sum((residuals / y_err) ** 2) / dof
        except Exception:
            fit_params = None

   
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        4, 1, figsize=(8.5, 9.5), sharex=True, 
        gridspec_kw={'height_ratios': [2.2, 2.0, 1.4, 1.6], 'hspace': 0.08}
    )

    # PANEL 1: X-RAY FLUX & FRED FIT
    if is_xray_driven and len(y_sub) > 0:
        ax1.plot(x_sub, y_sub, color='#b0bec5', alpha=0.5, lw=0.7, label='Raw Observations')
        ax1.scatter(x_sub, y_sub, color='#2e7d32', s=4, alpha=0.6)
        if fit_params is not None:
            x_smooth_abs = np.linspace(x_sub.min(), x_sub.max(), 2000)
            ax1.plot(x_smooth_abs, piecewise_fred_profile_standard(x_smooth_abs, *fit_params), color='#d32f2f', lw=1.8,
                     label=f'FRED Fit ($\\chi_\\nu^2$ = {reduced_chi_square:.2f})')
        ax1.legend(loc='upper right', fontsize=8)
    else:
        ax1.plot(x_sub, y_sub, color='#b0bec5', alpha=0.3, lw=0.5)
        ax1.text(0.5, 0.5, "Quiescent X-Ray Flux Level", transform=ax1.transAxes, 
                 ha='center', va='center', color='gray', fontsize=9.5, fontweight='bold')

    ax1.set_ylabel("4-10 keV X-Ray Flux\n[ph/s/cm²]", fontsize=8.5, fontweight='bold')
    ax1.set_title(f"{source_name} - {win_label}", fontsize=10.5, fontweight='bold')
    ax1.grid(True, linestyle=":", alpha=0.5)

    # PANEL 2: OPTICAL LIGHT CURVES (INVERTED MAGNITUDES)
    has_opt_lc = False
    if not g_sub.empty:
        ax2.scatter(g_sub[time_col], g_sub[mag_col], color='#2e7d32', s=6, alpha=0.7, label='ZTF g-band')
        ax2.plot(g_sub[time_col], g_sub[mag_col], color='#2e7d32', lw=0.6, alpha=0.4)
        has_opt_lc = True
    if not r_sub.empty:
        ax2.scatter(r_sub[time_col], r_sub[mag_col], color='#d32f2f', s=6, alpha=0.7, label='ZTF r-band')
        ax2.plot(r_sub[time_col], r_sub[mag_col], color='#d32f2f', lw=0.6, alpha=0.4)
        has_opt_lc = True

    if has_opt_lc:
        ax2.legend(loc='upper right', fontsize=8)
    else:
        ax2.text(0.5, 0.5, "No Optical Coverage in this MJD Window", transform=ax2.transAxes, 
                 ha='center', va='center', color='gray', fontsize=9)

    ax2.set_ylabel("Optical Magnitude\n[mag]", fontsize=8.5, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(True, linestyle=":", alpha=0.5)

    # PANEL 3: HARDNESS RATIO (PHYSICALLY CLEANED & INVERTED)
    valid_hr_mask = np.isfinite(s_sub) & np.isfinite(y_sub) & (s_sub > 0.001) & (y_sub > 0.0)
    if is_xray_driven and np.any(valid_hr_mask):
        hr_clean = y_sub[valid_hr_mask] / s_sub[valid_hr_mask]
        x_hr_clean = x_sub[valid_hr_mask]
        
        # Outlier filtering (removes division-by-zero noise spikes)
        q_low, q_high = np.percentile(hr_clean, [1, 98])
        inlier = (hr_clean >= q_low) & (hr_clean <= q_high)
        
        ax3.scatter(x_hr_clean[inlier], hr_clean[inlier], color='#546e7a', s=4, alpha=0.5, label='HR (4-10 / 2-4 keV)')
        hr_smooth = gaussian_filter1d(hr_clean[inlier], sigma=3) if np.sum(inlier) > 3 else hr_clean[inlier]
        ax3.plot(x_hr_clean[inlier], hr_smooth, color='#d32f2f', lw=1.6, label='X-Ray HR Trend')
        ax3.legend(loc='upper right', fontsize=8)
    else:
        ax3.text(0.5, 0.5, "Low X-Ray S/N Ratio (Quiescence)", transform=ax3.transAxes, 
                 ha='center', va='center', color='gray', fontsize=8.5)

    ax3.set_ylabel("Hardness Ratio\n[4-10 / 2-4 keV]", fontsize=8.5, fontweight='bold')
    ax3.invert_yaxis()
    ax3.grid(True, linestyle=":", alpha=0.5)

    # PANEL 4: OPTICAL COLOR DIFFERENCE (INVERTED MAGNITUDES)
    if len(t_o_sub) > 0:
        ax4.errorbar(t_o_sub, c_o_sub, yerr=err_c_sub, fmt='o', markersize=4, color='#78909c', alpha=0.6, capsize=0, label=' (g - r)')
        col_smooth = gaussian_filter1d(c_o_sub, sigma=2) if len(t_o_sub) > 3 else c_o_sub
        ax4.plot(t_o_sub, col_smooth, color='#1e88e5', lw=1.6, label='Color Trend')
        ax4.legend(loc='upper right', fontsize=8)
    else:
        ax4.text(0.5, 0.5, "No Paired (g - r) Data in Window", transform=ax4.transAxes, 
                 ha='center', va='center', color='gray', fontsize=8.5)

    ax4.set_ylabel("Color Difference\n[g - r mag]", fontsize=8.5, fontweight='bold')
    ax4.invert_yaxis()
    ax4.set_xlabel("Time (Modified Julian Date / MJD)", fontsize=9.5, fontweight='bold')
    ax4.set_xlim(w_start, w_end)
    ax4.grid(True, linestyle=":", alpha=0.5)

    fig.align_labels()
    plt.subplots_adjust(left=0.14, right=0.95, top=0.94, bottom=0.08)
    plt.show()
    print(f"--> Plotted Frame #{burst_num} for {source_name}: MJD {w_start} to {w_end}")
