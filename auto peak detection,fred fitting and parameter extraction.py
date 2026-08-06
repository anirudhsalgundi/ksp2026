# V 0332+53 
%matplotlib ipympl

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit

url = "https://maxi.riken.jp/star_data/J0334+531/J0334+531_g_lc_1day_all.dat"
df = pd.read_csv(url, sep=r'\s+', comment='#', header=None)
df = df[(df[0] >= 54000) & (df[0] <= 61500)].sort_values(0)

df = df[df[3] >= 0.0]

time_all = df[0].values
flux_all = df[3].values  # 4-10 keV band

valid = np.isfinite(time_all) & np.isfinite(flux_all)
time_all, flux_all = time_all[valid], flux_all[valid]

#   FRED PROFILE MODEL
def piecewise_fred_profile(t, t_peak, dt_rise_width, r, tau_decay, A, background):
    """
    Dynamically scaling Piecewise FRED Model:
    Eliminates fixed step-discontinuities by turning the rise duration 
    into a fittable parameter optimized by curve_fit.
    """
    fit = np.full_like(t, background, dtype=float)
    
    # Rise Phase
    cond_rise = (t <= t_peak) & (t > (t_peak - dt_rise_width))
    t_rise = t[cond_rise]
    
   
    dt_rise = (t_rise - (t_peak - dt_rise_width)) / dt_rise_width
    fit[cond_rise] = background + A * (dt_rise ** r)
    
    # Decay Phase
    cond_decay = (t > t_peak)
    t_decay = t[cond_decay]
    fit[cond_decay] = background + A * np.exp(-(t_decay - t_peak) / tau_decay)
    
    return fit

flux_smoothed = gaussian_filter1d(flux_all, sigma=7)

median_bg = np.median(flux_all)
mad = np.median(np.abs(flux_all - median_bg))
sigma_mad = 1.4826 * mad
threshold = median_bg + (5 * sigma_mad)


desired_distance_days = 260
data_resolution = np.median(np.diff(time_all))
index_distance = int(desired_distance_days / data_resolution)

smooth_peak_indices, _ = find_peaks(
    flux_smoothed, 
    height=threshold, 
    distance=index_distance, 
    prominence=0.3
)


true_peak_indices = []
for idx in smooth_peak_indices:
    start_window = int(max(0, idx - 15))
    end_window = int(min(len(flux_all), idx + 15))
    local_raw_max_idx = start_window + np.argmax(flux_all[start_window:end_window])
    true_peak_indices.append(int(local_raw_max_idx))

true_peak_indices = np.array(true_peak_indices, dtype=int)
print(f"Found {len(true_peak_indices)} macro-outbursts to model.\n")


window_before = 100
window_after = 250  

for n_peak, peak_idx in enumerate(true_peak_indices):
    burst_number = n_peak + 1
    
    if peak_idx - window_before < 0 or peak_idx + window_after >= len(time_all):
        print(f"Skipping Outburst {burst_number}: Window clips data boundaries.")
        continue
        
    x_data = time_all[peak_idx - window_before : peak_idx + window_after]
    y_data = flux_all[peak_idx - window_before : peak_idx + window_after]
    
    
    t_peak_guess = time_all[peak_idx]
    max_flux_guess = np.max(y_data)
    background_guess = 0.0
    
    # [t_peak, dt_rise_width, r, tau_decay, Amplitude, background]
    initial_guess = [t_peak_guess, 35.0, 1.8, 25.0, max_flux_guess, background_guess]
    
   
    bounds = (
        # Lower Bounds
        (t_peak_guess - 2.0,  15.0,  1.1,  5.0,   max_flux_guess * 0.99, -0.01),
        # Upper Bounds
        (t_peak_guess + 2.0,  65.0,  6.0,   120.0, max_flux_guess * 1.15,  0.01)
    )
    
   
    y_err = np.ones_like(y_data) * 0.03
    core_mask = (x_data >= t_peak_guess - 20) & (x_data <= t_peak_guess + 20)
    y_err[core_mask] = 0.018
    
    try:
        fit_params, fit_cov = curve_fit(
            piecewise_fred_profile,
            x_data,
            y_data,
            p0=initial_guess,
            bounds=bounds,
            sigma=y_err,
            absolute_sigma=False,  
            maxfev=25000
        )
        
        y_model = piecewise_fred_profile(x_data, *fit_params)
        residuals = y_data - y_model
        
        chi_square = np.sum((residuals / y_err) ** 2)
        dof = len(x_data) - len(fit_params)
        reduced_chi_square = chi_square / dof
        
    except RuntimeError:
        print(f" FRED Fit failed to converge for Outburst {burst_number}")
        continue
        
    
    fitted_dt_rise_width = fit_params[1]
    fitted_r = fit_params[2]
    calculated_tau_rise = fitted_dt_rise_width / fitted_r
        
    print("*" * 65)
    print(f"CHARACTERISTICS FOR OUTBURST {burst_number} (Apex MJD: {t_peak_guess:.2f})")
    print("-" * 65)
    print(f"Fitted Peak Epoch (t_peak)   : {fit_params[0]:.3f} MJD")
    print(f"Optimized Rise Width (T_rise): {fitted_dt_rise_width:.3f} Days")
    print(f"Rise Power Index (r)         : {fitted_r:.3f}")
    print(f"Derived Rise Scale (tau_rise): {calculated_tau_rise:.3f} Days") 
    print(f"Decay Timescale (tau_decay)  : {fit_params[3]:.3f} Days")
    print(f"Peak Amplitude (A)           : {fit_params[4]:.3f}")
    print(f"Local Background Level       : {fit_params[5]:.4f}")
    print(f"Reduced Chi-Square (χ²/v)    : {reduced_chi_square:.3f}")
    print("*" * 65 + "\n")
    
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, 
                                   gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.06})
    
    x_smooth_abs = np.linspace(x_data.min(), x_data.max(), 3000)
    
    ax1.plot(x_data, y_data, color='#b0bec5', alpha=0.4, lw=0.6, label='Raw Observations')
    ax1.scatter(x_data, y_data, color='#2e7d32', s=3, alpha=0.5, label='_nolegend_')
    ax1.plot(x_smooth_abs, piecewise_fred_profile(x_smooth_abs, *fit_params), color='#d32f2f', lw=1.8,
             label=f' FRED Fit ($\\chi_\\nu^2$ = {reduced_chi_square:.2f})')
    
    ax1.set_ylabel("Flux [ph/s/cm²]", fontsize=10)
    ax1.set_title(f"  FRED Modeling (Outburst {burst_number})", fontsize=11, fontweight='bold')
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc='upper right')
    
    ax2.scatter(x_data, residuals, color='#37474f', s=3, alpha=0.6)
    ax2.axhline(y=0, color='#d32f2f', linestyle='--', lw=1, alpha=0.8)
    ax2.set_ylabel("Residuals", fontsize=10)
    ax2.set_xlabel("Time (Modified Julian Date / MJD)", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.set_ylim(-4 * np.std(residuals), 4 * np.std(residuals))
    
    fig.align_labels()  
    plt.subplots_adjust(left=0.12, right=0.95, top=0.93, bottom=0.12)
    plt.show()
