
%matplotlib ipympl

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


url ="https://maxi.riken.jp/star_data/J0118+637/J0118+637_g_lc_1day_all.dat"
df = pd.read_csv(url, sep=r'\s+', comment='#', header=None)
df = df[(df[0] >= 54000) & (df[0] <= 61500)].sort_values(0)

time_all = df[0].values
flux_all = df[3].values  # 4-10 keV band

valid = np.isfinite(time_all) & np.isfinite(flux_all)
time_all, flux_all = time_all[valid], flux_all[valid]

flux_smoothed = gaussian_filter1d(flux_all, sigma=7)


median = np.median(flux_all)
mad = np.median(np.abs(flux_all - median))
sigma_mad = 1.4826 * mad
threshold = median + (5 * sigma_mad)

smooth_peak_indices, _ = find_peaks(
    flux_smoothed, 
    height=threshold, 
    distance=260, 
    prominence=0.05
)


true_peak_indices = []
for idx in smooth_peak_indices:
    start_window = int(max(0, idx - 20))
    end_window = int(min(len(flux_all), idx + 20))
    local_raw_max_idx = start_window + np.argmax(flux_all[start_window:end_window])
    true_peak_indices.append(int(local_raw_max_idx))

true_peak_indices = np.array(true_peak_indices, dtype=int)


plt.figure(figsize=(5, 4))


plt.plot(time_all, flux_all, color='#b0bec5', alpha=0.4, lw=0.5, label='Raw Data')
plt.scatter(time_all, flux_all, color='#cfd8dc', s=2, alpha=0.4, label='_nolegend_')


plt.plot(time_all, flux_smoothed, color='#1976d2', lw=1.2, alpha=0.8)


plt.scatter(time_all[true_peak_indices], flux_all[true_peak_indices], 
            color='#d32f2f', edgecolor='black', lw=0.7, s=35, zorder=5, label='Peaks')


plt.axhline(y=threshold, color='#ef6c00', linestyle='--', lw=1, alpha=0.7, label='5σ Threshold')

plt.title("4U 0115+63 ", fontsize=9, fontweight='bold')
plt.xlabel("Modified Julian Date (MJD)", fontsize=10)
plt.ylabel("Flux [ph/s/cm²]", fontsize=10)
plt.grid(True, linestyle=":", alpha=0.5)
plt.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
plt.ylim(-0.02, flux_all.max() * 1.05)

plt.tight_layout()
plt.show()


for idx in true_peak_indices:
    print(f" Outburst Apex -> MJD: {time_all[idx]:.2f} | Raw Flux: {flux_all[idx]:.4f}")
