# GX 339-4
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

maxi_url = "https://maxi.riken.jp/star_data/J1702-487/J1702-487_g_lc_1day_all.dat"
data = pd.read_csv(maxi_url, sep=r'\s+', comment='#', header=None)
maxi_filtered = data[(data[0] >= 54000) & (data[0] <= 61500)]

smarts_url = "http://www.astro.yale.edu/buxton/GX339/tables/gx339_master.tab"
#  999.0 is missing data (NaN)
data_smarts = pd.read_csv(smarts_url, sep=r'\s+', comment='#', header=None, na_values=['999.000', '999.00', '999.00000', '999.0'])

# Convert SMARTS time (JD - 2450000) to MJD so it aligns with MAXI
data_smarts[0] = data_smarts[0] + 49999.5
smarts_filtered = data_smarts[(data_smarts[0] >= 52000) & (data_smarts[0] <= 61500)]

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

if not maxi_filtered.empty:
    ax1.plot(maxi_filtered[0], maxi_filtered[3], color='red', alpha=0.7, linewidth=1)
    ax1.set_ylim(-0.005, maxi_filtered[3].max() * 1.1) 

    ax2.plot(maxi_filtered[0], maxi_filtered[5], color='green', alpha=0.7, linewidth=1)
    ax2.set_ylim(-0.005, maxi_filtered[5].max() * 1.1) 

ax1.set_ylabel("2-4 kev X-Ray\n[ph/s/cm²]", fontsize=7, fontweight='bold')
ax1.grid(True, linestyle=':', alpha=0.5)
ax2.set_ylabel("4-10 kev X-Ray\n[ph/s/cm²]", fontsize=7, fontweight='bold')
ax2.grid(True, linestyle=':', alpha=0.5)


if not smarts_filtered.empty:
    # V Band 
    v_data = smarts_filtered.dropna(subset=[1])
    ax3.scatter(v_data[0], v_data[1], color='red', s=4, alpha=0.7)
    # FIX 3: Invert Y-axis for magnitudes (brightest at the top)
    ax3.set_ylim(v_data[1].max() + 0.5, v_data[1].min() - 0.5) 

    # I Band 
    i_data = smarts_filtered.dropna(subset=[5])
    ax4.scatter(i_data[0], i_data[5], color='green', s=4, alpha=0.7)
    ax4.set_ylim(i_data[5].max() + 0.5, i_data[5].min() - 0.5) 

ax3.set_ylabel("V band\n[mag]", fontsize=7, fontweight='bold')
ax3.grid(True, linestyle=':', alpha=0.5)
ax4.set_ylabel("I band\n[mag]", fontsize=7, fontweight='bold')
ax4.grid(True, linestyle=':', alpha=0.5)

plt.xlim(52000, 61500)
ax4.set_xlabel("Time (MJD)", fontweight='bold', fontsize=11)

plt.tight_layout()
plt.subplots_adjust(hspace=0.08) 
plt.show()
