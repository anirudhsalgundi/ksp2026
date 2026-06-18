import pandas as pd
import matplotlib.pyplot as plt

maxi_url = "https://maxi.riken.jp/star_data/J0334+531/J0334+531_g_lc_1day_all.dat"
data = pd.read_csv(maxi_url, sep=r'\s+', comment='#', header=None)
maxi_filtered = data[(data[0] >= 57000) & (data[0] <= 61500)]

ztf_path = r"C:\Users\chowd\Downloads\ZTF18acbvefj_20260606\detections.csv"
df = pd.read_csv(ztf_path, sep=',')
filtered_df = df[(df['mjd'] >= 57000) & (df['mjd'] <= 61500)]

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(6, 4), sharex=True)

if not maxi_filtered.empty:
    ax1.plot(maxi_filtered[0], maxi_filtered[3], color='red', alpha=0.7, linewidth=1)
    
    ax1.set_ylim(-0.005, maxi_filtered[1].max() * 1.1) 

    ax2.plot(maxi_filtered[0], maxi_filtered[5], color='green', alpha=0.7, linewidth=1)
    
    ax1.set_ylim(-0.005, maxi_filtered[1].max() * 1.1) 

ax1.set_ylabel("2-4 kev X-Ray\n[ph/s/cm²]", fontsize=7, fontweight='bold')
ax1.grid(True, linestyle=':', alpha=0.5)
ax2.set_ylabel("4-10 kev X-Ray\n[ph/s/cm²]", fontsize=7, fontweight='bold')
ax2.grid(True, linestyle=':', alpha=0.5)

g_band_df = filtered_df[filtered_df['fid'] == 1]
if not g_band_df.empty:
    ax3.errorbar(g_band_df['mjd'], g_band_df['mag'], yerr=g_band_df['e_mag'],
                 fmt='o', markersize=3, color='green', alpha=0.7, capsize=0)
    ax3.set_ylim(g_band_df['mag'].max() + 0.2, g_band_df['mag'].min() - 0.2)

ax3.set_ylabel("g-band\nMagnitude", fontsize=9, fontweight='bold')
ax3.grid(True, linestyle=':', alpha=0.5)

r_band_df = filtered_df[filtered_df['fid'] == 2]
if not r_band_df.empty:
    ax4.errorbar(r_band_df['mjd'], r_band_df['mag'], yerr=r_band_df['e_mag'],
                 fmt='o', markersize=3, color='red', alpha=0.7, capsize=0)
    ax4.set_ylim(r_band_df['mag'].max() + 0.2, r_band_df['mag'].min() - 0.2)
else:
    ax4.set_ylim(20, 12)

ax4.set_ylabel("r-band\nMagnitude", fontsize=9, fontweight='bold')
ax4.grid(True, linestyle=':', alpha=0.5)

plt.xlim(57000, 61500)
ax4.set_xlabel("MJD", fontweight='bold', fontsize=11)


plt.tight_layout()
plt.subplots_adjust(hspace=0.08) 
plt.show()
