from astropy.io import fits
from astropy.table import Table
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#### SWIFT ####
lc_table = Table.read('~/Desktop/ksp2026/GX339_4/GX339-4.lc.fits', hdu='RATE')
x_swift = np.array(lc_table['TIME'])
y_swift = np.array(lc_table['RATE'])

#### MAXI ####
maxi_table = pd.read_csv("~/Desktop/ksp2026/GX339_4/gx339_4_maxi_data.csv", sep=' ', header=None)

x_maxi = maxi_table[0].to_numpy()
y_maxi = maxi_table[1].to_numpy()

# --- Hardness ratio: (4-10 keV) / (2-4 keV) ---
# columns: 3 = 2-4 keV flux (soft), 5 = 4-10 keV flux (hard)
soft = maxi_table[3].to_numpy()
hard = maxi_table[5].to_numpy()

with np.errstate(divide='ignore', invalid='ignore'):
    hr = np.where(soft > 0, hard / soft, np.nan)

x_hr = x_maxi

print(hr)

#### Optical ####
optical_table = pd.read_csv("~/Desktop/ksp2026/GX339_4/gx339_4_yale_data.csv", sep=r'\s+', header=None)

# filter out missing data (999.0 flag)
V_opt_table = optical_table[optical_table[3] != 999.0]
x_V = np.array(V_opt_table[0]) + 2450000 - 2400000.5
y_V = np.array(V_opt_table[1])

I_opt_table = optical_table[optical_table[7] != 999.0]
x_I = np.array(I_opt_table[0]) + 2450000 - 2400000.5
y_I = np.array(I_opt_table[5])

# --- Colour difference: V (bluer) - I (redder), nearest-neighbor match within 0.5 day ---
tol = 0.5  # days
x_VI, y_VI = [], []
for t_v, v in zip(x_V, y_V):
    idx = np.argmin(np.abs(x_I - t_v))
    if np.abs(x_I[idx] - t_v) <= tol:
        x_VI.append(t_v)
        y_VI.append(v - y_I[idx])
x_VI = np.array(x_VI)
y_VI = np.array(y_VI)

#### Time range input ####
start_time = 55045 # e.g. 55000, in MJD (same convention as x_swift/x_maxi)
end_time = 55753  # e.g. 56000

def clip(x, y, t0, t1):
    x = np.asarray(x); y = np.asarray(y)
    if t0 is None:
        t0 = x.min()
    if t1 is None:
        t1 = x.max()
    mask = (x >= t0) & (x <= t1)
    return x[mask], y[mask]

x_swift_c, y_swift_c = clip(x_swift, y_swift, start_time, end_time)
x_maxi_c, y_maxi_c = clip(x_maxi, y_maxi, start_time, end_time)
x_hr_c, hr_c = clip(x_hr, hr, start_time, end_time)
x_V_c, y_V_c = clip(x_V, y_V, start_time, end_time)
x_I_c, y_I_c = clip(x_I, y_I, start_time, end_time)
x_VI_c, y_VI_c = clip(x_VI, y_VI, start_time, end_time)

min_time = min(x_swift_c[0], x_maxi_c[0], x_V_c[0], x_I_c[0]) if start_time is None else start_time
max_time = max(x_swift_c[-1], x_maxi_c[-1], x_V_c[-1], x_I_c[-1]) if end_time is None else end_time

#### Original 4-panel light curve plot ####
fig, axs = plt.subplots(
    nrows=4, ncols=1, sharex=True, figsize=(8, 10), gridspec_kw={"hspace": 0}
)

axs[0].plot(x_swift_c, y_swift_c, "k.", markersize=2)
axs[0].set_title('Swift')
axs[1].plot(x_maxi_c, y_maxi_c, "r.", markersize=2)
axs[1].set_title('MAXI')
axs[2].plot(x_V_c, y_V_c, "g.", markersize=2)
axs[2].set_title('V optical')
axs[3].plot(x_I_c, y_I_c, "b.", markersize=2)
axs[3].set_title('I optical')

for i, ax in enumerate(axs):
    ax.axhline(0, color="black", linestyle=":", linewidth=0.8)
    ax.tick_params(axis="both", direction="in", top=True, right=True, which="both")
    if i < 3:
        ax.tick_params(axis="x", labelbottom=False)

axs[-1].set_xlim(min_time, max_time)
axs[0].set_title("GX 339-4, V821 Ara", loc="left", fontsize=12)

plt.savefig('/home/Vidhi/Desktop/ksp2026/plots_xrb/GX 339-4.svg')

#### Hardness ratio & colour difference plot ####
fig2, axs2 = plt.subplots(
    nrows=2, ncols=1, sharex=True, figsize=(8, 6), gridspec_kw={"hspace": 0}
)

axs2[0].plot(x_hr_c, hr_c, "r.", markersize=3)
axs2[0].set_ylabel("Hardness ratio\n(4-10 keV / 2-4 keV)")
axs2[0].set_title("MAXI hardness ratio", loc="left", fontsize=10)

axs2[1].plot(x_VI_c, y_VI_c, "m.", markersize=3)
axs2[1].set_ylabel("V - I magnitude difference")
axs2[1].set_title("Optical colour (V - I)", loc="left", fontsize=10)

axs2[1].set_xlabel("MJD")
for ax in axs2:
    ax.tick_params(axis="both", direction="in", top=True, right=True, which="both")

axs2[0].tick_params(axis="x", labelbottom=False)
axs2[-1].set_xlim(min_time, max_time)
axs2[0].set_title("GX 339-4, V821 Ara", loc="left", fontsize=12)

plt.savefig('/home/Vidhi/Desktop/ksp2026/plots_xrb/GX339-4_HR_color.svg')
plt.show()