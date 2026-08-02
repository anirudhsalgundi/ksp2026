from astropy.table import Table
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from alerce.core import Alerce

alerce = Alerce()
#### MAXI ####
maxi_table = pd.read_csv("~/Desktop/ksp2026/Aql_X_1/aql_x_1_maxi_data.csv", sep=' ', header=None)

x_maxi = maxi_table[0].to_numpy()
y_maxi = maxi_table[1].to_numpy()

# --- Hardness ratio: (4-10 keV) / (2-4 keV) ---
# columns: 3 = 2-4 keV flux (soft), 5 = 4-10 keV flux (hard)
soft = maxi_table[3].to_numpy()
hard = maxi_table[5].to_numpy()

with np.errstate(divide='ignore', invalid='ignore'):
    hr = np.where(soft > 0, hard / soft, np.nan)

x_hr = x_maxi

#### ZTF ####
oid = 'ZTF18accedau'
response_data = alerce.query_detections(oid, format="json")
df = pd.DataFrame(response_data)

if not df.empty:
    df_g = df[df['fid'] == 1][['mjd', 'magpsf']].dropna().reset_index(drop=True)
    df_r = df[df['fid'] == 2][['mjd', 'magpsf']].dropna().reset_index(drop=True)

    x_g = df_g['mjd'].to_numpy()
    y_g = df_g['magpsf'].to_numpy()

    x_r = df_r['mjd'].to_numpy()
    y_r = df_r['magpsf'].to_numpy()
else:
    x_g = np.array([]); y_g = np.array([])
    x_r = np.array([]); y_r = np.array([])

# --- ZTF colour: g (bluer) - r (redder), nearest-neighbor match within 0.5 day ---
tol = 0.5  # days
x_gr, y_gr = [], []
for t_g, g in zip(x_g, y_g):
    if len(x_r) == 0:
        break
    idx = np.argmin(np.abs(x_r - t_g))
    if np.abs(x_r[idx] - t_g) <= tol:
        x_gr.append(t_g)
        y_gr.append(g - y_r[idx])
x_gr = np.array(x_gr)
y_gr = np.array(y_gr)

#### Time range input ####
start_time = 60550  # e.g. 55000, in MJD
end_time = 60630    # e.g. 56000

def clip(x, y, t0, t1):
    x = np.asarray(x); y = np.asarray(y)
    if t0 is None:
        t0 = x.min()
    if t1 is None:
        t1 = x.max()
    mask = (x >= t0) & (x <= t1)
    return x[mask], y[mask]

print(x_g)
x_maxi_c, y_maxi_c = clip(x_maxi, y_maxi, start_time, end_time)
x_hr_c, hr_c = clip(x_hr, hr, start_time, end_time)
x_g_c, y_g_c = clip(x_g, y_g, start_time, end_time)
x_r_c, y_r_c = clip(x_r, y_r, start_time, end_time)
x_gr_c, y_gr_c = clip(x_gr, y_gr, start_time, end_time)

min_time = min(x_maxi_c[0], x_g_c[0], x_r_c[0]) if start_time is None else start_time
max_time = max(x_maxi_c[-1], x_g_c[-1], x_r_c[-1]) if end_time is None else end_time

#### Light curve panel plot (MAXI, ZTF g, ZTF r) ####
fig, axs = plt.subplots(
    nrows=3, ncols=1, sharex=True, figsize=(8, 8), gridspec_kw={"hspace": 0}
)

axs[0].plot(x_maxi_c, y_maxi_c, "r.", markersize=2)
axs[0].set_title('MAXI')
axs[1].plot(x_g_c, y_g_c, "c.", markersize=2)
axs[1].set_title('ZTF g')
axs[1].invert_yaxis()  # magnitudes: brighter = smaller number
axs[2].plot(x_r_c, y_r_c, "m.", markersize=2)
axs[2].set_title('ZTF r')
axs[2].invert_yaxis()

for i, ax in enumerate(axs):
    ax.axhline(0, color="black", linestyle=":", linewidth=0.8)
    ax.tick_params(axis="both", direction="in", top=True, right=True, which="both")
    if i < 2:
        ax.tick_params(axis="x", labelbottom=False)

axs[-1].set_xlim(min_time, max_time)
axs[0].set_title("Aql X-1", loc="left", fontsize=12)



#### Hardness ratio & ZTF colour plot ####
fig2, axs2 = plt.subplots(
    nrows=2, ncols=1, sharex=True, figsize=(8, 6), gridspec_kw={"hspace": 0}
)

axs2[0].plot(x_hr_c, hr_c, "r.", markersize=3)
axs2[0].set_ylabel("Hardness ratio\n(4-10 keV / 2-4 keV)")
axs2[0].set_title("MAXI hardness ratio", loc="left", fontsize=10)

axs2[1].plot(x_gr_c, y_gr_c, "c.", markersize=3)
axs2[1].set_ylabel("g - r magnitude difference")
axs2[1].set_title("ZTF colour (g - r)", loc="left", fontsize=10)

axs2[1].set_xlabel("MJD")
for ax in axs2:
    ax.tick_params(axis="both", direction="in", top=True, right=True, which="both")

axs2[0].tick_params(axis="x", labelbottom=False)
axs2[-1].set_xlim(min_time, max_time)
axs2[0].set_title("Aql X-1", loc="left", fontsize=12)

plt.show()