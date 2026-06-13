from astropy.io import fits
from astropy.table import Table
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#### SWIFT ####
lc_table = Table.read('GX339_4/GX339-4.lc.fits', hdu='RATE')


x_swift = (lc_table['TIME'])
y_swift = lc_table['RATE']


#### MAXI ####

maxi_table = pd.read_csv("GX339_4\gx339_4_maxi_data.csv", sep=' ', header = None)

x_maxi = maxi_table[0].tolist()
y_maxi = maxi_table[1].tolist()
print(y_maxi)

#### Optical ####

optical_table = pd.read_csv("GX339_4\gx339_4_yale_data.csv", sep=r'\s+', header = None)

#x_opt = optical_table[0].tolist()+2450000


V_opt_table = optical_table[optical_table[3] != 999.0]
x_V = (np.array(V_opt_table[0]) + 2450000 - 2400000.5).tolist()
y_V = V_opt_table[3].tolist()

I_opt_table = optical_table[optical_table[7] != 999.0]

x_I = (np.array(I_opt_table[0]) + 2450000 - 2400000.5).tolist()
y_I = I_opt_table[7].tolist()


min_time = max(x_swift[0],x_maxi[0],x_V[0],x_I[0])
max_time = min(x_swift[-1],x_maxi[-1],x_V[-1],x_I[-1])


print(min_time)
print(max_time)
fig, axs = plt.subplots(
    nrows=4, ncols=1, sharex=True, figsize=(8, 10), gridspec_kw={"hspace": 0}
)

axs[0].plot(x_swift, y_swift, "k.", markersize=2, )  
axs[0].set_title('Swift')
axs[1].plot(x_maxi, y_maxi, "r.", markersize=2)  
axs[1].set_title('MAXI')
axs[2].plot(x_V, y_V, "g.", markersize=2) 
axs[2].set_title('V optical')
axs[3].plot(x_I, y_I, "b.", markersize=2)  
axs[3].set_title('I optical')


for i, ax in enumerate(axs):
    ax.axhline(0, color="black", linestyle=":", linewidth=0.8)

    ax.tick_params(
        axis="both", direction="in", top=True, right=True, which="both"
    )

    if i < 3:
        ax.tick_params(axis="x", labelbottom=False)

axs[-1].set_xlim(min_time, max_time)

axs[0].set_title("GX 339-4, V821 Ara", loc="left", fontsize=12)

plt.savefig('GX339_4\GX339_4.png')
plt.show()