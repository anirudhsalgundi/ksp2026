
import numpy as np
import matplotlib.pyplot as plt

# Time vector (0 to 80 days)
t_vec = np.linspace(0, 80, 500)

xray_dim = np.zeros_like(t_vec)
opt_dim = np.zeros_like(t_vec)

# 1. Heating Front Delay Phase (Days 10 to 18)
t_start, t_peak, t_end = 10.0, 18.0, 80.0
mask_front = (t_vec >= t_start) & (t_vec < t_peak)
xray_dim[mask_front] = 7.5  # Low inner accretion rate during front propagation

# Optical rises early as the heating front propagates inward across the disk
dt_rise = (t_vec[mask_front] - t_start) / (t_peak - t_start)
opt_dim[mask_front] = 100.0 * (dt_rise ** 1.5)

# 2. Peak & Decay Phase (Days 18 to 80)
mask_decay = (t_vec >= t_peak) & (t_vec <= t_end)
tau_decay = 20.0
xray_dim[mask_decay] = 150.0 * np.exp(-(t_vec[mask_decay] - t_peak) / tau_decay)

# Inward propagating DIM assumption during decay: F_opt = 1.0 * (F_X ** 0.5)
opt_dim[mask_decay] = 8.16 * (xray_dim[mask_decay] ** 0.5)


np.random.seed(42)
xray_noisy = xray_dim + np.random.normal(0, 2.5, size=len(t_vec))
opt_noisy = opt_dim + np.random.normal(0, 2.0, size=len(t_vec))


fig, ax1 = plt.subplots(figsize=(10, 5.5))
ax2 = ax1.twinx()


ax1.axvspan(10, 18, color='gray', alpha=0.2, label='Heating Front Propagation (Delay)')


ax1.scatter(t_vec, np.maximum(0, xray_noisy), color='#1e88e5', s=12, alpha=0.4)
ax1.plot(t_vec, xray_dim, color='navy', lw=2.5, label='Toy DIM X-ray')
ax1.set_ylabel("X-ray Flux (Proxy for Inner Accretion)", color='#1e88e5', fontsize=11)
ax1.set_xlabel("Time (Days)", fontsize=11)


ax2.scatter(t_vec, np.maximum(0, opt_noisy), color='#f57c00', s=12, alpha=0.4)
ax2.plot(t_vec, opt_dim, color='maroon', linestyle='--', lw=2.5, label='Toy DIM Optical')
ax2.set_ylabel("Optical Flux (Proxy for Disk Area + Irradiation)", color='#f57c00', fontsize=11)

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', fontsize=9)

fig.subplots_adjust(left=0.10, right=0.90, top=0.92, bottom=0.12)
plt.show()
