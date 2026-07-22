import numpy as np
import matplotlib.pyplot as plt

def generate_xray_fred(t, t_start, t_rise, t_decay, amplitude):
    """
    Generates a Fast Rise Exponential Decay (FRED) X-ray lightcurve.
    """
    flux = np.zeros_like(t)
    mask = t > t_start
    t_eff = t[mask] - t_start
    
    # The peak occurs at t_peak = sqrt(t_rise * t_decay)
    # We normalize it so the maximum value exactly equals 'amplitude'
    norm_factor = np.exp(2 * np.sqrt(t_rise / t_decay))
    
    flux[mask] = amplitude * norm_factor * np.exp(-(t_rise / t_eff) - (t_eff / t_decay))
    return flux

def simulate_optical_reprocessing(t, xray_flux, beta=0.5, scaling_factor=1.0, lag_days=0.0):
    """
    Simulates optical flux based on X-ray heating of the accretion disk.
    Optical Flux is proportional to (X-ray Flux)^beta.
    """
    # Shift the time array to account for light-travel or viscous delays
    t_shifted = t - lag_days
    
    # Interpolate the X-ray flux at the delayed times
    xray_delayed = np.interp(t, t_shifted, xray_flux, left=0, right=0)
    
    # Apply the reprocessing power-law relation
    optical_flux = scaling_factor * (xray_delayed ** beta)
    return optical_flux

# ==========================================
# Run the Simulation
# ==========================================

# 1. Define the time array (e.g., 100 days of observations)
time = np.linspace(0, 100, 500)

# 2. Simulate the intrinsic X-ray outburst
xray_true = generate_xray_fred(time, t_start=10, t_rise=1.5, t_decay=15, amplitude=150)

# 3. Simulate the reprocessed optical outburst
# beta=0.5 is standard for a Shakura-Sunyaev disk irradiated by a central X-ray source
optical_true = simulate_optical_reprocessing(time, xray_true, beta=0.5, scaling_factor=4.0, lag_days=1.0)

# 4. Add Gaussian noise to simulate survey data (MAXI/Swift and ZTF/ATLAS)
np.random.seed(42)
xray_obs = xray_true + np.random.normal(0, 5, size=len(time))
optical_obs = optical_true + np.random.normal(0, 1.5, size=len(time))

# Make flux >= 0 to simulate real detector floors
xray_obs = np.maximum(xray_obs, 0)
optical_obs = np.maximum(optical_obs, 0)

# ==========================================
# Plotting
# ==========================================
fig, ax1 = plt.subplots(figsize=(10, 6))

color1 = 'tab:blue'
ax1.set_xlabel('Time (Days)')
ax1.set_ylabel('X-ray Flux (Counts/s)', color=color1)
ax1.scatter(time, xray_obs, color=color1, s=10, alpha=0.5, label='Simulated MAXI (X-ray)')
ax1.plot(time, xray_true, color='darkblue', linewidth=2, label='True X-ray FRED')
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()
color2 = 'tab:orange'
ax2.set_ylabel('Optical Flux (Arbitrary Units)', color=color2)
ax2.scatter(time, optical_obs, color=color2, s=10, alpha=0.5, label='Simulated ZTF/ATLAS (Optical)')
ax2.plot(time, optical_true, color='darkred', linewidth=2, linestyle='--', label='True Optical Reprocessing')
ax2.tick_params(axis='y', labelcolor=color2)

fig.suptitle('Simulated X-ray Binary Outburst: Reprocessing Model')
fig.tight_layout()
plt.show()