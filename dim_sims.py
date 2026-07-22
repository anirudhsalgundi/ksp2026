import numpy as np
import matplotlib.pyplot as plt

def toy_dim_lightcurves(time, t_start, t_reach_center, t_end):
    """
    Generates a qualitative toy model of an Outside-In Disk Instability.
    """
    # Initialize arrays
    optical_flux = np.zeros_like(time)
    xray_flux = np.zeros_like(time)
    
    # 1. The Heating Front (Outside-In)
    # The instability triggers in the outer disk at t_start.
    # The hot zone grows inward until it hits the compact object at t_reach_center.
    heating_phase = (time >= t_start) & (time < t_reach_center)
    
    # Optical rises smoothly as the area of the hot disk increases.
    # We use a quadratic curve to simulate the growing area of the hot annulus.
    optical_flux[heating_phase] = ((time[heating_phase] - t_start) / (t_reach_center - t_start))**2
    
    # X-rays remain at baseline (0 here) until the mass actually reaches the center.
    xray_flux[heating_phase] = 0.05  # Slight increase as inner disk gets slightly perturbed
    
    # 2. Outburst Peak & Viscous Decay
    # The heating front has reached the center. The whole disk is hot.
    decay_phase = time >= t_reach_center
    
    # X-rays suddenly spike because mass is now dumping onto the compact object.
    # They then decay exponentially as the disk viscously drains (King & Ritter 1998).
    viscous_timescale = 15.0 # days
    xray_flux[decay_phase] = 1.0 * np.exp(-(time[decay_phase] - t_reach_center) / viscous_timescale)
    
    # The optical flux now becomes dominated by X-ray irradiation (Reprocessing).
    # It decays alongside the X-rays, roughly proportional to L_x^0.5
    optical_flux[decay_phase] = 1.0 * (xray_flux[decay_phase]**0.5)
    
    # Scale fluxes for plotting purposes
    optical_flux *= 100 
    xray_flux *= 150
    
    return xray_flux, optical_flux

# ==========================================
# Run the Simulation
# ==========================================
time = np.linspace(0, 80, 500)

# Define the critical DIM timescales (in days)
t_start_instability = 10      # Thermal instability triggers in outer disk
t_mass_reaches_center = 18    # Time it takes the heating front to reach the X-ray emitting region

xray_dim, opt_dim = toy_dim_lightcurves(time, t_start_instability, t_mass_reaches_center, t_end=80)

# Add a little noise so it looks like survey data
np.random.seed(42)
xray_plot = xray_dim + np.random.normal(0, 3, size=len(time))
opt_plot = opt_dim + np.random.normal(0, 2, size=len(time))

xray_plot = np.maximum(xray_plot, 0)
opt_plot = np.maximum(opt_plot, 0)

# ==========================================
# Plotting the DIM Template
# ==========================================
fig, ax1 = plt.subplots(figsize=(10, 6))

color1 = 'tab:blue'
ax1.set_xlabel('Time (Days)')
ax1.set_ylabel('X-ray Flux (Proxy for Inner Accretion)', color=color1)
ax1.plot(time, xray_dim, color='darkblue', linewidth=3, label='Toy DIM X-ray')
ax1.scatter(time, xray_plot, color=color1, s=10, alpha=0.3)
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()
color2 = 'tab:orange'
ax2.set_ylabel('Optical Flux (Proxy for Disk Area + Irradiation)', color=color2)
ax2.plot(time, opt_dim, color='darkred', linewidth=3, linestyle='--', label='Toy DIM Optical')
ax2.scatter(time, opt_plot, color=color2, s=10, alpha=0.3)
ax2.tick_params(axis='y', labelcolor=color2)

# Highlight the propagation delay
ax1.axvspan(t_start_instability, t_mass_reaches_center, color='grey', alpha=0.2, label='Heating Front Propagation (Delay)')

# Combine legends
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper right')

fig.suptitle('Qualitative Template: Outside-In Disk Instability Model (DIM)')
fig.tight_layout()
plt.show()