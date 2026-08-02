import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pandas as pd
from scipy.optimize import curve_fit
from matplotlib.widgets import SpanSelector
from scipy.integrate import cumulative_trapezoid


def fred(t, A1, A2, tau_rise, tau_decay, t_peak):
    return np.where(
        t < t_peak,
        A1 * np.exp((t - t_peak) / tau_rise), 
        A2 * np.exp(-(t - t_peak) / tau_decay)
    )

maxi_table = pd.read_csv("~/Desktop/ksp2026/GX339_4/gx339_4_maxi_data.csv", sep=' ', header=None)

time = np.array(maxi_table[0].tolist())
flux = np.array(maxi_table[1].tolist())

counts, bin_edges = np.histogram(flux, bins=100)
max_bin_idx = np.argmax(counts)
quiescent_mode = (bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]) / 2

mad = np.median(np.abs(flux - np.median(flux)))
sigma_est = 1.5 * mad   
threshold = quiescent_mode + (5 * sigma_est)

peaks, _ = find_peaks(flux, height=threshold, prominence=sigma_est*5, distance=300, width=10)

fig, (ax, ax_res) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(10, 6))
ax.plot(time, flux, label='Lightcurve')
ax.axhline(quiescent_mode, color='green', linestyle='--', label='Quiescent Baseline')
ax.axhline(threshold, color='red', linestyle=':', label='Outburst Threshold')
ax.plot(time[peaks], flux[peaks], "x", color='black', label='Detected Peaks')
ax.set_xlim(np.min(time), np.max(time))
fit_line, = ax.plot([], [], color='red', lw=2.5, label='Curve Fit')
ax.legend(loc='upper right')
ax.set_title("Drag a span to fit to that region")


residual_line, = ax_res.plot([], [], color='purple', marker='o', linestyle='None', markersize=3, label='Residuals')
ax_res.axhline(0, color='gray', linestyle='--', alpha=0.7)
ax_res.set_ylabel("Residuals")
ax_res.set_xlabel("Time")

fit_text_box = None

def onselect(xmin, xmax):
    global fit_text_box # CRITICAL: Allows modifying the global tracker variable
    
    mask = (time >= xmin) & (time <= xmax)
    outburst_time = time[mask]
    outburst_flux = flux[mask]

    if len(outburst_time) >= 4:
        x = outburst_time - outburst_time[0]
        y = outburst_flux

        peak_y = np.max(y)
        peak_t = x[np.argmax(y)]
        base_y = np.min(y)
        approx_A1 = peak_y - base_y
        approx_A2 = peak_y - base_y

        total_duration = x[-1] - x[0]
        init_tau_rise = total_duration * 0.1   
        init_tau_decay = total_duration * 0.5  

        initial_guess = [approx_A1, approx_A2, init_tau_rise, init_tau_decay, peak_t]
        bounds = ([0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf])

        try:
            outburst_area = cumulative_trapezoid(outburst_flux,outburst_time,initial=0)
            outburst_area_norm = outburst_area/outburst_area[-1]
            t_5 = outburst_time[np.searchsorted(outburst_area_norm, 0.05)]
            t_95 = outburst_time[np.searchsorted(outburst_area_norm, 0.95)]
            t_90 = t_95 - t_5

            popt, pcov = curve_fit(fred, x, y, p0=initial_guess, bounds=bounds)
            y_model = fred(x, *popt)
            
            # Use safety check for error handling
            y_errors = np.std(y[:50]) if np.std(y[:50]) > 0 else 1.0
            chi_squared = np.sum(((y - y_model) / y_errors) ** 2)
            dof = len(x) - len(popt)
            
            if fit_text_box is not None:
                fit_text_box.remove()
                fit_text_box = None
                
            reduced_chi_squared = chi_squared / dof if dof > 0 else 0
            
            text_str = '\n'.join((
                r'$\mathbf{Fit\ Parameters:}$',
                r'$A_1 = {:.2f}$'.format(popt[0]),
                r'$A_2 = {:.2f}$'.format(popt[1]),
                r'$\tau_{{rise}} = {:.2f}$'.format(popt[2]),
                r'$\tau_{{decay}} = {:.2f}$'.format(popt[3]),
                r'$t_{{peak}} = {:.2f}$'.format(popt[4]),
                r'$t_{{90}} = {:.2f}$'.format(t_90),
                r'$\chi_\nu^2 = {:.2f}$'.format(reduced_chi_squared)
            ))

            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            fit_text_box = ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=11,
                                   verticalalignment='top', bbox=props)
            
            
            # --- FIX: Plot using unshifted X coordinates so it aligns with the data ---
            fit_line.set_data(outburst_time, y_model)

            residuals = outburst_flux - y_model

            residual_line.set_data(outburst_time, residuals)
            res_max = np.max(np.abs(residuals))
            ax_res.set_ylim(-res_max * 1.2, res_max * 1.2)
            fig.canvas.draw_idle()
            
        except Exception as e:
            print(f"Fit failed for window {outburst_time[0]} to {outburst_time[-1]}: {e}")
            fit_line.set_data([], [])
            if fit_text_box is not None:
                fit_text_box.remove()
                fit_text_box = None
            fig.canvas.draw_idle()
     
    else:
        print("Not enough points")
        fit_line.set_data([], [])
        if fit_text_box is not None:
            fit_text_box.remove()
            fit_text_box = None
        fig.canvas.draw_idle()

span = SpanSelector(
    ax,
    onselect,
    "horizontal",
    useblit=True,
    props=dict(alpha=0.5, facecolor="tab:blue"),
    interactive=True,
    drag_from_anywhere=True
)

plt.show()

"""
outburst_flux = flux[s:e]
outburst_time = time[s:e]


x = outburst_time - outburst_time[0]
y = outburst_flux

peak_y = np.max(y)
peak_t = x[np.argmax(y)]
base_y = np.min(y)
approx_A1 = peak_y - base_y
approx_A2 = peak_y - base_y

# Estimate timescales based on the actual window length
total_duration = x[-1] - x[0]
init_tau_rise = total_duration * 0.1   # Assume rise takes roughly 10% of window
init_tau_decay = total_duration * 0.5  # Assume decay takes roughly 50% of window

initial_guess = [approx_A1, approx_A2, init_tau_rise, init_tau_decay, peak_t]
bounds = ([0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf])

try:
    popt, pcov = curve_fit(fred, x, y, p0=initial_guess, bounds=bounds)
    
    # Plot using the updated function
    plt.plot(outburst_time, fred(x, *popt), 'r-', linewidth=2)

except Exception as e:
    print(f"Fit failed for window {s}-{e}: {e}")


y_model = fred(x, *popt)
y_errors = np.std(y[:50])
chi_squared = np.sum(((y-y_model)/y_errors)**2)

dof = len(x) - len(popt)

reduced_chi_squared = chi_squared/dof

print(reduced_chi_squared)
print(chi_squared)
print(popt)
plt.plot(outburst_time, outburst_flux, label='Lightcurve')

plt.legend()
plt.savefig("GX339_4_outburst.png")
plt.show()"""