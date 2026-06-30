import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = "data"
PLOT_DIR = "newplots"

# Ensure the plots directory exists
os.makedirs(PLOT_DIR, exist_ok=True)

def generate_4panel_plot(file_label, maxi_path, ztf_path):
    """
    Generates and saves a 4-panel light curve for a single target.
    """
    try:
        df_maxi = pd.read_csv(maxi_path)
        df_ztf = pd.read_csv(ztf_path)
    except Exception as e:
        print(f"⚠️ Error reading files for {file_label}: {e}")
        return False

    # Ensure consistent column casing
    df_maxi.columns = df_maxi.columns.str.upper()
    df_ztf.columns = df_ztf.columns.str.upper()

    # Filter ZTF Optical Bands
    if 'FILTERCODE' in df_ztf.columns:
        filter_col = 'FILTERCODE'
    elif 'BAND' in df_ztf.columns:
        filter_col = 'BAND'
    else:
        print(f"⚠️ Could not find a filter column in the ZTF data for {file_label}.")
        return False

    df_ztf[filter_col] = df_ztf[filter_col].astype(str).str.lower()
    df_g = df_ztf[df_ztf[filter_col] == 'zg']
    df_r = df_ztf[df_ztf[filter_col] == 'zr']

    # Set up the 4-panel plot
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 10), sharex=True)
    plt.subplots_adjust(hspace=0.05)

    ax_xray_soft = axes[0]
    ax_xray_hard = axes[1]
    ax_opt_g = axes[2]
    ax_opt_r = axes[3]

    # --- PANEL 1: MAXI Soft X-ray (2-4 keV) ---
    if 'FLUX_2_4' in df_maxi.columns:
        ax_xray_soft.errorbar(
            df_maxi['MJD'], df_maxi['FLUX_2_4'], yerr=df_maxi.get('ERR_2_4', 0),
            fmt='o', markersize=2, color='royalblue', alpha=0.7, label='MAXI 2-4 keV'
        )
    ax_xray_soft.set_ylabel("Soft X-ray\nFlux (2-4 keV)")
    ax_xray_soft.legend(loc="upper right")
    ax_xray_soft.grid(True, linestyle='--', alpha=0.5)

    # --- PANEL 2: MAXI Hard X-ray (10-20 keV) ---
    if 'FLUX_10_20' in df_maxi.columns:
        ax_xray_hard.errorbar(
            df_maxi['MJD'], df_maxi['FLUX_10_20'], yerr=df_maxi.get('ERR_10_20', 0),
            fmt='o', markersize=2, color='darkorange', alpha=0.7, label='MAXI 10-20 keV'
        )
    ax_xray_hard.set_ylabel("Hard X-ray\nFlux (10-20 keV)")
    ax_xray_hard.legend(loc="upper right")
    ax_xray_hard.grid(True, linestyle='--', alpha=0.5)

    # --- PANEL 3: ZTF g-band ---
    if not df_g.empty and 'MAG' in df_g.columns:
        ax_opt_g.errorbar(
            df_g['MJD'], df_g['MAG'], yerr=df_g.get('MAGERR', 0),
            fmt='o', markersize=3, color='seagreen', alpha=0.8, label='ZTF g-band'
        )
        ax_opt_g.invert_yaxis()  # Magnitudes: lower is brighter
    ax_opt_g.set_ylabel("Optical\nMag (g-band)")
    ax_opt_g.legend(loc="upper right")
    ax_opt_g.grid(True, linestyle='--', alpha=0.5)

    # --- PANEL 4: ZTF r-band ---
    if not df_r.empty and 'MAG' in df_r.columns:
        ax_opt_r.errorbar(
            df_r['MJD'], df_r['MAG'], yerr=df_r.get('MAGERR', 0),
            fmt='o', markersize=3, color='crimson', alpha=0.8, label='ZTF r-band'
        )
        ax_opt_r.invert_yaxis()  # Magnitudes: lower is brighter
    ax_opt_r.set_ylabel("Optical\nMag (r-band)")
    ax_opt_r.legend(loc="upper right")
    ax_opt_r.grid(True, linestyle='--', alpha=0.5)

    # --- Formatting & Saving ---
    axes[3].set_xlabel("Time (MJD)", fontsize=12)
    fig.suptitle(f"Multi-Wavelength Light Curve: {file_label}", fontsize=16, y=0.92)

    # Save to file
    output_filename = os.path.join(PLOT_DIR, f"{file_label}_4panel.png")
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    
    # CRITICAL: Close the figure to free up memory during batch processing
    plt.close(fig) 
    
    return True
if __name__ == "__main__":
    print(f"🚀 Scanning '{DATA_DIR}/' for optical ZTF data (the limiting reagent)...")
    
    # 1. Find all ZTF files
    ztf_files = glob.glob(os.path.join(DATA_DIR, "*_ztf.csv"))
    
    if not ztf_files:
        print("❌ No ZTF files found in the data directory.")
        exit()
        
    print(f"✅ Found {len(ztf_files)} ZTF files. Cross-referencing with MAXI data...")
    print("-" * 60)

    success_count = 0
    
    # 2. Iterate and Match
    for ztf_path in ztf_files:
        # Extract the base file label (e.g., "data/4U0614+091_ztf.csv" -> "4U0614+091")
        filename = os.path.basename(ztf_path)
        file_label = filename.replace("_ztf.csv", "")
        
        # Check for the corresponding MAXI file
        maxi_path = os.path.join(DATA_DIR, f"{file_label}_maxi.csv")
        
        if not os.path.exists(maxi_path):
            print(f"⏭️  Skipping {file_label}: Missing matching '_maxi.csv' file.")
            continue
            
        # 3. Generate the plot
        print(f"📊 Plotting {file_label}...")
        success = generate_4panel_plot(file_label, maxi_path, ztf_path)
        
        if success:
            success_count += 1

    print("-" * 60)
    print(f"🏁 Batch plotting complete! Successfully generated {success_count} plots in the '{PLOT_DIR}/' folder.")