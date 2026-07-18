

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Professional plot styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['ytick.major.size'] = 6

# ─────────────────────────────────────────────────────────────────────────────
# RIGOROUS ASTROPHYSICAL CLEANING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def clean_and_weight_bin_data(df, mjd_col, val_col, err_col, filter_col=None, filter_pattern=None, min_snr=1.5):
    """
    Cleans raw data by:
    1. Stripping out non-numeric and negative error artifacts.
    2. Filtering out low Signal-to-Noise Ratio (SNR) background junk.
    3. Performing Error-Weighted Binning per day to compress clumps cleanly.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    
    d = df.copy()
    
    # Filter by specific instrument band if requested
    if filter_col and filter_pattern:
        d = d[d[filter_col].astype(str).str.contains(filter_pattern, case=False, na=False)]
    
    if d.empty:
        return pd.DataFrame()
        
    # Standardize data types
    for col in [mjd_col, val_col, err_col]:
        d[col] = pd.to_numeric(d[col], errors='coerce')
    d = d.dropna(subset=[mjd_col, val_col, err_col])
    
    # Baseline checks: strip out zero or negative error bars
    d = d[d[err_col] > 0]
    
    # For optical magnitudes, lower numbers are brighter. For flux measurements (MAXI, ATLAS, BAT),
    # we filter out points that are buried deep in the noise floor using a Signal-to-Noise Ratio check.
    # We bypass this for magnitudes since they are logarithmic and don't center on 0.
    is_magnitude = 'mag' in val_col.lower()
    if not is_magnitude and min_snr > 0:
        d = d[np.abs(d[val_col]) / d[err_col] >= min_snr]
        
    if d.empty:
        return pd.DataFrame()
        
    # CRITICAL CLEANING STEP: Remove points with massive error bar spikes (Sigma Clipping on errors)
    median_err = d[err_col].median()
    d = d[d[err_col] <= (median_err * 3.0)] # Throw out anything 3x worse than the median quality
    
    if d.empty:
        return pd.DataFrame()
        
    # ANTI-CLUMPING TECHNIQUE: Group by integer Modified Julian Day (MJD)
    d['mjd_day'] = d[mjd_col].round()
    
    d['weight'] = 1.0 / (d[err_col] ** 2)
    d['weighted_val'] = d[val_col] * d['weight']
    
    # Perform the group aggregation
    binned_groups = d.groupby('mjd_day')
    
    binned_data = []
    for day, group in binned_groups:
        total_weight = group['weight'].sum()
        if total_weight == 0:
            continue
            
        # Standard error-weighted average calculations
        mean_mjd = group[mjd_col].mean()
        weighted_mean_val = group['weighted_val'].sum() / total_weight
        
        # Propagated error of a weighted mean: 1 / sqrt(sum(weights))
        propagated_err = 1.0 / np.sqrt(total_weight)
        
        binned_data.append({
            'mjd': mean_mjd,
            'val': weighted_mean_val,
            'err': propagated_err
        })
        
    binned_df = pd.DataFrame(binned_data)
    if not binned_df.empty:
        binned_df = binned_df.sort_values(by='mjd').reset_index(drop=True)
        
    return binned_df

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE ORCHESTRATION
# ─────────────────────────────────────────────────────────────────────────────

def load_safe_csv(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            return df if not df.empty else pd.DataFrame()
        except (pd.errors.EmptyDataError, Exception):
            return pd.DataFrame()
    return pd.DataFrame()


def generate_gold_plots(data_dir="newdata", output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    catalog_path = os.path.join(data_dir, "maxi_slist_catalog.csv")
    
    if not os.path.exists(catalog_path):
        print(f"[-] Catalog file not found at {catalog_path}.")
        return

    catalog = pd.read_csv(catalog_path)
    print(f"[+] Loaded catalog list. Scanning for sources with clean curves...")
    
    for _, row in catalog.iterrows():
        source_id = row["source_id"]
        safe_id = str(source_id).strip().replace(" ", "_")
        safe_id_maxi = safe_id.replace("+", "p").replace("-", "m")
        
        maxi_df  = load_safe_csv(os.path.join(data_dir, f"{safe_id_maxi}_maxi_lc.csv"))
        swift_df = load_safe_csv(os.path.join(data_dir, f"{safe_id}_swift.csv"))
        ztf_df   = load_safe_csv(os.path.join(data_dir, f"{safe_id}_ztf.csv"))
        atlas_df = load_safe_csv(os.path.join(data_dir, f"{safe_id}_atlas.csv"))
        
        # ── 1. Gather Cleaned X-Ray Channels ─────────────────────────────────
        xray_candidates = []
        maxi_bands_config = [
            ("flux_2_4", "MAXI 2–4 keV", "#1f77b4", "o"),
            ("flux_4_10", "MAXI 4–10 keV", "#ff7f0e", "o"),
            ("flux_10_20", "MAXI 10–20 keV", "#2ca02c", "o")
        ]
        
        if not maxi_df.empty:
            for col_name, label, color, marker in maxi_bands_config:
                err_name = col_name.replace("flux", "err")
                if col_name in maxi_df.columns and err_name in maxi_df.columns:
                    # Apply our updated cleaning and weighting engine
                    b_data = clean_and_weight_bin_data(maxi_df, 'mjd', col_name, err_name, min_snr=2.0)
                    if not b_data.empty and len(b_data) > 3: # Ignore curves with fewer than 3 valid points
                        xray_candidates.append({
                            "df": b_data, "label": label, "ylabel": f"{label}\n(phot/s/cm²)",
                            "color": color, "marker": marker, "invert": False
                        })
                        
        if not swift_df.empty:
            s_data = clean_and_weight_bin_data(swift_df, 'mjd', 'count_rate', 'count_rate_err', min_snr=1.5)
            if not s_data.empty and len(s_data) > 3:
                xray_candidates.append({
                    "df": s_data, "label": "Swift BAT 15–150 keV", "ylabel": "Swift/BAT\n(count/s/cm²)",
                    "color": "#d62728", "marker": "s", "invert": False
                })

        # ── 2. Gather Cleaned Optical Channels ────────────────────────────────
        optical_candidates = []
        if not ztf_df.empty:
            for b_char, b_color in [('g', '#2ca02c'), ('r', '#e377c2')]:
                for col in ("filtercode", "band", "filter", "F"):
                    if col in ztf_df.columns:
                        z_data = clean_and_weight_bin_data(ztf_df, 'mjd', 'mag', 'magerr', filter_col=col, filter_pattern=b_char)
                        if not z_data.empty and len(z_data) > 3:
                            optical_candidates.append({
                                "df": z_data, "label": f"ZTF {b_char}-band", "ylabel": f"ZTF {b_char}\n(Mag)",
                                "color": b_color, "marker": "^", "invert": True
                            })
                            break
                            
        if not atlas_df.empty:
            for b_char, b_color in [('o', '#ffbb78'), ('c', '#a179cf')]:
                for col in ("filt", "filter", "band"):
                    if col in atlas_df.columns:
                        a_data = clean_and_weight_bin_data(atlas_df, 'mjd', 'uJy', 'duJy', filter_col=col, filter_pattern=b_char, min_snr=2.0)
                        if not a_data.empty and len(a_data) > 3:
                            optical_candidates.append({
                                "df": a_data, "label": f"ATLAS {b_char}-band", "ylabel": f"ATLAS {b_char}\n(μJy)",
                                "color": b_color, "marker": "v", "invert": False
                            })
                            break

        # ── 3. Plot if the Source meets the 2 X-ray and 2 Optical Band condition ──
        if len(xray_candidates) >= 2 and len(optical_candidates) >= 2:
            print(f"[#] Plotting Cleaned Gold Candidate: {source_id}")
            
            fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(11, 12), sharex=True)
            fig.suptitle(f"Isolated Multi-Band Cleaned Light Curve: {source_id}", fontsize=13, fontweight='bold', y=0.96)
            
            selected_plots = [
                xray_candidates[0],    # Panel 1
                xray_candidates[1],    # Panel 2
                optical_candidates[0], # Panel 3
                optical_candidates[1]  # Panel 4
            ]
            
            for idx, p_config in enumerate(selected_plots):
                ax = axes[idx]
                b_df = p_config["df"]
                
                # Draw the binned path with error bars
                ax.errorbar(
                    b_df['mjd'], b_df['val'], yerr=b_df['err'],
                    fmt=p_config["marker"], markersize=4, color=p_config["color"],
                    alpha=0.85, linestyle='-', linewidth=0.8, label=p_config["label"], mec='none'
                )
                
                ax.set_ylabel(p_config["ylabel"], fontsize=9, fontweight='semibold')
                ax.legend(loc='upper right', frameon=True, fontsize=9)
                ax.grid(True, linestyle=':', alpha=0.5, color='#cbd5e1')
                ax.tick_params(axis='both', which='both', labelsize=9, top=True, right=True)
                
                if p_config["invert"]:
                    ax.invert_yaxis()
            
            axes[3].set_xlabel("Time (Modified Julian Day — MJD)", fontsize=11, fontweight='bold')
            plt.subplots_adjust(hspace=0.08)
            
            plot_filename = os.path.join(output_dir, f"{safe_id}_clean_4panel.png")
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"    [+] Saved: {plot_filename}")

if __name__ == "__main__":
    generate_gold_plots(data_dir="newdata", output_dir="plots")