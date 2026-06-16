import io
import os, re, time
import pandas as pd
import requests
from io import StringIO

from bs4 import BeautifulSoup
from astropy.coordinates import SkyCoord



def fetch_maxi_catalog_from_slist(output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "maxi_slist_catalog.csv")

    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 5:
        return pd.read_csv(csv_path, encoding="utf-8-sig")

    print("Scraping RIKEN slist.html for master catalog …")
    url     = "https://maxi.riken.jp/top/slist.html"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(url, headers=headers, timeout=20)
        soup     = BeautifulSoup(response.text, "html.parser")
    except Exception as e:
        print(f"[-] Failed to scrape master list: {e}")
        return pd.DataFrame()
    a_tags  = [a for a in soup.find_all("a", href=True) if "star_data" in a["href"]]
    targets = []

    for idx, a_tag in enumerate(a_tags, start=1):
        href  = a_tag["href"]
        parts = [p for p in href.split("/") if p]
        riken_target_name = parts[-2] if len(parts) >= 2 else parts[-1].replace(".html", "")

        raw_text   = a_tag.text.strip()
        clean_text = raw_text.encode("ascii", "ignore").decode("ascii").strip()
        source_id  = clean_text if clean_text else f"MAXI {riken_target_name}"

        print(f"    [Catalog {idx}/{len(a_tags)}] Resolving: {source_id} …", end="\r")

        ra, dec = None, None
        try:
            time.sleep(0.5)
            coord   = SkyCoord.from_name(source_id)
            ra, dec = coord.ra.deg, coord.dec.deg
        except Exception:
            match = re.search(r"J(\d{2})(\d{2})([+-]\d+)", riken_target_name)
            if match:
                h, m, d_raw = match.groups()
                d = d_raw[:-1] + "." + d_raw[-1]
                try:
                    coord   = SkyCoord(f"{h}h{m}m {d}d", frame="icrs")
                    ra, dec = coord.ra.deg, coord.dec.deg
                except Exception:
                    pass

        if ra is None or dec is None:
            ra, dec = 0.0, 0.0

        targets.append({
            "source_id":         source_id,
            "riken_target_name": riken_target_name,
            "ra":  ra,
            "dec": dec,
        })

    print("\n Catalog built.")
    df = pd.DataFrame(targets).drop_duplicates(subset=["source_id"])
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return df


def fetch_maxi_per_scan_lc(source_id, riken_target_name, output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    safe_id  = (str(source_id).strip()
                .replace(" ", "_").replace("+", "p").replace("-", "m"))
    csv_path = os.path.join(output_dir, f"{safe_id}_maxi_lc.csv")

    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            return df if not df.empty else pd.DataFrame()
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    urls_to_try = [
        f"http://maxi.riken.jp/star_data/{riken_target_name}/{riken_target_name}_gsc_orbit_all.dat",
        f"http://maxi.riken.jp/star_data/{riken_target_name}/{riken_target_name}_gsc_1d_all.dat",
        f"http://maxi.riken.jp/star_data/{riken_target_name}/{riken_target_name}_g_lc_1day_all.dat",
        f"https://maxi.riken.jp/star_data/{riken_target_name}/{riken_target_name}_gsc_orbit_all.txt",
        f"https://maxi.riken.jp/star_data/{riken_target_name}/{riken_target_name}_gsc_1d_all.txt",
    ]

    col_names = [
        "mjd",
        "flux_2_20", "err_2_20",
        "flux_2_4",  "err_2_4",
        "flux_4_10", "err_4_10",
        "flux_10_20","err_10_20",
    ]
    for url in urls_to_try:
        try:
            time.sleep(0.5)
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
            if r.status_code == 200 and "<html" not in r.text[:500].lower():
                lines = [l.strip() for l in r.text.splitlines()
                         if l.strip() and not l.startswith("#")]
                if not lines:
                    continue

                df = pd.read_csv(StringIO("\n".join(lines)), sep=r"\s+", header=None)
                df.columns = col_names[: len(df.columns)]

                for c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

                df = (df[df["flux_2_20"] > -900]
                        .dropna(subset=["mjd", "flux_2_20"])
                        .reset_index(drop=True))

                if not df.empty:
                    df.to_csv(csv_path, index=False)
                    print(f"    [+] MAXI: {len(df)} rows for {source_id}")
                    return df
        except Exception:
            continue
    pd.DataFrame().to_csv(csv_path, index=False)
    return pd.DataFrame()

def cross_match_swift_heasarc(source_id, ra, dec, radius_arcsec=30, output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    safe_source_id = str(source_id).strip().replace(" ", "_")
    csv_path = os.path.join(output_dir, f"{safe_source_id}_swift.csv")

    if os.path.exists(csv_path):
        try:
            return pd.read_csv(csv_path)
        except Exception:
            return pd.DataFrame()

    print(f"Querying Swift BAT Transient Monitor for: {source_id}")
    bat_name = str(source_id).replace(" ", "").replace("+", "p").replace("-", "m")
    candidate_urls = [
        f"https://swift.gsfc.nasa.gov/results/transients/weak/{bat_name}.lc.txt",
        f"https://swift.gsfc.nasa.gov/results/transients/{bat_name}.lc.txt",
    ]
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Referer":    "https://swift.gsfc.nasa.gov/",
    }

    response = None
    for url in candidate_urls:
        try:
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code == 200 and len(r.text.strip()) > 0:
                response = r
                print(f"    [+] Found Swift BAT data at: {url}")
                break
        except Exception as e:
            print(f"    [!] Request failed: {e}")
            continue

    if response is None:
        print(f"    [-] No Swift BAT data for {source_id}.")
        pd.DataFrame().to_csv(csv_path, index=False)
        return pd.DataFrame()

    try:
        lines = response.text.splitlines()
        data_lines = [l.strip() for l in lines
                      if l.strip() and not l.startswith("!") and not l.startswith("#")]
        if not data_lines:
            pd.DataFrame().to_csv(csv_path, index=False)
            return pd.DataFrame()

        df = pd.read_csv(StringIO("\n".join(data_lines)), sep=r"\s+", header=None)
        df = df.iloc[:, [0, 1, 2]]
        df.columns = ["mjd", "count_rate", "count_rate_err"]
        df["count_rate"]     = pd.to_numeric(df["count_rate"],     errors="coerce")
        df["count_rate_err"] = pd.to_numeric(df["count_rate_err"], errors="coerce")
        df["mjd"]            = pd.to_numeric(df["mjd"],            errors="coerce")
        df = df.dropna(subset=["mjd", "count_rate"])
        df = df[df["count_rate"] > -90].reset_index(drop=True)

        df.to_csv(csv_path, index=False)
        print(f"Swift/BAT: {len(df)} points for {source_id}.")
        return df

    except Exception as e:
        print(f"Swift BAT parse failed: {e}")
        pd.DataFrame().to_csv(csv_path, index=False)
        return pd.DataFrame()


def fetch_ztf_light_curve_bounded(source_id, ra, dec,
                                  mjd_start=None, mjd_end=None, output_dir="data"):
    if float(dec) < -30.0:
        return pd.DataFrame()

    os.makedirs(output_dir, exist_ok=True)
    safe_id  = str(source_id).strip().replace(" ", "_")
    csv_path = os.path.join(output_dir, f"{safe_id}_ztf.csv")

    if os.path.exists(csv_path):
        try:
            return pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    try:
        url    = "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves"
        params = {
            "POS":      f"CIRCLE {float(ra)} {float(dec)} {10.0/3600.0}",
            "BANDNAME": "g,r",
            "FORMAT":   "csv",
        }
        r = requests.get(url, params=params,
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=45)
        if r.status_code == 200 and "mjd" in r.text.lower():
            clean = "\n".join([l for l in r.text.splitlines() if not l.startswith("\\")])
            df    = pd.read_csv(StringIO(clean))
            if mjd_start:
                df = df[df["mjd"] >= float(mjd_start)]
            if mjd_end:
                df = df[df["mjd"] <= float(mjd_end)]
            df.to_csv(csv_path, index=False)
            print(f"ZTF: {len(df)} rows for {source_id}")
            return df
    except Exception:
        pass

    return pd.DataFrame()


def fetch_atlas_forced_photometry(source_id, ra, dec,
                                  mjd_start=None, mjd_end=None, output_dir="data"):
                                  
    if ra == 0.0 and dec == 0.0:
        print(f"ATLAS skipped for {source_id}: Coordinates are 0.0, 0.0")
        return pd.DataFrame()
        
    os.makedirs(output_dir, exist_ok=True)
    safe_id  = str(source_id).strip().replace(" ", "_")
    csv_path = os.path.join(output_dir, f"{safe_id}_atlas.csv")

    if os.path.exists(csv_path):
        try:
            return pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    def _save_empty_and_return(reason: str) -> pd.DataFrame:
        print(f"ATLAS skipped for {source_id}: {reason}")
        pd.DataFrame().to_csv(csv_path, index=False)
        return pd.DataFrame()

    username = os.environ.get("ATLAS_USER")
    password = os.environ.get("ATLAS_PASSWORD")
    if not username or not password:
        return _save_empty_and_return("ATLAS_USER / ATLAS_PASSWORD not set")

    BASEURL = "https://fallingstar-data.com/forcedphot"
    print(f"ATLAS: authenticating as {username} …")
    try:
        resp = requests.post(
            url=f"{BASEURL}/api-token-auth/",
            data={"username": username, "password": password},
        )
    except Exception as e:
        return _save_empty_and_return(f"auth request failed: {e}")

    if resp.status_code != 200:
        return _save_empty_and_return(f"auth returned HTTP {resp.status_code}")

    headers = {"Authorization": f"Token {resp.json()['token']}", "Accept": "application/json"}
    print(f"ATLAS: authenticated OK")

    if mjd_start is None:
        mjd_start = 57200.0
    if mjd_end is None:
        mjd_end = (time.time() / 86400.0) + 40587.0 

    chunk_size = 730.0 
    all_dfs = []

    def _fetch_chunk(chunk_mjd_min, chunk_mjd_max, chunk_idx, total_chunks):
        payload = {
            "ra": float(ra), "dec": float(dec),
            "mjd_min": float(chunk_mjd_min),
            "mjd_max": float(chunk_mjd_max)
        }
        
        print(f"\n ATLAS Chunk {chunk_idx}/{total_chunks}: "
              f"MJD {payload['mjd_min']:.1f} to {payload['mjd_max']:.1f} ...")

        task_url = None
        while not task_url:
            with requests.Session() as s:
                resp = s.post(f"{BASEURL}/queue/", headers=headers, data=payload)

                if resp.status_code == 201:
                    task_url = resp.json()["url"]
                elif resp.status_code == 429:
                    msg = resp.json().get("detail", "")
                    t_sec = re.findall(r"available in (\d+) seconds", msg)
                    t_min = re.findall(r"available in (\d+) minutes", msg)
                    waittime = int(t_sec[0]) if t_sec else (int(t_min[0]) * 60 if t_min else 10)
                    print(f"ATLAS rate-limited waiting {waittime}s …", end="\r")
                    time.sleep(waittime)
                else:
                    print(f"failed HTTP {resp.status_code}: {resp.text[:100]}")
                    return pd.DataFrame()
        result_url = None
        for poll_n in range(1, 61):
            with requests.Session() as s:
                try:
                    resp = s.get(task_url, headers=headers)
                except Exception:
                    time.sleep(10)
                    continue

                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("finishtimestamp"):
                        result_url = data["result_url"]
                        break
            time.sleep(10)

        if not result_url:
            requests.delete(task_url, headers=headers)
            print("    [-] Chunk timed out.")
            return pd.DataFrame()

        with requests.Session() as s:
            textdata = s.get(result_url, headers=headers).text
            s.delete(task_url, headers=headers)

        if not textdata.strip(): return pd.DataFrame()

        try:
            return pd.read_csv(io.StringIO(textdata.replace("###", "")), sep=r"\s+")
        except Exception:
            return pd.DataFrame()

    current_start = float(mjd_start)
    total_chunks = int((float(mjd_end) - current_start) // chunk_size) + 1
    current_chunk = 1

    while current_start < float(mjd_end):
        current_end = min(current_start + chunk_size, float(mjd_end))
        
        df_chunk = _fetch_chunk(current_start, current_end, current_chunk, total_chunks)
        if not df_chunk.empty:
            all_dfs.append(df_chunk)
            
        current_start += chunk_size
        current_chunk += 1
    if not all_dfs:
        return _save_empty_and_return("no data rows returned across any chunks")

    final_df = pd.concat(all_dfs, ignore_index=True)
    if "mjd" in final_df.columns:
        final_df = final_df.sort_values(by="mjd").drop_duplicates()

    final_df.to_csv(csv_path, index=False)
    print(f"\n ATLAS: Successfully merged and saved {len(final_df)} total rows for {source_id}")
    
    return final_df

def _count_xray_bands(maxi_df: pd.DataFrame, swift_df: pd.DataFrame) -> int:

    band_count = 0

    if not maxi_df.empty:
        maxi_sub_bands = ["flux_2_4", "flux_4_10", "flux_10_20"]
        for b in maxi_sub_bands:
            if b in maxi_df.columns and maxi_df[b].dropna().ne(0).any():
                band_count += 1
        if band_count == 0 and "flux_2_20" in maxi_df.columns:
            if maxi_df["flux_2_20"].dropna().ne(0).any():
                band_count += 1

    if not swift_df.empty:
        band_count += 1 

    return band_count


def _count_optical_bands(ztf_df: pd.DataFrame, atlas_df: pd.DataFrame) -> int:

    bands = set()
    if not ztf_df.empty:
        for col in ("filtercode", "band", "filter","F"):
            if col in ztf_df.columns:
                bands.update(ztf_df[col].dropna().unique())
                break
    if not atlas_df.empty:
        for col in ("filt", "filter", "band"):
            if col in atlas_df.columns:
                bands.update(atlas_df[col].dropna().unique())
                break

    return len(bands)

def categorise_source(
    source_id: str,
    maxi_df:   pd.DataFrame,
    swift_df:  pd.DataFrame,
    ztf_df:    pd.DataFrame,
    atlas_df:  pd.DataFrame,
) -> dict:

    has_maxi   = not maxi_df.empty
    has_swift  = not swift_df.empty
    has_xray   = has_maxi or has_swift
    has_ztf    = not ztf_df.empty
    has_atlas  = not atlas_df.empty
    has_optical= has_ztf or has_atlas

    n_xray_bands   = _count_xray_bands(maxi_df, swift_df)
    n_optical_bands= _count_optical_bands(ztf_df, atlas_df)

    if (has_maxi and has_optical
            and n_xray_bands >= 2 and n_optical_bands >= 2):
        tier = "Gold"

    elif has_xray and has_optical:
        tier = "Silver"

    else:
        tier = "Bronze"

    return {
        "source_id":       source_id,
        "tier":            tier,
        "has_maxi":        has_maxi,
        "has_swift":       has_swift,
        "has_ztf":         has_ztf,
        "has_atlas":       has_atlas,
        "n_xray_bands":    n_xray_bands,
        "n_optical_bands": n_optical_bands,
        "maxi_rows":       len(maxi_df),
        "swift_rows":      len(swift_df),
        "ztf_rows":        len(ztf_df),
        "atlas_rows":      len(atlas_df),
    }


def run_pipeline(
    output_dir:   str   = "newdata",
    mjd_start:    float = None,
    mjd_end:      float = None,
    max_sources:  int   = None,
) -> pd.DataFrame:

    os.makedirs(output_dir, exist_ok=True)

    catalog = fetch_maxi_catalog_from_slist(output_dir=output_dir)
    if catalog.empty:
        print("MAXI catalog is empty, aborting.")
        return pd.DataFrame()

    if max_sources:
        catalog = catalog.head(max_sources)

    n = len(catalog)
    sources_data = [] 
    print("\n" + "="*50)
    print("Phase 1: Fetching MAXI, Swift, and ZTF data")
    print("="*50)
    
    for i, row in catalog.iterrows():
        source_id         = row["source_id"]
        riken_target_name = row["riken_target_name"]
        ra                = row["ra"]
        dec               = row["dec"]

        print(f"\n[{i+1}/{n}] Fast Processing: {source_id}")

        maxi_df = fetch_maxi_per_scan_lc(
            source_id, riken_target_name, output_dir=output_dir
        )
        swift_df = cross_match_swift_heasarc(
            source_id, ra, dec, output_dir=output_dir
        )
        ztf_df = fetch_ztf_light_curve_bounded(
            source_id, ra, dec,
            mjd_start=mjd_start, mjd_end=mjd_end,
            output_dir=output_dir,
        )
        sources_data.append({
            "source_id":   source_id,
            "ra":          ra,
            "dec":         dec,
            "maxi_df":     maxi_df,
            "swift_df":    swift_df,
            "ztf_df":      ztf_df,
            "atlas_df":    pd.DataFrame(),
            "needs_atlas": ztf_df.empty
        })

    atlas_queue = [s for s in sources_data if s["needs_atlas"]]
    
    if atlas_queue:
        print("\n" + "="*50)
        print(f"PHASE 2: Fetching ATLAS fallback for {len(atlas_queue)} sources")
        print("="*50)
        
        for i, sdata in enumerate(atlas_queue):
            source_id = sdata["source_id"]
            ra        = sdata["ra"]
            dec       = sdata["dec"]
            
            print(f"\n[ATLAS {i+1}/{len(atlas_queue)}] Queuing: {source_id}")
            
            atlas_df = fetch_atlas_forced_photometry(
                source_id, ra, dec,
                mjd_start=mjd_start, mjd_end=mjd_end,
                output_dir=output_dir,
            )
            sdata["atlas_df"] = atlas_df
    else:
        print("\n" + "="*50)
        print("Phase 2: Skipped (No sources require ATLAS fallback)")
        print("="*50)


    print("\n" + "="*50)
    print("Phase 3: Categorising Sources")
    print("="*50)
    
    results = []
    for sdata in sources_data:
        result = categorise_source(
            sdata["source_id"], 
            sdata["maxi_df"], 
            sdata["swift_df"], 
            sdata["ztf_df"], 
            sdata["atlas_df"]
        )
        results.append(result)

        print(
            f"    → {result['source_id'][:15]:<15} | Tier: {result['tier']:6s} | "
            f"X-ray: {result['n_xray_bands']} | Opt: {result['n_optical_bands']} | "
            f"Rows (M:{result['maxi_rows']} S:{result['swift_rows']} Z:{result['ztf_rows']} A:{result['atlas_rows']})"
        )

    summary_df = pd.DataFrame(results)
    summary_path = os.path.join(output_dir, "pipeline_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n Summary saved → {summary_path}")
    print("\n" + "="*50)
    print("CATEGORISATION SUMMARY")
    print("="*50)
    for tier in ["Gold", "Silver", "Bronze"]:
        count = (summary_df["tier"] == tier).sum()
        print(f"  {tier:8s}: {count:>4d} sources")
    print("="*50)

    return summary_df

if __name__ == "__main__":
    df = run_pipeline(
        output_dir  = "newdata",
        mjd_start   = None,  
        mjd_end     = None,
        max_sources = None,
    )
    print(df)