import os
import re
import time
import logging
import io
from io import StringIO, BytesIO
import requests
import pandas as pd
from astroquery.heasarc import Heasarc
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table
import urllib3

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ZTF_API_URL = "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves"
ATLAS_API_URL = "https://fallingstar-data.com/forcedphot"

ZTF_TIMEOUT = 100
TIMEOUT = 45
ATLAS_CHUNK_SIZE = 500.0
DEFAULT_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler('pipeline.log')
file_handler.setFormatter(log_formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

logger = logging.getLogger("PIPELINE")
ztf_logger = logging.getLogger("ZTF")
atlas_logger = logging.getLogger("ATLAS")

def clean_maxi_name_for_url(maxi_catalog_name):
    name = str(maxi_catalog_name).strip()
    name = re.sub(r'^(?:\d+)?MAXI\s*', '', name, flags=re.IGNORECASE)
    name = name.replace(" ", "")
    return name

def fetch_live_maxi_by_coordinates(ra, dec, search_radius_arcmin=20.0):
    heasarc = Heasarc()
    coord = SkyCoord(ra=ra, dec=dec, unit=u.deg, frame='icrs')
    
    logger.info(f"Executing Cone Search at RA={ra:.4f}, Dec={dec:.4f}...")
    
    try:
        radius_str = f"{search_radius_arcmin} arcmin"
        results = heasarc.query_region(coord, catalog='maxigsc7yr', radius=radius_str)
        
        if results is None or len(results) == 0:
            logger.warning(f"Spatial Match Failed: No MAXI counterpart within {radius_str}.")
            return None
            
        for col in results.colnames:
            results.rename_column(col, col.upper())
            
        official_name = results['NAME'][0]
        logger.info(f"Spatial Match Found! Catalog Resolved ID: '{official_name}'")
        
        url_slug = clean_maxi_name_for_url(official_name)
        
        slug_variations = [url_slug]
        if url_slug and url_slug[-1].isdigit():
            base_slug = url_slug[:-1]
            last_digit = int(url_slug[-1])
            for delta in [1, -1]:
                new_digit = (last_digit + delta) % 10
                slug_variations.append(f"{base_slug}{new_digit}")
        
        columns = [
            'MJD', 'flux_2_20', 'err_2_20', 
            'flux_2_4', 'err_2_4', 
            'flux_4_10', 'err_4_10', 
            'flux_10_20', 'err_10_20'
        ]
        
        response = None
        chosen_slug = None
        
        for slug in slug_variations:
            url = f"https://maxi.riken.jp/star_data/{slug}/{slug}_g_lc_1day_all.dat"
            try:
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                res = requests.get(url, headers=DEFAULT_HEADERS, verify=False, timeout=TIMEOUT)
                if res.status_code == 200:
                    response = res
                    chosen_slug = slug
                    break
            except Exception:
                continue
                
        if response is None:
            logger.warning(f"RIKEN Server returned HTTP Status 404 for {url_slug} (and all rounding fallback variants)")
            return None
            
        logger.info(f"Success! Connected via matched URL slug: '{chosen_slug}'")
        df = pd.read_csv(StringIO(response.text), sep=r'\s+', comment='#', names=columns)
        
        df_modern = df[df['MJD'] >= 58119].copy()
        df_modern['datetime'] = pd.to_datetime(df_modern['MJD'] + 2400000.5, unit='D', origin='julian')
        
        logger.info(f"Pulled {len(df_modern)} modern data points for {chosen_slug}.")
        return df_modern

    except Exception as e:
        logger.error(f"Pipeline broke for this target. Error details: {e}")
        return None

def fetch_swift_bat_fits_lightcurves(df_targets, start_mjd=58119.0):
    heasarc = Heasarc()
    lightcurves = {}
    
    logger.info(f"Locating FITS light curves via direct HTTP translation for {len(df_targets)} targets...")
    
    for index, row in df_targets.iterrows():
        target_name = row['NAME'] 
        ra = row['RA']
        dec = row['DEC']
        
        try:
            coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
            bat_table = heasarc.query_region(coord, catalog='swbatmontr', radius=20.0*u.arcmin)
            
            if bat_table is None or len(bat_table) == 0:
                logger.warning(f"[{target_name}] No matching rows found in swbatmontr.")
                continue
            
            raw_catalog_name = bat_table['NAME'][0]
            clean_name = str(raw_catalog_name).replace(" ", "")
            
            base_url = "https://heasarc.gsfc.nasa.gov/docs/swift/results/transients/"
            urls_to_try = [
                f"{base_url}{clean_name}.orbit.lc.fits",
                f"{base_url}weak/{clean_name}.orbit.lc.fits"
            ]
            
            response = None
            for url in urls_to_try:
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                res = requests.get(url, headers=DEFAULT_HEADERS, verify=False, timeout=30)
                if res.status_code == 200:
                    response = res
                    break
            
            if response is None:
                logger.warning(f"[{target_name}] URL resolution failed. Asset does not exist on public directory.")
                continue
                
            fits_table = Table.read(BytesIO(response.content), format='fits', hdu=1)
            df_lc = fits_table.to_pandas()
            
            df_lc.columns = df_lc.columns.str.upper()
            if 'TIME' in df_lc.columns:
                df_lc = df_lc.rename(columns={'TIME': 'MJD'})
            
            df_modern = df_lc[df_lc['MJD'] >= start_mjd].copy()
            logger.info(f"[{target_name}] Swift BAT Success: Parsed {len(df_modern)} data points.")
            lightcurves[target_name] = df_modern
            
        except Exception as e:
            logger.error(f"[{target_name}] Error processing high-energy stream: {e}")
            continue
            
    return lightcurves

def fetch_ztf_light_curve_bounded(source_id, ra, dec, mjd_start=58119.0, mjd_end=None):
    if float(dec) < -30.0:
        ztf_logger.warning(f"ZTF skipped for {source_id}: Declination ({dec}) is below -30.0°")
        return pd.DataFrame()

    params = {
        "POS": f"CIRCLE {float(ra)} {float(dec)} {10.0/3600.0}",
        "BANDNAME": "g,r",
        "FORMAT": "csv",
    }
    
    if mjd_start or mjd_end:
        t_start = float(mjd_start) if mjd_start else "-Inf"
        t_end = float(mjd_end) if mjd_end else "Inf"
        params["TIME"] = f"{t_start} {t_end}"

    try:
        r = requests.get(ZTF_API_URL, params=params, headers=DEFAULT_HEADERS, timeout=ZTF_TIMEOUT)
        
        if r.status_code == 200 and "mjd" in r.text.lower():
            df = pd.read_csv(StringIO(r.text), comment='\\')
            
            if df.empty:
                ztf_logger.info(f"API returned empty table for {source_id}")
                return pd.DataFrame()

            df.columns = df.columns.str.upper()
            ztf_logger.info(f"{len(df)} rows parsed for {source_id}")
            return df
        else:
            ztf_logger.warning(f"No valid light curve returned for {source_id} (Status: {r.status_code})")
            return pd.DataFrame()
            
    except Exception as e:
        ztf_logger.error(f"Extraction failed for {source_id}: {e}")
        return pd.DataFrame()

def fetch_atlas_forced_photometry_memory(source_id, ra, dec, mjd_start=57200.0, mjd_end=None):
    if ra == 0.0 and dec == 0.0:
        atlas_logger.warning(f"ATLAS skipped for {source_id}: Coordinates are 0.0, 0.0")
        return pd.DataFrame()

    username = os.environ.get("ATLAS_USER")
    password = os.environ.get("ATLAS_PASSWORD")
    if not username or not password:
        atlas_logger.error("ATLAS skipped: ATLAS_USER / ATLAS_PASSWORD environment variables not set.")
        return pd.DataFrame()

    atlas_logger.info(f"Authenticating as {username}...")
    try:
        resp = requests.post(
            url=f"{ATLAS_API_URL}/api-token-auth/",
            data={"username": username, "password": password},
            timeout=TIMEOUT
        )
    except Exception as e:
        atlas_logger.error(f"Auth request error for {source_id}: {e}")
        return pd.DataFrame()

    if resp.status_code != 200:
        atlas_logger.error(f"Auth failed for {source_id} (HTTP {resp.status_code})")
        return pd.DataFrame()

    headers = {"Authorization": f"Token {resp.json()['token']}", "Accept": "application/json"}
    atlas_logger.info("Authentication verified successfully")

    if mjd_end is None:
        mjd_end = (time.time() / 86400.0) + 40587.0  

    all_dfs = []

    def _fetch_chunk(chunk_mjd_min, chunk_mjd_max, chunk_idx, total_chunks):
        payload = {
            "ra": float(ra), "dec": float(dec),
            "mjd_min": float(chunk_mjd_min),
            "mjd_max": float(chunk_mjd_max)
        }
        
        atlas_logger.info(f"Chunk {chunk_idx}/{total_chunks}: MJD {payload['mjd_min']:.1f} to {payload['mjd_max']:.1f}...")

        task_url = None
        while not task_url:
            with requests.Session() as s:
                resp = s.post(f"{ATLAS_API_URL}/queue/", headers=headers, data=payload, timeout=TIMEOUT)

                if resp.status_code == 201:
                    task_url = resp.json()["url"]
                elif resp.status_code == 429:
                    msg = resp.json().get("detail", "")
                    t_sec = re.findall(r"available in (\d+) seconds", msg)
                    t_min = re.findall(r"available in (\d+) minutes", msg)
                    waittime = int(t_sec[0]) if t_sec else (int(t_min[0]) * 60 if t_min else 10)
                    atlas_logger.info(f"Rate-limited, pausing for {waittime}s...")
                    time.sleep(waittime)
                else:
                    atlas_logger.error(f"Failure HTTP {resp.status_code}: {resp.text[:100]}")
                    return pd.DataFrame()
                    
        result_url = None
        for poll_n in range(1, 61):
            with requests.Session() as s:
                try:
                    resp = s.get(task_url, headers=headers, timeout=TIMEOUT)
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
            requests.delete(task_url, headers=headers, timeout=TIMEOUT)
            atlas_logger.warning("Stream chunk process timed out.")
            return pd.DataFrame()

        with requests.Session() as s:
            textdata = s.get(result_url, headers=headers, timeout=TIMEOUT).text
            s.delete(task_url, headers=headers, timeout=TIMEOUT)

        if not textdata.strip(): 
            return pd.DataFrame()

        try:
            return pd.read_csv(StringIO(textdata.replace("###", "")), sep=r"\s+")
        except Exception:
            return pd.DataFrame()

    current_start = float(mjd_start)
    total_chunks = int((float(mjd_end) - current_start) // ATLAS_CHUNK_SIZE) + 1
    current_chunk = 1

    while current_start < float(mjd_end):
        current_end = min(current_start + ATLAS_CHUNK_SIZE, float(mjd_end))
        
        df_chunk = _fetch_chunk(current_start, current_end, current_chunk, total_chunks)
        if not df_chunk.empty:
            all_dfs.append(df_chunk)
            
        current_start += ATLAS_CHUNK_SIZE
        current_chunk += 1
        
    if not all_dfs:
        atlas_logger.warning(f"No data rows found across all segments for {source_id}.")
        return pd.DataFrame()

    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df.columns = final_df.columns.str.upper()
    
    if "MJD" in final_df.columns:
        final_df = final_df.sort_values(by="MJD").drop_duplicates()

    atlas_logger.info(f"Completed. Returning {len(final_df)} total footprints for {source_id}")
    return final_df

def resume_pipeline():
    heasarc = Heasarc()
    
    logger.info("Fetching full master xrbcat catalog to cross-reference with local files...")
    try:
        full_catalog = heasarc.query_region(spatial='all-sky', catalog='xrbcat')
        df_xrb = full_catalog.to_pandas()
        df_xrb.columns = df_xrb.columns.str.upper()
        logger.info(f"Loaded master catalog with {len(df_xrb)} items.")
    except Exception as e:
        logger.error(f"CRITICAL: Failed to query HEASARC xrbcat. Error: {e}")
        return

    processing_queue = []
    atlas_fallback_queue = []

    logger.info("SCANNING LOCAL DISK FOR MAXI FILES")

    for index, row in df_xrb.iterrows():
        raw_name = row.get('NAME', f"Unknown_{index}")
        clean_filename = str(raw_name).strip().replace(" ", "_").replace("/", "-")
        ra = row.get('RA')
        dec = row.get('DEC')

        if pd.isna(ra) or pd.isna(dec):
            continue

        maxi_path = os.path.join(OUTPUT_DIR, f"{clean_filename}_maxi.csv")
        
        if os.path.exists(maxi_path):
            target_entry = {
                'NAME': raw_name,
                'RA': float(ra),
                'DEC': float(dec),
                'FILE_LABEL': clean_filename
            }
            processing_queue.append(target_entry)

    logger.info(f"Found {len(processing_queue)} targets with local MAXI data. Building execution queue...")

    if not processing_queue:
        logger.warning("Master queue empty (No local MAXI csv files found). Halting.")
        return

    df_batch = pd.DataFrame(processing_queue)

    logger.info("Launching Optical ZTF Queries...")
    
    for item in processing_queue:
        source_name = item['NAME']
        file_label = item['FILE_LABEL']
        
        ztf_path = os.path.join(OUTPUT_DIR, f"{file_label}_ztf.csv")
        atlas_path = os.path.join(OUTPUT_DIR, f"{file_label}_atlas.csv")
        
        if os.path.exists(ztf_path):
            logger.info(f"Skipping ZTF: '{ztf_path}' already exists.")
            continue
        if os.path.exists(atlas_path):
            logger.info(f"Skipping ZTF/ATLAS: '{atlas_path}' already exists.")
            continue
            
        df_ztf = fetch_ztf_light_curve_bounded(source_name, item['RA'], item['DEC'])
        
        if df_ztf is not None and not df_ztf.empty:
            df_ztf.to_csv(ztf_path, index=False)
            logger.info(f"ZTF optical stream stored for {source_name}")
        else:
            logger.warning(f"No ZTF data coverage for [{source_name}]. Routing to Phase 2 (ATLAS Fallback Queue).")
            atlas_fallback_queue.append(item)

    logger.info("Checking Swift FITS crossmatches...")
    
    swift_targets_to_run = []
    for index, row in df_batch.iterrows():
        swift_path = os.path.join(OUTPUT_DIR, f"{row['FILE_LABEL']}_swift.csv")
        if os.path.exists(swift_path):
            logger.info(f"Skipping Swift: '{swift_path}' already exists.")
        else:
            swift_targets_to_run.append(row)
            
    if swift_targets_to_run:
        df_swift_batch = pd.DataFrame(swift_targets_to_run)
        swift_lightcurves = fetch_swift_bat_fits_lightcurves(df_swift_batch)

        for target_name, df_swift in swift_lightcurves.items():
            if df_swift is not None and not df_swift.empty:
                matched_row = df_swift_batch[df_swift_batch['NAME'] == target_name]
                if not matched_row.empty:
                    file_label = matched_row['FILE_LABEL'].values[0]
                    swift_path = os.path.join(OUTPUT_DIR, f"{file_label}_swift.csv")
                    df_swift.to_csv(swift_path, index=False)
                    logger.info(f"Swift data written to {swift_path}")
    else:
        logger.info("All queued targets already have Swift data on disk.")

    logger.info("STARTING PHASE 2: ASYNCHRONOUS ATLAS FORCED PHOTOMETRY")
    logger.info(f"Queue size: {len(atlas_fallback_queue)} targets requiring deeper processing.")

    for item in atlas_fallback_queue:
        source_name = item['NAME']
        file_label = item['FILE_LABEL']
        
        atlas_path = os.path.join(OUTPUT_DIR, f"{file_label}_atlas.csv")
        if os.path.exists(atlas_path):
            logger.info(f"Skipping ATLAS: '{atlas_path}' already exists.")
            continue
        
        logger.info(f"Submitting async ATLAS request for [{source_name}]...")
        df_atlas = fetch_atlas_forced_photometry_memory(source_name, item['RA'], item['DEC'])
        
        if df_atlas is not None and not df_atlas.empty:
            df_atlas.to_csv(atlas_path, index=False)
            logger.info(f"ATLAS forced data written to {atlas_path}")

    logger.info("Resume Pipeline complete. All downstream streams written to the 'data/' folder.")

if __name__ == "__main__":
    resume_pipeline()