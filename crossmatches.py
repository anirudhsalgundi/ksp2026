# =====================================================
# Imports
# =====================================================
import logging
import os
import ssl
import re
import time
import json
import certifi
from matplotlib import text
import numpy as np
import pandas as pd
import urllib.request
from io import StringIO
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from astropy.coordinates import SkyCoord
import astropy.units as u
from alerce.core import Alerce
from time import perf_counter

# =====================================================
# Configuration
# =====================================================

# Logging configuration
logging.basicConfig(
    filename="crossmatches.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ── Credentials ───────────────────────────────────────────────────────────────
ATLAS_USER = os.environ.get("ATLAS_USER")
ATLAS_PASS = os.environ.get("ATLAS_PASS")

SWIFT_CSV  = os.path.join(BASE_DIR, "Swift_BAT_Transient_Sources.csv")
MAXI_CSV   = os.path.join(BASE_DIR, "MAXI_Sources.csv")

# ── Survey start dates (MJD) ──────────────────────────────────────────────────
# Restricting the ATLAS search window dramatically reduces query time.
# ZTF began: 2018-03-20  → MJD 58196
# ATLAS began: 2015-07-01 → MJD 57204 (but good all-sky from ~2017)
ZTF_START_MJD   = 58196   # 2018-03-20
ATLAS_START_MJD = 57940   # 2017-06-01  (conservative start for reliable coverage)

# ── Search radius ─────────────────────────────────────────────────────────────
CONE_RADIUS_ARCSEC = 5  

# ── File paths ────────────────────────────────────────────────────────────────
# CSV catalogs — expected in the same folder as this config.py
SWIFT_CSV  = os.path.join(BASE_DIR, "Swift_BAT_Transient_Sources.csv")
MAXI_CSV   = os.path.join(BASE_DIR, "MAXI_Sources.csv")

DATA_DIR        = os.path.join(BASE_DIR, "data")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")
LIGHTCURVE_DIR  = os.path.join(DATA_DIR, "lightcurves")

for d in [DATA_DIR, RESULTS_DIR, LIGHTCURVE_DIR, 
          os.path.join(LIGHTCURVE_DIR, "maxi"),
          os.path.join(LIGHTCURVE_DIR, "swift"),
          os.path.join(LIGHTCURVE_DIR, "ztf"),
          os.path.join(LIGHTCURVE_DIR, "atlas")]:
    os.makedirs(d, exist_ok=True)

# ── ATLAS API base URL ────────────────────────────────────────────────────────
ATLAS_BASEURL = "https://fallingstar-data.com/forcedphot"


# =====================================================
# Utility Functions
# =====================================================

# Instantiate clients and network settings globally
alerce_client = Alerce()
ssl_ctx = ssl.create_default_context(cafile=certifi.where())


def http_request(method, url, headers=None, data=None, timeout=30):
    """
        Generic HTTP wrapper used by the ATLAS API.

        Used for:
        - authentication
        - job submission
        - job status polling
        - result retrieval

        Returns:
        status code,
        response body,
        response headers
    """
    req_headers = headers or {}
    req_data = None
    if data is not None:
        req_data = urlencode(data).encode("utf-8")
        req_headers = {**req_headers, "Content-Type": "application/x-www-form-urlencoded"}
    req = Request(url, data=req_data, headers=req_headers, method=method)
    with urlopen(req, context=ssl_ctx, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return resp.status, body, dict(resp.headers)

# =====================================================
# Catalog Loading
# =====================================================

def load_catalogs():
    swift = pd.read_csv(SWIFT_CSV)
    maxi = pd.read_csv(MAXI_CSV)

    for df in [swift, maxi]:
        df['RA J2000 Degs'] = pd.to_numeric(
            df['RA J2000 Degs'], errors='coerce'
        )
        df['Dec J2000 Degs'] = pd.to_numeric(
            df['Dec J2000 Degs'], errors='coerce'
        )

    swift = swift.dropna(
        subset=['RA J2000 Degs', 'Dec J2000 Degs']
    ).reset_index(drop=True)

    maxi = maxi.dropna(
        subset=['RA J2000 Degs', 'Dec J2000 Degs']
    ).reset_index(drop=True)

    swift_coords = SkyCoord(
        ra=swift['RA J2000 Degs'].values * u.deg,
        dec=swift['Dec J2000 Degs'].values * u.deg
    )

    print(f"Loaded {len(maxi)} MAXI sources and {len(swift)} Swift sources.")

    return maxi, swift, swift_coords

# =====================================================
# Swift Functions
# =====================================================

def crossmatch_swift(
    maxi_ra,
    maxi_dec,
    swift_df,
    swift_coords,
    radius_arcsec=CONE_RADIUS_ARCSEC
):
    maxi_coord = SkyCoord(
        ra=maxi_ra * u.deg,
        dec=maxi_dec * u.deg
    )

# match_to_catalog_sky() already returns the nearest neighbour.
# idx corresponds to the minimum angular separation source.
    idx, sep2d, _ = maxi_coord.match_to_catalog_sky(
        swift_coords
    )

    sep = sep2d.arcsec.item()

    if sep <= radius_arcsec:
        return swift_df.iloc[idx], sep

    return None, None

def fetch_swift_lightcurve(source_name):
    failed_path = os.path.join(
    LIGHTCURVE_DIR,
    "swift_failed.txt"
    )
    failed_sources = set()
    if os.path.exists(failed_path):
        with open(failed_path) as f:
            failed_sources = {
                line.strip()
                for line in f
            }
    if source_name in failed_sources:
        return None
    
    os.makedirs(os.path.join(LIGHTCURVE_DIR, "swift"), exist_ok=True)
    save_path = os.path.join(LIGHTCURVE_DIR, "swift", f"{source_name.replace(' ', '_')}.csv")
    if os.path.exists(save_path):
        return pd.read_csv(save_path)

    name_fmt = source_name.replace(' ', '')

    for suffix in ['.lc.txt', '.orbit.lc.txt']:

        url = (
            "https://swift.gsfc.nasa.gov/results/transients/"
            f"{name_fmt}{suffix}"
        )

        t0 = perf_counter()

        try:
            with urllib.request.urlopen(
                url,
                context=ssl_ctx,
                timeout=3
            ) as resp:
                text = resp.read().decode('utf-8')

            dt = perf_counter() - t0

            print(
                f"    SUCCESS {url} "
                f"({dt:.2f}s, {len(text)} bytes)"
            )

            if len(text) > 100 and not text.strip().startswith("<!DOCTYPE"):
                df = pd.read_csv(
                StringIO(text),
                sep=r'\s+',
                comment='#',
                header=None,
                usecols=[0, 1, 2]
                )

                df.columns = [
                    'MJD',
                    'rate_15_50keV',
                    'err_15_50keV'
                ]
                df = df.apply(pd.to_numeric, errors='coerce').dropna()
                print(f"    Saving Swift CSV to: {save_path}")
                df.to_csv(save_path, index=False)
                print(f"  Swift: fetched {len(df)} points for {source_name}")
                return df
        except Exception as e:
            print(
                f"    FAIL {url} "
                f"({perf_counter()-t0:.2f}s)"
                )
            print(f"       {type(e).__name__}: {e}")
            pass  # fail fast, try next URL

    print(f"  Swift fetch completely failed for {source_name}")

    with open(failed_path, "a") as f:
        f.write(source_name + "\n")

    return None

# =====================================================
# MAXI Functions
# =====================================================

def radec_to_maxi_id(ra, dec):
    """
    Convert RA/Dec to MAXI source ID using precise coordinates.
    Format: J<HHMM><+/-><DD><T> where T is tenths of degree in dec.
    e.g. RA=84.727, Dec=26.316 -> J0538+263
    Tries rounded, floor and ceil variants to handle borderline cases.
    """
    coord   = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
    hh      = int(coord.ra.hms.h)
    mm      = int(coord.ra.hms.m)
    dec_abs = abs(coord.dec.deg)
    dd      = int(dec_abs)
    frac    = dec_abs - dd
    sign    = '+' if dec >= 0 else '-'

    t_round = int(round(frac * 10))
    if t_round == 10:
        dd_r = dd + 1
        t_round = 0
    else:
        dd_r = dd

    t_floor = int(np.floor(frac * 10))
    t_ceil  = min(int(np.ceil(frac * 10)), 9)

    id_round = f"J{hh:02d}{mm:02d}{sign}{dd_r:02d}{t_round}"
    id_floor = f"J{hh:02d}{mm:02d}{sign}{dd:02d}{t_floor}"
    id_ceil  = f"J{hh:02d}{mm:02d}{sign}{dd:02d}{t_ceil}"

    return list(dict.fromkeys([id_round, id_floor, id_ceil]))


def fetch_maxi_lightcurve(source_name, ra, dec):
    os.makedirs(os.path.join(LIGHTCURVE_DIR, "maxi"), exist_ok=True)
    save_path = os.path.join(LIGHTCURVE_DIR, "maxi", f"{source_name.replace(' ', '_')}.csv")
    if os.path.exists(save_path):
        return pd.read_csv(save_path)

    possible_ids = radec_to_maxi_id(ra, dec)
    print(f"  MAXI trying IDs: {possible_ids}")

    for target_id in possible_ids:
        base_url = f"http://maxi.riken.jp/star_data/{target_id}/{target_id}"
        for suffix in ['_g_lc_1day_all.dat', '_g_lc_1orb_all.dat']:
            url = base_url + suffix
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=8) as resp:  # ← 8s not 15s
                    text = resp.read().decode('utf-8')
                if len(text) > 100 and not text.strip().startswith('<!'):
                    df = pd.read_csv(
                        StringIO(text), sep=r'\s+', comment='#',
                        names=['MJD', 'rate_2_20keV', 'err_2_20keV',
                               'rate_2_4keV',   'err_2_4keV',
                               'rate_4_10keV',  'err_4_10keV',
                               'rate_10_20keV', 'err_10_20keV']
                    )
                    df = df.apply(pd.to_numeric, errors='coerce').dropna(subset=['MJD'])
                    df.to_csv(save_path, index=False)
                    print(f"  MAXI: fetched {len(df)} points for {source_name} ({target_id})")
                    return df
            except Exception as e:
                print(f"    MAXI tried: {url} -> {type(e).__name__}: {e}")
                continue

    print(f"  MAXI fetch completely failed for {source_name}")
    return None

# =====================================================
# ZTF Functions
# =====================================================

def query_ztf(ra, dec, source_name, radius_arcsec=CONE_RADIUS_ARCSEC):
    """Query ZTF via ALeRCE using query_lightcurve with nested detection extraction."""
    os.makedirs(os.path.join(LIGHTCURVE_DIR, "ztf"), exist_ok=True)
    save_path = os.path.join(LIGHTCURVE_DIR, "ztf", f"{source_name.replace(' ', '_')}.csv")

    # Load from cache
    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
        n_bands = int(df['fid'].nunique()) if 'fid' in df.columns else 0
        oid = df['oid'].iloc[0] if 'oid' in df.columns else None
        return True, oid, n_bands

    try:
        result = alerce_client.query_objects(
            survey="ztf", ra=ra, dec=dec, radius=radius_arcsec
        )
        time.sleep(0.1)

        if result is None or len(result) == 0:
            return False, None, 0

        oid = result.iloc[0]['oid']

        lc_raw = alerce_client.query_lightcurve(oid, format='pandas')
        time.sleep(0.1)

        if lc_raw is None or len(lc_raw) == 0:
            return True, oid, 0

        if 'detections' in lc_raw.columns:
            det_list = lc_raw['detections'].iloc[0]  # list of dicts
            if not det_list:
                return True, oid, 0
            lc = pd.json_normalize(det_list)
        elif 'fid' in lc_raw.columns:
            lc = lc_raw  # already flat
        else:
            return True, oid, 0

        if len(lc) == 0:
            return True, oid, 0

        lc['oid'] = oid
        lc.to_csv(save_path, index=False)

        n_bands = int(lc['fid'].nunique()) if 'fid' in lc.columns else 0
        return True, oid, n_bands

    except Exception as e:
        print(f"  ZTF error for {source_name}: {e}")
        return None, None, 0

# =====================================================
# ATLAS Functions
# =====================================================

def get_atlas_token():
    """Fetches ATLAS forced photometry authorization token."""
    payload = {"username": ATLAS_USER, "password": ATLAS_PASS}
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    status, body, _ = http_request("POST", f"{ATLAS_BASEURL}/api-token-auth/",
                                   headers=headers, data=payload)
    if status != 200:
        raise RuntimeError(f"ATLAS authentication failed with status code: {status}")
    return json.loads(body)["token"]

def submit_atlas_job(ra, dec, token):
    """
    Submit a single ATLAS forced photometry job.
    Returns the task URL string, or None if submission failed.
    Does NOT poll — call collect_atlas_result separately.
    """
    headers = {"Authorization": f"Token {token}", "Accept": "application/json"}
    payload = {
        "ra": float(ra), "dec": float(dec),
        "mjd_min": ATLAS_START_MJD,
        "send_email": False
    }
    try:
        status, body, _ = http_request(
            "POST", f"{ATLAS_BASEURL}/queue/",
            headers=headers, data=payload, timeout=30
        )
        if status == 201:
            return json.loads(body)["url"]
        print(f"    ATLAS submit returned status {status}")
    except Exception as e:
        print(f"    ATLAS submit error: {e}")
    return None

def collect_atlas_result(task_url, source_name, token, save_path):
    """
    Poll an already-submitted ATLAS job until it finishes, then save results.
    Returns (found: bool|None, n_bands: int).
    None = timed out or network error; False = queried, no detections.
    """
    print(f"    ATLAS polling URL: {task_url}")  
    headers = {"Authorization": f"Token {token}", "Accept": "application/json"}

    for attempt in range(120):  # poll up to 10 minutes (120 x 5s)
        time.sleep(5)
        try:
            status, result_body, _ = http_request("GET", task_url, headers=headers, timeout=30)
        except Exception as e:
            print(f"    ATLAS poll error (attempt {attempt+1}) for {source_name}: {e}")
            continue

        if status != 200:
            continue

        result = json.loads(result_body)
        if not result.get("finishtimestamp"):
            continue

        # Job finished
        phot_url = result.get("result_url")
        if not phot_url:
            print(f"  ATLAS: job finished but no result_url for {source_name}")
            return False, 0

        try:
            _, phot_text, _ = http_request("GET", phot_url, headers=headers, timeout=60)
        except Exception as e:
            print(f"  ATLAS: failed to fetch results for {source_name}: {e}")
            return None, 0

        df = pd.read_csv(StringIO(phot_text), sep=r'\s+', comment='#')  # ✅ fixed

        if df.empty:
            print(f"  ATLAS: empty result for {source_name}")
            return False, 0

        if 'uJy' in df.columns and 'duJy' in df.columns:
            df = df[df['uJy'] / df['duJy'] > 3]

        if len(df) == 0:
            print(f"  ATLAS: no detections above S/N>3 for {source_name}")
            return False, 0
        
        logger.info(f"Saving ATLAS CSV to: {save_path}")
        df.to_csv(save_path, index=False)
        n_bands = int(df['F'].nunique()) if 'F' in df.columns else 1
        return True, n_bands

    print(f"  ATLAS timed out for {source_name}")
    return None, 0

# =====================================================
# Classification
# =====================================================

def classify(row):
    """
    Gold:   >= 2 X-ray bands AND >= 2 optical bands
    Silver: >= 1 X-ray band  AND >= 1 optical band (but not Gold)
    Bronze: data in only one wavelength regime
    """
    xray_bands    = row['n_xray_bands']
    optical_bands = row['n_optical_bands']

    in_xray    = xray_bands > 0
    in_optical = optical_bands > 0

    if xray_bands >= 2 and optical_bands >= 2:
        return 'Gold'
    elif in_xray and in_optical:
        return 'Silver'
    elif in_xray or in_optical:
        return 'Bronze'
    else:
        return 'No match'

# =====================================================
# Main Pipeline
# =====================================================

def run_pipeline(base_dir="."):
    DATA_DIR = os.path.join(base_dir, "data")
    maxi_df, swift_df, swift_coords = load_catalogs()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(LIGHTCURVE_DIR, "atlas"), exist_ok=True)

    out_path = os.path.join(
        RESULTS_DIR,
        "crossmatch_results.csv"
    )

    try:
        atlas_token = get_atlas_token()
        logger.info("ATLAS token obtained")
    except Exception as e:
        atlas_token = None
        logger.warning(
            f"Warning: Could not get ATLAS token ({e}). "
            f"ATLAS queries will be skipped."
        )

    results = []
    atlas_jobs = {}

    logger.info(
        "\n=== PHASE 1: X-ray + ZTF crossmatch + "
        "ATLAS job submission ==="
    )

    for i, row in enumerate(
    maxi_df.itertuples(index=False, name=None),
    start=1
    ):
        name = row[0]
        ra   = row[1]
        dec  = row[2]

        logger.info(
            f"\n[{i}/{len(maxi_df)}] "
            f"{name}  (RA={ra:.3f}, Dec={dec:.3f})"
        )

        # ==========================================================
        # TOTAL SOURCE TIMER
        # ==========================================================
        source_start = perf_counter()

        # ==========================================================
        # MAXI
        # ==========================================================
        t0 = perf_counter()

        maxi_lc = fetch_maxi_lightcurve(
            name,
            ra,
            dec
        )

        maxi_time = perf_counter() - t0

        logger.info(
            f"  MAXI fetch took "
            f"{maxi_time:.2f} s"
        )

        n_maxi_bands = (
            2 if maxi_lc is not None and len(maxi_lc) > 0
            else 0
        )

        # ==========================================================
        # SWIFT CROSSMATCH
        # ==========================================================
        t0 = perf_counter()

        swift_match, swift_sep = crossmatch_swift(
            ra,
            dec,
            swift_df,
            swift_coords
        )

        logger.info(
            f"  Swift crossmatch took "
            f"{perf_counter()-t0:.3f} s"
        )

        swift_name = (
            swift_match['Source Name']
            if swift_match is not None
            else None
        )

        n_swift_bands = 0

        if swift_match is not None:

            t0 = perf_counter()

            swift_lc = fetch_swift_lightcurve(
                swift_name
            )

            logger.info(
                f"  Swift LC fetch took "
                f"{perf_counter()-t0:.2f} s"
            )

            n_swift_bands = (
                1
                if swift_lc is not None
                and len(swift_lc) > 0
                else 0
            )

            logger.info(
                f"  Swift match: "
                f"{swift_name} "
                f"({swift_sep:.1f} arcsec)"
            )

        else:
            logger.info("  Swift: no match")

        n_xray_bands = (
            n_maxi_bands +
            n_swift_bands
        )

        # ==========================================================
        # ZTF
        # ==========================================================
        t0 = perf_counter()

        ztf_found, ztf_oid, n_ztf_bands = query_ztf(
            ra,
            dec,
            name
        )

        logger.info(
            f"  ZTF query took "
            f"{perf_counter()-t0:.2f} s"
        )

        logger.info(
            f"  ZTF: "
            f"{'found' if ztf_found else 'not found'} "
            f"({n_ztf_bands} bands)"
        )

        # ==========================================================
        # ATLAS SUBMISSION
        # ==========================================================
        atlas_save_path = os.path.join(
            LIGHTCURVE_DIR,
            "atlas",
            f"{name.replace(' ', '_')}.csv"
        )

        atlas_cached = os.path.exists(
            atlas_save_path
        )

        if (
            not ztf_found
            and atlas_token
            and dec >= -50
        ):

            if atlas_cached:
                logger.info("  ATLAS: already cached")

            else:
                task_url = submit_atlas_job(
                    ra,
                    dec,
                    atlas_token
                )

                if task_url:
                    atlas_jobs[name] = task_url
                    logger.info("  ATLAS: job submitted")
                else:
                    logger.error("  ATLAS: submission failed")
                    

        elif ztf_found:
            logger.info(
                "  ATLAS: skipped "
                "(ZTF data available)"
            )

        elif dec < -50:
            logger.info(
                "  ATLAS: skipped "
                "(dec < -50)"
            )

        # ==========================================================
        # TOTAL TIME
        # ==========================================================
        logger.info(
            f"  TOTAL SOURCE TIME: "
            f"{perf_counter()-source_start:.2f} s"
        )

        results.append({
            'source_name': name,
            'ra': ra,
            'dec': dec,

            'maxi_found': n_maxi_bands > 0,
            'n_maxi_bands': n_maxi_bands,

            'swift_match': swift_name,
            'swift_sep_arcsec': swift_sep,
            'n_swift_bands': n_swift_bands,

            'n_xray_bands': n_xray_bands,

            'ztf_found': ztf_found,
            'ztf_oid': ztf_oid,
            'n_ztf_bands': n_ztf_bands,

            'atlas_found': None,
            'n_atlas_bands': 0,

            'n_optical_bands': n_ztf_bands,

            'classification': None
        })

    logger.info(
        f"\n=== PHASE 1 complete. "
        f"{len(atlas_jobs)} ATLAS jobs submitted. ==="
    )

    if atlas_jobs:
        logger.info(
            "Waiting 30s before polling "
            "ATLAS results..."
        )
        time.sleep(30)

# ── PHASE 2: Save ATLAS job URLs for standalone polling ───────────────────
    logger.info(f"\n=== PHASE 1 complete. {len(atlas_jobs)} ATLAS jobs submitted. ===")

    jobs_path = os.path.join(RESULTS_DIR, "atlas_jobs.json")
    with open(jobs_path, 'w') as f:
        json.dump(atlas_jobs, f, indent=2)
    logger.info(f"ATLAS job URLs saved to {jobs_path}")
    logger.info("Run poll_atlas.py in a new terminal to collect ATLAS results.\n")

    # Load any already-cached ATLAS results (from previous runs)
    results_by_name = {r['source_name']: r for r in results}
    for r in results:
        if r['atlas_found'] is None:
            atlas_save_path = os.path.join(
                LIGHTCURVE_DIR, "atlas", f"{r['source_name'].replace(' ', '_')}.csv"
            )
            if os.path.exists(atlas_save_path):
                df_a    = pd.read_csv(atlas_save_path)
                n_bands = int(df_a['F'].nunique()) if 'F' in df_a.columns else 1
                r['atlas_found']     = True
                r['n_atlas_bands']   = n_bands
                r['n_optical_bands'] += n_bands
            else:
                r['atlas_found']   = False
                r['n_atlas_bands'] = 0

    # ── PHASE 3: Classify all sources (ATLAS pending sources classified without it)
    logger.info("=== PHASE 3: Classifying sources ===")
    for r in results:
        r['classification'] = classify(pd.Series(r))
        logger.info(f"  {r['source_name']}: {r['classification']} "
                    f"| X-ray={r['n_xray_bands']} | optical={r['n_optical_bands']}")

    # Save preliminary results (ATLAS results will be added later)
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)

    logger.info("=" * 60)
    logger.info("PIPELINE EXECUTION COMPLETE")
    logger.info("=" * 60)

    logger.info(f"Total tracked targets: {len(df)}")
    logger.info(f"Saved preliminary results to: {out_path}")

    if atlas_jobs:
        logger.info(
            f"ATLAS jobs submitted: {len(atlas_jobs)}. "
            "Run poll_atlas.py later to collect ATLAS results."
        )

    logger.info("\nCurrent classification breakdown:")
    logger.info(df["classification"].value_counts().to_string())
    logger.info(
    "Results saved without ATLAS crossmatches. "
    "ATLAS results will be appended when poll_atlas.py completes."
    )
    return df

if __name__ == "__main__":
    run_pipeline()