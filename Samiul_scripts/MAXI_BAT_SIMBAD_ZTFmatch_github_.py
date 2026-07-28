#!/usr/bin/env python
# coding: utf-8

# ### Section 0
# ## Loaing Necessary Python Libraries

# In[2]:


pip install astroquery -q


# In[3]:


import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
from astroquery.xmatch import XMatch
from astroquery.simbad import Simbad
import requests
from alerce.core import Alerce
client = Alerce()

FITS_PATH    = "BAT_catalog.fits"


# ### Section 1
# ## Understanding the FITS structure

# In[4]:


# STEP 1: Read the FITS structure
fits_path = FITS_PATH

with fits.open(fits_path) as hdul:
    print("--- FITS Extensions Structure ---")
    hdul.info()


# In[5]:


bat_table = Table.read(fits_path, hdu=1)
bat_df = bat_table.to_pandas()
df=bat_df 
# Preview the column names and the first 5 rows
print("Columns in BAT catalog:", bat_df.columns.tolist())
bat_df.head()


# In[7]:


for col in bat_df.columns:
    # If the column contains byte strings, decode them to standard text strings
    if bat_df[col].dtype == object:
        bat_df[col] = bat_df[col].apply(lambda x: x.decode('utf-8').strip() if isinstance(x, bytes) else (x.strip() if isinstance(x, str) else x))

# Show the cleaned result
print("Cleaned column columns check:")
print(bat_df[['NAME','ALT_NAME', 'RA_OBJ', 'DEC_OBJ']].head())


# ### Section 2
# ## Searching Maxi

# In[8]:


def ra_dec_to_maxi_id(ra_deg: float, dec_deg: float) -> str:
    """
    Convert decimal RA/Dec degrees to a MAXI J-name string.Format: J + HHMM + sign + DDd
      HHMM  = zero-padded hours (HH) and minutes (MM), truncated (not rounded)
      DDd   = absolute degrees (DD) and tenths (d), truncated
      sign  = '+' or '-'
    """
    # RA: degrees → total hours, then split into HH and MM 
    ra_hours_total = ra_deg / 15.0          # 1 hour = 15 degrees
    hh = int(ra_hours_total)               # integer hours
    mm = int((ra_hours_total - hh) * 60)   # integer minutes (truncated)
    ra_str = f'{hh:02d}{mm:02d}'           

    # Dec: degrees → sign, integer degrees, and tenths 
    sign = '+' if dec_deg >= 0 else '-'
    abs_dec = abs(dec_deg)
    dd = int(abs_dec)                      # integer degrees
    d  = int((abs_dec - dd) * 10)          # tenths of a degree (truncated)
    dec_str = f'{dd:02d}{d}'               

    return f'J{ra_str}{sign}{dec_str}'     



# ###### Self Note: Python Type Hints & Function Architecture
# 
# The syntax `(ra_deg: float, dec_deg: float) -> str` uses **Python Type Hints** 
# 
# * **What it do:** it explicitly declare the expected data types for the inputs (`float` variables for coordinates) and the final output (`-> str` means the function will return a text string).
# * **Why we use it:** Type hints do not change how the code runs dynamically, but they provide self-documentation, prevent bugs.

# In[9]:


# --- Main MAXI check loop ---
MAXI_BASE = 'https://maxi.riken.jp/star_data'
MAXI_DELAY = 0.3   # seconds between requests (be polite to the server)
MAXI_TIMEOUT = 5   # seconds before giving up on a single request

# 1. ADD THIS: A custom User-Agent to disguise the request as a normal browser
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
}

maxi_matches = []  # will hold True/False for each source

for i, row in df.iterrows():
    source_name = row['NAME']
    ra          = row['RA_OBJ']
    dec         = row['DEC_OBJ']

    # Build the MAXI J-name and URL
    maxi_id  = ra_dec_to_maxi_id(ra, dec)
    maxi_url = f'{MAXI_BASE}/{maxi_id}/{maxi_id}.html'

    try:
        # 2. UPDATE THIS: Pass the HEADERS dictionary into your request
        resp = requests.get(maxi_url, headers=HEADERS, timeout=MAXI_TIMEOUT)
        matched = (resp.status_code == 200)
    except Exception as exc:
        # Network error, DNS failure, timeout, etc.
        print(f'  [WARNING] MAXI request failed for {source_name}: {exc}')
        matched = False

    maxi_matches.append(matched)

    # Progress report every 50 sources (1-indexed)
    if (i + 1) % 50 == 0:
        print(f'Checking MAXI: {i + 1}/{len(df)}...')

    time.sleep(MAXI_DELAY)  # polite delay

# Store results
df['MAXI_match'] = maxi_matches

print(f'\nMAXI check complete. Matches: {sum(maxi_matches)}/{len(df)}')


# ### Section 3
# ## Saving the matches in a CSV file

# In[10]:


# Filter to get only the 270 sources that have a MAXI match
matched_df = bat_df[bat_df['MAXI_match'] == True].copy()


# Defining  unpacking function i
def get_maxi_id_from_row(row):
    """Takes a single pandas row, extracts RA/Dec, and returns the MAXI ID."""
    ra = row['RA_OBJ']
    dec = row['DEC_OBJ']

    # Calculate the ID using our main function
    maxi_id = ra_dec_to_maxi_id(ra, dec)

    return maxi_id


#  Applying explicit function to every row across the dataframe (axis=1)
matched_df['MAXI_NAME'] = matched_df.apply(get_maxi_id_from_row, axis=1)


# Selecting and renaming only the columns we want in our table
csv_export_df = matched_df[[
    'NAME',     # BAT catalog name
    'RA_OBJ',   # RA from BAT
    'DEC_OBJ',  #  Dec from BAT
    'MAXI_NAME' # The matching MAXI web ID
]].copy()

csv_export_df.columns = ['BAT_Catalogue_Name', 'RA_BAT', 'DEC_BAT', 'MAXI_Name']


# Saving the local CSV file
output_filename = "BAT_MAXI_matched_sources_try2.csv"
csv_export_df.to_csv(output_filename, index=False)

print(f" Successfully saved {len(csv_export_df)} matched sources to: {output_filename}")
csv_export_df.head()


# #### Now I am going to use the SIMBAD first method:

# In[17]:


import pandas as pd
from astroquery.simbad import Simbad
import time

# 1. Load the MAXI-matched CSV
csv_path = "BAT_MAXI_matched_sources_try2.csv"
matched_df = pd.read_csv(csv_path)
print(f"Loaded {len(matched_df)} sources from: {csv_path}")

# 2. Configure SIMBAD
custom_simbad = Simbad()
custom_simbad.add_votable_fields('otype')  # ra/dec already returned by default in degrees

# 3. Result lists
simbad_main_ids = []
simbad_otypes   = []
simbad_ras      = []
simbad_decs     = []

print("\nQuerying SIMBAD row-by-row for precise optical coordinates...")

# 4. Loop
for index, row in matched_df.iterrows():
    name_to_search = row['BAT_Catalogue_Name']
    try:
        result_table = custom_simbad.query_object(name_to_search)
        if result_table is not None and len(result_table) > 0:
            cols = result_table.colnames

            # Main ID
            if 'main_id' in cols:   raw_main_id = result_table['main_id'][0]
            elif 'MAIN_ID' in cols: raw_main_id = result_table['MAIN_ID'][0]
            else:                   raw_main_id = None

            # Otype
            if 'otype' in cols:   raw_otype = result_table['otype'][0]
            elif 'OTYPE' in cols: raw_otype = result_table['OTYPE'][0]
            else:                 raw_otype = None

            # RA (degrees)
            if 'ra' in cols:   raw_ra = float(result_table['ra'][0])
            elif 'RA' in cols: raw_ra = float(result_table['RA'][0])
            else:              raw_ra = None

            # Dec (degrees)
            if 'dec' in cols:   raw_dec = float(result_table['dec'][0])
            elif 'DEC' in cols: raw_dec = float(result_table['DEC'][0])
            else:               raw_dec = None

            # Clean text fields
            clean_main_id = raw_main_id.decode('utf-8').strip() if isinstance(raw_main_id, bytes) else (str(raw_main_id).strip() if raw_main_id is not None else None)
            clean_otype   = raw_otype.decode('utf-8').strip()   if isinstance(raw_otype,   bytes) else (str(raw_otype).strip()   if raw_otype   is not None else None)

            simbad_main_ids.append(clean_main_id)
            simbad_otypes.append(clean_otype)
            simbad_ras.append(raw_ra)
            simbad_decs.append(raw_dec)

        else:
            simbad_main_ids.append(None)
            simbad_otypes.append(None)
            simbad_ras.append(None)
            simbad_decs.append(None)

    except Exception as error:
        print(f"  [SKIP] {name_to_search}: {error}")
        simbad_main_ids.append(None)
        simbad_otypes.append(None)
        simbad_ras.append(None)
        simbad_decs.append(None)

    if (index + 1) % 30 == 0:
        print(f"  Progress: {index + 1}/{len(matched_df)}...")
    time.sleep(0.1)

# 5. Attach columns
matched_df['SIMBAD_ID']    = simbad_main_ids
matched_df['SIMBAD_Otype'] = simbad_otypes
matched_df['SIMBAD_RA']    = simbad_ras
matched_df['SIMBAD_DEC']   = simbad_decs

# 6. Save
output_path = "BAT_MAXI_SIMBAD_precise_coords_copy.csv"
matched_df.to_csv(output_path, index=False)
print(f"\nDone. Saved {len(matched_df)} sources with precise coords → {output_path}")

matched_df[['BAT_Catalogue_Name', 'SIMBAD_ID', 'SIMBAD_Otype', 'SIMBAD_RA', 'SIMBAD_DEC']].head(10)


# In[16]:


result_table = custom_simbad.query_object("Aql X-1")
print(result_table.colnames)
print(result_table)


# In[19]:


client = Alerce()

# 1. Load the CSV containing our hyper-precise SIMBAD coordinates
input_path = "BAT_MAXI_SIMBAD_precise_coords_copy.csv"
df = pd.read_csv(input_path)
print(f"Loaded {len(df)} sources from: {input_path}")

# Filter out rows where SIMBAD couldn't resolve coordinates
valid_df = df.dropna(subset=['SIMBAD_RA', 'SIMBAD_DEC']).copy()
print(f"Found {len(valid_df)} sources with valid coordinates to target on ZTF.")

# 2. Storage arrays
ztf_ids    = []
ztf_ras    = []
ztf_decs   = []
separations = []

CONE_RADIUS_ARCSEC = 5.0
print(f"\nStarting precise ZTF cross-match (Radius: {CONE_RADIUS_ARCSEC} arcseconds)...")

# 3. Core matching loop
for index, row in valid_df.iterrows():
    name      = row['BAT_Catalogue_Name']
    ra_simbad  = row['SIMBAD_RA']
    dec_simbad = row['SIMBAD_DEC']

    center_coord = SkyCoord(ra=ra_simbad, dec=dec_simbad, unit=(u.deg, u.deg), frame='icrs')

    try:
        matches = client.query_objects(
            survey="ztf",
            ra=ra_simbad,
            dec=dec_simbad,
            radius=CONE_RADIUS_ARCSEC
        )

        if matches is not None and len(matches) > 0:
            ztf_coords  = SkyCoord(
                ra=matches['meanra'].values,
                dec=matches['meandec'].values,
                unit='deg'
            )
            seps        = center_coord.separation(ztf_coords).arcsec
            closest_idx = np.argmin(seps)
            best        = matches.iloc[closest_idx]

            ztf_ids.append(best.get('oid', None))
            ztf_ras.append(float(best['meanra']))
            ztf_decs.append(float(best['meandec']))
            separations.append(float(seps[closest_idx]))

        else:
            ztf_ids.append(None)
            ztf_ras.append(None)
            ztf_decs.append(None)
            separations.append(None)

    except Exception as e:
        print(f"  [ERROR] Failed to cross-match ZTF for {name}: {e}")
        ztf_ids.append(None)
        ztf_ras.append(None)
        ztf_decs.append(None)
        separations.append(None)

    if len(ztf_ids) % 30 == 0:
        print(f"  Progress: {len(ztf_ids)}/{len(valid_df)} processed...")

    time.sleep(0.1)

# 4. Attach columns
valid_df['ZTF_Object_ID']         = ztf_ids
valid_df['ZTF_RA']                = ztf_ras
valid_df['ZTF_DEC']               = ztf_decs
valid_df['SIMBAD_ZTF_Sep_Arcsec'] = separations

# 5. Save
output_path = "BAT_MAXI_SIMBAD_ZTF_perfect_match_copy.csv"
valid_df.to_csv(output_path, index=False)
print(f"\nSuccess! Saved perfectly matched master table → {output_path}")

valid_df[['BAT_Catalogue_Name', 'SIMBAD_ID', 'ZTF_Object_ID', 'SIMBAD_ZTF_Sep_Arcsec']].dropna().head(10)


# In[20]:


# Load the perfect match CSV
df = pd.read_csv("BAT_MAXI_SIMBAD_ZTF_perfect_match_copy.csv")

# Filter: only XRBs (LXB or HXB) with ZTF match
xrb_df = df[
    (df['SIMBAD_Otype'].isin(['LXB', 'HXB'])) &
    (df['ZTF_Object_ID'].notna())
].copy()

print(f"XRBs matched in BAT + MAXI + ZTF: {len(xrb_df)}")

xrb_df.to_csv("XRB_only_BAT_MAXI_ZTF_final.csv", index=False)
print("Saved → XRB_BAT_MAXI_ZTF_final.csv")

xrb_df.head(20)


# In[22]:


df = pd.read_csv("XRB_only_BAT_MAXI_ZTF_final.csv")
print(df.columns.tolist())


# In[26]:


df = pd.read_csv("XRB_only_BAT_MAXI_ZTF_final.csv")

final = df[[
    'BAT_Catalogue_Name',
    'RA_BAT',
    'DEC_BAT',
    'MAXI_Name',
    'SIMBAD_ID',
    'SIMBAD_Otype',
    'SIMBAD_RA',
    'SIMBAD_DEC',
    'ZTF_Object_ID'
]].copy()
final['SIMBAD_RA']  = final['SIMBAD_RA'].round(4)  # ADD HERE
final['SIMBAD_DEC'] = final['SIMBAD_DEC'].round(4) # ADD HERE

final.to_csv("XRB_BAT_MAXI_ZTF_final_report.csv", index=False)
print(f"Saved {len(final)} XRBs → XRB_BAT_MAXI_ZTF_final.csv")
final.head()


# In[ ]:





# In[ ]:




