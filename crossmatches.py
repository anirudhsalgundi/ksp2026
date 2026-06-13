import time
import os
import re
import sys
from io import StringIO

from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
import pandas as pd

from alerce.core import Alerce
import requests

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler("app.log")

log_format = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
console_handler.setFormatter(log_format)
file_handler.setFormatter(log_format)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.info("logger configured")



def normalise_name(name):
    return str(name).strip().lower().replace(" ", "")

def load_maxi_data():
    url = "https://maxi.riken.jp/top/slist.html"


    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers = headers)
        response.raise_for_status()

        tables = pd.read_html(response.text)
        maxi_data = tables[0]

        
        maxi_data.columns = maxi_data.iloc[0]

        maxi_data = maxi_data[1:].reset_index(drop=True)
        maxi_data.rename(columns={'source name': 'source_name'}, inplace=True)
        maxi_data[['RA', 'Dec']] = maxi_data['R.A., Dec'].str.split(', ', expand=True) # Separating RA and Dec into two columns

        maxi_data['RA'] = pd.to_numeric(maxi_data['RA'], errors='coerce')
        maxi_data['Dec'] = pd.to_numeric(maxi_data['Dec'], errors='coerce')

        logger.info("Maxi data loaded")
        return maxi_data
    except Exception as e:
        logger.error("Failed to load MAXI data:", {e})
        return None

def load_swift_data():
    try:
        with fits.open("BAT_catalog.fits") as hdul:
            data = hdul['INPUT_CATALOG'].data
        df = pd.DataFrame(data)

        logger.info("Swift data loaded")
        return df
    except Exception as e:
        logger.error("Failed to load Swift data:", {e})
        return None

def check_maxi_with_swift():
    maxi_data = load_maxi_data()
    swift_data = load_swift_data()
    maxi_coords = SkyCoord(ra=maxi_data['RA'].values*u.deg, dec=maxi_data['Dec'].values*u.deg)
    swift_coords = SkyCoord(ra=swift_data['RA_OBJ'].values*u.deg, dec=swift_data['DEC_OBJ'].values*u.deg)

    max_separation = 0.1 * u.deg

    idx, d2d, _ = maxi_coords.match_to_catalog_sky(swift_coords)

    matches = d2d < max_separation

    maxi_names = np.array(maxi_data['source_name'])
    matched_maxi_names = maxi_names[matches]

    logger.info("Matched maxi and swift")

    return maxi_data, swift_data, matched_maxi_names

def crossmatch_ztf(data):

    logger.info("Starting ZTF crossmatch")

    client = Alerce()

    radius = 5
    results = []
    #results_with_names = []
    list_of_names = []
    for name, ra, dec in zip(data['source_name'], data['RA'], data['Dec']):
        try:
            target_coord = SkyCoord(ra=float(ra)*u.deg, dec=float(dec)*u.deg)

            objects = client.query_objects(survey="ztf", 
                                        ra=float(ra), 
                                        dec=float(dec), 
                                        radius=radius
                                        )

        except Exception as e:
            logger.error(f"Error occurred while querying ZTF for {name}: {e}")
            objects = []

        if len(objects)>0:
            ztf_coords = SkyCoord(ra=objects['meanra'].values*u.deg, dec=objects['meandec'].values*u.deg)

            separations = target_coord.separation(ztf_coords)
            nearest_idx = separations.argmin()

            nearest_object = objects.iloc[nearest_idx]
            nearest_sep_arcsec = separations[nearest_idx].arcsec

            results.append({
                'target_name': name,
                'ztf_id': nearest_object['oid'],  # ZTF Object ID
                'separation_arcsec': nearest_sep_arcsec,
                'ztf_ra': nearest_object['meanra'],
                'ztf_dec': nearest_object['meandec']
            })
            
            list_of_names.append(name)

        logger.info("Match found for %s: count is %d", name, len(objects)) # This is to track the progress of the loop

        time.sleep(1)  # To avoid hitting API rate limits


    
    logger.info("Number of objects found: ", len(results), " out of ", len(data))

    df = pd.DataFrame(results)
    df.to_csv('ztf_results_maxi_v2.csv', index=False)

    logger.info("ZTF crossmatch done")
    return list_of_names

def crossmatch_atlas(input_data):

    logger.info("Starting Atlas crossmatch")

    BASEURL = "https://fallingstar-data.com/forcedphot"
    # BASEURL = "http://127.0.0.1:8000"

    if os.environ.get("ATLASFORCED_SECRET_KEY"):
        token = os.environ.get("ATLASFORCED_SECRET_KEY")
        logger.info("Using stored token")
    else:
        data = {"username": "vidhic", "password": "Scientia324"}

        resp = requests.post(url=f"{BASEURL}/api-token-auth/", data=data)

        if resp.status_code == 200:
            token = resp.json()["token"]
            logger.info(f"Your token is {token}")
            logger.info("Store this by running/adding to your .zshrc file:")
            logger.info(f'export ATLASFORCED_SECRET_KEY="{token}"')
        else:
            logger.error(f"ERROR {resp.status_code}")
            logger.error(resp.text)
            return None

    headers = {"Authorization": f"Token {token}", "Accept": "application/json"}


    results = []
    #results_with_names = []
    list_of_names = []
    for name, ra, dec in zip(input_data['source_name'], input_data['RA'], input_data['Dec']):
                
        task_url = None
        while not task_url:
            with requests.Session() as s:
                # alternative to token auth
                # s.auth = ('USERNAME', 'PASSWORD')
                resp = s.post(f"{BASEURL}/queue/", headers=headers, data={"ra": ra, "dec": dec, "mjd_min": 59000.0})

                if resp.status_code == 201:  # successfully queued
                    task_url = resp.json()["url"]
                    logger.info(f"The task URL is {task_url}")
                elif resp.status_code == 429:  # throttled
                    message = resp.json()["detail"]
                    logger.info(f"{resp.status_code} {message}")
                    t_sec = re.findall(r"available in (\d+) seconds", message)
                    t_min = re.findall(r"available in (\d+) minutes", message)
                    if t_sec:
                        waittime = int(t_sec[0])
                    elif t_min:
                        waittime = int(t_min[0]) * 60
                    else:
                        waittime = 10
                    logger.info(f"Waiting {waittime} seconds")
                    time.sleep(waittime)
                else:
                    logger.error(f"ERROR {resp.status_code}")
                    logger.error(resp.text)
                    return None

        result_url = None
        taskstarted_printed = False
        while not result_url:
            with requests.Session() as s:
                resp = s.get(task_url, headers=headers)

                if resp.status_code == 200:  # HTTP OK
                    if resp.json()["finishtimestamp"]:
                        result_url = resp.json()["result_url"]
                        logger.info(f"Task is complete with results available at {result_url}")
                    elif resp.json()["starttimestamp"]:
                        if not taskstarted_printed:
                            logger.info(f"Task is running (started at {resp.json()['starttimestamp']})")
                            taskstarted_printed = True
                        time.sleep(2)
                    else:
                        logger.info(f"Waiting for job to start (queued at {resp.json()['timestamp']})")
                        time.sleep(4)
                else:
                    logger.error(f"ERROR {resp.status_code}")
                    logger.error(resp.text)
                    return None

        with requests.Session() as s:
            textdata = s.get(result_url, headers=headers).text

            # if we'll be making a lot of requests, keep the web queue from being
            # cluttered (and reduce server storage usage) by sending a delete operation
            resp = s.delete(task_url, headers=headers)

            if resp.text.strip():
                logger.info(resp.json())


        dfresult = pd.read_csv(StringIO(textdata), sep=r"\s+").rename({"###MJD": "MJD"}, axis="columns")
        if len(dfresult) > 0:
            results.append({
                'target_name': name,
                'ra': dfresult['RA'].iloc[0],
                'dec': dfresult['Dec'].iloc[0]
            })
            logger.info(results)
            list_of_names.append(name)
        logger.info(f"{name} {len(dfresult)}")


        logger.info("Pausing for 3 seconds for the next target...")
        time.sleep(3)
    

    logger.info("Number of objects found: ", len(results), " out of ", len(input_data))

    df = pd.DataFrame(results)

    df.to_csv('atlas_results_maxi_v2.csv', index=False)

    logger.info("Atlas crossmatch done")
    return list_of_names

def make_summary():
    maxi_data, swift_data, maxi_swift = check_maxi_with_swift()

    ztf_names = crossmatch_ztf(maxi_data)
    ztf_data = pd.read_csv("ztf_results_maxi_v2.csv")
    ztf_names = ztf_data["target_name"].tolist()

    logger.info(ztf_names)

    if ztf_names is not None:
        not_ztf_names = maxi_data[~maxi_data["source_name"].isin(ztf_names)]
    else:
        not_ztf_names = maxi_data["source_name"].tolist()
    
    atlas_names = crossmatch_atlas(not_ztf_names)

    results_summary = []
    for name in maxi_data['source_name']:
        temp = [name, maxi_data["source_name"].isin([name]).any(), name in maxi_swift, name in ztf_names, name in atlas_names]
        results_summary.append(temp)

    df = pd.DataFrame(results_summary, columns=["name", "in_maxi", "in_swift", "in_ztf", "not_ztf_in_atlas"])
    df.to_csv("summary.csv", index=False) 
    return None
    

def main():

    make_summary()
    return None

if __name__ == "__main__":
    main()