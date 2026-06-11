import time

import numpy as np
import pandas as pd
from alerce.core import Alerce
import requests


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

        maxi_data[['RA', 'Dec']] = maxi_data['R.A., Dec'].str.split(', ', expand=True) # Separating RA and Dec into two columns

        return maxi_data
    except Exception as e:
        print("Failed to load MAXI data:", e)
        return None

def load_swift_data():
    url = "https://swift.gsfc.nasa.gov/results/transients/"


    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers = headers)
        response.raise_for_status()

        tables = pd.read_html(response.text)
        swift_data = tables[0]

        return swift_data
    except Exception as e:
        print("Failed to load Swift data:", e)
        return None

def check_maxi_with_swift():
    maxi_data = load_maxi_data()
    swift_data = load_swift_data()
    swift_names = set(swift_data['Source Name'].apply(normalise_name)) 
    maxi_swift = []

    for name in maxi_data['source name'].apply(normalise_name):
        if name in swift_names:
            maxi_swift.append(name)

    return maxi_data, swift_data, maxi_swift

def crossmatch_ztf(data):

    client = Alerce()

    radius = 5
    results = []
    results_with_names = []
    list_of_names = []
    for name, ra, dec in zip(data['source name'], data['RA'], data['Dec']):
        try:
            objects = client.query_objects(survey="ztf", 
                                        ra=float(ra), 
                                        dec=float(dec), 
                                        radius=radius
                                        )
        except Exception as e:
            print(f"Error occurred while querying ZTF for {name}: {e}")
            objects = []

        print(name, len(objects)) # This is to track the progress of the loop
    
            
        if len(objects) > 0:
                objects["input_ra"] = ra
                objects["input_dec"] = dec
                results.append(objects) 
                list_of_names.append(normalise_name(name))  # This is to keep track of the names of the matched sources
                for idx, row in objects.iterrows():  
                    temp = [name, ra, dec, row["ndet"]]   # This is to print the crossmatches with the names of the matched sources
                    results_with_names.append(temp)

        time.sleep(1)  # To avoid hitting API rate limits


    if len(results) > 0:
        final_results = pd.concat(results, ignore_index=True)
        final_results.to_csv("ztf_results_maxi.csv", index=False) # This contains the ZTF results

    print("Done! Results saved to ztf_results_maxi.csv")
    print("Number of objects found: ", len(final_results), " out of ", len(data))

    df = pd.DataFrame(results_with_names, columns=["Name", "RA", "Dec", "NDetections"])
    df.to_excel("ztf_matches_with_names_maxi.xlsx", index=False) # This contains names of the matched sources along with their RA, Dec and number of detections in ZTF
    print(df)

    return list_of_names # This is the list of names of the sources that have been crossmatched with ZTF

def crossmatch_atlas():

    return None

def make_summary():
    maxi_data, swift_data, maxi_swift = check_maxi_with_swift()

    ztf_names = crossmatch_ztf(maxi_data)
    ztf_res = pd.read_csv("ztf_results_maxi.csv")

    results_summary = []
    for name in maxi_data['source name'].apply(normalise_name):
        temp = [name, name in maxi_swift, name in ztf_names]
        results_summary.append(temp)

    df = pd.DataFrame(results_summary, columns=["Name", "In Swift?", "In ZTF?"])
    df.to_excel("summary.xlsx", index=False) 
    return None
 

def main():
    make_summary()
    return None

if __name__ == "__main__":
    main()