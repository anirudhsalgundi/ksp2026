from astropy.io import fits
from astropy.table import Table
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests
from bs4 import BeautifulSoup
import time

from alerce.core import Alerce

alerce = Alerce()

ztf_df = pd.read_csv("ztf_maxi_swift.csv")
ztf_names = ztf_df['target_name'].to_list()
print(ztf_names)
xrb_names=[]
#### SWIFT ####

maxi_swift = pd.read_csv("maxi_swift.csv")

base_url = "https://swift.gsfc.nasa.gov/results/transients/"
data_host = "https://swift.gsfc.nasa.gov"

response = requests.get(base_url)
soup = BeautifulSoup(response.text, 'html.parser')
table = soup.find('table')

swift_data = {}

for row in table.find_all('tr')[1:]:

    columns = row.find_all('td')
    if not columns:
        continue
    target_swift_name = columns[1].text.strip()
    source_type = columns[5].text.strip()
    maxi_swift['swift_name'] = maxi_swift['swift_name'].astype(str).str.strip()
    matched_row = maxi_swift[maxi_swift['swift_name'] == target_swift_name]

    if not matched_row.empty:
        target_name = matched_row['maxi_name'].values[0]


        if target_name in ztf_names:

            if 'HMXB' in source_type or 'LMXB' in source_type or 'HXMB' in source_type: 

                xrb_names.append(target_name)

                first_cell = columns[1].find('a')

                if first_cell and 'href' in first_cell.attrs:
                    href = first_cell['href']
                    
                    if href.startswith(".."):
                        full_url = data_host + href.lstrip('.')
                    else:
                        full_url = f"{data_host}/{href}"

                    print(full_url)
                    identifier = full_url.split('/')[-1]
                    
                    dat_url = f"https://swift.gsfc.nasa.gov/results/transients/weak/{identifier}.lc.fits"
                    print(f"{dat_url}")


                    try:
                        df = Table.read(dat_url, hdu='RATE')
                        swift_data[target_name] = df

                        print(df['RATE'])
                    except:
                        try:
                            dat_url = f"https://swift.gsfc.nasa.gov/results/transients/{identifier}.lc.fits"
                            print(f"{dat_url}")

                            df = Table.read(dat_url, hdu='RATE')
                            swift_data[target_name] = df

                            print(df['RATE'])
                        
                        except Exception as e:
                            print(f"Could not load data for {target_name}: {e}")
                else: 
                    print(f"No url for {target_swift_name}")


#### MAXI ####

print("MAXI")
base_url = "http://maxi.riken.jp/top/slist.html"
data_host = "http://maxi.riken.jp"

response = requests.get(base_url)
soup = BeautifulSoup(response.text, 'html.parser')
table = soup.find('table')

maxi_data = {}

for row in table.find_all('tr')[1:]:
    columns = row.find_all('td')
    if not columns:
        continue
    target_name = columns[0].text

    if target_name in xrb_names:
        third_cell = columns[2].find('a')

        if third_cell and 'href' in third_cell.attrs:
            href = third_cell['href']
            
            if href.startswith(".."):
                full_url = data_host + href.lstrip('.')
            else:
                full_url = f"{data_host}/{href}"
            identifier = full_url.split('/')[-2]

            dat_url = f"http://maxi.riken.jp/star_data/{identifier}/{identifier}_g_lc_1day_all.dat"
            print(f"{dat_url}")
            try:
                df = pd.read_csv(dat_url, sep=r'\s+', header = None)
                maxi_data[target_name] = df
            except Exception as e:
                print(f"Could not load data for {target_name}: {e}")
        else: 
            print("No url")
        

#### Optical ####
print("ZTF det")
optical_table = pd.read_csv("ztf_maxi_swift.csv", sep=r',')
all_r = {}
all_g = {}
for index, ztf_row in optical_table.iterrows():
    target_name = str(ztf_row['target_name'])
    oid = str(ztf_row['ztf_id'])

    
    response_data = alerce.query_detections(oid, format="json")
    df = pd.DataFrame(response_data)
    print(target_name)
    print(df.head())
    if not df.empty:
        df_g = df[df['fid'] == 1][['mjd', 'magpsf']].reset_index(drop=True)
        df_r = df[df['fid'] == 2][['mjd', 'magpsf']].reset_index(drop=True)


        all_g[target_name] = df_g
        all_r[target_name] = df_r
    else:
        all_g[target_name] = pd.DataFrame()
        all_r[target_name] = pd.DataFrame()

all_r_fp = {}
all_g_fp = {}

print("ZTF forced")
for index, ztf_row in optical_table.iterrows():
    target_name = str(ztf_row['target_name'])
    oid = str(ztf_row['ztf_id'])

    
    response_data = alerce.query_forced_photometry(oid, format="json")
    df = pd.DataFrame(response_data)
    print(target_name)
    print(df.head())
    if not df.empty:
        df_g = df[df['fid'] == 1][['mjd', 'mag']].reset_index(drop=True)
        df_r = df[df['fid'] == 2][['mjd', 'mag']].reset_index(drop=True)


        all_g_fp[target_name] = df_g
        all_r_fp[target_name] = df_r
    else:
        all_g_fp[target_name] = pd.DataFrame()
        all_r_fp[target_name] = pd.DataFrame()

all_r_nd = {}
all_g_nd = {}

print("Non detec")
for index, ztf_row in optical_table.iterrows():
    target_name = str(ztf_row['target_name'])
    oid = str(ztf_row['ztf_id'])

    
    response_data = alerce.query_non_detections(oid, format="json")
    df = pd.DataFrame(response_data)
    print(target_name)
    print(df.head())
    if not df.empty:

        df_g = df[df['fid'] == 1][['mjd', 'diffmaglim']].reset_index(drop=True)
        df_r = df[df['fid'] == 2][['mjd', 'diffmaglim']].reset_index(drop=True)


        all_g_nd[target_name] = df_g
        all_r_nd[target_name] = df_r
    else:
        all_g_nd[target_name] = pd.DataFrame()
        all_r_nd[target_name] = pd.DataFrame()

    
#x_opt = optical_table[0].tolist()+2450000
j = 0
for name in xrb_names:
    print(name)
    try:
        x_swift = (swift_data[name])['TIME']
        y_swift = (swift_data[name])['RATE']

        x_maxi = maxi_data[name][0].tolist()
        y_maxi = maxi_data[name][1].tolist()

        
        x_g = (all_g[name])['mjd'].tolist()
        y_g = (all_g[name])['magpsf'].tolist()
        x_g_fp = (all_g_fp[name])['mjd'].tolist()
        y_g_fp = (all_g_fp[name])['mag'].tolist()
        x_g_nd = (all_g_nd[name])['mjd'].tolist()
        y_g_nd = (all_g_nd[name])['diffmaglim'].tolist()


        x_r = (all_r[name])['mjd'].tolist()
        y_r = (all_r[name])['magpsf'].tolist()
        x_r_fp = (all_r_fp[name])['mjd'].tolist()
        y_r_fp = (all_r_fp[name])['mag'].tolist()
        x_r_nd = (all_r_nd[name])['mjd'].tolist()
        y_r_nd = (all_r_nd[name])['diffmaglim'].tolist()


        min_time = max(x_swift[0],x_maxi[0],x_g[0],x_r[0])
        max_time = min(x_swift[-1],x_maxi[-1],x_g[-1],x_r[-1])


        print(min_time)
        print(max_time)
        fig, axs = plt.subplots(
            nrows=4, ncols=1, sharex=True, figsize=(8, 10), gridspec_kw={"hspace": 0}
        )

        axs[0].plot(x_swift, y_swift, color = 'black', markersize=2, linestyle = '-')  
        axs[0].set_title('Swift')
        axs[0].set_ylabel('Counts/cm^2/sec (15-50 keV)')
        axs[1].plot(x_maxi, y_maxi, color = 'blue', markersize=2, linestyle = '-')  
        axs[1].set_title('MAXI')
        axs[1].set_ylabel('2-20keV [ph/s/cm2]')
        axs[2].plot(x_g, y_g, color = 'green', markersize=2, marker = 'o', linestyle = 'none') 
        axs[2].plot(x_g_fp, y_g_fp, color = 'green', markersize=2, marker = 's', linestyle = 'none')
        axs[2].plot(x_g_nd, y_g_nd, color = 'green', markersize=2, marker = 'v', linestyle = 'none')
        axs[2].set_title('g ZTF')
        axs[2].set_ylabel('Magnitude')
        axs[2].invert_yaxis()
        axs[3].plot(x_r, y_r, color = 'red', markersize=2, linestyle = 'none')  
        axs[3].plot(x_r_fp, y_r_fp,color = 'red', markersize=2, marker = 's', linestyle = 'none')
        axs[3].plot(x_r_nd, y_r_nd, color = 'red', markersize=2, marker = 'v', linestyle = 'none')
        axs[3].set_title('r ZTF')
        axs[3].set_ylabel('Magnitude')
        axs[3].invert_yaxis()

        axs[3].set_xlabel('MJD')


        for i, ax in enumerate(axs):
            ax.axhline(0, color="black", linestyle=":", linewidth=0.8)

            ax.tick_params(
                axis="both", direction="in", top=True, right=True, which="both"
            )

            if i < 3:
                ax.tick_params(axis="x", labelbottom=False)

        axs[-1].set_xlim(min_time, max_time)

        axs[0].set_title(name, loc="left", fontsize=12)
        j = j+1
        
        plt.savefig(f'./plots_xrb/{name}_common.png', format = "png")
        plt.show()
    except Exception as e:
        print(f"Could not plot {name} due to error {e}")

print(j)
print(xrb_names)


