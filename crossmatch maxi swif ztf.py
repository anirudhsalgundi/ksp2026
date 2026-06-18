#crossmatch maxi to swift to ztf

import pandas as pd
import numpy as np
import time

from astropy.coordinates import SkyCoord
import astropy.units as u
from alerce.core import Alerce

client = Alerce()

maxi = pd.read_csv("maxi_cat.csv")
swift = pd.read_csv("swift_cat.csv")

maxi.columns = maxi.columns.str.strip()
swift.columns = swift.columns.str.strip()


swift["CTPT_RA"] = pd.to_numeric(swift["CTPT_RA"], errors="coerce")
swift["CTPT_DEC"] = pd.to_numeric(swift["CTPT_DEC"], errors="coerce")

swift = swift.dropna(subset=["CTPT_RA", "CTPT_DEC"]).reset_index(drop=True)


maxi_coord = SkyCoord(maxi["RA"].astype(float),
                      maxi["DEC"].astype(float),
                      unit=u.deg)

swift_coord = SkyCoord(swift["CTPT_RA"].astype(float),
                       swift["CTPT_DEC"].astype(float),
                       unit=u.deg)

# maxi to swift crossmatch
radius_ms = 3 * u.arcmin

idx_swift, idx_maxi, sep_ms, _ = maxi_coord.search_around_sky(
    swift_coord,
    radius_ms
)

print("MAXI objects:", len(maxi))
print("Swift objects:", len(swift))
print("MAXI–Swift matches:", len(idx_swift))

#swift to ztf
radius_ztf = 5  # arcsec

results = []

for k in range(len(idx_swift)):

    sw_i = idx_swift[k]
    mx_i = idx_maxi[k]
    sep = sep_ms[k]

    ra = float(swift.loc[sw_i, "CTPT_RA"])
    dec = float(swift.loc[sw_i, "CTPT_DEC"])

    try:
        ztf_match = client.query_objects(
            ra=ra,
            dec=dec,
            radius=radius_ztf,
            format="pandas"
        )

        if ztf_match is None or ztf_match.empty:
            continue

        best = ztf_match.iloc[0]

        results.append({
            "MAXI_source": maxi.loc[mx_i, "Source IDs"],
            "MAXI_RA": maxi.loc[mx_i, "RA"],
            "MAXI_DEC": maxi.loc[mx_i, "DEC"],

            "Swift_source": swift.loc[sw_i, "BAT_NAME"],
            "Swift_RA": ra,
            "Swift_DEC": dec,

            "ZTF_oid": best.get("oid"),
            "ZTF_class": best.get("class"),

            "n_ztf_matches": len(ztf_match),

            
            "MAXI_SWIFT_sep_arcsec": sep.to(u.arcsec).value
        })

    except Exception as e:
        print(f"ERROR at {k}: {repr(e)}")

    time.sleep(0.2)


final_df = pd.DataFrame(results)

print("\n================ FINAL CATALOG ================")
print(final_df)

final_df.to_csv("MAXI_SWIFT_ZTF_catalog_fixed.csv", index=False)

print("\nTOTAL MATCHED SYSTEMS:", len(final_df))