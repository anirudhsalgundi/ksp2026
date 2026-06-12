#cross-match:  MAXI source → Crossmatch with Swift → Crossmatch with ZTF.
import pandas as pd
from alerce.core import Alerce
import numpy as np

maxi_df = pd.read_csv(r"C:\Users\chowd\Downloads\maxi_data.csv")
swift_df = pd.read_csv(r"C:\Users\chowd\Downloads\swift_source.csv")

client = Alerce()
final_results = []


for index, row in maxi_df.iterrows():
    ra_val = float(row["RA"])
    dec_val = float(row[" Dec"])

    match_record = {
        "maxi_source": row["source name"],
        "ra": ra_val,
        "dec": dec_val,
        "swift_match_found": False,
        "swift_source_name": None,
        "ztf_found": False,
        "ztf_oid": None,
        "ztf_class": None,
        "n_ztf_matches": 0
    }

    # (within ~5 arcseconds / 0.00139 degrees)
    swift_matches = swift_df[
        (np.abs(swift_df["RA J2000 Degs"] - ra_val) < 0.00139) &
        (np.abs(swift_df["Dec J2000 Degs"] - dec_val) < 0.00139)
    ]
    if not swift_matches.empty:
        match_record.update({
            "swift_match_found": True,
            "swift_source_name": swift_matches.iloc[0]["Source Name"]
        })

    #  Check ZTF via ALeRCE API
    try:
        ztf_matches = client.query_objects(
            survey="ztf", ra=ra_val, dec=dec_val, radius=5, format="pandas"
        )
        if ztf_matches is not None and not ztf_matches.empty:
            best_ztf = ztf_matches.iloc[0]
            match_record.update({
                "ztf_found": True,
                "ztf_oid": best_ztf.get("oid"),
                "ztf_class": best_ztf.get("classxf"),
                "n_ztf_matches": len(ztf_matches)
            })
    except Exception as e:
        print(f"Error querying ZTF for {row['source name']}: {e}")

  
    final_results.append(match_record)


ztf_crossmatch_ = pd.DataFrame(final_results)
display(ztf_crossmatch_)

print(f"\nSwift matches: {ztf_crossmatch_['swift_match_found'].sum()}")
print(f"ZTF matches: {ztf_crossmatch_['ztf_found'].sum()}")

ztf_crossmatch_.to_csv('maxi_swift_ztf_crossmatch_result.csv', index=False)
