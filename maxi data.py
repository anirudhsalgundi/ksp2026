#load maxi data


import pandas as pd

url = "https://maxi.riken.jp/pubdata/v7.7l/"
tables = pd.read_html(url)
maxi_df = tables[0]

maxi_df = maxi_df.dropna(subset=["Source IDs", "RA", "Dec"]).reset_index(
    drop=True
)
maxi_df["RA"] = pd.to_numeric(maxi_df["RA"], errors="coerce")
maxi_df["Dec"] = pd.to_numeric(maxi_df["Dec"], errors="coerce")

maxi_df = maxi_df.drop("Unnamed: 0", axis=1)

print(maxi_df.head())

maxi_df.to_csv("maxi_cat.csv", index=False)
print(
    "success!!"
)
