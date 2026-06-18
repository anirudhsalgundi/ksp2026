#load swift data

import pandas as pd
url='https://swift.gsfc.nasa.gov/bat_survey/bs105mon/data/BAT_105m_catalog_07jul2019.txt'
df=pd.read_csv(url,sep="\t")
df_bat = pd.read_csv(url, sep="|")

df_bat.columns = df_bat.columns.str.strip()
df_bat = df_bat.drop(columns=[0], errors="ignore")

df_bat.to_csv("swift_cat.csv", index=False)
print(df_bat.head(5))

df_bat=pd.read_csv('swift_cat.csv')
print('success!')