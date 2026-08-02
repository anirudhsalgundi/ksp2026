import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

outburst_data = pd.read_csv("outburst_statistics/outburst_stats.csv")

# Split the dataframe
ns = outburst_data[outburst_data["type"] == "NS"]
bh = outburst_data[outburst_data["type"] == "BH"]

fig, axes = plt.subplots(2, 2, figsize=(15, 7))
ns_rise = ns["t_rise"].dropna()
bh_rise = bh["t_rise"].dropna()
# t_rise
axes[0,0].hist(
    [ns["t_rise"].dropna(), bh["t_rise"].dropna()],
    bins="sturges",
    stacked=True,
    label=[f"NS (N={len(ns_rise)})", f"BH (N={len(bh_rise)})"],
    color=["skyblue", "tomato"],
    edgecolor="white"
)
axes[0,0].set_title("t_rise")
axes[0,0].set_xlabel("Time (days)")
axes[0,0].set_ylabel("Number of Outbursts")
axes[0,0].legend()


ns_decay = ns["t_decay"].dropna()
bh_decay = bh["t_decay"].dropna()
# t_decay
axes[0,1].hist(
    [ns["t_decay"].dropna(), bh["t_decay"].dropna()],
    bins="sturges",
    stacked=True,
    label=[f"NS (N={len(ns_decay)})", f"BH (N={len(bh_decay)})"],
    color=["skyblue", "tomato"],
    edgecolor="white"
)
axes[0,1].set_title("t_decay")
axes[0,1].set_xlabel("Time (days)")
axes[0,1].set_ylabel("Number of Outbursts")
axes[0,1].legend()


ns_peak = ns["t_peak"].dropna()
bh_peak = bh["t_peak"].dropna()
# t_peak
axes[1,0].hist(
    [ns["t_peak"].dropna(), bh["t_peak"].dropna()],
    bins="sturges",
    stacked=True,
    label=[f"NS (N={len(ns_peak)})", f"BH (N={len(bh_peak)})"],
    color=["skyblue", "tomato"],
    edgecolor="white"
)
axes[1,0].set_title("t_peak")
axes[1,0].set_xlabel("Time (days)")
axes[1,0].set_ylabel("Number of Outbursts")
axes[1,0].legend()

ns_90 = ns["t_90"].dropna()
bh_90 = bh["t_90"].dropna()
# t_90
axes[1,1].hist(
    [ns["t_90"].dropna(), bh["t_90"].dropna()],
    bins="sturges",
    stacked=True,
    label=[f"NS (N={len(ns_90)})", f"BH (N={len(bh_90)})"],
    color=["skyblue", "tomato"],
    edgecolor="white"
)
axes[1,1].set_title("t_90")
axes[1,1].set_xlabel("Time (days)")
axes[1,1].set_ylabel("Number of Outbursts")
axes[1,1].legend()


parameters = ["t_rise", "t_decay", "t_peak", "t_90"]

print("\nAverage Outburst Times")
print("-" * 50)

for param in parameters:
    ns_mean = ns[param].mean()
    bh_mean = bh[param].mean()

    ns_std = ns[param].std()
    bh_std = bh[param].std()

    ns_n = ns[param].count()   # Number of non-NaN values
    bh_n = bh[param].count()

    print(f"\n{param}:")
    print(f"  NS : {ns_mean:.2f} ± {ns_std:.2f} days (N={ns_n})")
    print(f"  BH : {bh_mean:.2f} ± {bh_std:.2f} days (N={bh_n})")
    
plt.tight_layout()
plt.show()