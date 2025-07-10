import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

print("Current working directory:", os.getcwd())

csv_path = 'cpu_reward_profile.csv'

if not os.path.isfile(csv_path):
    raise FileNotFoundError(f"CSV file not found at: {csv_path}")

df = pd.read_csv(csv_path)

df = df[df['n_candidates'] >= 400]

min_cand = df['n_candidates'].min()
max_cand = df['n_candidates'].max()
bins = np.logspace(np.log10(min_cand), np.log10(max_cand), 20)

hist_reward, bin_edges = np.histogram(df['n_candidates'], bins=bins, weights=df['reward_time_ms'])
hist_counts, _ = np.histogram(df['n_candidates'], bins=bins)

hist_counts = np.maximum(hist_counts, 1)

hist_reward = hist_reward / hist_counts
bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])

widths = np.diff(bin_edges)
plt.figure(figsize=(10, 6))
plt.bar(bin_centers, hist_reward, width=widths, align='center', label='Reward Calculation Time', color='#E41A1C', alpha=0.7)

sigma = 1.5
plt.plot(bin_centers, gaussian_filter1d(hist_reward, sigma), '-', color='#E41A1C', label='Reward Calc (smoothed)', linewidth=2)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Number of Candidates', fontsize=16)
plt.ylabel('Time (ms)', fontsize=16)
plt.legend(fontsize=13)
plt.grid(True, which="both", ls="--")
plt.tight_layout()