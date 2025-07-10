import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

print("Current working directory:", os.getcwd())

csv_path = 'cpu_reward_profile.csv'  # Update this if your file is named differently

if not os.path.isfile(csv_path):
    raise FileNotFoundError(f"CSV file not found at: {csv_path}")

df = pd.read_csv(csv_path)

# Remove outliers with very small n_candidates (e.g., < 100)
df = df[df['n_candidates'] >= 400]

# Use log bins for n_candidates
min_cand = df['n_candidates'].min()
max_cand = df['n_candidates'].max()
bins = np.logspace(np.log10(min_cand), np.log10(max_cand), 20)

# Compute histogram: sum of times and count of candidates per bin
hist_reward, bin_edges = np.histogram(df['n_candidates'], bins=bins, weights=df['reward_time_ms'])
hist_counts, _ = np.histogram(df['n_candidates'], bins=bins)

# Avoid division by zero
hist_counts = np.maximum(hist_counts, 1)

# Compute average time per candidate
hist_reward = hist_reward / hist_counts
bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])

# Prepare data for bar plot
widths = np.diff(bin_edges)
plt.figure(figsize=(10, 6))
plt.bar(bin_centers, hist_reward, width=widths, align='center', label='Reward Calculation Time', color='#E41A1C', alpha=0.7)

# Smoothed curve
sigma = 1.5
plt.plot(bin_centers, gaussian_filter1d(hist_reward, sigma), '-', color='#E41A1C', label='Reward Calc (smoothed)', linewidth=2)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# plt.xscale('log')
# plt.yscale('log')
plt.xlabel('Number of Candidates', fontsize=16)
plt.ylabel('Time (ms)', fontsize=16)
# plt.title('Reward Calculation Time vs Number of Candidates (CPU)', fontsize=16)
plt.legend(fontsize=13)
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()