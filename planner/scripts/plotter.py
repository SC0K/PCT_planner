import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

print("Current working directory:", os.getcwd())

csv_path = 'batched_ray_reward_profile.csv'

if not os.path.isfile(csv_path):
    raise FileNotFoundError(f"CSV file not found at: {csv_path}")

df = pd.read_csv(csv_path, header=None,
                 names=['num_candidates', 'setup_time', 'raycast_time', 'post_time', 'total_time'])

# Use log bins for num_candidates
min_cand = df['num_candidates'].min()
max_cand = df['num_candidates'].max()
bins = np.logspace(np.log10(min_cand), np.log10(max_cand), 20)

# Compute histogram: sum of times and count of candidates per bin
hist_setup, bin_edges = np.histogram(df['num_candidates'], bins=bins, weights=df['setup_time'])
hist_raycast, _ = np.histogram(df['num_candidates'], bins=bins, weights=df['raycast_time'])
hist_post, _ = np.histogram(df['num_candidates'], bins=bins, weights=df['post_time'])
hist_counts, _ = np.histogram(df['num_candidates'], bins=bins)

# Avoid division by zero
hist_counts = np.maximum(hist_counts, 1)

# Compute average time per candidate
hist_setup = hist_setup / hist_counts
hist_raycast = hist_raycast / hist_counts
hist_post = hist_post / hist_counts
bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
hist_setup = hist_setup * 1000
hist_raycast = hist_raycast * 1000
hist_post = hist_post * 1000

# Prepare data for stacked bar plot
widths = np.diff(bin_edges)
plt.figure(figsize=(10, 6))
plt.bar(bin_centers, hist_setup, width=widths, align='center', label='Setup Time', color='#E41A1C', alpha=0.7)
plt.bar(bin_centers, hist_raycast, width=widths, align='center', bottom=hist_setup, label='Raycast Time', color='#377EB8', alpha=0.7)
plt.bar(bin_centers, hist_post, width=widths, align='center', bottom=hist_setup+hist_raycast, label='Post Time', color='#4DAF4A', alpha=0.7)

# Smoothed curves for all components
sigma = 1.5
plt.plot(bin_centers, gaussian_filter1d(hist_setup, sigma), '-', color='#E41A1C', label='Setup Time (smoothed)', linewidth=2)
plt.plot(bin_centers, gaussian_filter1d(hist_raycast, sigma), '-', color='#377EB8', label='Raycast Time (smoothed)', linewidth=2)
plt.plot(bin_centers, gaussian_filter1d(hist_post, sigma), '-', color='#4DAF4A', label='Post Time (smoothed)', linewidth=2)

# Smoothed curve for total time
hist_total = hist_setup + hist_raycast + hist_post
plt.plot(bin_centers, gaussian_filter1d(hist_total, sigma), 'k-', label='Total Time (smoothed)', linewidth=2)


plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# plt.xscale('log')
# plt.yscale('log')
plt.xlabel('Number of Candidates',fontsize=16)
plt.ylabel('Time (ms)',fontsize=16)
plt.title('Time for Rewards Calculation of Candidates via Raycasting',fontsize=16)
plt.legend(fontsize=13)  # Increased font size for the legend
plt.grid(True, which="both", ls="--")
plt.tight_layout()
# Add more ticks to the x-axis and increase font size

plt.show()
