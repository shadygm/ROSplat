import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob

# Color palette for backends
color_map = {
    "CPU": "#33a02c",     # green
    "Cupy": "#1f78b4",    # blue
    "Torch": "#ff7f00",   # orange
}

# File patterns
backends = {
    "CPU": "./cpu*.csv",
    "Cupy": "./cupy*.csv",
    "Torch": "./torch*.csv"
}
# Set font to 18 point
plt.rcParams.update({'font.size': 18})

backend_results = {}

# --- Line Plot Prep ---
for name, pattern in backends.items():
    matched_files = sorted(glob.glob(pattern))
    print(f"[{name}] Matched files:", matched_files)
    
    fps_data = []
    for file in matched_files:
        df = pd.read_csv(file, sep=" ", header=None, names=["FPS", "Splats"])
        fps_data.append(df)

    if not fps_data:
        print(f"No files matched for {name}, skipping.")
        continue

    combined = pd.concat(fps_data)
    grouped = combined.groupby("Splats")["FPS"]
    mean_fps = grouped.mean()

    backend_results[name] = {
        "splats": mean_fps.index.to_numpy(),
        "mean_fps": mean_fps.to_numpy()
    }

# --- Plot: Smoothed FPS vs Splats ---
plt.figure(figsize=(10, 6))

for name, data in backend_results.items():
    splats = data["splats"]
    mean_fps = data["mean_fps"]

    smoothed_fps = np.convolve(mean_fps, np.ones(15)/15, mode='same')

    plt.plot(
        splats,
        smoothed_fps,
        label=name,
        color=color_map[name],
        linewidth=1.8
    )

# Styling for 18-point fonts
plt.xlabel("Number of Splats", fontsize=18)
plt.ylabel("FPS", fontsize=18)
plt.ylim(bottom=0)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend(fontsize=18, loc="upper right")
plt.tick_params(axis='both', labelsize=18)
plt.tight_layout()
plt.savefig("fps_vs_splats_styled.png", dpi=300)
plt.show()

# --- Bar Chart: Mean FPS Comparison with Std Dev ---

# --- Bar Chart: % Improvement Between Methods ---

# Compute mean FPS again
mean_fps_dict = {}
for name, pattern in backends.items():
    matched_files = sorted(glob.glob(pattern))
    fps_data = []

    for file in matched_files:
        df = pd.read_csv(file, sep=" ", header=None, names=["FPS", "Splats"])
        fps_data.append(df)

    if not fps_data:
        continue

    combined = pd.concat(fps_data)
    grouped = combined.groupby("Splats")["FPS"]
    mean_fps = grouped.mean()
    mean_fps_dict[name] = mean_fps.mean()

# Compute relative improvements
comparisons = [
    ("Torch", "CPU"),
    ("Cupy", "CPU"),
    ("Cupy", "Torch"),
]

labels = []
improvements = []

for a, b in comparisons:
    if b not in mean_fps_dict or a not in mean_fps_dict:
        continue
    rel_improvement = ((mean_fps_dict[a] - mean_fps_dict[b]) / mean_fps_dict[b]) * 100
    labels.append(f"{a} vs {b}")  # Only used for the x-axis labels.
    improvements.append(rel_improvement)

# Bar plot with adjacent bars.
x = np.arange(len(labels))
bar_width = 1  # Bars will directly touch one another.

plt.figure(figsize=(8, 6))
bars = plt.bar(x, improvements, color="#444444", width=bar_width, edgecolor="black")

# Add percentage labels below each bar.
offset = 1  # vertical offset from zero.
for bar, value in zip(bars, improvements):
    x_center = bar.get_x() + bar.get_width() / 2
    if value >= 0:
        # For positive bars, place the label below the bar (below zero).
        text_y = 0 - offset
        va = 'top'
    else:
        # For negative bars, place the label above the bar (above zero).
        text_y = 0 + offset
        va = 'bottom'
    
    plt.text(
        x_center,
        text_y,
        f"{value:.1f}%",
        ha='center',
        va=va,
        fontsize=18,
        color="black"
    )

plt.ylabel("Relative FPS Improvement (%)", fontsize=18)
plt.xticks(x, labels, fontsize=18)
plt.ylim(-10, 40)
plt.yticks(fontsize=18)
plt.grid(axis='y', linestyle='-', linewidth=0.5)
plt.tight_layout()
plt.savefig("bar_fps_percentage_comparison.png", dpi=300)
plt.show()
