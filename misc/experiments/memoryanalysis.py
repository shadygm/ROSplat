import pandas as pd
import matplotlib.pyplot as plt
import glob

# Set global font size to 18 pt for all plot elements
plt.rcParams.update({'font.size': 18})

def load_all_memory(files_pattern):
    files = sorted(glob.glob(files_pattern))
    all_dfs = []

    for f in files:
        df = pd.read_csv(f, sep=" ", header=None, names=["Splats", "Usage_MB"])
        df.sort_values("Splats", inplace=True)  # Ensure order for plotting
        all_dfs.append(df)

    return files, all_dfs

def plot_memory_usage( files, dfs, color, avg_label, output_filename):
    plt.figure(figsize=(10, 5))

    # Interpolate all files to align on common X values (union of all)
    all_splats = sorted(set().union(*[set(df["Splats"]) for df in dfs]))
    aligned_usages = []

    for df in dfs:
        interp = pd.Series(df["Usage_MB"].values, index=df["Splats"])
        interp = interp.reindex(all_splats).interpolate().fillna(method='bfill').fillna(method='ffill')
        aligned_usages.append(interp)

    # Compute mean across all files
    avg_usage = pd.concat(aligned_usages, axis=1).mean(axis=1)

    # Plot mean line only
    plt.plot(avg_usage.index, avg_usage.values, color=color, label=avg_label, linewidth=2.0)

    plt.xlabel("Number of Gaussians")
    plt.ylabel("Memory Usage (MB)")
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.show()

# --- RAM ---
ram_files, ram_dfs = load_all_memory("ram*.csv")
plot_memory_usage(
    files=ram_files,
    dfs=ram_dfs,
    color="#1f78b4",  # Changed to blue
    avg_label="Average RAM Usage",
    output_filename="gaussian_vs_ram_usage_avg.png"
)

# --- VRAM ---
vram_files, vram_dfs = load_all_memory("vram*.csv")
plot_memory_usage(
    files=vram_files,
    dfs=vram_dfs,
    color="#33a02c",  # Changed to green
    avg_label="Average VRAM Usage",
    output_filename="gaussian_vs_vram_usage_avg.png"
)
