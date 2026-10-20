
import os
import pandas as pd
import matplotlib.pyplot as plt

# Paths
log_file = os.path.join("outputs", "performance_log.csv")
output_dir = "outputs"

# Check whether the CSV exists
if not os.path.exists(log_file):
    print("Error: performance_log.csv not found.")
    print("Run main.py first to generate performance data.")
    raise SystemExit(1)

# Load performance data
df = pd.read_csv(log_file)

if df.empty:
    print("Error: CSV file contains no performance records.")
    raise SystemExit(1)

# Convert relevant columns to numeric values
numeric_columns = [
    "Total_Objects",
    "Extracted_Shapes",
    "Processing_Time_ms",
    "FPS",
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.dropna(subset=["Processing_Time_ms", "FPS"])

if df.empty:
    print("Error: No valid performance measurements found.")
    raise SystemExit(1)

# Calculate summary statistics
print("\n" + "=" * 45)
print("COMPUTER VISION PERFORMANCE SUMMARY")
print("=" * 45)
print(f"Valid frames analyzed: {len(df)}")
print(f"Average processing time: {df['Processing_Time_ms'].mean():.2f} ms")
print(f"Median processing time: {df['Processing_Time_ms'].median():.2f} ms")
print(f"Average processing FPS: {df['FPS'].mean():.2f}")
print(f"Median processing FPS: {df['FPS'].median():.2f}")
print(f"Maximum detected objects: {df['Total_Objects'].max():.0f}")
print(f"Average detected objects: {df['Total_Objects'].mean():.2f}")
print(f"Average extracted shapes: {df['Extracted_Shapes'].mean():.2f}")

# Save summary statistics
summary_file = os.path.join(output_dir, "performance_summary.txt")

with open(summary_file, "w", encoding="utf-8") as file:
    file.write("COMPUTER VISION PERFORMANCE SUMMARY\n")
    file.write("=" * 40 + "\n")
    file.write(f"Valid frames analyzed: {len(df)}\n")
    file.write(f"Average processing time: {df['Processing_Time_ms'].mean():.2f} ms\n")
    file.write(f"Median processing time: {df['Processing_Time_ms'].median():.2f} ms\n")
    file.write(f"Average processing FPS: {df['FPS'].mean():.2f}\n")
    file.write(f"Median processing FPS: {df['FPS'].median():.2f}\n")
    file.write(f"Maximum detected objects: {df['Total_Objects'].max():.0f}\n")
    file.write(f"Average detected objects: {df['Total_Objects'].mean():.2f}\n")
    file.write(f"Average extracted shapes: {df['Extracted_Shapes'].mean():.2f}\n")

# Graph 1: Processing time over frames
plt.figure()
plt.plot(df["Processing_Time_ms"].reset_index(drop=True))
plt.title("Processing Time per Frame")
plt.xlabel("Frame Record")
plt.ylabel("Processing Time (ms)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "processing_time.png"), dpi=150)
plt.close()

# Graph 2: FPS over frames
plt.figure()
plt.plot(df["FPS"].reset_index(drop=True))
plt.title("Processing FPS per Frame")
plt.xlabel("Frame Record")
plt.ylabel("FPS")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "processing_fps.png"), dpi=150)
plt.close()

# Graph 3: Object counts over frames
plt.figure()
plt.plot(
    df["Total_Objects"].reset_index(drop=True),
    label="YOLO Objects"
)
plt.plot(
    df["Extracted_Shapes"].reset_index(drop=True),
    label="Extracted Contours"
)
plt.title("Objects and Extracted Shapes per Frame")
plt.xlabel("Frame Record")
plt.ylabel("Count")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "object_counts.png"), dpi=150)
plt.close()

print("\nAnalysis completed successfully.")
print("Summary saved to:", summary_file)
print("Graphs saved in:", output_dir)
