
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load recorded performance data
file_path = "outputs/performance_log.csv"
df = pd.read_csv(file_path)

if df.empty:
    print("No performance data found.")
    raise SystemExit

# Convert measurement columns to numeric values
for column in [
    "Total_Objects",
    "Extracted_Shapes",
    "Processing_Time_ms",
    "FPS",
]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

print("\n===== COMPUTER VISION PERFORMANCE REPORT =====")
print(f"Total recorded frames: {len(df)}")
print(f"Average processing time: {df['Processing_Time_ms'].mean():.2f} ms")
print(f"Minimum processing time: {df['Processing_Time_ms'].min():.2f} ms")
print(f"Maximum processing time: {df['Processing_Time_ms'].max():.2f} ms")
print(f"Average estimated FPS: {df['FPS'].mean():.2f}")
print(f"Minimum estimated FPS: {df['FPS'].min():.2f}")
print(f"Maximum estimated FPS: {df['FPS'].max():.2f}")
print(f"Average objects per frame: {df['Total_Objects'].mean():.2f}")
print(f"Average extracted shapes per frame: {df['Extracted_Shapes'].mean():.2f}")

# Create folder for analysis results
os.makedirs("outputs", exist_ok=True)

# Plot processing time
plt.figure(figsize=(10, 5))
plt.plot(df["Processing_Time_ms"].dropna())
plt.title("Processing Time per Frame")
plt.xlabel("Recorded Frame")
plt.ylabel("Processing Time (ms)")
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/processing_time.png", dpi=150)
plt.close()

# Plot estimated FPS
plt.figure(figsize=(10, 5))
plt.plot(df["FPS"].dropna())
plt.title("Estimated FPS per Frame")
plt.xlabel("Recorded Frame")
plt.ylabel("Estimated FPS")
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/fps_analysis.png", dpi=150)
plt.close()

print("\nAnalysis complete.")
print("Charts saved in the outputs folder.")
