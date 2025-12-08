import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ensure_folders():
    folders = [
        "outputs",
        "outputs/processed",
        "outputs/plots",
        "outputs/tables"
    ]
    for f in folders:
        os.makedirs(f, exist_ok=True)

def run_monte_carlo():
    print("🎲 Running Monte Carlo simulation for illicit market value...\n")

    ensure_folders()

    # ----------------------------------------------------
    # 1️⃣ LOAD RAW SPECIES DATA WITH VALUE FIELD
    # ----------------------------------------------------
    print("🔍 Loading species value data...")

    species_raw = pd.read_csv("data/incident-summary-and-species-3322.csv", low_memory=False)

    # Rename to standard names
    species_raw = species_raw.rename(columns={
        "Full Scientific Name":"full_name",
        "Common Name":"species_name",
        "Value in USD":"value_usd"
    })

    # Clean
    species_raw["species_name"] = species_raw["species_name"].astype(str).str.strip().str.lower()
    species_raw["full_name"] = species_raw["full_name"].astype(str).str.strip().str.lower()
    species_raw["value_usd"] = pd.to_numeric(species_raw["value_usd"], errors="coerce")

    # Keep only rows with a valid positive value
    species_val = species_raw.dropna(subset=["value_usd"])
    species_val = species_val[species_val["value_usd"] > 0]

    print(f"✔ Valid value rows: {species_val.shape[0]}")
    if species_val.empty:
        print("⚠ No valid value data found. Aborting Monte Carlo.")
        return

    # ----------------------------------------------------
    # 2️⃣ OBSERVED MARKET VALUE (SEIZURE-BASED)
    # ----------------------------------------------------
    print("\n💰 Calculating observed seizure-based value...")

    observed_total = species_val["value_usd"].sum()
    print(f"Observed total value of seizures (USD): {observed_total:,.2f}")

    # Species-level observed value
    species_value = (
        species_val.groupby("species_name")["value_usd"]
        .sum()
        .reset_index()
        .sort_values("value_usd", ascending=False)
    )
    species_value["share_pct"] = 100 * species_value["value_usd"] / observed_total

    species_value.to_csv("outputs/tables/species_value_observed.csv", index=False)
    print("✔ Saved species_value_observed.csv\n")

    # ----------------------------------------------------
    # 3️⃣ MONTE CARLO: SCALE BY UNCERTAIN DETECTION RATE
    # ----------------------------------------------------
    print("🎲 Simulating true market value with detection uncertainty...")

    # ASSUMPTION: true detection rate is between 10% and 30%
    detection_low = 0.10
    detection_high = 0.30
    iterations = 10000

    detection_rates = np.random.uniform(detection_low, detection_high, size=iterations)
    # True total market value = Observed seizures / detection rate
    estimated_true_values = observed_total / detection_rates

    mc_df = pd.DataFrame({
        "iteration": np.arange(1, iterations + 1),
        "detection_rate": detection_rates,
        "estimated_true_value_usd": estimated_true_values
    })

    mc_df.to_csv("outputs/tables/monte_carlo_samples.csv", index=False)
    print("✔ Saved monte_carlo_samples.csv\n")

    # ----------------------------------------------------
    # 4️⃣ SUMMARY STATISTICS
    # ----------------------------------------------------
    print("📊 Computing summary statistics...")

    mean_val = estimated_true_values.mean()
    median_val = np.median(estimated_true_values)
    p05 = np.percentile(estimated_true_values, 5)
    p95 = np.percentile(estimated_true_values, 95)

    summary = pd.DataFrame([{
        "observed_total_usd": observed_total,
        "det_low": detection_low,
        "det_high": detection_high,
        "mean_true_value_usd": mean_val,
        "median_true_value_usd": median_val,
        "p05_true_value_usd": p05,
        "p95_true_value_usd": p95
    }])

    summary.to_csv("outputs/tables/monte_carlo_summary.csv", index=False)

    print("✔ Monte Carlo summary:")
    print(summary.to_string(index=False))
    print()

    # ----------------------------------------------------
    # 5️⃣ PLOT DISTRIBUTION
    # ----------------------------------------------------
    print("📈 Plotting distribution of estimated true market value...")

    plt.figure(figsize=(10, 6))
    plt.hist(estimated_true_values / 1e9, bins=50, alpha=0.7)
    plt.xlabel("Estimated True Market Value (Billion USD)")
    plt.ylabel("Frequency")
    plt.title("Monte Carlo Simulation of Global Illegal Wildlife Trade Value")
    plt.tight_layout()
    plt.savefig("outputs/plots/monte_carlo_total_value.png", dpi=300)
    plt.close()

    print("✔ Saved monte_carlo_total_value.png\n")
    print("🎉 Monte Carlo simulation complete!\n")

if __name__ == "__main__":
    run_monte_carlo()
