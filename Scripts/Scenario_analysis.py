import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ensure_folders():
    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)

def load_observed_total():
    """Load observed total value from species_value_observed."""
    species_val = pd.read_csv("outputs/tables/species_value_observed.csv")
    # This file has species_name, value_usd, share_pct
    observed_total = species_val["value_usd"].sum()
    return observed_total

def run_enforcement_scenarios():
    print("⚖️ Running enforcement scenarios (Monte Carlo)...\n")

    ensure_folders()

    observed_total = load_observed_total()
    print(f"Observed seizures total (USD): {observed_total:,.2f}\n")

    scenarios = {
        "baseline_10_30": (0.10, 0.30),
        "improved_20_40": (0.20, 0.40),
        "weak_05_20": (0.05, 0.20),
    }

    records = []
    all_samples = []

    iterations = 10000

    for name, (low, high) in scenarios.items():
        print(f"🎲 Scenario '{name}' with detection range [{low:.2f}, {high:.2f}]")

        detection_rates = np.random.uniform(low, high, size=iterations)
        est_true_values = observed_total / detection_rates

        # Summaries
        mean_val = est_true_values.mean()
        median_val = np.median(est_true_values)
        p05 = np.percentile(est_true_values, 5)
        p95 = np.percentile(est_true_values, 95)

        records.append({
            "scenario": name,
            "det_low": low,
            "det_high": high,
            "mean_true_value_usd": mean_val,
            "median_true_value_usd": median_val,
            "p05_true_value_usd": p05,
            "p95_true_value_usd": p95
        })

        all_samples.append(
            pd.DataFrame({
                "scenario": name,
                "detection_rate": detection_rates,
                "estimated_true_value_usd": est_true_values
            })
        )

    summary_df = pd.DataFrame(records)
    samples_df = pd.concat(all_samples, ignore_index=True)

    summary_df.to_csv("outputs/tables/scenario_enforcement_summary.csv", index=False)
    samples_df.to_csv("outputs/tables/scenario_enforcement_samples.csv", index=False)

    print("\n✔ Saved scenario_enforcement_summary.csv")
    print("✔ Saved scenario_enforcement_samples.csv\n")

    # Plot comparison (mean & 5–95% range, in billions)
    plt.figure(figsize=(10, 6))
    x = np.arange(len(summary_df))
    means = summary_df["mean_true_value_usd"] / 1e9
    lows = (summary_df["p05_true_value_usd"] / 1e9)
    highs = (summary_df["p95_true_value_usd"] / 1e9)
    err_low = means - lows
    err_high = highs - means

    plt.errorbar(
        x, means,
        yerr=[err_low, err_high],
        fmt="o",
        capsize=5
    )
    plt.xticks(x, summary_df["scenario"], rotation=20)
    plt.ylabel("Estimated True Market Size (Billion USD)")
    plt.title("Enforcement Scenarios — Illegal Wildlife Trade Market Size")
    plt.tight_layout()
    plt.savefig("outputs/plots/scenario_enforcement_comparison.png", dpi=300)
    plt.close()

    print("✔ Saved scenario_enforcement_comparison.png\n")


def run_chokepoint_scenario(top_n=3):
    print("🧬 Running chokepoint disruption scenario...\n")

    ensure_folders()

    # Load route edges & chokepoints
    edges = pd.read_csv("outputs/tables/route_edges.csv")
    chokepoints = pd.read_csv("outputs/tables/chokepoints.csv")

    if edges.empty or chokepoints.empty:
        print("⚠️ No edges or chokepoints found — skipping chokepoint scenario.")
        return

    # Compute baseline total flow
    baseline_total_flow = edges["count"].sum()

    # Get top N chokepoint countries
    top_choke = chokepoints.head(top_n)["country"].tolist()
    print(f"Top {top_n} chokepoints: {top_choke}")

    # Remove all routes involving these countries
    mask = ~(
        edges["origin_country"].isin(top_choke) |
        edges["destination_country"].isin(top_choke)
    )
    edges_disrupted = edges[mask]

    disrupted_total_flow = edges_disrupted["count"].sum()

    # Compute % reduction
    if baseline_total_flow > 0:
        reduction_pct = 100 * (baseline_total_flow - disrupted_total_flow) / baseline_total_flow
    else:
        reduction_pct = 0.0

    # Save table
    scenario_df = pd.DataFrame([{
        "baseline_total_flow": baseline_total_flow,
        "disrupted_total_flow": disrupted_total_flow,
        "reduction_pct": reduction_pct,
        "top_chokepoints_removed": ", ".join(top_choke)
    }])

    scenario_df.to_csv("outputs/tables/scenario_chokepoint_impact.csv", index=False)

    print("✔ Saved scenario_chokepoint_impact.csv\n")
    print(f"Baseline total flow: {baseline_total_flow}")
    print(f"After removing {top_n} chokepoints: {disrupted_total_flow}")
    print(f"Estimated reduction in trafficking routes: {reduction_pct:.2f}%\n")

    # Simple bar chart
    plt.figure(figsize=(6, 5))
    plt.bar(["Baseline", "After Disruption"], [baseline_total_flow, disrupted_total_flow])
    plt.ylabel("Total Route Flow (count)")
    plt.title(f"Impact of Removing Top {top_n} Chokepoints")
    plt.tight_layout()
    plt.savefig("outputs/plots/scenario_chokepoint_impact.png", dpi=300)
    plt.close()

    print("✔ Saved scenario_chokepoint_impact.png\n")


def run_all_scenarios():
    print("🚀 Running all scenario models...\n")
    run_enforcement_scenarios()
    run_chokepoint_scenario(top_n=3)
    print("🎉 All scenario modeling complete!")

if __name__ == "__main__":
    run_all_scenarios()
