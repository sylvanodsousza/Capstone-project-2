import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "outputs" / "tables"

def load(name):
    path = BASE / name
    print("\n--- Loading:", path)
    return pd.read_csv(path)

# 1. Monte Carlo simulation summary
mc = load("monte_carlo_samples.csv")
print("\nMonte Carlo Stats:")
print("Median:", mc.iloc[:,0].median())
print("P10:", mc.iloc[:,0].quantile(0.10))
print("P90:", mc.iloc[:,0].quantile(0.90))

# 2. Top 5 species
sp = load("species_frequency.csv")
print("\nTop 5 Species:")
print(sp.head(5))

# 3. Top 5 trafficking routes
rt = load("route_edges.csv")
print("\nTop 5 Routes:")
print(rt.sort_values('count', ascending=False).head(5))

# 4. Highest risk countries
rk = load("country_risk_scores.csv")
print("\nHighest-Risk Countries:")
print(rk.sort_values('risk_score', ascending=False).head(5))

# 5. Regression Results
reg = load("regression_results.csv")
print("\nRegression Results:")
print(reg)
