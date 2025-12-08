import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os

def ensure_folders():
    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)

def run_regression():

    print("📊 Running Regression Analysis...\n")
    ensure_folders()

    # ----------------------------------------------------
    # 1️⃣ LOAD DATASETS
    # ----------------------------------------------------
    print("🔍 Loading datasets...")

    cy = pd.read_csv("outputs/processed/country_year.csv")

    mam = pd.read_csv("data/WB_WDI_EN_MAM_THRD_NO.csv")
    bird = pd.read_csv("data/WB_WDI_EN_BIR_THRD_NO.csv")
    fish = pd.read_csv("data/WB_WDI_EN_FSH_THRD_NO.csv")

    # ----------------------------------------------------
    # 2️⃣ CLEAN WDI FORMAT (SDMX STRUCTURE)
    # ----------------------------------------------------
    print("🧽 Cleaning WDI SDMX files...")

    mam_clean = mam.rename(columns={
        "REF_AREA_LABEL": "country",
        "TIME_PERIOD": "year",
        "OBS_VALUE": "threatened_mammals"
    })[["country", "year", "threatened_mammals"]]

    bird_clean = bird.rename(columns={
        "REF_AREA_LABEL": "country",
        "TIME_PERIOD": "year",
        "OBS_VALUE": "threatened_birds"
    })[["country", "year", "threatened_birds"]]

    fish_clean = fish.rename(columns={
        "REF_AREA_LABEL": "country",
        "TIME_PERIOD": "year",
        "OBS_VALUE": "threatened_fish"
    })[["country", "year", "threatened_fish"]]

    # Convert year to numeric
    mam_clean["year"] = pd.to_numeric(mam_clean["year"], errors="coerce")
    bird_clean["year"] = pd.to_numeric(bird_clean["year"], errors="coerce")
    fish_clean["year"] = pd.to_numeric(fish_clean["year"], errors="coerce")

    # ----------------------------------------------------
    # 3️⃣ MERGE ALL COUNTRY-YEAR DATA
    # ----------------------------------------------------
    print("🔗 Merging datasets...")

    df = cy.merge(mam_clean, left_on=["incident_country", "year"],
                  right_on=["country", "year"], how="left")

    df = df.merge(bird_clean, on=["country", "year"], how="left")
    df = df.merge(fish_clean, on=["country", "year"], how="left")

    df = df.drop(columns=["country"])
    df = df.dropna(subset=["incident_count"])

    print("✔ Merged dataset shape:", df.shape, "\n")

    # ----------------------------------------------------
    # 4️⃣ REGRESSION MODEL
    # ----------------------------------------------------
    print("📐 Preparing regression model...")

    y = df["incident_count"]

    X = df[[
        "threatened_mammals",
        "threatened_birds",
        "threatened_fish"
    ]]

    X = X.fillna(0)  # missing values = 0 species reported
    X = sm.add_constant(X)

    print("📈 Running OLS regression...\n")

    model = sm.OLS(y, X).fit()

    # Save summary
    summary_text = model.summary().as_text()
    with open("outputs/tables/regression_summary.txt", "w") as f:
        f.write(summary_text)

    print(summary_text)

    # Save coefficients
    coef_df = pd.DataFrame({
        "variable": model.params.index,
        "coefficient": model.params.values
    })

    coef_df.to_csv("outputs/tables/regression_results.csv", index=False)

    # ----------------------------------------------------
    # 5️⃣ RISK SCORE
    # ----------------------------------------------------
    print("\n🔥 Creating risk score table...")

    df["risk_score"] = (
        df["threatened_mammals"] * model.params.get("threatened_mammals", 0) +
        df["threatened_birds"] * model.params.get("threatened_birds", 0) +
        df["threatened_fish"] * model.params.get("threatened_fish", 0)
    )

    risk = (
        df.groupby("incident_country")["risk_score"]
        .mean()
        .reset_index()
        .sort_values("risk_score", ascending=False)
    )

    risk.to_csv("outputs/tables/country_risk_scores.csv", index=False)

    # ----------------------------------------------------
    # 6️⃣ PLOT COEFFICIENTS
    # ----------------------------------------------------
    plt.figure(figsize=(8,5))
    plt.bar(coef_df["variable"], coef_df["coefficient"])
    plt.title("Predictors of Wildlife Trafficking Incidents")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("outputs/plots/regression_coefficients.png", dpi=300)
    plt.close()

    print("✔ Regression complete!")

if __name__ == "__main__":
    run_regression()
