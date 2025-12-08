import pandas as pd
import os

# ----------------------------------------------------
# Helper: ensure required folders exist
# ----------------------------------------------------
def ensure_folders():
    folders = [
        "outputs",
        "outputs/processed",
        "outputs/plots",
        "outputs/tables",
        "scripts",
        "data",
    ]
    for f in folders:
        os.makedirs(f, exist_ok=True)

# ----------------------------------------------------
# MAIN CLEANING LOGIC
# ----------------------------------------------------
def clean_all():
    print("🧹 Cleaning and preparing data...\n")

    ensure_folders()

    # ----------------------------------------------------
    # 1️⃣ LOAD CLEANED-READY DATASETS
    # ----------------------------------------------------
    print("🔍 Loading datasets...\n")

    incidents = pd.read_csv("data/incident-summary-and-locations-3322.csv")
    species = pd.read_csv("data/incident-summary-and-species-3322.csv")
    wildlife_xlsx = pd.read_excel("data/data_wildlife_trafficking (3).xlsx")

    # Load all trade DB CSVs
    trade_files = [f for f in os.listdir("data") if f.startswith("trade_db_") and f.endswith(".csv")]
    trade_dfs = []
    for f in trade_files:
        print(f"📥 Loading data/{f}")
        df = pd.read_csv(f"data/{f}", low_memory=False)
        trade_dfs.append(df)

    combined_trade = pd.concat(trade_dfs, ignore_index=True)
    print(f"\n✔ Combined Trade DB: {combined_trade.shape[0]} rows from {len(trade_files)} files.\n")

    # WDI datasets
    wdi_mam = pd.read_csv("data/WB_WDI_EN_MAM_THRD_NO.csv")
    wdi_bird = pd.read_csv("data/WB_WDI_EN_BIR_THRD_NO.csv")
    wdi_fish = pd.read_csv("data/WB_WDI_EN_FSH_THRD_NO.csv")

    print("✔ All datasets loaded.\n")

    print("📊 Shapes:")
    print(f"Incidents: {incidents.shape}")
    print(f"Species: {species.shape}")
    print(f"Wildlife XLSX: {wildlife_xlsx.shape}")
    print(f"Combined Trade DB: {combined_trade.shape}")
    print(f"WDI Mammals: {wdi_mam.shape}")
    print(f"WDI Birds: {wdi_bird.shape}")
    print(f"WDI Fish: {wdi_fish.shape}\n")

    # ----------------------------------------------------
    # 2️⃣ CLEAN INCIDENTS DATASET
    # ----------------------------------------------------
    print("🧼 Cleaning incidents dataset...")

    incidents = incidents.rename(columns={
        "Report ID": "incident_id",
        "Category of Incident": "category",
        "Country of Incident": "incident_country",
        "Date of Incident": "incident_date"
    })

    incidents["incident_date"] = pd.to_datetime(incidents["incident_date"], errors="coerce")

    print(f"✔ Incidents cleaned: {incidents.shape}\n")

    # ----------------------------------------------------
    # 3️⃣ CLEAN SPECIES DATASET
    # ----------------------------------------------------
    print("🦜 Cleaning species dataset...")

    species = species.rename(columns={
        "Report ID": "incident_id",
        "Full Scientific Name": "full_name",
        "Common Name": "species_name",
        "Class": "class"
    })

    species["species_name"] = species["species_name"].astype(str).str.strip().str.lower()
    species["full_name"] = species["full_name"].astype(str).str.strip().str.lower()

    # Combine species info into one table per incident
    species_grouped = (
        species.groupby("incident_id")
        .agg({
            "species_name": lambda x: list(x),
            "full_name": lambda x: list(x),
            "class": lambda x: list(x)
        })
        .reset_index()
    )

    print(f"✔ Species merged: {species_grouped.shape}\n")

    # ----------------------------------------------------
    # 4️⃣ CLEAN LOCATION DATA (Origin, Transit, Destination)
    # ----------------------------------------------------
    print("📍 Cleaning location dataset...\n")

    locs = pd.read_csv("data/incident-summary-and-locations-3322.csv")
    locs = locs.rename(columns={
        "Report ID": "incident_id",
        "Country": "country_loc",
        "Role": "role"
    })

    locs["role"] = locs["role"].astype(str).str.strip().str.lower()
    locs["country_loc"] = locs["country_loc"].astype(str).str.strip()

    # Define role groups
    origin_roles = ["origin location", "origin location"]
    destination_roles = ["destination location", "destination location"]
    transit_roles = ["transit location", "transit location"]

    origin = (
        locs[locs["role"].isin(origin_roles)]
        [["incident_id", "country_loc"]]
        .rename(columns={"country_loc": "origin_country"})
    )

    destination = (
        locs[locs["role"].isin(destination_roles)]
        [["incident_id", "country_loc"]]
        .rename(columns={"country_loc": "destination_country"})
    )

    transit = locs[locs["role"].isin(transit_roles)][["incident_id", "country_loc"]]

    transit_grouped = (
        transit.groupby("incident_id")["country_loc"]
        .apply(list)
        .reset_index()
        .rename(columns={"country_loc": "transit_countries"})
    )

    # Merge into full geo dataset
    incidents_geo = incidents.merge(origin, on="incident_id", how="left")
    incidents_geo = incidents_geo.merge(destination, on="incident_id", how="left")
    incidents_geo = incidents_geo.merge(transit_grouped, on="incident_id", how="left")
    incidents_geo = incidents_geo.merge(species_grouped, on="incident_id", how="left")

    print(f"✔ Incidents + Geo merged: {incidents_geo.shape}\n")

    # ----------------------------------------------------
    # 5️⃣ COUNTRY-YEAR SUMMARY
    # ----------------------------------------------------
    print("📊 Creating country-year summary...")

    country_year = (
        incidents_geo.groupby(["incident_country", incidents_geo["incident_date"].dt.year])
        .size()
        .reset_index(name="incident_count")
        .rename(columns={"incident_date": "year"})
    )

    print(f"✔ Country-year summary ready: {country_year.shape}\n")

    # ----------------------------------------------------
    # 6️⃣ SAVE ALL CLEANED OUTPUTS
    # ----------------------------------------------------
    print("💾 Saving cleaned datasets...")

    incidents.to_csv("outputs/processed/incidents_cleaned.csv", index=False)
    species_grouped.to_csv("outputs/processed/species_cleaned.csv", index=False)
    incidents_geo.to_csv("outputs/processed/incidents_geo.csv", index=False)
    country_year.to_csv("outputs/processed/country_year.csv", index=False)

    print("🎉 All cleaned data saved successfully!\n")
    print("Ready for NLP, Networks, Monte Carlo, Regression, and Scenario modeling.\n")


if __name__ == "__main__":
    clean_all()
