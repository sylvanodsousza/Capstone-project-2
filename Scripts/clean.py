import pandas as pd
from load import load_all_data

def clean_all():
    print("🧹 Cleaning and preparing data...\n")

    # Load all raw datasets
    data = load_all_data()

    incidents = data["incidents"]
    locs = data["locs"]
    species = data["species"]

    # ----------------------------------------------------
    # 1️⃣ CLEAN INCIDENTS DATA
    # ----------------------------------------------------
    print("🧼 Cleaning incidents dataset...")

    incidents = incidents.rename(columns={
        "Report ID":"incident_id",
        "Country of Incident":"country",
        "Date of Incident":"incident_date",
        "Category of Incident":"category",
        "Links to Corruption":"corruption",
        "Links to Serious or Organised Crime":"organized_crime"
    })

    # Convert dates
    incidents["incident_date"] = pd.to_datetime(incidents["incident_date"], errors="coerce")
    incidents["year"] = incidents["incident_date"].dt.year

    # Clean text values
    incidents["country"] = incidents["country"].astype(str).str.strip()

    # Drop invalid rows
    incidents = incidents.dropna(subset=["country", "year"])

    print(f"✔ Incidents cleaned: {incidents.shape}\n")

    # ----------------------------------------------------
    # 2️⃣ CLEAN SPECIES DATA
    # ----------------------------------------------------
    print("🦜 Cleaning species dataset...")

    species = species.rename(columns={
        "Report ID":"incident_id",
        "Full Scientific Name":"species_name",
        "Common Name":"common_name",
        "Count":"count",
        "Weight":"weight"
    })

    species["species_name"] = species["species_name"].astype(str).str.strip().str.lower()
    species["common_name"] = species["common_name"].astype(str).str.strip().str.lower()

    # Merge with incidents for country and year
    species_merged = species.merge(
        incidents[["incident_id", "country", "year"]],
        on="incident_id",
        how="left"
    )

    print(f"✔ Species merged: {species_merged.shape}\n")

    # ----------------------------------------------------
    # 3️⃣ CLEAN LOCATION DATA (Origin, Transit, Destination)
    # ----------------------------------------------------
    print("📍 Cleaning location dataset...")

    locs = locs.rename(columns={
        "Report ID":"incident_id",
        "Country":"country_loc",
        "Role":"role"
    })

    locs["country_loc"] = locs["country_loc"].astype(str).str.strip()
    locs["role"] = locs["role"].astype(str).str.strip().str.lower()

    # Extract origin
    origin = locs[locs["role"] == "origin"][["incident_id", "country_loc"]].rename(
        columns={"country_loc":"origin_country"}
    )

    # Extract destination
    destination = locs[locs["role"] == "destination"][["incident_id", "country_loc"]].rename(
        columns={"country_loc":"destination_country"}
    )

    # Extract multiple transit countries per incident
    transit = locs[locs["role"] == "transit"][["incident_id", "country_loc"]]
    transit_grouped = transit.groupby("incident_id")["country_loc"].apply(list).reset_index().rename(
        columns={"country_loc":"transit_countries"}
    )

    # Merge into a single geographic table
    incidents_geo = incidents.copy()
    incidents_geo = incidents_geo.merge(origin, on="incident_id", how="left")
    incidents_geo = incidents_geo.merge(destination, on="incident_id", how="left")
    incidents_geo = incidents_geo.merge(transit_grouped, on="incident_id", how="left")

    print(f"✔ Incidents + Geo merged: {incidents_geo.shape}\n")

    # ----------------------------------------------------
    # 4️⃣ CREATE COUNTRY-YEAR SUMMARY (Regression / Trends)
    # ----------------------------------------------------
    print("📊 Creating country-year summary...")

    country_year = (
        incidents.groupby(["country", "year"])
        .size()
        .reset_index(name="incident_count")
    )

    print(f"✔ Country-year summary ready: {country_year.shape}\n")

    # ----------------------------------------------------
    # 5️⃣ SAVE OUTPUT FILES
    # ----------------------------------------------------
    print("💾 Saving cleaned datasets...")

    incidents.to_csv("outputs/processed/incidents_cleaned.csv", index=False)
    species_merged.to_csv("outputs/processed/species_cleaned.csv", index=False)
    incidents_geo.to_csv("outputs/processed/incidents_geo.csv", index=False)
    country_year.to_csv("outputs/processed/country_year.csv", index=False)

    print("🎉 All cleaned data saved successfully!\n")
    print("Ready for NLP, Networks, Monte Carlo, Regression, and Scenarios.")

if __name__ == "__main__":
    clean_all()
