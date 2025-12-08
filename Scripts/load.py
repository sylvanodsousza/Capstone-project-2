import pandas as pd
import glob

def load_all_data():
    print("🔍 Loading datasets...\n")

    # Core TRAFFIC datasets
    incidents = pd.read_csv("data/incident-data-3322.csv")
    locs = pd.read_csv("data/incident-summary-and-locations-3322.csv")
    species = pd.read_csv("data/incident-summary-and-species-3322.csv")

    # Extended wildlife dataset (Excel)
    wildlife_xlsx = pd.read_excel("data/data_wildlife_trafficking (3).xlsx")

    # Load ALL trade_db files automatically (trade_db_30.csv → trade_db_50.csv)
    trade_files = glob.glob("data/trade_db_*.csv")
    trade_list = []

    for f in trade_files:
        print(f"📥 Loading {f}")
        df = pd.read_csv(f)
        df["source_file"] = f  # useful for debugging
        trade_list.append(df)

    # Combine all trade db files into one dataframe
    if len(trade_list) > 0:
        trade_db = pd.concat(trade_list, ignore_index=True)
        print(f"\n✔ Combined Trade DB: {trade_db.shape[0]} rows from {len(trade_files)} files.\n")
    else:
        trade_db = pd.DataFrame()
        print("⚠ No trade_db files found from trade_db_30.csv to trade_db_50.csv.\n")

    # World Bank datasets
    wdi_mam = pd.read_csv("data/WB_WDI_EN_MAM_THRD_NO.csv")
    wdi_bird = pd.read_csv("data/WB_WDI_EN_BIR_THRD_NO.csv")
    wdi_fish = pd.read_csv("data/WB_WDI_EN_FSH_THRD_NO.csv")

    print("✔ All datasets loaded.\n")

    print("📊 Shapes:")
    print(f"Incidents: {incidents.shape}")
    print(f"Locations: {locs.shape}")
    print(f"Species: {species.shape}")
    print(f"Wildlife XLSX: {wildlife_xlsx.shape}")
    print(f"Combined Trade DB: {trade_db.shape}")
    print(f"WDI Mammals: {wdi_mam.shape}")
    print(f"WDI Birds: {wdi_bird.shape}")
    print(f"WDI Fish: {wdi_fish.shape}\n")

    return {
        "incidents":incidents,
        "locs":locs,
        "species":species,
        "wildlife_xlsx":wildlife_xlsx,
        "trade_db":trade_db,
        "wdi_mam":wdi_mam,
        "wdi_bird":wdi_bird,
        "wdi_fish":wdi_fish
    }

if __name__ == "__main__":
    load_all_data()
