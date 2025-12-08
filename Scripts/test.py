import pandas as pd

mam = pd.read_csv("data/WB_WDI_EN_MAM_THRD_NO.csv")
bird = pd.read_csv("data/WB_WDI_EN_BIR_THRD_NO.csv")
fish = pd.read_csv("data/WB_WDI_EN_FSH_THRD_NO.csv")

print("\nMAM COLUMNS:\n", mam.columns.tolist())
print("\nBIRD COLUMNS:\n", bird.columns.tolist())
print("\nFISH COLUMNS:\n", fish.columns.tolist())
