import pandas as pd

df = pd.read_csv("data/incident-summary-and-locations-3322.csv")

print("\nUnique values in 'Role':")
print(df["Role"].dropna().unique())

print("\nSample of Role & Country columns:")
print(df[["Report ID", "Role", "Country"]].head(20))


