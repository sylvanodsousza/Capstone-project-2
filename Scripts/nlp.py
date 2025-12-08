import pandas as pd
import os
import re
from collections import Counter
import matplotlib.pyplot as plt

def run_nlp():
    print("🔍 Running NLP analysis...\n")

    # Ensure required folders exist
    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)

    # Load cleaned data
    incidents = pd.read_csv("outputs/processed/incidents_cleaned.csv")
    species = pd.read_csv("outputs/processed/species_cleaned.csv")

    # ----------------------------------------------------
    # 1️⃣ SPECIES FREQUENCY TABLE
    # ----------------------------------------------------
    print("🦜 Analyzing species frequency...\n")

    species_freq = (
        species["species_name"]
        .value_counts()
        .reset_index()
    )
    species_freq.columns = ["species_name", "count"]

    species_freq.to_csv("outputs/tables/species_frequency.csv", index=False)

    # Quick bar chart for your slides
    top_species = species_freq.head(15)
    plt.figure(figsize=(10,6))
    plt.barh(top_species["species_name"], top_species["count"])
    plt.title("Top Trafficked Species")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("outputs/plots/top_species.png")
    plt.close()

    print("✔ Species frequency saved and plotted.\n")

    # ----------------------------------------------------
    # 2️⃣ KEYWORD EXTRACTION (CATEGORY + SUBJECT)
    # ----------------------------------------------------
    print("📝 Extracting common trafficking-related keywords...\n")

    # Some datasets may have missing Subject column
    if "Subject" in incidents.columns:
        subjects = incidents["Subject"].fillna("").astype(str).str.lower()
    else:
        subjects = ""

    categories = incidents["category"].fillna("").astype(str).str.lower()

    text_series = categories + " " + subjects

    # Tokenize
    tokens = []
    for txt in text_series:
        tokens.extend(re.findall(r"[a-z]{4,}", txt))

    keyword_counts = Counter(tokens)
    keywords_df = pd.DataFrame(keyword_counts.most_common(200), columns=["keyword", "count"])

    keywords_df.to_csv("outputs/tables/keyword_frequency.csv", index=False)

    # Quick keyword bar chart
    top_keywords = keywords_df.head(20)
    plt.figure(figsize=(10,6))
    plt.barh(top_keywords["keyword"], top_keywords["count"])
    plt.title("Top Keywords in Trafficking Incidents")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("outputs/plots/top_keywords.png")
    plt.close()

    print("✔ Keyword frequency saved and plotted.\n")

    # ----------------------------------------------------
    # 3️⃣ SMUGGLING CODEWORDS (METHOD FIELDS)
    # ----------------------------------------------------
    print("🚨 Extracting smuggling method codewords...\n")

    if "Methods - Details" in incidents.columns:
        methods_text = incidents["Methods - Details"].fillna("").astype(str).str.lower()
    else:
        methods_text = ""

    method_tokens = []
    for txt in methods_text:
        method_tokens.extend(re.findall(r"[a-z]{4,}", txt))

    method_counts = Counter(method_tokens)
    method_df = pd.DataFrame(method_counts.most_common(150), columns=["keyword", "count"])

    method_df.to_csv("outputs/tables/method_codewords.csv", index=False)

    print("✔ Method codewords extracted.\n")

    print("🎉 NLP analysis completed successfully! All files saved.\n")

if __name__ == "__main__":
    run_nlp()
