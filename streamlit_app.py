# ===============================================================
# Wildlife Trafficking Intelligence Dashboard
# FINAL VERSION – with upgraded network & heatmaps
# ===============================================================

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network
from pathlib import Path

st.set_page_config(
    page_title="Wildlife Trafficking Dashboard",
    layout="wide",
)

# ======================================================
# PATHS & HELPERS
# ======================================================

BASE_DIR = Path(__file__).parent
TABLE_DIR = BASE_DIR / "outputs" / "tables"


@st.cache_data
def load_csv(name):
    path = TABLE_DIR / name
    if not path.exists():
        st.error(f"❌ File not found: {path}")
        return None
    return pd.read_csv(path)


@st.cache_data
def find_year_table():
    """Find any CSV in outputs/tables that has a 'year' column."""
    for f in TABLE_DIR.glob("*.csv"):
        df = pd.read_csv(f)
        if "year" in df.columns:
            return df, f.name
    return None, None


# ======================================================
# SIDEBAR NAVIGATION
# ======================================================

st.sidebar.title("Dashboard Navigation")
section = st.sidebar.radio(
    "Choose a view",
    [
        "Species Trafficking Overview",
        "Routes & Chokepoints (Network)",
        "Market Size (Monte Carlo)",
        "Country Risk (Regression)",
        "Scenario Analysis",
        "Raw Data Explorer",
    ],
)

# ======================================================
# 1. SPECIES TRAFFICKING OVERVIEW
# ======================================================

if section == "Species Trafficking Overview":
    st.title("Species Trafficking Overview")

    species_df = load_csv("species_frequency.csv")
    country_risk = load_csv("country_risk_scores.csv")
    routes = load_csv("route_edges.csv")

    # --------------------------------------------------
    # 1) Most-Trafficked Species
    # --------------------------------------------------
    st.subheader("1. Most-Trafficked Species")

    if species_df is not None:
        species_col = "species_name"
        count_col = "count"

        top_n = st.slider("Show top N species", 5, 30, 15)
        top_sp = species_df.sort_values(count_col, ascending=False).head(top_n)

        fig_sp = px.bar(
            top_sp,
            x=count_col,
            y=species_col,
            orientation="h",
            title="Top Trafficked Species",
        )
        fig_sp.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_sp, use_container_width=True)

        st.caption("Longer bars = species that appear more often in trafficking incidents.")
        st.dataframe(top_sp)

    # --------------------------------------------------
    # 2) Region-wise Trafficking – Country Heatmap
    # --------------------------------------------------
    st.subheader("2. Region-wise Trafficking (Country Heatmap)")

    if country_risk is not None:
        fig_map = px.choropleth(
            country_risk,
            locations="incident_country",
            locationmode="country names",
            color="risk_score",
            color_continuous_scale="Reds",
            title="Global Trafficking Risk Map",
            hover_name="incident_country",
        )
        fig_map.update_geos(
            showcountries=True,
            showcoastlines=True,
            projection_type="natural earth",
        )
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("country_risk_scores.csv not found.")

    # --------------------------------------------------
    # 3) Yearly Trends (uses any table with a 'year' column)
    # --------------------------------------------------
    st.subheader("3. Yearly Trends")

    yearly_df, yearly_name = find_year_table()
    if yearly_df is not None:
        st.caption(f"Using yearly data from: `{yearly_name}`")

        num_cols = [
            c
            for c in yearly_df.columns
            if yearly_df[c].dtype != "object" and c != "year"
        ]
        if num_cols:
            metric = num_cols[0]
            yearly = yearly_df.groupby("year")[metric].sum().reset_index()

            fig_year = px.line(
                yearly,
                x="year",
                y=metric,
                markers=True,
                title="Trafficking Activity Over Time",
            )
            fig_year.update_layout(
                xaxis_title="Year",
                yaxis_title="Activity (aggregated)",
            )
            st.plotly_chart(fig_year, use_container_width=True)
        else:
            st.info("Found a table with 'year', but no numeric metric to plot.")
    else:
        st.info("No dataset with a 'year' column found in outputs/tables.")

    # --------------------------------------------------
    # 4) Route Heatmap – Origin × Destination
    # --------------------------------------------------
    st.subheader("4. Route Heatmap (Origin → Destination)")

    if routes is not None:
        # drop unnamed index columns if present
        routes = routes.loc[:, ~routes.columns.str.contains("^Unnamed")]

        origin_col = "origin_country"
        dest_col = "destination_country"
        weight_col = "count"

        od = (
            routes.groupby([origin_col, dest_col], as_index=False)[weight_col]
            .sum()
            .rename(columns={weight_col: "FLOW"})
        )

        pivot = od.pivot(index=origin_col, columns=dest_col, values="FLOW").fillna(0)

        fig_heat = px.imshow(
            pivot,
            aspect="auto",
            color_continuous_scale="YlOrRd",
            labels=dict(color="Trafficking Flow"),
            title="Trafficking Route Heatmap (Origin → Destination)",
        )

        # annotate with flow values
        fig_heat.update_traces(
            text=pivot.values,
            texttemplate="%{text}",
            textfont={"size": 8},
        )
        fig_heat.update_layout(
            xaxis_title="Destination Country",
            yaxis_title="Origin Country",
            height=800,
            margin=dict(l=10, r=10, t=40, b=10),
        )

        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("route_edges.csv not found.")

# ======================================================
# 2. ROUTES & CHOKEPOINTS – INTERACTIVE NETWORK
# ======================================================

elif section == "Routes & Chokepoints (Network)":
    st.title("Trafficking Routes & Chokepoints — Interactive Network")

    routes = load_csv("route_edges.csv")
    choke = load_csv("chokepoints.csv")

    if routes is not None:
        routes = routes.loc[:, ~routes.columns.str.contains("^Unnamed")]

        origin_col = "origin_country"
        dest_col = "destination_country"
        weight_col = "count"

        # Build interactive PyVis network
        net = Network(
            height="750px",
            width="100%",
            bgcolor="#121212",
            font_color="white",
            notebook=False,
        )

        net.barnes_hut()  # physics layout

        for _, row in routes.iterrows():
            o = row[origin_col]
            d = row[dest_col]
            w = row[weight_col]

            net.add_node(o, label=o)
            net.add_node(d, label=d)
            net.add_edge(o, d, value=w, title=f"Flow: {w}")

        net.set_options(
            """
            var options = {
              "nodes": {
                "shape": "dot",
                "scaling": {
                  "min": 3,
                  "max": 20
                }
              },
              "edges": {
                "arrows": {"to": {"enabled": true}},
                "smooth": {
                  "enabled": true,
                  "type": "dynamic"
                }
              },
              "physics": {
                "stabilization": {
                  "enabled": true,
                  "iterations": 1000
                }
              }
            }
            """
        )

        net.save_graph("network.html")
        with open("network.html", "r", encoding="utf-8") as f:
            html = f.read()
        components.html(html, height=800, scrolling=True)
    else:
        st.info("route_edges.csv not found.")

    if choke is not None:
        st.subheader("Top Chokepoints")
        node_col = choke.columns[0]
        score_col = choke.columns[1]

        top_choke = choke.sort_values(score_col, ascending=False).head(20)

        fig_choke = px.bar(
            top_choke,
            x=score_col,
            y=node_col,
            orientation="h",
            title="Most Critical Chokepoints",
            color=score_col,
            color_continuous_scale="Cividis",
        )
        fig_choke.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_choke, use_container_width=True)

        st.dataframe(top_choke)

# ======================================================
# 3. MARKET SIZE – MONTE CARLO
# ======================================================

elif section == "Market Size (Monte Carlo)":
    st.title("Illegal Wildlife Market – Monte Carlo Simulation")

    samples = load_csv("monte_carlo_samples.csv")
    summary = load_csv("monte_carlo_summary.csv")

    if samples is not None:
        value_col = samples.columns[0]

        median_val = samples[value_col].median()
        p10 = samples[value_col].quantile(0.10)
        p90 = samples[value_col].quantile(0.90)

        c1, c2, c3 = st.columns(3)
        c1.metric("Typical Market Size", f"{median_val:.2f} B USD")
        c2.metric("Often Above", f"{p90:.2f} B USD")
        c3.metric("Often Below", f"{p10:.2f} B USD")

        fig_mc = px.histogram(
            samples,
            x=value_col,
            nbins=50,
            title="Distribution of Simulated Market Sizes",
        )
        fig_mc.update_layout(
            xaxis_title="Market Size (Billion USD)",
            yaxis_title="Number of simulations",
        )
        st.plotly_chart(fig_mc, use_container_width=True)

    if summary is not None:
        st.subheader("Simulation Summary")
        st.dataframe(summary)

# ======================================================
# 4. COUNTRY RISK – REGRESSION
# ======================================================

elif section == "Country Risk (Regression)":
    st.title("Country-Level Trafficking Risk – Regression Model")

    reg = load_csv("regression_results.csv")
    risk = load_csv("country_risk_scores.csv")

    if reg is not None:
        var_col = reg.columns[0]
        coef_col = reg.columns[1]

        fig_reg = px.bar(
            reg[reg[var_col] != "const"],
            x=coef_col,
            y=var_col,
            orientation="h",
            title="Effect of Risk Factors on Trafficking Incidents",
        )
        fig_reg.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_reg, use_container_width=True)

        st.dataframe(reg)

    if risk is not None:
        fig_risk = px.choropleth(
            risk,
            locations="incident_country",
            locationmode="country names",
            color="risk_score",
            color_continuous_scale="Reds",
            title="Country-Level Trafficking Risk",
        )
        st.plotly_chart(fig_risk, use_container_width=True)

        st.dataframe(risk)

# ======================================================
# 5. SCENARIO ANALYSIS
# ======================================================

elif section == "Scenario Analysis":
    st.title("Scenario Analysis – Impact of Interventions")

    scenario = load_csv("scenario_chokepoint_impact.csv")

    if scenario is not None:
        scen_col = scenario.columns[0]
        base_col = scenario.columns[1]
        after_col = scenario.columns[2]

        melted = scenario.melt(
            id_vars=[scen_col],
            value_vars=[base_col, after_col],
            var_name="State",
            value_name="Flow",
        )

        fig_scen = px.bar(
            melted,
            x=scen_col,
            y="Flow",
            color="State",
            barmode="group",
            title="Trafficking Flow – Baseline vs Intervention",
        )
        st.plotly_chart(fig_scen, use_container_width=True)

        st.dataframe(scenario)

# ======================================================
# 6. RAW DATA EXPLORER
# ======================================================

elif section == "Raw Data Explorer":
    st.title("Raw Data Explorer")

    files = [f.name for f in TABLE_DIR.glob("*.csv")]
    chosen = st.selectbox("Select a file", files)
    df = load_csv(chosen)

    if df is not None:
        st.dataframe(df)
