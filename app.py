# app.py
# Interactive GDP comparison (2020–2025), simplified:
# - Time Series: GDP, YoY%, Index(2020=100)
# - Scatter (2025): size vs growth
# No summary tiles, no bump chart, no "show in other views", no footnote.

'''
Time series chart: shows growth over time in either raw gdp, year over year growth (%), or relative to 2020 gdp
scatter plot: color represents quartile (speed of growth) over % 2025 growth over total gdp

Interactive elements
1. Users can select countries
2. For time series chart, users can display gdp/YoY/relative 2020GDP
3. Users can hover over points for details
4. Users can interact with graph (zoom/scale/etc)

Notes
CAGR is annual compound growth in % needed to go from first year gdp (2020) to final year gdp (2025)
YoY growth measures change in current from prev year %
Index 2020 shows relative growth from base year
Quartiles separates countries into groups based on how fast they grow in gdp (based on CAGR)
'''

from __future__ import annotations
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go


@dataclass
class GDPVisualizerConfig:
    csv_path: str = "Countries.csv"
    country_col_candidates: List[str] = None
    year_col_candidates: List[str] = None
    gdp_col_candidates: List[str] = None
    year_min: int = 2000
    year_max: int = 2022

    def __post_init__(self):
        if self.country_col_candidates is None:
            self.country_col_candidates = ["Country Name"]
        if self.year_col_candidates is None:
            self.year_col_candidates = ["Year", "year"]
        if self.gdp_col_candidates is None:
            self.gdp_col_candidates = ["GDP", "gdp", "Value", "value"]


class GDPApp:
    def __init__(self, cfg: GDPVisualizerConfig):
        self.cfg = cfg
        self.df_long = self._load_and_engineer()
        self.countries = sorted(self.df_long["Country Name"].unique())
        self.app = Dash(__name__)
        self._layout()
        self._callbacks()

    # --------------------------- Data prep --------------------------- #
    def _load_and_engineer(self) -> pd.DataFrame:
        df = pd.read_csv(self.cfg.csv_path)

        # Detect columns
        if "Country Name" in df.columns:
            country_col = "Country Name"
        else:
            country_col = next((c for c in self.cfg.country_col_candidates if c in df.columns), None)
        year_col = next((c for c in self.cfg.year_col_candidates if c in df.columns), None)
        gdp_col = next((c for c in self.cfg.gdp_col_candidates if c in df.columns), None)

        if country_col and year_col and gdp_col:
            df_long = df[[country_col, year_col, gdp_col]].rename(
                columns={country_col: "Country Name", year_col: "Year", gdp_col: "GDP"}
            )
        else:
            year_cols = []
            for col in df.columns:
                try:
                    y = int(str(col))
                    if self.cfg.year_min <= y <= self.cfg.year_max:
                        year_cols.append(col)
                except ValueError:
                    pass
            if not year_cols:
                raise ValueError("Expected 2000–2022 year columns or (Country, Year, GDP) long format.")

            if country_col is None:
                non_year = [c for c in df.columns if c not in year_cols]
                if not non_year:
                    raise ValueError("No country column detected.")
                country_col = next((c for c in self.cfg.country_col_candidates if c in non_year), non_year[0])

            df_long = df.melt(
                id_vars=[country_col],
                value_vars=year_cols,
                var_name="Year",
                value_name="GDP",
            ).rename(columns={country_col: "Country"})
            df_long["Year"] = df_long["Year"].astype(str).str.extract(r"(\d{4})").astype(int)

        # Clean & bound
        df_long["Country Name"] = df_long["Country Name"].astype(str)
        df_long["Year"] = pd.to_numeric(df_long["Year"], errors="coerce")
        df_long["GDP"] = pd.to_numeric(df_long["GDP"], errors="coerce")
        df_long = df_long.dropna(subset=["Country Name", "Year", "GDP"])
        df_long = df_long[(df_long["Year"] >= self.cfg.year_min) & (df_long["Year"] <= self.cfg.year_max)].copy()
        df_long.sort_values(["Country Name", "Year"], inplace=True)

        # YoY growth compared to previous year
        df_long["YoY_%"] = df_long.groupby("Country Name")["GDP"].pct_change() * 100.0

        # show relativity to base year 2020
        base = df_long.groupby("Country Name").apply(
            lambda g: g[g["Year"] == g["Year"].min()]["GDP"].iloc[0] if not g.empty else np.nan
        ).replace({0: np.nan})
        df_long["Index2000"] = df_long.apply(lambda r: 100.0 * r["GDP"] / base.get(r["Country Name"], np.nan), axis=1)

        # Country aggregates
        g_first = df_long.sort_values("Year").groupby("Country Name").first()["GDP"]
        g_last = df_long.sort_values("Year").groupby("Country Name").last()["GDP"]
        y_first = df_long.sort_values("Year").groupby("Country Name").first()["Year"]
        y_last = df_long.sort_values("Year").groupby("Country Name").last()["Year"]
        n_years = (y_last - y_first).replace(0, np.nan)

        cagr = (g_last / g_first).pow(1.0 / n_years) - 1.0
        cagr = cagr.replace([np.inf, -np.inf], np.nan)

        vol = df_long.groupby("Country Name")["YoY_%"].std(ddof=0)
        gdp_2022 = (df_long[df_long["Year"] == 2022].groupby("Country Name")["GDP"].mean())
       
        #print("CAGR index duplicates:", cagr.index.duplicated().sum())
        #print("Vol index duplicates:", vol.index.duplicated().sum())
        #print("GDP_2022 index duplicates:", gdp_2022.index.duplicated().sum())
        
        df_country = pd.DataFrame({"CAGR_20_22": cagr, "Volatility_YoY": vol, "GDP_20_22": gdp_2022})
        df_long = df_long.merge(df_country, how="left", left_on="Country Name", right_index=True)
        
        # CAGR quartiles (4 groups)
        df_long["CAGR_quartile"] = pd.qcut(
            df_long["CAGR_20_22"], q=4,
            labels=["Q1 (Slowest)", "Q2", "Q3", "Q4 (Fastest)"],
            duplicates="drop",
        )

        return df_long

    # ------------------------------ Layout ------------------------------ #
    def _layout(self):
        default_selection = [c for c in ["United States", "China", "India"] if c in self.countries][:2] or self.countries[:2]
        self.app.layout = html.Div(
            style={"fontFamily": "system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial", "padding": "16px"},
            children=[
                html.H2("GDP by Country (2000–2022)"),
                html.P("Explore levels, growth, and 2022 growth vs size. Hover for insights; click legend items to isolate countries."),
                html.Div(
                    style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "alignItems": "flex-end"},
                    children=[
                        html.Div(
                            style={"minWidth": "280px"},
                            children=[
                                html.Label("Countries"),
                                dcc.Dropdown(
                                    id="country-select",
                                    options=[{"label": c, "value": c} for c in self.countries],
                                    value=default_selection,
                                    multi=True,
                                    placeholder="Select countries…",
                                ),
                            ],
                        ),
                        html.Div(
                            style={"minWidth": "260px"},
                            children=[
                                html.Label("Series Metric"),
                                dcc.RadioItems(
                                    id="metric-mode",
                                    options=[
                                        {"label": "GDP Level", "value": "level"},
                                        {"label": "% YoY Growth", "value": "yoy"},
                                        {"label": "Index (2000=100)", "value": "index"},
                                    ],
                                    value="level",
                                    inline=True,
                                ),
                            ],
                        ),
                    ],
                ),
                dcc.Tabs(
                    id="tabs", value="tab-series", style={"marginTop": "12px"},
                    children=[
                        dcc.Tab(label="Time Series", value="tab-series"),
                        dcc.Tab(label="Scatter: 2022 Size vs Growth", value="tab-scatter"),
                    ],
                ),
                html.Div(id="tab-content", style={"marginTop": "12px"}),
            ],
        )

    # ------------------------------ Callbacks ------------------------------ #
    def _callbacks(self):
        @self.app.callback(
            Output("tab-content", "children"),
            Input("tabs", "value"),
            Input("country-select", "value"),
            Input("metric-mode", "value"),
        )
        def render_tab(tab, selected_countries, metric_mode):
            sel = selected_countries or []
            df = self.df_long.copy()

            # ----- TIME SERIES ----- #
            if tab == "tab-series":
                dff = df[df["Country Name"].isin(sel)] if sel else df.head(0)
                if dff.empty:
                    return html.Div("Select one or more countries to see the series.")

                # Always have GDP in trillions available for hover
                dff = dff.copy()
                dff["GDP_T"] = dff["GDP"] / 1e6  # millions -> trillions

                if metric_mode == "yoy":
                    y_col, title, y_title = "YoY_%", "% Year-over-Year GDP Growth", "% YoY"
                elif metric_mode == "index":
                    y_col, title, y_title = "Index2000", "Indexed GDP (Base Year = 100)", "Index (Base Year=100)"
                else:
                    y_col, title, y_title = "GDP_T", "GDP Level", "GDP (Trillions)"

                fig = px.line(
                    dff,
                    x="Year",
                    y=y_col,
                    color="Country Name",
                    markers=True,
                    title=title,
                    # Only the fields we want in the tooltip, in a known order:
                    custom_data=["Country Name", "GDP_T", "YoY_%", "Index2000", "CAGR_20_22"],
                    hover_data={},  # suppress extras
                )

                fig.update_traces(
                        hovertemplate=f"""
                    <b>%{{fullData.name}}</b><br>
                    Year: %{{x}}<br>
                    {y_title}: %{{y:.2f}}<br>
                    GDP (T): %{{customdata[0]:,.2f}}<br>
                    YoY%: %{{customdata[1]:.2f}}<br>
                    Index2000: %{{customdata[2]:.1f}}<br>
                    CAGR(’00→’22): %{{customdata[3]:.2%}}
                    <extra></extra>
                    """
                )

                fig.update_layout(
                    legend_title_text="Country",
                    hovermode="x unified",
                    yaxis_title=y_title,
                    xaxis_title="Year",
                )
                # If we're plotting GDP, show T suffix on the axis
                if metric_mode == "level":
                    fig.update_yaxes(tickformat=".0f", ticksuffix="T")

                return dcc.Graph(figure=fig, style={"height": "520px"})


            # ----- SCATTER (2022) ----- #
            if tab == "tab-scatter":
                d2022 = df[df["Year"] == 2022].copy()
                if not sel:
                    return html.Div("Select countries to populate this view.")
                dff = d2022[d2022["Country Name"].isin(sel)].copy()
                

                # Bubble size ~ GDP_2022 (normalized for display)
                dff["BubbleSize"] = dff["GDP_20_22"]
                smin, smax = dff["BubbleSize"].min(), dff["BubbleSize"].max()
                dff["BubbleSize"] = 20 + 60 * (dff["BubbleSize"] - smin) / (smax - smin) if pd.notna(smin) and pd.notna(smax) and smax > smin else 40

                # Scale GDP to trillions for axis + hover
                dff["GDP_2022_T"] = dff["GDP_20_22"] / 1e6  # millions -> trillions

                quartiles = ["Q1 (Slowest)", "Q2", "Q3", "Q4 (Fastest)"]
                color_map = {
                    "Q1 (Slowest)": "#F4A261",
                    "Q2": "#FFFF00",
                    "Q3": "#34D399",
                    "Q4 (Fastest)": "#8B5CF6",
                }

                fig = px.scatter(
                    dff,
                    x="GDP_2022_T", y="YoY_%",
                    size="BubbleSize",
                    color="CAGR_quartile",
                    category_orders={"CAGR_quartile": quartiles},
                    color_discrete_map=color_map,
                    title="2022: GDP Size vs % YoY Growth",
                    labels={"CAGR_quartile": "CAGR Quartile (’20→’22)"},
                    custom_data=["Country Name", "GDP_2022_T", "YoY_%", "CAGR_20_22", "CAGR_quartile"],
                    hover_data={},  # clean tooltip
                )

                fig.update_traces(
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "2022 GDP: %{customdata[1]:.2f}T<br>"
                        "2022 YoY: %{customdata[2]:.2f}%<br>"
                        "CAGR(’00→’22): %{customdata[3]:.2%}<br>"
                        "Quartile: %{customdata[4]}"
                        "<extra></extra>"
                    )
                )

                fig.update_layout(
                    xaxis_title="2022 GDP (Trillions)",
                    yaxis_title="% YoY in 2022",
                    legend_title_text="CAGR Quartile",
                )
                fig.update_xaxes(tickformat=".0f", ticksuffix="T")

                # Keep full legend even if some quartiles are absent
                present = {getattr(tr, "legendgroup", tr.name) for tr in fig.data}
                for q in quartiles:
                    if q not in present:
                        fig.add_trace(
                            go.Scatter(
                                x=[None], y=[None],
                                mode="markers",
                                marker=dict(size=0, color=color_map[q]),
                                name=q, legendgroup=q, showlegend=True, hoverinfo="skip",
                            )
                        )

                return dcc.Graph(figure=fig, style={"height": "520px"})


    def run(self, host="127.0.0.1", port=8050, debug=True):
        runner = getattr(self.app, "run", None) or getattr(self.app, "run_server")
        return runner(host=host, port=port, debug=debug)


if __name__ == "__main__":
    cfg = GDPVisualizerConfig(csv_path="Countries.csv")
    GDPApp(cfg).run()
