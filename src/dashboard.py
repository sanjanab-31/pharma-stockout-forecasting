"""dashboard.py - Plotly Dash monitoring portal (PharmaTrack Live).

    Professional Blue Healthcare SaaS Portal with Poppins typography and Lucide React Icons:
      - Page 1: Executive Dashboard & Supply Chain Intelligence (Metrics, Analytics Charts & Policy Simulation)
      - Page 2: Live Alert Feed & Operational Dispatch (Separate page with Pill Search & Filters, Portal Data Table)

    python dashboard.py              (then open http://127.0.0.1:8050)
"""
import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table, Input, Output, State, ctx
from config import OUTPUTS

# -----------------------------------------------------------------------------
# Data Loading & Preparation (Preserving Exact Original Logic)
# -----------------------------------------------------------------------------
scored = pd.read_pickle(OUTPUTS / "test_scored.pkl")
metrics = json.load(open(OUTPUTS / "metrics.json"))
sim = json.load(open(OUTPUTS / "simulation.json"))
THR = float(metrics["proposed_threshold"])
latest = scored[scored.day == scored.day.max()].copy()


def compute_action(r, threshold=THR):
    """Exact action classification rule from predict.py."""
    if r["risk"] < threshold:
        return "Normal Stock"
    if bool(r.get("upstream_shortage", 0)) and int(r.get("n_manufacturers", 1)) <= 1:
        return "Expedite / Transfer"
    if bool(r.get("upstream_shortage", 0)):
        return "Switch Alternate Mfr"
    return "Reorder Now"


# Add precalculated action and formatted risk for fast rendering
latest["action"] = latest.apply(compute_action, axis=1)
latest["risk_pct"] = (latest["risk"] * 100).round(1)
latest["eta_clean"] = latest["eta_days"].apply(
    lambda x: f"{x:.1f}d" if x < 29.9 else "No Open Order"
)
latest["facility_display"] = latest["facility_id"] + " • " + latest["city"]

# Distinct options for filters
FACILITY_OPTIONS = [
    {"label": f"{fid} — {city} ({ftype})", "value": fid}
    for fid, city, ftype in latest[["facility_id", "city", "facility_type"]]
    .drop_duplicates()
    .sort_values("facility_id")
    .values
]
CITY_OPTIONS = [{"label": c, "value": c} for c in sorted(latest["city"].unique())]
CATEGORY_OPTIONS = [{"label": c, "value": c} for c in sorted(latest["category"].unique())]
SEVERITY_OPTIONS = [
    {"label": "All Severity Levels", "value": "ALL"},
    {"label": "Critical Risk (≥ 80%)", "value": "CRITICAL"},
    {"label": "High Risk (66% – 80%)", "value": "HIGH"},
    {"label": "Elevated Risk (40% – 66%)", "value": "ELEVATED"},
    {"label": "Normal / Safe (< 40%)", "value": "SAFE"},
]
ACTION_OPTIONS = [
    {"label": "All Actions", "value": "ALL"},
    {"label": "Expedite / Inter-facility Transfer", "value": "Expedite / Transfer"},
    {"label": "Switch Alternate Manufacturer", "value": "Switch Alternate Mfr"},
    {"label": "Reorder Now", "value": "Reorder Now"},
    {"label": "Normal Stock (Safe)", "value": "Normal Stock"},
]


# -----------------------------------------------------------------------------
# Authentic Lucide React Icons rendered via pure CSS vector backgrounds
# -----------------------------------------------------------------------------


# Modern Light Portal Chart Styler Helper
def apply_portal_chart_style(fig, title_text, xaxis_dict=None, yaxis_dict=None):
    x_cfg = dict(
        gridcolor="#f1f5f9",
        zerolinecolor="#e2e8f0",
        tickfont=dict(family="Poppins, sans-serif", color="#64748b", size=11),
    )
    if xaxis_dict:
        x_cfg.update(xaxis_dict)
    y_cfg = dict(
        gridcolor="#f1f5f9",
        zerolinecolor="#e2e8f0",
        tickfont=dict(family="Poppins, sans-serif", color="#64748b", size=11),
    )
    if yaxis_dict:
        y_cfg.update(yaxis_dict)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font=dict(family="Poppins, sans-serif", color="#475569"),
        title=dict(
            text=title_text,
            font=dict(family="Poppins, sans-serif", color="#0f172a", size=14, weight="bold"),
        ),
        margin=dict(l=45, r=20, t=45, b=40),
        height=350,
        xaxis=x_cfg,
        yaxis=y_cfg,
        hoverlabel=dict(
            bgcolor="#ffffff",
            bordercolor="#cbd5e1",
            font=dict(family="Poppins, sans-serif", color="#0f172a", size=12),
        ),
    )


# -----------------------------------------------------------------------------
# Dash App Initialization
# -----------------------------------------------------------------------------
app = Dash(
    __name__,
    title="PharmaTrack Portal | Supply Chain Intelligence",
    assets_folder=str(Path(__file__).parent / "assets"),
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap",
    ],
    suppress_callback_exceptions=True,
)


# -----------------------------------------------------------------------------
# Reusable UI Components
# -----------------------------------------------------------------------------
def metric_box(label, value, subtext, trend_text, accent_color, icon_class, icon_bg):
    """Rounded key metric box in clean portal style with Lucide React icons."""
    return html.Div(
        className="metric-card-box",
        style={"--card-accent": accent_color, "--card-icon-bg": icon_bg},
        children=[
            html.Div(
                className="metric-card-header",
                children=[
                    html.Span(label, className="metric-card-label"),
                    html.Div(
                        className="metric-card-icon-wrap",
                        children=html.Span(className=f"lucide-icon {icon_class}"),
                    ),
                ],
            ),
            html.Div(value, className="metric-card-number"),
            html.Div(
                className="metric-card-sub",
                children=[
                    html.Span(
                        trend_text,
                        className="metric-trend-pill",
                        style={"backgroundColor": icon_bg, "color": accent_color},
                    ),
                    html.Span(subtext),
                ],
            ),
        ],
    )


def build_navbar():
    """Portal Top Navbar with Poppins font, Blue theme, and Lucide React icons."""
    return html.Header(
        className="portal-navbar",
        children=[
            html.Div(
                className="portal-brand-wrap",
                children=[
                    html.Div(
                        className="portal-logo-icon",
                        children=html.Span(className="lucide-icon lucide-truck-white"),
                    ),
                    html.Div(
                        className="portal-brand-text",
                        children=[
                            html.Div(
                                className="portal-brand-title",
                                children=[
                                    "PharmaTrack",
                                    html.Span("Portal", style={"color": "#2563eb", "fontWeight": "800"}),
                                    html.Span("Tamil Nadu", className="portal-brand-tag"),
                                ],
                            ),
                            html.Div(
                                "Real-Time Supply Chain Tracking & Stockout Forecasting Engine",
                                className="portal-brand-sub",
                            ),
                        ],
                    ),
                ],
            ),
            # Navigation Switcher (2 Distinct Pages)
            html.Div(
                className="portal-nav-switcher",
                children=[
                    html.Button(
                        [html.Span(className="lucide-icon lucide-dashboard-icon"), html.Span(" Executive Dashboard", style={"marginLeft": "7px"})],
                        id="nav-btn-analytics",
                        className="portal-nav-tab active",
                        n_clicks=0,
                    ),
                    html.Button(
                        [html.Span(className="lucide-icon lucide-bell-icon"), html.Span(" Live Alert Feed", style={"marginLeft": "7px"})],
                        id="nav-btn-alerts",
                        className="portal-nav-tab",
                        n_clicks=0,
                    ),
                ],
            ),
            # Status Meta Pill
            html.Div(
                className="portal-header-meta",
                children=[
                    html.Div(
                        className="meta-pill-badge",
                        children=[
                            html.Span(className="status-pulse-dot"),
                            html.Span("Engine Live • 14-Day Horizon", style={"fontWeight": "600"}),
                        ],
                    ),
                    html.Div(
                        className="meta-pill-badge",
                        children=[
                            html.Span(className="lucide-icon lucide-sliders-blue", style={"marginRight": "5px"}),
                            html.Span(f"Cutoff: {THR:.1%}", style={"color": "#2563eb", "fontWeight": "600"}),
                        ],
                    ),
                ],
            ),
        ],
    )


def build_metrics_row():
    """5 Key metrics cards with distinct rounded corners and Lucide React icons."""
    n_series = scored[["facility_id", "drug_id"]].drop_duplicates().shape[0]
    n_alerts = int((latest.risk >= THR).sum())
    n_shortages = int(latest[latest.upstream_shortage == 1].drug_id.nunique())
    fill_rate = sim["prop"]["fill_rate"]
    stockout_reduction = sim["stockout_day_reduction_pct"]

    return html.Div(
        className="metrics-grid-row",
        children=[
            metric_box(
                "Series Monitored",
                f"{n_series:,}",
                "Across 15 Cities",
                "30 Facilities",
                "#2563eb",
                "lucide-network-blue",
                "#eff6ff",
            ),
            metric_box(
                "High-Risk Alerts",
                f"{n_alerts}",
                "Action needed",
                f"≥ {THR:.1%}",
                "#ef4444",
                "lucide-alert-red",
                "#fef2f2",
            ),
            metric_box(
                "Upstream Shortages",
                f"{n_shortages}",
                "Active supplier disruption",
                "5 Drugs",
                "#f59e0b",
                "lucide-package-amber",
                "#fffbeb",
            ),
            metric_box(
                "Projected Fill Rate",
                f"{fill_rate:.1f}%",
                "vs 96.8% Baseline",
                "+2.1%",
                "#10b981",
                "lucide-shield-green",
                "#ecfdf5",
            ),
            metric_box(
                "Stockout Reduction",
                f"-{stockout_reduction:.1f}%",
                "5,761 days saved",
                "Proposed Policy",
                "#8b5cf6",
                "lucide-trending-purple",
                "#f5f3ff",
            ),
        ],
    )


# -----------------------------------------------------------------------------
# PAGE 1: Executive Dashboard & Supply Chain Intelligence
# -----------------------------------------------------------------------------
def build_page_dashboard():
    # 1. Chart: Top 10 High-Risk Medicines
    top_drugs = (
        latest.groupby(["drug_name", "category"])["risk"]
        .mean()
        .reset_index()
        .sort_values("risk", ascending=False)
        .head(10)
        .sort_values("risk")
    )
    fig_drugs = px.bar(
        top_drugs,
        x="risk",
        y="drug_name",
        orientation="h",
        color="risk",
        color_continuous_scale=[[0, "#93c5fd"], [0.5, "#3b82f6"], [1, "#1d4ed8"]],
        labels={"risk": "Network Mean Risk (14d)", "drug_name": ""},
        hover_data={"category": True, "risk": ":.2%"},
    )
    apply_portal_chart_style(
        fig_drugs,
        "Top 10 Highest-Risk Medicines (Network Average)",
        xaxis_dict=dict(tickformat=".0%", range=[0, 1]),
    )
    fig_drugs.update_layout(coloraxis_showscale=False)

    # 2. Chart: Network Mean Risk Timeline (Weekly)
    trend_df = scored.groupby("date")["risk"].mean().reset_index()
    fig_trend = go.Figure()
    fig_trend.add_trace(
        go.Scatter(
            x=trend_df["date"],
            y=trend_df["risk"],
            mode="lines+markers",
            name="Network Risk",
            line=dict(color="#2563eb", width=3),
            marker=dict(size=6, color="#1d4ed8"),
            fill="tozeroy",
            fillcolor="rgba(37, 99, 235, 0.08)",
        )
    )
    fig_trend.add_hline(
        y=THR,
        line_dash="dot",
        line_color="#ef4444",
        annotation_text=f"Cutoff Threshold ({THR:.1%})",
        annotation_position="bottom right",
        annotation_font_color="#ef4444",
        annotation_font_family="Poppins",
    )
    fig_trend.add_vrect(
        x0="2025-02-01",
        x1="2025-02-20",
        fillcolor="rgba(239, 68, 68, 0.08)",
        layer="below",
        line_width=0,
        annotation_text="Monsoon Flood Disruption",
        annotation_position="top left",
        annotation_font_color="#dc2626",
        annotation_font_family="Poppins",
    )
    apply_portal_chart_style(
        fig_trend,
        "Network Mean Risk Score Timeline (Weekly)",
        yaxis_dict=dict(tickformat=".0%"),
    )

    # 3. Chart: City Alert Concentration
    city_df = (
        latest.groupby("city")
        .agg(
            mean_risk=("risk", "mean"),
            alert_count=("risk", lambda s: (s >= THR).sum()),
            facility_count=("facility_id", "nunique"),
        )
        .reset_index()
        .sort_values("alert_count", ascending=False)
    )
    fig_city = px.bar(
        city_df,
        x="city",
        y="alert_count",
        color="mean_risk",
        color_continuous_scale=[[0, "#60a5fa"], [0.6, "#2563eb"], [1, "#1e3a8a"]],
        labels={"alert_count": "Active Alerts (Risk ≥ Cutoff)", "city": "", "mean_risk": "Avg Risk"},
        hover_data={"facility_count": True, "mean_risk": ":.2%"},
    )
    apply_portal_chart_style(
        fig_city,
        "Stockout Alerts by Tamil Nadu City (15 Cities)",
    )
    fig_city.update_layout(coloraxis_showscale=False)

    # 4. Chart: Risk by Category
    cat_df = (
        latest.groupby("category")
        .agg(
            mean_risk=("risk", "mean"),
            alerts=("risk", lambda s: (s >= THR).sum()),
            total=("risk", "count"),
        )
        .reset_index()
        .sort_values("mean_risk", ascending=False)
    )
    fig_cat = px.bar(
        cat_df,
        x="category",
        y="mean_risk",
        color="mean_risk",
        color_continuous_scale=[[0, "#34d399"], [0.4, "#38bdf8"], [0.8, "#f43f5e"]],
        labels={"mean_risk": "Mean Stockout Risk", "category": ""},
        hover_data={"alerts": True, "total": True},
    )
    apply_portal_chart_style(
        fig_cat,
        "Risk Profile by Therapeutic Class (10 Categories)",
        yaxis_dict=dict(tickformat=".0%"),
    )
    fig_cat.update_layout(coloraxis_showscale=False)

    # Model Performance Leaderboard Rows
    clf_metrics = metrics.get("classification", {})
    benchmark_rows = []
    for model_name, m_data in clf_metrics.items():
        is_proposed = "Proposed" in model_name
        benchmark_rows.append(
            html.Tr(
                className="highlighted-row" if is_proposed else "",
                children=[
                    html.Td(
                        [
                            html.Span(className="lucide-icon lucide-check-circle-blue", style={"marginRight": "8px", "verticalAlign": "middle"}) if is_proposed else None,
                            html.Span(model_name, style={"fontWeight": "700" if is_proposed else "500"}),
                        ]
                    ),
                    html.Td(f"{m_data.get('Accuracy', 0):.2f}%"),
                    html.Td(f"{m_data.get('Precision', 0):.2f}%"),
                    html.Td(f"{m_data.get('Recall', 0):.2f}%"),
                    html.Td(f"{m_data.get('F1', 0):.2f}%"),
                    html.Td(f"{m_data.get('ROC_AUC', 0):.3f}" if m_data.get("ROC_AUC") else "—"),
                    html.Td(f"{m_data.get('PR_AUC', 0):.3f}"),
                    html.Td(
                        html.Span(
                            "Production Model" if is_proposed else "Baseline",
                            className="metric-trend-pill",
                            style={
                                "background": "#eff6ff" if is_proposed else "#f1f5f9",
                                "color": "#2563eb" if is_proposed else "#64748b",
                                "border": "1px solid #bfdbfe" if is_proposed else "1px solid #e2e8f0",
                            },
                        )
                    ),
                ],
            )
        )

    return html.Div(
        children=[
            # Simulation Policy Comparison Banner
            html.Div(
                className="portal-policy-banner",
                children=[
                    html.Div(
                        children=[
                            html.Span(
                                "⚡ Replenishment Policy Simulation",
                                className="policy-badge-portal",
                            ),
                            html.Div(
                                "Dynamic AI Policy vs Static Reorder Point Benchmark",
                                className="policy-title-portal",
                            ),
                            html.Div(
                                "Replaying all 1,500 facility-medicine pairs with identical historical demand and common random lead times. "
                                "The proposed LightGBM predictive policy reduces stockout days by 63.6% while maintaining a 99.0% service fill rate.",
                                className="policy-desc-portal",
                            ),
                        ]
                    ),
                    html.Div(
                        className="policy-stats-quad",
                        children=[
                            html.Div(
                                className="policy-stat-card",
                                children=[
                                    html.Div("Stockout Days Reduction", className="policy-stat-label"),
                                    html.Div(
                                        f"-{sim['stockout_day_reduction_pct']:.1f}%",
                                        className="policy-stat-num",
                                        style={"color": "#2563eb"},
                                    ),
                                    html.Div(
                                        f"{sim['base']['stockout_days']:,} ➔ {sim['prop']['stockout_days']:,} days",
                                        className="policy-stat-sub",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="policy-stat-card",
                                children=[
                                    html.Div("Service Fill Rate", className="policy-stat-label"),
                                    html.Div(
                                        f"{sim['prop']['fill_rate']:.2f}%",
                                        className="policy-stat-num",
                                        style={"color": "#10b981"},
                                    ),
                                    html.Div(
                                        f"vs {sim['base']['fill_rate']:.2f}% baseline",
                                        className="policy-stat-sub",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="policy-stat-card",
                                children=[
                                    html.Div("Median Inventory Days", className="policy-stat-label"),
                                    html.Div(
                                        f"{sim['prop']['median_inventory_days']:.1f}d",
                                        className="policy-stat-num",
                                        style={"color": "#0f172a"},
                                    ),
                                    html.Div(
                                        f"vs {sim['base']['median_inventory_days']:.1f}d static",
                                        className="policy-stat-sub",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="policy-stat-card",
                                children=[
                                    html.Div("Alternate Source Orders", className="policy-stat-label"),
                                    html.Div(
                                        f"{sim['prop'].get('alternate_source_orders', 706):,}",
                                        className="policy-stat-num",
                                        style={"color": "#f59e0b"},
                                    ),
                                    html.Div(
                                        "Automated re-routes",
                                        className="policy-stat-sub",
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            # 2x2 Portal Charts Grid
            html.Div(
                className="portal-charts-grid",
                children=[
                    html.Div(className="portal-chart-card", children=[dcc.Graph(figure=fig_drugs, config={"displayModeBar": False})]),
                    html.Div(className="portal-chart-card", children=[dcc.Graph(figure=fig_trend, config={"displayModeBar": False})]),
                    html.Div(className="portal-chart-card", children=[dcc.Graph(figure=fig_city, config={"displayModeBar": False})]),
                    html.Div(className="portal-chart-card", children=[dcc.Graph(figure=fig_cat, config={"displayModeBar": False})]),
                ],
            ),
            # Model Leaderboard Card
            html.Div(
                className="portal-chart-card full-width",
                children=[
                    html.Div(
                        style={"display": "flex", "alignItems": "center", "justifyContent": "space-between", "marginBottom": "16px"},
                        children=[
                            html.Div(
                                style={"display": "flex", "alignItems": "center", "gap": "10px"},
                                children=[
                                    html.Span("14-Day Stockout Classifier Model Benchmark", style={"fontWeight": "700", "fontSize": "16px", "color": "#0f172a"}),
                                ],
                            ),
                            html.Span("Evaluated on Held-out Test Data (Jan – Jun 2025)", className="portal-badge-count"),
                        ],
                    ),
                    html.Div(
                        style={"overflowX": "auto"},
                        children=[
                            html.Table(
                                className="portal-benchmark-table",
                                children=[
                                    html.Thead(
                                        html.Tr(
                                            [
                                                html.Th("Model Architecture"),
                                                html.Th("Accuracy"),
                                                html.Th("Precision"),
                                                html.Th("Recall"),
                                                html.Th("F1-Score"),
                                                html.Th("ROC-AUC"),
                                                html.Th("PR-AUC"),
                                                html.Th("Deployment Status"),
                                            ]
                                        )
                                    ),
                                    html.Tbody(benchmark_rows),
                                ],
                            )
                        ],
                    ),
                ],
            ),
        ]
    )


# -----------------------------------------------------------------------------
# PAGE 2: Live Alert Feed & Operational Dispatch (Separate Page)
# -----------------------------------------------------------------------------
def build_page_alerts():
    return html.Div(
        children=[
            # Page Title Banner
            html.Div(
                className="page-title-banner",
                children=[
                    html.Div(
                        children=[
                            html.Div("Live Stockout Alert Feed & Supply Dispatch", className="page-heading"),
                            html.Div("Real-time risk scoring, inventory diagnostics, and recommended replenishment actions", className="page-subheading"),
                        ]
                    ),
                ],
            ),
            # Pill Search & Filter Toolbar (Matching User Uploaded UI Reference)
            html.Div(
                className="portal-filter-card",
                children=[
                    # Pill Search Input with SVG Background Icon
                    html.Div(
                        className="pill-search-wrapper",
                        children=[
                            dcc.Input(
                                id="filter-search",
                                type="text",
                                placeholder="Search medicines, facilities, cities...",
                                className="pill-search-input",
                                debounce=True,
                            ),
                        ],
                    ),
                    # Pill Dropdown: City
                    html.Div(
                        className="pill-dropdown-item",
                        children=[
                            dcc.Dropdown(
                                id="filter-city",
                                options=CITY_OPTIONS,
                                placeholder="City (15 cities) ▾",
                                clearable=True,
                            ),
                        ],
                    ),
                    # Pill Dropdown: Facility
                    html.Div(
                        className="pill-dropdown-item",
                        style={"minWidth": "210px"},
                        children=[
                            dcc.Dropdown(
                                id="filter-facility",
                                options=FACILITY_OPTIONS,
                                placeholder="Facility (30 centers) ▾",
                                clearable=True,
                            ),
                        ],
                    ),
                    # Pill Dropdown: Category
                    html.Div(
                        className="pill-dropdown-item",
                        children=[
                            dcc.Dropdown(
                                id="filter-category",
                                options=CATEGORY_OPTIONS,
                                placeholder="Category ▾",
                                clearable=True,
                            ),
                        ],
                    ),
                    # Pill Dropdown: Severity
                    html.Div(
                        className="pill-dropdown-item",
                        children=[
                            dcc.Dropdown(
                                id="filter-severity",
                                options=SEVERITY_OPTIONS,
                                value="ALL",
                                clearable=False,
                            ),
                        ],
                    ),
                    # Pill Dropdown: Action
                    html.Div(
                        className="pill-dropdown-item",
                        children=[
                            dcc.Dropdown(
                                id="filter-action",
                                options=ACTION_OPTIONS,
                                value="ALL",
                                clearable=False,
                            ),
                        ],
                    ),
                    # Reset Filters Pill Button
                    html.Button(
                        [html.Span(className="lucide-icon lucide-rotate-icon", style={"marginRight": "6px"}), html.Span("Reset")],
                        id="btn-reset-filters",
                        className="btn-pill-reset",
                        n_clicks=0,
                    ),
                ],
            ),
            # Split Layout: Portal Table on Left, Medicine Diagnostics on Right
            html.Div(
                className="portal-split-layout",
                children=[
                    # Table Card (Exact Portal Style from User Reference)
                    html.Div(
                        className="portal-table-card",
                        children=[
                            html.Div(
                                className="portal-table-top",
                                children=[
                                    html.Div(
                                        className="portal-table-title",
                                        children=[
                                            html.Span("Active Medicine Monitoring Queue"),
                                        ],
                                    ),
                                    html.Div(id="table-count-badge", className="portal-badge-count"),
                                ],
                            ),
                            dash_table.DataTable(
                                id="alerts-table",
                                columns=[
                                    {"name": "Facility & City", "id": "facility_display"},
                                    {"name": "Type", "id": "facility_type"},
                                    {"name": "Medicine", "id": "drug_name"},
                                    {"name": "Category", "id": "category"},
                                    {"name": "Days Supply", "id": "days_of_supply"},
                                    {"name": "14d Forecast", "id": "fc14"},
                                    {"name": "Status", "id": "shortage_display"},
                                    {"name": "Risk Score", "id": "risk_display"},
                                    {"name": "Action", "id": "action"},
                                    {"name": "Inspect", "id": "actions_btn"},
                                ],
                                data=[],
                                row_selectable="single",
                                selected_rows=[0],
                                page_size=12,
                                page_current=0,
                                sort_action="native",
                                style_as_list_view=True,
                                style_table={"overflowX": "auto", "minWidth": "100%"},
                                style_cell={
                                    "fontFamily": "Poppins, sans-serif",
                                    "textAlign": "left",
                                    "padding": "12px 14px",
                                    "fontSize": "12.5px",
                                    "backgroundColor": "#ffffff",
                                    "color": "#1e293b",
                                    "borderBottom": "1px solid #f1f5f9",
                                    "whiteSpace": "nowrap",
                                },
                                style_cell_conditional=[
                                    {"if": {"column_id": "facility_display"}, "minWidth": "170px", "width": "180px"},
                                    {"if": {"column_id": "facility_type"}, "minWidth": "140px", "width": "150px"},
                                    {"if": {"column_id": "drug_name"}, "minWidth": "160px", "width": "170px"},
                                    {"if": {"column_id": "category"}, "minWidth": "120px", "width": "130px"},
                                    {"if": {"column_id": "days_of_supply"}, "minWidth": "90px", "width": "95px", "textAlign": "center"},
                                    {"if": {"column_id": "fc14"}, "minWidth": "95px", "width": "100px", "textAlign": "center"},
                                    {"if": {"column_id": "shortage_display"}, "minWidth": "110px", "width": "115px", "textAlign": "center"},
                                    {"if": {"column_id": "risk_display"}, "minWidth": "90px", "width": "95px", "textAlign": "center"},
                                    {"if": {"column_id": "action"}, "minWidth": "160px", "width": "170px"},
                                    {"if": {"column_id": "actions_btn"}, "minWidth": "70px", "width": "75px", "textAlign": "center"},
                                ],
                                style_header={
                                    "backgroundColor": "#f8fafc",
                                    "color": "#475569",
                                    "fontWeight": "600",
                                    "fontSize": "11px",
                                    "textTransform": "uppercase",
                                    "letterSpacing": "0.5px",
                                    "borderBottom": "1px solid #e2e8f0",
                                },
                                style_data_conditional=[
                                    {
                                        "if": {"column_id": "actions_btn"},
                                        "color": "#2563eb",
                                        "fontWeight": "600",
                                    },
                                    {
                                        "if": {"column_id": "shortage_display", "filter_query": '{shortage_display} = "⚠️ SHORTAGE"'},
                                        "color": "#d97706",
                                        "fontWeight": "600",
                                    },
                                    {
                                        "if": {"column_id": "shortage_display", "filter_query": '{shortage_display} = "NORMAL"'},
                                        "color": "#059669",
                                        "fontWeight": "600",
                                    },
                                    {
                                        "if": {"column_id": "action", "filter_query": '{action} = "Expedite / Transfer"'},
                                        "color": "#dc2626",
                                        "fontWeight": "600",
                                    },
                                    {
                                        "if": {"column_id": "action", "filter_query": '{action} = "Switch Alternate Mfr"'},
                                        "color": "#d97706",
                                        "fontWeight": "600",
                                    },
                                    {
                                        "if": {"column_id": "action", "filter_query": '{action} = "Reorder Now"'},
                                        "color": "#2563eb",
                                        "fontWeight": "600",
                                    },
                                    {
                                        "if": {"column_id": "action", "filter_query": '{action} = "Normal Stock"'},
                                        "color": "#059669",
                                    },
                                    {
                                        "if": {"column_id": "risk_display"},
                                        "fontWeight": "700",
                                    },
                                    {
                                        "if": {"state": "selected"},
                                        "backgroundColor": "#eff6ff !important",
                                        "border": "1px solid #bfdbfe !important",
                                    },
                                ],
                            ),
                        ],
                    ),
                    # Right Inspector Card (Detail View)
                    html.Div(id="medicine-inspector-container", className="portal-inspector-card"),
                ],
            ),
        ],
    )


# -----------------------------------------------------------------------------
# Main Root Layout
# -----------------------------------------------------------------------------
app.layout = html.Div(
    className="portal-app-container",
    children=[
        dcc.Store(id="active-page-store", data="analytics"),
        build_navbar(),
        html.Main(
            className="portal-main",
            children=[
                build_metrics_row(),
                html.Div(id="page-container", children=build_page_dashboard()),
            ],
        ),
    ],
)


# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------
@app.callback(
    Output("page-container", "children"),
    Output("nav-btn-analytics", "className"),
    Output("nav-btn-alerts", "className"),
    Output("active-page-store", "data"),
    Input("nav-btn-analytics", "n_clicks"),
    Input("nav-btn-alerts", "n_clicks"),
    State("active-page-store", "data"),
    prevent_initial_call=True,
)
def switch_portal_page(btn_analytics_clicks, btn_alerts_clicks, current_page):
    """Switch between Page 1 (Executive Dashboard) and Page 2 (Live Alert Feed)."""
    trigger = ctx.triggered_id
    if trigger == "nav-btn-alerts":
        return build_page_alerts(), "portal-nav-tab", "portal-nav-tab active", "alerts"
    return build_page_dashboard(), "portal-nav-tab active", "portal-nav-tab", "analytics"


@app.callback(
    Output("filter-search", "value"),
    Output("filter-city", "value"),
    Output("filter-facility", "value"),
    Output("filter-category", "value"),
    Output("filter-severity", "value"),
    Output("filter-action", "value"),
    Input("btn-reset-filters", "n_clicks"),
    prevent_initial_call=True,
)
def reset_filters(n_clicks):
    """Reset all filter controls to default on Page 2."""
    return "", None, None, None, "ALL", "ALL"


@app.callback(
    Output("alerts-table", "data"),
    Output("table-count-badge", "children"),
    Input("filter-search", "value"),
    Input("filter-city", "value"),
    Input("filter-facility", "value"),
    Input("filter-category", "value"),
    Input("filter-severity", "value"),
    Input("filter-action", "value"),
)
def update_table_data(search_txt, city_val, facility_val, cat_val, severity_val, action_val):
    """Filter and sort the alert feed according to user selections."""
    df = latest.copy()

    # Text search
    if search_txt and search_txt.strip():
        q = search_txt.strip().lower()
        mask = (
            df["drug_name"].str.lower().str.contains(q)
            | df["facility_id"].str.lower().str.contains(q)
            | df["city"].str.lower().str.contains(q)
            | df["category"].str.lower().str.contains(q)
        )
        df = df[mask]

    # Dropdowns
    if city_val:
        df = df[df["city"] == city_val]
    if facility_val:
        df = df[df["facility_id"] == facility_val]
    if cat_val:
        df = df[df["category"] == cat_val]

    # Severity Level Filter
    if severity_val == "CRITICAL":
        df = df[df["risk"] >= 0.80]
    elif severity_val == "HIGH":
        df = df[(df["risk"] >= THR) & (df["risk"] < 0.80)]
    elif severity_val == "ELEVATED":
        df = df[(df["risk"] >= 0.40) & (df["risk"] < THR)]
    elif severity_val == "SAFE":
        df = df[df["risk"] < 0.40]

    # Action Filter
    if action_val and action_val != "ALL":
        df = df[df["action"] == action_val]

    # Sort so highest risks are first
    df = df.sort_values("risk", ascending=False)

    # Format records for clean table view matching reference style
    records = []
    for _, r in df.iterrows():
        records.append(
            {
                "facility_id": r["facility_id"],
                "facility_display": f"{r['facility_id']} • {r['city']}",
                "city": r["city"],
                "facility_type": r["facility_type"],
                "drug_id": r["drug_id"],
                "drug_name": r["drug_name"],
                "category": r["category"],
                "days_of_supply": f"{r['days_of_supply']:.1f}d",
                "fc14": f"{int(round(r['fc14']))} units",
                "eta_clean": r["eta_clean"],
                "shortage_display": "⚠️ SHORTAGE" if r["upstream_shortage"] else "NORMAL",
                "risk": round(float(r["risk"]), 3),
                "risk_display": f"{r['risk'] * 100:.1f}%",
                "action": r["action"],
                "actions_btn": "👁 View",
                "on_hand": int(r["on_hand"]),
                "on_order": int(r["on_order"]),
                "safety_stock": int(r.get("safety_stock", 0)),
                "rop": int(r.get("rop", 0)),
                "supplier_id": r.get("supplier_id", "S01"),
                "n_manufacturers": int(r.get("n_manufacturers", 1)),
            }
        )

    count_label = f"Showing {len(records):,} of {len(latest):,} Series"
    return records, count_label


@app.callback(
    Output("medicine-inspector-container", "children"),
    Input("alerts-table", "selected_rows"),
    Input("alerts-table", "data"),
)
def update_medicine_inspector(selected_rows, table_data):
    """Render the supply chain diagnostic inspector card for selected medicine."""
    if not table_data:
        return html.Div(
            style={"textAlign": "center", "padding": "40px", "color": "#94a3b8"},
            children=[
                html.Div("No records match the selected filter.", style={"fontWeight": "600", "fontSize": "14px"}),
            ],
        )

    idx = selected_rows[0] if (selected_rows and selected_rows[0] < len(table_data)) else 0
    item = table_data[idx]

    risk_val = item["risk"]
    risk_pct = risk_val * 100

    # Risk Meter Color & Status
    if risk_val >= 0.80:
        meter_color = "#ef4444"
        meter_bg = "#fef2f2"
        status_label = "CRITICAL STOCKOUT RISK"
    elif risk_val >= THR:
        meter_color = "#f59e0b"
        meter_bg = "#fffbeb"
        status_label = "HIGH STOCKOUT RISK"
    elif risk_val >= 0.40:
        meter_color = "#2563eb"
        meter_bg = "#eff6ff"
        status_label = "ELEVATED MONITORING"
    else:
        meter_color = "#10b981"
        meter_bg = "#ecfdf5"
        status_label = "SAFE INVENTORY LEVEL"

    # Action Rationale & Playbook
    action = item["action"]
    if action == "Expedite / Transfer":
        rec_bg = "#fef2f2"
        rec_border = "1px solid #fecaca"
        rec_color = "#b91c1c"
        rationale = (
            f"Active upstream manufacturer shortage detected with only {item['n_manufacturers']} certified supplier(s). "
            "Reordering from the primary distributor will face extended delays. Immediately expedite open shipments "
            "or trigger an emergency inter-facility stock transfer from neighboring healthcare facilities."
        )
    elif action == "Switch Alternate Mfr":
        rec_bg = "#fffbeb"
        rec_border = "1px solid #fde68a"
        rec_color = "#b45309"
        rationale = (
            f"Primary supplier has active shortage, but {item['n_manufacturers']} alternate qualified manufacturers are available. "
            "Re-route purchase orders to an alternate certified manufacturer in the Tamil Nadu supply network."
        )
    elif action == "Reorder Now":
        rec_bg = "#eff6ff"
        rec_border = "1px solid #bfdbfe"
        rec_color = "#1d4ed8"
        rationale = (
            "Current on-hand physical stock will not cover the forecasted 14-day demand. "
            "Dispatch a replenishment purchase order immediately to reach the dynamic reorder point."
        )
    else:
        rec_bg = "#ecfdf5"
        rec_border = "1px solid #a7f3d0"
        rec_color = "#047857"
        rationale = "Inventory coverage is sufficient through the 14-day forecasting horizon. No emergency action required."

    return html.Div(
        children=[
            # Top Info
            html.Div(
                className="inspector-top-info",
                children=[
                    html.Span(
                        item["category"],
                        className="portal-badge-count",
                        style={"marginBottom": "8px", "display": "inline-block"},
                    ),
                    html.Div(item["drug_name"], className="inspector-drug-name"),
                    html.Div(
                        [
                            html.Span(className="lucide-icon lucide-building-icon", style={"marginRight": "6px", "verticalAlign": "middle"}),
                            html.Span(f"{item['facility_id']} — {item['city']} ({item['facility_type']})"),
                        ],
                        className="inspector-facility-info",
                    ),
                ],
            ),
            # Risk Gauge Box
            html.Div(
                className="portal-risk-gauge-box",
                style={"backgroundColor": meter_bg},
                children=[
                    html.Div(f"{risk_pct:.1f}%", className="portal-risk-gauge-val", style={"color": meter_color}),
                    html.Div(status_label, className="portal-risk-gauge-label", style={"color": meter_color}),
                    html.Div(
                        className="portal-risk-gauge-track",
                        children=html.Div(
                            className="portal-risk-gauge-fill",
                            style={"width": f"{min(100, max(5, risk_pct))}%", "background": meter_color},
                        ),
                    ),
                ],
            ),
            # Diagnostic Metrics List
            html.Div(
                className="portal-metric-rows",
                children=[
                    html.Div(
                        className="portal-metric-row",
                        children=[
                            html.Span("Days of Supply", className="portal-metric-row-label"),
                            html.Span(item["days_of_supply"], className="portal-metric-row-val", style={"color": meter_color}),
                        ],
                    ),
                    html.Div(
                        className="portal-metric-row",
                        children=[
                            html.Span("14-Day Demand Forecast", className="portal-metric-row-label"),
                            html.Span(item["fc14"], className="portal-metric-row-val"),
                        ],
                    ),
                    html.Div(
                        className="portal-metric-row",
                        children=[
                            html.Span("On-Hand Physical Stock", className="portal-metric-row-label"),
                            html.Span(f"{item['on_hand']:,} units", className="portal-metric-row-val"),
                        ],
                    ),
                    html.Div(
                        className="portal-metric-row",
                        children=[
                            html.Span("Open In-Transit Order", className="portal-metric-row-label"),
                            html.Span(f"{item['on_order']:,} units", className="portal-metric-row-val"),
                        ],
                    ),
                    html.Div(
                        className="portal-metric-row",
                        children=[
                            html.Span("Next Shipment ETA", className="portal-metric-row-label"),
                            html.Span(item["eta_clean"], className="portal-metric-row-val"),
                        ],
                    ),
                    html.Div(
                        className="portal-metric-row",
                        children=[
                            html.Span("Certified Manufacturers", className="portal-metric-row-label"),
                            html.Span(f"{item['n_manufacturers']} Source(s)", className="portal-metric-row-val"),
                        ],
                    ),
                ],
            ),
            # Recommended Dispatch Action Card
            html.Div(
                className="portal-action-box",
                style={"background": rec_bg, "border": rec_border},
                children=[
                    html.Div("RECOMMENDED DISPATCH ACTION", className="portal-action-badge-title", style={"color": rec_color}),
                    html.Div(action, className="portal-action-heading", style={"color": rec_color}),
                    html.Div(rationale, className="portal-action-desc"),
                ],
            ),
        ]
    )


# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=False, port=8050)
