"""
DOE Designer — Plotly Dash GUI  (Design + Analysis)
Run:  python app.py  → http://127.0.0.1:8050
"""

import io, base64, json, uuid
from itertools import combinations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html, dash_table, Input, Output, State, ctx, ALL, no_update
import dash_bootstrap_components as dbc
from flask import request as flask_request, jsonify

from doe_generators  import generate_design, list_taguchi_arrays, apply_design_structure
from analysis import (
    ADMIN_COLS, fit_model, get_anova_table, get_coefficients, get_model_stats,
    get_equations, get_residuals, plot_pareto, plot_half_normal,
    plot_residuals, plot_main_effects, plot_interaction,
    predict_response, get_surface_data, optimize_response,
)

# ── App ───────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY, dbc.icons.BOOTSTRAP],
    title="DOE Designer",
    suppress_callback_exceptions=True,
)
server = app.server

# Agent state — shared between Flask endpoints and Dash callbacks
_agent_pending_config: dict = {}   # set by /agent/configure, cleared after Dash reads it
_agent_readable_state: dict = {}   # set by Dash callbacks, read by /agent/state

ACCENT  = "#2C7BE5"
CARD_SH = "0 2px 8px rgba(0,0,0,.09)"

DESIGNS = {
    "two_level_full":    {"label": "2-Level Full Factorial",       "group": "Factorial"},
    "fractional":        {"label": "Fractional Factorial",         "group": "Factorial"},
    "plackett_burman":   {"label": "Plackett-Burman (Screening)",  "group": "Screening"},
    "ccd":               {"label": "Central Composite (CCD)",      "group": "Response Surface"},
    "box_behnken":       {"label": "Box-Behnken",                  "group": "Response Surface"},
    "general_factorial": {"label": "General Full Factorial",       "group": "Factorial"},
    "taguchi":           {"label": "Taguchi Orthogonal Array",     "group": "Robust"},
    "simplex_lattice":   {"label": "Mixture — Simplex Lattice",    "group": "Mixture"},
    "simplex_centroid":  {"label": "Mixture — Simplex Centroid",   "group": "Mixture"},
}
GROUPS        = ["Factorial", "Screening", "Response Surface", "Robust", "Mixture"]
TAGUCHI_ARRAYS = list_taguchi_arrays()

# Admin column colours in DataTable
_ADMIN_STYLE = {
    "Std Order": ("#6c757d", "italic"),
    "Run Order": (ACCENT,    "normal"),
    "Block":     ("#0ca678", "normal"),
    "Replicate": ("#e67700", "normal"),
}


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

def _sidebar_section(group, keys):
    return html.Div([
        html.Div(group, className="px-3 pt-3 pb-1",
                 style={"fontSize": "0.68rem", "fontWeight": 700, "color": "#8e9aaf",
                        "textTransform": "uppercase", "letterSpacing": "0.07em"}),
        *[dbc.Button(DESIGNS[k]["label"],
                     id={"type": "design-btn", "index": k},
                     color="light",
                     className="text-start w-100 mb-1 py-1 px-3",
                     style={"fontSize": "0.81rem", "borderRadius": "5px", "border": "none"},
                     n_clicks=0)
          for k in keys],
    ])

def _design_type_card():
    """Card used inside the Design tab for selecting the design type."""
    return dbc.Card([
        dbc.CardHeader(html.Strong("Design Type")),
        dbc.CardBody(
            html.Div([
                item
                for grp in GROUPS
                for item in [
                    _sidebar_section(grp, [k for k, v in DESIGNS.items()
                                           if v["group"] == grp]),
                ]
            ], style={"overflowY": "auto", "maxHeight": "70vh"}),
            className="p-1",
        ),
    ], style={"boxShadow": CARD_SH})


# ═══════════════════════════════════════════════════════════════════════════════
# DESIGN TAB — helpers
# ═══════════════════════════════════════════════════════════════════════════════

def make_factor_row(idx, name="", low=-1, high=1, num_levels=3):
    return dbc.Row([
        dbc.Col(dbc.Input(id={"type": "f-name",   "index": idx}, value=name,
                          placeholder=f"Factor {idx+1}", debounce=True, size="sm"), width=3),
        dbc.Col(dbc.Input(id={"type": "f-low",    "index": idx}, value=low,
                          placeholder="Low",  type="number", debounce=True, size="sm"), width=2),
        dbc.Col(dbc.Input(id={"type": "f-high",   "index": idx}, value=high,
                          placeholder="High", type="number", debounce=True, size="sm"), width=2),
        dbc.Col(dbc.Input(id={"type": "f-levels", "index": idx}, value=num_levels,
                          placeholder="Lvls", type="number", min=2, debounce=True, size="sm"), width=2),
        dbc.Col(dbc.Button(html.I(className="bi bi-x"),
                           id={"type": "del-factor", "index": idx},
                           color="danger", outline=True, size="sm", n_clicks=0), width=1),
    ], className="mb-1 g-1 align-items-center", id={"type": "factor-row", "index": idx})


# All design-type option panels — always in DOM, visibility toggled by CSS
all_options = html.Div([
    html.Div(id="panel-fractional", style={"display": "none"},
             children=dbc.Card(dbc.CardBody([
                 html.H6("Fractional Factorial Options", className="fw-bold mb-2 small"),
                 dbc.Row([
                     dbc.Col([dbc.Label("Resolution", className="small mb-0"),
                              dbc.Select(id="opt-resolution",
                                         options=[{"label": f"Resolution {r}", "value": r} for r in [3,4,5]],
                                         value=3, size="sm")], width=5),
                     dbc.Col([dbc.Label("Custom generators (optional)", className="small mb-0"),
                              dbc.Input(id="opt-generators", placeholder="e.g.  a  b  ab",
                                        debounce=True, size="sm")], width=7),
                 ], className="g-2"),
             ]), className="border-0 mt-2", style={"background": "#eef4ff"})),

    html.Div(id="panel-ccd", style={"display": "none"},
             children=dbc.Card(dbc.CardBody([
                 html.H6("CCD Options", className="fw-bold mb-2 small"),
                 dbc.Row([
                     dbc.Col([dbc.Label("Face type", className="small mb-0"),
                              dbc.Select(id="opt-ccd-face",
                                         options=[{"label": "Circumscribed (CCC)", "value": "ccc"},
                                                  {"label": "Inscribed (CCI)",     "value": "cci"},
                                                  {"label": "Face-centered (CCF)", "value": "ccf"}],
                                         value="ccc", size="sm")], width=4),
                     dbc.Col([dbc.Label("Alpha", className="small mb-0"),
                              dbc.Select(id="opt-ccd-alpha",
                                         options=[{"label": "Orthogonal", "value": "orthogonal"},
                                                  {"label": "Rotatable",  "value": "rotatable"}],
                                         value="orthogonal", size="sm")], width=4),
                     dbc.Col([dbc.Label("Center pts (fact / star)", className="small mb-0"),
                              dbc.Row([dbc.Col(dbc.Input(id="opt-ccd-cf", value=4, type="number",
                                                         min=0, size="sm"), width=6),
                                       dbc.Col(dbc.Input(id="opt-ccd-cs", value=4, type="number",
                                                         min=0, size="sm"), width=6)],
                                      className="g-1")], width=4),
                 ], className="g-2"),
             ]), className="border-0 mt-2", style={"background": "#eef4ff"})),

    html.Div(id="panel-box-behnken", style={"display": "none"},
             children=dbc.Card(dbc.CardBody([
                 html.H6("Box-Behnken Options", className="fw-bold mb-2 small"),
                 dbc.Row([dbc.Col([dbc.Label("Center points", className="small mb-0"),
                                   dbc.Input(id="opt-bb-center", value=1, type="number",
                                             min=0, size="sm")], width=4)]),
             ]), className="border-0 mt-2", style={"background": "#eef4ff"})),

    html.Div(id="panel-taguchi", style={"display": "none"},
             children=dbc.Card(dbc.CardBody([
                 html.H6("Taguchi Options", className="fw-bold mb-2 small"),
                 dbc.Row([dbc.Col([dbc.Label("Orthogonal array", className="small mb-0"),
                                   dbc.Select(id="opt-taguchi-array",
                                              options=[{"label": a, "value": a} for a in TAGUCHI_ARRAYS],
                                              value="L8(2^7)", size="sm")], width=7)]),
             ]), className="border-0 mt-2", style={"background": "#eef4ff"})),

    html.Div(id="panel-simplex-lattice", style={"display": "none"},
             children=dbc.Card(dbc.CardBody([
                 html.H6("Simplex Lattice Options", className="fw-bold mb-2 small"),
                 dbc.Row([dbc.Col([dbc.Label("Degree (m)", className="small mb-0"),
                                   dbc.Input(id="opt-sl-degree", value=2, type="number",
                                             min=1, max=6, size="sm")], width=4)]),
             ]), className="border-0 mt-2", style={"background": "#eef4ff"})),

    html.Div(id="panel-center-points", style={"display": "none"},
             children=dbc.Card(dbc.CardBody([
                 html.H6("Center Points", className="fw-bold mb-2 small"),
                 dbc.Row([
                     dbc.Col([
                         dbc.Label("Center points per block", className="small mb-0"),
                         dbc.Input(id="opt-center-points", value=0, type="number",
                                   min=0, max=10, size="sm"),
                         html.Small("Added per block — all factors at midpoint",
                                    className="text-muted"),
                     ], width=8),
                 ], className="g-2"),
             ]), className="border-0 mt-2", style={"background": "#eef4ff"})),
])

_PANEL_IDS = {
    "fractional":      "panel-fractional",
    "ccd":             "panel-ccd",
    "box_behnken":     "panel-box-behnken",
    "taguchi":         "panel-taguchi",
    "simplex_lattice": "panel-simplex-lattice",
}

# Design types that support appending center-point runs
_CENTER_POINT_DESIGNS = {"two_level_full", "fractional", "plackett_burman"}

# Design structure card
design_structure_card = dbc.Card([
    dbc.CardHeader(dbc.Row([
        dbc.Col(html.Strong("Design Structure"), width="auto"),
        dbc.Col(dbc.Badge("Randomization · Replication · Blocking",
                          color="light", text_color="secondary", className="fw-normal"),
                width="auto", className="ms-auto d-flex align-items-center"),
    ], align="center")),
    dbc.CardBody(dbc.Row([
        dbc.Col([dbc.Label("Replicates", className="small mb-0 fw-bold"),
                 dbc.Input(id="opt-replicates", value=1, type="number", min=1, max=50, size="sm"),
                 html.Small("Repeat full design N times", className="text-muted")], width=4),
        dbc.Col([dbc.Label("Blocks",     className="small mb-0 fw-bold"),
                 dbc.Input(id="opt-blocks",     value=1, type="number", min=1, max=50, size="sm"),
                 html.Small("Divide runs into N blocks", className="text-muted")], width=4),
        dbc.Col([dbc.Label("Randomize",  className="small mb-0 fw-bold"),
                 dbc.Checklist(id="opt-randomize",
                               options=[{"label": " Run order", "value": "yes"}],
                               value=["yes"], switch=True, className="mt-1"),
                 html.Small("Shuffle within blocks", className="text-muted")], width=4),
    ], className="g-2"), className="pb-2"),
], className="mt-2", style={"boxShadow": CARD_SH, "borderLeft": f"3px solid {ACCENT}"})


def _build_design_table(df_disp, factor_cols):
    """Styled DataTable for the generated design matrix."""
    header_cond = (
        [{"if": {"column_id": col}, "backgroundColor": info[0], "color": "white"}
         for col, info in _ADMIN_STYLE.items() if col in df_disp.columns]
        + [{"if": {"column_id": c},  "backgroundColor": ACCENT,   "color": "white"}
           for c in factor_cols]
    )
    data_cond = [
        {"if": {"row_index": "odd"}, "backgroundColor": "#f7f9fc"},
        *[{"if": {"column_id": col}, "fontWeight": "bold",
           "color": info[0], "fontStyle": info[1]}
          for col, info in _ADMIN_STYLE.items() if col in df_disp.columns],
    ]
    return dash_table.DataTable(
        id="result-datatable",
        columns=[{"name": c, "id": c} for c in df_disp.columns],
        data=df_disp.to_dict("records"),
        page_size=25, sort_action="native", filter_action="native",
        style_table={"overflowX": "auto"},
        style_header={"fontWeight": "bold", "fontSize": "0.82rem",
                      "textAlign": "center", "border": "none"},
        style_header_conditional=header_cond,
        style_cell={"fontSize": "0.82rem", "padding": "5px 12px",
                    "textAlign": "center", "border": "1px solid #dee2e6"},
        style_data_conditional=data_cond,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# DESIGN TAB LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════

design_tab = dbc.Container([
    dbc.Row([
        # Column 1 — Design type selector
        dbc.Col(_design_type_card(), md=2),

        # Column 2 — Factors, options, structure
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(dbc.Row([
                    dbc.Col(html.Strong("Factors"), width="auto"),
                    dbc.Col(dbc.Button([html.I(className="bi bi-plus me-1"), "Add Factor"],
                                       id="add-factor-btn", color="primary",
                                       outline=True, size="sm", n_clicks=0),
                            width="auto", className="ms-auto"),
                ], align="center")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(html.Small("Name",   className="text-muted fw-bold"), width=3),
                        dbc.Col(html.Small("Low",    className="text-muted fw-bold"), width=2),
                        dbc.Col(html.Small("High",   className="text-muted fw-bold"), width=2),
                        dbc.Col(html.Small("# Lvls", className="text-muted fw-bold"), width=2),
                    ], className="mb-1 g-1"),
                    html.Div(id="factors-container", children=[
                        make_factor_row(0, "A", -1, 1),
                        make_factor_row(1, "B", -1, 1),
                        make_factor_row(2, "C", -1, 1),
                    ]),
                ]),
            ], style={"boxShadow": CARD_SH}),
            all_options,
            design_structure_card,
            dbc.Button([html.I(className="bi bi-play-fill me-2"), "Generate Design"],
                       id="generate-btn", color="primary", size="lg",
                       className="mt-3 w-100", n_clicks=0),
            html.Div(id="error-alert", className="mt-2"),
        ], md=4),

        # Column 3 — Design matrix
        dbc.Col(dbc.Card([
            dbc.CardHeader(dbc.Row([
                dbc.Col(html.Strong("Design Matrix"), width="auto"),
                dbc.Col(html.Div([
                    dbc.Button([html.I(className="bi bi-clipboard me-1"), "Copy"],
                               id="copy-btn", color="secondary", outline=True,
                               size="sm", n_clicks=0, className="me-2"),
                    dbc.Button([html.I(className="bi bi-filetype-csv me-1"), "CSV"],
                               id="export-csv-btn", color="success", outline=True,
                               size="sm", n_clicks=0, className="me-2"),
                    dbc.Button([html.I(className="bi bi-file-earmark-excel me-1"), "Excel"],
                               id="export-excel-btn", color="success", outline=True,
                               size="sm", n_clicks=0),
                    dcc.Download(id="download-csv"),
                    dcc.Download(id="download-excel"),
                    dcc.Clipboard(id="clipboard", style={"display": "none"}),
                ], className="d-flex"), width="auto", className="ms-auto"),
            ], align="center")),
            dbc.CardBody([
                html.Div(id="design-stats", className="mb-2"),
                html.Div(id="results-table"),
            ]),
        ], style={"boxShadow": CARD_SH, "minHeight": "500px"}), md=6),
    ], className="mt-3 g-3"),
], fluid=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS TAB LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════

# ── Section 1: Response data entry ───────────────────────────────────────────
data_entry_card = dbc.Card([
    dbc.CardHeader(html.Strong("Design & Response Data")),
    dbc.CardBody([
        # ── Import toolbar ────────────────────────────────────────────────────
        dbc.Row([
            dbc.Col([
                dbc.Button(
                    [html.I(className="bi bi-arrow-left-circle me-1"), "Load from Design Tab"],
                    id="load-design-btn", color="primary", outline=True, size="sm", n_clicks=0,
                ),
                html.Small("Use the design generated on the Design tab",
                           className="text-muted d-block mt-1"),
            ], md="auto"),
            dbc.Col([
                dcc.Upload(
                    id="upload-design",
                    children=dbc.Button(
                        [html.I(className="bi bi-file-earmark-arrow-up me-1"),
                         "Upload CSV / Excel"],
                        color="secondary", outline=True, size="sm",
                    ),
                    accept=".csv,.xlsx,.xls",
                    multiple=False,
                ),
                html.Small("Upload a design or full dataset (CSV or Excel)",
                           className="text-muted d-block mt-1"),
            ], md="auto"),
            dbc.Col([
                dbc.Button(
                    [html.I(className="bi bi-clipboard me-1"), "Paste Data"],
                    id="toggle-paste-btn", color="info", outline=True, size="sm", n_clicks=0,
                ),
                html.Small("Paste tab-separated data from Excel",
                           className="text-muted d-block mt-1"),
            ], md="auto"),
            dbc.Col([
                dbc.Button(
                    [html.I(className="bi bi-plus-circle me-1"), "Add Response Column"],
                    id="add-response-btn", color="success", outline=True, size="sm", n_clicks=0,
                ),
                html.Small("Add a new editable response column",
                           className="text-muted d-block mt-1"),
            ], md="auto"),
        ], className="g-3 mb-3"),

        # ── Paste area (collapsible) ──────────────────────────────────────────
        dbc.Collapse([
            dbc.Row([
                dbc.Col(
                    dcc.Textarea(
                        id="paste-textarea",
                        placeholder="Paste tab-separated data from Excel here (first row = headers)…",
                        style={"width": "100%", "height": "110px",
                               "fontSize": "0.78rem", "fontFamily": "monospace"},
                    ), width=10,
                ),
                dbc.Col(
                    dbc.Button("Parse & Load", id="parse-paste-btn",
                               color="primary", size="sm", n_clicks=0,
                               className="h-100 w-100"),
                    width=2,
                ),
            ], className="g-2 align-items-stretch mb-2"),
            html.Small(
                "Tip: copy your design + response columns from Excel and paste here.",
                className="text-muted d-block mb-3",
            ),
        ], id="paste-collapse", is_open=False),

        # ── Design + response table ───────────────────────────────────────────
        html.Div(
            id="analysis-table-container",
            children=html.Div(
                [html.I(className="bi bi-arrow-left-circle me-2"),
                 "Generate a design on the Design tab, or upload / paste data above."],
                className="text-muted text-center py-4",
            ),
        ),
    ]),
], style={"boxShadow": CARD_SH})

# ── Section 2: Model setup (left panel) ──────────────────────────────────────
model_setup_card = dbc.Card([
    dbc.CardHeader(html.Strong("Model Setup")),
    dbc.CardBody([
        dbc.Label("Response Variable", className="small fw-bold mb-0"),
        dcc.Dropdown(id="response-dropdown", placeholder="Select response…",
                     clearable=False, className="mb-1"),
        dbc.Row([
            dbc.Col(dbc.Input(id="response-rename-input",
                              placeholder="Rename column…",
                              debounce=True, size="sm"), width=8),
            dbc.Col(dbc.Button("Rename", id="rename-response-btn",
                               size="sm", color="light", outline=True,
                               n_clicks=0), width=4),
        ], className="g-1 mb-3"),

        dbc.Label("Factors to Include", className="small fw-bold mb-0"),
        html.Div(id="factor-checklist-container",
                 children=dbc.Checklist(id="factor-checklist", value=[], options=[]),
                 className="mb-3"),

        dbc.Label("Model Terms", className="small fw-bold mb-0"),
        dbc.ButtonGroup([
            dbc.Button("Main only",     id="qs-main-btn", size="sm", color="outline-primary",
                       n_clicks=0, className="flex-fill"),
            dbc.Button("+ 2FI",         id="qs-twfi-btn", size="sm", color="outline-primary",
                       n_clicks=0, className="flex-fill"),
            dbc.Button("Full factorial", id="qs-full-btn", size="sm", color="outline-primary",
                       n_clicks=0, className="flex-fill"),
        ], className="w-100 mb-2"),
        dash_table.DataTable(
            id="term-picker-table",
            columns=[
                {"name": "Term",     "id": "Term",     "editable": False},
                {"name": "In model", "id": "In model", "editable": True,
                 "presentation": "dropdown"},
            ],
            dropdown={
                "In model": {
                    "options": [{"label": "✓", "value": True},
                                {"label": "—", "value": False}]
                }
            },
            data=[],
            editable=True,
            style_table={"overflowY": "auto", "maxHeight": "260px"},
            style_header={"backgroundColor": ACCENT, "color": "white",
                          "fontWeight": "bold", "fontSize": "0.78rem",
                          "textAlign": "center"},
            style_cell={"fontSize": "0.78rem", "padding": "2px 8px",
                        "textAlign": "center", "border": "1px solid #dee2e6"},
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "#f7f9fc"},
                {"if": {"filter_query": "{disabled} = true"},
                 "color": "#adb5bd", "fontStyle": "italic"},
            ],
            style_cell_conditional=[
                {"if": {"column_id": "Term"}, "textAlign": "left", "width": "70%"},
                {"if": {"column_id": "In model"}, "width": "30%"},
            ],
            css=[{"selector": ".Select-value-label", "rule": "color: #2C7BE5 !important"}],
        ),
        html.Div(id="term-picker-tooltip-area", className="mb-3"),

        dbc.Button([html.I(className="bi bi-calculator me-2"), "Fit Model"],
                   id="fit-model-btn", color="primary",
                   className="w-100", n_clicks=0),
        html.Div(id="fit-error", className="mt-2"),

        # Model statistics summary (populated after fitting)
        html.Div(id="model-stats-summary", className="mt-3"),
    ]),
], style={"boxShadow": CARD_SH})

# ── Section 3: Results accordion (right panel) ───────────────────────────────
results_accordion = dbc.Accordion([

    dbc.AccordionItem([
        html.Div(id="anova-table-div",
                 children=html.Div("Fit a model to see the ANOVA table.",
                                   className="text-muted text-center py-3")),
    ], title="ANOVA Table", item_id="anova"),

    dbc.AccordionItem([
        dbc.Row([
            dbc.Col(dcc.Graph(id="pareto-plot",    config={"displayModeBar": False},
                              style={"minHeight": "300px"}), md=6),
            dbc.Col([
                dbc.Switch(
                    id="lenth-lines-toggle",
                    label="Show Lenth ME / SME",
                    value=True,
                    className="mb-1",
                    style={"fontSize": "0.8rem"},
                ),
                dcc.Graph(id="halfnormal-plot",
                          config={"displayModeBar": False},
                          style={"minHeight": "300px"}),
            ], md=6),
        ], className="mb-3"),
        dcc.Graph(id="main-effects-plot", config={"displayModeBar": False}),
        html.Hr(className="my-2"),
        dbc.Row([
            dbc.Col([dbc.Label("Interaction plot — Factor A", className="small"),
                     dcc.Dropdown(id="ia-factor-a", placeholder="Select…", clearable=False)],
                    md=3),
            dbc.Col([dbc.Label("Factor B", className="small"),
                     dcc.Dropdown(id="ia-factor-b", placeholder="Select…", clearable=False)],
                    md=3),
            dbc.Col(dbc.Button([html.I(className="bi bi-graph-up me-1"), "Plot Interaction"],
                               id="plot-ia-btn", color="secondary", outline=True,
                               size="sm", n_clicks=0, className="mt-4"),
                    md=2),
        ], className="g-2 align-items-end"),
        dcc.Graph(id="interaction-plot", config={"displayModeBar": False}),
    ], title="Effects Analysis (Pareto · Half-Normal · Main Effects · Interactions)", item_id="effects"),

    dbc.AccordionItem([
        dcc.Graph(id="residual-plots", config={"displayModeBar": True}),
    ], title="Residual Analysis — Model Adequacy Checking", item_id="residuals"),

], id="results-accordion", always_open=True, active_item=["anova", "effects"])

analysis_tab = dbc.Container([
    html.Div(id="block-detect-banner"),
    dbc.Row(dbc.Col(data_entry_card), className="mt-3"),
    dbc.Row([
        dbc.Col(model_setup_card,  md=3),
        dbc.Col(results_accordion, md=9),
    ], className="mt-3 g-3"),
], fluid=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION & OPTIMIZATION TAB LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════

prediction_tab = dbc.Container([
    dbc.Row([
        # Left: Coefficients + Equation
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.Strong("Regression Coefficients")),
                dbc.CardBody(html.Div(
                    id="pred-coefficients-div",
                    children=html.Div("Fit a model on the Analysis tab first.",
                                      className="text-muted text-center py-3"),
                )),
            ], style={"boxShadow": CARD_SH}),
            dbc.Card([
                dbc.CardHeader(html.Strong("Model Equation")),
                dbc.CardBody(html.Div(id="pred-equation-div")),
            ], className="mt-3", style={"boxShadow": CARD_SH}),
        ], md=7),

        # Right: Optimization
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.Strong("Response Optimization")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Goal", className="small fw-bold mb-1"),
                            dbc.RadioItems(
                                id="opt-goal",
                                options=[
                                    {"label": " Maximize", "value": "maximize"},
                                    {"label": " Minimize", "value": "minimize"},
                                    {"label": " Target",   "value": "target"},
                                ],
                                value="maximize",
                                className="mb-2",
                            ),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Target Value", className="small fw-bold mb-1"),
                            dbc.Input(id="opt-target-value", type="number", size="sm",
                                      placeholder="Enter target…"),
                            html.Small("(used when goal = Target)",
                                       className="text-muted d-block mt-1"),
                        ], width=6),
                    ], className="g-2"),
                    dbc.Button(
                        [html.I(className="bi bi-bullseye me-2"), "Run Optimization"],
                        id="optimize-btn", color="primary",
                        className="w-100 mt-2", n_clicks=0,
                    ),
                    html.Div(id="optimization-results", className="mt-2"),
                ]),
            ], style={"boxShadow": CARD_SH}),
        ], md=5),
    ], className="mt-3 g-3"),

    # Response Surface
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.Strong("Response Surface Plots")),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Factor X", className="small fw-bold mb-0"),
                        dcc.Dropdown(id="surface-fa", clearable=False,
                                     placeholder="Select…"),
                    ], md=3),
                    dbc.Col([
                        dbc.Label("Factor Y", className="small fw-bold mb-0"),
                        dcc.Dropdown(id="surface-fb", clearable=False,
                                     placeholder="Select…"),
                    ], md=3),
                    dbc.Col([
                        dbc.Label("Other factors held at:",
                                  className="small fw-bold mb-0"),
                        html.Div(id="surface-constants-container",
                                 className="d-flex flex-wrap gap-2 mt-1"),
                    ], md=4),
                    dbc.Col(
                        dbc.Button(
                            [html.I(className="bi bi-graph-up me-1"), "Plot Surfaces"],
                            id="plot-surface-btn", color="primary",
                            size="sm", n_clicks=0, className="w-100 mt-4",
                        ), md=2,
                    ),
                ], className="g-2 align-items-end mb-3"),
                dbc.Row([
                    dbc.Col(dcc.Graph(id="contour-plot",
                                     config={"displayModeBar": True},
                                     style={"minHeight": "420px"}), md=6),
                    dbc.Col(dcc.Graph(id="surface-3d-plot",
                                     config={"displayModeBar": True},
                                     style={"minHeight": "420px"}), md=6),
                ]),
            ]),
        ], style={"boxShadow": CARD_SH})),
    ], className="mt-3"),
], fluid=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════

main = html.Div([
    dbc.Tabs([
        dbc.Tab(design_tab,     label="Design",                tab_id="tab-design",
                label_style={"fontWeight": 600}),
        dbc.Tab(analysis_tab,   label="Analysis",              tab_id="tab-analysis",
                label_style={"fontWeight": 600}),
        dbc.Tab(prediction_tab, label="Prediction & Optimization", tab_id="tab-prediction",
                label_style={"fontWeight": 600}),
    ], id="main-tabs", active_tab="tab-design",
       className="px-3 pt-2",
       style={"borderBottom": "1px solid #dee2e6",
              "boxShadow": CARD_SH, "background": "white"}),

    # Stores
    dcc.Store(id="active-design",       data="two_level_full"),
    dcc.Store(id="factor-count",        data=3),
    dcc.Store(id="design-df",           data=None),   # generated design JSON
    dcc.Store(id="analysis-df",         data=None),   # design + response data
    dcc.Store(id="block-col-store",     data=None),   # "Block" when detected, else None
    dcc.Store(id="fit-info-store",      data=None),   # fitted model metadata
    dcc.Store(id="surf-const-factors",  data=[]),     # factor names for surface constants
    dcc.Store(id="response-cols-store", data=["Response"]),  # list of response column names
    # Agent bridge stores
    dcc.Store(id="agent-config-store",  storage_type="memory"),
    dcc.Store(id="agent-state-sync",    storage_type="memory"),
    dcc.Store(id="session-id",          storage_type="memory", data=str(uuid.uuid4())),
    dcc.Interval(id="agent-poll",       interval=2000, n_intervals=0),
])

app.layout = main


# ═══════════════════════════════════════════════════════════════════════════════
# FLASK API ENDPOINTS — agent bridge
# ═══════════════════════════════════════════════════════════════════════════════

@server.route("/agent/configure", methods=["POST"])
def agent_configure():
    payload = flask_request.get_json(force=True)
    _agent_pending_config.clear()
    _agent_pending_config.update(payload)
    # Immediately reflect design_type in readable state so test scripts don't
    # depend on a live browser session to trigger the Dash poll callback.
    if "design_type" in payload:
        _agent_readable_state["design_type"] = payload["design_type"]
        _agent_readable_state["factors"] = payload.get("factors", [])
        _agent_readable_state["options"] = payload.get("options", {})
    return jsonify({"status": "ok"})


@server.route("/agent/state", methods=["GET"])
def agent_state():
    return jsonify(_agent_readable_state)


# ═══════════════════════════════════════════════════════════════════════════════
# CALLBACKS — DESIGN TAB
# ═══════════════════════════════════════════════════════════════════════════════

# 1. Select design type → update active store + show correct option panel
@app.callback(
    Output("active-design",         "data"),
    Output("panel-fractional",      "style"),
    Output("panel-ccd",             "style"),
    Output("panel-box-behnken",     "style"),
    Output("panel-taguchi",         "style"),
    Output("panel-simplex-lattice", "style"),
    Output("panel-center-points",   "style"),
    Input({"type": "design-btn", "index": ALL}, "n_clicks"),
    State("active-design", "data"),
    prevent_initial_call=False,
)
def select_design(_, current):
    triggered = ctx.triggered_id
    design = triggered["index"] if isinstance(triggered, dict) else (current or "two_level_full")
    show, hide = {"display": "block"}, {"display": "none"}
    all_panels = set(_PANEL_IDS.values()) | {"panel-center-points"}
    styles = {p: hide for p in all_panels}
    if (pid := _PANEL_IDS.get(design)):
        styles[pid] = show
    if design in _CENTER_POINT_DESIGNS:
        styles["panel-center-points"] = show
    return (design,
            styles["panel-fractional"], styles["panel-ccd"],
            styles["panel-box-behnken"], styles["panel-taguchi"],
            styles["panel-simplex-lattice"], styles["panel-center-points"])


# 2. Add / delete factor rows
@app.callback(
    Output("factors-container", "children"),
    Output("factor-count",      "data"),
    Input("add-factor-btn", "n_clicks"),
    Input({"type": "del-factor", "index": ALL}, "n_clicks"),
    State("factors-container", "children"),
    State("factor-count",      "data"),
    prevent_initial_call=True,
)
def manage_factors(_, del_ns, rows, count):
    triggered = ctx.triggered_id
    if triggered == "add-factor-btn":
        label = chr(ord("A") + (count % 26)) if count < 26 else f"F{count+1}"
        rows.append(make_factor_row(count, label, -1, 1))
        return rows, count + 1
    if isinstance(triggered, dict) and triggered.get("type") == "del-factor":
        del_idx = triggered["index"]
        rows = [r for r in rows if r["props"]["id"]["index"] != del_idx]
        if not rows:
            rows = [make_factor_row(count, "A", -1, 1)]
            return rows, count + 1
        return rows, count
    return rows, count


# 3. Generate design
@app.callback(
    Output("results-table",   "children"),
    Output("design-df",       "data"),
    Output("error-alert",     "children"),
    Output("design-stats",    "children"),
    Input("generate-btn", "n_clicks"),
    State("active-design",     "data"),
    State({"type": "f-name",   "index": ALL}, "value"),
    State({"type": "f-low",    "index": ALL}, "value"),
    State({"type": "f-high",   "index": ALL}, "value"),
    State({"type": "f-levels", "index": ALL}, "value"),
    State("opt-resolution",    "value"),
    State("opt-generators",    "value"),
    State("opt-ccd-face",      "value"),
    State("opt-ccd-alpha",     "value"),
    State("opt-ccd-cf",        "value"),
    State("opt-ccd-cs",        "value"),
    State("opt-bb-center",     "value"),
    State("opt-taguchi-array", "value"),
    State("opt-sl-degree",     "value"),
    State("opt-replicates",    "value"),
    State("opt-blocks",        "value"),
    State("opt-randomize",     "value"),
    State("opt-center-points", "value"),
    prevent_initial_call=True,
)
def generate(n_clicks, design_type,
             names, lows, highs, num_levels,
             frac_res, frac_gen, ccd_face, ccd_alpha, ccd_cf, ccd_cs,
             bb_center, taguchi_arr, sl_degree,
             n_replicates, n_blocks, randomize_val,
             n_center_points):
    NU = no_update
    factors = []
    for name, lo, hi, nl in zip(names, lows, highs, num_levels):
        if not name:
            continue
        try:
            lo_v, hi_v = float(lo or 0), float(hi or 1)
        except (TypeError, ValueError):
            lo_v, hi_v = 0.0, 1.0
        if lo_v >= hi_v:
            return NU, NU, dbc.Alert(f"Factor '{name}': Low ≥ High.", color="warning"), NU
        factors.append({"name": name, "low": lo_v, "high": hi_v,
                        "num_levels": int(nl) if nl else 3})
    if not factors:
        return NU, NU, dbc.Alert("Add at least one factor.", color="warning"), NU

    opts = dict(
        resolution=int(frac_res or 3), generators=frac_gen or None,
        face=ccd_face or "ccc", alpha=ccd_alpha or "orthogonal",
        center_factorial=int(ccd_cf or 4), center_star=int(ccd_cs or 4),
        center=int(bb_center or 1), array_name=taguchi_arr or "L8(2^7)",
        degree=int(sl_degree or 2),
    )
    try:
        df_base = generate_design(design_type, factors, opts)
    except Exception as e:
        return NU, NU, dbc.Alert(str(e), color="danger", dismissable=True), NU

    reps = max(1, int(n_replicates or 1))
    blks = max(1, int(n_blocks     or 1))
    rand = bool(randomize_val)
    try:
        df, warning = apply_design_structure(df_base, reps, blks, rand)
    except Exception as e:
        return NU, NU, dbc.Alert(str(e), color="danger", dismissable=True), NU

    factor_cols = [f["name"] for f in factors]

    # Append center-point runs if requested (factorial / fractional / PB only;
    # CCD embeds its own center points via pyDOE3).
    n_cp = int(n_center_points or 0)
    if n_cp > 0 and design_type in _CENTER_POINT_DESIGNS:
        # Midpoint for each factor in actual (uncoded) units
        mid = {f["name"]: (float(f["low"]) + float(f["high"])) / 2.0
               for f in factors}
        # n_cp center points PER BLOCK — iterate over unique blocks in order
        unique_blocks = list(dict.fromkeys(df["Block"].tolist()))
        cp_all = []
        std_counter = len(df) + 1
        for blk in unique_blocks:
            rep_val = df.loc[df["Block"] == blk, "Replicate"].iloc[-1]
            for _ in range(n_cp):
                row = {fname: mid[fname] for fname in factor_cols}
                row["Std Order"] = std_counter
                row["Run Order"] = std_counter
                row["Block"]     = blk
                row["Replicate"] = rep_val
                row["Point Type"] = "Center"
                cp_all.append(row)
                std_counter += 1
        df["Point Type"] = "Factorial"
        df_cp = pd.DataFrame(cp_all)
        df = pd.concat([df, df_cp], ignore_index=True)
        # Keep column order: admin cols → factor cols → Point Type
        admin_present = [c for c in ("Std Order", "Run Order", "Block", "Replicate")
                         if c in df.columns]
        df = df[admin_present + factor_cols + ["Point Type"]]

    df_disp = df.copy()
    df_disp[factor_cols] = df_disp[factor_cols].round(4)

    table = _build_design_table(df_disp, factor_cols)

    n_runs = len(df)
    k      = len(factors)
    stats_row = dbc.Row([
        dbc.Col(dbc.Badge(f"{n_runs} total runs",  color="primary",   pill=True), width="auto"),
        dbc.Col(dbc.Badge(f"{len(df_base)} runs/replicate", color="info", pill=True), width="auto"),
        dbc.Col(dbc.Badge(f"{reps} replicate{'s' if reps>1 else ''}",
                          color="warning", text_color="dark", pill=True), width="auto"),
        dbc.Col(dbc.Badge(f"{blks} block{'s' if blks>1 else ''}",
                          color="success", pill=True), width="auto"),
        dbc.Col(dbc.Badge("Randomized" if rand else "Standard order",
                          color="secondary", pill=True), width="auto"),
    ], className="g-1")

    if n_cp > 0 and design_type in _CENTER_POINT_DESIGNS:
        total_cp = n_cp * blks
        badge_label = (f"{n_cp}/block × {blks} = {total_cp} center pts"
                       if blks > 1 else
                       f"{n_cp} center pt{'s' if n_cp > 1 else ''}")
        stats_row.children.append(
            dbc.Col(dbc.Badge(badge_label, color="info", pill=True), width="auto")
        )

    alert = dbc.Alert(warning, color="warning", dismissable=True) if warning else None
    return table, df.to_json(orient="split"), alert, stats_row


# 4. Copy to clipboard
@app.callback(
    Output("clipboard", "content"),
    Output("clipboard", "n_clicks"),
    Input("copy-btn", "n_clicks"),
    State("design-df", "data"),
    prevent_initial_call=True,
)
def copy_table(_, j):
    if not j:
        return no_update, no_update
    df = pd.read_json(io.StringIO(j), orient="split")
    return df.to_csv(sep="\t", index=False), 1


# 5. Download CSV
@app.callback(
    Output("download-csv", "data"),
    Input("export-csv-btn", "n_clicks"),
    State("design-df", "data"),
    prevent_initial_call=True,
)
def export_csv(_, j):
    if not j:
        return no_update
    df = pd.read_json(io.StringIO(j), orient="split")
    return dcc.send_data_frame(df.to_csv, "doe_design.csv", index=False)


# 6. Download Excel
@app.callback(
    Output("download-excel", "data"),
    Input("export-excel-btn", "n_clicks"),
    State("design-df",    "data"),
    State("active-design","data"),
    prevent_initial_call=True,
)
def export_excel(_, j, design_type):
    if not j:
        return no_update
    df  = pd.read_json(io.StringIO(j), orient="split")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Design Matrix", index=False)
        pd.DataFrame({
            "Property": ["Design Type", "Total Runs", "Factors"],
            "Value":    [DESIGNS.get(design_type, {}).get("label", design_type),
                         len(df), len(df.columns) - 4],
        }).to_excel(writer, sheet_name="Info", index=False)
    buf.seek(0)
    return dcc.send_bytes(buf.read(), "doe_design.xlsx")


# ═══════════════════════════════════════════════════════════════════════════════
# CALLBACKS — ANALYSIS TAB
# ═══════════════════════════════════════════════════════════════════════════════

# Toggle paste panel
@app.callback(
    Output("paste-collapse", "is_open"),
    Input("toggle-paste-btn", "n_clicks"),
    State("paste-collapse",   "is_open"),
    prevent_initial_call=True,
)
def toggle_paste(_, is_open):
    return not is_open


def _parse_upload(contents, filename):
    """Decode a dcc.Upload payload into a DataFrame. Returns (df, error_str)."""
    try:
        _hdr, content_string = contents.split(",", 1)
        decoded = base64.b64decode(content_string)
        ext = (filename or "").rsplit(".", 1)[-1].lower()
        if ext in ("xlsx", "xls"):
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        return df, None
    except Exception as e:
        return None, str(e)


def _apply_resp_cols(df, resp_cols):
    """Ensure resp_cols exist as columns in df (add as None if missing)."""
    for rc in resp_cols:
        if rc not in df.columns:
            df[rc] = None
    return df


def _block_banner(df):
    """Return (banner_children, block_col_name) if df has a Block column, else (None, None)."""
    if "Block" in df.columns:
        n_blocks = df["Block"].nunique()
        banner = dbc.Alert(
            [
                html.I(className="bi bi-layers me-2"),
                "Detected blocking variable 'Block' with ",
                html.Strong(f"{n_blocks} levels"),
                " — it will be included in the analysis automatically.",
            ],
            color="info",
            dismissable=True,
            className="mb-2",
        )
        return banner, "Block"
    return None, None


# A. Populate analysis table
@app.callback(
    Output("analysis-table-container", "children"),
    Output("analysis-df",              "data"),
    Output("response-cols-store",      "data"),
    Output("block-detect-banner",      "children"),
    Output("block-col-store",          "data"),
    # Triggers
    Input("load-design-btn",  "n_clicks"),   # explicit button
    Input("design-df",        "data"),       # auto-load when design generated
    Input("upload-design",    "contents"),   # file upload (CSV or Excel)
    Input("parse-paste-btn",  "n_clicks"),   # paste
    Input("add-response-btn", "n_clicks"),   # add response column
    # States
    State("upload-design",       "filename"),
    State("paste-textarea",      "value"),
    State("analysis-df",         "data"),
    State("response-cols-store", "data"),
    prevent_initial_call=True,
)
def populate_analysis_table(load_n, design_json, upload_contents, parse_n, add_n,
                             upload_filename, paste_value,
                             prev_analysis_json, resp_cols):
    triggered_id = ctx.triggered_id
    resp_cols    = list(resp_cols or ["Response"])
    NU = no_update

    # ── File upload (CSV or Excel) ────────────────────────────────────────────
    if triggered_id == "upload-design" and upload_contents:
        df, err = _parse_upload(upload_contents, upload_filename)
        if err:
            return dbc.Alert(f"Upload error: {err}", color="danger"), NU, NU, None, None
        detected_resp = [c for c in df.columns if c.lower().startswith("response")]
        if not detected_resp:
            detected_resp = ["Response"]
            df["Response"] = None
        factor_cols = [c for c in df.columns
                       if c not in ADMIN_COLS and c not in detected_resp]
        banner, block_col = _block_banner(df)
        return (_make_analysis_table(df, factor_cols, detected_resp),
                df.to_json(orient="split"), detected_resp, banner, block_col)

    # ── Parse pasted data ─────────────────────────────────────────────────────
    if triggered_id == "parse-paste-btn":
        if not paste_value or not paste_value.strip():
            return NU, NU, NU, NU, NU
        try:
            df = pd.read_csv(io.StringIO(paste_value.strip()), sep="\t")
        except Exception as e:
            return dbc.Alert(f"Paste parse error: {e}", color="danger"), NU, NU, None, None
        detected_resp = [c for c in df.columns if c.lower().startswith("response")]
        if not detected_resp:
            detected_resp = ["Response"]
            df["Response"] = None
        factor_cols = [c for c in df.columns
                       if c not in ADMIN_COLS and c not in detected_resp]
        banner, block_col = _block_banner(df)
        return (_make_analysis_table(df, factor_cols, detected_resp),
                df.to_json(orient="split"), detected_resp, banner, block_col)

    # ── Add a new response column ─────────────────────────────────────────────
    if triggered_id == "add-response-btn":
        if not prev_analysis_json:
            return (dbc.Alert("Load a design first, then add a response column.",
                              color="warning", dismissable=True),
                    NU, NU, NU, NU)
        df        = pd.read_json(io.StringIO(prev_analysis_json), orient="split")
        n         = len(resp_cols)
        new_name  = f"Response {n + 1}" if n > 0 else "Response"
        resp_cols = resp_cols + [new_name]
        df        = _apply_resp_cols(df, resp_cols)
        factor_cols = [c for c in df.columns
                       if c not in ADMIN_COLS and c not in set(resp_cols)]
        banner, block_col = _block_banner(df)
        return (_make_analysis_table(df, factor_cols, resp_cols),
                df.to_json(orient="split"), resp_cols, banner, block_col)

    # ── Load from Design Tab (explicit button or auto-trigger on design-df change) ──
    if triggered_id in ("load-design-btn", "design-df"):
        if not design_json:
            return NU, NU, NU, NU, NU
        df_design = pd.read_json(io.StringIO(design_json), orient="split")

        # Preserve response data entered in a previous session for same design
        if prev_analysis_json:
            try:
                df_prev = pd.read_json(io.StringIO(prev_analysis_json), orient="split")
                for rc in resp_cols:
                    if rc in df_prev.columns:
                        if ("Std Order" in df_design.columns
                                and "Std Order" in df_prev.columns):
                            mapping = df_prev.set_index("Std Order")[rc]
                            df_design[rc] = df_design["Std Order"].map(mapping)
                        elif len(df_prev) == len(df_design):
                            df_design[rc] = df_prev[rc].values
            except Exception:
                pass  # if merge fails, just start fresh

        df_design = _apply_resp_cols(df_design, resp_cols)
        factor_cols = [c for c in df_design.columns
                       if c not in ADMIN_COLS and c not in set(resp_cols)]
        banner, block_col = _block_banner(df_design)
        return (_make_analysis_table(df_design, factor_cols, resp_cols),
                df_design.to_json(orient="split"), resp_cols, banner, block_col)

    return NU, NU, NU, NU, NU


def _make_analysis_table(df: pd.DataFrame, factor_cols: list, resp_cols: list):
    """Editable DataTable: design columns read-only, response columns editable."""
    resp_set = set(resp_cols)
    columns  = []
    for c in df.columns:
        editable = c in resp_set
        columns.append({"name": c, "id": c, "editable": editable,
                         "type": "numeric" if c not in ADMIN_COLS else "any"})

    # Colour coding
    header_cond = (
        [{"if": {"column_id": col}, "backgroundColor": info[0], "color": "white"}
         for col, info in _ADMIN_STYLE.items() if col in df.columns]
        + [{"if": {"column_id": c}, "backgroundColor": ACCENT, "color": "white"}
           for c in factor_cols]
        + [{"if": {"column_id": rc}, "backgroundColor": "#0ca678", "color": "white"}
           for rc in resp_cols]
    )
    data_cond = [
        {"if": {"row_index": "odd"}, "backgroundColor": "#f7f9fc"},
        *[{"if": {"column_id": col}, "fontWeight": "bold",
           "color": info[0], "fontStyle": info[1]}
          for col, info in _ADMIN_STYLE.items() if col in df.columns],
        *[{"if": {"column_id": rc}, "backgroundColor": "#f0fff8"}
          for rc in resp_cols],
    ]
    df_disp = df.copy()
    for c in df_disp.select_dtypes(include="number").columns:
        if c not in resp_set and c not in {"Std Order", "Run Order", "Block", "Replicate"}:
            df_disp[c] = df_disp[c].round(4)

    return dash_table.DataTable(
        id="analysis-datatable",
        columns=columns,
        data=df_disp.to_dict("records"),
        editable=True,
        page_size=20,
        sort_action="native",
        style_table={"overflowX": "auto"},
        style_header={"fontWeight": "bold", "fontSize": "0.82rem",
                      "textAlign": "center", "border": "none"},
        style_header_conditional=header_cond,
        style_cell={"fontSize": "0.82rem", "padding": "5px 12px",
                    "textAlign": "center", "border": "1px solid #dee2e6",
                    "minWidth": "80px"},
        style_data_conditional=data_cond,
    )


# B. Sync analysis-df when user edits the datatable
@app.callback(
    Output("analysis-df", "data", allow_duplicate=True),
    Input("analysis-datatable", "data"),
    State("response-cols-store", "data"),
    prevent_initial_call=True,
)
def sync_analysis_df(table_data, resp_cols):
    if not table_data:
        return no_update
    df = pd.DataFrame(table_data)
    # Coerce response columns to numeric
    for rc in (resp_cols or []):
        if rc in df.columns:
            df[rc] = pd.to_numeric(df[rc], errors="coerce")
    return df.to_json(orient="split")


# C. Update response dropdown, factor checklist, and surface dropdowns when analysis-df changes
@app.callback(
    Output("response-dropdown",          "options"),
    Output("response-dropdown",          "value"),
    Output("factor-checklist-container", "children"),
    Output("ia-factor-a",                "options"),
    Output("ia-factor-b",                "options"),
    Output("surface-fa",                 "options"),
    Output("surface-fa",                 "value"),
    Output("surface-fb",                 "options"),
    Output("surface-fb",                 "value"),
    Input("analysis-df",       "data"),
    State("response-cols-store","data"),
    prevent_initial_call=True,
)
def update_model_inputs(analysis_json, resp_cols):
    empty = ([], None,
             dbc.Checklist(id="factor-checklist", options=[], value=[]),
             [], [], [], None, [], None)
    if not analysis_json:
        return empty

    df          = pd.read_json(io.StringIO(analysis_json), orient="split")
    resp_cols   = resp_cols or []
    factor_cols = [c for c in df.columns
                   if c not in ADMIN_COLS and c not in resp_cols]

    resp_opts   = [{"label": rc, "value": rc} for rc in resp_cols if rc in df.columns]
    resp_val    = resp_cols[0] if resp_cols else None

    factor_opts = [{"label": fc, "value": fc} for fc in factor_cols]
    checklist   = dbc.Checklist(
        id="factor-checklist",
        options=factor_opts,
        value=factor_cols,
        inputStyle={"marginRight": "6px"},
    )
    val_a = factor_cols[0] if len(factor_cols) > 0 else None
    val_b = factor_cols[1] if len(factor_cols) > 1 else None
    return (resp_opts, resp_val, checklist,
            factor_opts, factor_opts,
            factor_opts, val_a, factor_opts, val_b)


# ── Shared helpers for table rendering ────────────────────────────────────────

def _fmt(val):
    if pd.isna(val):    return ""
    if isinstance(val, float):
        if abs(val) < 0.001: return f"{val:.3e}"
        return f"{val:.4f}"
    return str(val)


def _sig_color(p):
    if pd.isna(p): return None
    if p < 0.001:  return "#ffc9c9"
    if p < 0.05:   return "#fff3bf"
    return None


def _coef_table(coef_df):
    coef_data = [{k: _fmt(v) for k, v in r.items()} for r in coef_df.to_dict("records")]
    coef_colors = [
        {"if": {"row_index": i, "column_id": "p-value"},
         "backgroundColor": _sig_color(coef_df.iloc[i]["p-value"])}
        for i in range(len(coef_df))
        if _sig_color(coef_df.iloc[i]["p-value"])
    ]
    return dash_table.DataTable(
        columns=[{"name": c, "id": c} for c in coef_df.columns],
        data=coef_data,
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": ACCENT, "color": "white",
                      "fontWeight": "bold", "fontSize": "0.82rem", "textAlign": "center"},
        style_cell={"fontSize": "0.82rem", "padding": "4px 10px",
                    "textAlign": "center", "border": "1px solid #dee2e6"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#f7f9fc"},
            *coef_colors,
        ],
    )


def _eq_display(eqs):
    return dbc.Card(dbc.CardBody([
        html.H6("Regression Equation (Coded Variables)", className="fw-bold mb-1"),
        html.Pre(eqs["coded"],  className="bg-light p-2 rounded",
                 style={"fontSize": "0.83rem", "whiteSpace": "pre-wrap"}),
        html.H6("Actual-Unit Equation", className="fw-bold mb-1 mt-3"),
        html.Pre(eqs["actual"], className="bg-light p-2 rounded",
                 style={"fontSize": "0.83rem", "whiteSpace": "pre-wrap"}),
        html.Small("Coded variables: x = (X − center) / half-range",
                   className="text-muted"),
    ]), className="border-0", style={"background": "#f8f9fa"})


# C2. Build term-picker table when factor checklist or analysis-df changes
@app.callback(
    Output("term-picker-table", "data"),
    Input("factor-checklist",   "value"),
    State("analysis-df",        "data"),
    prevent_initial_call=True,
)
def build_term_picker(factor_cols, analysis_json):
    if not factor_cols:
        return []
    k = len(factor_cols)

    # Detect whether any factor has center points / 3+ levels
    has_curvature = False
    if analysis_json:
        try:
            df = pd.read_json(io.StringIO(analysis_json), orient="split")
            for fc in factor_cols:
                if fc in df.columns:
                    vals = df[fc].dropna().unique()
                    lo, hi = float(vals.min()), float(vals.max())
                    half = (hi - lo) / 2.0 if (hi - lo) != 0 else 1.0
                    coded = set(round((v - (lo + hi) / 2) / half, 6) for v in vals)
                    if not coded.issubset({-1.0, 1.0}):
                        has_curvature = True
                        break
        except Exception:
            pass

    rows = []
    # Generate all non-empty subsets in order: 1-way, 2-way, …, k-way
    for arity in range(1, k + 1):
        for combo in combinations(range(k), arity):
            term_str = " × ".join(factor_cols[i] for i in combo)
            rows.append({"Term": term_str, "In model": arity == 1,
                         "arity": arity, "disabled": False,
                         "_factors": [factor_cols[i] for i in combo]})

    # Add quadratic rows
    for fc in factor_cols:
        quad_label = f"{fc}²"
        rows.append({"Term": quad_label, "In model": False,
                     "arity": "quad", "disabled": not has_curvature,
                     "_factors": [fc]})

    return rows


# C3. Quick-select buttons update the term-picker checkboxes
@app.callback(
    Output("term-picker-table", "data", allow_duplicate=True),
    Input("qs-main-btn", "n_clicks"),
    Input("qs-twfi-btn", "n_clicks"),
    Input("qs-full-btn", "n_clicks"),
    State("term-picker-table", "data"),
    prevent_initial_call=True,
)
def quick_select(n_main, n_twfi, n_full, table_data):
    if not table_data:
        return no_update
    triggered = ctx.triggered_id
    updated = []
    for row in table_data:
        row = dict(row)
        if row.get("disabled"):
            row["In model"] = False
        elif triggered == "qs-main-btn":
            row["In model"] = (row["arity"] == 1)
        elif triggered == "qs-twfi-btn":
            row["In model"] = (row["arity"] in (1, 2))
        elif triggered == "qs-full-btn":
            row["In model"] = (row["arity"] != "quad")
        updated.append(row)
    return updated


# D. Fit model → populate ANOVA, stats, plots
@app.callback(
    Output("anova-table-div",    "children"),
    Output("model-stats-summary","children"),
    Output("pareto-plot",        "figure"),
    Output("main-effects-plot",  "figure"),
    Output("residual-plots",     "figure"),
    Output("fit-error",          "children"),
    Output("fit-info-store",     "data"),
    Input("fit-model-btn", "n_clicks"),
    State("analysis-df",         "data"),
    State("response-dropdown",   "value"),
    State("factor-checklist",    "value"),
    State("term-picker-table",   "data"),
    State("block-col-store",     "data"),
    prevent_initial_call=True,
)
def fit_and_display(n_clicks, analysis_json, response_col, factor_cols, term_table,
                    block_col_store):
    NU    = no_update
    EMPTY = go.Figure().update_layout(template="plotly_white")

    if not analysis_json or not response_col or not factor_cols:
        return (NU,) * 7

    # Derive custom_terms from the term-picker table
    custom_terms = None
    if term_table:
        selected = [row for row in term_table if row.get("In model") and not row.get("disabled")]
        if selected:
            custom_terms = []
            for row in selected:
                term_display = row["Term"]
                arity = row.get("arity")
                if arity == "quad":
                    # e.g. "A²" → "A^2"
                    factor_name = term_display.replace("²", "")
                    custom_terms.append(f"{factor_name}^2")
                else:
                    # e.g. "A × B × C" → "A*B*C"
                    custom_terms.append(term_display.replace(" × ", "*"))

    df = pd.read_json(io.StringIO(analysis_json), orient="split")
    df[response_col] = pd.to_numeric(df[response_col], errors="coerce")
    n_valid = df[response_col].notna().sum()
    if n_valid < 3:
        err = dbc.Alert(f"Need at least 3 response observations (got {n_valid}).",
                        color="warning")
        return NU, NU, EMPTY, EMPTY, EMPTY, err, NU

    try:
        fi = fit_model(df, response_col, list(factor_cols),
                       custom_terms=custom_terms,
                       block_col=block_col_store)
    except Exception as e:
        err = dbc.Alert(str(e), color="danger", dismissable=True)
        return NU, NU, EMPTY, EMPTY, EMPTY, err, NU

    # ── ANOVA table ───────────────────────────────────────────────────────────
    aov = get_anova_table(fi)
    aov_data   = [{k: _fmt(v) for k, v in row.items()} for row in aov.to_dict("records")]
    aov_colors = [
        {"if": {"row_index": i, "column_id": "p-value"},
         "backgroundColor": _sig_color(aov.iloc[i]["p-value"])}
        for i in range(len(aov))
        if _sig_color(aov.iloc[i]["p-value"])
    ]
    aov_table = dash_table.DataTable(
        columns=[{"name": c, "id": c} for c in aov.columns],
        data=aov_data,
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": ACCENT, "color": "white",
                      "fontWeight": "bold", "fontSize": "0.82rem", "textAlign": "center"},
        style_cell={"fontSize": "0.82rem", "padding": "4px 10px",
                    "textAlign": "center", "border": "1px solid #dee2e6"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#f7f9fc"},
            {"if": {"filter_query": "{Source} contains 'Model'"}, "fontWeight": "bold"},
            {"if": {"filter_query": "{Source} contains 'Total'"},
             "fontWeight": "bold", "borderTop": "2px solid #dee2e6"},
            *aov_colors,
        ],
    )

    # ── Model statistics ──────────────────────────────────────────────────────
    ms = get_model_stats(fi)

    def _stat_badge(label, val):
        text = f"{val:.4f}" if isinstance(val, float) and not np.isnan(val) else "—"
        return dbc.Col([
            html.Div(label, className="small text-muted mb-0"),
            html.Div(text,  className="fw-bold", style={"fontSize": "1.05rem"}),
        ], className="text-center border rounded p-2", style={"background": "#f8f9fa"})

    stats_card = dbc.Card(dbc.CardBody([
        html.H6("Model Summary", className="fw-bold mb-2"),
        dbc.Row([
            _stat_badge("R²", ms["R2"]), _stat_badge("Adj R²", ms["AdjR2"]),
            _stat_badge("Pred R²", ms["PredR2"]), _stat_badge("PRESS", ms["PRESS"]),
            _stat_badge("S (σ̂)", ms["S"]),
        ], className="g-2"),
        html.Small(f"n = {ms['n']}   df(Model) = {ms['df_mod']}   "
                   f"df(Residual) = {ms['df_res']}",
                   className="text-muted mt-2 d-block"),
    ]), className="border-0", style={"background": "#eef4ff"})

    fit_store = {"factor_cols": fi["factor_cols"], "response_col": fi["response_col"],
                 "custom_terms": custom_terms, "block_col": block_col_store,
                 "analysis_df": analysis_json}

    return (aov_table, stats_card,
            plot_pareto(fi), plot_main_effects(fi), plot_residuals(fi),
            None, json.dumps(fit_store))


# E. Lenth toggle → re-render half-normal plot from stored fit
@app.callback(
    Output("halfnormal-plot", "figure"),
    Input("lenth-lines-toggle", "value"),
    Input("fit-info-store",     "data"),
    State("analysis-df",        "data"),
    prevent_initial_call=True,
)
def update_halfnormal(show_lenth, fit_store_json, analysis_json):
    EMPTY = go.Figure().update_layout(template="plotly_white")
    if not fit_store_json or not analysis_json:
        return EMPTY
    store = json.loads(fit_store_json)
    df    = pd.read_json(io.StringIO(analysis_json), orient="split")
    rc    = store["response_col"]
    df[rc] = pd.to_numeric(df[rc], errors="coerce")
    try:
        fi = fit_model(df, rc, store["factor_cols"],
                       custom_terms=store.get("custom_terms"),
                       block_col=store.get("block_col"))
        return plot_half_normal(fi, lenth_lines=bool(show_lenth))
    except Exception:
        return EMPTY


# F. Interaction plot
@app.callback(
    Output("interaction-plot", "figure"),
    Input("plot-ia-btn",   "n_clicks"),
    State("ia-factor-a",   "value"),
    State("ia-factor-b",   "value"),
    State("fit-info-store","data"),
    State("analysis-df",   "data"),
    prevent_initial_call=True,
)
def update_interaction_plot(n, fa, fb, fit_store_json, analysis_json):
    if not fit_store_json or not analysis_json or not fa or not fb or fa == fb:
        return go.Figure().add_annotation(
            text="Select two different factors and click 'Plot Interaction'.",
            showarrow=False, font=dict(size=13, color="#6c757d"),
        ).update_layout(template="plotly_white")

    store = json.loads(fit_store_json)
    df    = pd.read_json(io.StringIO(analysis_json), orient="split")
    rc    = store["response_col"]
    df[rc] = pd.to_numeric(df[rc], errors="coerce")
    try:
        fi = fit_model(df, rc, store["factor_cols"], custom_terms=store.get("custom_terms"),
               block_col=store.get("block_col"))
        return plot_interaction(fi, fa, fb)
    except Exception as e:
        return go.Figure().add_annotation(text=str(e), showarrow=False)


# F. Rename response column
@app.callback(
    Output("analysis-table-container", "children",      allow_duplicate=True),
    Output("analysis-df",              "data",          allow_duplicate=True),
    Output("response-cols-store",      "data",          allow_duplicate=True),
    Output("response-dropdown",        "options",       allow_duplicate=True),
    Output("response-dropdown",        "value",         allow_duplicate=True),
    Input("rename-response-btn", "n_clicks"),
    State("response-rename-input", "value"),
    State("response-dropdown",     "value"),
    State("analysis-df",           "data"),
    State("response-cols-store",   "data"),
    prevent_initial_call=True,
)
def rename_response_col(n, new_name, old_name, analysis_json, resp_cols):
    NU = no_update
    if not n or not new_name or not new_name.strip() or not old_name or not analysis_json:
        return NU, NU, NU, NU, NU
    new_name = new_name.strip()
    if new_name == old_name:
        return NU, NU, NU, NU, NU
    df = pd.read_json(io.StringIO(analysis_json), orient="split")
    if old_name not in df.columns:
        return NU, NU, NU, NU, NU
    df        = df.rename(columns={old_name: new_name})
    resp_cols = [new_name if rc == old_name else rc for rc in (resp_cols or [])]
    factor_cols = [c for c in df.columns if c not in ADMIN_COLS and c not in set(resp_cols)]
    table     = _make_analysis_table(df, factor_cols, resp_cols)
    resp_opts = [{"label": rc, "value": rc} for rc in resp_cols]
    return table, df.to_json(orient="split"), resp_cols, resp_opts, new_name


# ── PREDICTION & OPTIMIZATION TAB CALLBACKS ───────────────────────────────────

# G. Populate coefficients + equation when fit-info-store changes
@app.callback(
    Output("pred-coefficients-div", "children"),
    Output("pred-equation-div",     "children"),
    Input("fit-info-store", "data"),
    State("analysis-df",    "data"),
    prevent_initial_call=True,
)
def populate_prediction_tab(fit_store_json, analysis_json):
    NU = no_update
    if not fit_store_json or not analysis_json:
        return NU, NU
    store = json.loads(fit_store_json)
    df    = pd.read_json(io.StringIO(analysis_json), orient="split")
    rc    = store["response_col"]
    df[rc] = pd.to_numeric(df[rc], errors="coerce")
    try:
        fi = fit_model(df, rc, store["factor_cols"], custom_terms=store.get("custom_terms"),
               block_col=store.get("block_col"))
    except Exception as e:
        err = dbc.Alert(str(e), color="danger")
        return err, err
    return _coef_table(get_coefficients(fi)), _eq_display(get_equations(fi))


# H. Update surface constants form when FA/FB selection changes
@app.callback(
    Output("surface-constants-container", "children"),
    Output("surf-const-factors",          "data"),
    Input("surface-fa",      "value"),
    Input("surface-fb",      "value"),
    State("fit-info-store",  "data"),
    State("analysis-df",     "data"),
    prevent_initial_call=True,
)
def update_surface_constants(fa, fb, fit_store_json, analysis_json):
    if not fit_store_json or not analysis_json or not fa or not fb:
        return html.Div(), []
    store = json.loads(fit_store_json)
    df    = pd.read_json(io.StringIO(analysis_json), orient="split")
    rc    = store["response_col"]
    df[rc] = pd.to_numeric(df[rc], errors="coerce")
    try:
        fi = fit_model(df, rc, store["factor_cols"], custom_terms=store.get("custom_terms"),
               block_col=store.get("block_col"))
    except Exception:
        return html.Div(), []

    other = [f for f in fi["factor_cols"] if f not in (fa, fb)]
    if not other:
        return html.Small("(no other factors)", className="text-muted"), []

    enc     = fi["encoding"]
    inputs  = []
    for i, fname in enumerate(other):
        e = enc[fname]
        inputs.append(html.Div([
            html.Small(f"{fname}:", className="text-muted fw-bold me-1"),
            dbc.Input(
                id={"type": "surf-const", "index": i},
                type="number", value=round(e["mid"], 4),
                min=e["low"], max=e["high"],
                size="sm", style={"width": "90px"},
                debounce=True,
            ),
        ], className="d-flex align-items-center"))
    return html.Div(inputs, className="d-flex flex-wrap gap-3"), other


# I. Run optimization
@app.callback(
    Output("optimization-results", "children"),
    Input("optimize-btn",     "n_clicks"),
    State("opt-goal",         "value"),
    State("opt-target-value", "value"),
    State("fit-info-store",   "data"),
    State("analysis-df",      "data"),
    prevent_initial_call=True,
)
def run_optimization_cb(n, goal, target_val, fit_store_json, analysis_json):
    if not n or not fit_store_json or not analysis_json:
        return no_update
    store = json.loads(fit_store_json)
    df    = pd.read_json(io.StringIO(analysis_json), orient="split")
    rc    = store["response_col"]
    df[rc] = pd.to_numeric(df[rc], errors="coerce")
    try:
        fi = fit_model(df, rc, store["factor_cols"], custom_terms=store.get("custom_terms"),
               block_col=store.get("block_col"))
        best_point, pred_val = optimize_response(fi, goal=goal,
                                                  target=float(target_val or 0))
    except Exception as e:
        return dbc.Alert(str(e), color="danger", dismissable=True)

    goal_label = {"maximize": "Maximum", "minimize": "Minimum",
                  "target": "Target"}.get(goal, goal)
    rows = [{"Factor": col,
             "Optimal Setting": f"{val:.4f}",
             "Range": f"[{fi['encoding'][col]['low']:.4f}, {fi['encoding'][col]['high']:.4f}]"}
            for col, val in best_point.items()]

    return dbc.Card(dbc.CardBody([
        dbc.Row([dbc.Col([
            html.Div(f"Predicted {goal_label} Response",
                     className="small text-muted mb-0"),
            html.Div(f"{pred_val:.4f}", className="fw-bold",
                     style={"fontSize": "1.6rem", "color": ACCENT}),
        ], className="text-center")], className="mb-3"),
        dash_table.DataTable(
            columns=[{"name": c, "id": c}
                     for c in ["Factor", "Optimal Setting", "Range"]],
            data=rows,
            style_header={"backgroundColor": ACCENT, "color": "white",
                          "fontWeight": "bold", "fontSize": "0.82rem",
                          "textAlign": "center"},
            style_cell={"fontSize": "0.82rem", "padding": "4px 10px",
                        "textAlign": "center", "border": "1px solid #dee2e6"},
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "#f7f9fc"}],
        ),
    ]), className="border-0 mt-2", style={"background": "#eef4ff"})


# J. Plot response surface (contour + 3D)
@app.callback(
    Output("contour-plot",    "figure"),
    Output("surface-3d-plot", "figure"),
    Input("plot-surface-btn", "n_clicks"),
    State("surface-fa",       "value"),
    State("surface-fb",       "value"),
    State({"type": "surf-const", "index": ALL}, "value"),
    State("surf-const-factors", "data"),
    State("fit-info-store",   "data"),
    State("analysis-df",      "data"),
    prevent_initial_call=True,
)
def plot_surface_cb(n, fa, fb, const_vals, const_factors,
                    fit_store_json, analysis_json):
    empty = go.Figure().update_layout(template="plotly_white",
                                       margin=dict(l=40, r=20, t=40, b=40))
    if not n or not fit_store_json or not analysis_json or not fa or not fb or fa == fb:
        return empty, empty

    store = json.loads(fit_store_json)
    df    = pd.read_json(io.StringIO(analysis_json), orient="split")
    rc    = store["response_col"]
    df[rc] = pd.to_numeric(df[rc], errors="coerce")
    try:
        fi = fit_model(df, rc, store["factor_cols"], custom_terms=store.get("custom_terms"),
               block_col=store.get("block_col"))
    except Exception as e:
        ann = go.Figure().add_annotation(text=str(e), showarrow=False)
        return ann, ann

    constants_actual = {fname: float(v)
                        for fname, v in zip(const_factors or [], const_vals or [])
                        if v is not None}

    try:
        X, Y, Z = get_surface_data(fi, fa, fb, constants_actual, n=50)
    except Exception as e:
        ann = go.Figure().add_annotation(text=str(e), showarrow=False)
        return ann, ann

    hold_str = "  |  ".join(f"{f} = {v:.4g}" for f, v in constants_actual.items())
    subtitle = f"Hold: {hold_str}" if hold_str else ""
    title_sfx = f"<br><sup style='color:#6c757d'>{subtitle}</sup>" if subtitle else ""

    # ── Contour ───────────────────────────────────────────────────────────────
    fig_c = go.Figure(go.Contour(
        x=X[0], y=Y[:, 0], z=Z,
        colorscale="RdBu_r",
        colorbar=dict(title=dict(text=rc, side="right")),
        contours=dict(showlabels=True, labelfont=dict(size=10)),
        hovertemplate=f"{fa}=%{{x:.3f}}<br>{fb}=%{{y:.3f}}<br>{rc}=%{{z:.4f}}<extra></extra>",
    ))
    fig_c.update_layout(
        title=dict(text=f"Contour — {rc}{title_sfx}", font_size=13),
        xaxis_title=fa, yaxis_title=fb,
        template="plotly_white", height=440,
        margin=dict(l=65, r=20, t=70, b=55),
    )

    # ── 3D surface ────────────────────────────────────────────────────────────
    fig_3 = go.Figure(go.Surface(
        x=X[0], y=Y[:, 0], z=Z,
        colorscale="RdBu_r",
        colorbar=dict(title=dict(text=rc, side="right")),
    ))
    fig_3.update_layout(
        title=dict(text=f"3D Surface — {rc}{title_sfx}", font_size=13),
        scene=dict(xaxis_title=fa, yaxis_title=fb, zaxis_title=rc),
        template="plotly_white", height=440,
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig_c, fig_3


# ═══════════════════════════════════════════════════════════════════════════════
# CALLBACKS — AGENT BRIDGE
# ═══════════════════════════════════════════════════════════════════════════════

@app.callback(
    Output("agent-config-store", "data"),
    Input("agent-poll", "n_intervals"),
    prevent_initial_call=True,
)
def poll_agent_config(n):
    """Read pending agent config and hand it to the Dash callback chain."""
    if _agent_pending_config:
        config = dict(_agent_pending_config)
        _agent_pending_config.clear()
        return config
    return dash.no_update


@app.callback(
    Output("results-table",   "children",  allow_duplicate=True),
    Output("design-df",       "data",      allow_duplicate=True),
    Output("error-alert",     "children",  allow_duplicate=True),
    Output("design-stats",    "children",  allow_duplicate=True),
    Output("active-design",   "data",      allow_duplicate=True),
    Output("factor-count",    "data",      allow_duplicate=True),
    Output("factors-container","children", allow_duplicate=True),
    Output("panel-fractional",      "style", allow_duplicate=True),
    Output("panel-ccd",             "style", allow_duplicate=True),
    Output("panel-box-behnken",     "style", allow_duplicate=True),
    Output("panel-taguchi",         "style", allow_duplicate=True),
    Output("panel-simplex-lattice", "style", allow_duplicate=True),
    Output("panel-center-points",   "style", allow_duplicate=True),
    Input("agent-config-store", "data"),
    prevent_initial_call=True,
)
def apply_agent_config(config):
    """Apply a design config pushed by the agent and generate the design matrix."""
    NU = no_update
    if not config:
        return (NU,) * 13

    design_type = config.get("design_type", "two_level_full")
    raw_factors = config.get("factors", [])
    options     = config.get("options", {})

    factors = []
    for f in raw_factors:
        name = f.get("name", "")
        if not name:
            continue
        try:
            lo_v = float(f.get("low",  0))
            hi_v = float(f.get("high", 1))
        except (TypeError, ValueError):
            lo_v, hi_v = 0.0, 1.0
        if lo_v >= hi_v:
            alert = dbc.Alert(f"Agent config: Factor '{name}' low >= high.", color="warning")
            return NU, NU, alert, NU, NU, NU, NU, NU, NU, NU, NU, NU, NU
        factors.append({"name": name, "low": lo_v, "high": hi_v,
                        "num_levels": int(f.get("num_levels", 2))})

    if not factors:
        alert = dbc.Alert("Agent config: no valid factors provided.", color="warning")
        return NU, NU, alert, NU, NU, NU, NU, NU, NU, NU, NU, NU, NU

    opts = dict(
        resolution=int(options.get("resolution", 3) or 3),
        generators=options.get("generators"),
        face=options.get("ccd_face", "ccc"),
        alpha=options.get("ccd_alpha", "orthogonal"),
        center_factorial=int(options.get("center_factorial", 4) or 4),
        center_star=int(options.get("center_star", 4) or 4),
        center=int(options.get("bb_center", 1) or 1),
        array_name=options.get("taguchi_array", "L8(2^7)"),
        degree=int(options.get("sl_degree", 2) or 2),
    )
    try:
        df_base = generate_design(design_type, factors, opts)
    except Exception as e:
        alert = dbc.Alert(f"Agent config generate error: {e}", color="danger", dismissable=True)
        return NU, NU, alert, NU, NU, NU, NU, NU, NU, NU, NU, NU, NU

    reps = max(1, int(options.get("replicates", 1) or 1))
    blks = max(1, int(options.get("blocks",     1) or 1))
    rand = bool(options.get("randomize", True))
    try:
        df, warning = apply_design_structure(df_base, reps, blks, rand)
    except Exception as e:
        alert = dbc.Alert(f"Agent config structure error: {e}", color="danger", dismissable=True)
        return NU, NU, alert, NU, NU, NU, NU, NU, NU, NU, NU, NU, NU

    factor_cols = [f["name"] for f in factors]

    n_cp = int(options.get("center_points", 0) or 0)
    if n_cp > 0 and design_type in _CENTER_POINT_DESIGNS:
        mid = {f["name"]: (float(f["low"]) + float(f["high"])) / 2.0 for f in factors}
        unique_blocks = list(dict.fromkeys(df["Block"].tolist()))
        cp_all = []
        std_counter = len(df) + 1
        for blk in unique_blocks:
            rep_val = df.loc[df["Block"] == blk, "Replicate"].iloc[-1]
            for _ in range(n_cp):
                row = {fname: mid[fname] for fname in factor_cols}
                row["Std Order"]  = std_counter
                row["Run Order"]  = std_counter
                row["Block"]      = blk
                row["Replicate"]  = rep_val
                row["Point Type"] = "Center"
                cp_all.append(row)
                std_counter += 1
        df["Point Type"] = "Factorial"
        df = pd.concat([df, pd.DataFrame(cp_all)], ignore_index=True)
        admin_present = [c for c in ("Std Order", "Run Order", "Block", "Replicate")
                         if c in df.columns]
        df = df[admin_present + factor_cols + ["Point Type"]]

    df_disp = df.copy()
    df_disp[factor_cols] = df_disp[factor_cols].round(4)
    table = _build_design_table(df_disp, factor_cols)

    n_runs = len(df)
    stats_row = dbc.Row([
        dbc.Col(dbc.Badge(f"{n_runs} total runs",  color="primary",   pill=True), width="auto"),
        dbc.Col(dbc.Badge(f"{len(df_base)} runs/replicate", color="info", pill=True), width="auto"),
        dbc.Col(dbc.Badge(f"{reps} replicate{'s' if reps>1 else ''}",
                          color="warning", text_color="dark", pill=True), width="auto"),
        dbc.Col(dbc.Badge(f"{blks} block{'s' if blks>1 else ''}",
                          color="success", pill=True), width="auto"),
        dbc.Col(dbc.Badge("Randomized" if rand else "Standard order",
                          color="secondary", pill=True), width="auto"),
    ], className="g-1")

    alert = dbc.Alert(warning, color="warning", dismissable=True) if warning else None

    # Rebuild factor rows to reflect agent config in the Design tab
    factor_rows = [make_factor_row(i, f["name"], f["low"], f["high"])
                   for i, f in enumerate(factors)]

    # Panel visibility — same logic as select_design
    show, hide = {"display": "block"}, {"display": "none"}
    all_panels = set(_PANEL_IDS.values()) | {"panel-center-points"}
    panel_styles = {p: hide for p in all_panels}
    if (pid := _PANEL_IDS.get(design_type)):
        panel_styles[pid] = show
    if design_type in _CENTER_POINT_DESIGNS:
        panel_styles["panel-center-points"] = show

    # Also update readable state with design info (for agent state polling)
    _agent_readable_state.update({
        "design_type":  design_type,
        "factor_count": len(factors),
        "factors":      factors,
    })

    return (
        table,
        df.to_json(orient="split"),
        alert,
        stats_row,
        design_type,
        len(factors),
        factor_rows,
        panel_styles["panel-fractional"],
        panel_styles["panel-ccd"],
        panel_styles["panel-box-behnken"],
        panel_styles["panel-taguchi"],
        panel_styles["panel-simplex-lattice"],
        panel_styles["panel-center-points"],
    )


@app.callback(
    Output("agent-state-sync", "data"),
    Input("fit-info-store",    "data"),
    State("active-design",     "data"),
    State("factor-count",      "data"),
    prevent_initial_call=True,
)
def sync_state_to_agent(fit_info, active_design, factor_count):
    """Expose fitted model results to the agent via _agent_readable_state."""
    if fit_info:
        store = json.loads(fit_info) if isinstance(fit_info, str) else fit_info
        _agent_readable_state.update({
            "design_type":  active_design,
            "factor_count": factor_count,
            "anova":        store.get("anova"),
            "model_stats":  store.get("model_stats"),
            "terms":        store.get("custom_terms"),
            "response_col": store.get("response_col"),
        })
    return dash.no_update


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=8050)
