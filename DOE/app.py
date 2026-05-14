"""
DOE Designer — Plotly Dash GUI  (Design + Analysis + AI Assistant)
Run:  python app.py  → http://127.0.0.1:8050
"""

import io, base64, json, uuid, os, sys
from dotenv import load_dotenv

# Ensure repo root is in sys.path so `agent` package is importable when
# the script is launched as  python DOE/app.py  from the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Load .env from repo root before any agent/API imports
load_dotenv(dotenv_path=os.path.join(_REPO_ROOT, ".env"), override=True)

from itertools import combinations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html, dash_table, Input, Output, State, ctx, ALL, MATCH, no_update
import dash_bootstrap_components as dbc
from flask import request as flask_request, jsonify

from doe_generators  import generate_design, list_taguchi_arrays, apply_design_structure
from analysis import (
    ADMIN_COLS, fit_model, get_anova_table, get_coefficients, get_model_stats,
    get_equations, get_residuals, plot_pareto, plot_half_normal,
    plot_residuals, plot_main_effects, plot_interaction,
    predict_response, get_surface_data, optimize_response,
    get_residual_stats, fig_to_b64,
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

# Interviewer session registry — keyed by browser-tab UUID stored in session-id Store
from agent.interviewer  import Interviewer
from agent.interpreter  import Interpreter
from agent.recommender  import recommend_next_step

_sessions: dict[str, Interviewer] = {}


def _get_interviewer(session_id: str) -> Interviewer:
    if session_id not in _sessions:
        _sessions[session_id] = Interviewer()
    return _sessions[session_id]


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
    "Std Order":  ("#6c757d", "italic"),
    "Run Order":  (ACCENT,    "normal"),
    "Block":      ("#0ca678", "normal"),
    "Replicate":  ("#e67700", "normal"),
    "Point Type": ("#6f42c1", "normal"),   # purple — CCD / design-structure label
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

def make_factor_row(idx, name="", low=-1, high=1, num_levels=2, ftype="numeric", cat_levels=""):
    is_num = ftype == "numeric"
    return dbc.Row([
        dbc.Col(dbc.Input(id={"type": "f-name", "index": idx}, value=name,
                          placeholder=f"Factor {idx+1}", debounce=True, size="sm"), width=3),
        dbc.Col(dbc.Select(id={"type": "f-type", "index": idx},
                           options=[{"label": "Numeric",   "value": "numeric"},
                                    {"label": "Categoric", "value": "categoric"}],
                           value=ftype, size="sm"), width=2),
        dbc.Col(dbc.Input(id={"type": "f-low",  "index": idx}, value=low,
                          placeholder="Low",  type="number", debounce=True, size="sm"),
                id={"type": "f-low-col",  "index": idx}, width=2,
                style={} if is_num else {"display": "none"}),
        dbc.Col(dbc.Input(id={"type": "f-high", "index": idx}, value=high,
                          placeholder="High", type="number", debounce=True, size="sm"),
                id={"type": "f-high-col", "index": idx}, width=2,
                style={} if is_num else {"display": "none"}),
        dbc.Col(dbc.Input(id={"type": "f-levels-cat", "index": idx}, value=cat_levels,
                          placeholder="A, B, C", debounce=True, size="sm"),
                id={"type": "f-cat-col",  "index": idx}, width=4,
                style={"display": "none"} if is_num else {}),
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
        style_table={"overflowX": "auto", "maxHeight": "400px", "overflowY": "auto"},
        style_header={"fontWeight": "bold", "fontSize": "0.82rem",
                      "textAlign": "center", "border": "none"},
        style_header_conditional=header_cond,
        style_cell={"fontSize": "0.82rem", "padding": "5px 12px",
                    "textAlign": "center", "border": "1px solid #dee2e6"},
        style_data_conditional=data_cond,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# AI ASSISTANT PANEL (embedded in Design tab — must be defined before design_tab)
# ═══════════════════════════════════════════════════════════════════════════════

agent_tab = dbc.Container([
    dbc.Row([
        # Chat column
        dbc.Col([
            html.Div(
                id="agent-chat-display",
                style={
                    "height": "60vh",
                    "overflowY": "scroll",
                    "border": "1px solid #dee2e6",
                    "padding": "1rem",
                    "borderRadius": "4px",
                    "background": "#fafbfc",
                },
            ),
            dbc.Row([
                dbc.Col(
                    dbc.Input(
                        id="agent-chat-input",
                        placeholder="Describe your experiment or answer the question above…",
                        type="text",
                        debounce=False,
                        n_submit=0,
                    ),
                    width=9,
                ),
                dbc.Col(
                    dbc.Button(
                        [html.I(className="bi bi-send me-1"), "Send"],
                        id="agent-send-btn",
                        color="primary",
                        n_clicks=0,
                        className="w-100",
                    ),
                    width=3,
                ),
            ], className="mt-2 g-2"),
            html.Div([
                html.Hr(className="my-2"),
                html.Small(
                    "Once the assistant has recommended a design, click below to "
                    "populate the Design tab automatically.",
                    className="text-muted d-block mb-2",
                ),
                dbc.Button(
                    [html.I(className="bi bi-gear-fill me-2"),
                     "Apply Recommended Design →"],
                    id="agent-configure-btn",
                    color="success",
                    className="w-100",
                    n_clicks=0,
                    disabled=True,
                ),
            ], id="agent-configure-panel", style={"display": "none"}),
        ], md=8),

        # Info sidebar
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.Strong("About this assistant")),
                dbc.CardBody([
                    html.P(
                        "This AI consultant guides you through 7 questions and "
                        "recommends the best experimental design for your situation, "
                        "grounded in the NIST Statistical Handbook.",
                        className="small",
                    ),
                    html.P(
                        "Once a recommendation appears, click "
                        "'Configure App from Recommendation' to automatically "
                        "populate the Design tab.",
                        className="small",
                    ),
                    html.Hr(),
                    html.H6("Available design types:", className="small fw-bold"),
                    html.Ul([
                        html.Li("2-Level Full / Fractional Factorial", className="small"),
                        html.Li("Plackett-Burman (screening)", className="small"),
                        html.Li("CCD / Box-Behnken (response surface)", className="small"),
                        html.Li("General Factorial", className="small"),
                        html.Li("Taguchi orthogonal arrays", className="small"),
                        html.Li("Simplex Lattice / Centroid (mixtures)", className="small"),
                    ]),
                    html.Hr(),
                    html.P(
                        html.Small(
                            "Powered by claude-sonnet-4-6.",
                            className="text-muted",
                        )
                    ),
                ]),
            ], style={"boxShadow": CARD_SH}),
        ], md=4),
    ], className="mt-3 g-3"),
], fluid=True)


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
                        dbc.Col(html.Small("Name",         className="text-muted fw-bold"), width=3),
                        dbc.Col(html.Small("Type",         className="text-muted fw-bold"), width=2),
                        dbc.Col(html.Small("Low / Levels", className="text-muted fw-bold"), width=4),
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
    html.Hr(className="my-4"),
    dbc.Row([
        dbc.Col([
            dbc.Button(
                [html.I(className="bi bi-robot me-2"), "🤖 AI Design Assistant"],
                id="toggle-agent-panel-btn",
                color="outline-primary", outline=True,
                className="w-100 mb-2", n_clicks=0,
            ),
            dbc.Collapse(
                agent_tab,
                id="agent-panel-collapse",
                is_open=False,
            ),
        ])
    ], className="g-3"),
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
            dbc.Col([
                dbc.InputGroup([
                    dbc.Input(id="response-rename-input", placeholder="Rename response…",
                              debounce=True, size="sm"),
                    dbc.Button("Rename", id="rename-response-btn",
                               size="sm", color="outline-secondary", n_clicks=0),
                ], size="sm"),
                html.Small("Rename selected response column", className="text-muted d-block mt-1"),
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
                     clearable=False, className="mb-3"),

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
            dbc.Button("+ Quadratic",   id="qs-quad-btn", size="sm", color="outline-secondary",
                       n_clicks=0, className="flex-fill",
                       title="Select all main effects, 2FI, and quadratic terms (RSM)"),
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
                    "options": [{"label": "✓", "value": "yes"},
                                {"label": "—", "value": "no"}]
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
], style={"boxShadow": CARD_SH, "position": "sticky", "top": "1rem"})

# ── Helper: AI interpretation result card ─────────────────────────────────────
def _interp_card(markdown_text: str) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(dcc.Markdown(markdown_text, style={"fontSize": "0.88rem"})),
        style={"borderLeft": "4px solid #17a2b8", "background": "#f8fffe", "marginTop": "12px"},
    )


def _interp_btn(btn_id: str, label: str = "🤖 AI Interpret") -> dbc.Button:
    return dbc.Button(
        label, id=btn_id, color="outline-info", size="sm",
        className="mt-3 w-100", n_clicks=0,
        disabled=True,                          # enabled once fit-info-store is set
    )


# ── Section 3: Results accordion (right panel) ───────────────────────────────
results_accordion = dbc.Accordion([

    # ── 0: Design Summary ────────────────────────────────────────────────────
    dbc.AccordionItem([
        html.Div(id="design-summary-div",
                 children=html.Div("Load a design to see the summary.",
                                   className="text-muted text-center py-3")),
        _interp_btn("interp-design-btn", "🤖 AI Interpretation: Design Summary"),
        dcc.Loading(html.Div(id="interp-design-div"), type="dot", color="#17a2b8"),
    ], title="Design Summary", item_id="design-summary"),

    # ── 1: ANOVA Table ───────────────────────────────────────────────────────
    dbc.AccordionItem([
        html.Div(id="anova-table-div",
                 children=html.Div("Fit a model to see the ANOVA table.",
                                   className="text-muted text-center py-3")),
        _interp_btn("interp-anova-btn", "🤖 AI Interpretation: ANOVA & Model Stats"),
        dcc.Loading(html.Div(id="interp-anova-div"), type="dot", color="#17a2b8"),
    ], title="ANOVA Table", item_id="anova"),

    # ── 2: Effects & Interaction Analysis ────────────────────────────────────
    dbc.AccordionItem([
        html.Small("Effect Estimates", className="text-muted fw-bold d-block mb-2"),
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
        html.Small("Interaction Plot", className="text-muted fw-bold d-block mb-2"),
        dbc.Row([
            dbc.Col([dbc.Label("Factor A", className="small"),
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
        _interp_btn("interp-effects-btn", "🤖 AI Interpretation: Effects & Interactions"),
        dcc.Loading(html.Div(id="interp-effects-div"), type="dot", color="#17a2b8"),
    ], title="Effects and Interaction Analysis (Pareto · Half-Normal · Main Effects · Interactions)",
       item_id="effects"),

    # ── 3: Residual Analysis ─────────────────────────────────────────────────
    dbc.AccordionItem([
        dcc.Graph(id="residual-plots", config={"displayModeBar": True}),
        _interp_btn("interp-resid-btn", "🤖 AI Interpretation: Residual Analysis"),
        dcc.Loading(html.Div(id="interp-resid-div"), type="dot", color="#17a2b8"),
    ], title="Residual Analysis — Model Adequacy Checking", item_id="residuals"),

], id="results-accordion", always_open=True, active_item=["design-summary", "anova", "effects"])

analysis_tab = dbc.Container([
    html.Div(id="block-detect-banner"),
    dbc.Row(dbc.Col(data_entry_card), className="mt-3"),
    dbc.Row([
        dbc.Col(model_setup_card,  md=3),
        dbc.Col([
            results_accordion,
            dbc.Button(
                [html.I(className="bi bi-file-earmark-code me-2"), "Save Report as HTML"],
                id="save-html-btn", color="outline-secondary",
                className="w-100 mt-3", n_clicks=0, disabled=True,
            ),
            dcc.Download(id="download-html"),
        ], md=9),
    ], className="mt-3 g-3"),
], fluid=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION & OPTIMIZATION TAB LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════

prediction_tab = dbc.Container([
    # Top: Coefficients + Equation
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.Strong("Regression Coefficients")),
            dbc.CardBody(html.Div(id="pred-coefficients-div",
                                  children=html.Div("Fit a model on the Analysis tab first.",
                                                    className="text-muted text-center py-3"))),
        ], style={"boxShadow": CARD_SH}), md=7),
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.Strong("Model Equation")),
            dbc.CardBody(html.Div(id="pred-equation-div")),
        ], style={"boxShadow": CARD_SH}), md=5),
    ], className="mt-3 g-3"),

    # Bottom: Controls left, Surface right
    dbc.Row([
        # Left: Optimization controls (sticky)
        dbc.Col(dbc.Card([
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
                            value="maximize", className="mb-2",
                        ),
                    ], width=6),
                    dbc.Col([
                        dbc.Label("Target Value", className="small fw-bold mb-1"),
                        dbc.Input(id="opt-target-value", type="number", size="sm",
                                  placeholder="Enter target…"),
                        html.Small("(used when goal = Target)", className="text-muted d-block mt-1"),
                    ], width=6),
                ], className="g-2"),
                dbc.Button(
                    [html.I(className="bi bi-bullseye me-2"), "Run Optimization"],
                    id="optimize-btn", color="primary",
                    className="w-100 mt-2", n_clicks=0,
                ),
                dcc.Loading(
                    html.Div(id="optimization-results", className="mt-2"),
                    type="circle", color=ACCENT,
                ),
            ]),
        ], style={"boxShadow": CARD_SH, "position": "sticky", "top": "1rem"}), md=4),

        # Right: Response surface
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.Strong("Response Surface Plots")),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Factor X", className="small fw-bold mb-0"),
                        dcc.Dropdown(id="surface-fa", clearable=False, placeholder="Select…"),
                    ], md=3),
                    dbc.Col([
                        dbc.Label("Factor Y", className="small fw-bold mb-0"),
                        dcc.Dropdown(id="surface-fb", clearable=False, placeholder="Select…"),
                    ], md=3),
                    dbc.Col([
                        dbc.Label("Other factors held at:", className="small fw-bold mb-0"),
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
                    dbc.Col(dcc.Graph(id="contour-plot", config={"displayModeBar": True},
                                     style={"minHeight": "420px"}), md=6),
                    dbc.Col(dcc.Graph(id="surface-3d-plot", config={"displayModeBar": True},
                                     style={"minHeight": "420px"}), md=6),
                ]),
            ]),
        ], style={"boxShadow": CARD_SH}), md=8),
    ], className="mt-3 g-3"),
    dbc.Row(
        dbc.Col([
            dbc.Button(
                [html.I(className="bi bi-file-earmark-code me-2"), "Save Report as HTML"],
                id="save-html-pred-btn", color="outline-secondary",
                className="w-100 mt-2", n_clicks=0, disabled=True,
            ),
            dcc.Download(id="download-html-pred"),
        ]), className="mt-2 g-3",
    ),
], fluid=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════

main = html.Div([
    dbc.Tabs([
        dbc.Tab(design_tab,     label="⊞ Design",                    tab_id="tab-design",
                label_style={"fontWeight": 600}),
        dbc.Tab(analysis_tab,   label="📊 Analysis",                  tab_id="tab-analysis",
                label_style={"fontWeight": 600}),
        dbc.Tab(prediction_tab, label="🎯 Prediction & Optimization", tab_id="tab-prediction",
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
    # Per-section AI interpretation text (raw markdown for HTML export)
    dcc.Store(id="interp-design-store",  data=""),
    dcc.Store(id="interp-anova-store",   data=""),
    dcc.Store(id="interp-effects-store", data=""),
    dcc.Store(id="interp-resid-store",   data=""),
    # Agent bridge stores
    dcc.Store(id="agent-config-store",  storage_type="memory"),
    dcc.Store(id="agent-state-sync",    storage_type="memory"),
    dcc.Store(id="session-id",          storage_type="memory", data=str(uuid.uuid4())),
    # Chat history — survives callback re-invocations within the same browser session
    dcc.Store(id="agent-chat-history",  storage_type="memory", data=[]),
    dcc.Store(id="agent-pending-input", storage_type="memory"),
    # Sidebar help stores
    dcc.Store(id="help-chat-history",   storage_type="memory", data=[]),

    # Floating "?" help button
    dbc.Button(
        "?",
        id="help-toggle",
        color="primary",
        n_clicks=0,
        style={
            "position":     "fixed",
            "bottom":       "2rem",
            "right":        "2rem",
            "zIndex":       1000,
            "borderRadius": "50%",
            "width":        "3rem",
            "height":       "3rem",
            "fontSize":     "1.2rem",
            "lineHeight":   "1",
            "padding":      "0",
            "boxShadow":    "0 4px 12px rgba(0,0,0,0.25)",
        },
    ),

    # Sidebar help Offcanvas
    dbc.Offcanvas(
        [
            html.P(
                "Ask anything about the current tab. "
                "I have context about your active design and model.",
                className="small text-muted mb-2",
            ),
            html.Div(
                id="help-chat-display",
                style={
                    "height":     "55vh",
                    "overflowY":  "scroll",
                    "border":     "1px solid #dee2e6",
                    "padding":    "0.75rem",
                    "borderRadius": "4px",
                    "background": "#fafbfc",
                    "fontSize":   "0.84rem",
                },
            ),
            dbc.Row([
                dbc.Col(
                    dbc.Input(
                        id="help-chat-input",
                        placeholder="Ask about this tab…",
                        type="text",
                        size="sm",
                        n_submit=0,
                    ),
                    width=9,
                ),
                dbc.Col(
                    dbc.Button(
                        "Ask", id="help-send-btn",
                        color="primary", size="sm",
                        n_clicks=0, className="w-100",
                    ),
                    width=3,
                ),
            ], className="mt-2 g-1"),
            html.Small(
                "Powered by claude-haiku-4-5 · context-aware for active tab",
                className="text-muted d-block mt-1",
            ),
        ],
        id="help-panel",
        title="Quick Help",
        placement="end",
        is_open=False,
        style={"width": "400px"},
    ),
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


# 2b. Toggle factor type UI (numeric vs categoric)
@app.callback(
    Output({"type": "f-low-col",  "index": MATCH}, "style"),
    Output({"type": "f-high-col", "index": MATCH}, "style"),
    Output({"type": "f-cat-col",  "index": MATCH}, "style"),
    Input({"type":  "f-type",     "index": MATCH}, "value"),
    prevent_initial_call=True,
)
def toggle_factor_type_ui(ftype):
    is_num = ftype == "numeric"
    return ({} if is_num else {"display": "none"},
            {} if is_num else {"display": "none"},
            {"display": "none"} if is_num else {})


# 3. Generate design
@app.callback(
    Output("results-table",   "children"),
    Output("design-df",       "data"),
    Output("error-alert",     "children"),
    Output("design-stats",    "children"),
    Input("generate-btn", "n_clicks"),
    State("active-design",     "data"),
    State({"type": "f-name",       "index": ALL}, "value"),
    State({"type": "f-low",        "index": ALL}, "value"),
    State({"type": "f-high",       "index": ALL}, "value"),
    State({"type": "f-type",       "index": ALL}, "value"),
    State({"type": "f-levels-cat", "index": ALL}, "value"),
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
             names, lows, highs, ftypes, cat_levels_strs,
             frac_res, frac_gen, ccd_face, ccd_alpha, ccd_cf, ccd_cs,
             bb_center, taguchi_arr, sl_degree,
             n_replicates, n_blocks, randomize_val,
             n_center_points):
    NU = no_update
    factors = []
    for name, ftype, lo, hi, cat_lvls_str in zip(names, ftypes, lows, highs, cat_levels_strs):
        if not name:
            continue
        if ftype == "categoric":
            levels = [l.strip() for l in (cat_lvls_str or "A,B").split(",") if l.strip()]
            if len(levels) < 2:
                return NU, NU, dbc.Alert(f"Factor '{name}': provide at least 2 categoric levels.", color="warning"), NU
            factors.append({"name": name, "type": "categoric", "levels": levels,
                            "num_levels": len(levels)})
        else:
            try:
                lo_v, hi_v = float(lo or 0), float(hi or 1)
            except (TypeError, ValueError):
                lo_v, hi_v = 0.0, 1.0
            if lo_v >= hi_v:
                return NU, NU, dbc.Alert(f"Factor '{name}': Low ≥ High.", color="warning"), NU
            factors.append({"name": name, "type": "numeric", "low": lo_v, "high": hi_v,
                            "num_levels": 2})
    if not factors:
        return NU, NU, dbc.Alert("Add at least one factor.", color="warning"), NU

    CAT_UNSUPPORTED = {"ccd", "box_behnken", "simplex_lattice", "simplex_centroid", "plackett_burman"}
    has_cat = any(f.get("type") == "categoric" for f in factors)
    if has_cat and design_type in CAT_UNSUPPORTED:
        return NU, NU, dbc.Alert(
            "Categorical factors are not supported for this design type. "
            "Use General Factorial or Two-Level Factorial.", color="warning"
        ), NU

    opts = dict(
        resolution=int(frac_res or 3), generators=frac_gen or None,
        face=ccd_face or "ccc", alpha=ccd_alpha or "orthogonal",
        center_factorial=(4 if ccd_cf is None else int(ccd_cf)),
        center_star=(4 if ccd_cs is None else int(ccd_cs)),
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
        # Midpoint for each numeric factor in actual (uncoded) units
        mid = {f["name"]: (float(f["low"]) + float(f["high"])) / 2.0
               for f in factors if f.get("type") == "numeric"}
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
    numeric_factor_cols = [f["name"] for f in factors if f.get("type") != "categoric"]
    if numeric_factor_cols:
        df_disp[numeric_factor_cols] = df_disp[numeric_factor_cols].round(4)

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
    State("analysis-df", "data"),
    prevent_initial_call=True,
)
def sync_analysis_df(table_data, resp_cols, prev_json):
    if not table_data:
        return no_update
    df = pd.DataFrame(table_data)
    # Prevent paste from adding extra rows beyond the original design
    if prev_json:
        try:
            original_n = len(pd.read_json(io.StringIO(prev_json), orient="split"))
            if len(df) > original_n:
                df = df.iloc[:original_n].copy()
        except Exception:
            pass
    # Coerce response columns to numeric
    for rc in (resp_cols or []):
        if rc in df.columns:
            df[rc] = pd.to_numeric(df[rc], errors="coerce")
    return df.to_json(orient="split")


# C. Update response dropdown, factor checklist, surface dropdowns, and
#    design-summary-div when analysis-df changes
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
    Output("design-summary-div",         "children"),
    Input("analysis-df",       "data"),
    State("response-cols-store","data"),
    State("active-design",     "data"),
    State("block-col-store",   "data"),
    prevent_initial_call=True,
)
def update_model_inputs(analysis_json, resp_cols, active_design, block_col):
    empty_summary = html.Div("Load a design to see the summary.",
                             className="text-muted text-center py-3")
    empty = ([], None,
             dbc.Checklist(id="factor-checklist", options=[], value=[]),
             [], [], [], None, [], None, empty_summary)
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

    # ── Design summary card ──────────────────────────────────────────────────
    design_label = (active_design or "Unknown").replace("_", " ").title()
    n_runs  = len(df)
    n_blocks = df["Block"].nunique()     if "Block"     in df.columns else 1
    n_reps   = df["Replicate"].nunique() if "Replicate" in df.columns else 1
    pt_counts = {}
    if "Point Type" in df.columns:
        pt_counts = df["Point Type"].value_counts().to_dict()

    # Factor table rows
    fac_rows = []
    for fc in factor_cols:
        is_num = pd.api.types.is_numeric_dtype(df[fc])
        fac_rows.append(html.Tr([
            html.Td(fc, style={"fontWeight": "600"}),
            html.Td("Numeric" if is_num else "Categoric",
                    className="text-muted", style={"fontSize": "0.8rem"}),
            html.Td(f"{df[fc].min():.4g}" if is_num else str(df[fc].unique().tolist()),
                    style={"fontSize": "0.82rem"}),
            html.Td(f"{df[fc].max():.4g}" if is_num else "—",
                    style={"fontSize": "0.82rem"}),
        ]))
    fac_table = dbc.Table(
        [html.Thead(html.Tr([html.Th("Factor"), html.Th("Type"),
                             html.Th("Low / Levels"), html.Th("High")])),
         html.Tbody(fac_rows)],
        size="sm", hover=True, className="mb-2",
        style={"fontSize": "0.82rem"},
    )
    badges = dbc.Stack([
        dbc.Badge(f"Design: {design_label}", color="primary",   className="me-1"),
        dbc.Badge(f"{n_runs} runs",          color="info",      className="me-1",
                  text_color="white"),
        dbc.Badge(f"{n_blocks} block(s)",    color="success",   className="me-1"),
        dbc.Badge(f"{n_reps} replicate(s)",  color="warning",   className="me-1",
                  text_color="dark"),
        *[dbc.Badge(f"{v} {k}", color="secondary", className="me-1")
          for k, v in pt_counts.items()],
    ], direction="horizontal", gap=1, className="flex-wrap mb-2")

    design_summary = html.Div([badges, fac_table])

    return (resp_opts, resp_val, checklist,
            factor_opts, factor_opts,
            factor_opts, val_a, factor_opts, val_b,
            design_summary)


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
    Input("analysis-df",        "data"),
    prevent_initial_call=True,
)
def build_term_picker(factor_cols, analysis_json):
    if not factor_cols:
        return []
    k = len(factor_cols)

    # Detect categoric vs numeric columns and curvature
    cat_cols = set()
    has_curvature = False
    if analysis_json:
        try:
            df = pd.read_json(io.StringIO(analysis_json), orient="split")
            for fc in factor_cols:
                if fc in df.columns:
                    if not pd.api.types.is_numeric_dtype(df[fc]):
                        cat_cols.add(fc)
                    else:
                        vals = df[fc].dropna().unique()
                        lo, hi = float(vals.min()), float(vals.max())
                        half = (hi - lo) / 2.0 if (hi - lo) != 0 else 1.0
                        coded = set(round((v - (lo + hi) / 2) / half, 6) for v in vals)
                        if not coded.issubset({-1.0, 1.0}):
                            has_curvature = True
        except Exception:
            pass

    def term_label(cols):
        parts = []
        for c in cols:
            parts.append(f"C({c})" if c in cat_cols else c)
        return " × ".join(parts)

    rows = []
    # Generate all non-empty subsets in order: 1-way, 2-way, …, k-way
    for arity in range(1, k + 1):
        for combo in combinations(range(k), arity):
            cols = [factor_cols[i] for i in combo]
            term_str = term_label(cols)
            rows.append({"Term": term_str, "In model": "yes" if arity == 1 else "no",
                         "arity": arity, "disabled": False})

    # Add quadratic rows only for numeric factors
    for fc in factor_cols:
        if fc not in cat_cols:
            quad_label = f"{fc}²"
            rows.append({"Term": quad_label, "In model": "no",
                         "arity": "quad", "disabled": not has_curvature})

    return rows


# C3. Quick-select buttons update the term-picker checkboxes
@app.callback(
    Output("term-picker-table", "data", allow_duplicate=True),
    Input("qs-main-btn",  "n_clicks"),
    Input("qs-twfi-btn",  "n_clicks"),
    Input("qs-full-btn",  "n_clicks"),
    Input("qs-quad-btn",  "n_clicks"),
    State("term-picker-table", "data"),
    prevent_initial_call=True,
)
def quick_select(n_main, n_twfi, n_full, n_quad, table_data):
    if not table_data:
        return no_update
    triggered = ctx.triggered_id
    updated = []
    for row in table_data:
        row = dict(row)
        if row.get("disabled"):
            # truly unavailable (e.g. quad when no curvature) — always off
            row["In model"] = "no"
        elif triggered == "qs-main-btn":
            row["In model"] = "yes" if row["arity"] == 1 else "no"
        elif triggered == "qs-twfi-btn":
            row["In model"] = "yes" if row["arity"] in (1, 2) else "no"
        elif triggered == "qs-full-btn":
            row["In model"] = "yes" if row["arity"] != "quad" else "no"
        elif triggered == "qs-quad-btn":
            # Main effects + 2FI + pure quadratic (standard RSM model)
            row["In model"] = "yes" if row["arity"] in (1, 2, "quad") else "no"
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
    Output("interp-design-btn",  "disabled"),
    Output("interp-anova-btn",   "disabled"),
    Output("interp-effects-btn", "disabled"),
    Output("interp-resid-btn",      "disabled"),
    Output("save-html-btn",         "disabled"),
    Output("save-html-pred-btn",    "disabled"),
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
        return (NU,) * 13

    # Derive custom_terms from the term-picker table
    custom_terms = None
    if term_table:
        selected = [row for row in term_table if row.get("In model") == "yes" and not row.get("disabled")]
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
        return NU, NU, EMPTY, EMPTY, EMPTY, err, NU, NU, NU, NU, NU, NU, NU

    # Only use blocking if there are genuinely multiple blocks
    effective_block_col = None
    if block_col_store and block_col_store in df.columns:
        if df[block_col_store].nunique() > 1:
            effective_block_col = block_col_store
    try:
        fi = fit_model(df, response_col, list(factor_cols),
                       custom_terms=custom_terms,
                       block_col=effective_block_col)
    except Exception as e:
        err = dbc.Alert(str(e), color="danger", dismissable=True)
        return NU, NU, EMPTY, EMPTY, EMPTY, err, NU, NU, NU, NU, NU, NU, NU

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

    # Serialise ANOVA rows — convert NaN/inf to None for JSON safety
    import math as _math
    def _clean(v):
        if isinstance(v, float) and (_math.isnan(v) or _math.isinf(v)):
            return None
        return v

    anova_records = [{k: _clean(val) for k, val in row.items()}
                     for row in aov.to_dict("records")]
    ms_clean = {k: _clean(v) for k, v in ms.items()}

    fit_store = {
        "factor_cols":  fi["factor_cols"],
        "response_col": fi["response_col"],
        "custom_terms": custom_terms,
        "block_col":    effective_block_col,
        "analysis_df":  analysis_json,
        "anova":        anova_records,
        "model_stats":  ms_clean,
    }

    # Enable all interpret buttons and save-html on success
    return (aov_table, stats_card,
            plot_pareto(fi), plot_main_effects(fi), plot_residuals(fi),
            None, json.dumps(fit_store),
            False, False, False, False, False, False)


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

    fi_cat_cols  = fi.get("cat_cols", [])
    fi_cat_levels = fi.get("cat_levels", {})
    enc      = fi["encoding"]
    inputs   = []
    for i, fname in enumerate(other):
        if fname in fi_cat_cols:
            lvls = fi_cat_levels.get(fname, [])
            inputs.append(html.Div([
                html.Small(f"{fname}:", className="text-muted fw-bold me-1"),
                dbc.Select(
                    id={"type": "surf-const", "index": i},
                    options=[{"label": l, "value": l} for l in lvls],
                    value=lvls[0] if lvls else None,
                    size="sm",
                    style={"width": "90px"},
                ),
            ], className="d-flex align-items-center"))
        else:
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
    fi_cat_cols = fi.get("cat_cols", [])
    rows = []
    for col, val in best_point.items():
        if col in fi_cat_cols:
            rows.append({"Factor": col, "Optimal Setting": str(val), "Range": "categoric"})
        else:
            enc = fi["encoding"][col]
            rows.append({"Factor": col,
                         "Optimal Setting": f"{val:.4f}",
                         "Range": f"[{enc['low']:.4f}, {enc['high']:.4f}]"})

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
        center_factorial=(4 if options.get("center_factorial") is None else int(options["center_factorial"])),
        center_star=(4 if options.get("center_star") is None else int(options["center_star"])),
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
    numeric_agent_cols = [f["name"] for f in factors if f.get("type") != "categoric"]
    if numeric_agent_cols:
        df_disp[numeric_agent_cols] = df_disp[numeric_agent_cols].round(4)
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

    # Rebuild factor rows to reflect agent config in the Design tab (assume numeric)
    factor_rows = [make_factor_row(i, f["name"], f["low"], f["high"], ftype="numeric")
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


# ═══════════════════════════════════════════════════════════════════════════════
# CALLBACKS — AI ASSISTANT TAB
# ═══════════════════════════════════════════════════════════════════════════════

def _chat_bubble(role: str, content: str) -> html.Div:
    """Render one chat message as a styled bubble."""
    is_user = role == "user"
    return html.Div(
        html.Div(
            content,
            style={
                "background":    "#2C7BE5" if is_user else "#e9ecef",
                "color":         "white"   if is_user else "#212529",
                "padding":       "0.5rem 1rem",
                "borderRadius":  "12px",
                "maxWidth":      "80%",
                "display":       "inline-block",
                "whiteSpace":    "pre-wrap",
                "fontSize":      "0.88rem",
                "lineHeight":    "1.5",
            },
        ),
        style={
            "textAlign":    "right" if is_user else "left",
            "marginBottom": "0.6rem",
        },
    )


@app.callback(
    Output("agent-chat-display",   "children",  allow_duplicate=True),
    Output("agent-chat-input",     "value"),
    Output("agent-chat-history",   "data",      allow_duplicate=True),
    Output("agent-pending-input",  "data"),
    Input("agent-send-btn",        "n_clicks"),
    Input("agent-chat-input",      "n_submit"),
    State("agent-chat-input",      "value"),
    State("session-id",            "data"),
    State("agent-chat-history",    "data"),
    prevent_initial_call=True,
)
def stage_user_message(n_clicks, n_submit, user_input, session_id, history):
    """Immediately show user message + thinking bubble; store input for get_agent_reply.
    Does NOT modify interviewer state or history — get_agent_reply owns that."""
    if not user_input or not user_input.strip():
        return no_update, no_update, no_update, no_update

    # Render existing history plus the new user message visually (display only)
    existing = list(history or [])
    user_text = user_input.strip()

    thinking_bubble = html.Div(
        html.Div(
            html.Em("Thinking…"),
            style={
                "background": "#e9ecef",
                "color": "#6c757d",
                "padding": "0.5rem 1rem",
                "borderRadius": "12px",
                "maxWidth": "80%",
                "display": "inline-block",
                "fontSize": "0.88rem",
            },
        ),
        id="thinking-bubble",
        style={"textAlign": "left", "marginBottom": "0.6rem"},
    )
    # Show existing messages + user bubble + thinking bubble immediately
    chat_children = [_chat_bubble(m["role"], m["content"]) for m in existing]
    chat_children.append(_chat_bubble("user", user_text))
    chat_children.append(thinking_bubble)

    # Pass user text to get_agent_reply; do NOT update history store yet
    return chat_children, "", no_update, {"input": user_text, "session_id": session_id or str(uuid.uuid4())}


@app.callback(
    Output("agent-chat-display",    "children",  allow_duplicate=True),
    Output("agent-chat-history",    "data",      allow_duplicate=True),
    Output("agent-configure-panel", "style"),
    Output("agent-configure-btn",   "disabled"),
    Output("agent-configure-btn",   "children"),
    Input("agent-pending-input",    "data"),
    State("session-id",             "data"),
    State("agent-chat-history",     "data"),
    prevent_initial_call=True,
)
def get_agent_reply(pending, session_id, history):
    """Call Anthropic API; interviewer._history is the single source of truth."""
    if not pending or not pending.get("input"):
        return no_update, no_update, no_update, no_update, no_update

    session_id  = session_id or pending.get("session_id", "")
    interviewer = _get_interviewer(session_id)
    user_text   = pending["input"]

    # On first ever message, start the interviewer (adds opening to self._history)
    if not interviewer._history:
        interviewer.start()

    try:
        reply = interviewer.chat(user_text)   # appends user msg + reply to self._history
    except Exception as e:
        reply = f"[Error contacting Claude API: {e}]"
        interviewer._history.append({"role": "assistant", "content": reply})

    # Rebuild display and store from interviewer's authoritative history
    chat_children = [_chat_bubble(m["role"], m["content"]) for m in interviewer._history]

    # Panel always visible once conversation has started
    panel_style = {"display": "block"}

    # Button enabled only once a design type keyword appears in the recommendation
    user_turns = sum(1 for m in interviewer._history if m["role"] == "user")
    has_rec = interviewer.has_recommendation()
    print(f"[get_agent_reply] user_turns={user_turns} has_recommendation={has_rec} last_reply_snippet={reply[:80]!r}")
    btn_disabled = not has_rec
    btn_label = (
        [html.I(className="bi bi-gear-fill me-2"), "Apply Recommended Design →"]
        if has_rec else
        [html.I(className="bi bi-hourglass-split me-2"), "Waiting for recommendation…"]
    )

    return chat_children, list(interviewer._history), panel_style, btn_disabled, btn_label


@app.callback(
    Output("agent-config-store", "data",     allow_duplicate=True),
    Output("main-tabs",          "active_tab"),
    Input("agent-configure-btn", "n_clicks"),
    State("session-id",          "data"),
    prevent_initial_call=True,
)
def configure_app_from_agent(n_clicks, session_id):
    """Extract design config from interviewer, write store + switch to Design tab atomically."""
    if not n_clicks:
        return no_update, no_update

    session_id  = session_id or ""
    interviewer = _get_interviewer(session_id)
    try:
        config = interviewer.extract_design_config()
    except Exception as e:
        print(f"[configure_app_from_agent] extraction failed: {e}")
        return no_update, no_update

    payload = {
        "design_type": config.get("design_type", "two_level_full"),
        "factors":     config.get("factors", []),
        "options":     config.get("options", {}),
    }
    print(f"[configure_app_from_agent] pushing config: {payload}")
    return payload, "tab-design"


@app.callback(
    Output("agent-panel-collapse", "is_open"),
    Input("toggle-agent-panel-btn", "n_clicks"),
    State("agent-panel-collapse", "is_open"),
    prevent_initial_call=True,
)
def toggle_agent_panel(n, is_open):
    return not is_open if n else is_open


# ═══════════════════════════════════════════════════════════════════════════════
# CALLBACKS — PER-SECTION AI INTERPRETATION
# ═══════════════════════════════════════════════════════════════════════════════

def _rebuild_fi(fit_store_json: str, analysis_json: str):
    """Re-run fit_model from the JSON snapshot so we get the live fi dict."""
    store        = json.loads(fit_store_json) if isinstance(fit_store_json, str) else fit_store_json
    df           = pd.read_json(io.StringIO(analysis_json), orient="split")
    response_col = store["response_col"]
    factor_cols  = store["factor_cols"]
    df[response_col] = pd.to_numeric(df[response_col], errors="coerce")
    return fit_model(df, response_col, list(factor_cols),
                     custom_terms=store.get("custom_terms"),
                     block_col=store.get("block_col"))


# ── I0. Interpret Design ─────────────────────────────────────────────────────
@app.callback(
    Output("interp-design-div",   "children"),
    Output("interp-design-store", "data"),
    Input("interp-design-btn",    "n_clicks"),
    State("analysis-df",          "data"),
    State("active-design",        "data"),
    State("response-cols-store",  "data"),
    State("block-col-store",      "data"),
    prevent_initial_call=True,
)
def interpret_design_section(n_clicks, analysis_json, active_design, resp_cols, block_col):
    if not n_clicks or not analysis_json:
        return no_update, no_update
    try:
        df          = pd.read_json(io.StringIO(analysis_json), orient="split")
        resp_cols   = resp_cols or []
        factor_cols = [c for c in df.columns if c not in ADMIN_COLS and c not in resp_cols]
        fac_info = []
        for fc in factor_cols:
            is_num = pd.api.types.is_numeric_dtype(df[fc])
            fac_info.append({
                "name": fc,
                "type": "numeric" if is_num else "categoric",
                "low":  float(df[fc].min()) if is_num else None,
                "high": float(df[fc].max()) if is_num else None,
                "levels": df[fc].unique().tolist() if not is_num else None,
            })
        design_data = {
            "design_type":       active_design or "unknown",
            "n_runs":            int(len(df)),
            "n_factors":         len(factor_cols),
            "factors":           fac_info,
            "n_blocks":          int(df["Block"].nunique())     if "Block"     in df.columns else 1,
            "n_replicates":      int(df["Replicate"].nunique()) if "Replicate" in df.columns else 1,
            "n_center_points":   int((df["Point Type"] == "Center").sum())
                                 if "Point Type" in df.columns else 0,
            "point_type_counts": df["Point Type"].value_counts().to_dict()
                                 if "Point Type" in df.columns else {},
        }
        text = Interpreter().interpret_design(design_data)
        return _interp_card(text), text
    except Exception as e:
        return dbc.Alert(str(e), color="danger"), ""


# ── I1. Interpret ANOVA ──────────────────────────────────────────────────────
@app.callback(
    Output("interp-anova-div",   "children"),
    Output("interp-anova-store", "data"),
    Input("interp-anova-btn",    "n_clicks"),
    State("fit-info-store",      "data"),
    prevent_initial_call=True,
)
def interpret_anova_section(n_clicks, fit_store_json):
    if not n_clicks or not fit_store_json:
        return no_update, no_update
    try:
        store        = json.loads(fit_store_json) if isinstance(fit_store_json, str) else fit_store_json
        text = Interpreter().interpret_anova(
            store.get("anova", []),
            store.get("model_stats", {}),
            store.get("factor_cols", []),
            store.get("response_col", "Response"),
        )
        return _interp_card(text), text
    except Exception as e:
        return dbc.Alert(str(e), color="danger"), ""


# ── I2. Interpret Effects & Interactions ─────────────────────────────────────
@app.callback(
    Output("interp-effects-div",   "children"),
    Output("interp-effects-store", "data"),
    Input("interp-effects-btn",    "n_clicks"),
    State("fit-info-store",        "data"),
    State("analysis-df",           "data"),
    State("ia-factor-a",           "value"),
    State("ia-factor-b",           "value"),
    prevent_initial_call=True,
)
def interpret_effects_section(n_clicks, fit_store_json, analysis_json, fa, fb):
    if not n_clicks or not fit_store_json or not analysis_json:
        return no_update, no_update
    try:
        fi = _rebuild_fi(fit_store_json, analysis_json)
        # Export plot images
        pareto_b64   = fig_to_b64(plot_pareto(fi))
        halfnorm_b64 = fig_to_b64(plot_half_normal(fi))
        maineff_b64  = fig_to_b64(plot_main_effects(fi))
        images = [pareto_b64, halfnorm_b64, maineff_b64]
        if fa and fb:
            images.append(fig_to_b64(plot_interaction(fi, fa, fb)))
        else:
            images.append(None)
        images = [img for img in images if img]  # drop any failed renders
        if not images:
            return _interp_card("⚠️ Could not render plot images (kaleido may not be installed). Run `pip install kaleido` and retry."), ""
        # Coefficient table as JSON string
        coef_df   = get_coefficients(fi)
        coef_json = coef_df.to_json(orient="records")
        text = Interpreter().interpret_effects(
            images, coef_json,
            fi["response_col"], fi["factor_cols"],
            ia_factor_a=fa, ia_factor_b=fb,
        )
        return _interp_card(text), text
    except Exception as e:
        return dbc.Alert(str(e), color="danger"), ""


# ── I3. Interpret Residuals ──────────────────────────────────────────────────
@app.callback(
    Output("interp-resid-div",   "children"),
    Output("interp-resid-store", "data"),
    Input("interp-resid-btn",    "n_clicks"),
    State("fit-info-store",      "data"),
    State("analysis-df",         "data"),
    prevent_initial_call=True,
)
def interpret_residuals_section(n_clicks, fit_store_json, analysis_json):
    if not n_clicks or not fit_store_json or not analysis_json:
        return no_update, no_update
    try:
        fi          = _rebuild_fi(fit_store_json, analysis_json)
        resid_img   = fig_to_b64(plot_residuals(fi), width=1000, height=540)
        if not resid_img:
            return _interp_card("⚠️ Could not render residual plot image (kaleido may not be installed). Run `pip install kaleido` and retry."), ""
        resid_stats = get_residual_stats(fi)
        text = Interpreter().interpret_residuals(
            resid_img, resid_stats, fi["response_col"]
        )
        return _interp_card(text), text
    except Exception as e:
        return dbc.Alert(str(e), color="danger"), ""


# ═══════════════════════════════════════════════════════════════════════════════
# CALLBACKS — SAVE HTML REPORT
# ═══════════════════════════════════════════════════════════════════════════════

@app.callback(
    Output("download-html",       "data"),
    Input("save-html-btn",        "n_clicks"),
    State("fit-info-store",       "data"),
    State("analysis-df",          "data"),
    State("active-design",        "data"),
    State("interp-design-store",  "data"),
    State("interp-anova-store",   "data"),
    State("interp-effects-store", "data"),
    State("interp-resid-store",   "data"),
    prevent_initial_call=True,
)
def save_html_report(n_clicks, fit_store_json, analysis_json, active_design,
                     interp_design, interp_anova, interp_effects, interp_resid):
    if not n_clicks or not fit_store_json or not analysis_json:
        return no_update
    try:
        import markdown as _md
        _md_ok = True
    except ImportError:
        _md_ok = False

    def _md_to_html(text):
        if not text:
            return "<p><em>AI interpretation not generated for this section.</em></p>"
        if _md_ok:
            return _md.markdown(text)
        # Minimal fallback: bold + bullets
        import re
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'^- (.+)$', r'<li>\1</li>', text, flags=re.MULTILINE)
        text = re.sub(r'(<li>.*</li>)', r'<ul>\1</ul>', text, flags=re.DOTALL)
        return f"<p>{text.replace(chr(10), '<br>')}</p>"

    try:
        store = json.loads(fit_store_json) if isinstance(fit_store_json, str) else fit_store_json
        fi    = _rebuild_fi(fit_store_json, analysis_json)
        df    = pd.read_json(io.StringIO(analysis_json), orient="split")

        # Plotly figures → embeddable HTML divs
        def _pfig(fig):
            return fig.to_html(full_html=False, include_plotlyjs=False)

        pareto_html   = _pfig(plot_pareto(fi))
        halfnorm_html = _pfig(plot_half_normal(fi))
        maineff_html  = _pfig(plot_main_effects(fi))
        resid_html    = _pfig(plot_residuals(fi))

        # Raw data table
        data_html = df.to_html(classes="report-table", index=False, border=0)

        # ANOVA table as HTML
        aov = get_anova_table(fi)
        aov_html = aov.to_html(classes="report-table", index=False, border=0)

        # Coefficients table
        coef = get_coefficients(fi)
        coef_html = coef.to_html(classes="report-table", index=False, border=0)

        # Equations
        eqs = get_equations(fi)

        # Design summary table
        resp_cols   = store.get("response_cols", [store.get("response_col", "Response")])
        factor_cols = [c for c in df.columns if c not in ADMIN_COLS and c not in resp_cols]
        design_label = (active_design or "Unknown").replace("_", " ").title()

        from datetime import datetime as _dt
        now = _dt.now().strftime("%Y-%m-%d %H:%M")

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>DOE Analysis Report</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 2rem 3rem; color: #212529; line-height: 1.6; }}
  h1   {{ color: #2C7BE5; border-bottom: 2px solid #2C7BE5; padding-bottom: 0.4rem; }}
  h2   {{ color: #495057; margin-top: 2.5rem; border-left: 4px solid #2C7BE5; padding-left: 0.7rem; }}
  .report-table {{ border-collapse: collapse; width: 100%; font-size: 0.88rem; margin: 1rem 0; }}
  .report-table th {{ background: #2C7BE5; color: white; padding: 6px 12px; text-align: center; }}
  .report-table td {{ padding: 5px 12px; border: 1px solid #dee2e6; text-align: center; }}
  .report-table tr:nth-child(even) {{ background: #f7f9fc; }}
  .ai-card {{ border-left: 4px solid #17a2b8; background: #f8fffe; padding: 1rem 1.2rem;
              margin: 1rem 0; border-radius: 4px; font-size: 0.9rem; }}
  .ai-card h3 {{ color: #17a2b8; margin-top: 0; font-size: 1rem; }}
  .no-interp {{ color: #6c757d; font-style: italic; }}
  pre  {{ background: #f0f4f8; padding: 0.8rem; border-radius: 4px; font-size: 0.84rem;
          white-space: pre-wrap; word-wrap: break-word; }}
  footer {{ margin-top: 3rem; color: #868e96; font-size: 0.8rem; border-top: 1px solid #dee2e6;
            padding-top: 0.5rem; }}
  @media print {{ body {{ margin: 1rem; }} }}
</style>
</head>
<body>
<h1>DOE Analysis Report</h1>
<p><strong>Design:</strong> {design_label} &nbsp;|&nbsp;
   <strong>Response:</strong> {store.get('response_col','—')} &nbsp;|&nbsp;
   <strong>Factors:</strong> {', '.join(store.get('factor_cols',[]))} &nbsp;|&nbsp;
   <strong>Runs:</strong> {len(df)} &nbsp;|&nbsp;
   <strong>Generated:</strong> {now}</p>

<h2>1. Design Summary</h2>
<div class="ai-card"><h3>🤖 AI Interpretation</h3>{_md_to_html(interp_design)}</div>

<h2>2. ANOVA Table &amp; Model Statistics</h2>
{aov_html}
<h3>Regression Equation (Coded)</h3>
<pre>{eqs.get('coded','')}</pre>
<h3>Regression Equation (Actual Units)</h3>
<pre>{eqs.get('actual','')}</pre>
<h3>Coefficients</h3>
{coef_html}
<div class="ai-card"><h3>🤖 AI Interpretation</h3>{_md_to_html(interp_anova)}</div>

<h2>3. Effects and Interaction Analysis</h2>
{pareto_html}
{halfnorm_html}
{maineff_html}
<div class="ai-card"><h3>🤖 AI Interpretation</h3>{_md_to_html(interp_effects)}</div>

<h2>4. Residual Analysis</h2>
{resid_html}
<div class="ai-card"><h3>🤖 AI Interpretation</h3>{_md_to_html(interp_resid)}</div>

<h2>5. Experimental Data</h2>
{data_html}

<footer>Generated by <strong>DOE Assistant</strong> &middot; {now}</footer>
</body>
</html>"""

        filename = f"doe_report_{_dt.now().strftime('%Y%m%d_%H%M%S')}.html"
        return dcc.send_string(html_content, filename=filename)
    except Exception as e:
        return no_update


@app.callback(
    Output("download-html-pred", "data"),
    Input("save-html-pred-btn",  "n_clicks"),
    State("fit-info-store",       "data"),
    State("analysis-df",          "data"),
    State("active-design",        "data"),
    State("interp-design-store",  "data"),
    State("interp-anova-store",   "data"),
    State("interp-effects-store", "data"),
    State("interp-resid-store",   "data"),
    prevent_initial_call=True,
)
def save_html_report_pred(n_clicks, fit_store_json, analysis_json, active_design,
                           interp_design, interp_anova, interp_effects, interp_resid):
    return save_html_report(n_clicks, fit_store_json, analysis_json, active_design,
                            interp_design, interp_anova, interp_effects, interp_resid)


# ═══════════════════════════════════════════════════════════════════════════════
# CALLBACKS — SIDEBAR HELP PANEL
# ═══════════════════════════════════════════════════════════════════════════════

def _make_help_system_prompt(active_tab: str, app_state: dict) -> str:
    state_summary = (
        json.dumps(
            {k: v for k, v in app_state.items() if k in
             ("design_type", "factor_count", "response_col", "anova", "model_stats")},
            indent=2,
        )
        if app_state else "No model fitted yet."
    )
    tab_names = {
        "tab-design":      "Design",
        "tab-analysis":    "Analysis",
        "tab-prediction":  "Prediction & Optimization",
    }
    tab_label = tab_names.get(active_tab or "", active_tab or "unknown")
    return (
        f"You are a concise help assistant for a DOE (Design of Experiments) "
        f"Dash application. The user is on the '{tab_label}' tab.\n\n"
        f"Current app state:\n{state_summary}\n\n"
        "Provide short, practical guidance (2-4 sentences). "
        "Reference NIST / Montgomery best practices where relevant. "
        "Be direct — the user needs actionable help, not a lecture."
    )


@app.callback(
    Output("help-panel", "is_open"),
    Input("help-toggle", "n_clicks"),
    State("help-panel",  "is_open"),
    prevent_initial_call=True,
)
def toggle_help(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open


@app.callback(
    Output("help-chat-display", "children"),
    Output("help-chat-input",   "value"),
    Output("help-chat-history", "data"),
    Input("help-send-btn",      "n_clicks"),
    Input("help-chat-input",    "n_submit"),
    State("help-chat-input",    "value"),
    State("main-tabs",          "active_tab"),
    State("help-chat-history",  "data"),
    prevent_initial_call=True,
)
def send_help_message(n_clicks, n_submit, user_input, active_tab, history):
    """Handle a question in the sidebar help panel using claude-haiku-4-5."""
    if not user_input or not user_input.strip():
        return no_update, no_update, no_update

    from anthropic import Anthropic as _Anthropic
    history = list(history or [])
    history.append({"role": "user", "content": user_input.strip()})

    system = _make_help_system_prompt(active_tab or "tab-design", _agent_readable_state)

    try:
        client   = _Anthropic()
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            system=system,
            messages=history,
        )
        reply = response.content[0].text
    except Exception as e:
        reply = f"[Help unavailable: {e}]"

    history.append({"role": "assistant", "content": reply})
    children = [_chat_bubble(m["role"], m["content"]) for m in history]
    return children, "", history


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=8050)
