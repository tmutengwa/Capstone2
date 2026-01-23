import os
import dash
import base64
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np
from itertools import combinations
from config import settings

from orchestrator import DuckOrchestrator
from profiler import SmartProfiler
from llm_consultant import LLMConsultant
from logger import logger

# --- Initialization ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)
app.title = "AutoEDA Platform"

# Injecting Global Styles for AI Report Formatting (Bold 14/13, Regular 12)
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .ai-report h2 { font-weight: bold; font-size: 14pt; margin-top: 20px; }
            .ai-report h3 { font-weight: bold; font-size: 13pt; margin-top: 15px; }
            .ai-report p, .ai-report li { font-weight: normal; font-size: 12pt; line-height: 1.6; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

orchestrator = DuckOrchestrator()
default_file = "data/train.csv"
if os.path.exists(default_file):
    orchestrator.load_data(default_file)

consultant = LLMConsultant()

# --- Structural Discovery & Simulation Logic ---
class BlackFridayEDA:
    def __init__(self, df):
        self.df = df

    def get_missingness_price_signature(self):
        """Discovers the Price Signature: Correlation between nulls in Category 3 and Purchase."""
        if 'Product_Category_3' in self.df.columns and 'Purchase' in self.df.columns:
            null_flag = self.df['Product_Category_3'].isnull().astype(int)
            return round(null_flag.corr(self.df['Purchase']), 4)
        return 0.0

    def bottom_up_lattice_check(self, cols=['User_ID', 'Product_ID']):
        """Math-driven discovery of structural hierarchy via Lattice Logic."""
        total = len(self.df)
        available = [c for c in cols if c in self.df.columns]
        for r in range(1, 3):
            for combo in combinations(available, r):
                if len(self.df.groupby(list(combo))) == total:
                    return f"Lattice Root: {' + '.join(combo)}"
        return "Non-Hierarchical/Flat Structure"

    def simulate_variance_data(self):
        """Generates the Simulation Variance Table data to quantify cleaning bias."""
        global_mean = self.df['Purchase'].mean()
        dropped_df = self.df.dropna(subset=['Product_Category_3'])
        dropped_mean = dropped_df['Purchase'].mean()
        
        null_mask = self.df['Product_Category_3'].isnull()
        budget_segment_mean = self.df[null_mask]['Purchase'].mean()
        
        data = [
            {"Strategy": "Global Baseline (Actual)", "Purchase Mean": f"${global_mean:,.2f}", "Bias Error": "$0.00", "Data Retention": "100%"},
            {"Strategy": "Top-Down (Dropping Rows)", "Purchase Mean": f"${dropped_mean:,.2f}", "Bias Error": f"+${dropped_mean - global_mean:,.2f}", "Data Retention": f"{(len(dropped_df)/len(self.df))*100:.1f}%"},
            {"Strategy": "Bottom-Up (Labelling)", "Purchase Mean": f"${budget_segment_mean:,.2f}", "Bias Error": "Segment Signal", "Data Retention": "100%"}
        ]
        return pd.DataFrame(data)

    def get_decision_matrix(self):
        """Structural state mapped to grounded management strategies for the PromptFactory."""
        root = self.bottom_up_lattice_check()
        if "User_ID" in root and "Product_ID" in root:
            return "STRICT TREE (Strategy: Backfill / Inherit)"
        elif "Lattice" in root:
            return "NETWORK (Strategy: Categorical 'None' / Labeling)"
        return "LEVEL-BASED (Strategy: Path Truncation)"

# --- Layout Components ---

navbar = dbc.Navbar(
    dbc.Container(
        [
            dbc.NavbarBrand("AI-Driven EDA Platform", className="mx-auto", style={"fontWeight": "bold", "fontSize": "1.5rem"}),
        ],
        fluid=True,
    ),
    color="primary", dark=True,
)

sidebar_width = "10.5rem"
content_margin = "11.5rem"

sidebar = html.Div(
    [
        html.H5("Controls", className="display-7"),
        html.Hr(),
        html.P("Current Dataset:", className="lead", style={"fontSize": "0.9rem"}),
        html.Div(id="dataset-info-sidebar", style={"fontSize": "0.85rem", "marginBottom": "10px"}),
        dbc.Input(id="load-path", placeholder="Path to CSV/Parquet", type="text", value=default_file, className="mb-2", size="sm"),
        dbc.Button("Load Data", id="btn-load", color="secondary", className="mb-3", size="sm", style={"width": "100%"}),
        html.Hr(),
        html.Label("SQL Transform:", style={"fontSize": "0.9rem"}),
        dbc.Textarea(id="sql-query", placeholder="SELECT * FROM data_view WHERE ...", className="mb-2", style={"height": "80px", "fontSize": "0.85rem"}),
        dbc.Button("Apply Transform", id="btn-transform", color="info", className="mb-3", size="sm", style={"width": "100%"}),
    ],
    style={
        "position": "fixed", "top": "56px", "left": 0, "bottom": 0, "width": sidebar_width,
        "padding": "1rem", "backgroundColor": "#f8f9fa", "overflowY": "auto"
    },
)

content = html.Div(id="page-content", style={"marginLeft": content_margin, "padding": "2rem"})

tabs = dbc.Tabs(
    [
        dbc.Tab(label="Dashboard", tab_id="tab-dashboard"),
        dbc.Tab(label="Visual Analysis", tab_id="tab-visuals"),
        dbc.Tab(label="Report Summary", tab_id="tab-ai"),
    ],
    id="tabs",
    active_tab="tab-dashboard",
)

app.layout = html.Div([
    dcc.Store(id='store-health', storage_type='session'),
    dcc.Store(id='store-visual', storage_type='session'),
    dcc.Store(id='store-summary', storage_type='session'),
    dcc.Store(id='store-ask-history', storage_type='session'),
    navbar,
    sidebar,
    html.Div([
        tabs,
        content
    ], style={"marginLeft": content_margin, "padding": "1rem"})
])

# --- Callbacks ---

@app.callback(
    Output("dataset-info-sidebar", "children"),
    Input("btn-load", "n_clicks"),
    Input("btn-transform", "n_clicks"),
    State("load-path", "value"),
    State("sql-query", "value")
)
def update_data(n_load, n_trans, path, query):
    ctx = dash.callback_context
    if not ctx.triggered:
        df = orchestrator.get_data()
        return f"{df.shape[0]} rows, {df.shape[1]} cols" if not df.empty else "No data loaded."

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if button_id == "btn-load" and path:
        try:
            df = orchestrator.load_data(path)
            return f"{df.shape[0]} rows, {df.shape[1]} cols"
        except Exception as e:
            return f"Error: {str(e)}"
    
    if button_id == "btn-transform" and query:
        try:
            df = orchestrator.transform(query)
            return f"{df.shape[0]} rows, {df.shape[1]} cols"
        except Exception as e:
            return f"Error: {str(e)}"
            
    df = orchestrator.get_data()
    return f"{df.shape[0]} rows, {df.shape[1]} cols" if not df.empty else "No data loaded."

@app.callback(
    Output("page-content", "children"),
    Input("tabs", "active_tab"),
    State('store-health', 'data'),
    State('store-visual', 'data'),
    State('store-summary', 'data'),
    State('store-ask-history', 'data')
)
def render_tab_content(active_tab, health_data, visual_data, summary_data, ask_data):
    df = orchestrator.get_data()
    if df.empty:
        return html.Div("Please load a dataset first.", className="text-danger")
    
    profiler = SmartProfiler(df)
    eda = BlackFridayEDA(df)

    if active_tab == "tab-dashboard":
        # 1. PRESERVED: Dataset Overview Logic
        summary = profiler.get_dataset_summary()
        
        # This renders the structural anchors discovered via Bottom-Up Logic
        purchase_stats = {}
        if 'Purchase' in df.columns:
            purchase_stats = {
                'Mean': round(df['Purchase'].mean(), 2),
                'Min': df['Purchase'].min(),
                'Max': df['Purchase'].max(),
                'Median': df['Purchase'].median(),
                'Mode': df['Purchase'].mode()[0] if not df['Purchase'].mode().empty else "N/A"
            }
        else:
            purchase_stats = {k: 'N/A' for k in ['Mean', 'Min', 'Max', 'Median', 'Mode']}
        
        # --- ADDITIVE: Lattice Node Visualization ---
        lattice_viz = html.Div([
            html.H4("Lattice Node Mapping (Structural Dependency)"),
            dbc.Alert(f"Root Discovery: {eda.bottom_up_lattice_check()}", color="success"),
            html.P("This lattice verifies the Parent-Child relationship between User and Category."),
        ], className="mb-4")

        # UI Implementation of the Simulation Variance Table
        variance_df = eda.simulate_variance_data()
        variance_table = dbc.Table.from_dataframe(variance_df, striped=True, bordered=True, hover=True, responsive=True)
        
        diagnostics = profiler.run_diagnostic_tests()
        skew_kurt = profiler.get_skewness_kurtosis()
        
        return html.Div([
            html.H3("Dataset Overview"),
            dbc.Row([
                dbc.Col(dbc.Card([dbc.CardBody([html.H4(summary['rows']), html.P("Rows")])], color="light"), width=3),
                dbc.Col(dbc.Card([dbc.CardBody([html.H4(summary['columns']), html.P("Columns")])], color="light"), width=3),
                dbc.Col(dbc.Card([dbc.CardBody([html.H4(f"{summary['memory_mb']} MB"), html.P("Memory")])], color="light"), width=3),
                dbc.Col(dbc.Card([dbc.CardBody([html.H4(summary['duplicates']), html.P("Duplicates")])], color="light"), width=3),
            ], className="mb-2"),
            
            dbc.Row([
                dbc.Col(dbc.Card([dbc.CardBody([html.H4(purchase_stats['Mean']), html.P("Purchase Mean")])], color="info", inverse=True), width=2),
                dbc.Col(dbc.Card([dbc.CardBody([html.H4(purchase_stats['Median']), html.P("Purchase Median")])], color="info", inverse=True), width=2),
                dbc.Col(dbc.Card([dbc.CardBody([html.H4(purchase_stats['Mode']), html.P("Purchase Mode")])], color="info", inverse=True), width=2),
                dbc.Col(dbc.Card([dbc.CardBody([html.H4(purchase_stats['Min']), html.P("Purchase Min")])], color="secondary", inverse=True), width=3),
                dbc.Col(dbc.Card([dbc.CardBody([html.H4(purchase_stats['Max']), html.P("Purchase Max")])], color="secondary", inverse=True), width=3),
            ], className="mb-4"),
            
            html.H4("Simulation Variance Table (Bias Detection)"),
            variance_table,
            html.P("Insight: Switching to labelling removes artificial mean inflation.", style={"fontSize": "0.85rem", "color": "blue"}),

            html.H4("Data Health Assessment"),
            dbc.Button("Generate Analysis", id="btn-generate-health", color="info", size="sm", className="mb-2"),
            dcc.Loading(id="loading-health", children=[html.Div(id="ai-health-output", children=dcc.Markdown(health_data, className="ai-report") if health_data else None, className="p-3 border bg-light mb-4")]),

            html.H4("Diagnostic Tests"),
            dbc.Table.from_dataframe(diagnostics, striped=True, bordered=True, hover=True, responsive=True),
            
            html.H4("Skewness & Kurtosis"),
            dbc.Table.from_dataframe(skew_kurt, striped=True, bordered=True, hover=True, responsive=True),
        ])

    elif active_tab == "tab-visuals":
        plots = profiler.generate_all_plots()
        plot_components = []
        
        # Sort keys to ensure 1-7 order
        for name in sorted(plots.keys()):
            path = plots[name]
            if os.path.exists(path):
                encoded = base64.b64encode(open(path, 'rb').read()).decode('ascii')
                plot_components.append(
                    dbc.Row([
                        dbc.Col([
                            html.H5(name, className="text-center"),
                            html.Img(src=f'data:image/png;base64,{encoded}', style={'width': '100%', 'height': 'auto', 'border': '1px solid #ddd', 'borderRadius': '5px', 'boxShadow': '2px 2px 5px rgba(0,0,0,0.1)'})
                        ], width={"size": 10, "offset": 1})
                    ], className="mb-5")
                )

        return html.Div([
            html.H3("Visual Analysis"),
            dbc.Button("Generate Visual Insights", id="btn-generate-visual-insights", color="warning", className="mb-3"),
            dcc.Loading(id="loading-visual-insights", children=[html.Div(id="ai-visual-output", children=dcc.Markdown(visual_data, className="ai-report") if visual_data else None, className="p-3 border bg-light mb-4")]),
            html.Div(plot_components)
        ])

    elif active_tab == "tab-ai":
        return html.Div([
            html.H3("Report Summary"),
            dbc.Button("Generate Executive Summary", id="btn-ai-summary", color="primary", className="mb-3"),
            dcc.Loading(html.Div(id="ai-summary-output", children=dcc.Markdown(summary_data, className="ai-report") if summary_data else None, className="mb-4 p-3 border bg-light")),
            html.Hr(),
            html.H5("Ask Godobori"),
            dbc.Input(id="ai-question", placeholder="e.g., What is the relationship between Age and Purchase?", className="mb-2"),
            dbc.Button("Ask Godobori", id="btn-ai-ask", color="success"),
            dcc.Loading(html.Div(id="ai-answer-output", children=dcc.Markdown(ask_data, className="ai-report") if ask_data else None, className="mt-3 p-3 border bg-light"))
        ])
    return html.Div("Tab not found.")

# --- Callbacks for Storage and Generation (Refactored for PromptFactory Integration) ---

@app.callback(
    [Output("ai-health-output", "children"), Output("store-health", "data")],
    Input("btn-generate-health", "n_clicks"),
    prevent_initial_call=True
)
def generate_health_assessment(n):
    if not n: return dash.no_update, dash.no_update
    df = orchestrator.get_data(); eda = BlackFridayEDA(df)
    if df.empty: return "No data.", None
    
    # Passing grounded metadata to reflect new Factory signatures
    eda_metadata = {
        "Price_Signature": eda.get_missingness_price_signature(),
        "Lattice_Root": eda.bottom_up_lattice_check()
    }
    explanation = consultant.analyze_diagnostics(df.head(5).to_string(), eda_metadata)
    return dcc.Markdown(explanation, className="ai-report"), explanation

@app.callback(
    [Output("ai-visual-output", "children"), Output("store-visual", "data")],
    Input("btn-generate-visual-insights", "n_clicks"),
    prevent_initial_call=True
)
def generate_visual_insights(n):
    if not n: return dash.no_update, dash.no_update
    df = orchestrator.get_data(); profiler = SmartProfiler(df); eda = BlackFridayEDA(df)
    if df.empty: return "No data.", None
    
    eda_metadata = {"Lattice_Root": eda.bottom_up_lattice_check()}
    insight = consultant.generate_grounded_report(profiler.generate_all_plots(), eda_metadata)
    return dcc.Markdown(insight, className="ai-report"), insight

@app.callback(
    [Output("ai-summary-output", "children"), Output("store-summary", "data")],
    Input("btn-ai-summary", "n_clicks"),
    prevent_initial_call=True
)
def generate_ai_summary(n):
    if not n: return dash.no_update, dash.no_update
    df = orchestrator.get_data(); eda = BlackFridayEDA(df)
    if df.empty: return "No data.", None
    
    eda_metadata = {
        "Lattice_Root": eda.bottom_up_lattice_check(),
        "Price_Signature": eda.get_missingness_price_signature()
    }
    insight = consultant.generate_summary_insight(df.dtypes.to_string(), eda_metadata)
    return dcc.Markdown(insight, className="ai-report"), insight

@app.callback(
    [Output("ai-answer-output", "children"), Output("store-ask-history", "data")],
    Input("btn-ai-ask", "n_clicks"),
    State("ai-question", "value"),
    State("store-ask-history", "data"),
    prevent_initial_call=True
)
def ask_ai(n, question, existing_history):
    if not n or not question: return dash.no_update, dash.no_update
    df = orchestrator.get_data(); eda = BlackFridayEDA(df)
    
    eda_metadata = {"Lattice_Root": eda.bottom_up_lattice_check()}
    answer = consultant.answer_question(df.head(10).to_string(), question, eda_metadata)
    
    new_history = f"**Q: {question}**\n\n{answer}\n\n---\n\n" + (existing_history if existing_history else "")
    return dcc.Markdown(new_history, className="ai-report"), new_history

# WORKS IN LAMBDA
if __name__ == "__main__":
    # Debug must be False for production/Lambda
    app.run(debug=False, host="0.0.0.0", port=8050)