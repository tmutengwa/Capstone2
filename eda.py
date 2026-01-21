import os
import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np
from config import settings

from orchestrator import DuckOrchestrator
from profiler import SmartProfiler
from llm_consultant import LLMConsultant
from logger import logger

# --- Initialization ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)
app.title = "AI-Driven EDA Platform"

orchestrator = DuckOrchestrator()
default_file = "data/train.csv"
if os.path.exists(default_file):
    orchestrator.load_data(default_file)

consultant = LLMConsultant()

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
    # State Stores
    dcc.Store(id='store-health'),
    dcc.Store(id='store-visual'),
    dcc.Store(id='store-summary'),
    dcc.Store(id='store-ask-history'),
    
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
        if not df.empty:
             return f"{df.shape[0]} rows, {df.shape[1]} cols"
        return "No data loaded."

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
    if not df.empty:
         return f"{df.shape[0]} rows, {df.shape[1]} cols"
    return "No data loaded."

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

    if active_tab == "tab-dashboard":
        summary = profiler.get_dataset_summary()
        purchase_stats = {}
        if 'Purchase' in df.columns:
            purchase_stats['Mean'] = round(df['Purchase'].mean(), 2)
            purchase_stats['Min'] = df['Purchase'].min()
            purchase_stats['Max'] = df['Purchase'].max()
            purchase_stats['Median'] = df['Purchase'].median()
            mode_series = df['Purchase'].mode()
            purchase_stats['Mode'] = mode_series[0] if not mode_series.empty else "N/A"
        else:
            purchase_stats = {k: 'N/A' for k in ['Mean', 'Min', 'Max', 'Median', 'Mode']}
        
        diagnostics = profiler.run_diagnostic_tests()
        diag_table = dbc.Table.from_dataframe(diagnostics, striped=True, bordered=True, hover=True, responsive=True) if not diagnostics.empty else html.P("No diagnostic tests available.")
        
        skew_kurt = profiler.get_skewness_kurtosis()
        skew_table = dbc.Table.from_dataframe(skew_kurt, striped=True, bordered=True, hover=True, responsive=True) if not skew_kurt.empty else html.P("No numerical columns for skewness analysis.")
        
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
            
            html.H4("Data Health Assessment"),
            dbc.Button("Generate Analysis", id="btn-generate-health", color="info", size="sm", className="mb-2"),
            dcc.Loading(id="loading-health", children=[html.Div(id="ai-health-output", children=dcc.Markdown(health_data) if health_data else None, className="p-3 border bg-light mb-4")]),

            html.H4("Diagnostic Tests"),
            diag_table,
            
            html.H4("Skewness & Kurtosis"),
            skew_table,
        ])

    elif active_tab == "tab-visuals":
        plots = profiler.generate_all_plots()
        plot_components = []
        plot_items = list(plots.items())
        i = 0
        while i < len(plot_items):
            name, fig = plot_items[i]
            if name == "Pairwise Plot":
                plot_components.append(dbc.Row(dbc.Col(dcc.Graph(figure=fig), width=12), className="mb-4"))
                i += 1
                continue
            row_children = [dbc.Col(dcc.Graph(figure=fig), width=6)]
            if i + 1 < len(plot_items):
                name2, fig2 = plot_items[i+1]
                if name2 != "Pairwise Plot":
                    row_children.append(dbc.Col(dcc.Graph(figure=fig2), width=6))
                    i += 2
                else: i += 1
            else: i += 1
            plot_components.append(dbc.Row(row_children, className="mb-4"))

        return html.Div([
            html.H3("Visual Analysis"),
            dbc.Button("Generate Visual Insights", id="btn-generate-visual-insights", color="warning", className="mb-3"),
            dcc.Loading(id="loading-visual-insights", children=[html.Div(id="ai-visual-output", children=dcc.Markdown(visual_data) if visual_data else None, className="p-3 border bg-light mb-4")]),
            html.Div(plot_components)
        ])

    elif active_tab == "tab-ai":
        return html.Div([
            html.H3("Report Summary"),
            dbc.Button("Generate Executive Summary", id="btn-ai-summary", color="primary", className="mb-3"),
            dcc.Loading(html.Div(id="ai-summary-output", children=dcc.Markdown(summary_data) if summary_data else None, className="mb-4 p-3 border bg-light")),
            html.Hr(),
            html.H5("Ask Godobori"),
            dbc.Input(id="ai-question", placeholder="e.g., What is the relationship between Age and Purchase?", className="mb-2"),
            dbc.Button("Ask Godobori", id="btn-ai-ask", color="success"),
            dcc.Loading(html.Div(id="ai-answer-output", children=dcc.Markdown(ask_data) if ask_data else None, className="mt-3 p-3 border bg-light"))
        ])
    
    return html.Div("Tab not found.")

# --- Callbacks for Storage and Generation ---

@app.callback(
    [Output("ai-health-output", "children"), Output("store-health", "data")],
    Input("btn-generate-health", "n_clicks"),
    prevent_initial_call=True
)
def generate_health_assessment(n):
    df = orchestrator.get_data()
    if df.empty: return "No data.", None
    profiler = SmartProfiler(df)
    diagnostics = profiler.run_diagnostic_tests()
    skew_kurt = profiler.get_skewness_kurtosis()
    context = f"Diagnostic Tests:\n{diagnostics.to_string()}\n\nSkewness & Kurtosis:\n{skew_kurt.to_string()}"
    explanation = consultant.analyze_diagnostics(context)
    return dcc.Markdown(explanation), explanation

@app.callback(
    [Output("ai-visual-output", "children"), Output("store-visual", "data")],
    Input("btn-generate-visual-insights", "n_clicks"),
    prevent_initial_call=True
)
def generate_visual_insights(n):
    df = orchestrator.get_data()
    if df.empty: return "No data.", None
    profiler = SmartProfiler(df)
    corr_matrix_str = profiler.get_correlation_matrix_str()
    insight = consultant.analyze_visuals(corr_matrix_str)
    return dcc.Markdown(insight), insight

@app.callback(
    [Output("ai-summary-output", "children"), Output("store-summary", "data")],
    Input("btn-ai-summary", "n_clicks"),
    prevent_initial_call=True
)
def generate_ai_summary(n):
    df = orchestrator.get_data()
    if df.empty: return "No data.", None
    profiler = SmartProfiler(df)
    summary_stats = profiler.get_dataset_summary()
    context = f"Summary Stats: {summary_stats}\nColumns: {df.dtypes.to_string()}"
    insight = consultant.generate_summary_insight(context)
    return dcc.Markdown(insight), insight

@app.callback(
    [Output("ai-answer-output", "children"), Output("store-ask-history", "data")],
    Input("btn-ai-ask", "n_clicks"),
    State("ai-question", "value"),
    State("store-ask-history", "data"),
    prevent_initial_call=True
)
def ask_ai(n, question, existing_history):
    if not n or not question: return dash.no_update, dash.no_update
    df = orchestrator.get_data()
    advanced_stats = SmartProfiler(df).generate_advanced_stats()
    target = 'Purchase' if 'Purchase' in df.columns else None
    corr_info = ""
    if target:
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 1:
            corr_info = f"\nCorrelations with {target}:\n{df[num_cols].corr()[target].sort_values(ascending=False).to_string()}"

    context = f"Columns: {list(df.columns)}\n{advanced_stats}\n{corr_info}\nData Sample:\n{df.head(10).to_string()}"
    answer = consultant.answer_question(context, question)
    
    new_history = f"**Q: {question}**\n\n{answer}\n\n---\n\n" + (existing_history if existing_history else "")
    return dcc.Markdown(new_history), new_history

if __name__ == "__main__":
    import webbrowser
    from threading import Timer
    def open_browser(): webbrowser.open_new("http://localhost:8050")
    logger.info("Starting Dash server on http://0.0.0.0:8050")
    Timer(1, open_browser).start()
    app.run(debug=True, host="0.0.0.0", port=8050, use_reloader=False)