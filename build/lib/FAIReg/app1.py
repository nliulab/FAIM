import numpy as np
import pandas as pd
import plotly.graph_objs as go
import seaborn as sns

from shiny import App, reactive, ui, render
from shinywidgets import output_widget, register_widget, render_widget
from .fairness_plotting import *


dataset = "SGH_ED"

if dataset == "mimiciv":
    data_name = "hospitalization"
elif dataset == "SGH_ED":
    data_name = "admission_diabete_fullyear"
    
fairness_metrics_df = pd.read_csv(f"./{data_name}/fairness_metrics_df.csv", index_col=0)
metrics = fairness_metrics_df.columns
num_metrics = len(metrics)
metrics_dic = dict(zip(metrics, metrics))
  

app_ui = ui.page_fluid(
    ui.tags.style(
        """
        .app-col {
            border: 1px solid black;
            border-radius: 5px;
            background-color: #eee;
            padding: 8px;
            margin-top: 5px;
            margin-bottom: 5px;
        }
        """
    ),
    ui.h2({"style": "text-align: center;"}, "Model Selection"),
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.input_checkbox_group("x", "Metrics", metrics_dic, selected=metrics.tolist()),
            ui.input_slider("r", label="Thresh_show", min=0, max=0.3, value=20), # thresh_show
        ),
        ui.panel_main(
            ui.output_text_verbatim("txt"),
            ui.output_plot("histogram", width="100%", height="200px"),
            output_widget("radar", width="100%", height="100%"),
        ),
    ),
)
  
def server(input, output, session):
    # df = fairness_metrics_df[input.x()]
    @output
    @render.text
    def txt():
        return f'{input.x()}'
    
    @output
    @render_widget
    def radar():
        p = plot_radar(fairness_metrics_df[list(input.x())], input.r(), "Fairness metrics")
        return go.FigureWidget(p)
    # register_widget("scatterpolarplot", go.FigureWidget(p))
    
    @output
    @render.plot(alt="A histogram")
    def histogram():
        fairness_metrics_df[list(input.x())].hist(layout=(1, num_metrics), figsize=(16, 2), bins=50)
        
    
    # @reactive.Effect
    # def _():
    #     pass


app = App(app_ui, server)

app.run()