import numpy as np
import pandas as pd
import plotly.graph_objs as go
from shiny import App, reactive, ui
from shinywidgets import output_widget, register_widget

import seaborn as sns

plot_df_overall = pd.read_csv("./admission_diabete_fullyear/plot_df_overall.csv", index_col=0)
plot_df_by_group_1 = pd.read_csv("./admission_diabete_fullyear/plot_df_by_group_1.csv", index_col=0)
plot_df_by_group_2 = pd.read_csv("./admission_diabete_fullyear/plot_df_by_group_2.csv", index_col=0)

def rgb01_hex(col):
   col_hex =  [round(i* 255) for i in col]
   col_hex = '#%02x%02x%02x' % tuple(col_hex)
   return col_hex

def plot_radar(plot_df_overall, thresh_show, title, **kwargs):
      fig = go.Figure()
      # fig = sp.make_subplots(rows=1, cols=2)
      cmap = sns.diverging_palette(200,20,sep=10, s=50, as_cmap=False, n = plot_df_overall.shape[0])
      theta = plot_df_overall.columns.tolist()
      theta += theta[:1]

      for i, id in enumerate(plot_df_overall.index): 
            values = plot_df_overall.loc[id, :]
            values = values.values.flatten().tolist()
            values += values[:1]
            info = [f"{theta[j]}: {v:.3f}" for j, v in enumerate(values) if j != len(values)-1]
            fig.add_trace(go.Scatterpolar(
                  r=values,
                  theta=theta, 
                  fill = "toself" if id == "FRC" else "none",
                  text= "\n".join(info),
                  name = f"{id}",
                  line = dict(color=rgb01_hex(cmap[i]), dash="dot")
            ))
      
      fig.update_layout(
            title = title,
            font = dict(family = "Arial"),
            polar=dict(
                  # bgcolor = "#1e2130",
                  radialaxis=dict(
                        visible=True,
                        range=[0, thresh_show]
                  )
            ),
            legend = dict(
                  x=0.25, y= -0.1, orientation="h"),
            showlegend=True,
            **kwargs
      )
      return fig
  
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
    ui.h2({"style": "text-align: center;"}, "Model Comparison"),
    ui.input_checkbox("show_by_group", "Show differences across subgroups", value=True),
    ui.row( 
           ui.column(
            4,
            output_widget("scatterpolarplot", width="100%", height="100%")
        ),
    ),
    ui.row(
        ui.column(
            4,
            output_widget("scatterpolarplot1", width="100%", height="100%"),
        ),
        ui.column(
            4,
            output_widget("scatterpolarplot2", width="100%", height="100%"),
        ),
    ),
)
  
def server(input, output, session):
    p = plot_radar(plot_df_overall, 0.3, "Fairness metrics")
    p1 = plot_radar(plot_df_by_group_1, 0.9, "TPR by group")
    p2 = plot_radar(plot_df_by_group_2, 0.9, "TNR by group") #width=480, height=480
    
    register_widget("scatterpolarplot", p)
    register_widget("scatterpolarplot1", p1)
    register_widget("scatterpolarplot2", p2)
    
    @reactive.Effect
    def _():
        pass
        # p1.layout.visibility = "hidden" if not input.show_by_group() else "none"
        # p2.data[1].visible = input.show_by_group()


app = App(app_ui, server)

app.run()