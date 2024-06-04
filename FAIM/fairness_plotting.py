import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import math

# from FairLite.fair_evaluator import FairEvaluator

import plotnine as pn
from plotnine import *

from .utils import *


def rgb01_hex(col):
    col_hex = [round(i * 255) for i in col]
    col_hex = "#%02x%02x%02x" % tuple(col_hex)
    return col_hex


def compute_area(fairness_metrics):
    n_metric = len(fairness_metrics)
    tmp = fairness_metrics.values.flatten().tolist()
    tmp_1 = tmp[1:] + tmp[:1]

    if n_metric > 2:
        theta_c = 2 * np.pi / n_metric
        area = np.sum(np.array(tmp) * np.array(tmp_1) * np.sin(theta_c))
    elif n_metric == 2:
        area = np.sum(np.array(tmp) * np.array(tmp_1))
    else:
        area = np.abs(tmp[0])

    return area


def plot_perf_metric(
    perf_metric, eligible, x_range, select=None, plot_selected=False, x_breaks=None
):
    """ Plot performance metrics of sampled models

        Parameters
        ----------
        perf_metric : numpy.array or pandas.Series
            Numeric vector of performance metrics for all sampled models
        eligible : numpy.array or pandas.Series
            Boolean vector of the same length of 'perf_metric', indicating \
                whether each sample is eligible.
        x_range : list
            Numeric vector indicating the range of eligible values for \
                performance metrics. 
            Will be indicated by dotted vertical lines in plots.
        select : list or numpy.array, optional (default: None)
            Numeric vector of indexes of 'perf_metric' to be selected
        plot_selected : bool, optional (default: False)
            Whether performance metrics of selected models should be plotted in \
                a secondary figure.
        x_breaks : list, optional (default: None)
            If selected models are to be plotted, the breaks to use in the \
                histogram

        Returns
        -------
        plot : plotnine.ggplot
            Histogram(s) of model performance made using ggplot
    """
    m = len(perf_metric)
    perf_df = pd.DataFrame(perf_metric, columns=["perf_metric"], index=None)
    plot = (
        pn.ggplot(perf_df, pn.aes(x="perf_metric"))
        + pn.geoms.geom_histogram(
            breaks=np.linspace(np.min(perf_metric), np.max(perf_metric), 40)
        )
        + pn.geoms.geom_vline(xintercept=x_range, linetype="dashed", size=0.7)
        + pn.labels.labs(
            x="Ratio of loss to minimum loss",
            title="""Loss of {m:d} sampled models
                \n{n_elg:d} ({per_elg:.1f}%) sampled models are eligible""".format(
                m=m, n_elg=np.sum(eligible), per_elg=np.sum(eligible) * 100 / m
            ),
        )
        + pn.themes.theme_bw()
        + pn.themes.theme(
            title=pn.themes.element_text(ha="left"),
            axis_title_x=pn.themes.element_text(ha="center"),
            axis_title_y=pn.themes.element_text(ha="center"),
        )
    )
    if plot_selected:
        if select is None:
            print("'select' vector is not specified!\nUsing all models instead")
            select = [i for i in range(len(perf_df))]
        try:
            perf_select = perf_df.iloc[select]
        except:
            print(
                "Invalid indexes detected in 'select' vector!\nUsing all models instead"
            )
            select = [i for i in range(len(perf_df))]
            perf_select = perf_df.iloc[select]
        plot2 = (
            pn.ggplot(perf_select, pn.aes(x="perf_metric"))
            + pn.geoms.geom_histogram(breaks=x_breaks)
            + pn.labels.labs(
                x="Ratio of loss to minimum loss",
                title="{n_select:d} selected models".format(n_select=len(select)),
            )
            + pn.themes.theme_bw()
            + pn.themes.theme(
                title=pn.themes.element_text(ha="left"),
                axis_title_x=pn.themes.element_text(ha="center"),
                axis_title_y=pn.themes.element_text(ha="center"),
            )
        )
        return (plot, plot2)
    else:
        return plot


def plot_distribution(df, s=4):
    num_metrics = df.shape[1] - 2
    labels = df.sen_var_exclusion.unique()
    for i in range(len(labels)):
        if labels[i] == "":
            labels[i] = "No exclusion"
        elif len(labels[i].split("_")) == 2:
            labels[i] = f"Exclusion of {' and '.join(labels[i].split('_'))}"
        elif len(labels[i].split("_")) > 2:
            sens = labels[i].split("_")
            labels[i] = f"Exclusion of {', '.join(sens[:-1])} and {sens[-1]}"
        else:
            labels[i] = f"Exclusion of {labels[i]}"

    fig, axes = plt.subplots(nrows=1, ncols=num_metrics, figsize=(s * num_metrics, s))
    for i, x in enumerate(df.columns[:-2]):
        ax = axes[i]
        # sns.jointplot(data=df, x=x, y="auc", hue="sen_var_exclusion",  ax=ax, legend=False)
        sns.histplot(
            data=df, x=x, hue="sen_var_exclusion", bins=50, ax=ax, legend=False
        )  # layout=(1, num_metrics), figsize=(4, 4), color="#595959",
        ax.set_title(x)
        ax.set_xlabel("")
        ax.set_ylabel("Count" if i == 0 else "")

    plt.legend(
        loc="center left",
        title="",
        labels=labels[::-1],
        ncol=1,
        bbox_to_anchor=(1.04, 0.5),
        borderaxespad=0,
    )
    # plt.tight_layout() bbox_transform=fig.transFigure,
    plt.show()

    return fig


def plot_scatter(df, perf, sen_var_exclusion, title, c1=20, c2=0.15, **kwargs):
    ### basic settings ###
    np.random.seed(0)
    if "figsize" not in kwargs.keys():
        fig_h = 400
        figsize = [fig_h * df.shape[1] * 2.45 / 3, fig_h]
    else:
        figsize = kwargs["figsize"]
    caption_size = figsize[1] / c1  # control font size / figure size
    fig_caption_ratio = 0.8
    fig_font_size = caption_size * fig_caption_ratio

    font_family = "Arial"
    highlight_color = "#D4AF37"
    fig_font_unit = c2  # control the relative position of elements
    caption_font_unit = fig_font_unit * fig_caption_ratio
    d = fig_font_unit / 8
    legend_pos_y = 1 + fig_font_unit
    subtitle_pos = [legend_pos_y + d, legend_pos_y + d + caption_font_unit]
    xlab_pos_y = -fig_font_unit * 2

    area_list = []
    for i, id in enumerate(df.index):
        values = df.loc[id, :]
        area_list.append(1 / compute_area(values))
    ranking = np.argsort(np.argsort(area_list)[::-1])

    # jittering for display
    jitter_control = np.zeros(len(ranking))
    for idx in range(len(ranking)):
        if ranking[idx] == 0:
            jitter_control[idx] = 0
        elif ranking[idx] <= 10 and ranking[idx] != 0:
            jitter_control[idx] = 0.01 * np.random.uniform(0, 1)
        elif ranking[idx] <= 10**2 and ranking[idx] > 10:
            jitter_control[idx] = 0.015 * np.random.uniform(0, 1)
        elif ranking[idx] <= 10**3 and ranking[idx] > 10**2:
            jitter_control[idx] = 0.015 * np.random.uniform(0, 1)
        else:
            jitter_control[idx] = 0.02 * np.random.uniform(-1, 1)

    ### plot ###
    best_id = df.index[np.where(ranking == 0)][0]
    worst_id = df.index[np.argmin(area_list)]
    meduim_id = df.index[np.argsort(area_list)[int(len(area_list) / 2)]]

    num_metrics = df.shape[1]
    num_models = df.shape[0]

    fig = make_subplots(cols=num_metrics, rows=1, horizontal_spacing=0.13)
    cmap = sns.light_palette("steelblue", as_cmap=False, n_colors=df.shape[0])
    cmap = cmap[::-1]
    colors = [rgb01_hex(cmap[x]) if x != 0 else highlight_color for x in ranking]
    sizes = [10 if x != 0 else 20 for x in ranking]

    shapes = sen_var_exclusion.copy().tolist()
    cases = sen_var_exclusion.unique()
    shapes_candidates = ["square", "circle", "triangle-up", "star"][: len(cases)]
    for i, case in enumerate(cases):
        for j, v in enumerate(sen_var_exclusion):
            if v == case:
                shapes[j] = shapes_candidates[i]

        if cases[i] == "":
            cases[i] = "No exclusion"
        elif len(cases[i].split("_")) == 2:
            cases[i] = f"Exclusion of {' and '.join(cases[i].split('_'))}"
        elif len(cases[i].split("_")) > 2:
            sens = cases[i].split("_")
            cases[i] = f"Exclusion of {', '.join(sens[:-1])} and {sens[-1]}"
        else:
            cases[i] = f"Exclusion of {cases[i]}"

    fair_index_df = pd.DataFrame(
        {
            "model id": df.index,
            "fair_index": area_list,
            "ranking": ranking,
            "eod": df["Equalized Odds"],
            "colors": colors,
            "shapes": shapes,
            "sizes": sizes,
            "cases": sen_var_exclusion,
            "jitter": jitter_control,
        }
    )

    # Add scatter plots to the subplots
    for k, s in enumerate(shapes_candidates):
        for i in range(num_metrics):
            # index of sen_var_exclusion(shape) == s
            s_idx = [idx for idx, x in enumerate(shapes) if x == s]
            x = df.iloc[s_idx, i].values
            js = fair_index_df.loc[fair_index_df.shapes == s, "jitter"].values
            jittered_x = x + js

            col = fair_index_df.loc[fair_index_df.shapes == s, "colors"]
            size = fair_index_df.loc[fair_index_df.shapes == s, "sizes"]
            fair_index = fair_index_df.loc[fair_index_df.shapes == s, "fair_index"]
            ids = fair_index_df.loc[fair_index_df.shapes == s, "model id"]
            rank_text = fair_index_df.loc[fair_index_df.shapes == s, "ranking"]
            r = (
                fair_index_df.loc[fair_index_df.shapes == s, "ranking"]
                .apply(lambda x: math.log10(x + 1))
                .values
            )
            sen_case = fair_index_df.loc[fair_index_df.shapes == s, "cases"]

            hovertext = [
                f"Fairness index: {f:.3f}. Ranking: {x}. Model id: {i}"
                for f, x, i in zip(fair_index, rank_text, ids)
            ]
            fig.add_trace(
                go.Scatter(
                    x=r,
                    y=jittered_x,
                    customdata=hovertext,
                    mode="markers",
                    marker=dict(
                        color=col,
                        symbol=s,
                        size=size,
                        line=dict(color=col, width=1),
                        opacity=0.8,
                    ),
                    hovertemplate="%{customdata}.",
                    hoverlabel=None,
                    hoverinfo="name+z",
                    name=cases[k],
                ),
                col=i + 1,
                row=1,
            )

            if i == int((df.shape[1] + 0.5) / 2):
                fig.update_xaxes(
                    title_text=None,
                    tickvals=[0, 1, 2, 3],
                    ticktext=[1, 10, 100, 1000],
                    col=i + 1,
                    row=1,
                    tickangle=0,
                )
            else:
                fig.update_xaxes(
                    title_text=None,
                    tickvals=[0, 1, 2, 3],
                    ticktext=[1, 10, 100, 1000],
                    col=i + 1,
                    row=1,
                    tickangle=0,
                )
            fig.update_yaxes(
                title_text=df.columns[i],
                col=i + 1,
                row=1,
                showticksuffix="none",
                titlefont={"size": caption_size},
            )

            fig.add_vline(
                x=0,
                line_width=2,
                line_dash="dot",
                line_color=highlight_color,
                col=i + 1,
                row=1,
            )

            min_metric = df.loc[ranking == 0, df.columns[i]].values[0]
            max_metric = df.loc[ranking == num_models - 1, df.columns[i]].values[0]
            meduim_metric = df.loc[
                ranking == int(num_models / 2), df.columns[i]
            ].values[0]

            # add annotations
            anno_size = caption_size * 0.7
            if k == 0:
                fig.add_hline(
                    y=min_metric,
                    line_width=2,
                    line_dash="dot",
                    line_color=highlight_color,
                    col=i + 1,
                    row=1,
                )

                # position_y = np.mean(df.iloc[:, i])
                min_annotation = {
                    "x": 0,
                    "y": min_metric,
                    "text": f"Model ID {best_id}<br> Rank No.1",
                    "showarrow": True,
                    "arrowhead": 6,
                    "xanchor": "left",
                    "yanchor": "bottom",
                    "xref": "x",
                    "yref": "y",
                    "font": {"size": anno_size},
                    "ax": -10,
                    "ay": -10,
                    "xshift": 0,
                    "yshift": 0,
                }
                fig.add_annotation(min_annotation, col=i + 1, row=1)
            if meduim_id in ids:
                medium_annotation = {
                    "x": math.log10(int(num_models / 2) + 1),
                    "y": meduim_metric + jitter_control[meduim_id],
                    "text": f"Model ID {meduim_id}<br> Rank No.{int(num_models/2)}",
                    "showarrow": True,
                    "arrowhead": 6,
                    "xanchor": "left",
                    "yanchor": "bottom",
                    "xref": "x",
                    "yref": "y",
                    "font": {"size": anno_size},
                    "ax": -10,
                    "ay": -10,
                    "xshift": 0,
                    "yshift": 0,
                }
                fig.add_annotation(medium_annotation, col=i + 1, row=1)
            if worst_id in ids:
                max_annotation = {
                    "x": math.log10(num_models + 1),
                    "y": max_metric + jitter_control[worst_id],
                    "text": f"Model ID {worst_id}<br> Rank No.{num_models}",
                    "showarrow": True,
                    "arrowhead": 6,
                    "xanchor": "left",
                    "yanchor": "bottom",
                    "xref": "x",
                    "yref": "y",
                    "font": {"size": anno_size},
                    "ax": -10,
                    "ay": -10,
                    "xshift": 0,
                    "yshift": 0,
                    "align": "left",
                }
                fig.add_annotation(max_annotation, col=i + 1, row=1)

    colorbar_trace = go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        hoverinfo="none",
        marker=dict(
            colorscale=[
                rgb01_hex(np.array((243, 244, 245)) / 255),
                "steelblue",
            ],  # "magma",
            showscale=True,
            cmin=0,
            cmax=2,
            colorbar=dict(
                title=None,
                thickness=10,
                tickvals=[0, 2],
                ticktext=["Low", "High"],
                outlinewidth=0,
                orientation="v",
                x=1,
                y=0.5,
            ),
        ),
    )
    fig.add_trace(colorbar_trace)

    fig.update_layout(
        title=title,
        font=dict(family="Arial", size=fig_font_size),
        hovermode="closest",
        width=figsize[0],
        height=figsize[1],
        showlegend=True,
        template="simple_white",
        legend=dict(x=0, y=legend_pos_y, orientation="h"),
    )

    rectangle = {
        "type": "rect",
        "x0": -0.1,
        "y0": subtitle_pos[0],
        "x1": 1.1,
        "y1": subtitle_pos[1],
        "xref": "paper",
        "yref": "paper",
        "fillcolor": "steelblue",
        "opacity": 0.1,
    }  # 'line': {'color': 'red', 'width': 2},
    fig.add_shape(rectangle)
    subtitle_annotation = {
        "x": -0.1,
        "y": subtitle_pos[1],
        "text": f"<i> The FAIM model (i.e., fairness-aware model) is with model ID {best_id}, out of {num_models} nearly-optimal models.</i>",
        "showarrow": False,
        "xref": "paper",
        "yref": "paper",
        "font": {"size": caption_size * 1.1},
        "align": "left",
    }
    xaxis_annotation = {
        "x": 0.5,
        "y": xlab_pos_y,
        "text": "Model Rank",
        "showarrow": False,
        "xref": "paper",
        "yref": "paper",
        "font": {"size": caption_size},
    }
    colorbar_title = {
        "x": 1.05,
        "y": 0.5,
        "text": "Fairness Ranking Index (FRI)",
        "showarrow": False,
        "xref": "paper",
        "yref": "paper",
        "font": {"size": anno_size * 0.9},
        "textangle": 90,
    }
    fig.add_annotation(subtitle_annotation)
    fig.add_annotation(xaxis_annotation)
    fig.add_annotation(colorbar_title)

    for i, trace in enumerate(fig.data):
        if i % num_metrics == 1:
            trace.update(showlegend=True)
        else:
            trace.update(showlegend=False)
    # fig.show()

    return fig, fair_index_df


def plot_radar(df, thresh_show, title, **kwargs):
    fig = go.Figure()
    # fig = sp.make_subplots(rows=1, cols=2)
    cmap = sns.diverging_palette(200, 20, sep=10, s=50, as_cmap=False, n=df.shape[0])
    theta = df.columns.tolist()
    theta += theta[:1]
    area_list = []

    for i, id in enumerate(df.index):
        values = df.loc[id, :]
        area_list.append(compute_area(values))
        values = values.values.flatten().tolist()
        values += values[:1]
        info = [
            f"{theta[j]}: {v:.3f}" for j, v in enumerate(values) if j != len(values) - 1
        ]
        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=theta,
                fill="toself" if id == "FAIReg" else "none",
                text="\n".join(info),
                name=f"{id}",
                line=dict(color=rgb01_hex(cmap[i]), dash="dot"),
            )
        )

    ranking = np.argsort(np.argsort(area_list))
    best_id = df.index[np.where(ranking == 0)][0]
    print(
        f"The best model is No.{best_id} with metrics on validation set:\n {df.loc[best_id, :]}"
    )
    values = df.loc[best_id, :].values.flatten().tolist()
    values += values[:1]
    info = [
        f"{theta[j]}: {v:.3f}" for j, v in enumerate(values) if j != len(values) - 1
    ]
    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=theta,
            fill="toself",
            text="\n".join(info),
            name=f"model {best_id}",
            line=dict(color="royalblue", dash="solid"),
        )
    )

    fig.update_layout(
        # title = title,
        font=dict(family="Arial", size=16),
        polar=dict(
            # bgcolor = "#1e2130",
            radialaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor="lightgray",
                visible=True,
                range=[0, thresh_show],
            )
        ),
        legend=dict(x=0.25, y=-0.1, orientation="h"),
        showlegend=False,
        **kwargs,
    )
    return fig


def plot_bar(
    shap_values, feature_names, original_feature_names, coef=None, title=None, **kwargs
):
    """Plot the bar chart of feature importance"""
    if "color" not in kwargs.keys():
        color = "steelblue"
    else:
        color = kwargs["color"]

    def get_prefix(v):
        if "_" in v and (v not in original_feature_names):
            tmp = ["_".join(v.split("_")[:i]) for i in range(len(v.split("_")))]
            return [s for s in tmp if s in original_feature_names][0]
        else:
            return v

    if shap_values is not None:

        grouped_df = pd.DataFrame({"values": shap_values}, index=feature_names).groupby(
            by=get_prefix, axis=0
        )
        df = {k: np.mean(np.abs(g.values)) for k, g in grouped_df}
        df = pd.DataFrame.from_dict(df, orient="index").reset_index()
        df.columns = ["Var", "Value"]

        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "color": ["grey" if i < 0 else "steelblue" for i in df.Value],
                        "order": np.abs(df.Value),
                    }
                ),
            ],
            axis=1,
        )

    elif coef is not None:
        df = pd.DataFrame(
            {
                "Var": coef.index,
                "Value": coef.values,
                "color": ["grey" if i < 0 else "steelblue" for i in coef.values],
                "order": np.abs(coef.values),
            }
        )
    else:
        raise ValueError("Either shap_value or coef should be provided")

    df = df.loc[df["Var"] != "const", :]
    df = df.sort_values(by="order", ascending=True)
    df["Var"] = pd.Categorical(df["Var"], categories=df["Var"].tolist(), ordered=True)

    common_theme = theme(
        text=element_text(size=24),
        panel_grid_major_y=element_line(colour="lightgrey"),
        panel_grid_minor=element_blank(),
        panel_background=element_blank(),
        axis_line_x=element_line(colour="black"),
        axis_ticks_major_y=element_blank(),
    )

    x_lab = "Feature importance"

    p = (
        ggplot(data=df, mapping=aes(x="Var", y="Value", fill="color"))
        + geom_hline(yintercept=0, color="grey")
        + geom_bar(stat="identity")
        + common_theme
        + coord_flip()
        + labs(x="", y=x_lab, title=title)
        + theme(legend_position="none")
        + scale_fill_manual(values=[color])
    )
    return p

