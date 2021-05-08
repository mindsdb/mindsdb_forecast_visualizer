import plotly.graph_objects as go
import plotly.io as pio

from mindsdb_forecast_visualizer.config import COLORS


# from ipywidgets import interact, interactive, fixed, interact_manual


def plot(time, real, predicted, confa, confb, labels, anomalies=None):
    """ We use Plotly to generate forecasting visualizations
    """

    # TODO check it works okay
    pio.renderers.default = "browser"  # turn this off to see graphs inline

    fig = go.Figure()

    if confa is not None and confb is not None:
        fig.add_trace(go.Scatter(x=time, y=confa,
                                 name='Confidence',
                                 fill=None,
                                 mode='lines',
                                 # TODO: is this one too strong?
                                 line=dict(color=COLORS.SLATEGREY, width=0)))  # '#919EA5'

        fig.add_trace(go.Scatter(x=time, y=confb,
                                 name='Confidence',
                                 fill='tonexty',
                                 mode='lines',
                                 line=dict(color=COLORS.SLATEGREY, width=0)))

    fig.add_trace(go.Scatter(x=time, y=real,
                             name='Real',
                             line=dict(color=COLORS.SHAMROCK, width=3)))

    fig.add_trace(go.Scatter(x=time, y=predicted,
                             name='Predicted',
                             showlegend=True,
                             line=dict(color=COLORS.BLUEBERRY, width=3)))

    if anomalies:
        for (t_idx, t), anomaly in zip(enumerate(time), anomalies):
            if anomaly:
                t1 = time[t_idx - 1] if t_idx > 0 else t
                t3 = time[t_idx + 1] if t_idx < len(time) - 1 else t
                fig.add_vrect(x0=t1, x1=t3, line_width=0, opacity=0.25, fillcolor=COLORS.WHEAT)  # "orange"

    fig.update_layout(
        xaxis=dict(
            showline=True,
            showgrid=True,
            showticklabels=True,
            gridwidth=1,
            gridcolor=COLORS.GRIDCOLOR,
            linecolor=COLORS.LINECOLOR,
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Source Sans Pro',
                size=14,
                color=COLORS.TICKCOLOR,
            ),
        ),
        yaxis=dict(
            showgrid=True,
            zeroline=True,
            showline=True,
            linecolor=COLORS.LINECOLOR,
            linewidth=2,
            showticklabels=True,
            gridwidth=1,
            gridcolor=COLORS.GRIDCOLOR,
            tickfont=dict(
                family='Source Sans Pro',
                size=14,
                color=COLORS.LINECOLOR,
            ),

        ),
        autosize=True,
        showlegend=True,
        plot_bgcolor='white',
        hovermode='x',

        font_family="Courier New",
        font_color=COLORS.FONTCOLOR,
        title_font_family="Times New Roman",
        title_font_color=COLORS.FONTCOLOR,
        legend_title_font_color=COLORS.FONTCOLOR,

        title=labels['title'],
        xaxis_title=labels['xtitle'],
        yaxis_title=labels['ytitle'],
        legend_title=labels['legend_title'],
    )

    return fig
