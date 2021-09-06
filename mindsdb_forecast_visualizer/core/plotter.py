import plotly.io as pio
import plotly.graph_objects as go

from mindsdb_forecast_visualizer.config import COLORS


def plot(time, real, predicted, confa, confb, labels, fh_idx, anomalies=None, renderer="browser"):
    """ We use Plotly to generate forecasting visualizations
    """
    pio.renderers.default = renderer  # comment to use default plotter instead of persistent web browser tabs

    fig = go.Figure()

    if confa is not None and confb is not None:
        fig.add_trace(go.Scatter(x=time, y=confa,
                                 name='Confidence',
                                 fill=None,
                                 mode='lines',
                                 line=dict(color=COLORS.SLATEGREY, width=0)))

        fig.add_trace(go.Scatter(x=time, y=confb,
                                 name='Confidence',
                                 fill='tonexty',
                                 mode='lines',
                                 line=dict(color=COLORS.SLATEGREY, width=0)))

    fig.add_trace(go.Scatter(x=time, y=real[:fh_idx],
                             name='Real',
                             line=dict(color=COLORS.SHAMROCK, width=3)))

    fig.add_trace(go.Scatter(x=time, y=[None for _ in range(fh_idx)] + real[fh_idx:],
                             name='Real',
                             line=dict(color=COLORS.SHAMROCK, width=3)))

    fig.add_trace(go.Scatter(x=time, y=predicted,
                             name='Predicted',
                             showlegend=True,
                             line=dict(color=COLORS.BLUEBERRY, width=3)))

    fig.add_vline(x=fh_idx-1, line_width=2, line_dash="dash", line_color="black")

    if anomalies and time:
        for (t_idx, t), anomaly in zip(enumerate(time), anomalies):
            if anomaly:
                t1 = time[t_idx - 1] if t_idx > 0 else t
                t3 = time[t_idx + 1] if t_idx < len(time) - 1 else t
                fig.add_vrect(x0=t1, x1=t3, line_width=0, opacity=0.25, fillcolor=COLORS.WHEAT)  # "orange"

    # @TODO: get dticks from inferred time deltas
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
            # dtick= 1,
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
                color=COLORS.TICKCOLOR,
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
