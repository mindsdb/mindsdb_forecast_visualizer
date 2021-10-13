import plotly.io as pio
import plotly.graph_objects as go

from mindsdb_forecast_visualizer.config import COLORS, RENDERER


def plot(time,
         real,
         predicted,
         confa,
         confb,
         labels,
         fh_idx,
         renderer='browser',
         anomalies=None,
         separate=False):

    pio.renderers.default = renderer
    fig = go.Figure()

    cutoffs = [
        (0, fh_idx),                    # plot real & predicted up until latest observed data point
        (fh_idx, None)                  # plot predicted forecasts starting latest observed data point
    ] if separate else [(0, None)]

    preambles = [
        [],                             # offset list for first cutoff (None, starts from beginning)
        [None for _ in range(fh_idx)]   # offset list for second cutoff
    ] if separate else [[]]

    for preamble, (start_idx, end_idx) in zip(preambles, cutoffs):

        if confa is not None and confb is not None:
            fig.add_trace(go.Scatter(x=time,
                                     y=preamble + confa[start_idx:end_idx],
                                     name='Confidence',
                                     fill=None,
                                     mode='lines',
                                     line=dict(
                                         color=COLORS.SLATEGREY,
                                         width=0)))

            fig.add_trace(go.Scatter(x=time,
                                     y=preamble + confb[start_idx:end_idx],
                                     name='Confidence',
                                     fill='tonexty',
                                     mode='lines',
                                     line=dict(
                                         color=COLORS.SLATEGREY,
                                         width=0)))

        fig.add_trace(go.Scatter(x=time,
                                 y=preamble + real[start_idx:end_idx],
                                 name='Real',
                                 line=dict(
                                     color=COLORS.SHAMROCK,
                                     width=3)))

        fig.add_trace(go.Scatter(x=time,
                                 y=preamble + predicted[start_idx:end_idx],
                                 name='Predicted',
                                 showlegend=True,
                                 line=dict(
                                     color=COLORS.BLUEBERRY,
                                     width=3)))

    fig.add_vline(x=time[fh_idx-1], line_width=2, line_dash="dash", line_color="black")  # mark start of forecasting window

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
