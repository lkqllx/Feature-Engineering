import pandas as pd
import numpy as np
from pyecharts.charts import *
import pyecharts.options as opts


def plot(df, title:str):
    df.index = df['date']
    df = df.drop(['date'], axis=1)
    return (
        Line(init_opts=opts.InitOpts(width="1200px", height="400px"))
            .add_xaxis(xaxis_data=df.index.strftime('%Y-%m-%d').values.tolist())
            .add_yaxis(
            series_name=title.upper(),
            y_axis=np.round(df['r2'].values, 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2)
        )
            .set_global_opts(
            legend_opts=opts.LegendOpts(pos_bottom="0%", pos_right='25%'),
            title_opts=opts.TitleOpts(title=title.upper(), pos_right='48%', ),
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
            toolbox_opts=opts.ToolboxOpts(is_show=True),
            xaxis_opts=opts.AxisOpts(boundary_gap=False),
            yaxis_opts=opts.AxisOpts(
                axislabel_opts=opts.LabelOpts(formatter="{value}"),
                splitline_opts=opts.SplitLineOpts(is_show=True),
            ),
        )
            .set_series_opts(
            markpoint_opts=opts.MarkPointOpts(data=[opts.MarkPointItem(type_='max', name='Max'),
                                                    opts.MarkPointItem(type_='min', name='Min')])
        )
    )

if __name__ == '__main__':
    df = pd.read_csv('linear_reg.csv')
    plot(df, 'linear regression r-sqaure')