import pandas as pd
import numpy as np
from pyecharts.charts import *
import pyecharts.options as opts
import os

def plot_one_y(df, title:str):
    df.index = pd.to_datetime(df['date'])
    df.sort_index(inplace=True)
    df = df.drop(['date'], axis=1)
    return (
        Line(init_opts=opts.InitOpts(width="1200px", height="400px"))
            .add_xaxis(xaxis_data=df.index.strftime('%Y-%m-%d').values.tolist())
            .add_yaxis(
            series_name=title.upper(),
            y_axis=np.round(df['r2'].values, 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .set_global_opts(
            legend_opts=opts.LegendOpts(pos_bottom="0%", pos_right='45%'),
            title_opts=opts.TitleOpts(title=title.upper(), pos_left='0%', ),
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
                                                    opts.MarkPointItem(type_='min', name='Min')]),
        )
    )

def plot_multi_y(df, title:str):
    df.index = pd.to_datetime(df['date'])
    df.sort_index(inplace=True)
    df = df.drop(['date'], axis=1)
    return (
        Line(init_opts=opts.InitOpts(width="1200px", height="400px"))
            .add_xaxis(xaxis_data=df.index.strftime('%Y-%m-%d').values.tolist())
            .add_yaxis(
            series_name=df.columns[0],
            y_axis=np.round(df.values[:,0], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[1],
            y_axis=np.round(df.values[:, 1], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[2],
            y_axis=np.round(df.values[:, 2], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[3],
            y_axis=np.round(df.values[:, 3], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[4],
            y_axis=np.round(df.values[:, 4], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[5],
            y_axis=np.round(df.values[:, 5], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[6],
            y_axis=np.round(df.values[:, 6], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[7],
            y_axis=np.round(df.values[:, 7], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .set_global_opts(
            legend_opts=opts.LegendOpts(pos_bottom="0%", pos_right='25%'),
            title_opts=opts.TitleOpts(title=title.upper(), pos_left='0%', ),
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
                                                    opts.MarkPointItem(type_='min', name='Min')]),
        )
    )


def plot_compare_y(df, df2, title:str):
    df.index = pd.to_datetime(df['date'])
    df.sort_index(inplace=True)
    df = df.drop(['date'], axis=1)
    df2 = df2.drop(['date'], axis=1)
    return (
        Line(init_opts=opts.InitOpts(width="1200px", height="600px"))
            .add_xaxis(xaxis_data=df.index.strftime('%Y-%m-%d').values.tolist())
            .add_yaxis(
            series_name=df.columns[0],
            y_axis=np.round(df.values[:,0], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[1],
            y_axis=np.round(df.values[:, 1], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[2],
            y_axis=np.round(df.values[:, 2], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[3],
            y_axis=np.round(df.values[:, 3], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[4],
            y_axis=np.round(df.values[:, 4], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[5],
            y_axis=np.round(df.values[:, 5], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[6],
            y_axis=np.round(df.values[:, 6], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df.columns[7],
            y_axis=np.round(df.values[:, 7], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df2.columns[0],
            y_axis=np.round(df2.values[:, 0], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df2.columns[1],
            y_axis=np.round(df2.values[:, 1], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df2.columns[2],
            y_axis=np.round(df2.values[:, 2], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df2.columns[3],
            y_axis=np.round(df2.values[:, 3], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df2.columns[4],
            y_axis=np.round(df2.values[:, 4], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df2.columns[5],
            y_axis=np.round(df2.values[:, 5], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df2.columns[6],
            y_axis=np.round(df2.values[:, 6], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .add_yaxis(
            series_name=df2.columns[7],
            y_axis=np.round(df2.values[:, 7], 2).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
            .set_global_opts(
            legend_opts=opts.LegendOpts(pos_bottom="0%", pos_left='10%'),
            title_opts=opts.TitleOpts(title=title.upper(), pos_left='0%', ),
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
                                                    opts.MarkPointItem(type_='min', name='Min')]),
        )
    )


if __name__ == '__main__':
    df = pd.read_csv('result/linear_252/linear_reg.csv')
    ori_reg_plot = plot_one_y(df, 'linear_252 regression r-sqaure of original data')

    files = os.listdir('result/linear_252/')
    name = []
    for file in files:
        if 'normed' in file.split('.')[0].split('_'):
            curr_df = pd.read_csv('result/linear_252/' + file)
            try:
                df = pd.concat([df, curr_df['r2']], axis=1)
                name.append(file.split('.')[0].split('_')[-2]+'-w norm')
            except:
                df = curr_df[['date', 'r2']]
                name.append('date')
                name.append(file.split('.')[0].split('_')[-2]+'-w/ norm')
    df.columns = name
    sort_dict = {key:key.split('-')[1] for key in name if key != 'date'}
    sort_dict = sorted(sort_dict.items(), key=lambda x: int(x[1]))
    name = ['date'] + [item[0] for item in sort_dict]
    df = df[name]
    normed_pca_plot = plot_multi_y(df, 'linear_252 regression r-sqaure after normed pca')

    files = os.listdir('result/linear_252/')
    name = []
    for file in files:
        if 'normed' in file.split('.')[0].split('_'):
            curr_df = pd.read_csv('result/linear_252/' + file)
            try:
                df_pca = pd.concat([df_pca, curr_df['pca_ratio']], axis=1)
                name.append(file.split('.')[0].split('_')[-2]+'-w norm')
            except:
                df_pca = curr_df[['date', 'pca_ratio']]
                name.append('date')
                name.append(file.split('.')[0].split('_')[-2]+'-w/ norm')
    df_pca.columns = name
    sort_dict = {key:key.split('-')[1] for key in name if key != 'date'}
    sort_dict = sorted(sort_dict.items(), key=lambda x: int(x[1]))
    name = ['date'] + [item[0] for item in sort_dict]
    df_pca = df_pca[name]
    pca_plot_normed = plot_multi_y(df_pca, 'normed pca eigenvalue ratio')


    files = os.listdir('result/linear_252/')
    name = []
    for file in files:
        if 'without' in file.split('.')[0].split('_'):
            curr_df = pd.read_csv('result/linear_252/' + file)
            try:
                df2 = pd.concat([df2, curr_df['r2']], axis=1)
                name.append(file.split('.')[0].split('_')[-3]+'-w/o norm')
            except:
                df2 = curr_df[['date', 'r2']]
                name.append('date')
                name.append(file.split('.')[0].split('_')[-3]+'-w/o norm')
    df2.columns = name
    sort_dict = {key:key.split('-')[1] for key in name if key != 'date'}
    sort_dict = sorted(sort_dict.items(), key=lambda x: int(x[1]))
    name = ['date'] + [item[0] for item in sort_dict]
    df2 = df2[name]
    without_norm_pca_plot = plot_multi_y(df2, 'linear_252 regression r-sqaure after pca without norm')

    files = os.listdir('result/linear_252/')
    name = []
    for file in files:
        if 'without' in file.split('.')[0].split('_'):
            curr_df = pd.read_csv('result/linear_252/' + file)
            try:
                df_pca2 = pd.concat([df_pca2, curr_df['pca_ratio']], axis=1)
                name.append(file.split('.')[0].split('_')[-3]+'-w/o norm')
            except:
                df_pca2 = curr_df[['date', 'pca_ratio']]
                name.append('date')
                name.append(file.split('.')[0].split('_')[-3]+'-w/o norm')
    df_pca2.columns = name
    sort_dict = {key:key.split('-')[1] for key in name if key != 'date'}
    sort_dict = sorted(sort_dict.items(), key=lambda x: int(x[1]))
    name = ['date'] + [item[0] for item in sort_dict]
    df_pca2 = df_pca2[name]
    pca_plot_without_norm = plot_multi_y(df_pca2, 'pca without normed eigenvalue ratio')

    files = os.listdir('result/linear_252/')
    name = []
    for file in files:
        if 'normed' in file.split('.')[0].split('_'):
            curr_df = pd.read_csv('result/linear_252/' + file)
            try:
                df = pd.concat([df, curr_df['r2']], axis=1)
                name.append(file.split('.')[0].split('_')[-2]+'-w norm')
            except:
                df = curr_df[['date', 'r2']]
                name.append('date')
                name.append(file.split('.')[0].split('_')[-2]+'-w/ norm')
    df.columns = name
    sort_dict = {key:key.split('-')[1] for key in name if key != 'date'}
    sort_dict = sorted(sort_dict.items(), key=lambda x: int(x[1]))
    name = ['date'] + [item[0] for item in sort_dict]
    df = df[name]
    normed_pca_plot = plot_multi_y(df, 'linear_252 regression r-sqaure after normed pca')
    compare_plot = plot_compare_y(df, df2, 'comparison between pca w/ and w/o norm')

    Page().add(*[ori_reg_plot, normed_pca_plot, pca_plot_normed, without_norm_pca_plot, pca_plot_without_norm, compare_plot]).render(path='render_252.html')