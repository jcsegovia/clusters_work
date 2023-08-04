import random
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

PLOT_COLOR_FOUND = 'green'
PLOT_COLOR_NEW = 'violet'
PLOT_COLOR_NEW_IN_RANGE = 'orange'
PLOT_COLOR_MISSED = 'red'
PLOT_COLOR_NOISE = 'gray'

# PLOT_COLOR_FOUND = 'green'
# PLOT_COLOR_NEW = 'violet'
# PLOT_COLOR_NEW_IN_RANGE = 'red'
# PLOT_COLOR_MISSED = 'orange'
# PLOT_COLOR_NOISE = 'gray'

# PLOT_COLOR_FOUND = 'green'
# PLOT_COLOR_NEW = 'violet'
# PLOT_COLOR_NEW_IN_RANGE = 'cyan'
# PLOT_COLOR_MISSED = 'red'
# PLOT_COLOR_NOISE = 'gray'

# PLOT_COLOR_FOUND = 'green'
# PLOT_COLOR_NEW = 'cyan'
# PLOT_COLOR_NEW_IN_RANGE = 'orange'
# PLOT_COLOR_MISSED = 'red'
# PLOT_COLOR_NOISE = 'gray'

ALGORITHM_MODELS = ['DBSCAN', 'OPTICS', 'MeanShift']
ALGORITHM_COLORS = ['#53b854', '#3d91c5', '#ff8e00']


class Utils:

    def __init__(self):
        pass

    @staticmethod
    def log_msg(report, msg):
        print(msg)
        report.write(msg + "\n")
        report.flush()

    @staticmethod
    def log_msg_array(report, msg_array):
        for msg in msg_array:
            print(msg)
            report.write(msg + "\n")
        report.flush()

    @staticmethod
    def read_properties(prop_path):
        props = {}
        prop_file = Path(prop_path)
        with prop_file.open() as f:
            while True:
                line = f.readline()
                if not line:
                    break
                key_value = line.strip()
                if not key_value:
                    continue
                if key_value.startswith('#'):
                    continue;
                sep_index = key_value.find('=')
                if sep_index >= 0:
                    key = key_value[0:sep_index].strip()
                    value = key_value[sep_index + 1:].strip()
                # print(f'key: [{key}], value: [{value}]')
                props[key] = value
        return props

    @staticmethod
    def get_argument_value(arguments, key):
        # key: eg.: '-dir'
        if not key.startswith('-'):
            tmp = f'-{key}'
        else:
            tmp = key
        if not key.endswith('='):
            tmp = tmp + '='
        for item in arguments:
            if item.startswith(tmp):
                return item[len(tmp):]
        return None

    @staticmethod
    def overwrite_properties(properties, command_line_arguments):
        for argument in command_line_arguments:
            if argument.startswith('-'):
                p = argument.find('=')
                key = argument[1:p]
                properties[key] = argument[p+1:]
        return properties

    @staticmethod
    def get_arg_multiple(argument):
        a = argument.strip()
        if a[0] == "'":
            a = a[1:]
        if a[-1] == "'":
            a = a[0:len(a)-1]
        if a[0] == '"':
            a = a[1:]
        if a[-1] == '"':
            a = a[0:len(a)-1]
        all_items_arg = a.split(',')
        all_items = []
        for i in all_items_arg:
            all_items.append(i.strip())
        return all_items

    @staticmethod
    def get_html_relative_root(path):
        p = path.rfind('/')
        if p >= 0:
            return f'.{path[p:]}'
        else:
            return f'./{path}'

    @staticmethod
    def get_col_specs(colspecs_arg):
        colspecs = []
        items = colspecs_arg.split(';')
        for item in items:
            i = item.split(',')
            t = (int(i[0].strip()), int(i[1].strip()))
            colspecs.append(t)
        return colspecs

    @staticmethod
    def read_file_by_lines(input_file):
        with open(input_file, 'r') as file:
            return file.readlines()

    @staticmethod
    def generate_sources_for_query(sources_as_serie):
        sources_ids = ""
        for i in range(len(sources_as_serie)):
            if i == 0:
               sources_ids = str(sources_as_serie[0])
            else:
               sources_ids = sources_ids + "," + str(sources_as_serie[i])
        return sources_ids

    @staticmethod
    def check_sources(src_sources, box_sources):
        counter = 0
        not_found_sources = []
        for src in src_sources:
            #print(src)
            if src in box_sources.array:
                counter += 1
            else:
                # print(f'Not found: {src}')
                not_found_sources.append(src)
        return counter, not_found_sources

    @staticmethod
    def get_samples(samples):
        items = samples.split(',')
        res = []
        for i in items:
            res.append(int(i.strip()))
        return res

    @staticmethod
    def do_sample(src_sources, box_data, sample, gaia_source_column):
        # src_sources: array of sources to be increased
        # box_data: gaia box query results
        # sample: percentage to increase
        sources_size = len(src_sources)
        new_size = int(sources_size + (sample * sources_size / 100))
        indexes_found = box_data[box_data[gaia_source_column].isin(src_sources)].index
        box_indexes = np.arange(0, len(box_data), 1).tolist()
        search_subset = random.sample(box_indexes, k=new_size)

        found = []
        for i in range(len(search_subset)):
            if search_subset[i] in indexes_found.values:
                found.append(search_subset[i])

        new_subset = []
        for i in range(len(search_subset)):
            if search_subset[i] not in found:
                new_subset.append(search_subset[i])

        pending_to_add = sources_size - len(found)
        indexes_for_updating = random.sample(np.arange(0, len(new_subset), 1).tolist(), k=pending_to_add)

        index_in_new_subset = []
        for i in indexes_for_updating:
            index_in_new_subset.append(new_subset[i])

        sources_to_add = []
        for i in indexes_found:
            if i not in found:
                sources_to_add.append(i)

        ii = 0
        for i in range(len(search_subset)):
            # print(i, search_subset[i])
            if search_subset[i] in index_in_new_subset:
                search_subset[i] = sources_to_add[ii]
                ii = ii + 1

        return box_data.loc[search_subset]

    @staticmethod
    def generate_main_plot(df, labels, clusters_all_output_file, title):
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        fig, axes = plt.subplots(1,2, figsize=(10,5))
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            xy = df[labels == k]
            axes[0].plot(
                xy['ra'],
                xy['dec'],
                ".",
                alpha=0.7,
                markerfacecolor=tuple(col),
                label='Cluster '+ str(k)
            )
            axes[0].set_xlabel('RA $(deg)$')
            axes[0].set_ylabel('Dec $(deg)$')
            axes[0].legend()
            axes[0].set_title("RA / Dec")

            axes[1].plot(
                xy['pmra'],
                xy['pmdec'],
                ".",
                alpha=0.7,
                markerfacecolor=tuple(col),
                label='Cluster '+ str(k)
            )
            axes[1].set_xlabel('pmRA $(mas,yr^{-1})$')
            axes[1].set_ylabel('pmDec $(mas,yr^{-1})$')
            axes[1].legend()
            axes[1].set_title("pmRA / pmDec")

        fig.suptitle(title)
        # plt.show()
        fig.savefig(clusters_all_output_file)

    @staticmethod
    def generate_cluster_plot(df, sources_df, sources_found_in_cluster, sources_not_found_in_cluster, new_sources,
                              noise_sources, new_sources_in_range, cluster_output_file, title):
        fig, axes = plt.subplots(1,2, figsize=(10,5))
        source_in_cluster_filter = sources_df[sources_df['source_id'].isin(sources_found_in_cluster)]
        source_not_in_cluster_filter = sources_df[sources_df['source_id'].isin(sources_not_found_in_cluster)]
        new_items_cluster_filter = df[df['source_id'].isin(new_sources)]
        noise_filter = df[df['source_id'].isin(noise_sources)]
        new_items_in_range_filter = df[df['source_id'].isin(new_sources_in_range)]

        axes[0].plot(
            noise_filter['ra'],
            noise_filter['dec'],
            ".",
            alpha=0.5,
            label='Noise',
            color=PLOT_COLOR_NOISE
            )
        axes[0].plot(
            new_items_cluster_filter['ra'],
            new_items_cluster_filter['dec'],
            ".",
            alpha=0.5,
            label='New',
            color=PLOT_COLOR_NEW
            )
        axes[0].plot(
            source_in_cluster_filter['ra'],
            source_in_cluster_filter['dec'],
            ".",
            alpha=0.5,
            label='Found',
            color=PLOT_COLOR_FOUND
            )
        axes[0].plot(
            new_items_in_range_filter['ra'],
            new_items_in_range_filter['dec'],
            ".",
            alpha=0.5,
            label='New+',
            color=PLOT_COLOR_NEW_IN_RANGE
            )
        axes[0].plot(
            source_not_in_cluster_filter['ra'],
            source_not_in_cluster_filter['dec'],
            ".",
            alpha=0.5,
            label='Missed',
            color=PLOT_COLOR_MISSED
            )
        axes[0].set_xlabel('RA $(deg)$')
        axes[0].set_ylabel('Dec $(deg)$')
        axes[0].legend()
        axes[0].set_title("RA / Dec")

        axes[1].plot(
            noise_filter['pmra'],
            noise_filter['pmdec'],
            ".",
            alpha=0.5,
            label='Noise',
            color=PLOT_COLOR_NOISE
            )
        axes[1].plot(
            new_items_cluster_filter['pmra'],
            new_items_cluster_filter['pmdec'],
            ".",
            alpha=0.5,
            label='New',
            color=PLOT_COLOR_NEW
            )
        axes[1].plot(
            source_in_cluster_filter['pmra'],
            source_in_cluster_filter['pmdec'],
            ".",
            alpha=0.5,
            label='Found',
            color=PLOT_COLOR_FOUND
            )
        axes[1].plot(
            new_items_in_range_filter['pmra'],
            new_items_in_range_filter['pmdec'],
            ".",
            alpha=0.5,
            label='New+',
            color=PLOT_COLOR_NEW_IN_RANGE
            )
        axes[1].plot(
            source_not_in_cluster_filter['pmra'],
            source_not_in_cluster_filter['pmdec'],
            ".",
            alpha=0.5,
            label='Missed',
            color=PLOT_COLOR_MISSED
            )
        axes[1].set_xlabel('pmRA $(mas,yr^{-1})$')
        axes[1].set_ylabel('pmDec $(mas,yr^{-1})$')
        axes[1].legend()
        axes[1].set_title("pmRA / pmDec")

        fig.suptitle(title)
        # plt.show()
        fig.savefig(cluster_output_file)

    @staticmethod
    def plot_item_metrics(df, field, axis, xlabel, ylabel, title, algorithms, colors):
        for index in range(len(algorithms)):
            model = algorithms[index]
            inc_percents = df[df.model == model]['inc_percent']
            color = colors[index]
            axis.plot(inc_percents, df[df.model == model][field], label=model, color=color)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.legend()

    def plot_metrics_new_items(df, model, field, axis, column, xlabel, ylabel, title, color, model_title):
        inc_percents = df[df.model == model]['inc_percent']
        axis[0][column].plot(inc_percents, df[df.model == model][field], label=f'{model} {model_title}', color=color)
        axis[0][column].set_xlabel(xlabel)
        axis[0][column].set_ylabel(ylabel)
        axis[0][column].set_title(title)
        axis[0][column].legend()
        axis[1][column].plot(inc_percents, df[df.model == model]['silhouette'], label=f'{model} silhouette score',
                             color='#fc345c')
        axis[1][column].set_xlabel(xlabel)
        axis[1][column].set_ylabel('Silhouette score')
        # axis[1][column].set_title(title)
        axis[1][column].legend()

    @staticmethod
    def generate_main_metrics_plot(df, title, metrics_output_file, title_new, metrics_new_output_file,
                                   title_new_plus, metrics_new_plus_output_file):
        df_work = df.copy()
        df_work['cluster'] = df['cluster'].apply(lambda x: x + 1)
        df_clusters = df_work.groupby(by=['model', 'inc_percent'])[['cluster', 'silhouette']].max().reset_index()
        fig, axis = plt.subplots(1, 2, figsize=(14, 5))
        Utils.plot_item_metrics(df, 'cluster', axis[0], '% of increased sources', 'Number of clusters',
                  'Number of clusters per increased set(%)', ALGORITHM_MODELS, ALGORITHM_COLORS)
        Utils.plot_item_metrics(df, 'silhouette', axis[1], '% of increased sources', 'Silhouette score',
                  'Silhouette score per increased set(%)', ALGORITHM_MODELS, ALGORITHM_COLORS)
        fig.suptitle(title)
        # plt.show()
        fig.savefig(metrics_output_file)
        df_found = df_work[df_work['sources'] == df_work['found_count']]
        df_new = df_found.groupby(by=['model', 'inc_percent'])[
            ['new_count', 'cluster', 'silhouette']].max().reset_index()
        df_new_plus = df_found.groupby(by=['model', 'inc_percent'])[
            ['new_in_range_count', 'cluster', 'silhouette']].max().reset_index()
        fig, axis = plt.subplots(2, 3, figsize=(16, 11))
        Utils.plot_metrics_new_items(df_new, 'DBSCAN', 'new_count', axis, 0, '% of increased sources', 'Number of clusters',
                 'New sources per increased set(%)', ALGORITHM_COLORS[0], 'New')
        Utils.plot_metrics_new_items(df_new, 'OPTICS', 'new_count', axis, 1, '% of increased sources', 'Number of clusters',
                 'New sources per increased set(%)', ALGORITHM_COLORS[1], 'New')
        Utils.plot_metrics_new_items(df_new, 'MeanShift', 'new_count', axis, 2, '% of increased sources', 'Number of clusters',
                 'New sources per increased set(%)', ALGORITHM_COLORS[2], 'New')
        fig.suptitle(title_new)
        fig.savefig(metrics_new_output_file)
        fig, axis = plt.subplots(2, 3, figsize=(16, 11))
        Utils.plot_metrics_new_items(df_new_plus, 'DBSCAN', 'new_in_range_count', axis, 0, '% of increased sources',
                 'Number of clusters', 'New+ sources per increased set(%)', ALGORITHM_COLORS[0], 'New+')
        Utils.plot_metrics_new_items(df_new_plus, 'OPTICS', 'new_in_range_count', axis, 1, '% of increased sources',
                 'Number of clusters', 'New+ sources per increased set(%)', ALGORITHM_COLORS[1], 'New+')
        Utils.plot_metrics_new_items(df_new_plus, 'MeanShift', 'new_in_range_count', axis, 2, '% of increased sources',
                 'Number of clusters', 'New+ sources per increased set(%)', ALGORITHM_COLORS[2], 'New+')
        fig.suptitle(title_new_plus)
        fig.savefig(metrics_new_plus_output_file)

    @staticmethod
    def plot_main_metrics_item(df, model, field, axis, row, column, xlabel, ylabel, title, color, model_title, run):
        inc_percents = df[df.model == model]['inc_percent']
        axis[row+0][column].plot(inc_percents, df[df.model == model][field], label=f'#{run} {model} {model_title}', color=color)
        # axis[row+0][column].set_xlabel(xlabel)
        # axis[row+0][column].set_ylabel(ylabel)
        # axis[row+0][column].set_title(title)
        axis[row+0][column].legend()
        c = '#fc345c'
        axis[row+1][column].plot(inc_percents, df[df.model == model]['silhouette'], label=f'#{run} Silhouette score',
                             color=c)
        # axis[row+1][column].set_xlabel(xlabel)
        # axis[row+1][column].set_ylabel('Silhouette score')
        # axis[row+1][column].set_title(title)
        axis[row+1][column].legend()

    @staticmethod
    def plot_main_metrics(model_index, df_all, type, type_column, type_title, main_metrics_summary_file):
        num_runs = len(df_all)
        if num_runs < 3:
            return
        fig, axis = plt.subplots(int(num_runs/3) * 2, 3, figsize=(10, 12))
        col_index = 0
        row_index = 0
        run = 0
        model = ALGORITHM_MODELS[model_index]
        color = ALGORITHM_COLORS[model_index]
        for key in df_all.keys():
            df_data = df_all[key]
            Utils.plot_main_metrics_item(df_data[type], model, type_column, axis, row_index, col_index, '% of increased sources',
                                   'Number of clusters', 'New sources per increased set(%)', color, type_title, run)
            col_index += 1
            if col_index == 3:
                col_index = 0
                row_index += 2
            run = run + 1
        plt.suptitle(model)
        # plt.show()
        fig.savefig(main_metrics_summary_file)

    @staticmethod
    def generate_metrics_summary(df_all, subdir):
        new_files = []
        new_plus_files = []
        for i in range(len(ALGORITHM_MODELS)):
            model = ALGORITHM_MODELS[i]
            main_metrics_dbscan_new_summary_file = f'{subdir}/metrics_summary_{model}_new.png'
            Utils.plot_main_metrics(i, df_all, 'new', 'new_count', 'New', main_metrics_dbscan_new_summary_file)
            print(f'Generated: {main_metrics_dbscan_new_summary_file}')
            new_files.append(main_metrics_dbscan_new_summary_file)

            main_metrics_dbscan_new_plus_summary_file = f'{subdir}/metrics_summary_{model}_new_plus.png'
            Utils.plot_main_metrics(i, df_all, 'new_plus', 'new_in_range_count', 'New+',
                                    main_metrics_dbscan_new_plus_summary_file)
            print(f'Generated: {main_metrics_dbscan_new_plus_summary_file}')
            new_plus_files.append(main_metrics_dbscan_new_plus_summary_file)
        return new_files, new_plus_files

    @staticmethod
    def generate_metrics_summary_html(subdir, new_files, new_plus_files, clusters_all, hyper_params_all):
        html_output_file = f'{subdir}/metrics_summary.html'
        with open(html_output_file, "w") as html:
            html.write("<html>\n")
            html.write("<head>\n")
            html.write("<style>\n")
            html.write("body { font-family: Arial, Helvetica, sans-serif; }\n")
            html.write("</style>\n")
            html.write("</head>\n")
            html.write("<body>\n")
            html.write('<h2>Cluster IDs</h2>')
            for cluster in clusters_all:
                html.write(f"{'<br/>'.join(cluster)}")
                html.write('<br/>\n')
            html.write('<h2>Hyper-params</h2>')
            for hyper_params in hyper_params_all:
                html.write(f"{'<br/>'.join(hyper_params)}")
                html.write('<br/>\n')
            html.write(f'<h2>New sources (100% original sources found)</h2>\n')
            for file in new_files:
                html.write(f'<img src="{Utils.get_html_relative_root(file)}"></br>\n')
            html.write(f'<h2>New+ sources (in range pmRA/pmDec, 100% original sources found)</h2>\n')
            for file in new_plus_files:
                html.write(f'<img src="{Utils.get_html_relative_root(file)}"></br>\n')
            html.write("</body>\n")
            html.write("</html>\n")
        print(f'Generated: {html_output_file}')
        pass

    @staticmethod
    def generate_cluster_data(k, df, sources_df, labels, min_max_pm):
        in_cluster_sources = df[labels == k]['source_id']
        in_cluster_pmra = df[labels == k]['pmra']
        in_cluster_pmdec = df[labels == k]['pmdec']

        sources_found_in_cluster = []
        sources_not_found_in_cluster = []
        for id in sources_df['source_id']:
            if id in in_cluster_sources.values:
                sources_found_in_cluster.append(id)
            else:
                sources_not_found_in_cluster.append(id)

        new_sources = []
        for id in in_cluster_sources.values:
            if id not in sources_df['source_id'].values:
                new_sources.append(id)

        new_sources_in_range = []
        for index in range(len(in_cluster_sources.values)):
            id = in_cluster_sources.values[index]
            pmra = in_cluster_pmra.values[index]
            pmdec = in_cluster_pmdec.values[index]
            if id not in sources_df['source_id'].values and \
                    min_max_pm['pmra_max'] >= pmra >= min_max_pm['pmra_min'] and \
                    min_max_pm['pmdec_max'] >= pmdec >= min_max_pm['pmdec_min']:
                new_sources_in_range.append(id)

        noise_sources = []
        for id in df['source_id'].values:
            if id not in in_cluster_sources.values:
                noise_sources.append(id)

        return sources_found_in_cluster, sources_not_found_in_cluster, new_sources, noise_sources, new_sources_in_range

    @staticmethod
    def generate_html(html_output_file, report_file, output_files, metrics, cluster, model, model_params):
        with open(html_output_file, "w") as html:
            html.write("<html>\n")
            html.write("<head>\n")
            html.write("<style>\n")
            html.write("body { font-family: Arial, Helvetica, sans-serif; }\n")
            html.write("</style>\n")
            html.write("</head>\n")
            html.write("<body>\n")
            # html.write('<center>\n')
            html.write(f'<h2>Report {cluster} {model}</h2>\n')
            html.write(f'<p>Model parameters:</p>')
            html.write(f'<p>{model_params}</p>')
            html.write(f'<br/><br/>')
            for m in metrics:
                html.write(f'<p>{m}')
            html.write(f'<br/><br/><br/><br/>')
            html.write(f'<h2>Plots</h2>\n')
            html.write('<center>\n')
            plot_counter = 1
            for o in output_files:
                if o.find('all_clusters') > 0:
                    html.write(f'<br/><br/><br/><br/>')
                html.write(f'<image src="./{o}"><br/>')
                html.write(f'Plot {plot_counter}<br/><br/><br/><br/>')
                plot_counter = plot_counter + 1
            html.write('</center>\n')
            html.write('<h2>Log</h2>\n')
            html.write(f'<br/><br/>')
            html.write('<p><pre>\n')
            with open(report_file) as report:
                while True:
                    line = report.readline()
                    if not line:
                        break
                    html.write(f'{line}')
            html.write('</pre></p>\n')
            html.write('<p>End log</p>')
            html.write("</body>\n")
            html.write("</html>\n")
        pass

    @staticmethod
    def remove_end_dot(value):
        if value.endswith('.'):
            return value[0:len(value)-1]
        else:
            return value

    @staticmethod
    def extract_content_from_par(data):
        start = data.find('(')
        end = data.find(')')
        if start >= 0 and end >= 0:
            return data[start+1:end]
        else:
            return data

    @staticmethod
    def get_metric_items_cluster(line):
        items = line.split(' ')
        res = []
        res.append(Utils.remove_end_dot(items[5]))
        res.append(Utils.remove_end_dot(items[7]))
        res.append(Utils.extract_content_from_par(Utils.remove_end_dot(items[8])))
        res.append(Utils.remove_end_dot(items[10]))
        news = Utils.extract_content_from_par(Utils.remove_end_dot(items[12])).split('/')
        res.append(news[0])
        res.append(news[1])
        res.append(Utils.remove_end_dot(items[14]))
        return res

    @staticmethod
    def get_metric_title(prev_msg, m):
        items = prev_msg.split(' ')
        p = m.find('-')
        tmp = m[p:]
        silhouette_start = tmp.find('(')
        #silhouette_end = tmp.find(')')
        #sub_title = f'{tmp[0:silhouette_start]}(<b>{tmp[silhouette_start+1:silhouette_end]}</b>)'
        silhouette = Utils.extract_content_from_par(tmp)
        sub_title = f'{tmp[0:silhouette_start]}(<b>{silhouette}</b>)'
        return f'<b>{items[0]} ({items[1]}+{items[2]}%) {items[3]}</b> {sub_title}'

    @staticmethod
    def get_metric_model_title(m):
        items = m.split(' ')
        return f'<b>{items[0]} {items[1]} source(s) - {items[3]}</b>'

    @staticmethod
    def get_popup_windows_name(href):
        tmp = href.replace('/', '_')
        return tmp.replace('.', '_')

    @staticmethod
    def get_model_parameters(model_params):
        lines = []
        buffer = None
        num_keys = 0
        for k in model_params.keys():
            if num_keys == 0:
                buffer = f'{k}: {model_params[k]}'
            else:
                buffer = buffer + f', {k}: {model_params[k]}'
            num_keys = num_keys + 1
            if num_keys > 3:
                lines.append(buffer)
                num_keys = 0
                buffer = ''
        return lines

    @staticmethod
    def generate_metrics_html(html_output_file, metrics, output_files, subdirs, html_reports, model_parameters,
                              metrics_plot_file, metrics_new_plot_file, metrics_new_plus_plot_file):
        with open(html_output_file, "w") as html:
            html.write("<html>\n")
            html.write("<head>\n")
            html.write("<style>\n")
            html.write("body { font-family: Arial, Helvetica, sans-serif; }\n")
            html.write("</style>\n")
            html.write("</head>\n")
            html.write("<body>\n")
            html.write('<h2>Metrics</h2>\n')
            html.write('<table border="1" cellspacing="0" cellpadding="10">\n')
            html.write('<thead>\n')
            html.write('<tr style="background-color: #eeeeee">\n')
            html.write('<th># Cluster</th>\n')
            html.write('<th># Found</th>\n')
            html.write('<th>% Found</th>\n')
            html.write('<th># Missed</th>\n')
            html.write('<th># New</th>\n')
            html.write('<th># New In Range</th>\n')
            html.write('<th># Noise</th>\n')
            html.write('<th>Plot</th>\n')
            html.write('</tr>\n')
            html.write('</thead>\n')
            html.write('<tbody>\n')
            prev_msg = None
            for index in range(len(metrics)):
                m_arr = metrics[index]
                plots = output_files[index]
                subdir = subdirs[index]
                html_report = html_reports[index]
                model_params = model_parameters[index]
                plot_index = 0
                last_title = None
                for i in range(len(m_arr)):
                    m = m_arr[i]
                    html.write('<tr>')
                    if m.find('Estimated') > 0:
                        # sample entry
                        html.write(f'<td colspan="7">{Utils.get_metric_title(prev_msg, m)}</td>')
                        plot = plots[plot_index]
                        plot_index = plot_index + 1
                        href = f'./{subdir}/{plot}'
                        window_name = Utils.get_popup_windows_name(href)
                        args = f"'{href}','{window_name}','width=1000, height=500, menubar=no, toolbar=no'"
                        html.write(f'<td><input type="button" value="Plot" onclick="window.open({args})" title="{href}"/></td>')
                    elif m.find('Cluster') > 0:
                        # Normal cluster entry
                        plot = plots[plot_index]
                        plot_index = plot_index + 1
                        items = Utils.get_metric_items_cluster(m)
                        for item in items:
                            html.write(f'<td>{item}</td>')
                        href = f'./{subdir}/{plot}'
                        # html.write(f'<td><a href="{href}" target="_blank">Plot</a></td>')
                        window_name = Utils.get_popup_windows_name(href)
                        args = f"'{href}','{window_name}','width=1000, height=500, menubar=no, toolbar=no'"
                        html.write(f'<td><input type="button" value="Plot" onclick="window.open({args})" title="{href}"/></td>')
                    else:
                        # Section entry
                        section_entry_title = Utils.get_metric_model_title(m)
                        if last_title is None or section_entry_title != last_title:
                            last_title = section_entry_title
                            # model_param_lines = '<br/>'.join(Utils.get_model_parameters(model_params))
                            # html.write(f'<td colspan="6" style="background-color: #eeeeee">{section_entry_title}<br/>Model parameters:<br/>{model_param_lines}</td>')
                            model_param_lines = '<pre>' + '\n'.join(Utils.get_model_parameters(model_params)) + '</pre>'
                            html.write(f'<td colspan="7" style="background-color: #eeeeee">{section_entry_title}{model_param_lines}</td>')
                            href = f'./{subdir}/{html_report}'
                            window_name = Utils.get_popup_windows_name(href)
                            args = f"'{href}','{window_name}','width=1000, height=1000, menubar=no, toolbar=no'"
                            html.write(f'<td style="background-color: #eeeeee"><input type="button" value="Report" onclick="window.open({args})" title="{href}"/></td>')
                        prev_msg = m
                    html.write('</tr>')
            html.write('</tbody>\n')
            html.write("</table>\n")
            html.write(f'<br/><img src="{Utils.get_html_relative_root(metrics_plot_file)}"><br/>')
            html.write(f'<br/><img src="{Utils.get_html_relative_root(metrics_new_plot_file)}"><br/>')
            html.write(f'<br/><img src="{Utils.get_html_relative_root(metrics_new_plus_plot_file)}"><br/>')
            html.write("</body>\n")
            html.write("</html>\n")
        pass

    @staticmethod
    def generate_hyper_params(output_file, model_names, hyper_params):
        with open(output_file, "w") as hyper_params_file:
            for i in range(len(model_names)):
                hyper_params_file.write(f'{model_names[i]}: {hyper_params[i]}\n')
        pass

    @staticmethod
    def create_cvs_item(line, runid, silhouette):
        items = line.split(' ')
        res = []
        res.append(str(runid))
        res.append(Utils.remove_end_dot(items[0]))
        # res.append(Utils.remove_end_dot(items[1]))
        sources_inc = Utils.extract_content_from_par(Utils.remove_end_dot(items[1]))
        source_inc_items = sources_inc.split('+')
        res.append(source_inc_items[0])
        res.append(source_inc_items[1].replace('%',''))
        res.append(Utils.remove_end_dot(items[2]))
        res.append(Utils.remove_end_dot(items[5]))
        res.append(Utils.remove_end_dot(items[7]))
        res.append(Utils.extract_content_from_par(Utils.remove_end_dot(items[8])).replace('%',''))
        res.append(Utils.remove_end_dot(items[10]))
        news = Utils.extract_content_from_par(Utils.remove_end_dot(items[12])).split('/')
        res.append(news[0])
        res.append(news[1])
        res.append(Utils.remove_end_dot(items[14]))
        res.append(silhouette)
        return res

    @staticmethod
    def get_csv_silhouette(line):
        p = line.find('-')
        tmp = line[p:]
        silhouette = Utils.extract_content_from_par(tmp)
        i = silhouette.find(':')
        return silhouette[i+1:].strip()

    @staticmethod
    def generate_metrics_csv(csv_output_file, metrics, runid):
        with open(csv_output_file, "w") as csv:
            csv.write('runid,id,sources,inc_percent,model,cluster,found_count,found_percent,missed_count,new_count,new_in_range_count,noise_count,silhouette\n')
            for index in range(len(metrics)):
                m_arr = metrics[index]
                for i in range(len(m_arr)):
                    m = m_arr[i]
                    if m.find('Estimated') > 0:
                        silhouette = Utils.get_csv_silhouette(m)
                    if m.find('Cluster') > 0:
                        lines = Utils.create_cvs_item(m, runid, silhouette)
                        csv.write(f'{",".join(lines)}\n')
        pass

    @staticmethod
    def generate_main_csv(sources_df, df, labels, csv_output_file, is_in_range):
        df_classify = pd.DataFrame()
        df_classify['source_id'] = df['source_id']
        df_classify['ra'] = df['ra']
        df_classify['dec'] = df['dec']
        df_classify['pmra'] = df['pmra']
        df_classify['pmdec'] = df['pmdec']
        df_classify['parallax'] = df['parallax']
        df_classify['group'] = labels
        df_classify['member'] = df['source_id'].apply(lambda x: True if x in sources_df['source_id'].values else False)
        df_classify['in_range'] = is_in_range
        df_classify.to_csv(csv_output_file)
        pass

    @staticmethod
    def is_in_range(min_max_pm, df):
        df.apply(lambda x: True if ((x['pmra'] <= min_max_pm['pmra_max']) and \
                                    (x['pmra'] >= min_max_pm['pmra_min']) and \
                                    (x['pmdec'] <= min_max_pm['pmdec_max']) and \
                                    (x['pmdec'] >= min_max_pm['pmdec_min'])) else False, axis=1)

    @staticmethod
    def get_model_params(properties, model_prop, sample):
        params = {}
        Utils.get_model_params_by_key(properties, model_prop, params)
        if sample is not None:
            sample_key = model_prop + "." + sample
            Utils.get_model_params_by_key(properties, sample_key, params)
        return params

    @staticmethod
    def get_model_params_by_key(properties, key, params):
        if key in properties.keys():
            model_properties = properties[key]
            if model_properties is None:
                return params
            items = model_properties.split(',')
            for item in items:
                kv = item.split('=')
                params[kv[0].strip()] = kv[1].strip()
        return params
