import pandas as pd
from astroquery.gaia import Gaia
from utils import Utils
import datetime
import os
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import  Pipeline
from sklearn.cluster import DBSCAN, OPTICS, MeanShift
from sklearn.metrics import silhouette_score

PROPERTY_SOURCE_INPUT_FILE_COLUMNS = 'SOURCE_INPUT_FILE_COLUMNS'
PROPERTY_SOURCE_INPUT_FILE = 'SOURCE_INPUT_FILE'
PROPERTY_SOURCE_INPUT_FILE_INDEXES = 'SOURCE_INPUT_FILE_INDEXES'
PROPERTY_SOURCE_FILE = 'SOURCE_FILE'
PROPERTY_SOURCE_CLUSTER_COLUMN = 'SOURCE_CLUSTER_COLUMN'
PROPERTY_SOURCE_COLUMN = 'SOURCE_COLUMN'
PROPERTY_GAIA_SOURCE_COLUMN = 'GAIA_SOURCE_COLUMN'
PROPERTY_GAIA_TABLE = 'GAIA_TABLE'
PROPERTY_SAMPLES = 'SAMPLES'


class Processor:

    def __init__(self, properties, directory, main_subdir, main_report, report_dir):
        self.properties = properties
        self.directory = directory
        self.main_subdir = main_subdir
        self.main_report = main_report
        self.report_dir = report_dir

    def prepare(self):
        self.input_file = './' + self.directory + '/' + self.properties[PROPERTY_SOURCE_FILE]
        self.data_source = pd.read_csv(self.input_file)
        self.cluster_column = self.properties[PROPERTY_SOURCE_CLUSTER_COLUMN]
        self.source_column = self.properties[PROPERTY_SOURCE_COLUMN]
        self.gaia_source_column = self.properties[PROPERTY_GAIA_SOURCE_COLUMN]
        self.gaia_table = self.properties[PROPERTY_GAIA_TABLE]

    def get_cluster_data(self, cluster):
        src_cluster_df = self.data_source[self.data_source[self.cluster_column] == cluster]
        src_cluster_data = src_cluster_df[self.source_column]
        Utils.log_msg(self.main_report, f'Found {src_cluster_data.shape[0]} sources for cluster {cluster} in file {self.input_file}\n')
        return src_cluster_data

    def preproc(self):
        columns = []
        for c in self.properties[PROPERTY_SOURCE_INPUT_FILE_COLUMNS].split(','):
            columns.append(c.strip())
        source_input_file = './' + self.directory + '/' + str(self.properties[PROPERTY_SOURCE_INPUT_FILE])
        Utils.log_msg(self.main_report, f'Reading input file: {source_input_file}')
        colspecs = Utils.get_col_specs(self.properties[PROPERTY_SOURCE_INPUT_FILE_INDEXES])
        # Colums size are provided in readme file
        src_df = pd.read_fwf(source_input_file, header=None, colspecs=colspecs)
        src_df.columns = columns
        output_file = './' + self.directory + '/' + self.properties[PROPERTY_SOURCE_FILE]
        Utils.log_msg(self.main_report, f'Saving output in CSV: {output_file}')
        src_df.to_csv(output_file)
        return src_df

    def list_clusters(self):
        clusters = self.data_source.groupby(by=self.cluster_column).count()
        res = clusters.sort_values(by=self.source_column, ascending=False)
        Utils.log_msg(self.main_report, f'Cluster|Num sources')
        for i in range(len(res[self.source_column].index)):
            Utils.log_msg(self.main_report, f'{res[self.source_column].index[i]}\t{res[self.source_column][i]}')
        Utils.log_msg(self.main_report, f'Num clusters: {len(res[self.source_column])}')

    def get_gaia_output_files(self, cluster):
        gaia_output_file = './' + self.directory + '/output/gaia_query_' + cluster + '_' + self.gaia_table + '.csv'
        gaia_box_query_output_file = './' + self.directory + '/output/gaia_box_query_' + cluster + '_' + self.gaia_table + '.csv'
        return gaia_output_file, gaia_box_query_output_file

    def load(self, cluster_data, gaia_output_file, gaia_box_query_output_file):
        src_sources = cluster_data.array
        sources_for_query = Utils.generate_sources_for_query(src_sources)

        # Calculate max min for RA and DEC
        Utils.log_msg(self.main_report, f'Searching sources in Gaia Archive...')
        # Gaia Query to get sources
        # gaia_table = properties['GAIA_TABLE']
        query = 'SELECT source_id, ra, dec, parallax, pmra, pmdec FROM ' + self.gaia_table + ' WHERE source_id IN (' + sources_for_query + ')'
        Utils.log_msg(self.main_report, f'Gaia query: {query}\n')
        res = Gaia.launch_job_async(query)
        data = res.get_data()
        Utils.log_msg(self.main_report, f'Retrieved from Gaia Archive: {len(data)} sources')
        # gaia_output_file =
        data.write(gaia_output_file, format='ascii', delimiter=',', overwrite=True)
        Utils.log_msg(self.main_report, f'Gaia query saved at: {gaia_output_file}\n')

        ra_max = data['ra'].max()
        ra_min = data['ra'].min()
        dec_max = data['dec'].max()
        dec_min = data['dec'].min()
        Utils.log_msg(self.main_report, f'ra_max: {ra_max}, ra_min: {ra_min}, dec_max: {dec_max}, dec_min: {dec_min}\n')
        Utils.log_msg(self.main_report, f'Units: RA/Dec: degrees, pmRA/pmDec: mas yr**-1 (milliarcseconds per Julian year\n')

        # Increase sources
        print(f'Searching by box...\n')

        # Box search
        box_query = "SELECT source_id, ra, dec, parallax, pmra, pmdec FROM " + self.gaia_table + " WHERE ra >= " + str(
            ra_min) + " AND ra <= " + str(ra_max) + " AND dec >= " + str(dec_min) + " AND dec <= " + str(
            dec_max) + " AND pmra IS NOT null AND pmdec IS NOT null"
        Utils.log_msg(self.main_report, f'Gaia Box Search query: {box_query}\n')
        box_res = Gaia.launch_job_async(box_query)
        box_data = box_res.get_data()
        Utils.log_msg(self.main_report, f'\nBox query num sources: {len(box_data)}\n')
        box_data.write(gaia_box_query_output_file, format='ascii', delimiter=',', overwrite=True)
        Utils.log_msg(self.main_report, f'Box query saved at: {gaia_box_query_output_file}\n')

    def check_box_query_sources(self, cluster_data, gaia_box_query_output_file):
        src_sources = cluster_data.array
        box_data = pd.read_csv(gaia_box_query_output_file)

        # Check original sources are in output box file

        print(f'Checking sources are in returned box...')

        box_data = pd.read_csv(gaia_box_query_output_file)

        box_sources = box_data[self.gaia_source_column]
        Utils.log_msg(self.main_report, f'Box sources: {len(box_sources)}')

        found, not_found_sources = Utils.check_sources(src_sources, box_sources)
        Utils.log_msg(self.main_report, f'Expected: {len(src_sources)}, found: {found} -> {"OK" if len(src_sources) == found else "ERROR"}')
        if len(not_found_sources) > 0:
            Utils.log_msg(self.main_report, f'Not found sources: {",".join(not_found_sources)}')
        if len(src_sources) != found:
            raise ValueError(f'Error: not found sources in returned results while processing: {gaia_box_query_output_file}')
        pass

    def sample(self, cluster_data, gaia_box_query_output_file, cluster):
        src_sources = cluster_data.array
        box_data = pd.read_csv(gaia_box_query_output_file)
        samples = Utils.get_samples(self.properties[PROPERTY_SAMPLES])
        for sample in samples:
            sample_output_file = './' + self.directory + '/output/gaia_query_' + cluster + '_' + self.gaia_table +\
                                 '_sample_' + str(sample) + '.csv'
            Utils.log_msg(self.main_report, f'Generating sample file {sample_output_file}')
            df = Utils.do_sample(src_sources, box_data, sample, self.gaia_source_column)
            df.to_csv(sample_output_file)
            found, not_found_sources = Utils.check_sources(src_sources, df[self.gaia_source_column])
            Utils.log_msg(self.main_report, f'Expected: {len(src_sources)}, found: {found} -> {"OK" if len(src_sources == found) else "ERROR"}')
            if len(not_found_sources) > 0:
                Utils.log_msg(self.main_report, f'Not found sources: {",".join(not_found_sources)}')
        pass

    def log_msg(self, low_level_report, msg):
        Utils.log_msg(self.main_report, msg)
        low_level_report.write(msg + "\n")
        low_level_report.flush()

    def apply_model_parameters(self, model, pipeline, sample):
        if 'OPTICS' == model.upper():
            params = Utils.get_model_params(self.properties, 'CLASSIFIER.OPTICS', sample)
            for key in params.keys():
                # algorithm: auto, cluster_method: xi, eps: None, leaf_size: 30
                # max_eps: inf, memory: None, metric: minkowski, metric_params: None
                # min_cluster_size: 0.5, min_samples: 5, n_jobs: None, p: 2
                if 'algorithm' == key:
                    pipeline.named_steps['classifier'].algorithm = params[key]
                elif 'cluster_method' == key:
                    pipeline.named_steps['classifier'].cluster_method = params[key]
                elif 'xi' == key:
                    pipeline.named_steps['classifier'].xi = float(params[key])
                elif 'eps' == key:
                    pipeline.named_steps['classifier'].eps = float(params[key])
                elif 'leaf_size' == key:
                    pipeline.named_steps['classifier'].leaf_size = params[key]
                elif 'max_eps' == key:
                    pipeline.named_steps['classifier'].max_eps = params[key]
                elif 'memory' == key:
                    pipeline.named_steps['classifier'].memory = params[key]
                elif 'metric' == key:
                    pipeline.named_steps['classifier'].metric = params[key]
                elif 'metric_params' == key:
                    pipeline.named_steps['classifier'].metric_params = params[key]
                elif 'min_cluster_size' == key:
                    pipeline.named_steps['classifier'].min_cluster_size = float(params[key])
                elif 'min_samples' == key:
                    pipeline.named_steps['classifier'].min_samples = params[key]
                elif 'n_jobs' == key:
                    pipeline.named_steps['classifier'].n_jobs = params[key]
                elif 'p' == key:
                    pipeline.named_steps['classifier'].p = params[key]
        elif 'DBSCAN' == model.upper():
            params = Utils.get_model_params(self.properties, 'CLASSIFIER.DBSCAN', sample)
            for key in params.keys():
                # algorithm: auto, eps: 0.5, leaf_size: 30, metric: euclidean
                # metric_params: None, min_samples: 5, n_jobs: None, p: None
                if 'algorithm' == key:
                    pipeline.named_steps['classifier'].algorithm = params[key]
                elif 'eps' == key:
                    pipeline.named_steps['classifier'].eps = float(params[key])
                elif 'leaf_size' == key:
                    pipeline.named_steps['classifier'].leaf_size = params[key]
                elif 'metric' == key:
                    pipeline.named_steps['classifier'].metric = params[key]
                elif 'metric_params' == key:
                    pipeline.named_steps['classifier'].metric_params = params[key]
                elif 'min_samples' == key:
                    pipeline.named_steps['classifier'].min_samples = params[key]
                elif 'n_jobs' == key:
                    pipeline.named_steps['classifier'].n_jobs = params[key]
                elif 'p' == key:
                    pipeline.named_steps['classifier'].p = params[key]
        elif 'MEANSHIFT' == model.upper():
            params = Utils.get_model_params(self.properties, 'CLASSIFIER.MEANSHIFT', sample)
            for key in params.keys():
                # bandwidth: None, bin_seeding: False, cluster_all: True, max_iter: 300
                if 'bandwidth' == key:
                    pipeline.named_steps['classifier'].bandwidth = float(params[key])
                elif 'bin_seeding' == key:
                    pipeline.named_steps['classifier'].bin_seeding = params[key]
                elif 'cluster_all' == key:
                    pipeline.named_steps['classifier'].cluster_all = params[key]
                elif 'max_iter' == key:
                    pipeline.named_steps['classifier'].max_iter = params[key]
        pass

    def process_classify(self, subdir, cluster, sample, sources_df, pipeline, report, output_files, metrics, model):
        sample_file = './' + self.directory + '/output/gaia_query_' + cluster + '_' + self.gaia_table + '_sample_' + str(
            sample) + '.csv'
        self.log_msg(report, f'Sources count: {sources_df.shape[0]} +{sample}%\n')
        self.log_msg(report, f'Reading {sample_file}\n')

        metrics.append(f'{cluster} {sources_df.shape[0]} {sample} {model}')

        df = pd.read_csv(sample_file)
        X = df[['pmra', 'pmdec']]
        clustering = pipeline.fit(X)
        labels = clustering.named_steps['classifier'].labels_
        scoring = silhouette_score(X, labels)
        # core_sample_indices = clustering.named_steps['classifier'].core_sample_indices_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        self.log_msg(report, "Estimated number of clusters: %d" % n_clusters_)
        self.log_msg(report, "Estimated number of noise points: %d" % n_noise_)
        self.log_msg(report, f'Silhoutte score: {scoring}')
        self.log_msg(report, '\n')

        clusters_all_output_file_name = 'gaia_query_' + cluster + '_' + self.gaia_table + '_sample_' + str(sample) + '_all_clusters.png'
        output_files.append(clusters_all_output_file_name)
        clusters_all_output_file = subdir + '/' + clusters_all_output_file_name
        self.log_msg(report, f'Main plot {clusters_all_output_file}')

        clusters_all_output_csv_file_name = 'gaia_query_' + cluster + '_' + self.gaia_table + '_sample_' + str(sample) + '_all_clusters.csv'
        clusters_all_output_csv_file = subdir + '/' + clusters_all_output_csv_file_name
        self.log_msg(report, f'Main csv {clusters_all_output_csv_file}')
        min_max_pm = {}
        min_max_pm['pmra_max'] = sources_df['pmra'].max()
        min_max_pm['pmra_min'] = sources_df['pmra'].min()
        min_max_pm['pmdec_max'] = sources_df['pmdec'].max()
        min_max_pm['pmdec_min'] = sources_df['pmdec'].min()
        is_in_range = Utils.is_in_range(min_max_pm, df)
        Utils.generate_main_csv(sources_df, df, labels, clusters_all_output_csv_file, is_in_range)

        main_plot_title = f"{cluster} ({sources_df.shape[0]}+{sample}%) {model} - Estimated number of clusters: {n_clusters_} (Silhouette: {scoring:.4f})"
        metrics.append(main_plot_title)
        Utils.generate_main_plot(df, labels, clusters_all_output_file, main_plot_title)
        self.log_msg(report, '\n')

        cluster_output_file_base = 'gaia_query_' + cluster + '_' + self.gaia_table + '_sample_' + str(sample)
        unique_labels = set(labels)
        for k in unique_labels:
            if k == -1:
                continue
            cluster_output_file_name = cluster_output_file_base + '_cluster_' + str(k) + '.png'
            output_files.append(cluster_output_file_name)
            cluster_output_file = subdir + '/' + cluster_output_file_name
            Utils.log_msg(report, f'Cluster {k} -> {cluster_output_file}')
            sources_found_in_cluster, sources_not_found_in_cluster, new_sources, noise_sources, new_sources_in_range = \
                Utils.generate_cluster_data(k, df, sources_df, labels, min_max_pm)

            percentage = len(sources_found_in_cluster) * 100 / sources_df.shape[0]
            self.log_msg(report,
                    f'Catalogued sources found in cluster {k}: {len(sources_found_in_cluster)} ({percentage:.1f}%)')
            self.log_msg(report, f'Extra elements found: {len(new_sources)}')
            self.log_msg(report, f'Catalogued sources not found in cluster: {len(sources_not_found_in_cluster)}')

            cluster_plot_title = f'{cluster} ({sources_df.shape[0]}+{sample}%) {model} - Cluster {k}. Found: {len(sources_found_in_cluster)} ({percentage:.1f}%). Missed: {len(sources_not_found_in_cluster)}. New: {len(new_sources)}/{len(new_sources_in_range)}. Noise: {len(noise_sources)}'
            metrics.append(cluster_plot_title)
            Utils.generate_cluster_plot(df, sources_df, sources_found_in_cluster, sources_not_found_in_cluster,
                                  new_sources, noise_sources, new_sources_in_range, cluster_output_file, cluster_plot_title)
            self.log_msg(report, '\n')
        return output_files

    def classify_by_model(self, cluster, model, pipeline):
        samples = Utils.get_samples(self.properties[PROPERTY_SAMPLES])

        now = datetime.datetime.now()
        date_formatted = now.strftime("%Y%m%dT%H%M%S")
        # subdir = './' + self.directory + '/output/' + cluster + '_' + model + '_' + date_formatted
        model_subdir = cluster + '_' + model + '_' + date_formatted
        subdir = self.main_subdir + '/' + model_subdir
        os.mkdir(subdir)

        report_file = subdir + '/gaia_query_' + cluster + '_' + self.gaia_table + '_report.txt'
        html_report_name = 'gaia_query_' + cluster + '_' + self.gaia_table + '_report.html'
        html_output_file = subdir + '/' + html_report_name

        sources_file = './' + self.directory + '/output/gaia_query_' + cluster + '_' + self.gaia_table + '.csv'
        sources_df = pd.read_csv(sources_file)

        model_params = pipeline.named_steps['classifier'].get_params()

        output_files = []
        metrics = []
        with open(report_file, 'w') as report:
            Utils.log_msg(report, f'Sources file: {sources_file}\n')
            Utils.log_msg(report, f'Model parameters: {model_params}\n')
            for sample in samples:
                self.process_classify(subdir, cluster, sample, sources_df, pipeline, report, output_files, metrics, model)

        Utils.generate_html(html_output_file, report_file, output_files, metrics, cluster, model, model_params)

        return metrics, output_files, model_subdir, html_report_name, model_params

    def classify_single_model(self, cluster, model):
        if 'DBSCAN' == model.upper():
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', DBSCAN(eps=0.5))
            ])
        elif 'OPTICS' == model.upper():
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', OPTICS(xi=0.05, min_cluster_size=0.5))
            ])
        elif 'MEANSHIFT' == model.upper():
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', MeanShift())
            ])
        else:
            raise  ValueError(f'Unknown model {model}. Available models: DBSCAN, OPTICS and MeanShift')
        self.apply_model_parameters(model, pipeline, None)
        return self.classify_by_model(cluster, model, pipeline)

    def classify(self, cluster, model):
        metrics_html_file = self.main_subdir + '/main_metrics.html'
        metrics_csv_file = self.main_subdir + '/main_metrics.csv'
        metrics_plot_file = self.main_subdir + '/main_metrics_plot.png'
        metrics_new_plot_file = self.main_subdir + '/main_metrics_new_plot.png'
        metrics_new_plus_plot_file = self.main_subdir + '/main_metrics_new_plus_plot.png'
        hyper_params_file = self.main_subdir + '/hyper_params.txt'
        cluster_output_file = self.main_subdir + '/cluster_id.txt'
        if 'all_models' == model:
            metrics_dbscan, output_files_dbscan, subdir_dbscan, html_report_dbscan, model_params_dbscan = self.classify_single_model(cluster, 'DBSCAN')
            metrics_optics, output_files_optics, subdir_optics, html_report_optics, model_params_optics = self.classify_single_model(cluster, 'OPTICS')
            metrics_meanshift, output_files_meanshift, subdir_meanshift, html_report_meanshift, model_params_meanshift = self.classify_single_model(cluster, 'MeanShift')
            Utils.log_msg(self.main_report, f'Multi-model metrics:')
            Utils.log_msg_array(self.main_report, metrics_dbscan)
            Utils.log_msg(self.main_report, "\n")
            Utils.log_msg(self.main_report, f"Params: {model_params_dbscan}\n")
            Utils.log_msg_array(self.main_report, metrics_optics)
            Utils.log_msg(self.main_report, "\n")
            Utils.log_msg(self.main_report, f"Params: {model_params_optics}\n")
            Utils.log_msg_array(self.main_report, metrics_meanshift)
            Utils.log_msg(self.main_report, "\n")
            Utils.log_msg(self.main_report, f"Params: {model_params_meanshift}\n")
            Utils.generate_metrics_csv(metrics_csv_file, [metrics_dbscan, metrics_optics, metrics_meanshift], self.report_dir)
            df_metrics = pd.read_csv(metrics_csv_file)
            Utils.generate_main_metrics_plot(df_metrics, 'Main metrics', metrics_plot_file, 'New Items Metrics',
                                             metrics_new_plot_file, 'New+ Items Metrics', metrics_new_plus_plot_file)
            Utils.generate_metrics_html(metrics_html_file, [metrics_dbscan, metrics_optics, metrics_meanshift],
                                        [output_files_dbscan, output_files_optics, output_files_meanshift],
                                        [subdir_dbscan, subdir_optics, subdir_meanshift],
                                        [html_report_dbscan, html_report_optics, html_report_meanshift],
                                        [model_params_dbscan, model_params_optics, model_params_meanshift],
                                        metrics_plot_file, metrics_new_plot_file, metrics_new_plus_plot_file)
            Utils.generate_hyper_params(hyper_params_file, ['DBSCAN', 'OPTICS', 'MeanShift'],
                                        [model_params_dbscan, model_params_optics, model_params_meanshift])
        else:
            metrics, output_files, subdir, html_report, model_params = self.classify_single_model(cluster, model)
            Utils.log_msg(self.main_report, f'Metrics:')
            Utils.log_msg_array(self.main_report, metrics)
            Utils.log_msg(self.main_report, "\n")
            Utils.log_msg(self.main_report, f"Params: {model_params}\n")
            Utils.generate_metrics_csv(metrics_csv_file, [metrics], self.report_dir)
            df_metrics = pd.read_csv(metrics_csv_file)
            Utils.generate_main_metrics_plot(df_metrics, 'Main metrics', metrics_plot_file, 'New Items Metrics',
                                             metrics_new_plot_file, 'New+ Items Metrics', metrics_new_plus_plot_file)
            Utils.generate_metrics_html(metrics_html_file, [metrics], [output_files], [subdir], [html_report],
                                        [model_params], metrics_plot_file, metrics_new_plot_file,
                                        metrics_new_plus_plot_file)
            Utils.generate_hyper_params(hyper_params_file, [model], [model_params])
        with open(cluster_output_file, "w") as cluster_file:
            cluster_file.write(f'Cluster: {cluster}\n')
        pass

