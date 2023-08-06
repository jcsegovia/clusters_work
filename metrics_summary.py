import os
import sys
import pandas as pd
import datetime
from utils import Utils


def print_help():
    print(f'Wrong numer of arguments.')
    print(f'Expected: python launcher.py <directory>')
    print(f'\t e.g: python launcher.py ./test_1')


if len(sys.argv) < 2:
    print_help()
    exit()

DIRECTORY = sys.argv[1]

if not os.path.exists(DIRECTORY):
    raise ValueError(f"Not found source directory {DIRECTORY}")

DIRECTORIES = [
    f'./{DIRECTORY}/AA_618_A59/output',
    f'./{DIRECTORY}/AA_635_A45/output',
    f'./{DIRECTORY}/AA_661_A118/output'
]

MAIN_METRICS_REPORT_DIR = './main_metrics_reports'

if not os.path.exists(MAIN_METRICS_REPORT_DIR):
    os.mkdir(MAIN_METRICS_REPORT_DIR)

now = datetime.datetime.now()
date_formatted = now.strftime("%Y%m%dT%H%M%S")
report_dir = 'main_metrics_report_' + date_formatted
subdir = f'{MAIN_METRICS_REPORT_DIR}/{report_dir}'
os.mkdir(subdir)

df_all = {}
clusters_all = []
hyper_params_all = []
for work_dir in DIRECTORIES:
    report_directories = os.listdir(work_dir)
    for report_dir in report_directories:
        if report_dir.startswith('report_'):
            path = f'{work_dir}/{report_dir}'
            # cluster
            cluster_id = Utils.read_file_by_lines(f'{path}/cluster_id.txt')
            if cluster_id not in clusters_all:
                clusters_all.append(cluster_id)
            # hyperparams
            hyper_params = Utils.read_file_by_lines(f'{path}/hyper_params.txt')
            if hyper_params not in hyper_params_all:
                hyper_params_all.append(hyper_params)
            # dataframe
            csv_data = f'{path}/main_metrics.csv'
            df = pd.read_csv(csv_data)
            df['cluster'] = df['cluster'].apply(lambda x: x+1)
            df_data = {}
            df_data['src'] = df
            df_found = df[df['sources'] == df['found_count']]
            df_data['found'] = df_found
            df_data['new'] = df_found.groupby(by=['model','inc_percent'])[['new_count','cluster','silhouette']].max().reset_index()
            df_data['new_plus'] = df_found.groupby(by=['model','inc_percent'])[['new_in_range_count','cluster','silhouette']].max().reset_index()
            df_all[str(path)] = df_data
            print(report_dir, path)

new_files, new_plus_files = Utils.generate_metrics_summary(df_all, subdir)
Utils.generate_metrics_summary_html(subdir, new_files, new_plus_files, clusters_all, hyper_params_all)

print('Done.')
