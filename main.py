import sys
import os
from utils import Utils
from processor import Processor
import datetime

CONFIG_PROPERTIES_FILE = 'config.properties'

TASK_LIST_CLUSTERS = 'list'
TASK_PREPROC = 'preproc'
TASK_LOAD = 'load'
TASK_SAMPLING= 'sampling'
TASK_CLASSIFY='classify'
TASK_ALL='all'
TASKS_ALL=[TASK_PREPROC, TASK_LOAD, TASK_SAMPLING, TASK_CLASSIFY]


def print_help():
    print(f'Wrong numer of arguments.')
    print(f'Expected: python main.py -dir=<dir> -task=<task> -cluster=<cluster> -model=<model> -sample=<sample>')
    print(f'\t 0) preproc required only ONCE')
    print(f'\t eg.: python main.py -dir=./test_1/AA_618_A59 -task=preproc')
    print(f'\t eg.: python main.py -dir=./test_1/AA_635_A45 -task=preproc')
    print(f'\t eg.: python main.py -dir=./test_1/AA_661_A118 -task=preproc')
    print(f'\t Once preproc has been executed')
    print(f'\t 1) To list available clusters')
    print(f'\t eg.: python main.py -dir=./test_1/AA_618_A59 -task=list')
    print(f'\t eg.: python main.py -dir=./test_1/AA_635_A45 -task=list')
    print(f'\t eg.: python main.py -dir=./test_1/AA_661_A118 -task=list')
    print(f'\t 2) Normal run (load,sampling,classify). Several clusters can be specified:')
    print(f'\t eg.: python main.py -dir=./test_1/AA_618_A59 -task="load,sampling,classify" -cluster="UBC17_a"')
    print(f'\t eg.: python main.py -dir=./test_1/AA_618_A59 -task=all -cluster=UBC17_a (all=load,sampling,classify using DBSCAN)')
    print(f'\t eg.: python main.py -dir=./test_1/AA_618_A59 -task=classify -cluster=UBC17_a -model=dbscan (classify with DBSCAN)')
    print(f'\t eg.: python main.py -dir=./test_1/AA_618_A59 -task=classify -cluster=UBC17_a -model=all_models (classify with dbscan, optics and MeanShift)')
    print(f'\t eg.: python main.py -dir=./test_1/AA_618_A59 -task="sampling,classify" -cluster=UBC17_a -model=OPTICS -SAMPLES=300')
    print(f'\t eg.: python main.py -dir=./test_1/AA_618_A59 -task=classify -cluster=UBC17_a -model=OPTICS -SAMPLES=300 -CLASSIFIER.OPTICS="min_cluster_size=0.5,xi=0.05"')


if len(sys.argv) < 3:
    print_help()
    raise ValueError('Missing required arguments')

# args
DIR_ARG = Utils.get_argument_value(sys.argv, '-dir=')
if DIR_ARG is None:
    print_help()
    raise ValueError('Missing "dir" argument')

TASK_ARG = Utils.get_argument_value(sys.argv, '-task=')
if TASK_ARG is None:
    print_help()
    raise ValueError('Missing "task" argument')

if 'all' == TASK_ARG or '*' == TASK_ARG:
    TASK_ARG = TASK_LOAD + ',' + TASK_SAMPLING + ',' + TASK_CLASSIFY

CLUSTER_ARG = Utils.get_argument_value(sys.argv, '-cluster=')
MODEL_ARG = Utils.get_argument_value(sys.argv, '-model=')
if MODEL_ARG is None:
    MODEL_ARG = 'DBSCAN'
elif 'all' == MODEL_ARG or '*' == MODEL_ARG:
    MODEL_ARG = 'all_models'

now = datetime.datetime.now()
date_formatted = now.strftime("%Y%m%dT%H%M%S")
report_dir = 'report_' + date_formatted
subdir = './' + DIR_ARG + '/output/' + report_dir
os.mkdir(subdir)
main_report_file = subdir + '/main_report_' + date_formatted

with open('./execs_cmd.log', 'a') as runs_log:
    run_date_formatted = now.strftime("%Y-%m-%d T %H:%M:%S")
    runs_log.write(f'Run: {run_date_formatted}. Comand line: {" ".join(sys.argv)} -> Report dir: {subdir}\n')


with open(main_report_file, 'w') as main_report:
    Utils.log_msg(main_report, f'Command line: {" ".join(sys.argv)}')

    prop_file = './' + DIR_ARG + '/' + CONFIG_PROPERTIES_FILE
    Utils.log_msg(main_report, f'Using properties file: {prop_file}')
    properties = Utils.read_properties(prop_file)

    # Overwrite properties with command line arguments
    properties = Utils.overwrite_properties(properties, sys.argv)

    processor = Processor(properties, DIR_ARG, subdir, main_report, report_dir)

    all_tasks = Utils.get_arg_multiple(TASK_ARG)

    if TASK_PREPROC in all_tasks:
        processor.preproc()
        if len(all_tasks) == 1:
            exit()

    processor.prepare()

    if TASK_LIST_CLUSTERS in all_tasks:
        processor.list_clusters()
        if len(all_tasks) == 1:
            exit()

    if CLUSTER_ARG is None:
        Utils.log_msg(main_report, 'Mandatory cluster argument nof found')
        print_help()
        raise ValueError('Mandatory cluster argument nof found')

    cluster = CLUSTER_ARG
    Utils.log_msg(main_report, f'Clusters to process: {cluster}')

    print(f'\nProcessing cluster {cluster} for {DIR_ARG}')

    src_cluster_data = processor.get_cluster_data(cluster)
    gaia_output_file, gaia_box_query_output_file = processor.get_gaia_output_files(cluster)

    if TASK_LOAD in all_tasks:
        processor.load(src_cluster_data, gaia_output_file, gaia_box_query_output_file)
        processor.check_box_query_sources(src_cluster_data, gaia_box_query_output_file)
    if TASK_SAMPLING in all_tasks:
        processor.sample(src_cluster_data, gaia_box_query_output_file, cluster)
    if TASK_CLASSIFY in all_tasks:
        processor.classify(cluster, MODEL_ARG)

print('\nDone.')
