import os
import sys
import argparse
sys.path.append(os.getcwd().split('Arabic-MultiTask-Learning')[0] + 'Arabic-MultiTask-Learning')
import json
import math
import pandas as pd
from icecream import ic
from config import *

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_results", default="", type=str, help="Experiment results directory")
args = parser.parse_args()

# check information config
if args.experiment_results == '':
    print('Specifying --experiment_results path is required')
    exit(1)

if not os.path.exists('logs' + '/' + args.experiment_results):
    print('The --experiment_results does not exist path')
    print('Enter with a valid path')
    exit(1)

class Analyses:
    '''Class for caculate the confidence interval'''
    def __init__(self, experiment_name, winners={}):
        self.experiment_name = experiment_name
        self.winners = winners
        self.data_path = BASELINE_DATA_PATH
        self.log_path = BASELINE_MACHAMP_LOGS_PATH + '/' + experiment_name
        self.config_path = BASELINE_MACHAMP_CONFIG_PATH + '/' + experiment_name
        
    def number_of_directories(self, path):
        return len([entry.name for entry in os.scandir(self.log_path + '/' + path) if entry.is_dir()])
    
    def check_logs(self):
        self.log_directories = [entry.name for entry in os.scandir(self.log_path) if entry.is_dir()]
        self.number_sub_log_directories = [self.number_of_directories(dir) for dir in self.log_directories]
        
        # Different number of logs sub-directories
        if len(set(self.number_sub_log_directories)) > 1:
            print('Different number of logs sub-directories')
            print('Check the logs directory')
            exit(1)
            
        # numero de subdiretorios different from  config
        with open(self.config_path + '/' + self.experiment_name + '_information_config' + '.json', 'r') as file:
            conf_dict = json.load(file)
            
        if self.number_sub_log_directories[0] > 1  and self.number_sub_log_directories[0] != (conf_dict['folds_number']):
            print('Number of sub-directories differ from config file')
            print('Check the logs directory and the information config file')
            exit(1)
    
    def create_results_dict(self):
        self.results = {item.upper():{} for item in set([dir.split('_')[0] for dir in self.log_directories])}
        for dir_name in self.log_directories:
            model_type, heads = dir_name.split('_')[0].upper(), dir_name.split('_')[1]
                
            if len(heads.split('-')) > 1:
                self.results[model_type][heads] = {h:0 for h in heads.split('-')}
            else:
                self.results[model_type][heads] = 0
                    
    def add_winners_results(self):
        self.results['WINNERS'] = self.winners
    
    def find_results(self, source_file, experiment_type):
        for log_dir in self.log_directories:
            model_type, heads = log_dir.split('_')[0].upper(), log_dir.split('_')[1]
            log_subdir = [entry.name for entry in os.scandir(self.log_path + '/' + log_dir) if entry.is_dir()][0]
            
            #load config -> dataset
            with open(self.log_path + '/' + log_dir + '/' + log_subdir + '/config.json', 'r') as f:
                dict_config = json.load(f)
            dataset_task = [(k, list(v['tasks'].keys())[0]) for k,v in dict_config['dataset_reader']['datasets'].items()]
            
            #load metrics -> results
            extra_path = '/' + log_subdir if experiment_type == 'train-test' else ''
            with open(self.log_path + '/' + log_dir + extra_path + source_file, 'r') as f:
                dict_metrics = json.load(f)
            dataset_task_result =  [l + (v,) for k,v in dict_metrics.items() for l in dataset_task if 'best' in k and l[1] in k]
    
            # add results to dict
            for tup in dataset_task_result:
                head, value = tup[0], tup[2]

                if len(dataset_task_result) > 1:
                    self.results[model_type][heads][head] = value
                else:
                    self.results[model_type][head] = value
            
            #Save results
            with open(self.log_path + '/' + 'summary_' + experiment_type +'.json', 'w') as f:
                json.dump(self.results, f, indent=4)

    def ttest(self, v,n):
        z = 1.96
        SE = math.sqrt((v*(1-v))/n)
        inter = round(1.96*SE,4)
        inf = round(v - 1.96*SE,4)
        sup = round(v + 1.96*SE,4)
        return inter, inf, sup
        
    def sample_size(self, partition, data):
        data_file = [file for file  in os.listdir(self.data_path) if partition in file and data in file][0]
        return pd.read_csv(self.data_path + '/' + data_file, sep='\t').shape[0]
    
    def print_results(self, model_type, heads, head, val, inter, inf, sup):
        print('Model: {}'.format(model_type))
        if heads:
            print('Heads: {}'.format(heads))
        print('Heads: {}'.format(head))
        print('Value: {}'.format(val))
        print('Interval: +/- {}'.format(inter))
        print('Value {} is contained in ({},{})'.format(val,inf,sup))
        print('\n')
        
    def caculate_conf_intervals(self, partition, experiment_type):
        results_plus_interval = self.results.copy()
        
        for model_type, heads_values in self.results.items():
            for heads, value in heads_values.items():
                if isinstance(value, dict):
                    for head, v in value.items():
                        n = self.sample_size(partition, head)
                        i,inf,sup = self.ttest(v,n)
                        results_plus_interval[model_type][heads][head] = {'value': round(v,4), 'inter': i, 'inf': inf, 'sup': sup}
                        self.print_results(model_type, heads, head, v, i, inf, sup)
                        
                else:
                    head, v = heads, value
                    n = self.sample_size(partition, head)
                    i,inf,sup = self.ttest(v,n)
                    results_plus_interval[model_type][head] = {'value': round(v,4), 'inter': i, 'inf': inf, 'sup': sup}
                    self.print_results(model_type, None, head, v, i, inf, sup)
        
        with open(self.log_path + '/' + 'summary_' + experiment_type + '_plus_interval.json', 'w') as f:
            json.dump(results_plus_interval, f, indent=4)
        
    def get_results(self):
        # 1 sub directoery -> train test pipelines
        if max(set(self.number_sub_log_directories)) == 1:
            self.add_winners_results()
            self.find_results('/metrics.json', 'train-test')
            self.caculate_conf_intervals('test', 'train-test')
            
        # more than 1 subdirectory -> cross validation
        elif max(set(self.number_sub_log_directories)) > 1:
            self.find_results('/average.json', 'cross-validation')
            self.caculate_conf_intervals('merge', 'cross-validation')
        
        else:
            print('Error: There is no results in the log subdirectories')
            exit(1)
    
    def main(self):
        self.check_logs() 
        self.create_results_dict()
        self.get_results()


if __name__ == '__main__':
    ANALYSES = Analyses(experiment_name=args.experiment_results,
                        winners={'ArMI2021':0.919, 'OSACT2022':0.852})
    ANALYSES.main()