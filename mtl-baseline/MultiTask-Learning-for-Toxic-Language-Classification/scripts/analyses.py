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
    def __init__(self, results, experiment_name):
        self.data_path = BASELINE_DATA_PATH
        self.log_path = BASELINE_MACHAMP_LOGS_PATH + '/' + experiment_name
        self.config_path = BASELINE_MACHAMP_CONFIG_PATH + '/' + experiment_name
        
    def number_of_directories(self, path):
        list_of_contents = os.listdir(self.log_path + '/' + path)
        all_directories = [item for item in list_of_contents if os.path.isdir(self.log_path + '/' + path + '/' + item)]
        return len(all_directories)
    
    def check_logs(self):
        logs = os.listdir(self.log_path)
        self.log_directories = [item for item in logs if os.path.isdir(self.log_path + '/' + item)]
        self.number_sub_log_directories = [self.number_of_directories(dir) for dir in self.log_directories]
        
        # Different number of logs sub-directories
        if len(set(self.number_sub_log_directories)) > 1:
            print('Different number of logs sub-directories')
            print('Check the logs directory')
            exit(1)
            
        # numero de subdiretorios different from  config
        with open(self.config_path + '/' + 'EA0_information_config' + '.json', 'r') as file:
            conf_dict = json.load(file)
            
        if self.number_sub_log_directories[0] > 1  and set(self.number_sub_log_directories).pop() != (conf_dict['folds_number']):
            print('Number of sub-directories differ from config file')
            print('Check the logs directory and the information config file')
            exit(1)
    
    def create_results_dict(self):
        self.results = {item.upper():{} for item in set([dir.split('_')[0] for dir in self.log_directories])}
        for model_type, heads in zip([dir.split('_')[0] for dir in self.log_directories], [dir.split('_')[1] for dir in self.log_directories]):
                
            if len(heads.split('-')) > 1:
                self.results[model_type.upper()][heads] = {h:0 for h in heads.split('-')}
            else:
                self.results[model_type.upper()][heads] = 0
                    
        # add manualy winnners
        self.results['WINNERS'] = {'ArMI2021':0.919, 'OSACT2022':0.852}
    
    def get_results(self):
        # 1 sub directoery -> train test pipelines
        if max(set(self.number_sub_log_directories)) == 1:
            for log_dir in self.log_directories:
                log_subdir = os.listdir(self.log_path + '/' + log_dir)[0]
                
                with open(self.log_path + '/' + log_dir + '/' + log_subdir + '/config.json', 'r') as f:
                    dict_config = json.load(f)
                dataset_task = [(k, list(v['tasks'].keys())[0]) for k,v in dict_config['dataset_reader']['datasets'].items()]
                
                #load metrics -> results
                with open(self.log_path + '/' + log_dir + '/' + log_subdir + '/metrics.json', 'r') as f:
                    dict_metrics = json.load(f)
                dataset_task_result =  [l + (v,) for k,v in dict_metrics.items() for l in dataset_task if 'best' in k and l[1] in k]
        
                # add results to dict
                for tup in dataset_task_result:
                    if len(dataset_task_result) > 1:
                        self.results[log_dir.split('_')[0].upper()][log_dir.split('_')[1]][tup[0]] = tup[2]
                    else:
                        self.results[log_dir.split('_')[0].upper()][tup[0]] = tup[2]
                
                #Save results
                with open(self.log_path + '/' + 'summary_train-test' + '.json', 'w') as f:
                    json.dump(self.results, f, indent=4)
            
            
        # more than 1 subdirectory -> cross validation
        ##TODO: I STOPED HERE -> I NEED TO code bellow &b try the if-else
        elif max(set(self.number_sub_log_directories)) > 1:
            for log_dir in self.log_directories:
                log_subdir = os.listdir(self.log_path + '/' + log_dir)[0]
                
                #load config
                with open(self.log_path + '/' + log_dir + '/' + log_subdir + '/config.json', 'r') as f:
                    dict_config = json.load(f)
                dataset_task = [(k, list(v['tasks'].keys())[0]) for k,v in dict_config['dataset_reader']['datasets'].items()]
                
                #load metrics -> results
                with open(self.log_path + '/' + log_dir + '/average.json', 'r') as f:
                    dict_metrics = json.load(f)
                dataset_task_result =  [l + (v,) for k,v in dict_metrics.items() for l in dataset_task if 'best' in k and l[1] in k]
        
                # add results to dict
                for tup in dataset_task_result:
                    if len(dataset_task_result) > 1:
                        self.results[log_dir.split('_')[0].upper()][log_dir.split('_')[1]][tup[0]] = tup[2]
                    else:
                        self.results[log_dir.split('_')[0].upper()][tup[0]] = tup[2]
                
                #Save results
                with open(self.log_path + '/' + 'summary_cross-validation' + '.json', 'w') as f:
                    json.dump(self.results, f, indent=4)

        else:
            print('Error: There is no results in the log subdirectories')
            exit(1)
            
            
    def main(self):
        self.check_logs() 
        self.create_results_dict()
        self.get_results()


if __name__ == '__main__':
    ANALYSES = Analyses(results=None, 
                        experiment_name=args.experiment_results)
    ANALYSES.main()
    
    ic('WORKING')