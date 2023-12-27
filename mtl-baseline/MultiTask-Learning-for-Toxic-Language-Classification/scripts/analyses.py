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
        
        # Diferente number of logs sub-directories
        if len(set(self.number_sub_log_directories)) > 1:
            print('Different number of logs sub-directories')
            print('Check the logs directory')
            exit(1)
            
        # numero de subdiretorios diferente from  config
        with open(self.config_path + '/' + 'EA0_information_config' + '.json', 'r') as file:
            conf_dict = json.load(file)
            
        if self.number_sub_log_directories[0] > 1  and set(self.number_sub_log_directories).pop() != (conf_dict['folds_number']):
            print('Number of sub-directories differ from config file')
            print('Check the logs directory and the information config file')
            exit(1)
    
    def get_results(self):
        results = {}

        ##TODO: STOPED HERE !!!!
        
        # 1 sub directoery -> train test pipelines
        if len(set(self.number_sub_log_directories)) == 1:
            
            
        
        # more than 1 subdirectory -> cross validation
        elif len(set(self.number_sub_log_directories)) > 1:
            
        else:
            print('Error: There is no results in the log subdirectories')
            exit(1)
            
            
    def main(self):
        self.check_logs() 
        self.get_results()


if __name__ == '__main__':
    ANALYSES = Analyses(results=None, 
                        experiment_name=args.experiment_results)
    ANALYSES.main()
    
    ic('WORKING')