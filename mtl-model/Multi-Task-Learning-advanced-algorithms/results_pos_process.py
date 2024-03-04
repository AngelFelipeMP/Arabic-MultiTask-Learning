# from config import *
import os
import json
import math
import pandas as pd
from icecream import ic
from config import LOGS_PATH, DATA_PATH, INFO_DATA
import sys
sys.path.append(os.getcwd().split('Arabic-MultiTask-Learning')[0] + 'Arabic-MultiTask-Learning')
# from config import BASELINE_LOGS_PATH

## TODO [ ]: file the functions

class ResultsExtraction:
    def __init__(self):
        pass
    
    def get_results_files(self):
        pass
    
    def get_number_of_samples(self):
        pass
    
    def get_best_models(self):
        INFO_DATA
        pass
    
    def add_conf_interval(self):
        pass
    
    def ttest(self, v,n):
        z = 1.96
        SE = math.sqrt((v*(1-v))/n)
        inter = round(1.96*SE,4)
        inf = round(v - 1.96*SE,4)
        sup = round(v + 1.96*SE,4)
        return inter, inf, sup
    
    def flatten_nested_dict_to_dataframe(nested_dict):
        flat_data = []

        # Iterate over each model and each head
        for model in nested_dict.key():
            for heads in nested_dict[model].keys():
                for data, value in nested_dict[model][heads].items():
                    row = {
                            'model': model,
                            'head': heads,
                            'data': data,
                            'value': value}
                    if value == dict():
                        for metric, value in value.items():
                            row[metric] = value
                            
                    flat_data.append(row)

        # Create DataFrame from the flat data
        df = pd.DataFrame(flat_data)
            
        return df
    
    def save_json_csv(self, dict_table, file_name):
        #Save results as json
        with open(LOGS_PATH + '/' + 'summary_' + file_name +'.json', 'w') as f:
            json.dump(dict_table, f, indent=4)
        
        #Save results as csv
        df = self.flatten_nested_dict_to_dataframe(dict_table)
        df.to_csv(LOGS_PATH + '/' + 'summary_' + file_name + '.csv', index=False)
    
    def main(self):
        for file_name in self.get_results_files(LOGS_PATH):
            number_of_samples = self.get_number_of_samples(DATA_PATH, file_name)
            
            best_models_dict = self.get_best_models(file_name)
            best_models_dict_plus_conf_interval = self.add_conf_interval(best_models_dict, number_of_samples)

            self.save_json_csv(best_models_dict, file_name)
            self.save_json_csv(best_models_dict_plus_conf_interval, file_name + 'plus_intervals')

if __name__ == '__main__':
    Extract_processed_results = ResultsExtraction()
    Extract_processed_results.main()