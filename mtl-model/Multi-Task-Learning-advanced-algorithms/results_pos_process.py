import os
import re
import json
import math
import pandas as pd
from icecream import ic
from config import LOGS_PATH, DATA_PATH, INFO_DATA

class ResultsExtraction:
    def __init__(self):
        self.LOGS_PATH = LOGS_PATH
        self.DATA_PATH = DATA_PATH
        self.INFO_DATA = INFO_DATA
    
    def get_results_files(self):
        return [file for file in os.listdir(self.LOGS_PATH) if file in  ['train-test.csv', 'cross-validation.csv']]
    
    def get_datasets(self):
        return list(set([file.split('_')[0] for file in os.listdir(self.DATA_PATH)]))
    
    def get_number_of_samples(self, file_name): 
        partition = 'test' if 'train-test' in file_name else 'merge'
        dict_data_size = dict()
        
        for dataset in self.get_datasets():
            for file in os.listdir(self.DATA_PATH):
                if all(item in file for item in [dataset, partition, 'processed']):
                    # dict_data_size[self.remove_numbers(dataset)] = len(pd.read_csv(os.path.join(self.DATA_PATH, file))) ## COMMENT: I mat remove REMOVE YEAR
                    dict_data_size[dataset] = len(pd.read_csv(os.path.join(self.DATA_PATH, file)))
                    
        return dict_data_size
    
    def remove_numbers(self, s):
        return re.sub(r'\d+', '', s)
    
    def datasets_ranking_metrics(self):
        ranking_column = dict()
        for dataset in self.get_datasets():
            # dataset_without_year = self.remove_numbers(dataset) ## COMMENT: I mat remove REMOVE YEAR
            # metric = re.sub(r'\b-score\b', '', self.INFO_DATA[dataset_without_year]['metric'], flags=re.IGNORECASE)
            # ranking_column[dataset_without_year] = metric.lower().replace("-", "_") + "_val"
            metric = re.sub(r'\b-score\b', '', self.INFO_DATA[dataset]['metric'], flags=re.IGNORECASE)
            ranking_column[dataset] = metric.lower().replace("-", "_") + "_val"
        return ranking_column
    
    def get_best_models(self, file_name):
        df = pd.read_csv(os.path.join(self.LOGS_PATH,file_name))
        ranking_metric = self.datasets_ranking_metrics()
        output = dict()
        
        for model in df['model'].unique():
            output[model] = {}
            for heads in df.loc[df['model'] == model]['heads'].unique():
                output[model][heads] = {}
                for dataset in df.loc[(df['model'] == model) & (df['heads'] == heads)]['data'].unique():
                    max_value = df.loc[(df['model'] == model) & (df['heads'] == heads) & (df['data'] == dataset)][ranking_metric[dataset]].max()
                    output[model][heads][dataset] = round(max_value,4)
        return output
    
    def add_conf_interval(self, best_models_dict, number_of_samples):
        for model in best_models_dict.keys():
            for heads in best_models_dict[model].keys():
                for dataset, value in best_models_dict[model][heads].items():
                    
                    interval = self.ttest(value, number_of_samples[dataset])
                    best_models_dict[model][heads][dataset] = { "value": value,
                                                                "inter": interval[0],
                                                                "inf": interval[1],
                                                                "sup": interval[2]
                                                                }
        return best_models_dict
    
    def ttest(self, v,n):
        z = 1.96
        SE = math.sqrt((v*(1-v))/n)
        inter = round(z*SE,4)
        inf = round(v - z*SE,4)
        sup = round(v + z*SE,4)
        return [inter, inf, sup]
    
    def flatten_nested_dict_to_dataframe(self, nested_dict):
        flat_data = []

        # Iterate over each model and each head
        for model in nested_dict.keys():
            for heads in nested_dict[model].keys():
                for data, value in nested_dict[model][heads].items():
                    row = {
                            'model': model,
                            'head': heads,
                            'data': data}
                    if type(value) == dict:
                        for metric, value in value.items():
                            row[metric] = value
                    else:
                        row['value'] = value
                            
                    flat_data.append(row)

        # Create DataFrame from the flat data
        df = pd.DataFrame(flat_data)
            
        return df
    
    def save_json_csv(self, dict_table, file_name):
        #Save results as json
        with open(self.LOGS_PATH + '/' + 'summary_' + file_name.split('.')[0] +'.json', 'w') as f:
            json.dump(dict_table, f, indent=4)
        
        #Save results as csv
        df = self.flatten_nested_dict_to_dataframe(dict_table)
        df.to_csv(self.LOGS_PATH + '/' + 'summary_' + file_name.split('.')[0] + '.csv', index=False)
    
    def main(self, winners={}):
        for file_name in self.get_results_files():
            number_of_samples = self.get_number_of_samples(file_name)
            
            best_models_dict = self.get_best_models(file_name)
            if winners and 'train-test' in file_name:
                best_models_dict['WINNERS'] = winners
                
            best_models_dict_plus_conf_interval = self.add_conf_interval(best_models_dict, number_of_samples)
            
            self.save_json_csv(best_models_dict, file_name)
            self.save_json_csv(best_models_dict_plus_conf_interval, file_name + 'plus_intervals')

if __name__ == '__main__':
    Extract_processed_results = ResultsExtraction()
    Extract_processed_results.main(winners={'UM6P-NLP':{'ArMI2021':0.919}, 
                                            'GOF':{'OSACT2022': 0.852}})
