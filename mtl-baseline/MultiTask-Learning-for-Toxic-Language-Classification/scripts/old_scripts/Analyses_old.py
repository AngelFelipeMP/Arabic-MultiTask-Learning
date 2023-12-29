import math
import os
import os
import sys
sys.path.append(os.getcwd().split('Arabic-MultiTask-Learning')[0] + 'Arabic-MultiTask-Learning')
import pandas as pd
from icecream import ic
from config import *

class Analyses:
    '''Class for train a mtl model'''
    def __init__(self, results):
        self.results = results
        ##TODO: Remove comments
        # self.path = '/' + '/'.join(os.getcwd().split('/')[1:-2])
        # self.repo_path = '/' + '/'.join(os.getcwd().split('/')[1:-1])
        # self.data_path = self.path + '/data'
        self.data_path = BASELINE_DATA_PATH 
        
    def ttest(self, v,n):
        z = 1.96
        SE = math.sqrt((v*(1-v))/n)
        inf = round(v - 1.96*SE,4)
        sup = round(v + 1.96*SE,4)
        print('Interval: +/- {}'.format(round(1.96*SE,4)))
        print('Value {} is contained in ({},{})'.format(v,inf,sup))
    
    def data(self, term):
        ##TODO: Remove comments
        # val_data = [file for file  in os.listdir(self.data_path) if 'VAL' in file and term in file.lower()]
        val_data = [file for file  in os.listdir(self.data_path) if 'test' in file and term in file.lower()]
        merde_data = [file for file  in os.listdir(self.data_path) if 'merge' in file and term in file.lower()]
        val_data = val_data[0] if val_data else None
        merde_data = merde_data[0] if merde_data else None

        return val_data, merde_data
            
    def conf_interval(self, v, dataset):
        dataset_val, dataset_merge = self.data(dataset)
        
        if dataset_val:
            df_val = pd.read_csv(self.data_path + '/' + dataset_val, sep="\t")
            n_val = df_val.shape[0]
        
        if dataset_merge:
            df_merge = pd.read_csv(self.data_path + '/' + dataset_merge, sep="\t")
            n_merge = df_merge.shape[0]
        
        print('Dataset: {}'.format(dataset_merge))
        print('Instances: '.format(n_merge))
        self.ttest(v,n_merge)
        

    def data_size(self):
        for dataset in self.merde_data:
            df = pd.read_csv(self.data_path + '/' + dataset, sep="\t")
            n = df.shape[0]
            
            print('Dataset: {}'.format(dataset))
            print('Instances: {}'.format(n))
            
if __name__ == '__main__':
    
    results = {
    'WINNERS': {'armi':0.919, 'osact':0.852},
    'STL': {'armi':0.891713268937468, 'hsarabic':0.604895114898681, 'osact':0.774980783462524},
    'MTL':{'armi-hsarabic':{'armi':0.892221657346212, 'hsarabic':0.617238223552703},
    'armi-osact':{'armi':0.889679715302491, 'osact':0.766920208930969},
    'hsarabic-osact':{'hsarabic':0.613496959209442, 'osact':0.767723739147186},
    'hsarabic-osact-armi':{'hsarabic':0.609965682029724, 'osact':0.790371298789978, 'armi':0.884087442806304}}}
    
    Analyses = Analyses(results)
    for model_type in Analyses.results.keys():
        print('####################### {} ######################'.format(model_type))
        
        if model_type == 'STL' or model_type == 'WINNERS':
            for model, performace in Analyses.results[model_type].items():
                print('\nModel: {}'.format(model))             
                # print('Confidence interval: {}'.format(Analyses.conf_interval(performace)))
                Analyses.conf_interval(performace, model)
                print('\n')
    
        else:
            for model in Analyses.results[model_type].keys():
                print('Model: {}'.format(model))
                for head, performace in Analyses.results[model_type][model].items():
                    print('Head: {}'.format(head))
                    # print('Performance: {}'.format(performace))
                    # print('Confidence interval: {}'.format(Analyses.conf_interval(performace)))
                    Analyses.conf_interval(performace, head)
                    print('\n')