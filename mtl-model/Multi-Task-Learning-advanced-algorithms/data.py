import os
import shutil
import pandas as pd
import gdown
from config import *

class DataProcessClass:
    '''Class with useful functions to process data'''
    def __init__(self):
        self.run()
    
    def run(self):            
        self.check_files()
        self.process_data()
        self.merge_data()
        self.sanitization()
            
    def check_files(self):
        missing_file = []
        # check if the dataset in config file exist
        for data in INFO_DATA.keys():
            for file in INFO_DATA[data]['datasets'].values():
                if file not in os.listdir(DATA_PATH):
                    print('File does not exist: {}'.format(file))
                    missing_file.append(file)
        
        if len(missing_file) > 0:
            print('Missing files: {}'.format(missing_file))
            exit()
        else:
            print('\nConfirmed the existence of all datasets from the config file')
    
    def process_data(self):
        # get dataset from config file
        for data in INFO_DATA.keys():
            for file in INFO_DATA[data]['datasets'].values():
                
                # read csv file
                divide_columns = ',' if file[-4:] == '.csv' else '\t'
                df = pd.read_csv(DATA_PATH + '/'  + file, sep = divide_columns)
                
                #remove non-target language text
                if TARGET_LANGUAGE != INFO_DATA[data]['language']:
                    if TARGET_LANGUAGE not in INFO_DATA[data]['language']:
                        print('Target language does not exist in the dataset: {}'.format(file))
                        exit()
                    else:
                        df = df.query('language == "{}"'.format(TARGET_LANGUAGE)).reset_index(drop=True)
                        
                # add task/had column to the dataframe
                df['task'] = INFO_DATA[data]['task']
                
                # convert label to number
                if type(INFO_DATA[data]['positive_class']) == str and not INFO_DATA[data]['positive_class'] == '1':
                    df[INFO_DATA[data]['label_col']] = df[INFO_DATA[data]['label_col']].apply(lambda x: 1 if x == INFO_DATA[data]['positive_class'] else 0)
                    
                # save as a csv file
                df.to_csv(DATA_PATH + '/' + file[:-4] + '_processed.csv', index=False)
                
    def merge_data(self):
        # get data from config file
        for data in INFO_DATA.keys():
            merge_list = []
            
            # create a list with data files to merge
            for file in INFO_DATA[data]['datasets'].values():
                df = pd.read_csv(DATA_PATH + '/'  + file[:-4] + '_processed.csv')
                merge_list.append(df)
                
                
            if len(merge_list) == 0:
                print('No processed files for data: {}'.format(data))
                print('No merging procedure for data: {}'.format(data))
                exit()
            
            elif len(merge_list) > 0:
                
                if len(merge_list) == 1:
                    print('Only one processed file for data: {}'.format(data))
                    print('No merging procedure for data: {}'.format(data))
                
                else:
                    df = pd.concat(merge_list, ignore_index=True)
                    
                
                # remove duplicate instances
                df.drop_duplicates(INFO_DATA[data]['text_col'], inplace=True)
                
                # save merged dataset
                df.to_csv(DATA_PATH + '/' + file.split('_')[0] + '_merge' + '_processed.csv', index=False)
                print('\nMerged file for data: {}'.format(data))


    def sanitization(self):
        # add a space ad the end of test deduplicate samples
        for task in INFO_DATA.keys():
    
            df  = pd.read_csv(DATA_PATH + '/' + str(INFO_DATA[task]['datasets']['test'].split('.')[0]) + '_processed.csv')
            rows = df.duplicated(subset=[INFO_DATA[task]['text_col']], keep='first')
            df.loc[rows, INFO_DATA[task]['text_col']] += (df.loc[rows, INFO_DATA[task]['text_col']] + ' ')
            df.to_csv(DATA_PATH + '/' + str(INFO_DATA[task]['datasets']['test'].split('.')[0]) + '_processed.csv', index=False)


if __name__ == "__main__":
    DataProcessClass()
    print('\nData processing finished!!!!!')