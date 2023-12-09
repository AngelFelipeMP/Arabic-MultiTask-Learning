import pandas as pd
from config import *
from icecream import ic
from sklearn.model_selection import train_test_split
import math
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

class ArMI2021:
    '''Class for ArMI_2021 dataset'''
    def __init__(self):
        self.df_training = self.read_tsv('training')
        self.df_test = self.read_tsv('test')
        self.df_labels = self.read_tsv('labels')

    def read_tsv(self, file):
        return pd.read_csv(ORIGINAL_DATA_PATH + '/' + ArMI_2021['directory']+ '/' + ArMI_2021[file], sep='\t', index_col= 'tweet_id')
    
    def add_labels_to_test(self):
        self.df_test_with_labels = pd.merge(self.df_test, self.df_labels, on='tweet_id')
        
    def join_train_test(self):
        self.df_training['data'], self.df_test_with_labels['data'] = 'training', 'test'
        self.df_merge = pd.concat([self.df_training, self.df_test_with_labels])
        
    def summary(self):
        print('******************')
        print('**** ArMI2021 ****')
        print('******************')
        for split in ['training', 'test_with_labels', 'merge']:
            df = getattr(self, 'df_' + split)
            ic(df.columns)
            ic(df.index)
            print(f'Number of samples in {split}: {df.shape[0]}')
            print(f'Label distribution in {split}:')
            distribution = df['misogyny'].value_counts(normalize=True).to_dict()
            for k,v in distribution.items():
                print(f'{k}: {v}')
            print()
    
    def main(self):
        self.add_labels_to_test()
        self.join_train_test()
        self.summary()
        

class OSACT2022:
    ##TODO: create a new data split/slot "train_deve" it will be training data (80% of the whole data)
    '''Class for ArMI_2021 dataset'''
    def __init__(self):
        self.df_training = self.read_tsv('training')
        self.df_dev = self.read_tsv('dev')
        self.df_test = self.read_tsv('test')
        self.df_labels = self.read_tsv('labels')

    def map_labels_FGHS_to_HS(self):
        self.df_training['Hate-Speech'] = self.df_training ['Fine-Grained-Hate-Speech'].apply(lambda x: 'NOT_HS' if x == 'NOT_HS' else 'HS')
        self.df_dev['Hate-Speech'] = self.df_dev['Fine-Grained-Hate-Speech'].apply(lambda x: 'NOT_HS' if x == 'NOT_HS' else 'HS')
        
    def read_tsv(self, file):
        col_names = ['tweet_id', 'text', 'Offencive-Language', 'Fine-Grained-Hate-Speech', 'vulgar', 'violence']
        columns = ['Hate-Speech'] if file == 'labels' else  (col_names[:2] if file == 'test' else col_names)
        
        df = pd.read_csv(ORIGINAL_DATA_PATH + '/' + OSACT_2022['directory']+ '/' + OSACT_2022[file], 
                                sep='\t', 
                                header=None,
                                names=columns,
                                index_col= 'tweet_id' if file != 'labels' else None)
        if file == 'labels':
            df.index = df.index + 10158
            
        return df
    
    def add_labels_to_test(self):
        self.df_test_with_labels = pd.concat([self.df_test, self.df_labels], axis=1)
        
    def join_train_test(self):
        self.df_training['data'], self.df_dev['data'], self.df_test_with_labels['data'] = 'training', 'dev', 'test'
        self.df_merge = pd.concat([self.df_training, self.df_dev, self.df_test])
        
    def summary(self):
        print('******************')
        print('**** OSACT202 ****')
        print('******************')
        for split in ['training', 'dev', 'test_with_labels', 'merge']:
            df = getattr(self, 'df_' + split)
            ic(df.columns)
            ic(df.index)
            print(f'Number of samples in {split}: {df.shape[0]}')
            print(f'Label distribution in {split}:')
            distribution = df['Hate-Speech'].value_counts(normalize=True).to_dict()
            for k,v in distribution.items():
                print(f'{k}: {v}')
            print()
    
    def main(self):
        self.add_labels_to_test()
        self.map_labels_FGHS_to_HS()
        self.join_train_test()
        self.summary()

        
class HSARABIC:
    '''Class for HSARABIC dataset'''
    def __init__(self):
        self.df = self.read_tsv('single_partition')

    def read_tsv(self, file):
        return pd.read_excel(ORIGINAL_DATA_PATH + '/' + HSArabic['directory']+ '/' + HSArabic[file],
                                usecols=['NUM', 'info_text', 'Q4.1: Offensive/Vulgar لغة مسيئة/بذيئة'],
                                index_col= 'NUM',
                                engine='openpyxl')
    
    def preprocess(self):
        self.df = self.df.rename(columns={'info_text': 'text', 'Q4.1: Offensive/Vulgar لغة مسيئة/بذيئة': 'offensive_vulgar'})
        self.df = self.df[self.df.index.notna()] # Remove the instances with empty indexes: 2
        self.df.index = self.df.index.astype('int64') # Convert the indexes from float64 to int64
        self.df.index.name = 'tweet_id' # Rename the index column
        self.df = self.df.dropna() # Remove the instances with empty annotations/labels: 596
        self.df = self.df[self.df['offensive_vulgar'] != 'no-not-directed'] # Remove the instances with labels that appear only once: 1
        self.df = self.df[self.df['offensive_vulgar'] != 'neutral-or-combination'] # Remove the instances with labels that appear only once: 1
        self.df['offensive_vulgar'] = self.df['offensive_vulgar'].apply(lambda x: x.upper()) # Normalise the labels
    
    def split_data(self):
        self.df_training, self.df_test_with_labels = train_test_split(self.df, test_size=0.2, stratify=self.df['offensive_vulgar'], random_state=42)
        
    def join_train_test(self):
        self.df_training['data'], self.df_test_with_labels['data'] = 'training', 'test'
        self.df_merge = pd.concat([self.df_training, self.df_test_with_labels])
        
    def summary(self):
        print('******************')
        print('**** HSARABIC ****')
        print('******************')
        for split in ['training', 'test_with_labels', 'merge']:
            df = getattr(self, 'df_' + split)
            ic(df.columns)
            ic(df.index)
            print(f'Number of samples in {split}: {df.shape[0]}')
            print(f'Label distribution in {split}:')
            distribution = df['offensive_vulgar'].value_counts(normalize=True).to_dict()
            for k,v in distribution.items():
                print(f'{k}: {v}')
            print()
    
    def main(self):
        self.preprocess()
        self.split_data()
        self.join_train_test()
        self.summary()
        
            
if __name__ == '__main__':
    print('\n')
    
    ArMI_2021 = ArMI2021()
    ArMI_2021.main()
    
    OSACT2022 = OSACT2022()
    OSACT2022.main()
    
    HSArabic = HSARABIC()
    HSArabic.main()
        

    
##TODO: remove ic() from the code
##TODO: rewrite in in notes
# Regarding the HSArabicDataset, there are some inconsistencies in the data:
# For example, for Hate Speech, the same label was written in different 
# ways: Yes, Yes, and YES.
# There are two labels that appear only once: ‘no-not-directed’ and 
# ‘Neutral-or-combination’.
# In addition, there are a few missing values. So, whatever label/task 
# we use from this dataset,
# we will need to clean it.

# There are only two binary tasks embedded in the HSArabicDataset, which are
#    - Q4.1: Offensive/Vulgar


#### DAMIANO
# I'd use the following to pre-process the dataset:
# ignore instances with empty annotations (there is a filtering task before the ones we're aiming for)
# normalise the labels by applying lowercasing and keeping the first character as label ('y', 'n').