import pandas as pd
from config import *
from icecream import ic
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

class ArMI2021:
    '''Class for ArMI_2021 dataset'''
    def __init__(self):
        self.df_training = self.read_tsv('training')
        self.df_test = self.read_tsv('test')
        self.df_labels = self.read_tsv('labels')
        self.main()

    def read_tsv(self, file):
        return pd.read_csv(ORIGINAL_DATA_PATH + '/' + ArMI_2021['directory']+ '/' + ArMI_2021[file], sep='\t', index_col= 'tweet_id')
    
    def preprocess(self):
        for split in ['training', 'test']:
            df = getattr(self, 'df_' + split)
            df['text'] = df['text'].str.replace('مستخدم@', '@USER', regex=True) # Replace users
            df['text'] = df['text'].str.replace(r'htt\S+|www.\S+', 'URL', regex=True) # Replace URLs
            setattr(self, 'df_' + split, df)  # Update the DataFrame attribute
    
    def add_labels_to_test(self):
        self.df_test_with_labels = pd.merge(self.df_test, self.df_labels, on='tweet_id')
        
    def join_train_test(self):
        self.df_training['data'], self.df_test_with_labels['data'] = 'training', 'test'
        self.df_merge = pd.concat([self.df_training, self.df_test_with_labels])
        
    def save_tsv(self, path, splits):
        all_splits = ['training', 'test_with_labels', 'merge']
        splits = splits if splits else all_splits
        for data in splits:
            df = getattr(self, 'df_' + data)
            df.to_csv(path  + '/ArMI2021_' + data +'.tsv', sep='\t')
        
    def summary(self):
        print('******************')
        print('**** ArMI2021 ****')
        print('******************')
        for split in ['training', 'test_with_labels', 'merge']:
            df = getattr(self, 'df_' + split)
            print(f'Number of samples in {split}: {df.shape[0]}')
            print(f'Label distribution in {split}:')
            distribution = df['misogyny'].value_counts(normalize=True).to_dict()
            for k,v in distribution.items():
                print(f'{k}: {v}')
            print()
    
    def main(self):
        self.preprocess()
        self.add_labels_to_test()
        self.join_train_test()
        
class OSACT2022:
    '''Class for ArMI_2021 dataset'''
    def __init__(self):
        self.df_training = self.read_tsv('training')
        self.df_dev = self.read_tsv('dev')
        self.df_test = self.read_tsv('test')
        self.df_labels = self.read_tsv('labels')
        self.main()

    def read_tsv(self, file):
        col_names = ['tweet_id', 'text', 'Offencive-Language', 'Fine-Grained-Hate-Speech', 'vulgar', 'violence']
        columns = ['Hate-Speech'] if file == 'labels' else  (col_names[:2] if file == 'test' else col_names[:])
        
        df = pd.read_csv(ORIGINAL_DATA_PATH + '/' + OSACT_2022['directory']+ '/' + OSACT_2022[file], 
                                sep='\t', 
                                header=None,
                                names=columns,
                                index_col= 'tweet_id' if file != 'labels' else None)
        if file == 'labels':
            df.index = df.index + 10158
            df.index.name = 'tweet_id'
            
        return df
    
    def add_labels_to_test(self):
        self.df_test_with_labels = pd.concat([self.df_test, self.df_labels], axis=1)
        
    def map_labels_FGHS_to_HS(self):
        self.df_training['Hate-Speech'] = self.df_training['Fine-Grained-Hate-Speech'].apply(lambda x: 'NOT_HS' if x == 'NOT_HS' else 'HS')
        self.df_dev['Hate-Speech'] = self.df_dev['Fine-Grained-Hate-Speech'].apply(lambda x: 'NOT_HS' if x == 'NOT_HS' else 'HS')
        
    def remove_non_needed_columns(self):
        self.df_training = self.df_training[['text', 'Hate-Speech']]
        self.df_dev = self.df_dev[['text', 'Hate-Speech']]
        self.df_test_with_labels = self.df_test_with_labels[['text', 'Hate-Speech']]
        
    def join_train_dev(self):
        self.df_training['data'], self.df_dev['data'], self.df_test_with_labels['data'] = 'training', 'dev', 'test'
        self.df_train_plus_dev = pd.concat([self.df_training, self.df_dev])
        
    def join_train_dev_test(self):
        self.df_training['data'] = 'test'
        self.df_merge = pd.concat([self.df_train_plus_dev, self.df_test])
        
    def save_tsv(self, path, splits):
        all_splits = ['training', 'dev', 'train_plus_dev','test_with_labels', 'merge']
        splits = splits if splits else all_splits
        for data in splits:
            df = getattr(self, 'df_' + data)
            df.to_csv(path  + '/OSACT2022_' + data +'.tsv', sep='\t')
        
    def summary(self):
        print('******************')
        print('**** OSACT2022 ***')
        print('******************')
        for split in ['training', 'dev', 'train_plus_dev','test_with_labels', 'merge']:
            df = getattr(self, 'df_' + split)
            print(f'Number of samples in {split}: {df.shape[0]}')
            print(f'Label distribution in {split}:')
            distribution = df['Hate-Speech'].value_counts(normalize=True).to_dict()
            for k,v in distribution.items():
                print(f'{k}: {v}')
            print()
    
    def main(self):
        self.add_labels_to_test()
        self.map_labels_FGHS_to_HS()
        self.remove_non_needed_columns()
        self.join_train_dev()
        self.join_train_dev_test()

        
class HSARABIC:
    '''Class for HSARABIC dataset'''
    def __init__(self):
        self.df = self.read_tsv('single_partition')
        self.main()

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
        self.df['text'] = self.df['text'].str.replace(r'@\w+', '@USER', regex=True) # Replace users
        self.df['text'] = self.df['text'].str.replace(r'htt\S+|www.\S+', 'URL', regex=True) # Replace URLs
    
    def split_data(self):
        self.df_training, self.df_test_with_labels = train_test_split(self.df, test_size=0.2, stratify=self.df['offensive_vulgar'], random_state=42)
        
    def join_train_test(self):
        self.df_training['data'], self.df_test_with_labels['data'] = 'training', 'test'
        self.df_merge = pd.concat([self.df_training, self.df_test_with_labels])
        
    def save_tsv(self, path, splits):
        all_splits = ['training', 'test_with_labels', 'merge']
        splits = splits if splits else all_splits
        for data in splits:
            df = getattr(self, 'df_' + data)
            df.to_csv(path  + '/HSARABIC_' + data +'.tsv', sep='\t')
        
    def summary(self):
        print('******************')
        print('**** HSARABIC ****')
        print('******************')
        for split in ['training', 'test_with_labels', 'merge']:
            df = getattr(self, 'df_' + split)
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
        
            
if __name__ == '__main__':
    print('\n')
    
    ArMI_2021 = ArMI2021()
    ArMI_2021.summary()
    
    OSACT2022 = OSACT2022()
    OSACT2022.summary()
    
    HSArabic = HSARABIC()
    HSArabic.summary()
