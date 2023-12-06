import pandas as pd
from config import *
from icecream import ic

class ArMI2021:
    '''Class for ArMI_2021 dataset'''
    def __init__(self):
        self.df_training = self.read_tsv('training')
        self.df_test = self.read_tsv('test')
        self.df_labels = self.read_tsv('labels')

    def read_tsv(self, file):
        return pd.read_csv(ORIGINAL_DATA_PATH + '/' + ArMI_2021['directory']+ '/' + ArMI_2021[file], sep='\t')
    
    def add_labels_to_test(self):
        self.df_test_with_labels = pd.merge(self.df_test, self.df_labels, on='tweet_id')
        
    def join_train_test(self):
        df_training, df_test = self.df_training, self.df_test_with_labels
        df_training['data'], df_test['data'] = 'training', 'test'
        self.df_merge = pd.concat([df_training, df_test])
        
    def summary(self):
        for split in ['training', 'test_with_labels', 'merge']:
            df = getattr(self, 'df_' + split)
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
        for split in ['training', 'dev', 'test_with_labels', 'merge']:
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
        self.join_train_test()
        self.summary()
        
            
if __name__ == '__main__':
        # ArMI_2021 = ArMI2021()
        # ArMI_2021.main()
        

                
                
        OSACT2022 = OSACT2022()
        OSACT2022.main()
        
        
        
        
#   - OSACT-2022: Hate speech detection
#   - HSArabicDataset: Offensive/vulgar language detection
#   - AraMi 2021 and/or ArMIS 2023 : Misogyny detection