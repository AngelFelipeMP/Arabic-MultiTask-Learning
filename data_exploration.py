import pandas as pd
from config import *
from icecream import ic

if __name__ == "__main__":
    
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
        
    ArMI_2021 = ArMI2021()
    ArMI_2021.main()