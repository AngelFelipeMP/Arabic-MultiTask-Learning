import os
import sys
sys.path.append(os.getcwd().split('Arabic-MultiTask-Learning')[0] + 'Arabic-MultiTask-Learning')
from data_exploration import ArMI2021, OSACT2022, HSARABIC
from config import *

if __name__ == '__main__':
    ArMI_2021 = ArMI2021()
    ArMI_2021.save_tsv(
                path=BASELINE_DATA_PATH,
                splits=['training','test_with_labels'])
    
    OSACT2022 = OSACT2022()
    OSACT2022.save_tsv(
                path=BASELINE_DATA_PATH,
                splits=['train_plus_dev','test_with_labels'])
    
    HSArabic = HSARABIC()
    HSArabic.save_tsv(
                path=BASELINE_DATA_PATH,
                splits=['training','test_with_labels'])
    
    print('Datasets saved!')