import os
import sys
sys.path.append(os.getcwd().split('Arabic-MultiTask-Learning')[0] + 'Arabic-MultiTask-Learning')
from data_exploration import ArMI2021, OSACT2022, HSARABIC
from config import *

if __name__ == '__main__':
    ArMI_2021 = ArMI2021()
    ArMI_2021.save_tsv(BASELINE_DATA_PATH)
    
    OSACT2022 = OSACT2022()
    OSACT2022.save_tsv(BASELINE_DATA_PATH)
    
    HSArabic = HSARABIC()
    HSArabic.save_tsv(BASELINE_DATA_PATH)
    
    print('Datasets saved!')