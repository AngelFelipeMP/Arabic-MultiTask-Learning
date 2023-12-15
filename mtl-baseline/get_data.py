import os
import sys
import shutil
sys.path.append(os.getcwd().split('Arabic-MultiTask-Learning')[0] + 'Arabic-MultiTask-Learning')
from data_exploration import ArMI2021, OSACT2022, HSARABIC
from config import *

def clear_directory(dir_path):
    # Check if the directory exists
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        # Iterate over all files in the directory
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove file or link
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove subdirectories and files within
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        print(f"The directory {dir_path} does not exist.")

if __name__ == '__main__':
    #remove old data
    clear_directory(BASELINE_DATA_PATH)
    
    #process data
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