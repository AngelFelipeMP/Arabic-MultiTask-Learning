import os

TRANSFORMERS = ['aubmindlab/bert-base-arabert']

REPO_PATH = os.getcwd().split('Arabic-MultiTask-Learning')[0] + 'Arabic-MultiTask-Learning'
ORIGINAL_DATA_PATH = REPO_PATH + '/arabic-data-mtl'
GRAPHICS_PATH = REPO_PATH + '/graphics'
PRE_TRAINED_MODELS_PATH = REPO_PATH + '/pre-trained-models'

BASELINE_FOLDER_PATH = REPO_PATH + '/mtl-baseline'

BASELINE_LOGS_PATH = BASELINE_FOLDER_PATH + '/logs'
BASELINE_DATA_PATH = BASELINE_FOLDER_PATH + '/data'

BASELINE_MACHAMP_LOGS_PATH = BASELINE_FOLDER_PATH + '/MultiTask-Learning-for-Toxic-Language-Classification/machamp/logs'
BASELINE_MACHAMP_CONFIG_PATH = BASELINE_FOLDER_PATH + '/MultiTask-Learning-for-Toxic-Language-Classification/config'


MODEL_FOLDER_PATH = REPO_PATH + '/mtl-model'

MODEL_LOGS_PATH = MODEL_FOLDER_PATH + '/logs'
MODEL_DATA_PATH = MODEL_FOLDER_PATH + '/data'

ArMI_2021 = {'directory':'ArMI-2021',
                'labels':'ArMI2021_gold.tsv',
                'training':'ArMI2021_training.tsv',
                'dev': None,
                'test':'ArMI2021_test.tsv',
                'single_partition': None}

HSArabic = {'directory':'HSArabic',
                'labels': None,
                'training': None,
                'dev': None,
                'test': None,
                'single_partition':'HSArabicDataset.xlsx'}

OSACT_2022 = {'directory':'OSACT2022',
                'labels': 'OSACT2022-sharedTask-test-taskB-gold-labels.txt',
                'training': 'OSACT2022-sharedTask-train.txt',
                'dev': 'OSACT2022-sharedTask-dev.txt',
                'test': 'OSACT2022-sharedTask-test-tweets.txt',
                'single_partition': None}