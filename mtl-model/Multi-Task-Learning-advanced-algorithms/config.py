import os

#Hiper-parameters
SPLITS = 2 #2
EPOCHS = 3 #15
MAX_LEN = [128] #[64]
DROPOUT = [0.3] #[0.3]
LR = [1e-4] #[5e-6, 1e-5, 5e-5, 1e-4]
BATCH_SIZE = [64] #[64] #[12] #[16]
TRANSFORMERS = ['aubmindlab/bert-base-arabert']
ENCODER_FEATURE_LAYERS = [3] #[1,2,3] #[0]
DECODER_FEATURE_LAYERS = [0]

N_ROWS= None #128
SEED = 17
CODE_PATH = os.getcwd()
REPO_PATH = '/'.join(CODE_PATH.split('/')[0:-1])
DATA_PATH = REPO_PATH + '/' + 'data'
LOGS_PATH = REPO_PATH + '/' + 'logs'

TARGET_LANGUAGE = 'ar'
PROCESS_DATA = True
DEVICE = 'cuda:0' #0 #'cuda:1'
DATA_PARALLEL = True #None 

DOMAIN_CROSS_VALIDATION = 'cross-validation'
DOMAIN_TRAIN_TEST = 'train-test'

TRAIN_WORKERS = 1
VAL_WORKERS = 1 


INFO_DATA = {'ArMI2021': {
                    'task': 'sexism detection',
                    'url':'',
                    'text_col':'text',
                    'label_col':'misogyny',
                    'positive_class':'misogyny',
                    'metric':'Accuracy',
                    'language':'ar',
                    'datasets': {
                        'train': 'ArMI2021_training.tsv',
                        'test': 'ArMI2021_test_with_labels.tsv'
                    }
                },
            'HSARABIC': {
                    'task': 'offencive language detection',
                    'url':'',
                    'text_col':'text',
                    'label_col':'offensive_vulgar',
                    'positive_class':'YES',
                    'metric':'F1-score',
                    'language':'ar',
                    'datasets': {
                        'train': 'HSARABIC_training.tsv',
                        'test': 'HSARABIC_test_with_labels.tsv'
                    }
                },
            'OSACT2022': {
                    'task': 'hate speech detection',
                    'url':'',
                    'text_col':'text',
                    'label_col':'Hate-Speech',
                    'positive_class':'HS',
                    'metric':'F1-macro',
                    'language':'ar',
                    'datasets': {
                        'train': 'OSACT2022_train_plus_dev.tsv',
                        'test': 'OSACT2022_test_with_labels.tsv'
                    }
                }
        }

MODELS = {
        'MTL0': {
            'decoder': {
                'model':'classifier',
                'heads':['ArMI2021-HSARABIC']},
            'encoder': {
                'model':'transformer', 
                'input':['text']}
            }
        }


# MODELS = {
#         'MTL0': {
#             'decoder': {
#                 'model':'classifier',
#                 'heads':['ArMI2021-HSARABIC', 'ArMI2021-HSARABIC-OSACT2022']},
#             'encoder': {
#                 'model':'transformer', 
#                 'input':['text']}
#         },
#         'MTL1': {
#             'decoder': {
#                 'model':'classifier',
#                 'heads':['ArMI2021-HSARABIC', 'ArMI2021-HSARABIC-OSACT2022']},
#             'encoder': {
#                 'model':'transformer',
#                 'input':['text', 'task-identification-text']}
#             },
#         'MTL2': {
#             'decoder': {
#                 'model':'classifier',
#                 'heads':['ArMI2021-HSARABIC', 'ArMI2021-HSARABIC-OSACT2022']},
#             'encoder': {
#                 'model':'task-identification-encoder', 
#                 'input':['text', 'task-identification-vector']}
#             }
#         }

# MODELS = {
#         'MTL0': {
#             'decoder': {
#                 'model':'classifier',
#                 'heads':['DETOXIS2021-HatEval2019', 'EXIST2021-DETOXIS2021-HatEval2019']},
#             'encoder': {
#                 'model':'transformer', 
#                 'input':['text']}
#             }
#         }


# MODELS = {
#         'MTL0': {
#             'decoder': {
#                 'model':'classifier',
#                 'heads':['EXIST-DETOXIS', 'EXIST-HatEval','DETOXIS-HatEval', 'EXIST-DETOXIS-HatEval']},
#             'encoder': {
#                 'model':'transformer', 
#                 'input':['text']}
#         },
#         'MTL1': {
#             'decoder': {
#                 'model':'classifier',
#                 'heads':['EXIST-DETOXIS', 'EXIST-HatEval','DETOXIS-HatEval', 'EXIST-DETOXIS-HatEval']},
#             'encoder': {
#                 'model':'transformer',
#                 'input':['text', 'task-identification-text']}
#             },
#         'MTL2': {
#             'decoder': {
#                 'model':'classifier',
#                 'heads':['EXIST-DETOXIS', 'EXIST-HatEval','DETOXIS-HatEval', 'EXIST-DETOXIS-HatEval']},
#             'encoder': {
#                 'model':'task-identification-encoder', 
#                 'input':['text', 'task-identification-vector']}
#             }
#         }
        


# INFO_DATA = {'DETOXIS2021': {
#                     'task': 'toxicity detection',
#                     'url':'https://drive.google.com/drive/folders/1KnFDh6oykkhW0h3AS1OhGqnjp-akH10J?usp=sharing',
#                     'text_col':'comment',
#                     'label_col':'toxicity',
#                     'positive_class':1,
#                     'metric':'F1-score',
#                     'language':'es',
#                     'datasets': {
#                         'train': 'DETOXIS2021_train.csv',
#                         'test': 'DETOXIS2021_test_with_labels.csv'
#                     }
#                 },
#             'EXIST2021': {
#                     'task': 'sexism detection',
#                     'url':'https://drive.google.com/drive/folders/1UlxE4jeze3tzfwrwrRywsP2nA4C96U13?usp=sharing',
#                     'text_col':'text',
#                     'label_col':'task1',
#                     'positive_class':'sexist',
#                     'metric':'Accuracy',
#                     'language':'en-es',
#                     'datasets': {
#                         'train': 'EXIST2021_training.tsv',
#                         'test': 'EXIST2021_test_with_labeled.tsv'
#                     }
#                 },
#             'HatEval2019': {
#                     'task': 'hate speech detection',
#                     'url':'https://drive.google.com/drive/folders/1hFmufsQLuku4W21DkLClZ4xOdkCuW85l?usp=sharing',
#                     'text_col':'text',
#                     'label_col':'HS',
#                     'positive_class':1,
#                     'metric':'F1-macro',
#                     'language':'es',
#                     'datasets': {
#                         'train': 'HatEval2019_es_train.csv',
#                         'test': 'HatEval2019_es_test.csv'
#                     }
#                 }
#         }