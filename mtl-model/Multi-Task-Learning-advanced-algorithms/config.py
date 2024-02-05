import os

#Hiper-parameters
SPLITS = 1 #2
EPOCHS = 1 #15
MAX_LEN = [64] #[128]
DROPOUT = [0.3] #[0.3]
LR = [1e-4] #[5e-6, 1e-5, 5e-5, 1e-4]
BATCH_SIZE = [16] #[64] #[12]
TRANSFORMERS = ['dccuchile/bert-base-spanish-wwm-cased']
ENCODER_FEATURE_LAYERS = [3] #[1,2,3] #[0]
DECODER_FEATURE_LAYERS = [0]

N_ROWS=64 #None 
SEED = 17
CODE_PATH = os.getcwd()
REPO_PATH = '/'.join(CODE_PATH.split('/')[0:-1])
DATA_PATH = REPO_PATH + '/' + 'data'
LOGS_PATH = REPO_PATH + '/' + 'logs'

TARGET_LANGUAGE = 'es'
DOWLOAD_DATA = False
PROCESS_DATA = True
DEVICE = 'cuda:0' #0 #'cuda:1'
DATA_PARALLEL = True #None 

# DOMAIN_GRID_SEARCH = 'gridsearch'
DOMAIN_CROSS_VALIDATION = 'cross-validation'
DOMAIN_TRAIN_TEST = 'train-test'
# DOMAIN_TRAIN = 'training'
# DOMAIN_VALIDATION = 'validation'
# DOMAIN_TRAIN_ALL_DATA = 'all_data_training'
# DOMAIN_TEST = 'test'

TRAIN_WORKERS = 1
VAL_WORKERS = 1 


INFO_DATA = {'DETOXIS': {
                    'task': 'toxicity detection',
                    'url':'https://drive.google.com/drive/folders/1KnFDh6oykkhW0h3AS1OhGqnjp-akH10J?usp=sharing',
                    'text_col':'comment',
                    'label_col':'toxicity',
                    'positive_class':1,
                    'metric':'F1-score',
                    'language':'es',
                    'datasets': {
                        'train': 'DETOXIS2021_train.csv',
                        'test': 'DETOXIS2021_test_with_labels.csv'
                    }
                },
            'EXIST': {
                    'task': 'sexism detection',
                    'url':'https://drive.google.com/drive/folders/1UlxE4jeze3tzfwrwrRywsP2nA4C96U13?usp=sharing',
                    'text_col':'text',
                    'label_col':'task1',
                    'positive_class':'sexist',
                    'metric':'Accuracy',
                    'language':'en-es',
                    'datasets': {
                        'train': 'EXIST2021_training.tsv',
                        'test': 'EXIST2021_test_with_labeled.tsv'
                    }
                },
            'HatEval': {
                    'task': 'hate speech detection',
                    'url':'https://drive.google.com/drive/folders/1hFmufsQLuku4W21DkLClZ4xOdkCuW85l?usp=sharing',
                    'text_col':'text',
                    'label_col':'HS',
                    'positive_class':1,
                    'metric':'F1-score',
                    'language':'es',
                    'datasets': {
                        'train': 'HatEval2019_es_train.csv',
                        'test': 'HatEval2019_es_test.csv'
                    }
                }
        }

MODELS = {
        'MTL0': {
            'decoder': {
                'model':'classifier',
                'heads':['EXIST-DETOXIS', 'EXIST-HatEval','DETOXIS-HatEval', 'EXIST-DETOXIS-HatEval']},
            'encoder': {
                'model':'transformer', 
                'input':['text']}
            }
        }

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
            
            
            


# MODELS = {
#         'STL': {
#             'decoder': {
#                 'model':'classifier',
#                 'heads':['EXIST', 'DETOXIS', 'HatEval']},
#             'encoder': {
#                 'model':'transformer', 
#                 'input':['text']}
#             },
#         'MTL0': {
#             'decoder': {
#                 'model':'classifier',
#                 'heads':['EXIST-DETOXIS', 'EXIST-HatEval','DETOXIS-HatEval', 'EXIST-DETOXIS-HatEval']},
#             'encoder': {
#                 'model':'transformer', 
#                 'input':['text']}
#             },
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
#             },
#         'MTL3': {
#             'decoder': {
#                 'model':'deep-classifier',
#                 'heads':['EXIST-DETOXIS', 'EXIST-HatEval','DETOXIS-HatEval', 'EXIST-DETOXIS-HatEval']},
#             'encoder': {
#                 'model':'transformer', 
#                 'input':['text']}
#             },
#         'MTL4': {
#             'decoder': {
#                 'model':'deep-classifier',
#                 'heads':['EXIST-DETOXIS', 'EXIST-HatEval','DETOXIS-HatEval', 'EXIST-DETOXIS-HatEval']},
#             'encoder': {
#                 'model':'transformer',
#                 'input':['text', 'task-identification-text']}
#             },
#         'MTL5': {
#             'decoder': {
#                 'model':'deep-classifier',
#                 'heads':['EXIST-DETOXIS', 'EXIST-HatEval','DETOXIS-HatEval', 'EXIST-DETOXIS-HatEval']},
#             'encoder': {
#                 'model':'task-identification-encoder', 
#                 'input':['text', 'task-identification-vector']}
#             }
#         }



# INFO_DATA = {'DETOXIS': {
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
#             'EXIST': {
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
#             'HatEval': {
#                     'task': 'hate speech detection',
#                     'url':'https://drive.google.com/drive/folders/1XAcXmF-jerbQNy_nwzjkBuhjnU7TektN?usp=sharing',
#                     'text_col':'text',
#                     'label_col':'HS',
#                     'positive_class':1,
#                     'metric':'F1-score',
#                     'language':'es',
#                     'datasets': {
#                         'train': 'HatEval2019_es_train.csv',
#                         'dev':'HatEval2019_es_dev.csv',
#                         'test': 'HatEval2019_es_test.csv'
#                     }
#                 }
#         }
#
