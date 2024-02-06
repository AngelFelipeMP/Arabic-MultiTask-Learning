import torch
import pandas as pd
import numpy as np
import random
import config
from utils import rename_logs, tdqm_gridsearch, parameters
from cross_validation import TrainValidaion
from tqdm import tqdm

random.seed(config.SEED)
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)
torch.cuda.manual_seed_all(config.SEED)

#rename old log files adding date YMD-HMS
rename_logs()

# create progress bar
grid_search_bar = tqdm(total=tdqm_gridsearch(splits=1), desc='TRAIN-TEST', position=0)

# metric results dataset
df_results = None

# get model_name/framework_name such as 'STL', 'MTL0' and etc & parameters
for model_name, model_characteristics in config.MODELS.items():
    
    # start model -> get datasets/heads
    for group_heads in model_characteristics['decoder']['heads']:
        
        # Model script starts Here!
        data_dict = dict()
        for head in sorted(group_heads.split('-')):
            
            # load datasets & create StratifiedKFold splitter
            data_dict[head] = {}
            data_dict[head]['train'] = pd.read_csv(config.DATA_PATH + '/' + str(config.INFO_DATA[head]['datasets']['train'].split('.')[0]) + '_processed.csv', nrows=config.N_ROWS)
            data_dict[head]['val'] = pd.read_csv(config.DATA_PATH + '/' + str(config.INFO_DATA[head]['datasets']['test'].split('.')[0]) + '_processed.csv', nrows=config.N_ROWS)
            data_dict[head]['num_class'] = len(data_dict[head]['train'][config.INFO_DATA[head]['label_col']].unique().tolist())
            data_dict[head]['rows'] = data_dict[head]['val'].shape[0]
        
        # train-test 
        for num_efl in parameters(model_name)['task-identification-vector']:
            for num_dfl in parameters(model_name)['deep-classifier']:
                for transformer in config.TRANSFORMERS:
                    for max_len in config.MAX_LEN:
                        for batch_size in config.BATCH_SIZE:
                            for drop_out in config.DROPOUT:
                                for lr in config.LR:
                                            
                                    tqdm.write(f'\nModel: {model_name} Heads: {group_heads} Encode-feature-layers: {num_efl} Decoder-feature-layers: {num_dfl} Dropout: {drop_out} lr: {lr} Max_len: {max_len} Batch_size: {batch_size}')
                                    
                                    cv = TrainValidaion(model_name, 
                                                        group_heads,
                                                        data_dict, 
                                                        max_len, 
                                                        transformer, 
                                                        batch_size, 
                                                        drop_out,
                                                        lr,
                                                        df_results,
                                                        1,
                                                        num_efl,
                                                        num_dfl,
                                                        config.DOMAIN_TRAIN_TEST
                                    )
                                    
                                    df_results = cv.run()
                                    grid_search_bar.update(1)
