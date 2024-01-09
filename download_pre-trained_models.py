from transformers import AutoModel, AutoTokenizer
from config import *

for transformer in TRANSFORMERS:
    #Donwload model and tokenizer
    model = AutoModel.from_pretrained(transformer)
    tokenizer = AutoTokenizer.from_pretrained(transformer)
    
    # Save model and tokenizer locally
    model.save_pretrained(PRE_TRAINED_MODELS_PATH + '/' + transformer.split('/')[-1] + '_model')
    tokenizer.save_pretrained(PRE_TRAINED_MODELS_PATH + '/' + transformer.split('/')[-1] +'_tokenizer')
    
print('Done!')