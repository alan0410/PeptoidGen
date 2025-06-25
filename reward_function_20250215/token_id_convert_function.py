from transformers import BertTokenizer
import torch
import numpy as np

def seq_list_to_ids(seq_list, max_len = 100,  ): # data format: list with strings ['AAAA', 'BBBB', 'CCC', ....]
    
    #tokenizer = BertTokenizer.from_pretrained('c:/Users/yeramazing/Desktop/yeram/code/saved_model/peptoidtokenizer', local_files_only=True) #BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
    tokenizer = BertTokenizer.from_pretrained('c:\\Users\\G\\OneDrive\\바탕 화면\\KIST\\code/saved_model/peptoidtokenizer', local_files_only=True)

    pad_id = tokenizer.pad_token_id
    start_id = tokenizer.cls_token_id
    end_id = tokenizer.sep_token_id
    
    output_list = [ tokenizer.tokenize(string) for string in seq_list]
        
    seq_token_list_src = []
    seq_token_list_trg = []

    for index in output_list:
        if len(index) + 1 < max_len:
            seq_token_list_src.append(  tokenizer.convert_tokens_to_ids(index) + [pad_id] * (max_len - len(index) ))
            seq_token_list_trg.append(  tokenizer.convert_tokens_to_ids(index) + [pad_id] * (max_len - len(index) ))

        else:
            seq_token_list_src.append(  tokenizer.convert_tokens_to_ids(index)[:max_len-1] +[end_id] )
            seq_token_list_trg.append( tokenizer.convert_tokens_to_ids(index) [:max_len-1] +[end_id] )
            
    dataset = [torch.tensor(seq_token_list_src), torch.tensor(seq_token_list_trg)]
    
    return dataset