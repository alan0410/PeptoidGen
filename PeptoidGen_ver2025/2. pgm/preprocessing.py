from transformers import BertTokenizer
import torch
import numpy as np
import pandas as pd
import json
import os
import pickle
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from torch.utils.data import DataLoader, Dataset

#setting = open_json('setting.json')
DATA_PATH = 'c:\\Users\\G\\OneDrive\\바탕 화면\\KIST\\code\\PeptoidGen_ver2025\\1. data'

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], index


def sequence_flatten(original_sequence_list):
    new_sequence_list = []

    for seq in original_sequence_list:
        par_start_idx, par_end_idx = seq.find("("), seq.find(")")
        if par_start_idx != -1 and par_end_idx != -1:
            new_seq = seq[:par_start_idx] + (seq[par_start_idx+1:par_end_idx] + '-') *int(seq[par_end_idx+1]) + seq[par_end_idx+3:]
            new_seq = re.sub('-',' ', new_seq)
            new_sequence_list.append(new_seq)
        else:
            seq = re.sub('-',' ', seq)
            new_sequence_list.append(seq)
    
    return new_sequence_list

def seq_list_to_ids(seq_list, max_len = 100,  ): # data format: list with strings ['AAAA', 'BBBB', 'CCC', ....]
    
    tokenizer = BertTokenizer.from_pretrained( os.path.join( DATA_PATH,'saved_model/peptoidtokenizer'), local_files_only=True) #BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
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

def open_json(file):
    with open(file) as f:
        file = json.load(f)
    return file

def open_txt(file):
    with open(file, 'rb') as fp:
        new_sequence_list = pickle.load(fp)
    return new_sequence_list

def tokens_to_smiles_to_pc(seq):
    
    # new_df = pd.DataFrame([] , columns =   ['EState_VSA11', 'PEOE_VSA2', 'PEOE_VSA12', 'VSA_EState2', 'EState_VSA1',
    #    'Kappa1', 'Chi0', 'HeavyAtomMolWt', 'ExactMolWt',  'Chi1','LabuteASA', 
        
    #     'FractionCSP3', 'VSA_EState7', 'FpDensityMorgan3', 'SMR_VSA7',
    #    'FpDensityMorgan2', 'MolLogP', 'SlogP_VSA6', 'VSA_EState6', 'PEOE_VSA6',
    #    'EState_VSA8', 'FpDensityMorgan1'] )
    
    with open(os.path.join( DATA_PATH,'submonomers_SMILES_dictionary.json')) as f:
        smiles_dict = json.loads(f.read())
    
    #for i in range(len(tokens_list)):
    #seq = tokens_list[i]
    smiles = ''
    
    for j in list(seq):
        smiles += smiles_dict[j]
            
    mols = Chem.MolFromSmiles(smiles)
    descriptor_names = [desc[0] for desc in Descriptors._descList]
    # descriptor 계산기 생성
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

    # 분자에 대해 descriptor 계산
    descriptor_values = calculator.CalcDescriptors(mols)

    # 결과를 Pandas 데이터프레임으로 변환
    df = pd.DataFrame([descriptor_values], columns=descriptor_names)
    # properties1 = [ Descriptors.EState_VSA11(mols) , Descriptors.PEOE_VSA2(mols), Descriptors.PEOE_VSA12(mols), 
    #                 Descriptors.VSA_EState2(mols), Descriptors.EState_VSA1(mols), Descriptors.Kappa1(mols), Descriptors.Chi0(mols), 
    #                 Descriptors.HeavyAtomMolWt(mols), Descriptors.ExactMolWt(mols), Descriptors.Chi1(mols), Descriptors.LabuteASA(mols)] 

    # properties2 = [Descriptors.FractionCSP3(mols),Descriptors.VSA_EState7(mols),Descriptors.FpDensityMorgan3(mols),Descriptors.SMR_VSA7(mols),
    #                 Descriptors.FpDensityMorgan2(mols),Descriptors.MolLogP(mols),Descriptors.SlogP_VSA6(mols),Descriptors.VSA_EState6(mols),
    #                 Descriptors.PEOE_VSA6(mols), Descriptors.EState_VSA8(mols), Descriptors.FpDensityMorgan1(mols)]

    # properties = properties1 +  properties2 #[1] * (len(corr_feature_list)-10) 
    
    #new_df = pd.concat([new_df, pd.DataFrame([properties], columns =  corr_feature_list)], axis = 0) #new_df.append(properties)
                  
    return df


def tokens_to_smiles_to_pc_encoder(seq):
    
    new_df = pd.DataFrame([] , columns =   ['EState_VSA11', 'PEOE_VSA2', 'PEOE_VSA12', 'VSA_EState2', 'EState_VSA1',
       'Kappa1', 'Chi0', 'HeavyAtomMolWt', 'ExactMolWt',  'Chi1','LabuteASA', 
        
        'FractionCSP3', 'VSA_EState7', 'FpDensityMorgan3', 'SMR_VSA7',
       'FpDensityMorgan2', 'MolLogP', 'SlogP_VSA6', 'VSA_EState6', 'PEOE_VSA6',
       'EState_VSA8', 'FpDensityMorgan1'] )
    
    with open(os.path.join( DATA_PATH,'submonomers_SMILES_dictionary.json')) as f:
        smiles_dict = json.loads(f.read())
    
    #for i in range(len(tokens_list)):
    #seq = tokens_list[i]
    smiles = ''
    
    for j in list(seq):
        smiles += smiles_dict[j]
            
    mols = Chem.MolFromSmiles(smiles)
    #descriptor_names = [desc[0] for desc in Descriptors._descList]
    # descriptor 계산기 생성
    #calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

    # 분자에 대해 descriptor 계산
    # descriptor_values = calculator.CalcDescriptors(mols)

    # 결과를 Pandas 데이터프레임으로 변환
    #df = pd.DataFrame([descriptor_values], columns=descriptor_names)
    properties1 = [ Descriptors.EState_VSA11(mols) , Descriptors.PEOE_VSA2(mols), Descriptors.PEOE_VSA12(mols), 
                    Descriptors.VSA_EState2(mols), Descriptors.EState_VSA1(mols), Descriptors.Kappa1(mols), Descriptors.Chi0(mols), 
                    Descriptors.HeavyAtomMolWt(mols), Descriptors.ExactMolWt(mols), Descriptors.Chi1(mols), Descriptors.LabuteASA(mols)] 

    properties2 = [Descriptors.FractionCSP3(mols),Descriptors.VSA_EState7(mols),Descriptors.FpDensityMorgan3(mols),Descriptors.SMR_VSA7(mols),
                    Descriptors.FpDensityMorgan2(mols),Descriptors.MolLogP(mols),Descriptors.SlogP_VSA6(mols),Descriptors.VSA_EState6(mols),
                    Descriptors.PEOE_VSA6(mols), Descriptors.EState_VSA8(mols), Descriptors.FpDensityMorgan1(mols)]

    properties = properties1 +  properties2 #[1] * (len(corr_feature_list)-10) 
    
    new_df = pd.concat([new_df, pd.DataFrame([properties], columns =  ['EState_VSA11', 'PEOE_VSA2', 'PEOE_VSA12', 'VSA_EState2', 'EState_VSA1',
       'Kappa1', 'Chi0', 'HeavyAtomMolWt', 'ExactMolWt',  'Chi1','LabuteASA', 
        
        'FractionCSP3', 'VSA_EState7', 'FpDensityMorgan3', 'SMR_VSA7',
       'FpDensityMorgan2', 'MolLogP', 'SlogP_VSA6', 'VSA_EState6', 'PEOE_VSA6',
       'EState_VSA8', 'FpDensityMorgan1'])], axis = 0) #new_df.append(properties)
                  
    return new_df


def softmax_temp(x, temperature=1.5):
#     # Subtracting the maximum value along the appropriate axis for numerical stability
    max_x = torch.max(x, dim=1, keepdim=True)[0]
    exp_values = torch.exp((x - max_x) / temperature)
    sum_exp_values = torch.sum(exp_values, dim=1, keepdim=True)     
    probabilities = exp_values / sum_exp_values
    return probabilities