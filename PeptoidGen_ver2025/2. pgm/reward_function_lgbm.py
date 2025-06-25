import json
import warnings
warnings.filterwarnings('ignore')

from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from rdkit.ML.Descriptors import MoleculeDescriptors
import joblib
import os 

#setting = open_json('setting.json')
DATA_PATH = 'c:\\Users\\G\\OneDrive\\바탕 화면\\KIST\\code\\PeptoidGen_ver2025\\1. data'

with open( os.path.join(DATA_PATH, 'submonomers_SMILES_dictionary.json') ) as f:
    smiles_dict = json.loads(f.read())

###################################
# def tokens_to_smiles_to_pc(seq):

#     smiles = ''
    
#     for j in list(seq):
#         smiles += smiles_dict[j]
            
#     mols = Chem.MolFromSmiles(smiles)  
#     descriptor_names = [desc[0] for desc in Descriptors._descList]
#     # descriptor 계산기 생성
#     calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

#     # 분자에 대해 descriptor 계산
#     descriptor_values = calculator.CalcDescriptors(mols)

#     # 결과를 Pandas 데이터프레임으로 변환
#     df = pd.DataFrame([descriptor_values], columns=descriptor_names)
    
#     #new_df = pd.concat([new_df, pd.DataFrame([properties], columns =  corr_feature_list)], axis = 0) #new_df.append(properties)
                  
#     return df

def reward_predict( src_pc ):

    clf1= LGBMClassifier(
    max_depth=5, 
    min_gain_to_split=0, 
    learning_rate=0.05, 
    feature_fraction=0.9, 
    min_child_samples=5,
    verbose = -1)

    clf2= LGBMClassifier(
    max_depth=5, 
    min_gain_to_split=0, 
    learning_rate=0.05, 
    feature_fraction=0.9, 
    min_child_samples=5,
    verbose = -1)

    clf1 = joblib.load(os.path.join(DATA_PATH, 'lgb_anti.pkl') )
    clf2 = joblib.load( os.path.join(DATA_PATH,'lgb_hemolytic.pkl'))

    result1 = clf1.predict_proba(src_pc)
    result2 = clf2.predict_proba(src_pc)

    return torch.tensor(result1[:,1]), torch.tensor(result2[:,1])


