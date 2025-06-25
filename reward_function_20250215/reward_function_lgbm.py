#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import random
import math
import time
import warnings
from collections import deque, namedtuple
import pickle, json

from tqdm import tqdm 
import re
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from token_id_convert_function import seq_list_to_ids
#import reward_function_peptoid

warnings.filterwarnings('ignore')
from transformers import BertTokenizer

from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score, precision_score, confusion_matrix

tokenizer = BertTokenizer.from_pretrained('c:\\Users\\G\\OneDrive\\바탕 화면\\KIST\\code/saved_model/peptoidtokenizer', local_files_only=True)

seed = 10
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed) 
# %%
df1 = pd.read_excel('c:\\Users\\G\\OneDrive\\바탕 화면\\KIST\\code\\20230522\\peptoid_values_with_hemolytic_train.xlsx')
df1 = df1.iloc[df1[['E_coli','S_aureus']].dropna().index].reset_index()

df1.loc[ (df1['E_coli'] < 16) & (df1['S_aureus'] < 16) , 'Anti_label'] = 1
df1.loc[ df1['Anti_label'] != 1  ,'Anti_label'] = 0

df2 = df1.copy()
df2 = df2.dropna(subset = ['Hemolytic_final'])

df2['Hemolytic_label'] = 0
df2.loc[df2['Hemolytic_final'] >= 8, 'Hemolytic_label'] = 1
df2
# %%
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

def tokens_to_smiles_to_pc(seq):
    
    # new_df = pd.DataFrame([] , columns =   ['EState_VSA11', 'PEOE_VSA2', 'PEOE_VSA12', 'VSA_EState2', 'EState_VSA1',
    #    'Kappa1', 'Chi0', 'HeavyAtomMolWt', 'ExactMolWt',  'Chi1','LabuteASA', 
        
    #     'FractionCSP3', 'VSA_EState7', 'FpDensityMorgan3', 'SMR_VSA7',
    #    'FpDensityMorgan2', 'MolLogP', 'SlogP_VSA6', 'VSA_EState6', 'PEOE_VSA6',
    #    'EState_VSA8', 'FpDensityMorgan1'] )
    
    with open('c:\\Users\\G\\OneDrive\\바탕 화면\\KIST\\code\\reward_function_20250215\\submonomers_SMILES_dictionary.json') as f:
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

# def scaling(df):
        
#     data_mic = df[['EState_VSA11', 'PEOE_VSA2', 'PEOE_VSA12', 'VSA_EState2', 'EState_VSA1',
#        'Kappa1', 'Chi0', 'HeavyAtomMolWt', 'ExactMolWt',  'Chi1','LabuteASA']]
    
#     data_hemo = df[['FractionCSP3', 'VSA_EState7', 'FpDensityMorgan3', 'SMR_VSA7',
#        'FpDensityMorgan2', 'MolLogP', 'SlogP_VSA6', 'VSA_EState6', 'PEOE_VSA6',
#        'EState_VSA8', 'FpDensityMorgan1']]
    
#     return data_mic, data_hemo 

# def encoder_data_preparation(pc_list):
#     new_df = pd.DataFrame(pc_list, columns =  ['EState_VSA11', 'PEOE_VSA2', 'PEOE_VSA12', 'VSA_EState2', 'EState_VSA1',
#        'Kappa1', 'Chi0', 'HeavyAtomMolWt', 'ExactMolWt',  'Chi1','LabuteASA', 
        
#         'FractionCSP3', 'VSA_EState7', 'FpDensityMorgan3', 'SMR_VSA7',
#        'FpDensityMorgan2', 'MolLogP', 'SlogP_VSA6', 'VSA_EState6', 'PEOE_VSA6',
#        'EState_VSA8', 'FpDensityMorgan1']) #tokens_to_smiles_to_pc(seq)
    
#     data_MIC, data_hemo = scaling(new_df) 

#     intermediate_input = pd.concat([data_MIC, data_hemo], axis = 1)
#     return intermediate_input
# %%
#from token_id_convert_function import seq_list_to_ids
df2= df2.sample(frac= 1)
new_sequence_list = sequence_flatten(df2['sequence'].tolist())
seq_len = 18
num_train_data = 160
 
#new_sequence_list = sequence_flatten(df1['sequence'].tolist())
train_dataset = seq_list_to_ids(new_sequence_list[:num_train_data], max_len = 18)
test_dataset = seq_list_to_ids(new_sequence_list[num_train_data:], max_len = 18)
#train_dataset = seq_list_to_ids(new_sequence_list[:num_train_data], max_len = 18)
#test_dataset = seq_list_to_ids(new_sequence_list[num_train_data:], max_len = 18)

train_dataset_trg = torch.tensor (df2['Hemolytic_label'][:num_train_data].tolist())
test_dataset_trg = torch.tensor (df2['Hemolytic_label'][num_train_data:].tolist())

sampled_item_pc_list= []

for seq in tqdm(new_sequence_list ):
    pc = tokens_to_smiles_to_pc(seq.split(" "))
    sampled_item_pc_list.append(pc)

#%%
source_pc_input = pd.concat(sampled_item_pc_list)
train_source_pc_input = source_pc_input[:num_train_data]
test_source_pc_input = source_pc_input[num_train_data:]

train_source_pc_input['class'] = train_dataset_trg
test_source_pc_input['class'] = test_dataset_trg

#%%
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

def cohen_d(x, y):
    """Calculate Cohen's d effect size."""
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std

# 샘플 데이터 생성
result_df = pd.DataFrame()

def draw_feature(feature, save_dir = 'c:\\Users\\G\\OneDrive\\바탕 화면\\KIST\\code\\reward_function_20250215\\3. result_hemolytic\\'):

    # Cohen's d 계산
    class_a_values = train_source_pc_input[train_source_pc_input['class'] == 1][feature]
    class_b_values = train_source_pc_input[train_source_pc_input['class'] == 0][feature]
    d_value = cohen_d(class_a_values, class_b_values)

    # 각 class의 샘플 수 계산
    class_counts = train_source_pc_input['class'].value_counts()

    title = f"{feature} | Cohen's d: {d_value:.2f}"

    # 그래프 그리기
    plt.figure(figsize=(3, 3))
    #sns.violinplot(x='class', y=feature, data=train_source_pc_input, inner=None, palette={1: 'red', 0: 'blue'}, alpha=0.6)

    sns.violinplot(x='class', y=feature, data=train_source_pc_input, inner=None, 
    palette={1: 'red', 0: 'blue'}, alpha=0.1)
    sns.boxplot(x='class', y= feature, data=train_source_pc_input, showcaps=True,
    boxprops={'facecolor': 'white', 'edgecolor': 'black', 'linewidth': 1.5},
    whiskerprops={'linewidth': 2}, width=0.2, zorder= 3)
    
    y_min = train_source_pc_input[feature].min()
    y_max = train_source_pc_input[feature].max()
    
    plt.ylim(y_min - 2, y_max + 2) 
    for i, class_name in enumerate([1, 0]):
        text_y_position = max(y_min - 1, plt.ylim()[0] + 1)
        plt.text(i, text_y_position, f"n={class_counts[class_name]}",
                ha='left', va='center', fontsize=13, fontweight='bold', color='blue')

    plt.title(title)
    plt.xlabel("Hemolytic", fontsize = 14)
    plt.ylabel("Value")
    plt.savefig(os.path.join( save_dir, f'{feature}.png') )
    #plt.show()
    
    sub_result_df = pd.DataFrame({'feature':[feature],
                                  'CohensD': [d_value],
                                  'link': [os.path.join( save_dir, f'{feature}.png')]})

    return sub_result_df

for feature in tqdm(train_source_pc_input.columns[:-1]):
    sub_result_df = draw_feature(feature, save_dir ='c:\\Users\\G\\OneDrive\\바탕 화면\\KIST\\code\\reward_function_20250215\\3. result_hemolytic\\')
    result_df = pd.concat([result_df, sub_result_df])

result_df.to_excel('c:\\Users\\G\\OneDrive\\바탕 화면\\KIST\\code\\reward_function_20250215\\result_df_hemo.xlsx', index= False)
# %%
from lightgbm import LGBMClassifier

cols = train_source_pc_input.columns[:-1]
tgt = 'class'

clf = LGBMClassifier(
    max_depth=5, 
    min_gain_to_split=0, 
    learning_rate=0.05, 
    feature_fraction=0.9, 
    min_child_samples=5,
    verbose = -1
)
clf.fit(train_source_pc_input[cols], train_source_pc_input[tgt])

#%%
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
pred =clf.predict(train_source_pc_input[cols])
print(f"Accuracy: {accuracy_score(train_source_pc_input[tgt], pred)}")
print(f"Precision: {precision_score(train_source_pc_input[tgt], pred)}")
print(f"Confusion matrix: {confusion_matrix(train_source_pc_input[tgt], pred)}")

test_pred =clf.predict(test_source_pc_input[cols])
print(f"Accuracy: {accuracy_score(test_source_pc_input[tgt], test_pred)}")
print(f"Precision: {precision_score(test_source_pc_input[tgt], test_pred)}")
print(f"Confusion matrix: {confusion_matrix(test_source_pc_input[tgt], test_pred)}")
# %%
importance_df = pd.DataFrame(clf.feature_importances_, 
                             index= cols, columns = ['importance'],
                             ).sort_values(by= 'importance', ascending = False).reset_index()
importance_df.to_csv(os.path.join(save_dir, "Hemo_importance.csv") , index= False)
# %%
import joblib
import os 

save_dir = 'c:/Users/G/OneDrive/바탕 화면/KIST/code/reward_function_20250215/model/'
if not os.path.exists(save_dir): 
    os.makedirs(save_dir, exist_ok = True)

# booster= clf.booster_
# booster.save_model(filename = os.path.join(save_dir , 'hemolytic_model.txt'))
joblib.dump(clf, os.path.join(save_dir, 'lgb_hemolytic.pkl') )

# %%
