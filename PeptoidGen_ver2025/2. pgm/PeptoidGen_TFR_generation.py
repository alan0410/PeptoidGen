#%%
import torch

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as dist
from torch.optim.lr_scheduler import ExponentialLR
import torch.distributions as dist
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
import pickle, json, joblib
from tqdm import tqdm 
import re
from multiprocessing import Pool

from preprocessing import *
from reward_function_lgbm import *

warnings.filterwarnings('ignore')
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score

#%%
#setting = open_json('setting.json')
DATA_PATH = 'c:\\Users\\김영성\\Desktop\\PeptoidGen-main\\PeptoidGen_ver2025\\1. data'
PGM_PATH= 'c:\\Users\\김영성\\Desktop\\PeptoidGen-main\\\PeptoidGen_ver2025\\2. pgm'
RESULT_PATH = 'c:\\Users\\김영성\\Desktop\\PeptoidGen-main\\PeptoidGen_ver2025\\3. result'

tokenizer = BertTokenizer.from_pretrained('peptoidtokenizer', local_files_only=True)
seed = 950410
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed) 
#Please fix all seeds so that the reproducibility is ensured

# %%
#데이터셋 열기
df1 = pd.read_excel(os.path.join(DATA_PATH,'peptoid_values_with_hemolytic_train.xlsx'))
new_sequence_list = open_txt(os.path.join(DATA_PATH, 'new_sequence_list.txt'))
train_dataset =seq_list_to_ids(new_sequence_list, max_len = 18)
# %%
#Encoder 용으로 만들려면 꼭 필요
sampled_item_pc_list= []
for seq in tqdm(new_sequence_list ):
    pc = tokens_to_smiles_to_pc_encoder(seq.split(" "))
    sampled_item_pc_list.append(pc)

sampled_item_pc_list =  pd.concat(sampled_item_pc_list, ignore_index=True)

sampled_item_pc_list_for_reward= []

for seq in tqdm(new_sequence_list ):
    pc = tokens_to_smiles_to_pc(seq.split(" "))
    sampled_item_pc_list_for_reward.append(pc)

sampled_item_pc_list_for_reward =  pd.concat(sampled_item_pc_list_for_reward)

sampled_item_pc_list.to_csv(os.path.join(DATA_PATH, 'sampled_item_pc_list_encoder.csv'), index=False)
sampled_item_pc_list_for_reward.to_csv(os.path.join(DATA_PATH, 'sampled_item_pc_list_for_reward.csv'), index=False)
# print(train_dataset[0].shape)
#%%##########################################################################################
sampled_item_pc_df =            pd.read_csv(os.path.join(DATA_PATH, 'sampled_item_pc_list_encoder.csv'))
sampled_item_pc_df_for_reward = pd.read_csv(os.path.join(DATA_PATH, 'sampled_item_pc_list_for_reward.csv'))

anti_result, hemodel_result = reward_predict(sampled_item_pc_df_for_reward )
reward_original_seq = 3 * (torch.exp(torch.tensor(anti_result)) -1) * torch.where(torch.tensor(hemodel_result) > 0.5, 1, 0)

#%%
seq_len = 18
train_dataset = seq_list_to_ids(new_sequence_list, max_len = seq_len)
batch_size= 350

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], index

train_dataset_src = MyDataset(train_dataset[0])
train_dataset_trg = MyDataset(train_dataset[1])
train_dataloader_src =  DataLoader( train_dataset_src, batch_size=batch_size )
train_dataloader_trg = DataLoader( train_dataset_trg, batch_size=batch_size)
train_dataloader_src_pc = DataLoader( torch.tensor(sampled_item_pc_df.values, dtype=torch.float32), batch_size=batch_size  )

# %%
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout = dropout, bidirectional = True)
        self.dropout = nn.Dropout(dropout)
        self.lrelu = nn.LeakyReLU()
        self.pc_linear = nn.Linear(22+hid_dim, hid_dim) 
        
    def forward(self, src, src_pc):

        embedded = self.dropout(self.embedding(src))

        outputs, hidden = self.rnn(self.lrelu(embedded))
        
        # print("hidden shape", hidden.shape) # [2, batch_size, hid_dim]
        # print("src_pc shape", src_pc.shape) # [batch_size, 22]

        src_pc = torch.tensor([src_pc.tolist(), src_pc.tolist()]).squeeze(2).transpose(1,2) * 5
        
        # print("src_pc shape", src_pc.shape) # [batch_size, 22]
        # print("hidden shape", hidden.shape) # [2, batch_size, hid_dim]
        hidden = torch.cat([hidden, src_pc], axis = 2)
        
        #print("hidden shape after concat", hidden.shape) # [2, batch_size, 22+hid_dim]
        hidden = self.pc_linear(hidden)
        
        return outputs, hidden

# Attention 들어갈 자리
class Attention(nn.Module):
    def __init__(self, hidden_dim, units, ):    # the argment 'units' inherits from decoder.
        super().__init__()
        self.W1 = nn.Linear(hidden_dim * 4, units)
        self.W2 = nn.Linear(hidden_dim * 2 ,units)
        self.V = nn.Linear(units,1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim = 1)
        
    def forward(self, dec_hidden, all_enc_hiddens):
        
        dec_hidden = dec_hidden.view( [dec_hidden.shape[1], -1])
   
        query_with_time_axis = torch.unsqueeze(dec_hidden, dim = 1)        
        
        all_enc_hiddens = all_enc_hiddens.view([all_enc_hiddens.shape[1], all_enc_hiddens.shape[0], -1])

        bahdanau_additive = self.W1(query_with_time_axis) + self.W2(all_enc_hiddens) 

        attention_score = self.V(self.tanh(bahdanau_additive))

        attention_weights = self.softmax(attention_score)
        
        context_vector = torch.sum(attention_weights * all_enc_hiddens, axis = 1)
        
        return context_vector, attention_weights
    
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, unit, attention_usage = False):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.dropout = nn.Dropout(dropout)
        self.lrelu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim = 1)
        self.attention_usage = attention_usage
        self.units = unit
        
        if self.attention_usage == True:
            self.Attention = Attention(hidden_dim = hid_dim, units = self.units) 
            self.pc_linear = nn.Linear(22+hid_dim, hid_dim) 
            self.rnn = nn.GRU(emb_dim * 3 , hid_dim , n_layers * 2 , dropout=dropout, bidirectional = False)
            self.fc_out = nn.Linear(hid_dim , output_dim) 
        else:
            self.pc_linear = nn.Linear(22+hid_dim, hid_dim) 
            self.rnn = nn.GRU(emb_dim, hid_dim , n_layers * 2, dropout=dropout, bidirectional = False)
            self.batchnorm1 = nn.BatchNorm1d(hid_dim)
            self.fc_out = nn.Linear(hid_dim , output_dim) 
            self.batchnorm2 = nn.BatchNorm1d(51)

    def forward(self, input,src_pc,  hidden, enc_output):

        input = input.unsqueeze(0)
        
        embedded = self.dropout(self.embedding(input))

        embedded  = self.lrelu(embedded)

        if self.attention_usage == True:
            context_vector, attention_weight = self.Attention(hidden, enc_output)

            embedded = torch.cat((torch.unsqueeze(context_vector, dim = 0), embedded ), dim= -1)
            #src_pc = torch.tensor([src_pc.tolist(), src_pc.tolist()]).transpose(1,2) * 5
            #hidden = torch.cat([hidden, src_pc], axis = 2)
            #hidden = self.pc_linear(hidden)            
            #[1,batch_size, emb_dim] 이 되어야
            output, hidden = self.rnn(embedded, hidden)
            prediction = self.fc_out(output.squeeze(0)) 
            #prediction = self.softmax(prediction)
            return prediction, hidden, attention_weight


        else:
            #src_pc = torch.tensor([src_pc.tolist(), src_pc.tolist()]).transpose(1,2) * 5
            #hidden = torch.cat([hidden, src_pc], axis = 2)
            #hidden = self.pc_linear(hidden)        
            output, hidden = self.rnn(embedded, hidden)
            output = self.batchnorm1( output.squeeze(0) )
            #print("out after batchnorm shape", output.shape)

            prediction = self.fc_out(output)   
            #print("prediction 1", prediction[0,:])
            #prediction = self.softmax(prediction)
            #print("prediction 2", prediction[0,:])
            prediction = self.batchnorm2(prediction.squeeze(0))
            
            return prediction, hidden, 0

# %%
class Seq2Seq(nn.Module):
   def __init__(self, encoder, decoder, device):
       super().__init__()
       
       self.encoder = encoder
       self.decoder = decoder
       self.device = device
       self.softmax = softmax_temp #nn.Softmax(dim = 1)
       
   def forward(self, src, src_pc, trg, teacher_forcing_ratio):
       # src = [src len, batch size]
       # trg = [trg len, batch size]
       
       trg_len = trg.shape[0]
       batch_size = trg.shape[1]
       trg_vocab_size = self.decoder.output_dim
       
       # decoder 결과를 저장할 텐서
       outputs = torch.zeros(trg_len, batch_size, trg_vocab_size)
       actions = torch.zeros(trg_len, batch_size)
       
       # Encoder의 마지막 은닉 상태가 Decoder의 초기 은닉상태로 쓰임
       enc_output, hidden = self.encoder(src, src_pc)
       
       # Decoder에 들어갈 첫 input은 <sos> 토큰
       input = trg[0, :]

       for t in range(0, trg_len):

           output, hidden, _ = self.decoder(input, src_pc, hidden, enc_output)
           
           output = self.softmax(output)
           
           if t <= 2 :  #(trg_len *  teacher_forcing_ratio ) :
               
               output[:,[0,1,2,3,4]] = 1e-8
               outputs[t] = output

               sample = dist.Categorical(probs = output) 
           
               action = sample.sample()
               actions[t] = action
                              
               input = action #trg[t] #action # trg[t] #output.argmax(1) #trg[t]
               
           else:
               output[:,[1,2,4]] = 1e-8
               outputs[t] = output
               
               sample = dist.Categorical(probs = output) 

               action = sample.sample()
               actions[t] = action
               
               input = action #trg[t] #action ##output.argmax(1) 

       return outputs, actions

#%%
input_dim = tokenizer.vocab_size #len(SRC.vocab)
output_dim = tokenizer.vocab_size #len(TRG.vocab)

# Encoder embedding dim
enc_emb_dim = 64
# Decoder embedding dim
dec_emb_dim = 64

hid_dim= 64
n_layers= 1

attention_unit = 10

enc_dropout = 0.4
dec_dropout= 0.4
device= 'cpu'

enc = Encoder(input_dim, enc_emb_dim, hid_dim, n_layers, enc_dropout)
dec = Decoder(output_dim, dec_emb_dim, hid_dim, n_layers, dec_dropout, attention_unit, attention_usage = False)
model = Seq2Seq(enc, dec, device)

#%%
# optimizer 그리고 사전학습된 모델 
model.load_state_dict(torch.load(os.path.join(DATA_PATH, 'saved_model_20250626/TFR_best_REINFORCED/exponential_250626/E_1255.pt')))
optimizer = optim.Adam(model.parameters(), lr = 0.001)
#criterion = nn.CrossEntropyLoss(reduction = 'none', ignore_index = 0)  # [PAD] index 가 0임

print(model)
print("num_parameters of the model: ",  sum(p.numel() for p in model.parameters()))

#%%
def generation(target_network, behavior_network,  optimizer, clip, epoch, n_epochs,  seq_len ):
    target_network.eval()
    epoch_loss=0
    epoch_reward = 0
    length = 0
    neg = -1e18
    alpha = 0.1
    k,j = 0, 0    
    good_sample_idx_len_epoch = 0
    
    #train_dataloader_src_pc = DataLoader( torch.tensor(sampled_item_pc_list.values, dtype=torch.float32), batch_size=batch_size, shuffle=False,num_workers=6, pin_memory=False)
    # train_dataloader_src =  DataLoader( train_dataset_src, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=False )
    # train_dataloader_trg = DataLoader( train_dataset_trg, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=False)

    #teacher_force_ratio_list = [1.0] * 6000
    #(batch0, indices0), (batch1, indices1) in zip(train_dataloader_src, train_dataloader_trg)
    
    for (batch1, indices1), (batch2, indices2), (batch3) in zip(train_dataloader_src, train_dataloader_trg, train_dataloader_src_pc):
        src = batch1.T
        src_pc = batch3.T
        trg = batch2.T
        
        optimizer.zero_grad()
        #with torch.no_grad():
        output, actions = behavior_network(src, src_pc, trg, teacher_forcing_ratio = 1.0 )

        output_dim = output.shape[-1]
        sample_for_buffer = torch.zeros(seq_len * actions.shape[1],) #action
        mask = torch.zeros(seq_len * actions.shape[1], dtype = torch.float32)  # pad token 에 0 곱해

        #print("output before reshpae", output.shape)
        #print("output before reshpae ex", output[:,0,0])
        
        output = output.transpose(0,1).reshape(-1, output_dim)
        
        #print("actions shape", actions.shape)
        #print("actions", actions[:,0])
        
        actions = actions.T
        
        #print("actions shape", actions.shape) # 343, 18
        #print(" actions[1]", actions[1] )

        actions = actions.reshape(output.shape[0], )         
        
        token = tokenizer.convert_ids_to_tokens(actions) # id 를 token 으로, 
        
        # print("token", token)
        
        i= 0
        sampled_item_token_list = []
        sampled_item_pc_list = [] # sampled item PC로 변환용
        reward_list = []
        
        #beta = []
        #print("len token", len(token))
        
        while i < len(token) :
            
            #print("i" , i)
            
            datapoint = actions[i:i+seq_len] #.tolist()
            datapoint_token = token[i:i+seq_len]
            
            #print("datapoint", datapoint)
            #print("datapoiunt_token", datapoint_token)
            
            # sep, pad 둘 다 있을 경우
            
            if (tokenizer.pad_token in datapoint_token) and (tokenizer.sep_token in datapoint_token): 
                stop_idx = min( torch.where(datapoint == tokenizer.pad_token_id )[0][0].item() ,  torch.where(datapoint == tokenizer.sep_token_id )[0][0].item() )
                
                datapoint =  datapoint[:stop_idx ].tolist() + [tokenizer.sep_token_id ] + [tokenizer.pad_token_id] * (seq_len - stop_idx -1)  
                
                sample_token = datapoint_token[:stop_idx + 1]
                
                
                sample_token[-1] = tokenizer.sep_token
                actions[i:i+seq_len] = torch.tensor(datapoint)
                sample_for_buffer[i:i+seq_len] = torch.tensor( [tokenizer.cls_token_id]  + datapoint[:-1] )
                mask[i:i+stop_idx] = torch.ones(stop_idx, dtype = torch.float32)      
                
                #print("sample token 1" , sample_token)
                
                
            # pad token 만 있을 경우
            elif tokenizer.pad_token in datapoint_token:
                pad_location = torch.where(datapoint == tokenizer.pad_token_id )[0][0].item()
                datapoint = datapoint[:pad_location].tolist() + [tokenizer.sep_token_id ] + [tokenizer.pad_token_id] * (seq_len -pad_location -1)  
                
                sample_token = datapoint_token[:np.where(np.array(datapoint_token) == tokenizer.pad_token)[0][0].item()+1]
                sample_token[-1] = tokenizer.sep_token
                
                
                #print("datapointttt", datapoint)
                #print("sample_token", sample_token)
                
                actions[i:i+seq_len] = torch.tensor(datapoint)
                
                #print("cross entropy 에 들어가는 actionssss", actions[i:i+seq_len])
                #print("reward function 계산에 들어가는 ", [tokenizer.cls_token_id]  + datapoint[:-1] )
                
                sample_for_buffer[i:i+seq_len] = torch.tensor( [tokenizer.cls_token_id]  + datapoint[:-1]  )
                
                mask[i:i + pad_location] = torch.ones(pad_location, dtype = torch.float32)
                #print("sample token 2" , sample_token)
                
            elif tokenizer.mask_token in datapoint_token:
                mask_location = torch.where(datapoint == tokenizer.mask_token_id )[0][0].item()
                datapoint = datapoint[:mask_location].tolist() + [tokenizer.sep_token_id ] + [tokenizer.pad_token_id] * (seq_len - mask_location -1)  
                sample_token = datapoint_token[:np.where(np.array(datapoint_token) == tokenizer.mask_token)[0][0].item()+1]
                sample_token[-1] = tokenizer.sep_token
                actions[i:i+seq_len] = torch.tensor(datapoint)
                sample_for_buffer[i:i+seq_len] = torch.tensor( [tokenizer.cls_token_id]  + datapoint[:-1]  )
                mask[i:i + mask_location] = torch.ones(mask_location, dtype = torch.float32)
            # END token (NH2) 만 있을 경우
            
            elif tokenizer.sep_token in datapoint_token:
                end_location = torch.where(datapoint == tokenizer.sep_token_id )[0][0].item()
                datapoint = datapoint[:end_location+1].tolist() + [tokenizer.pad_token_id] * (seq_len - end_location-1)  
                sample_token = datapoint_token[:np.where(np.array(datapoint_token) == tokenizer.sep_token)[0][0].item()+1]
                actions[i:i+seq_len] = torch.tensor(datapoint)
                sample_for_buffer[i:i+seq_len] = torch.tensor( [tokenizer.cls_token_id] + datapoint[:-1])
                mask[i:i + end_location] = torch.ones(end_location, dtype = torch.float32)
                #print("sample_token 3", sample_token)
            else:
                datapoint = datapoint      
                sample_token = datapoint_token
                                
                sample_token[-1] = tokenizer.sep_token
                
                #print("sample_token_else", sample_token)
                
                actions[i:i+seq_len] = torch.tensor (datapoint[:-1].tolist() + [tokenizer.sep_token_id])
                
                sample_for_buffer[i:i+seq_len] = torch.tensor( [tokenizer.cls_token_id]  + datapoint[:-1].tolist() )
                mask[i:i+ seq_len] = torch.ones(len(datapoint), dtype = torch.float32)
        
            pc = tokens_to_smiles_to_pc(sample_token)
            sampled_item_pc_list.append(pc)
            sampled_item_token_list.append(sample_token)

            # sample_tokens = [
            # token[j : j + seq_len].tolist()
            # for j in range(0, len(token), seq_len)]
            # # ▶ Pool 을 이용해 병렬 실행 (코어 수 −1 = 7개 프로세스)
            # #ith Pool(processes=7) as pool:
            # sampled_item_pc_list = pool.map(tokens_to_smiles_to_pc, sample_tokens)
            # sampled_item_token_list = sample_tokens
            
            # train_dataloader_src_pc = DataLoader(
            # torch.tensor(sampled_item_pc_list.values, dtype=torch.float32),
            # batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=False)

            i += seq_len
        
        #print("sample_for_buffer", sample_for_buffer[:54])

        sample_for_buffer = sample_for_buffer.reshape(-1, seq_len).long()

        ##############################################################################################################
        ##############################################################################################################
        ##############################################################################################################

        ########################################Loss 계산부분

        ##############################################################################################################
        ##############################################################################################################
        ##############################################################################################################
        ##############################################################################################################
        

        #if epoch % 10 == 0:
        #    print("sample_for_buffer", tokenizer.convert_ids_to_tokens (sample_for_buffer[0,]))
        #    print("sample_for_buffer", tokenizer.convert_ids_to_tokens (sample_for_buffer[100,]))
        #    print("sample_for_buffer", tokenizer.convert_ids_to_tokens (sample_for_buffer[200,]))
        #    print("sample_for_buffer", tokenizer.convert_ids_to_tokens(sample_for_buffer[300,]))
        #         
        antimicrobial_prob, hemodel_prob = reward_predict( pd.concat(sampled_item_pc_list) )
        
        #reward_generated_sequence = torch.tensor(antimicrobial_prob + hemodel_prob)       
        reward_generated_sequence = 3 * (torch.exp(torch.tensor(antimicrobial_prob)) -1) * torch.where(torch.tensor(hemodel_prob) > 0.5, 1, 0) 
        
        # if epoch % 5 == 0:
        #     print("reward", reward_generated_sequence)

        good_sample_idx = torch.where( (antimicrobial_prob > 0.5 ) & (hemodel_prob > 0.5 ) )[0]

        good_sample_idx_len_epoch += len(good_sample_idx)
        
        #print("reward generated_sequence", reward_generated_sequence)
        #bad_sample_reward[epoch,:] = reward_generated_sequence[[0, 100, 200]
        # output: [batch*seq_len, vocab_size]  (softmax 확률)
        # actions: [batch*seq_len]           (샘플된 토큰 id)
        pad_idx = 0
        mask = (actions != pad_idx).float()                   # [B*T]

        dist_cat = torch.distributions.Categorical(probs=output)
        log_prob = dist_cat.log_prob(actions)         # [B*T]
        entropy_per_step = dist_cat.entropy()                 # [B*T]

        # device & dtype 일치시키기
        reward_flat = torch.tensor(
            reward_generated_sequence,
            device=log_prob.device,
            dtype=log_prob.dtype
        ).repeat_interleave(seq_len)

        # Policy loss (PAD 제외 평균)
        policy_loss  = - (log_prob * reward_flat * mask).sum() / mask.sum()
        # Entropy regularization (PAD 제외 평균)
        entropy_loss =   (entropy_per_step * mask).sum()   / mask.sum()

        total_loss = policy_loss - alpha * entropy_loss
        #print("total_loss", total_loss)    
        
        #REINFORCE_loss = (criterion(output , actions.to(torch.int64) ).type(torch.float) * torch.tensor(reward_generated_sequence).repeat_interleave(seq_len)).sum() / mask.sum()
        #total_loss = REINFORCE_loss  - alpha * Entropy
        
        #print("loss ", criterion(output , actions.to(torch.int64) ).type(torch.float)[:54]  )
        #print("reward", torch.tensor(reward_generated_sequence).repeat_interleave(seq_len)[:54] )
        
        #total_loss.requires_grad = True
                
        epoch_loss += total_loss.item()
        
        epoch_reward += torch.sum( reward_generated_sequence ).item()
        
        length += len(reward_generated_sequence )

        #memory.put( epoch, batch1, sample_for_buffer, reward_generated_sequence, indices1 )
        
        k += 1
        
        #total_loss.backward()
    
        # Gradient exploding 막기 위해 clip
        torch.nn.utils.clip_grad_norm_(target_network.parameters(), clip)

        #optimizer.step()
        #scheduler.step()
        
        #print("loss", epoch_loss, "len ", k)
    
    # if epoch % 5 == 0:
    #     print("Length of Good sample idx" , len(good_sample_idx))
    return epoch_loss/ k ,  reward_generated_sequence , actions, np.std(reward_generated_sequence.numpy()), good_sample_idx   #epoch_accuracy/ j , epoch_top_3_accuracy/ j

#%%
# class Buffer():
#     def __init__(self):
#         self.M = 10
#         self.action_buffer = torch.zeros(self.M, seq_len, len(new_sequence_list)).float() #(M, 18, 1000)
#         self.reward_buffer = torch.broadcast_to(torch.tensor(reward_original_seq).T, (self.M, len(reward_original_seq))).float() #(M, 1000)
        
#     def put(self, epoch, sequence_original, generated_sequence, reward_generated_sequence, idx):
        
#         # original sequence 의 reward vs 생성된 sequence reward 를 비교한다. 
#         # 십입을 기존에 buffer에 있는 max 애들이랑 비교해서 이기면 넣어주네
        
#         reward_generated_sequence = reward_generated_sequence #* beta
                
#         reward_comparison = torch.tensor( [ np.array(self.reward_buffer[ epoch % self.M , idx]).tolist(), 
#                                            reward_generated_sequence.squeeze().tolist()] ).T
        
#         sequence = torch.tensor([sequence_original.tolist(),
#                                  generated_sequence.tolist()]).T
        
#         argmax = torch.argmax(reward_comparison, axis = 1)
#         maximal = torch.max(reward_comparison, axis =1).values.type(torch.float32)
        
#         self.action_buffer[ epoch % self.M , : , idx] = sequence[:, torch.arange(len(idx)), argmax].float()
#         self.reward_buffer[ epoch % self.M , idx] = maximal.float() #reward_generated_sequence
        
#         # 초기화는 할까 말까 고민중
#         # if epoch % self.M == 0 :
#         #     self.action_buffer = torch.zeros(self.M, seq_len, len(augmented_sequence_list))
#         #     self.reward_buffer = torch.broadcast_to(torch.tensor(reward_original_seq), (self.M, len(reward_original_seq)))
        
#     def sampling(self):
#         actions_array = torch.tensor(self.action_buffer)
#         reward_array = torch.tensor(self.reward_buffer)
        
#         #print( "good sample percentage in buffer:", round( len(np.where (reward_array > 0.5)[0]) / (self.M * len(reward_original_seq))* 100 , 2) , "%" )
        
#         sampled_sequence = actions_array[torch.randint(0, self.M, (len(reward_original_seq),)), :, torch.arange(len(reward_original_seq))]
#         #sampled_sequence = actions_array[torch.argmax(reward_array, axis = 0), :, torch.arange(len(reward_original_seq))]
        
#         return torch.tensor(sampled_sequence)

# def epoch_time(start_time, end_time):
#     sec = end_time - start_time
#     mins = sec // 60
#     remains = sec % 60
#     return int(mins), round(remains, 2)

#%%

N_EPOCHS = 1
behavior_network = model
target_network = model
# behavior_network = torch.compile(behavior_network)
# target_network   = torch.compile(target_network)

#best_valid_loss = float('inf')
#torch.set_num_threads(8)
#torch.set_num_interop_threads(8)
#pool = Pool(processes=7)    
#memory = Buffer()
#bad_sample_reward = torch.zeros((3000,3))
total_df = pd.DataFrame()

for epoch in tqdm(range(N_EPOCHS)):
    #Buffer 사용 
    # if epoch >= 10 and epoch % 10 == 0:    
    #     sample_data = memory.sampling()
    #     train_dataset_trg = MyDataset(torch.tensor(sample_data, dtype = torch.int32))
    #     train_dataloader_trg = DataLoader(train_dataset_trg, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=False)
        
    #start_time = time.time()
    with torch.no_grad():
        
        train_loss,  reward, actions, train_reward_std, good_sample_idx = generation(target_network, target_network,  optimizer, CLIP, epoch, N_EPOCHS, seq_len)
    

    str_array = np.array(tokenizer.convert_ids_to_tokens(action_list[0])).reshape(-1,18)
    sequence_list = ['H-' + '-'.join(arr[arr != '[PAD]']) for arr in str_array]

    sub_df = pd.DataFrame({'시퀀스': sequence_list, '리워드': reward.numpy()})
    sub_df['good_sample_여부'] = np.where(sub_df.index.isin(np.array(good_sample_idx)) ,'Y', 'N')
    total_df = pd.concat([total_df, sub_df], ignore_index=True)


# %%
# %%
total_df.to_csv(os.path.join(RESULT_PATH, 'TFR_exponential_generation_result.csv'), index=False)
# %%
