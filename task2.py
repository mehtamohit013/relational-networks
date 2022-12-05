#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[ ]:


import pandas as pd
import numpy as np
import os

from collections import Counter, OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils import rnn

import torchtext as tt

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger


# # Parameters

# In[ ]:


data_dir = './data'
log_dir = './LSTM_RN_logs'

BATCH_SIZE = 64
n_worker = 8

input_dim = 13


# # Utilities

# In[ ]:


def top_k_acc(y_true:torch.Tensor,y_pred:torch.Tensor,relations,k=1):
    
    y_pred_tpk = torch.topk(y_pred,k,dim=1)[1]
    
    unary_ovr = 0
    unary_pos = 0
    
    binary_ovr = 0
    binary_pos = 0

    for i in range(0,len(y_pred_tpk)):
        if(y_true[i] in y_pred_tpk[i]):
            if relations[i]==1:  
                unary_pos+=1
            else:
                binary_pos+=1
        if relations[i]==1:
            unary_ovr+=1
        else:
            binary_ovr+=1
    
    binary_acc = float(binary_pos)/float(binary_ovr)
    unary_acc = float(unary_pos)/float(unary_ovr)
    return binary_acc*100, unary_acc*100


# # Loading Data

# In[ ]:


train_df = pd.read_pickle(os.path.join(data_dir,'train_df.pkl'))
test_df = pd.read_pickle(os.path.join(data_dir,'test_df.pkl'))

n_train = len(train_df)
n_test = len(test_df)


# In[ ]:


train_df.info()


# # Creating Vocab 

# In[ ]:


tokenizer = tt.data.utils.get_tokenizer('basic_english')

corpus = ''

for i in train_df.index.values:
    corpus+=' '+train_df.at[i,'Question']
    corpus+=' '+str(train_df.at[i,'Answer'])

corpus = corpus.lower()
corpus_token = tokenizer(corpus)

counter = Counter(corpus_token)
sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
ordered_dict = OrderedDict(sorted_by_freq_tuples)
vocab = tt.vocab.Vocab(ordered_dict,specials=[])


# # Dataset and Dataloader

# ## Dataset Class

# In[ ]:


class state_dataset(Dataset):
    def __init__(self,df:pd.DataFrame,
                vocab:tt.vocab.Vocab,
                tokenizer,
                input_dim = input_dim) -> None:
        super().__init__()
        self.df = df
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.vocab_size = len(vocab)
        self.input_dim = input_dim
        
        self.color_map = {
            'red': 1,
            'green': 2,
            'blue': 3,
            'orange':4,
            'grey': 5,
            'yellow':6
        }
        self.shape_map = {
            'rectangle': 1,
            'circle': 2
        }

    def __getitem__(self, index):
        sample = dict()

        # State Description
        state = {}
        state['center'] = np.stack(self.df.at[index,'State']['center'].to_numpy(),axis=0).astype('float32')
        
        color = self.df.at[index,'State']['color'].values
        state['color'] = np.array([self.color_map[i.lower()] for i in color]).astype('float32')

        shape = self.df.at[index,'State']['shape'].values
        state['shape'] = np.array([self.shape_map[i.lower()] for i in shape]).astype('float32')

#         state['size'] = self.df.at[index,'State']['size'].values.astype('float32')
        sample['state'] = np.concatenate((state['center'],state['color'][...,None],state['shape'][...,None]),axis=1)

        # Tokenizing Question and Answer
        ques = self.tokenizer(self.df.at[index,'Question'])
        sample['ques'] = np.array([self.one_hot(self.vocab[i.lower()]) for i in ques]).astype('float32')
        sample['lengths'] = sample['ques'].shape[0]
        
        sample['ques'] = np.concatenate([sample['ques'],
                                   np.zeros((self.input_dim-sample['lengths'],self.vocab_size))],
                                  axis=0).astype('float32')
        
        ans = [str(self.df.at[index,'Answer'])]
        sample['ans'] = np.array([self.vocab[i.lower()] for i in ans]).astype('float32')

        #  Additional Information
        if self.df.at[index,'Relation'].lower() == 'unary':
            sample['relations']=1
        else:
            sample['relations']=2

#         sample['ques type'] = self.df.at[index,'Ques type']

        return sample
    
    def __len__(self):
        return len(self.df)
    
    def one_hot(self,a:int):
        tmp = np.zeros(self.vocab_size)
        tmp[a] = 1
        return tmp.astype('float32')


# In[ ]:


train_dataset = state_dataset(
    df=train_df,
    vocab=vocab,
    tokenizer=tokenizer
)

test_dataset = state_dataset(
    df=test_df,
    vocab=vocab,
    tokenizer=tokenizer
)


# In[ ]:


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    pin_memory=True,
    num_workers = n_worker,
    drop_last = True,
    shuffle = True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    pin_memory=True,
    num_workers = n_worker,
    drop_last = True,
    shuffle = True
)


# # Model

# In[ ]:


class lstm_RN(nn.Module):
    def __init__(self,
                 vocab_size:int = len(vocab),
                 batch_size:int = BATCH_SIZE):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        
        self.hidden_dim = 128
        self.lstm = nn.LSTM(input_size=vocab_size,
                           hidden_size =128,
                           batch_first=True)
        
        self.g_fc1 = nn.Linear((4)*2+128, 256)
        
        self.g_fc2 = nn.Linear(256, 256)
        self.g_fc3 = nn.Linear(256, 256)
        self.g_fc4 = nn.Linear(256, 256)

        self.f_fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, self.vocab_size)
        
        
        
    def forward(self,ques,state,lengths):
        curr_device = ques.device
        
        h0 = torch.randn(1,self.batch_size,self.hidden_dim).to(curr_device)
        c0 = torch.randn(1,self.batch_size,self.hidden_dim).to(curr_device)
        
        cpu = torch.device('cpu')
        lengths = lengths.to(cpu)
        ques = rnn.pack_padded_sequence(ques.to('cpu'),lengths,batch_first=True,enforce_sorted=False)
        ques = ques.to(curr_device)
        

        ques,_ = self.lstm(ques,(h0,c0))
        ques,lengths = rnn.pad_packed_sequence(ques, batch_first=True)
        ques = ques[np.arange(ques.shape[0]), lengths - 1, :]
        
        '''
        Making Pairs:
        ques shape: [BATCH_SIZE,128]
        state shape: [BATCH_SIZE,6,4]
        6 objects with each object having 4 features
        '''
        
        obj_cnt = state.shape[1]
        
#         From model.py line 172
#         Add Questions Everywhere
        qst = torch.unsqueeze(ques,1)
        qst = qst.repeat(1,obj_cnt,1)
        qst = torch.unsqueeze(qst,2)
        
#         Cast all pairs against one another
        x_i = torch.unsqueeze(state,1)
        x_i = x_i.repeat(1,obj_cnt,1,1)
        
        x_j = torch.unsqueeze(state,2)
        x_j = torch.cat([x_j,qst],3)
        x_j = x_j.repeat(1,1,obj_cnt,1)
        
        # Concat all together
        x_full = torch.cat([x_i,x_j],3)
        
        # Reshape for passing through network
        x_ = x_full.view(self.batch_size * obj_cnt * obj_cnt, x_full.shape[-1])
        
        
        x_ = self.g_fc1(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc2(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc3(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc4(x_)
        x_ = F.relu(x_)
        
        x_g = x_.view(self.batch_size,obj_cnt*obj_cnt, 256)
        x_g = x_g.sum(1).squeeze()
        
        x_f = self.f_fc1(x_g)
        x_f = F.relu(x_f)
        
        x = self.fc2(x_f)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


# In[ ]:


model = lstm_RN()


# # Lightning Trainer

# In[ ]:


class lstm_rn_trainer(pl.LightningModule):
    def __init__(self,
                learning_rate = 1e-3,
                vocab_size = len(vocab),
                batch_size = BATCH_SIZE):
        super().__init__()
        self.model = lstm_RN()
        
        self.vocab_size = vocab_size
        self.nll_loss = nn.NLLLoss()
        self.lr = learning_rate
        self.batch_size = batch_size
       
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(),self.lr)
        return opt
    
    def forward(self,ques,state,lengths):
        out = self.model(ques,state,lengths)
        return out
    
    def training_step(self,batch,batch_idx):
        ques = batch['ques']
        state = batch['state']
        lengths = batch['lengths']
        label = batch['ans'][:,0].long()
        
        yhat = self(ques,state,lengths)
        loss = self.nll_loss(yhat,label)
        
        self.log('Train Loss',loss)
        
        return {'loss':loss,'pred':yhat.cpu().detach(),
                'label':label.cpu().detach(),
                'relations': batch['relations'].cpu().detach()}
    
    def training_epoch_end(self,train_out):
        len_out = len(train_out)
        y_pred = torch.Tensor(len_out*self.batch_size,self.vocab_size)
        y_true = torch.Tensor(len_out*self.batch_size)
        relations = []
        
        for i in range(0,len_out):
            y_pred[i*self.batch_size:(i+1)*self.batch_size,:] = train_out[i]['pred'] 
            y_true[i*self.batch_size:(i+1)*self.batch_size] = train_out[i]['label']
            relations.append(train_out[i]['relations'])
        
        relations = torch.concat(relations).numpy()

        # Calculating Avg loss
        avg_loss = torch.stack([x['loss'] for x in train_out]).mean()

        top1_bin,top1_una = top_k_acc(y_true,y_pred,relations,k=1)

        self.logger.experiment.add_scalar('Loss-Train per epoch',avg_loss,self.current_epoch)
        self.logger.experiment.add_scalar('Binary Train Accuracy',top1_bin,self.current_epoch)
        self.logger.experiment.add_scalar('Unary Train Accuracy',top1_una,self.current_epoch)

        print(f'Binary Train accuracy is {top1_bin}')
        print(f'Unary Train accuracy is {top1_una}')
        
    def validation_step(self,batch,batch_idx):
        ques = batch['ques']
        state = batch['state']
        lengths = batch['lengths']
        label = batch['ans'][:,0].long()
        
        yhat = self(ques,state,lengths)
        
        return [yhat.cpu().detach(),label.cpu().detach(),
                batch['relations'].cpu().detach()]
     
    def validation_epoch_end(self,val_out):
        len_out = len(val_out)
        y_pred = torch.Tensor(len_out*self.batch_size,self.vocab_size)
        y_true = torch.Tensor(len_out*self.batch_size)
        relations = []
        for i in range(0,len_out):
            y_pred[i*self.batch_size:(i+1)*self.batch_size,:] = val_out[i][0]
            y_true[i*self.batch_size:(i+1)*self.batch_size] = val_out[i][1]
            relations.append(val_out[i][2])
            
        relations = torch.concat(relations).numpy()
        top1_bin,top1_una = top_k_acc(y_true,y_pred,relations,k=1)

        self.logger.experiment.add_scalar('Binary Validation Accuracy',top1_bin,self.current_epoch)
        self.logger.experiment.add_scalar('Unary Validation Accuracy',top1_una,self.current_epoch)


        print(f'Binary Validation accuracy is {top1_bin}')
        print(f'Unary Validation accuracy is {top1_una}')
    
    def test_step(self,batch,batch_idx):
        ques = batch['ques']
        state = batch['state']
        lengths = batch['lengths']
        label = batch['ans'][:,0].long()
        
        yhat = self(ques,state,lengths)
        
        return [yhat.cpu().detach(),label.cpu().detach(),
                batch['relations'].cpu().detach()]
     
    def test_epoch_end(self,val_out):
        len_out = len(val_out)
        y_pred = torch.Tensor(len_out*self.batch_size,self.vocab_size)
        y_true = torch.Tensor(len_out*self.batch_size)
        relations = []
        for i in range(0,len_out):
            y_pred[i*self.batch_size:(i+1)*self.batch_size,:] = val_out[i][0]
            y_true[i*self.batch_size:(i+1)*self.batch_size] = val_out[i][1]
            relations.append(val_out[i][2])
            
        relations = torch.concat(relations).numpy()
        top1_bin,top1_una = top_k_acc(y_true,y_pred,relations,k=1)

        self.logger.experiment.add_scalar('Binary Test Accuracy',top1_bin,self.current_epoch)
        self.logger.experiment.add_scalar('Unary Test Accuracy',top1_una,self.current_epoch)
        
        self.logger.experiment.add_hparams(
            {
                'LR' : self.lr,
                "Batch Size": self.batch_size,
                "Vocab Size": self.vocab_size,
                'overall params' : sum(p.numel() for p in self.model.parameters())
            },
            {
                'hparam/test_bin_acc' : top1_bin,
                'hparam/test_una_acc' : top1_una
            }
        ) 

        print(f'Binary Test accuracy is {top1_bin}')
        print(f'Unary Test accuracy is {top1_una}')


# # Training

# In[ ]:


logger = TensorBoardLogger(log_dir, name="LSTM_RN",log_graph=True,default_hp_metric=False)
lstm_model = lstm_rn_trainer()


# In[ ]:


lstm_pl_trainer = pl.Trainer(
                    accelerator='gpu', devices=1,
                    max_epochs = 40,
                    logger = logger,
                    deterministic = False,
                    auto_lr_find = False
                     )


# In[ ]:


# lstm_pl_trainer.tune(lstm_model,train_loader,test_loader)
lstm_pl_trainer.fit(lstm_model,train_loader,test_loader)
lstm_pl_trainer.test(lstm_model,test_loader)

