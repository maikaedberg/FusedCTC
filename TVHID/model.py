import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchsummary import summary
from encoder import TransformerEncoderLayer, TransformerEncoder

# structure of MLP inspired by
# Consistency-Aware Graph Network for Human Interaction Understanding by Wang et al.
# https://github.com/deepgogogo/CAGNet?v=1

class BaselineAudio(nn.Module):

    def __init__(self, target_dim, p=0.1):

        super(BaselineAudio, self).__init__()

        NFB = 29*128
        NDIM = 256
        NMID = 64

        self.fc_action_node=nn.Linear(NFB,NDIM)
        self.fc_action_mid=nn.Linear(NDIM,NMID)
        self.nl_action_mid=nn.LayerNorm([NMID])
        self.fc_action_final=nn.Linear(NMID, target_dim)
        
        self.dropout=nn.Dropout(p)

        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x, mode='train'):

        action = self.fc_action_node(x)
        action = F.relu(action)
        action = self.fc_action_mid(action)
        action = self.nl_action_mid(action)
        action = F.relu(action)
        action = self.fc_action_final(action)
        
        if mode == 'train':
            action = self.dropout(action)

        return action

class BaselineInteractionVision2(nn.Module):

    def __init__(self, p=0.1):

        super(BaselineInteractionVision2, self).__init__()

        NFB = 512
        NDIM = 256
        NMID = 64

        self.fc_action_node=nn.Linear(2*NFB,NDIM)
        self.fc_action_mid=nn.Linear(NDIM,NMID)
        self.nl_action_mid=nn.LayerNorm([NMID])
        self.fc_action_final=nn.Linear(NMID, 2)
        
        self.dropout=nn.Dropout(p)

        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x, mode='train'):

        action = self.fc_action_node(x)
        action = F.relu(action)
        action = self.fc_action_mid(action)
        action = self.nl_action_mid(action)
        action = F.relu(action)
        action = self.fc_action_final(action)
        
        if mode == 'train':
            action = self.dropout(action)

        return action

class BaselineInteractionVision(nn.Module):

    def __init__(self, p=0.1):

        super(BaselineInteractionVision, self).__init__()

        NFB = 512
        NDIM = 256

        self.fc_interactions_mid=nn.Linear(2*NFB,NDIM)
        self.fc_interactions_final=nn.Linear(NDIM,2)
        
        self.dropout=nn.Dropout(p)     

        #for m in self.modules():
        #    if isinstance(m,nn.Linear):
        #        nn.init.kaiming_normal_(m.weight)

    def forward(self, x, mode='train'):

        interacs = self.fc_interactions_mid(x)
        interacs = self.fc_interactions_final(interacs)

        if mode == 'train':
            interacs = self.dropout(interacs)

        return interacs

def make_models(qmodes = [], kmodes=[], N=2):

    models = []

    for (qmode, kmode) in zip(qmodes, kmodes):

        base_interac = BaselineInteractionVision() if qmode=='vision' else BaselineAudio(2)
        encoder =  EncoderClassifier(qmode, kmode, 8, base_interac, dropout_p=0.1, N=N)

        models.append(encoder)

    return models


class EncoderClassifier(nn.Module):
    
    def __init__(self, qmode, kmode, heads, base, dropout_p=0.1, N=4):

        super(EncoderClassifier, self).__init__()

        self.qmode = qmode
        self.kmode = kmode
        self.nheads = heads
        self.N = N

        qdim = 512 if qmode == 'vision' else 128
        kdim = 512 if kmode == 'vision' else 128
        qlen = 2 if qmode == 'vision' else 29
        klen = 2 if kmode == 'vision' else 29

        self.n_heads = heads

        self.pos_encoder_q = PositionalEncoding(qdim, dropout_p, max_len=qlen)
        self.pos_encoder_k = PositionalEncoding(kdim, dropout_p, max_len=klen)
        
        self.encoder_layer = TransformerEncoderLayer(d_model=qdim, kdim=kdim, nhead=heads, batch_first=True)
        self.encoder =  TransformerEncoder(self.encoder_layer, num_layers=N)
        self.dropout = nn.Dropout(p=dropout_p)
        self.base = base

        self.init_weights()


    def init_weights(self):
        for p in self.parameters():
            if isinstance(p,nn.Linear):
                nn.init.kaiming_normal_(p.weight)
            elif p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, f1, f2, aud, pad_aud, mode='train'):
        query = torch.stack((f1,f2),dim=1) if self.qmode == 'vision' else aud
        key   = torch.stack((f1,f2),dim=1) if self.kmode == 'vision' else aud

        if self.N != 0:
            if self.qmode == 'audio':
                query = self.pos_encoder_q(query)
            if self.kmode == 'audio':
                key   = self.pos_encoder_k(key)
        
        batchsize = f1.shape[0]
        pad_query = torch.zeros(16, 2) if self.qmode == 'vision' else pad_aud
        pad_key   = torch.zeros(16, 2) if self.kmode == 'vision' else pad_aud

        final_mask = []
        for batch_no in range(batchsize):
            curr_mask = [] # should be of size query * audio
            mask_q = pad_query[batch_no] # size query vision 2 audio 5
            mask_k = pad_key[batch_no]   #      key   vision 2 audio 5
            for i in range(len(mask_q)):
                if mask_q[i] == 0:
                    curr_mask.append(mask_k)
                else:
                    curr_mask.append(torch.ones_like(mask_k))

            final_mask += [torch.stack(curr_mask) for _ in range(self.nheads)]

        final_mask = torch.stack(final_mask)

        query = self.encoder(query, key, mask = final_mask)
        
        query = torch.flatten(query, start_dim=1)
        key   = torch.flatten(key, start_dim=1)

        return self.base(query)
    
    def save(self, path: str) -> None:
        '''Saves the model to a file '''
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        '''Loads the model from a file '''
        self.load_state_dict(torch.load(path))

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)