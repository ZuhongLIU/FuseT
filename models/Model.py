import torch
import torch.nn as nn
import numpy as np
#from models.Layers import EncoderLayer, DecoderLayer
import torch.nn.functional as F
import torch_dct as dct #https://github.com/zh217/torch-dct

from models.SceneModel import SceneEncoder
#from utils import EnvBC_Loss,calculate_part_sep

#from models.GC import GCN,GCN_attention
#from models.GC_scatterversion import GCN


#from models.ST import STBlock,STFN

from dataset.mice_dataset import get_split_indices
from consts import NUM_MICE
from typing import Dict
import pytorch_lightning as pl


class Pooler(nn.Module):
    def __init__(self, input_size):
        super(Pooler, self).__init__()

        self.attn = nn.Linear(input_size, 1)
    
    def forward(self, embeddings):
        weights = F.softmax(self.attn(embeddings), dim=1)
        return torch.sum(weights * embeddings, dim=1)

class LinearClassifier(nn.Module):
    '''
        Simple linear classifier on top of embeddings (single layer perceptron)
        Optional dropout
    '''
    def __init__(self, hidden_dim, output_dim, dropout=False):
        super(LinearClassifier, self).__init__()
        self.classifier = nn.Linear(hidden_dim, output_dim)
        self.loss = nn.BCEWithLogitsLoss()
        self.dropout = nn.Dropout(0.2)
        self.do_dropout = dropout

    def forward(self, embedding, labels):
        if self.do_dropout:
            embedding = self.dropout(embedding)
        
        preds = self.classifier(embedding)
        loss = self.loss(preds, labels)
        #print("2222222222222222222222")
        #print(labels)
        return loss, preds


class RegressionHead(nn.Module):
    '''
        General regression head for continuous features
        Can be used with dropout, or with a mask
    '''

    def __init__(self, hidden_dim, output_dim, dropout=0.0, return_pred=False, two_layer=False) -> None:
        super(RegressionHead, self).__init__()
        if two_layer:
            self.regression_layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, output_dim)
            )
        else:
            self.regression_layer = nn.Linear(hidden_dim, output_dim)

        self.loss = nn.MSELoss(reduction='none')
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.return_pred = return_pred

    def forward(self, x, labels, mask=None):
        if self.dropout:
            x = self.dropout(x)

        preds = self.regression_layer(x)
        #print("preds:",preds.shape)
        #print("label:",labels.shape)
        if mask is not None:
            loss = ((mask * self.loss(preds, labels)).sum(dim=1)/mask.sum(dim=1)).mean()
        else:
            loss = self.loss(preds, labels).mean()
        #print("##################")
        #print(labels)
        if self.return_pred:
            return loss, preds
        else:
            return loss




def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s, *_ = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class SceneTransformer(pl.LightningModule):
    def __init__(self, device, output_size,in_feat_dim, time_steps, feature_dim, head_num, k):
        super(SceneTransformer, self).__init__()
        # self.device = device
        self.output_size = output_size
        self.in_feat_dim = in_feat_dim
        self.time_steps = time_steps
        self.feature_dim = feature_dim
        self.head_num = head_num
        self.k = k

        self.fc= nn.Linear(feature_dim,output_size)

        self.encoder=SceneEncoder(device, in_feat_dim, time_steps, feature_dim, head_num, k)
        #self.encoder=SceneEncoder(device, in_feat_dim*(self.J**2+1), time_steps, feature_dim, head_num, k)
        self.mm_head = RegressionHead(self.output_size, in_feat_dim)

        self.chases_head = LinearClassifier(self.output_size, 1, dropout=0.2)
        self.lights_head = LinearClassifier(self.output_size, 1, dropout=0.2)

        # Todo don't hard code...
        self.task_head = RegressionHead(self.output_size, 60, dropout=0.1)
        self.inter_mouse_task_head = RegressionHead(self.output_size, 13, dropout=0.1)
        self.group_task_head = RegressionHead(NUM_MICE * self.output_size, 1, dropout=0.2)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def extract_embeddings(self, embeddings: torch.tensor) -> torch.tensor:
        '''
        Extracts the embeddings for each frame from raw model outputs
        Removes special tokens and extra embeddings added to input 
        '''
        
        mouse_embeds = []
        for m in range(NUM_MICE):
            mouse_embeds.append(embeddings[:,:,m,:])

        return mouse_embeds

    def create_inter_mouse_embeddings(self, embeddings) -> torch.tensor:
        '''
            This function creates embeddings for inter-mouse interactions
            We create an embedding for every possible pairing
            (m0, m1), (m0, m2), (m1, m0), (m1, m2)... etc
            We use these embeddings to predict inter (between) mouse tasks.
        '''
        pairs = {}

        # Now create the pairings:
        for m1 in range(NUM_MICE):
            pairs[m1] = {}

            for m2 in range(NUM_MICE):
                if m1 == m2:
                    continue
                pairs[m1][m2] = embeddings[m2] - embeddings[m1]
        
        return pairs

    def calculate_inter_mouse_loss(self, embeddings, annotations):
        '''
            This function calculates the loss for tasks with inter mouse annotations
        '''
        pair_embeddings = self.create_inter_mouse_embeddings(embeddings)
        loss = 0
        num = 0
        for m1 in range(NUM_MICE):
            for m2 in range(NUM_MICE):
                if m1 == m2:
                    continue
                #print(pair_embeddings[m1][m2].shape)
                #print(annotations[m1][m2].shape)
                loss += self.inter_mouse_task_head(pair_embeddings[m1][m2], annotations[m1][m2])
                num += 1

        return loss / num

    def forward(self, keypoints: torch.tensor = None, 
                        label_mask: torch.tensor = None,
                        labels: torch.tensor = None,
                        chases: torch.tensor = None,
                        lights: torch.tensor = None,
                        task_annotations: torch.tensor = None,
                        inter_mouse_task_annotations: Dict = None,
                        group_task_annotations: torch.tensor = None,
                        train=True,
                        ):
        
        #print(keypoints)
        #torch.manual_seed(1234)
        input_seq=keypoints.clone()
        b,seq_len,n_person,feat_dim=keypoints.shape
        

        batch_mask=np.kron(np.diag([1 for i in range(input_seq.shape[0])]),np.ones((3,3),dtype='int'))
        batch_mask=torch.BoolTensor(batch_mask).to(self.device)

        input_seq=input_seq.permute(0,2,1,3)
        input_seq=input_seq.reshape(b*n_person,seq_len,-1)
    
        #print(input_seq.shape)

        if train:
            m=label_mask.permute(0,2,1,3)
            m=m.reshape(b*n_person,seq_len,-1).repeat(1,1,self.in_feat_dim)
        else:
            m=None

        enc_out,*_=self.encoder(input_seq,batch_mask,None,m)
        #print(enc_out.shape)

        enc_out=enc_out.reshape(-1,n_person,seq_len,self.feature_dim)
        
        enc_out=enc_out.permute(0,2,1,3)

        embeddings=self.fc(enc_out)

        #print(embeddings.shape)
        
        extracted_embeddings = self.extract_embeddings(embeddings) 
        combined_embeddings = torch.cat(extracted_embeddings, dim=2) #16*78*126

        #print(combined_embeddings.shape)    

        # Losses
        losses = {}
        loss = 0

        if chases is not None:
            chases_loss, _ = self.chases_head(embeddings, chases)
            losses['chases_loss'] = chases_loss
            loss += 0.05 * chases_loss
            #print(chases.shape)
        
        if lights is not None:
            lights_loss, _ = self.lights_head(embeddings, lights)
            losses['lights_loss'] = lights_loss
            loss += 0.5 * lights_loss
        
        if task_annotations is not None:
            #print("task:",task_annotations.shape)
            task_loss = self.task_head(embeddings, task_annotations)
            losses['task_loss'] = task_loss
            loss += 0.8 * task_loss
        
        if inter_mouse_task_annotations is not None:
            inter_task_loss = self.calculate_inter_mouse_loss(
                                    extracted_embeddings, 
                                    inter_mouse_task_annotations,
                                    #label_mask
                                    )
            losses['inter_task_loss'] = inter_task_loss
            loss += 0.8 * inter_task_loss
            
        if group_task_annotations is not None:
            group_task_loss = self.group_task_head(combined_embeddings, group_task_annotations)
            losses['group_task_loss'] = group_task_loss
            loss += 0.4 * group_task_loss

        if label_mask is not None:
            mm_loss = self.mm_head(embeddings, labels, label_mask)
            losses['masked_modelling_loss'] = mm_loss
            loss += mm_loss
        print(losses)

        return {
            'loss': loss, 
            'embeds':combined_embeddings,
            **losses
        }


    def training_step(self, batch, batch_idx):

        keypoints,label_mask,labels,\
            chases,lights,\
                task_annotations,inter_mouse_task_annotations,\
                    group_task_annotations=batch["keypoints"],batch["label_mask"],batch["labels"],\
                        batch["chases"],batch["lights"],\
                            batch["task_annotations"],batch["inter_mouse_task_annotations"],batch["group_task_annotations"]
        #inputs["train"]=torch.BoolTensor(1)
        #print(inputs["keypoints"])
        outputs=self.forward(keypoints,label_mask,labels,chases,lights,task_annotations,inter_mouse_task_annotations,group_task_annotations)
        
        loss=outputs["loss"]
        return loss


    def configure_optimizers(self):
        lrate=3e-5
        optimizer = torch.optim.Adam(self.parameters(),lr=lrate)
        #torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.3, last_epoch=-1)
        return optimizer