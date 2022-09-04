from gc import callbacks
from dataset.masked_modelling_sep_dataset import MaskedModelingSepDataset
from dataset.mice_dataset import get_split_indices
from dataset.full_sequence_sep_dataset import FullSequenceSepDataset
from torch.utils.data import DataLoader
import torch
import numpy as np
from consts import DEFAULT_NUM_TRAINING_POINTS, NUM_MICE
import argparse
from pathlib import Path
from typing import Union, List, Any, Dict,Mapping
from transforms.augmentations import training_augmentations
from transforms.keypoints_transform import get_svd_from_dataset
from transformers.integrations import TensorBoardCallback
from datetime import datetime
from dataset.utils import get_multitask_datasets
from tqdm import tqdm

from models.Model import SceneTransformer
#import models.GC as nnmodel
#from utils import disc_l2_loss,adv_disc_l2_loss
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import init

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

def build_submission(submission_dict):
    frame_number_map = {}
    embeddings = []

    i = 0
    for key, seq_embed in submission_dict.items():
        seq_len = seq_embed.shape[0]
        frame_number_map[key] = (i, i + seq_len)
        embeddings.append(seq_embed)

        i += seq_len
    
    embeddings = np.stack(embeddings).reshape(i, -1)

    return {'frame_number_map': frame_number_map, 'embeddings': embeddings}

def _prepare_input(data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        """
        Stolen from huggingface
        """
        if isinstance(data, Mapping):
            return type(data)({k: _prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(_prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            return data.to(device)
        return data

def get_test_embeddings(model, dataset, tasks=[]):
    model.to(device)
    model.eval()

    dataloader = dataset.get_dataloader()
    embedding_dict = {}
    annotation_dict = {task: {} for task in tasks}
    #print("#################################") 
    for seq_id, batch in tqdm(dataloader):
        #if cnt>=2598:
        #print(batch.keys())
        batch = _prepare_input(batch)
        
        keypoints,inter_mouse_task_annotations=batch["keypoints"],batch["inter_mouse_task_annotations"]
        
        #keypoints=keypoints.to(device)
        #print(keypoints.shape)
        outputs = model.forward(keypoints=keypoints,inter_mouse_task_annotations=inter_mouse_task_annotations,train=False)
        
        embeddings = outputs['embeds'].cpu().detach()
    
        embedding_dict[seq_id] = dataset.unsplit_overlap(embeddings)
       
        for task in tasks:
            annotation_dict[task][seq_id] = batch[task]    
        
        #cnt+=1
        #return embedding_dict,annotation_dict
        #if cnt==2602:
        #    return embedding_dict,annotation_dict
        #print(seq_id)
    #print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    return embedding_dict, annotation_dict


def test(model, test_dataset):
    '''Runs testing, return a submission'''
    embedding_dict, _ = get_test_embeddings(model, test_dataset)
    #print(embedding_dict)
    submission = build_submission(embedding_dict)
    return submission

def get_args() -> argparse.Namespace:
    """
    Loads args for main()
    """
    parser = argparse.ArgumentParser(
        description="Embedding training facility for mice."
    )
    # Training settings

    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default = 32,
        help="batch size",
    )

    parser.add_argument(
        "--max_seq_length",
        type=int,
        required=False,
        default=80,
        help="batch size",
    )

    parser.add_argument(
        "--sample_frequency",
        type=int,
        required=False,
        help="Size to scale the frame samples.",
    )

    parser.add_argument(
        "--mask_prob",
        type=float,
        required=False,
        default=0.4,
        help="Float ratio to split between train and val.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        required=False,
        default=10,
        help="number of epochs",
    )

    parser.add_argument(
        "--train_path",
        type=Path,
        required=False,
        default='./mouse_data/user_train.npy',
        help="Path to training dataset.",
    )
    parser.add_argument(
        "--test_path",
        type=Path,
        required=False,
        default='./mouse_data/submission_data.npy',
        help="Path to test dataset.",
    )

    parser.add_argument(
        "--train_ratio",
        type=float,
        required=False,
        default=0.95,
        help="Float ratio to split between train and val.",
    )

    args = parser.parse_args()

    return args


device="cuda:0"

args=get_args()

MaskedDataset=MaskedModelingSepDataset
train_datasets, val_datasets, feature_transformers = get_multitask_datasets(args, MaskedDataset)
#print(len(train_datasets))
train_dataloader = DataLoader(train_datasets[0], batch_size=args.batch_size, collate_fn=train_datasets[0].collator, num_workers=8)
val_dataloader= DataLoader(train_datasets[1], batch_size=args.batch_size, collate_fn=train_datasets[1].collator, num_workers=8)

#checkpoint_callback = ModelCheckpoint(monitor='val_loss')


#model = SceneTransformer.load_from_checkpoint('./scene_cls_mask/lightning_logs/version_29/checkpoints/epoch=204-step=9840.ckpt').to(device)
#model = FuseT(batch_size=16,num_stage=6,J=2,output_size=42, in_feat_dim=24, time_steps=60, feature_dim=256,device=device).to(device)
#model = FuseT(batch_size=32,num_stage=8,J=2,output_size=42, in_feat_dim=28, time_steps=80, feature_dim=256,device=device).to(device)
model= SceneTransformer(device=device,output_size=42, in_feat_dim=28, time_steps=80, feature_dim=256, head_num=4, k=4)
print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

trainer = pl.Trainer(max_epochs=450, gpus=1,default_root_dir="scene_cls_mask_nofeat")
#trainer.fit(model,train_dataloader,None,ckpt_path='./scene_cls_mask/lightning_logs/version_3/checkpoints/epoch=300-step=14448.ckpt')
trainer.fit(model, train_dataloader,None,None)


FullDataset = FullSequenceSepDataset


if args.test_path:
    test_dataset = FullDataset(
        path=args.test_path,  
        max_seq_length=args.max_seq_length,
        **feature_transformers
        )

    with torch.no_grad():
        submission = test(model, test_dataset)
print(submission["embeddings"].shape)
np.save('test_mouse_submission10.npy', submission) 