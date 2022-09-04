from cProfile import label
from dataset.mice_dataset import BaseMouseDataset
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Dict
from torchvision import transforms
#from tomcat.consts import NUM_MICE

NUM_MICE=3

class MaskedModelingSepDataset(BaseMouseDataset):
    '''
        Masked Modeling Dataset

        Notes:
            Masking is not done here, we only decide on the sequence indices where masking should take place,
            mask embeddings are then applied in the forward pass.

            Labels are identical to the input.
        
        Output:
            Collator function takes all the inputs and outputs
    '''
    
    def __init__(self, 
                max_seq_length: int = 50, 
                mask_prob: float = 0.15, 
                augmentations: transforms.Compose = None, 
                **kwargs):
        
        super().__init__(**kwargs)

        self.max_keypoints_len = max_seq_length - 2 # account for special tokens
        self.max_seq_length = max_seq_length
        self.mask_prob = mask_prob
        self.random_sequence_replace_prob = None

        self.augmentations = augmentations
    
    def sample_random_sequence(self):
        '''Collect a training sequence at random'''
        idx = np.random.randint(0, len(self))
        keypoints = self.keypoints[idx]
    
        start = np.random.randint(0, keypoints.shape[0] - self.max_keypoints_len)
        end = start + self.max_keypoints_len
        keypoints = keypoints[start:end, :]

        return keypoints

    #def sample_random_keypoints(self, length=1):
    #    '''Collect a training sample at random'''
    #    seq = self.sample_random_sequence()
    #    seq_idx = np.random.randint(0, len(seq)-length)
    #    mouse_idx = np.random.randint(0, NUM_MICE)
        #print("seq:",len(seq))
    #    return self.featurise_keypoints(seq[seq_idx:seq_idx + length])[:, mouse_idx]

    def sample_random_keypoints(self, length=1):
        '''Collect a training sample at random'''
        seq = self.sample_random_sequence()
        seq_idx = np.random.randint(0, len(seq)-length)
        mouse_idx = np.random.randint(0, NUM_MICE)
        keypoints = seq[seq_idx:seq_idx + length]
        feats = self.featurise_keypoints(keypoints)[:, mouse_idx]
        mouse_task_annotations = self.mouse_feats(keypoints)[:, mouse_idx]
        #print(mouse_task_annotations)
        return torch.cat([feats, mouse_task_annotations], dim=1)

    def mask_keypoints(self, feats: torch.tensor) -> Dict[str, torch.tensor]:
        '''
        Mask input keypoints according to mask_prob:
        Follow BERT:    
            80% masked points are denoted for masking in forward pass
            10% are replaced with a randomly sampled input
            10% are left

        Returns:
            Feats (torch.tensor): input feats
            Attention Mask (torch.tensor): attention mask
            Mask (torch.tensor): denoted input positions to be replaced with mask embedding
            Labels (torch.tensor): simply a copy of the input
        '''
        seq_len = len(feats)
        #print(seq_len)
        labels = feats.clone()
        label_mask = torch.zeros((seq_len, NUM_MICE, 1), dtype=torch.bool)
        masked_positions = torch.zeros((seq_len, NUM_MICE, 1), dtype=torch.bool)
        attention_mask = torch.ones((seq_len, NUM_MICE, 1), dtype=torch.bool)

        for m in range(NUM_MICE):
            for i in range(seq_len):

                rand = np.random.random()
                if rand < self.mask_prob:
                    label_mask[i, m] = 1

                    if rand < 0.8 * self.mask_prob:
                        # mask, don't need to change the features, these will instead be masked in the model forward pass
                        masked_positions[i, m] = 1

                    #elif rand < 0.9 * self.mask_prob:
                        # random
                        #print(i,m)
                        #print("feats:",feats[i,m].shape)
                        #print(self.sample_random_keypoints().shape)
                    #    feats[i, m] = self.sample_random_keypoints()

        return feats, attention_mask, masked_positions, labels, label_mask

    def add_special_tokens(self, tensor, default=0):
        length, mice, dim = tensor.shape
        new_tensor = torch.full((mice * length + 4, dim), fill_value=default, dtype=tensor.dtype)
        pos = 1
        for m in range(mice):
            start = pos
            end = pos + length
            new_tensor[start:end] = tensor[:, m]
            pos += length + 1

        return new_tensor

    def create_segments(self, total_seq_len, seq_len):
        segments = torch.zeros((total_seq_len), dtype=torch.long)
        pos = seq_len 
        for m in range(NUM_MICE):
            segments[pos:] += 1
            pos += 1 + seq_len

        return segments
    
    def random_sequence_replace(self, keypoints):
        '''
        Replaces one mouse with a randomly sampled mouse sequence.
        if hit, then half the time it is randomly sampled from dataset
        '''
        label = None
        if self.random_sequence_replace_prob:
            
            label = torch.zeros(1, dtype=torch.float32)

            if np.random.random() < self.random_sequence_replace_prob:
                random_sequence = self.sample_random_sequence()
                mouse_idx = np.random.randint(0, NUM_MICE)
                keypoints[:, mouse_idx, :] = random_sequence[:, mouse_idx, :]
                label += 1

        return keypoints, label

    def get_sub_sequence(self, idx: int):
        sequence = self.keypoints[idx]

        start = np.random.randint(0, sequence.shape[0] - self.max_keypoints_len-1)
        end = start + self.max_seq_length
        #end = start + self.max_keypoints_len
        keypoints = sequence[start:end, :]

        annotations = {}
        if self.has_annotations:
            #print(self.annotations["lights"][idx].shape)
            #print(self.annotations["lights"][idx][start:end].shape)
            annotations = {k: torch.tensor(v[idx][start:end]) for (k,v) in self.annotations.items()}
            annotations = {k: v[:, None, None].repeat(1, NUM_MICE, 1) for (k,v) in annotations.items()}
        #print(annotations["lights"].shape)


        return keypoints, annotations
    
    def shuffle_mice(self, sequence):
        mice_indices = np.arange(NUM_MICE)
        np.random.shuffle(mice_indices)
        sequence = sequence[:, mice_indices, :]
        return sequence

    def get_training_sample(self, idx: int) -> dict:
        '''
        Returns a training sample
        
        Randomly samples a section with length self.max_keypoints_len of the input sequence.
        Adds dummy inputs for special CLS and SEP tokens, these are replaced in the forward pass with learnable embeddings

        Applies masking and returns model inputs

        '''
        keypoints, annotations = self.get_sub_sequence(idx)
        #print("keypoints:",keypoints.shape)

        keypoints = self.shuffle_mice(keypoints)
        
        # Randomly replace a keypoint
        #keypoints, random_replace_label = self.random_sequence_replace(keypoints)

        if self.augmentations:
            keypoints = self.augmentations(keypoints)
        
        feats = self.featurise_keypoints(keypoints)
        #print("feats:",feats.shape) [seq_length,agent,28]
        mouse_task_annotations, inter_mouse_task_annotations, group_task_annotations = self.create_task_annotations(keypoints)
        #print("feat1:",feats.shape)
        #print("mouse_task_annotations:",mouse_task_annotations.shape) #[seq_length,agent,60]
        #print(11111111111111111111111)
        #feats = torch.cat([feats, mouse_task_annotations], dim=2)
        #print("feat2:",feats.shape)
        feats, attention_mask, masked_positions, labels, label_mask = self.mask_keypoints(feats)
        #print("attention_mask:",attention_mask.shape)
        #rint("masked_positions:",masked_positions.shape)
        #print("label_mask:",label_mask.shape)
        '''
        feats = self.add_special_tokens(feats)
        attention_mask = self.add_special_tokens(attention_mask, 1)
        masked_positions = self.add_special_tokens(masked_positions)
        labels = self.add_special_tokens(labels)
        label_mask = self.add_special_tokens(label_mask)
        task_annotations = self.add_special_tokens(mouse_task_annotations)
        '''
        # Do segment ids
        #segments = self.create_segments(len(feats), self.max_seq_length)
        #print('inter_mouse_task_annotations:', inter_mouse_task_annotations.shape)
        #print('group_task_annotations:', group_task_annotations.shape)
        inputs = {
            'keypoints': feats, 
            #'attention_mask': attention_mask, 
            #'masked_positions': masked_positions, 
            'labels': labels,
            'label_mask': label_mask,
            #'segments': segments,
            #'sequence_replace_label': random_replace_label,
            'task_annotations': mouse_task_annotations,
            'inter_mouse_task_annotations': inter_mouse_task_annotations,
            'group_task_annotations': group_task_annotations
            }
        
        #print(annotations["lights"].shape)
        
        #print(inputs["keypoints"].shape)
        return {**inputs, **annotations} 

    def __getitem__(self, idx: int) -> dict:
        #print(1111111111111111111111111)
        inputs = self.get_training_sample(idx)
        #print("inputs:",inputs["keypoints"].shape)
        return inputs
    
    def collator(self, batch: tuple) -> dict:
        '''
        Data Collator
        Stacks the input batch
        
        We need special treatment for inter_mouse_task_annotations, since it is a nested dict
        '''
        #print("batch:",batch["keypoints"].shape)
        inputs = {}
        keys = batch[0].keys()
        #print(keys)
        #for i in batch:
            #print(i["lights"].shape)
        for k in keys:
            if batch[0][k] is None:
                continue
            
            #print(k)

            if k == 'inter_mouse_task_annotations':
                inter_mouse_task_annotations = {}
                for m1 in range(NUM_MICE):
                    inter_mouse_task_annotations[m1] = {m2: torch.stack([i[k][m1][m2] for i in batch]) for m2 in batch[0][k][m1].keys()}
                
                inputs[k] = inter_mouse_task_annotations
            
            else:
                inputs[k] = torch.stack([i[k] for i in batch])
        #print("inputs:",inputs)
        return inputs
