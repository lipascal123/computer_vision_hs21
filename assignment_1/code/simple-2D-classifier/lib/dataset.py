import numpy as np

import os

import torch
from torch._C import TracingState
from torch.utils.data import Dataset


class Simple2DDataset(Dataset):
    def __init__(self, split='train'):
        super().__init__()
        assert split in ['train', 'valid'], f'Split parameters "{split}" must be either "train" or "valid".'
        # Read either train or validation data from disk based on split parameter using np.load.
        # Data is located in the folder "data".
        # Hint: you can use os.path.join to obtain a path in a subfolder.
        # Save samples and annotations to class members self.samples and self.annotations respectively.
        # Samples should be an Nx2 numpy array. Annotations should be Nx1.
        
        # Load data and save it to np array 
        # dircetory of this file      
        dirname = os.path.dirname(__file__) 
        # filename of data files to be loaded 
        filename = os.path.abspath(os.path.join(dirname, "..", "data/"+split+".npz"))
        # load data as npz
        npz = np.load(filename) 

        # extract the files in the compressed npz to np array
        #print("FILES",npz.files)
        self.samples = npz['samples']
        self.anotations =npz['annotations']
            
    def __len__(self):
        # Returns the number of samples in the dataset.
        return self.samples.shape[0]
    
    def __getitem__(self, idx):
        # Returns the sample and annotation with index idx.

        # get a single data sample
        sample = self.samples[idx,:]
        annotation = self.anotations[idx]
        
        # Convert to tensor.
        return {
            'input': torch.from_numpy(sample).float(),
            'annotation': torch.from_numpy(annotation[np.newaxis]).float()
        }


class Simple2DTransformDataset(Dataset):
    def __init__(self, split='train'):
        super().__init__()
        assert split in ['train', 'valid'], f'Split parameters "{split}" must be either "train" or "valid".'
        # Read either train or validation data from disk based on split parameter.
        # Data is located in the folder "data".
        # Hint: you can use os.path.join to obtain a path in a subfolder.
        # Save samples and annotations to class members self.samples and self.annotations respectively.
        # Samples should be an Nx2 numpy array. Annotations should be Nx1.
        
        # Load data and save it to np array 
        # dircetory of this file      
        dirname = os.path.dirname(__file__) 
        # filename of data files to be loaded 
        filename = os.path.abspath(os.path.join(dirname, "..", "data/"+split+".npz"))
        # load data as npz
        npz = np.load(filename) 

        # extract the files in the compressed npz to np array
        #print("FILES",npz.files)
        self.samples = npz['samples']
        self.anotations =npz['annotations']
            
    def __len__(self):
        # Returns the number of samples in the dataset.
        return self.samples.shape[0]
    
    def __getitem__(self, idx):
        # Returns the sample and annotation with index idx.
        
        # get a single data sample
        sample = self.samples[idx,:]
        annotation = self.anotations[idx]
        
        # Transform the sample to a different coordinate system.
        sample = transform(sample)

        # Convert to tensor.
        return {
            'input': torch.from_numpy(sample).float(),
            'annotation': torch.from_numpy(annotation[np.newaxis]).float()
        }


def transform(sample):
    # create new np.array by coping by slices
    new_sample = sample[:]   
    new_sample[0] = np.sqrt((sample[0]**2) + (sample[1]**2))
    new_sample[1] = np.arctan2(sample[1],sample[0])
    return new_sample
