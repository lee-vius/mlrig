import os
import csv

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# from tqdm import tqdm # Displays a progress bar

# import torch
# from torch import nn
# from torch import optim
# import torch.nn.functional as F
# from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, DataLoader, random_split


class DeformData(Dataset):  # inherant from Dataset
    def __init__(self, root_dir, inputD_dir, transform=None):
        self.root_dir = root_dir # 数据集的根目录
        self.input_dir = inputD_dir
        self.transform = transform # 自定义的transform
        temp_folder = os.listdir(self.root_dir)
        self.data_folders = [i for i in temp_folder if i[:6] == "rigged"]

        self.data_set = {}

    
    def __len__(self): # return the size of the dataset
        return len(self.data_folders)


    def __getitem__(self, index): # return dataset[index] according to the index
        # check memory if have stored the data
        if index in self.data_set:
            return self.data_set[index]
        folder_index = self.data_folders[index]
        folder_path = os.path.join(self.root_dir, folder_index)
        input_path = os.path.join(self.input_dir, folder_index + '.csv')
        # read in data
        pose_data = self.read_in_data_two(folder_path, input_path)

        if self.transform:
            pose_data = self.transform(pose_data)

        self.data_set[index] = pose_data
        return pose_data

    
    @staticmethod
    def read_in_data(filepath):
        # get the data files to read
        files = []
        for _, _, fs in os.walk(filepath):
            files = fs
        # read in the result
        pose_data = {}
        for file_name in files:
            # for each file, read in as a dict
            f = open(filepath + '/' + file_name, 'r')
            temp = {}
            reader = csv.reader(f)
            content = list(reader)
            if file_name[:-4] in ['anchorPoints', 'differentialOffset', 'localOffset', 'worldOffset', 'worldPos']:
                for line in content:
                    temp[int(line[0])] = [float(coord) for coord in line[1:]]
            elif file_name[:-4] in ['jointLocalMatrix', 'jointLocalQuaternion', 'jointWorldMatrix', 'jointWorldQuaternion']:
                for line in content:
                    temp[line[0]] = [float(coord) for coord in line[1:]]
            else:
                for line in content:
                    temp[line[0]] = line[1:]
            f.close()
            pose_data[file_name[:-4]] = temp

        return pose_data

    @staticmethod
    def read_in_data_two(filepath, mover_file):
        # get the data files to read
        files = os.listdir(filepath)
        # read in the label data
        pose_data = {}
        for file_name in files:
            if file_name[:-4] in ['differentialOffset', 'localOffset']:
                data_type = {
                    'index': np.int32,
                    'x': np.float,
                    'y': np.float,
                    'z': np.float
                }
                df = pd.read_csv(filepath + '/' + file_name, header=None, names=list(data_type.keys()), dtype=data_type)
                if 'vtx_index' not in pose_data:
                    pose_data['vtx_index'] = np.array(df['index'], dtype=np.int32)
                pose_data[file_name[:-4]] = np.array(df[['x', 'y', 'z']], dtype=np.float)
            
            elif file_name[:-4] == 'anchorPoints':
                data_type = {
                    'index': np.int32,
                    'x': np.float,
                    'y': np.float,
                    'z': np.float
                }
                df = pd.read_csv(filepath + '/' + file_name, header=None, names=list(data_type.keys()), dtype=data_type)
                pose_data['anchor_index'] = np.array(df['index'], dtype=np.int32)
                pose_data['anchor_offsets'] = np.array(df[['x', 'y', 'z']], dtype=np.float)
                
            elif file_name[:-4] in ['jointLocalMatrix', 'jointLocalQuaternion', 'jointWorldMatrix', 'jointWorldQuaternion']:
                pass
            else:
                pass
        
        # read in mover data
        df = pd.read_csv(mover_file, header=0)
        attrs = list(df.keys())
        pose_data['mover_name'] = list(df[attrs[0]])
        pose_data['mover_value'] = np.array(df[attrs[1:]], dtype=np.float)
        return pose_data

    @staticmethod
    def read_in_rig(filepath):
        # read in the result as a dict
        f = open(filepath, 'r')
        reader = list(csv.reader(f))
        pose_rig = {}
        # get the attribute to set
        attribute = reader[0][1:]
        for line in reader[1:]:
            mover = line[0]
            # get each attribute
            for i, value in enumerate(line[1:]):
                pose_rig[mover + '.' + attribute[i]] = float(value)
        f.close()
        return pose_rig 
