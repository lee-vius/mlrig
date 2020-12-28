import sys
import os
from mlmodel import Network_diff, Network_anchor
from data_handler import DeformData, TestData
import numpy as np
import pandas as pd
import scipy
import scipy.sparse.linalg as linalg
from scipy.sparse import csr_matrix
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm # Displays a progress bar

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, DataLoader, random_split


# Define a class containing data specially for reconstruction
class reconstru():
    def __init__(self, topology_path):
        # Variable denoting whether the matrices have been calculated
        self.isCalculated = False
        # Variables for reconstruction
        self.VALENCES = []
        self.LAPLACIAN_MTX = None
        self.REVERSE_CUTHILL = None
        self.REVERSE_CUTHILL_MTX = None
        self.INVERSE_CMO = None
        self.CHOLESKY_MTX = None

        self.topology_path = topology_path


    def cal_Lmatrice(self, anchor_index, anchor_offsets, diffOffset):
        # Calculate the laplacian matrix, cholesky matrix and reverse_cuthill_mckee_order
        connectionMap = {}
        # Read in the connection map
        f = open(self.topology_path, 'r')
        reader = csv.reader(f)
        content = list(reader)
        for line in content:
            if len(line) > 1:
                connectionMap[int(line[0])] = [int(c) for c in line[1:]]
            else:
                connectionMap[int(line[0])] = []
        f.close()

        # handle essential data
        valences = []

        num_vtx = diffOffset.shape[0]
        vid_dict = np.array(range(num_vtx))
        col_ct = num_vtx
        row_ct = num_vtx + anchor_index.shape[0]

        # calculate the variables
        valences = np.array([len(connectionMap[i]) for i in vid_dict], dtype=np.float)
        laplacian_raw = np.zeros((row_ct, col_ct), dtype=np.float)
        for i in vid_dict:
            laplacian_raw[i, i] = valences[i]
            for j in connectionMap[i]:
                laplacian_raw[i, j] = -1.0

        for i, anchor_i in enumerate(anchor_index):
            # This place need to fill the anchor weight, currently set as 1.0
            laplacian_raw[i + num_vtx, anchor_i] = 1.0

        # calculate laplacian matrix
        laplacian_matrix = csr_matrix(laplacian_raw, shape=(row_ct, col_ct))
        # calculate normal matrix
        normal_matrix = laplacian_matrix.transpose() * laplacian_matrix

        # calculate reverse cuthill
        reverse_cuthill = scipy.sparse.csgraph.reverse_cuthill_mckee(normal_matrix, symmetric_mode=True)
        reverse_cuthill_mtx = np.ndarray(shape=(col_ct, col_ct))
        inverse_ary = [0] * col_ct
        reverse_cuthill_mtx = normal_matrix[reverse_cuthill, :]
        reverse_cuthill_mtx = reverse_cuthill_mtx[:, reverse_cuthill].A

        for i in range(col_ct):
            row_index = reverse_cuthill[i]
            inverse_ary[row_index] = i

        cholesky_matrix = scipy.linalg.cholesky(reverse_cuthill_mtx, lower=True, overwrite_a=True, check_finite=False)

        print("Laplacian matrix calculated")
        # Assign values
        self.VALENCES = valences.reshape((num_vtx, 1))
        self.LAPLACIAN_MTX = laplacian_matrix
        self.INVERSE_CMO = inverse_ary
        self.REVERSE_CUTHILL = reverse_cuthill
        self.REVERSE_CUTHILL_MTX = reverse_cuthill_mtx
        self.CHOLESKY_MTX = cholesky_matrix


# Define test file path
test_dir_root = "/Users/levius/Desktop/高级图像图形学/项目/code/ml/maya_split/gen_data/data_set"
root_dir = "/Users/levius/Desktop/高级图像图形学/项目/code/ml/maya_split/gen_data/temp_data"
input_dir = "/Users/levius/Desktop/高级图像图形学/项目/code/ml/maya_split/gen_data/mover_rigged"
anchor_dir = "/Users/levius/Desktop/高级图像图形学/项目/code/ml/maya_split/gen_data/anchorPoints.csv"
param_save_path = "/Users/levius/Desktop/高级图像图形学/项目/code/ml/maya_split/mlmodel/model_param/"
topology_path = "/Users/levius/Desktop/高级图像图形学/项目/code/ml/maya_split/gen_data/topology_Mery_geo_cn_body.csv"

# create the reconstructor
RECONSTRUCTOR = reconstru(topology_path)

if sys.platform == 'win32':
    # Redefine folder path if the system is windows
    test_dir_root = "D:/ACG/project/ml/maya_split/gen_data/data_set"
    root_dir = "D:/ACG/project/ml/maya_split/gen_data/temp_data"
    input_dir = "D:/ACG/project/ml/maya_split/gen_data/mover_rigged"
    anchor_dir = "D:/ACG/project/ml/maya_split/gen_data/anchorPoints.csv"
    param_save_path = "D:/ACG/project/ml/maya_split/mlmodel/model_param/trained/"
    topology_path = "D:/ACG/project/ml/maya_split/gen_data/topology_Mery_geo_cn_body.csv"


# Load in the test dataset
print("Loading testing datasets...")
DATA_TEST = TestData(root_dir=test_dir_root, input_dir=anchor_dir)
DATA_VALID = DeformData(root_dir=root_dir, inputD_dir=input_dir)
test_loader = DataLoader(DATA_TEST, batch_size=1)
valid_loader = DataLoader(DATA_VALID, batch_size=1)
print("Loading Done!")

device = "cpu"  # currently gpu not available

LOAD = True

# create three training models for x, y, z coordinates
models = {'x': Network_diff(540, 12942, 2048, dropout=0.5).to(device),
          'y': Network_diff(540, 12942, 2048, dropout=0.5).to(device),
          'z': Network_diff(540, 12942, 2048, dropout=0.5).to(device)}

if LOAD:
    # read in the model last time trained
    models['x'].load_state_dict(torch.load(param_save_path + 'model_states_x.pth'), strict=True)
    models['y'].load_state_dict(torch.load(param_save_path + 'model_states_y.pth'), strict=True)
    models['z'].load_state_dict(torch.load(param_save_path + 'model_states_z.pth'), strict=True)

# create anchor points models
ANCHOR_NUM = 107
anchor_models = []
for i in range(ANCHOR_NUM):
    anchor_models.append(Network_anchor(540, 3, 64, dropout=0.0))

if LOAD:
    for i, m in enumerate(anchor_models):
        m.load_state_dict(torch.load(param_save_path + '/anchor_models/model{}.pth'.format(i)))

# create three training models for localoffset x, y, z coordinates
local_models = {'x': Network_diff(540, 12942, 2048, dropout=0.5).to(device),
                'y': Network_diff(540, 12942, 2048, dropout=0.5).to(device),
                'z': Network_diff(540, 12942, 2048, dropout=0.5).to(device)}

if LOAD:
    # read in the model last time trained
    local_models['x'].load_state_dict(torch.load(param_save_path + 'local_states_x.pth'))
    local_models['y'].load_state_dict(torch.load(param_save_path + 'local_states_y.pth'))
    local_models['z'].load_state_dict(torch.load(param_save_path + 'local_states_z.pth'))


def predict_localoffset(diff_models, anchor_models, anchor_num, test_index, loader):
    # the function predict the final localoffset result
    diffOffset = predict_diffoffset(diff_models, test_index, loader)
    anchor_offsets, anchor_index = predict_anchors(anchor_models, anchor_num, test_index, loader)
    # reconstruct coord
    recons_localoffset = reconstruct_localoffset(anchor_index, anchor_offsets, diffOffset)
    return diffOffset, anchor_offsets , recons_localoffset


def predict_diffoffset(diff_models, test_index, loader):
    # the function predict the final differential result
    diff_models['x'].eval()
    diff_models['y'].eval()
    diff_models['z'].eval()
    with torch.no_grad(): # Do not calculate grident to speed up computation
        batch_data = None
        for i, b in enumerate(loader):
            if i == test_index:
                batch_data = b
        batch = torch.tensor(batch_data['mover_value'].reshape(1, -1), dtype=torch.float32).to(device)
        # predict the results
        pred_x = diff_models['x'](batch)
        pred_y = diff_models['y'](batch)
        pred_z = diff_models['z'](batch)
        pred_x = np.expand_dims(pred_x, axis=2)
        pred_y = np.expand_dims(pred_y, axis=2)
        pred_z = np.expand_dims(pred_z, axis=2)
        result = np.concatenate((pred_x, pred_y, pred_z), axis=2)
        return np.array(result[0])


def predict_anchors(anchor_models, anchor_num, test_index, loader):
    # the function predicts the anchors localoffset
    for i in range(anchor_num):
        anchor_models[i].eval()
    with torch.no_grad(): # Do not calculate gradient to speed up computation
        batch_data = None
        for i, b in enumerate(loader):
            if i == test_index:
                batch_data = b
        batch = torch.tensor(batch_data['mover_value'].reshape(1, -1), dtype=torch.float32).to(device)
        anchor_index = batch_data['anchor_index']
        # predict results
        pred = np.concatenate([np.expand_dims(ml(batch), axis=1) for ml in anchor_models], axis=1)
        return np.array(pred[0]), np.array(anchor_index[0])


def reconstruct_localoffset(anchor_index, anchor_offsets, diffoffset):
    # the function reconstruct the localoffset
    col_ct = diffoffset.shape[0]
    # based on anchors and differential offset
    if not RECONSTRUCTOR.isCalculated:
        RECONSTRUCTOR.cal_Lmatrice(anchor_index, anchor_offsets, diffoffset)
        RECONSTRUCTOR.isCalculated = True
    print("Laplacian calculated")

    # construct differential offsets timed by their valance
    diff_coord = diffoffset * RECONSTRUCTOR.VALENCES
    anchor_weight = 1.0
    anchor_coord = anchor_offsets * anchor_weight
    # concatenate anchor and differential together
    diff_coord = np.concatenate([diff_coord, anchor_coord], axis=0)
    # calculate modified diff coordinates
    modified_diff_coord = RECONSTRUCTOR.LAPLACIAN_MTX.transpose() * diff_coord

    # reorder coordinates based on reverse cuthill
    reordered_diff_coord = np.array([modified_diff_coord[RECONSTRUCTOR.REVERSE_CUTHILL[i], :] for i in range(col_ct)])

    # back substitution to solve triangular matrices
    solve_coord = scipy.linalg.solve_triangular(RECONSTRUCTOR.CHOLESKY_MTX, reordered_diff_coord, lower=True, overwrite_b=True, check_finite=False)
    solve_real_coord = scipy.linalg.solve_triangular(RECONSTRUCTOR.CHOLESKY_MTX.T, solve_coord, lower=False, overwrite_b=True, check_finite=False)

    real_coord = np.array([solve_real_coord[RECONSTRUCTOR.INVERSE_CMO[i]][:] for i in range(col_ct)])

    return real_coord


def get_label(valid_index, loader):
    batch_data = None
    for i, b in enumerate(loader):
        if i == valid_index:
            batch_data = b
    diff_label = np.array(batch_data['differentialOffset'], dtype=np.float)
    local_label = np.array(batch_data['localOffset'], dtype=np.float)
    anchor_ind = np.array(batch_data['anchor_index'], dtype=np.int)
    anchor_offset = np.array(batch_data['anchor_offsets'], dtype=np.float)
    return diff_label[0], local_label[0], anchor_ind[0], anchor_offset[0]


diffOffset, anchor_offsets, recons_localoffset = predict_localoffset(models, anchor_models, ANCHOR_NUM, 5, valid_loader)
diff_label, local_label, anchor_ind, anchor_label = get_label(5, valid_loader)
real_coord = reconstruct_localoffset(anchor_ind, anchor_label, diff_label)

print(np.mean(local_label - real_coord, axis=0))
print(np.mean(local_label - recons_localoffset, axis=0))

print(np.mean(abs(diff_label - diffOffset), axis=0))
# print(np.mean(abs(anchor_label - anchor_offsets) , axis=0))
print(np.mean(anchor_label, axis=0))
print(np.mean(anchor_offsets, axis=0))
