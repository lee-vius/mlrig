import sys
import os
from mlmodel import Network_anchor, Network_diff
from data_handler import DeformData, TestData
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # Displays a progress bar

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, DataLoader, random_split


# Load the dataset and train, val, test splits
# TODO: Define the folder path of data
root_dir = "/Users/levius/Desktop/高级图像图形学/项目/code/ml/maya_split/gen_data/temp_data"
input_dir = "/Users/levius/Desktop/高级图像图形学/项目/code/ml/maya_split/gen_data/mover_rigged"
fig_dir = "/Users/levius/Desktop/高级图像图形学/项目/code/ml/maya_split/mlmodel/fig_output"
param_save_path = "/Users/levius/Desktop/高级图像图形学/项目/code/ml/maya_split/mlmodel/model_param/"

if sys.platform == 'win32':
    # Redefine folder path if the system is windows
    root_dir = "D:/ACG/project/ml/maya_split/gen_data/temp_data"
    input_dir = "D:/ACG/project/ml/maya_split/gen_data/mover_rigged"
    fig_dir = "D:/ACG/project/ml/maya_split/mlmodel/fig_output"
    param_save_path = "D:/ACG/project/ml/maya_split/mlmodel/model_param/"


# Load in the dataset
print("Loading datasets...")
DATA_DEFORM = DeformData(root_dir=root_dir, inputD_dir=input_dir)
DATA_TRAIN = Subset(DATA_DEFORM, range(400))
DATA_VAL = Subset(DATA_DEFORM, range(400, 500))
# TODO: adjust the batch size
train_loader = DataLoader(DATA_TRAIN, batch_size=100, shuffle=True)
val_loader = DataLoader(DATA_VAL, batch_size=100, shuffle=True)

# for things in train_loader:
#     print(things['mover_value'].size())

print("Done!")

# TODO: Construct train process
# device = "cuda" if torch.cuda.is_available() else "cpu"  # Configure device
device = "cpu"  # currently gpu not available
print("Train Device is {}".format(device))

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

# TODO: choose your loss function
criterion = nn.L1Loss()

# TODO: adjust optimizer, learning rate, weight decay according to your need
optimizers = {'x': optim.SGD(models['x'].parameters(), lr=0.1, weight_decay=1e-6),
              'y': optim.SGD(models['y'].parameters(), lr=0.1, weight_decay=1e-6),
              'z': optim.SGD(models['z'].parameters(), lr=0.1, weight_decay=1e-6)}


# TODO: create anchor points models
ANCHOR_NUM = 107
anchor_criterion = nn.MSELoss()

anchor_models = []
for i in range(ANCHOR_NUM):
    anchor_models.append(Network_anchor(540, 3, 64, dropout=0.0))

if LOAD:
    for i, m in enumerate(anchor_models):
        m.load_state_dict(torch.load(param_save_path + '/anchor_models/model{}.pth'.format(i)), strict=True)

anchor_optimizers = []
for i in range(ANCHOR_NUM):
    anchor_optimizers.append(optim.SGD(anchor_models[i].parameters(), lr=0.1, weight_decay=1e-6))

# create three training models for localoffset x, y, z coordinates
local_models = {'x': Network_diff(540, 12942, 2048, dropout=0.5).to(device),
                'y': Network_diff(540, 12942, 2048, dropout=0.5).to(device),
                'z': Network_diff(540, 12942, 2048, dropout=0.5).to(device)}

if LOAD:
    # read in the model last time trained
    local_models['x'].load_state_dict(torch.load(param_save_path + 'local_states_x.pth'), strict=True)
    local_models['y'].load_state_dict(torch.load(param_save_path + 'local_states_y.pth'), strict=True)
    local_models['z'].load_state_dict(torch.load(param_save_path + 'local_states_z.pth'), strict=True)

# TODO: choose your loss function
local_criterion = nn.MSELoss()

# TODO: adjust optimizer, learning rate, weight decay according to your need
local_optimizers = {'x': optim.SGD(local_models['x'].parameters(), lr=0.1, weight_decay=1e-6),
                    'y': optim.SGD(local_models['y'].parameters(), lr=0.1, weight_decay=1e-6),
                    'z': optim.SGD(local_models['z'].parameters(), lr=0.1, weight_decay=1e-6)}


# TODO: choose an appropriate number of epoch
num_epoch = 10
ROUND = 1


def train(models, train_loader, val_loader, num_epoch = 10, fig_prefix=0): # Train the model
    print("Start training...")
    # Set the model to training mode
    models['x'].train()
    models['y'].train()
    models['z'].train()
    # Define loss
    train_loss = {"x": [], "y": [], "z": []}
    valid_loss = {"x": [], "y": [], "z": []}
    for i in range(num_epoch):
        running_loss = {'x': [], 'y': [], 'z': []}
        for batch_data in tqdm(train_loader):
            batch = torch.tensor(batch_data['mover_value'].reshape(100, -1), dtype=torch.float32).to(device)
            label = torch.tensor(batch_data['differentialOffset'], dtype=torch.float32).to(device)
            # initialize optimizers
            optimizers['x'].zero_grad()
            optimizers['y'].zero_grad()
            optimizers['z'].zero_grad()
            # predict x results
            pred_x = models['x'](batch)
            loss_x = criterion(pred_x, label[:, :, 0])
            running_loss['x'].append(loss_x.item())
            loss_x.backward()
            optimizers['x'].step()
            # predict y results
            pred_y = models['y'](batch)
            loss_y = criterion(pred_y, label[:, :, 1])
            running_loss['y'].append(loss_y.item())
            loss_y.backward()
            optimizers['y'].step()
            # predict z results
            pred_z = models['z'](batch)
            loss_z = criterion(pred_z, label[:, :, 2])
            running_loss['z'].append(loss_z.item())
            loss_z.backward()
            optimizers['z'].step()

        # Print the average loss for this epoch
        print("Epoch {} loss X:{}, Y:{}, Z:{}".format(
            i+1 ,np.mean(running_loss['x']), np.mean(running_loss['y']), np.mean(running_loss['z'])
            ))
        train_loss['x'].append(np.mean(running_loss['x']))
        train_loss['y'].append(np.mean(running_loss['y']))
        train_loss['z'].append(np.mean(running_loss['z']))
        # Get validation loss
        val_loss_x, val_loss_y, val_loss_z = evaluate(models, val_loader)
        valid_loss['x'].append(val_loss_x)
        valid_loss['y'].append(val_loss_y)
        valid_loss['z'].append(val_loss_z)

    print("Done!")
    # TODO: comment following if not want to output loss history
    plot_loss_history(train_loss['x'], valid_loss['x'], type='X_round{}'.format(fig_prefix))
    plot_loss_history(train_loss['y'], valid_loss['y'], type='Y_round{}'.format(fig_prefix))
    plot_loss_history(train_loss['z'], valid_loss['z'], type='Z_round{}'.format(fig_prefix))


def evaluate(models, loader): # Evaluate accuracy on validation / test set
    # Set the model to evaluation mode
    models['x'].eval() 
    models['y'].eval()
    models['z'].eval()
    with torch.no_grad(): # Do not calculate grident to speed up computation
        running_loss = {'x': [], 'y': [], 'z': []}
        for batch_data in tqdm(loader):
            batch = torch.tensor(batch_data['mover_value'].reshape(100, -1), dtype=torch.float32).to(device)
            label = torch.tensor(batch_data['differentialOffset'], dtype=torch.float32).to(device)
            # predict x results
            pred_x = models['x'](batch)
            loss_x = criterion(pred_x, label[:, :, 0])
            running_loss['x'].append(loss_x.item())
            # predict y results
            pred_y = models['y'](batch)
            loss_y = criterion(pred_y, label[:, :, 1])
            running_loss['y'].append(loss_y.item())
            # predict z results
            pred_z = models['z'](batch)
            loss_z = criterion(pred_z, label[:, :, 2])
            running_loss['z'].append(loss_z.item())

    print("Evaluation loss X: {}, Y: {}, Z: {}".format(
        np.mean(running_loss['x']), np.mean(running_loss['y']), np.mean(running_loss['z'])
        ))
    return np.mean(running_loss['x']), np.mean(running_loss['y']), np.mean(running_loss['z'])


def train_local(models, train_loader, val_loader, num_epoch = 10, fig_prefix=0): # Train the model
    print("Start training localoffset...")
    # Set the model to training mode
    models['x'].train()
    models['y'].train()
    models['z'].train()
    # Define loss
    train_loss = {"x": [], "y": [], "z": []}
    valid_loss = {"x": [], "y": [], "z": []}
    for i in range(num_epoch):
        running_loss = {'x': [], 'y': [], 'z': []}
        for batch_data in tqdm(train_loader):
            batch = torch.tensor(batch_data['mover_value'].reshape(100, -1), dtype=torch.float32).to(device)
            label = torch.tensor(batch_data['localOffset'], dtype=torch.float32).to(device)
            # initialize optimizers
            local_optimizers['x'].zero_grad()
            local_optimizers['y'].zero_grad()
            local_optimizers['z'].zero_grad()
            # predict x results
            pred_x = models['x'](batch)
            loss_x = local_criterion(pred_x, label[:, :, 0])
            running_loss['x'].append(loss_x.item())
            loss_x.backward()
            local_optimizers['x'].step()
            # predict y results
            pred_y = models['y'](batch)
            loss_y = local_criterion(pred_y, label[:, :, 1])
            running_loss['y'].append(loss_y.item())
            loss_y.backward()
            local_optimizers['y'].step()
            # predict z results
            pred_z = models['z'](batch)
            loss_z = local_criterion(pred_z, label[:, :, 2])
            running_loss['z'].append(loss_z.item())
            loss_z.backward()
            local_optimizers['z'].step()

        # Print the average loss for this epoch
        print("Epoch {} loss X:{}, Y:{}, Z:{}".format(
            i+1 ,np.mean(running_loss['x']), np.mean(running_loss['y']), np.mean(running_loss['z'])
            ))
        train_loss['x'].append(np.mean(running_loss['x']))
        train_loss['y'].append(np.mean(running_loss['y']))
        train_loss['z'].append(np.mean(running_loss['z']))
        # Get validation loss
        val_loss_x, val_loss_y, val_loss_z = evaluate_local(models, val_loader)
        valid_loss['x'].append(val_loss_x)
        valid_loss['y'].append(val_loss_y)
        valid_loss['z'].append(val_loss_z)

    print("Done!")
    # TODO: comment following if not want to output loss history
    plot_loss_history(train_loss['x'], valid_loss['x'], type='local_X_round{}'.format(fig_prefix))
    plot_loss_history(train_loss['y'], valid_loss['y'], type='local_Y_round{}'.format(fig_prefix))
    plot_loss_history(train_loss['z'], valid_loss['z'], type='local_Z_round{}'.format(fig_prefix))


def evaluate_local(models, loader): # Evaluate accuracy on validation / test set
    # Set the model to evaluation mode
    models['x'].eval() 
    models['y'].eval()
    models['z'].eval()
    with torch.no_grad(): # Do not calculate grident to speed up computation
        running_loss = {'x': [], 'y': [], 'z': []}
        for batch_data in tqdm(loader):
            batch = torch.tensor(batch_data['mover_value'].reshape(100, -1), dtype=torch.float32).to(device)
            label = torch.tensor(batch_data['localOffset'], dtype=torch.float32).to(device)
            # predict x results
            pred_x = models['x'](batch)
            loss_x = local_criterion(pred_x, label[:, :, 0])
            running_loss['x'].append(loss_x.item())
            # predict y results
            pred_y = models['y'](batch)
            loss_y = local_criterion(pred_y, label[:, :, 1])
            running_loss['y'].append(loss_y.item())
            # predict z results
            pred_z = models['z'](batch)
            loss_z = local_criterion(pred_z, label[:, :, 2])
            running_loss['z'].append(loss_z.item())

    print("Evaluation loss X: {}, Y: {}, Z: {}".format(
        np.mean(running_loss['x']), np.mean(running_loss['y']), np.mean(running_loss['z'])
        ))
    return np.mean(running_loss['x']), np.mean(running_loss['y']), np.mean(running_loss['z'])


def train_anchor(anchor_models, anchor_num, train_loader, val_loader, num_epoch = 10, fig_prefix=0): # Train the model
    print("Start training...")
    # Set train mode
    for model in anchor_models:
        model.train()
    # Define loss
    train_loss = [[] for i in range(anchor_num)]
    valid_loss = [[] for i in range(anchor_num)]
    for epo in range(num_epoch):
        running_loss = []
        for i in range(anchor_num):
            running_loss.append([])
        for batch_data in tqdm(train_loader):
            batch = torch.tensor(batch_data['mover_value'].reshape(100, -1), dtype=torch.float32).to(device)
            label = torch.tensor(batch_data['anchor_offsets'], dtype=torch.float32).to(device)
            # initialize optimizers
            for opt in anchor_optimizers:
                opt.zero_grad()
            # predict x results
            pred = [ml(batch) for ml in anchor_models]
            loss = [anchor_criterion(pred[i], label[:,i,:]) for i in range(len(pred))]
            # backward
            for i in range(anchor_num):
                running_loss[i].append(loss[i].item())
                loss[i].backward()
                anchor_optimizers[i].step()

        # Print the average loss for this epoch
        print("Epoch {}".format(epo))
        print("Anchors mean losses: {}".format(np.mean([np.mean(running_loss[i]) for i in range(anchor_num)])))
        for i in range(anchor_num):
            train_loss[i].append(np.mean(running_loss[i]))
        # Get validation loss
        val_loss = evaluate_anchor(anchor_num, val_loader)
        for i in range(anchor_num):
            valid_loss[i].append(val_loss[i])
    print("Done!")
    # TODO: comment following if not want to output loss history
    plot_loss_history(train_loss[0], valid_loss[0], type='Anchor0_round{}'.format(fig_prefix))


def evaluate_anchor(anchor_num, loader): # Evaluate accuracy on validation / test set
    # Set the model to evaluation mode
    for i in range(anchor_num):
        anchor_models[i].eval()
    with torch.no_grad(): # Do not calculate grident to speed up computation
        running_loss = []
        for i in range(anchor_num):
            running_loss.append([])
        for batch_data in tqdm(loader):
            batch = torch.tensor(batch_data['mover_value'].reshape(100, -1), dtype=torch.float32).to(device)
            label = torch.tensor(batch_data['anchor_offsets'], dtype=torch.float32).to(device)
            # predict results
            pred = [ml(batch) for ml in anchor_models]
            loss = [anchor_criterion(pred[i], label[:,i,:]) for i in range(len(pred))]
            for i in range(anchor_num):
                running_loss[i].append(loss[i].item())
    eva_loss = [np.mean(running_loss[i]) for i in range(anchor_num)]
    print("Evaluation loss: {}".format(np.mean(eva_loss)))

    return eva_loss
    

def plot_loss_history(train_loss, valid_loss, type='X'):
    x_axis = list(range(len(train_loss)))
    plt.plot(x_axis, train_loss)
    plt.plot(x_axis, valid_loss)
    plt.legend(['train-loss', 'valid-loss'])
    plt.xlabel("epoch number")
    plt.ylabel("Loss")
    plt.title("Loss for train and validation of coordinate {}".format(type))
    plt.savefig(fig_dir + '/Loss_history_{}.png'.format(type))
    plt.close()


for i in range(ROUND):
    print("Round{}".format(i))
    # save the parameters after training
    train(models, train_loader, val_loader, num_epoch=num_epoch, fig_prefix=i)
    torch.save(obj=models['x'].state_dict(), f=param_save_path + 'model_states_x.pth')
    torch.save(obj=models['y'].state_dict(), f=param_save_path + 'model_states_y.pth')
    torch.save(obj=models['z'].state_dict(), f=param_save_path + 'model_states_z.pth')

    # # save the localoffset parameters after training
    # train_local(local_models, train_loader, val_loader, num_epoch=num_epoch, fig_prefix=i)
    # torch.save(obj=local_models['x'].state_dict(), f=param_save_path + 'local_states_x.pth')
    # torch.save(obj=local_models['y'].state_dict(), f=param_save_path + 'local_states_y.pth')
    # torch.save(obj=local_models['z'].state_dict(), f=param_save_path + 'local_states_z.pth')

    # # train the anchor points
    # train_anchor(anchor_models, ANCHOR_NUM, train_loader, val_loader, num_epoch=num_epoch, fig_prefix=i)
    # # save the anchor models
    # for i in range(ANCHOR_NUM):
    #     if not os.path.exists(param_save_path + '/anchor_models/model{}.pth'.format(i)):
    #         fd = open(param_save_path + '/anchor_models/model{}.pth'.format(i), 'w', encoding="utf-8")
    #         fd.close()
    #     torch.save(obj=anchor_models[i].state_dict(), f=param_save_path + '/anchor_models/model{}.pth'.format(i))
