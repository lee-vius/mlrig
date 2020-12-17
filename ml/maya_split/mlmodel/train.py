from mlmodel import Network
from data_handler import DeformData
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
device = "cuda" if torch.cuda.is_available() else "cpu" # Configure device
# device = "cpu"
param_save_path = "/Users/levius/Desktop/高级图像图形学/项目/code/ml/maya_split/mlmodel/model_param/"
LOAD = False

# create three traning models for x, y, z coordinates
models = {}
models['x'] = Network(540, 12942, 2, 2048).to(device)
models['y'] = Network(540, 12942, 2, 2048).to(device)
models['z'] = Network(540, 12942, 2, 2048).to(device)

if LOAD:
    # read in the model last time trained
    models['x'].load_state_dict(torch.load(param_save_path + 'model_states_x.pth'))
    models['y'].load_state_dict(torch.load(param_save_path + 'model_states_y.pth'))
    models['z'].load_state_dict(torch.load(param_save_path + 'model_states_z.pth'))

# TODO: choose your loss function
criterion = nn.L1Loss()

# TODO: adjust optimizer, learning rate, weight decay according to your need
optimizers = {}
optimizers['x'] = optim.SGD(models['x'].parameters(), lr=0.1, weight_decay=1e-6)
optimizers['y'] = optim.SGD(models['y'].parameters(), lr=0.1, weight_decay=1e-6)
optimizers['z'] = optim.SGD(models['z'].parameters(), lr=0.1, weight_decay=1e-6)

# TODO: choose an appropriate number of epoch
num_epoch = 10


def train(models, train_loader, val_loader, num_epoch = 10): # Train the model
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
    plot_loss_history(train_loss['x'], valid_loss['x'], type='X')
    plot_loss_history(train_loss['y'], valid_loss['y'], type='Y')
    plot_loss_history(train_loss['z'], valid_loss['z'], type='Z')


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

# save the parameters after training
train(models, train_loader, val_loader, num_epoch=num_epoch)
torch.save(obj=models['x'].state_dict(), f=param_save_path + 'model_states_x.pth')
torch.save(obj=models['y'].state_dict(), f=param_save_path + 'model_states_y.pth')
torch.save(obj=models['z'].state_dict(), f=param_save_path + 'model_states_z.pth')

# evaluate the model
# print("Evaluate on validation set...")
# evaluate(model, valloader)
# print("Evaluate on test set")
# evaluate(model, testloader)
