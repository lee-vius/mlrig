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
param_save_path = "/Users/levius/Desktop/高级图像图形学/项目/code/ml/maya_split/mlmodel/model_param/model_states.pth"
LOAD = False
model = Network(540, 12942, 2, 2048).to(device)
if LOAD:
    # read in the model last time trained
    model.load_state_dict(torch.load(param_save_path))
# TODO: choose your loss function
# criterion = nn.CrossEntropyLoss()
criterion = nn.L1Loss()
# Debug
# print(model)
# TODO: adjust optimizer, learning rate, weight decay according to your need
optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-6)
# optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# TODO: choose an appropriate number of epoch
num_epoch = 10


def train(model, loader, num_epoch = 10): # Train the model
    print("Start training...")
    model.train() # Set the model to training mode
    train_loss = []
    # valid_loss = []
    for i in range(num_epoch):
        running_loss = []
        for batch_data in tqdm(loader):
            # print(i_batch)
            # print(batch_data['mover_value'].size())
            # print(batch_data['localOffset'].size())
            batch = torch.tensor(batch_data['mover_value'].reshape(100, -1), dtype=torch.float32).to(device)
            label = torch.tensor(batch_data['differentialOffset'], dtype=torch.float32).to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = criterion(pred, label[:, :, 0])
            running_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        print("Epoch {} loss:{}".format(i+1,np.mean(running_loss))) # Print the average loss for this epoch
        train_loss.append(np.mean(running_loss))
#         _, val_loss = evaluate(model, valloader)
#         valid_loss.append(val_loss)
    print("Done!")
    # TODO: If needed, you can use following lines to print loss history
    # x_axis = list(range(len(train_loss)))
    # plt.plot(x_axis, train_loss)
    # plt.plot(x_axis, valid_loss)
    # plt.legend(['train-loss', 'valid-loss'])
    # plt.xlabel("epoch number")
    # plt.ylabel("Loss")
    # plt.title('Loss for train and validation')
    # plt.savefig('Loss_history.png')


def evaluate(model, loader): # Evaluate accuracy on validation / test set
    model.eval() # Set the model to evaluation mode
    correct = 0
    with torch.no_grad(): # Do not calculate grident to speed up computation
        running_loss = []
        for batch, label in tqdm(loader):
            batch = batch.to(device)
            label = label.to(device)
            pred = model(batch)
            loss = criterion(pred, label) # Calculate the loss
            running_loss.append(loss)
            correct += (torch.argmax(pred, dim=1) == label).sum().item()
    acc = correct/len(loader.dataset)
    print("Evaluation accuracy: {}".format(acc))
    return acc, np.mean(running_loss)


# save the parameters after training
train(model, train_loader, num_epoch)
torch.save(obj=model.state_dict(), f=param_save_path)

# evaluate the model
# print("Evaluate on validation set...")
# evaluate(model, valloader)
# print("Evaluate on test set")
# evaluate(model, testloader)
