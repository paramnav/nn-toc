
import torch
from pathlib import Path
import pickle
import numpy as np
from matplotlib import pyplot as plt
#from sklearn.model_selection import StratifiedKFold, KFold
#from sklearn.model_selection import train_test_split
import os
from torch.utils.data import Dataset, TensorDataset, DataLoader

from os import walk  

import torch.nn.functional as F

training_accuracy = []
training_losses = []
training_mae = []
learning_rate =  [] 

evaluation_losses = []
evaluation_mae = []
true_labels = []
pred_labels = []

##########################################################################
#build_dnn()
"""
We present here an autoencoder structure here for the DNN.
The decoder part is frozen and is not trained, but can be used for unsupervised learning for future.
than the supervised approach. Though here, we show the entire autoencoder architecture, we would be using only the encoder and the supervised part for training and predictions. But the entire architecture provides flexibility for other use cases where the amount of data available is low and requires unsupervised approach.
"""

def initialize_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0)
    
def buildDNN(layer_widths, features):
    class DNN(torch.nn.Module):
        def __init__(self, layer_width):
            super(DNN, self).__init__()

            self.do_prob = 0.2
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(features.shape[1], layer_width[0]),
                torch.nn.BatchNorm1d(layer_width[0]),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.do_prob),
                torch.nn.Linear(layer_width[0], layer_width[1]),
                torch.nn.BatchNorm1d(layer_width[1]),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.do_prob),
                torch.nn.Linear(layer_width[1], layer_width[2]),
                torch.nn.BatchNorm1d(layer_width[2]),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.do_prob),
                torch.nn.Linear(layer_width[2], layer_width[3]),
                torch.nn.BatchNorm1d(layer_width[3]),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.do_prob),
                torch.nn.Linear(layer_width[3], layer_width[4]),
                torch.nn.BatchNorm1d(layer_width[4]),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.do_prob),
                torch.nn.Linear(layer_width[4], layer_width[5]),
                torch.nn.BatchNorm1d(layer_width[5]),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.do_prob),
                torch.nn.Linear(layer_width[5], layer_width[6]),
                torch.nn.BatchNorm1d(layer_width[6]),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.do_prob),
            )

            self.decoder = torch.nn.Sequential(

                torch.nn.Linear(layer_width[6], layer_width[5]),
                #torch.nn.BatchNorm1d(layer_width[5]),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.do_prob),
                torch.nn.Linear(layer_width[5], layer_width[4]),
                #torch.nn.BatchNorm1d(layer_width[4]),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.do_prob),
                torch.nn.Linear(layer_width[4], layer_width[3]),
                #torch.nn.BatchNorm1d(layer_width[3]),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.do_prob),
                torch.nn.Linear(layer_width[3], layer_width[2]),
                #torch.nn.BatchNorm1d(layer_width[2]),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.do_prob),
                torch.nn.Linear(layer_width[2], layer_width[1]),
                #torch.nn.BatchNorm1d(layer_width[1]),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.do_prob),
                torch.nn.Linear(layer_width[1], layer_width[0]),
                #torch.nn.BatchNorm1d(layer_width[0]),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.do_prob),
                torch.nn.Linear(layer_width[0], features.shape[1])
            )

            self.supervised = torch.nn.Sequential(
                torch.nn.Linear(layer_width[6], layer_width[7]),
                torch.nn.BatchNorm1d(layer_width[7]),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.do_prob),
                torch.nn.Linear(layer_width[7], layer_width[8]),
                torch.nn.BatchNorm1d(layer_width[8]),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.do_prob),
                torch.nn.Linear(layer_width[8], layer_width[9]),
                torch.nn.BatchNorm1d(layer_width[9]),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.do_prob),
                torch.nn.Linear(layer_width[9], 1),

            )

        def forward(self, x):
            encoded = self.encoder(x)
            targets = self.supervised(encoded)
            return targets
        
    def initialize_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0)
            
    layer_widths = [512, 512, 512, 512, 512, 512, 512, 512, 512, 512]  
    model = DNN(layer_widths)
    model.apply(initialize_weights)
    model.double()
    model = DNN(layer_widths)
    model = model.double()
    model.apply(initialize_weights)

    return model


def train(epoch, model, trainloader, optimizer, lr_scheduler, loss_fn, change_lr):
    model.train()   
    running_loss = 0
    counter = 0
    mae_epoch = 0
    running_vall_loss = 0.0 

    
    for features, labels in trainloader:
        if torch.cuda.is_available():
            features = features.cuda()
            labels = labels.cuda()
        
        features = features.to(dtype = torch.double)

        labels   = labels.to(dtype = torch.double) 

        optimizer.zero_grad()
        

        targets = model(features) #predictions

        labels = torch.reshape(labels, (labels.shape[0],1))
        output = torch.reshape(targets, (targets.shape[0],1))
        loss = loss_fn(output, labels)
        mae_epoch+=torch.mean(torch.abs(targets- labels))

        #model learns by back prop
        loss.backward()
        
        #optimize weights
        optimizer.step()
        if change_lr == True:
            lr_scheduler.step(loss)
        
        running_loss += loss.item()


        lr = optimizer.param_groups[0]['lr']
        counter+=1

    training_loss = running_loss/len(trainloader)
    
    mae = mae_epoch/counter
    learning_rate.append(lr)
    training_mae.append(mae.cpu().detach().numpy())
    training_losses.append(training_loss)

    print('epoch: %d | Training Loss: %.3f | MAE: %.3f | LR: %.7f |'%(epoch, training_loss,   mae, lr))

def validation(epoch, model, testloader, loss_fn, store_labels=False):
    model.eval() #similar to model.train(), model.eval() tells that you are testing. 
    running_loss = 0

    counter = 0
    mae_epoch = 0

    for features, labels in testloader:
        if torch.cuda.is_available():
            features = features.cuda()
            labels = labels.cuda()
        
        features = features.to(dtype = torch.double)
        labels   = labels.to(dtype = torch.double)   
 
        targets = model(features) #predictions

        #if train_step == 2:
        labels = torch.reshape(labels, (labels.shape[0],1))
        output = torch.reshape(targets, (targets.shape[0],1))
        loss = loss_fn(output, labels)
        mae_epoch+=torch.mean(torch.abs(targets- labels))

        
        running_loss += loss.item()
        counter+=1


    testing_loss = running_loss/len(testloader)

    mae = mae_epoch/counter

    evaluation_losses.append(testing_loss)
    evaluation_mae.append(mae.cpu().detach().numpy())
    print('epoch: %d | Testing Loss: %.3f | MAE: %.3f  |'%(epoch, testing_loss, mae))

def get_model_performance_metrics():
    return training_accuracy,  
    training_losses, 
    training_mae,
    learning_rate,

    evaluation_losses,
    evaluation_mae,
    true_labels,
    pred_labels

########################################################################

def preprocess_features_labels(features, labels, X_mean, X_std):
    """
    Preprocesses features and labels.

    Parameters:
    features (numpy.ndarray): Input features.
    labels (numpy.ndarray): Input labels.
    X_mean (numpy.ndarray): Mean values for normalization.
    X_std (numpy.ndarray): Standard deviation values for normalization.

    Returns:
    torch Tensor: Preprocessed features.
    torch Tensor: Preprocessed labels.
    """

    # Save rows with NaN values
    nan_rows = np.isnan(features).any(axis=1)
    np.save("FeatureNanRows", nan_rows)
    features = features[~nan_rows, :]
    labels = labels[~nan_rows]

    # Drop features with NaN values
    nan_rows = np.isnan(features).any(axis=0)
    np.save("FeatureNanRows", nan_rows)
    features = features[:, ~nan_rows]

    # Normalize features
    features = np.divide((features - X_mean), X_std)

    # Replace duplicate rows with NaN values and adjust labels
    arr = features
    unique_rows, row_counts = np.unique(arr, axis=0, return_counts=True)
    duplicate_indices = np.where(row_counts > 1)[0]
    groups = np.split(unique_rows, np.where(np.diff(unique_rows, axis=0).any(axis=1))[0] + 1)

    for idx in duplicate_indices:
        group_idx = [i for i, group in enumerate(groups) if (group == unique_rows[idx]).all()][0]
        group_labels = labels[np.where((arr == unique_rows[idx]).all(axis=1))[0]]
        # print(group_labels)
        # print(group_idx)
        # print(unique_rows[idx])
        if np.std(group_labels) > 0.2 * np.max(group_labels):
            row_idx = np.where((arr == unique_rows[idx]).all(axis=1))[0]
            arr[row_idx,:] = np.nan
            labels[row_idx] = np.nan
            print(f"Duplicate row with index {row_idx} was in group {group_idx} and was replaced with NaN values")

        else:
            row_idx = np.where((arr == unique_rows[idx]).all(axis=1))[0]
            arr[row_idx[1:], :] = np.nan  # remove duplicate rows except the first occurrence

            for z in range(len(row_idx)):
                if z == 0:
                    labels[row_idx[z]] = np.mean(group_labels)  # replace the label of the first occu
                else:
                    labels[row_idx[z]] = np.nan

            # print(row_idx)

    # remove rows with NaN values
    num_nan_values_before = np.sum(np.isnan(arr), axis=1)
    arr = arr[~np.isnan(arr).any(axis=1)]
    num_nan_values_after = np.sum(np.isnan(arr), axis=1)
    labels = labels[~np.isnan(labels)]
    features = arr

    print(f"Number of rows in features before: {len(num_nan_values_before)}")
    print(f"Number of rows in features after: {len(num_nan_values_after)}")
    print(f"Number of rows in labels: {len(labels)}")

    #features = torch.tensor(features)
    #labels = torch.tensor(labels)

    return features, labels


def create_train_test_loader(features, labels, batch_size = 200, test_size_ratio=1/7, shuffle=True):
    """
    Creates train and test loaders.

    Parameters:
    features (numpy.ndarray): Input features.
    labels (numpy.ndarray): Input labels.
    batch_size (int): Batch size for DataLoader.
    test_size_ratio (float): Ratio of test set size to total dataset size (default: 0.2).
    shuffle (bool): Whether to shuffle the dataset before creating loaders (default: True).

    Returns:
    torch.utils.data.DataLoader: Train loader.
    torch.utils.data.DataLoader: Test loader.
    """

    dataset=TensorDataset(features, labels)
    dataloader=DataLoader(dataset, batch_size=batch_size)



    testsize=int(features.shape[0]*test_size_ratio)
    trainsize=features.shape[0]-testsize


    train_set, test_set = torch.utils.data.random_split(dataset, [trainsize, testsize], generator=torch.Generator().manual_seed(42))



    trainloader= torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    testloader= torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

    num_channels = features.shape[1]

    #####################################################summary of training set
    dataiter = iter(trainloader)
    features, labels = next(dataiter)
    
    print("Train data size")
    print(trainsize)
    print("Test data size")
    print(testsize)
    ######################################################


    return trainloader, testloader

def train_model(model, loss_fn, trainloader, testloader, num_epochs=500):
    """
    Train the model.

    Parameters:
    model (torch.nn.Module): The model to be trained.
    loss_fn: Loss function used for training.
    optimizer: Optimizer used for training.
    train_loader: DataLoader for training data.
    test_loader: DataLoader for test data.
    train_step (int): Training step (default: 2).
    num_epochs (int): Number of epochs for training (default: 500).
    """

    # Freeze the parameters of the decoder
    for param in model.decoder.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, 
                                                          patience=50, threshold=0.0001, threshold_mode='rel', 
                                                          cooldown=0, min_lr=0.00001, eps=1e-08, verbose=False)
    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        loss_fn = loss_fn.cuda()

    # Training loop
    for epoch in range(num_epochs):
        change_LR = True  # Set to True to change learning rate

        # Train for one epoch
        train(epoch, model, trainloader, optimizer, lr_scheduler, loss_fn,  change_LR)

        # Validate the model
        validation(epoch, model, testloader, loss_fn, store_labels=True)
        print()
    return training_accuracy, training_losses, training_mae, learning_rate, evaluation_losses, evaluation_mae, true_labels, pred_labels



