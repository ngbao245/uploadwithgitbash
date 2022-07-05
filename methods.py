from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import numpy as np
import sklearn.tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
import torch
import torchvision
from torchvision.transforms import ToTensor, Lambda
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import tqdm

from utils import *


#Decision Trees
def tune_DT(X_train, y_train, X_validation, y_validation, circuit=None, RAND_SEED=251289):
    #Hyperparams
    depths = np.arange(1, 11)

    #Initialize
    bestNRMSE = np.inf
    best_y_pred = []
    bestdepth = -1
    training = []
    validation = []

    #Training and Tuning
    for depth in depths:
        regressor = DecisionTreeRegressor(random_state=RAND_SEED, max_depth=depth) 
        regressor.fit(X_train, y_train)

        #Construct the Predictions
        y_pred_train = regressor.predict(X_train)
        y_pred_train = np.reshape(y_pred_train, (len(y_pred_train), 1))
        train_NRMSE = get_NRMSE(y_train, y_pred_train)
        training.append(train_NRMSE)


        y_pred = regressor.predict(X_validation)
        y_pred = np.reshape(y_pred, (len(y_pred), 1))
        validation_NRMSE = get_NRMSE(y_validation, y_pred)
        validation.append(validation_NRMSE)


        if (validation_NRMSE < bestNRMSE):
            bestNRMSE = validation_NRMSE
            bestdepth = depth
            best_y_pred = y_pred
        #print("Iter Depth: ",  regressor.get_depth())

    plt.figure(figsize=(12, 8))
    plt.plot(depths, training, label="Training NRMSE")
    plt.plot(depths, validation, label="Validation NRMSE")
    plt.title("NRMSE vs Depth for Decision Tree Regressor")
    plt.legend()

    if circuit:
        plt.savefig(f"images/trees_and_ensemble_learning/NRMSE_vs_depth_{circuit}.png")
    else:
        plt.show()

    print("Best NRMSE: " + str(bestNRMSE))
    print("Best Depth: " + str(bestdepth))
    return bestNRMSE, bestdepth


#Visualizing Decision Trees
def visualize_DT(feature_name, X_train, y_train, depth, show_depth, circuit=None, RAND_SEED=251289):
    
    assert show_depth <= depth, "show_depth has to be less than equal to depth"
    
    regressor = DecisionTreeRegressor(random_state = RAND_SEED, max_depth = depth) 
    regressor.fit(X_train, y_train)
    
    text_representation = sklearn.tree.export_text(regressor, feature_names = feature_name)
    fig = plt.figure(figsize=(25,20))
    _ = sklearn.tree.plot_tree(regressor, max_depth = show_depth, feature_names = feature_name)

    if circuit:
        plt.savefig(f"images/trees_and_ensemble_learning/tree_visual_{circuit}.png")
    else:
        plt.show()

    return text_representation
    

#Random Forest
def random_forest(X_train, y_train, X_validation, y_validation, n_estimators = 100, RAND_SEED = 251289):
    # m = d/3 - taking m features out of d is the best for regression random forest
    regressor = RandomForestRegressor(n_estimators = n_estimators, random_state = RAND_SEED, max_features = 0.3) 
    regressor.fit(X_train, y_train.ravel())

    #Construct the Predictions
    y_pred = regressor.predict(X_validation)
    y_pred = np.reshape(y_pred, (len(y_pred), 1))

    NRMSE = get_NRMSE(y_validation, y_pred)
    firstdepth = regressor.estimators_[0].get_depth()

    print("Depth: " + str(firstdepth))
    print("NRMSE: " + str(NRMSE))
    
    return NRMSE, firstdepth


#AdaBoost
def adaboost_DT(X_train, y_train, X_validation, y_validation, bestdepth, n_estimators = 100, RAND_SEED = 251289):
    regressor = AdaBoostRegressor(
    DecisionTreeRegressor(random_state = RAND_SEED, max_depth = bestdepth),
    n_estimators = n_estimators,
    random_state = RAND_SEED
    ) 
    regressor.fit(X_train, y_train.ravel())

    #Construct the Predictions
    y_pred = regressor.predict(X_validation)
    y_pred = np.reshape(y_pred, (len(y_pred), 1))
    NRMSE = get_NRMSE(y_validation, y_pred)

    print("Depth: " + str(bestdepth))            
    print("NRMSE: " + str(NRMSE))
    
    return NRMSE, bestdepth


def linear_regression(X_train, y_train, X_validation, y_validation):
    reg = LinearRegression().fit(X_train, y_train)
    y_hat_validation = reg.predict(X_validation)
    NRMSE = get_NRMSE(y=y_validation, y_hat=y_hat_validation)

    return NRMSE


def neural_network(X_train, y_train, X_validation, y_validation, epochs, batch_size, learning_rate, MODEL):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MODEL(input_size=X_train.shape[1]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    X_train_tensor, X_validation_tensor, y_train_tensor, y_validation_tensor = torch.Tensor(X_train), torch.Tensor(X_validation), torch.Tensor(y_train), torch.Tensor(y_validation)
    train_set = TensorDataset(X_train_tensor, y_train_tensor)
    validation_set = TensorDataset(X_validation_tensor, y_validation_tensor)
    dataloader_train = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    dataloader_validation = DataLoader(validation_set, batch_size=batch_size, shuffle=True)

    model.train() # Put model in training mode

    for epoch in range(epochs):
        # Training
        per_batch_training_loss = []
        per_batch_training_NRMSE = []
        for x, y in tqdm.tqdm(dataloader_train, unit="batch"):
            # bring samples and labels to device
            x, y = x.to(device), y.to(device)

            # perform forward pass: make predictions and compute loss
            pred = model(x)
            loss = criterion(pred, y)

            # remove gradients from previous step, perform backward pass, and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # compute the NRMSE
            per_batch_training_loss.append(loss.item())
            per_batch_training_NRMSE.append( get_NRMSE(pred.detach().clone().cpu().numpy(), y.detach().clone().cpu().numpy()) )

        per_epoch_training_loss = np.mean(per_batch_training_loss)
        per_epoch_training_NMRSE = np.mean(per_batch_training_NRMSE)
        print(f"Finished epoch {epoch + 1}, Training Loss: {per_epoch_training_loss}, Training NMRSE: {per_epoch_training_NMRSE}")

        # Validation
        with torch.no_grad():
            model.eval() # Put model in eval mode
            per_batch_validation_losses = []
            per_batch_validation_NRMSE = []
            for x, y in dataloader_validation:
                # bring samples and labels to device
                x, y = x.to(device), y.to(device)

                # perform forward pass: make predictions and compute loss
                pred = model(x)
                loss = criterion(pred, y)

                # compute the NRMSE
                per_batch_validation_losses.append(loss.item())
                per_batch_validation_NRMSE.append( get_NRMSE(pred.detach().clone().cpu().numpy(), y.detach().clone().cpu().numpy()) )

            per_epoch_validation_loss = np.mean(per_batch_validation_losses)
            per_epoch_validation_NMRSE = np.mean(per_batch_validation_NRMSE)
            print(f"Validation Loss: {per_epoch_validation_loss}, Validation NMRSE: {per_epoch_validation_NMRSE}")
            model.train() # Put model back in train mode

    ### FINAL TRAINING AND VALIDATION ACCURACIES ###
    with torch.no_grad():
        model.eval() # Put model in eval mode
        per_batch_training_NRMSE = []
        for x, y in dataloader_train:
            # bring samples and labels to device
            x, y = x.to(device), y.to(device)

            # perform forward pass: make predictions and compute loss
            pred = model(x)

            # compute the NRMSE
            per_batch_training_NRMSE.append( get_NRMSE(pred.detach().clone().cpu().numpy(), y.detach().clone().cpu().numpy()) )

        per_model_train_NRMSE = np.mean(per_batch_training_NRMSE)
        print("Final Training NMRSE:", per_model_train_NRMSE)
        model.train() # Put model back in train mode

    with torch.no_grad():
        model.eval() # Put model in eval mode
        per_batch_validation_NRMSE = []
        for x, y in dataloader_validation:
            # bring samples and labels to device
            x, y = x.to(device), y.to(device)

            # perform forward pass: make predictions and compute loss
            pred = model(x)

            # compute the NRMSE
            per_batch_validation_NRMSE.append( get_NRMSE(pred.detach().clone().cpu().numpy(), y.detach().clone().cpu().numpy()) )

        per_model_validation_NRMSE = np.mean(per_batch_validation_NRMSE)
        print("Final Validation NMRSE:", per_model_validation_NRMSE)
        model.train() # Put model back in train mode

    return per_model_validation_NRMSE