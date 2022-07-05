from sklearn.model_selection import train_test_split
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt


def load_and_split_dataset(file="dataset/processed_power_data_alu32.mat", validation_size=0.2, RAND_SEED=251289):
    '''
        Load switching activity/power dataset
        Split into train and validation set
    '''
    data = loadmat(file)
    features = data['features']
    X_train, X_validation, y_train, y_validation = train_test_split(data['X'], data['y'], test_size=validation_size, shuffle=True, random_state=RAND_SEED)

    return X_train, X_validation, y_train, y_validation, features


#Make sure y and y_hat have the EXACT same shape
def get_NRMSE(y, y_hat): 
    return 1/(np.max(y) - np.min(y)) * np.sqrt(np.sum((y - y_hat)**2)/len(y))


#TODO: Add more methodology as we increase
def bar_graph(module_names, dt, rf, ab_dt):
    
    barWidth = 0.25
    fig = plt.subplots(figsize =(12, 8))

    # set height of bar
    DT = dt
    RF = rf
    AB_DT = ab_dt

    # Set position of bar on X axis
    br1 = np.arange(len(DT))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    # Make the plot
    plt.bar(br1, DT, color ='r', width = barWidth,
            edgecolor ='grey', label ='Decision Tree')
    plt.bar(br2, RF, color ='g', width = barWidth,
            edgecolor ='grey', label ='Random Forest')
    plt.bar(br3, AB_DT, color ='b', width = barWidth,
            edgecolor ='grey', label ='AdaBoost Decision Tree')

    # Adding Xticks
    plt.xlabel('Module', fontweight ='bold', fontsize = 15)
    plt.ylabel('NRMSE', fontweight ='bold', fontsize = 15)
    plt.xticks([r + barWidth for r in range(len(DT))],
            module_names)

    plt.legend()
    plt.show()