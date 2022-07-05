from utils import *
from methods import *
from models import *
import numpy as np
import matplotlib.pyplot as plt


circuit_names = ["32-bit 2-input ALU", "64-bit 2-input ALU", "32-bit 2-input MUX", 'BCD to 7-seg decoder']
circuits = ["alu32", "alu64", "mux32", "7sg"]
best_neural_net = {
    "alu32": MLP_3H, 
    "alu64": MLP_1H_DROP, 
    "mux32": MLP_3H, 
    "7sg": MLP_3H
}
best_depth_boosted_dt = {
    "alu32": 7, 
    "alu64": 4, 
    "mux32": 5, 
    "7sg": 6
}

NRMSE = {
    "Linear Regression": [], 
    "Boosted Decision Tree": [], 
    "Random Forest": [], 
    "Neural Net": []
}

RAND_SEED = 251289

n_estimators = 100

epochs = 30
batch_size = 100
learning_rate = 1e-3

for circuit in circuits:
    X_train, X_validation, y_train, y_validation, features = load_and_split_dataset(file=f"dataset/processed_power_data_{circuit}.mat", 
                                                                                    validation_size=0.2,
                                                                                    RAND_SEED = RAND_SEED)

    nrmse_linear = linear_regression(X_train, y_train, X_validation, y_validation)
    NRMSE["Linear Regression"].append(nrmse_linear)

    nrmse_rf, _ = random_forest(X_train, y_train, X_validation, y_validation, n_estimators=n_estimators, RAND_SEED=RAND_SEED)
    NRMSE["Random Forest"].append(nrmse_rf)

    nrmse_boosted_dt, _ = adaboost_DT(X_train, y_train, X_validation, y_validation, bestdepth=best_depth_boosted_dt[circuit], n_estimators=n_estimators, RAND_SEED=RAND_SEED)
    NRMSE["Boosted Decision Tree"].append(nrmse_boosted_dt)

    nrmse_nn = neural_network(X_train, y_train, X_validation, y_validation, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, MODEL=best_neural_net[circuit])
    NRMSE["Neural Net"].append(nrmse_nn)

print(circuit_names)
print(NRMSE)

num_circuits = len(circuit_names)

# set width of bar
barWidth = 0.15
fig = plt.subplots(figsize=(12, 8))
 
# Set position of bar on X axis
br = []
for id, (model, error) in enumerate(NRMSE.items()):
    if id == 0:
        br.append(np.arange(num_circuits))
    else:
        br.append( [x + barWidth for x in br[id-1]] )

    # Make the plot
    plt.bar(br[id], error, width=barWidth, edgecolor='grey', label=model)

# Adding Xticks
plt.xlabel('Circuit', fontweight='bold', fontsize=15)
plt.ylabel('NRMSE', fontweight='bold', fontsize=15)
plt.xticks([r + 1.5*barWidth for r in range(num_circuits)], circuit_names)
 
plt.legend()
plt.savefig("images/final_comparison.png")
