import os
import numpy as np
import pandas as pd
from scipy.io import savemat
import sys
import argparse


OP2DEC = {
    "ALU_ADD"   : 0 ,
    "ALU_SUB"   : 1 ,
    "ALU_AND"   : 2 ,
    "ALU_OR"    : 3 ,
    "ALU_XOR"   : 4 ,
    "ALU_SLT"   : 5 ,
    "ALU_SLTU"  : 6 ,
    "ALU_SLL"   : 7 ,
    "ALU_SRA"   : 8 ,
    "ALU_SRL"   : 9 ,
    "ALU_COPY_B": 10,
    "ALU_XXX"   : 15
}

def dec2bcd(dec, width):
    '''
        Convert width-bit decimal to string of BCD values
        Args:
            dec (int): original decimal value
            width (int): the number of bits that can represent the decimal dec
        Returns:
            padded_bcd (str): BCD values
    '''
    max_dec = 2**width-1
    num_digits = len(list(str(max_dec)))
    total_num_digits = 4 * num_digits
    unpadded_bcd = ''.join([np.binary_repr(int(d), 4) for d in list(str(dec))])
    padded_bcd = '0' * (total_num_digits - len(unpadded_bcd)) + unpadded_bcd
    return padded_bcd

parser = argparse.ArgumentParser(description='Preprocess Collected Training Data')
parser.add_argument('--circuit', type=str, default='alu32', help='Specify the RTL module to collect training data on (alu32, alu64, etc.')
args = vars(parser.parse_args())


def preprocess(raw_csv, width=32, opwidth=4):
    '''
        Read the CSV generated during data collection (in collect_training_data.py)
        Return:
            data (dict): training/validation/test data, contains
                X (np.ndarray): samples (switching activities on inputs and opcodes)
                y (np.ndarray): labels (RTL-level simulation switching power from Joules)
                features (list): operands_A, operands_B, opcode
    '''
    df = pd.read_csv(raw_csv)

    y = df["switching_power"].to_numpy().reshape(-1, 1)

    df_X = df.drop(["switching_power"], axis=1)
    features = list(df_X.columns)

    if circuit_name != 'alu':
        if width == 32:
            X_raw = df_X.to_numpy(dtype='uint32')
        elif width == 64:
            X_raw = df_X.to_numpy(dtype='uint64')
    else:
        X_raw = df_X.to_numpy()

    # Convert to binary
    X_bin = []
    for idx, feature in enumerate(features):
        if feature in ["opcode", "sel"]:
            if feature == "opcode":
                x_temp = np.array([OP2DEC[x] for x in X_raw[:, idx]]).astype(int)
                X_bin.append( np.array(list(map(np.binary_repr, x_temp, [opwidth]*X_raw.shape[0]))) )
            elif feature == "sel":
                X_bin.append( np.array(list(map(np.binary_repr, X_raw[:, idx].astype(int), [opwidth]*X_raw.shape[0]))) )
        else:
            if circuit_name != "7sg":
                if width in [2, 4, 8, 16, 32, 64]:
                    if width < 8:
                        uwidth = 8
                    else:
                        uwidth = width
                    X_bin.append( np.array(list(map(np.binary_repr, X_raw[:, idx].astype('uint{}'.format(uwidth)), [width]*X_raw.shape[0]))) )
                else:
                    raise Exception("Width has to be a power of 2 and of range (0, 64]")
            else:
                X_bin.append( np.array(list(map(dec2bcd, X_raw[:, idx].astype('uint32'), [width]*X_raw.shape[0]))) )

    X_bin = np.array(X_bin).T

    # One-hot encode each bit in features
    if circuit_name != '7sg':
        X = np.zeros((X_bin.shape[0], 2*width+opwidth))
        X[:, 0:width]       = np.vstack([np.array(list(x)).astype(int) for x in X_bin[:, 0]])
        X[:, width:2*width] = np.vstack([np.array(list(x)).astype(int) for x in X_bin[:, 1]])
        X[:, 2*width:]      = np.vstack([np.array(list(x)).astype(int) for x in X_bin[:, 2]])
    else:
        # X = np.zeros((X_bin.shape[0], 40))
        X = np.vstack([np.array(list(x)).astype(int) for x in X_bin[:, 0]])

    data = {
        "X": X,
        "y": y,
        "features": features
    }

    return data


if __name__ == "__main__":
    # Check if requirements are met:
    if sys.version_info[0] != 3:
        raise Exception("Does not support Python 2. Please use Python 3")

    circuit = args['circuit'].lower()

    circuit_name = circuit[:3]
    if circuit_name == 'alu':
        opwidth = 4 # number of bits for ALU opcode
        width = int(circuit[3:]) # number of bits for ALU input A and B
    elif circuit_name == 'mux':
        opwidth = 1 # number of bits for MUX sel
        width = int(circuit[3:]) # number of bits for MUX input A and B
    elif circuit_name == '7sg':
        width = 32
        opwidth = None

    raw_power_data_path = "dataset/raw_power_data_{}.csv".format(circuit)
    processed_power_data_path = "dataset/processed_power_data_{}.mat".format(circuit)

    # Preprocess power data
    data = preprocess(raw_csv=raw_power_data_path, width=width, opwidth=opwidth)

    # Save .mat file contain the dataset ready for training/validation/testing. One can read the .mat file using scipy.io.loadmat(<file_name>)
    savemat(file_name=processed_power_data_path, mdict=data)
