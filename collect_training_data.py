#!/usr/bin/python3

# Desc: Automatically generate Verilog testbenches for ALU, MUX, or BCD to 7-segment converter circuits, 
# then run RTL simulations and collect training samples and labels.

# Author: Hoang Nguyen <hoang_nguyen@berkeley.edu>
#         Yuki Ito     <itoyuki@berkeley.edu>


import os
import subprocess
import random
from string import Template
import time
import pandas as pd
import sys
import argparse


parser = argparse.ArgumentParser(description='Collect Training Data')
parser.add_argument('--circuit', type=str, default='alu32', help='Specify the RTL module to collect training data on (alu32, alu64, mux32, 7sg, etc.)')
parser.add_argument('--num_samples', type=int, default=1000, help='Specify the number of samples to collect')
args = vars(parser.parse_args())

START_TIME = time.time()

ALUOP = {
    0 : "ALU_ADD"   ,
    1 : "ALU_SUB"   ,
    2 : "ALU_AND"   ,
    3 : "ALU_OR"    ,
    4 : "ALU_XOR"   ,
    5 : "ALU_SLT"   ,
    6 : "ALU_SLTU"  ,
    7 : "ALU_SLL"   ,
    8 : "ALU_SRA"   ,
    9 : "ALU_SRL"   ,
    10: "ALU_COPY_B",
    15: "ALU_XXX"
}

PROJ_PATH = os.getcwd()
RTL_PATH  = os.path.join(PROJ_PATH, 'src/MODULE.v')
TB_PATH   = os.path.join(PROJ_PATH, 'src/MODULE_tb.v')
LIB_PATH  = os.path.join(PROJ_PATH, 'src/ALUop.vh')


def generate_test_vector(width=32, opwidth=4, circuit='alu'):
    '''
        Randomly generate the 32-bit ALU input A, B and 4-bit ALU opcode
        opcode's generator avoids value 11, 12, 13, 14 (undefined ALU operations)
        Args:
            width (int): number of bits for ALU input A and B
            opwidth (int): number of bits for ALU opcode
        Returns:
            A (int): randomly generated value for ALU input A
            B (int): randomly generated value for ALU input B
            ALUOP[op] (str): randomly generated ALU opcode
    '''
    A  = random.randint(0, 2**width-1)
    B  = random.randint(0, 2**width-1)

    op = random.randint(0, 2**opwidth-1)
    while op in list(range(11, 15)):
        op = random.randint(0, 2**opwidth-1)

    if (circuit == 'alu'):
        return A, B, ALUOP[op]
    elif (circuit == 'mux'):
        return A, B, op
    elif (circuit == '7sg'):
        return A


def generate_alu_mux_rtl(rtl_tmpl_path='templates/alu_template.txt', width=32):
    '''
        Write an ALU/MUX module based on the width
        The generated files are located in the src/ directory
        Args:
            rtl_tmpl_path (str): path to template file for ALU/MUX module
            width (int): number of bits for ALU/MUX input A and B
    '''
    data = {
        "width": width,
        "lib_path": LIB_PATH
    }

    with open(rtl_tmpl_path, 'r') as tmpl_file:
        template = Template(tmpl_file.read())
    content = template.substitute(data)

    with open(RTL_PATH, 'w') as tb_file:
        tb_file.write(content)


def generate_alu_mux_tb(tb_tmpl_path='templates/alu_tb_template.txt', width=32, opwidth=4, circuit='alu'):
    '''
        Write an ALU/MUX testbench based on the width, randomly generated input and opcode values
        The generated files are located in the src/ directory
        Args:
            tb_tmpl_path (str): path to template file for ALU/MUX testbench
            width (int): number of bits for ALU/MUX input A and B
            opwidth (int): number of bits for ALU/MUX opcode
    '''
    A, B, opcode = generate_test_vector(width=width, opwidth=opwidth, circuit=circuit)

    data = {
        "width": width,
        "A": A,
        "B": B,
        "opcode": opcode,
        "lib_path": LIB_PATH
    }

    with open(tb_tmpl_path, 'r') as tmpl_file:
        template = Template(tmpl_file.read())
    content = template.substitute(data)

    with open(TB_PATH, 'w') as tb_file:
        tb_file.write(content)

    return A, B, opcode


def generate_7segment_rtl(rtl_tmpl_path='templates/seg7_template.txt'):
    '''
        Write an BCD to 7-segment converter module
        The generated files are located in the src/ directory
        Args:
            rtl_tmpl_path (str): path to template file for BCD to 7-segment converter module
    '''
    data = {}

    with open(rtl_tmpl_path, 'r') as tmpl_file:
        template = Template(tmpl_file.read())
    content = template.substitute(data)

    with open(RTL_PATH, 'w') as tb_file:
        tb_file.write(content)


def generate_7segment_tb(tb_tmpl_path='templates/seg7_tb_template.txt', width=32):
    '''
        Write an BCD to 7-segment converter testbench based on the width and randomly generated decimal input
        The generated files are located in the src/ directory
        Args:
            tb_tmpl_path (str): path to template file for BCD to 7-segment converter testbench
            width (int): number of bits of the integer to convert to 7-segment
    '''
    A = generate_test_vector(width=width, opwidth=1, circuit='7sg')

    # Make a list of all digits in the generated integer and pad 0 in front if less than 10 digits (max number of digit for a 32-bit integer) 
    in_list = list(str(A))
    in_list = ['0']*(10-len(in_list)) + in_list

    data = {
        "in0": in_list[0],
        "in1": in_list[1],
        "in2": in_list[2],
        "in3": in_list[3],
        "in4": in_list[4],
        "in5": in_list[5],
        "in6": in_list[6],
        "in7": in_list[7],
        "in8": in_list[8],
        "in9": in_list[9]
    }

    with open(tb_tmpl_path, 'r') as tmpl_file:
        template = Template(tmpl_file.read())
    content = template.substitute(data)

    with open(TB_PATH, 'w') as tb_file:
        tb_file.write(content)

    return A


def run_sim():
    '''
        Invoke Hammer and run RTL simulation for the generated testbench
        At the same time convert vpd format to vcd format to be consumed by Joules
    '''
    # Remove the cache file to bypass the "The design hasn't changed and need not be recompiled." Error
    try:
        os.remove('build/sim/sim-rundir/simv.daidir/.vcs.timestamp')
    except OSError:
        pass

    cmd = "make vpd2vcd"
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print('RUNNING SIMULATION')
    # print(output)
    # print(error)


def run_joules():
    '''
        Invoke Joules and generate power and activity report
    '''
    cmd = "make joules"
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print('RUNNING JOULES')
    # print(output)
    # print(error)


def extract_label(power_report="build/joules/power_report.time_based.csv"):
    '''
        Scrape the power report generated by Joules
        Label: total switching power (in build/joules/power_report.time_based.txt)
        Features: switching activity of inputs and opcodes
    '''
    df = pd.read_csv(power_report).iloc[8][0]
    df = str(df)
    contents = df.split(" ")
    contents = list(filter(None, contents))
    assert contents[0] == 'logic'

    return contents[3]
    

def initialize_csv(datafile, circuit='alu'):
    if circuit == 'alu':
        df = pd.DataFrame({"operands_A": [], "operands_B": [], "opcode": [], "switching_power": []})
    elif circuit == 'mux':
        df = pd.DataFrame({"operands_A": [], "operands_B": [], "sel": [], "switching_power": []})
    elif circuit == '7sg':
        df = pd.DataFrame({"input": [], "switching_power": []})
    df.to_csv(datafile, index=False)


def save_to_csv(datafile, data, circuit='alu'):
    '''
        Save/Append the samples and labels to a CSV
    '''
    if circuit == 'alu':
        df2 = pd.DataFrame({
            "operands_A": [data["operands_A"]],
            "operands_B": [data["operands_B"]],
            "opcode": [data["opcode"]],
            "switching_power": [data["switching_power"]]
        })
    elif circuit == 'mux':
        df2 = pd.DataFrame({
            "operands_A": [data["operands_A"]],
            "operands_B": [data["operands_B"]],
            "sel": [data["opcode"]],
            "switching_power": [data["switching_power"]]
        })
    elif circuit == '7sg':
        df2 = pd.DataFrame({
            "input": [data["operands_A"]],
            "switching_power": [data["switching_power"]]
        })

    df = pd.read_csv(datafile)
    df = pd.concat([df, df2], ignore_index=True, axis=0)
    df.to_csv(datafile, index=False)


if __name__ == '__main__':
    # Check if requirements are met:
    if sys.version_info[0] != 3:
        raise Exception("Does not support Python 2. Please use Python 3")

    circuit = args['circuit'].lower()
    num_tests = args['num_samples']

    circuit_name = circuit[:3]
    if circuit_name == 'alu':
        width = int(circuit[3:]) # number of bits for ALU / MUX input A and B
        opwidth = 4 # number of bits for ALU opcode
    elif circuit_name == 'mux':
        width = int(circuit[3:])
        opwidth = 1 # width of sel
    elif circuit_name == '7sg':
        width = 32
        opwidth = None

    # Set up random seed
    random.seed(os.urandom(width))

    # Initialize csv file that contains all the data if it does not already exist
    if not os.path.exists("dataset/"):
        os.makedirs("dataset/")

    csv_file_path = "dataset/raw_power_data_{}.csv".format(circuit)
    if not os.path.exists(csv_file_path):
        initialize_csv(datafile=csv_file_path, circuit=circuit_name)

    # Generate RTL module
    if circuit_name in ['alu', 'mux']:
        generate_alu_mux_rtl(rtl_tmpl_path='templates/{}_template.txt'.format(circuit_name), width=width)
    elif circuit_name == '7sg':
        generate_7segment_rtl(rtl_tmpl_path='templates/seg7_template.txt'.format(circuit_name))

    # Collect sample point: Generate testbench, run simulation/joules, and extract and save data
    for test_id in range(num_tests):
        print("Sample #{}".format(test_id + 1))

        if circuit_name in ['alu', 'mux']:
            operands_A, operands_B, opcode = generate_alu_mux_tb(tb_tmpl_path='templates/{}_tb_template.txt'.format(circuit_name), width=width, opwidth=opwidth, circuit=circuit_name)
        elif circuit_name == '7sg':
            operands_A = generate_7segment_tb(tb_tmpl_path='templates/seg7_tb_template.txt', width=width)
            operands_B = None
            opcode = None

        run_sim()
        run_joules()

        switching_power = extract_label()

        csv_data = {
            "operands_A": operands_A,
            "operands_B": operands_B,
            "opcode": opcode,
            "switching_power": switching_power
        }
        save_to_csv(datafile=csv_file_path, data=csv_data, circuit=circuit_name)

        print('\n')
        print("Timestamp after Sample #{}: {} seconds".format(test_id + 1, round(time.time() - START_TIME, 5)))
        print('\n')

    print("Total Runtime: {}".format(round(time.time() - START_TIME, 5)))
