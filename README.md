# Repository for SPEED: <ins>S</ins>witching <ins>P</ins>ower <ins>E</ins>stimation using Machine Learning with Time-<ins>E</ins>fficient <ins>D</ins>ata Collection

## EECS 251B - CS 289A Graduate Project

## Authors:
* Yuki Ito (itoyuki@berkeley.edu)
* Hoang Nguyen (hoang_nguyen@berkeley.edu)

## File/Directory Descriptions:
* [`src/`](https://github.com/ucb-eecs251b/sp22-project-ito-nguyen/tree/main/src): contains the Verilog modules and testbenches
    * `src/ALUop.vh`: library file for ALU opcodes
    * `src/MODULE.v`: Verilog module for DUT created during data collection from templates
    * `src/MODULE_tb.v`: Verilog module for DUT testbench created during data collection from templates


* [`templates/`](https://github.com/ucb-eecs251b/sp22-project-ito-nguyen/tree/main/templates): contains the templates for Verilog modules and testbenches to be consumed by `collect_training_data.py`
    * `alu_template.txt`: Template for ALU module
    * `alu_tb_template.txt`: Template for ALU testbench
    * `mux_template.txt`: Template for MUX module
    * `mux_tb_template.txt`: Template for MUX testbench
    * `seg7_template.txt`: Template for BCD to 7-segment converter module
    * `seg7_tb_template.txt`: Tamplate for BCD to 7-segment converter testbench

* [`dataset`](https://github.com/ucb-eecs251b/sp22-project-ito-nguyen/tree/main/dataset): contains the raw and processed data for training

* [`images`](https://github.com/ucb-eecs251b/sp22-project-ito-nguyen/tree/main/images): contains generated plots and figures

* Python files:
   * [`collect_training_data.py`](https://github.com/ucb-eecs251b/sp22-project-ito-nguyen/blob/main/collect_training_data.py): Automatically generate Verilog testbenches for a 2-input 32-bit ALU DUT, then run RTL simulations and collect training samples and labels.
   * [`preprocess_data.py`](https://github.com/ucb-eecs251b/sp22-project-ito-nguyen/blob/main/preprocess_data.py): Perform preprocessing on the raw data collected from `collect_training_data.py`.
   * [`models.py`](https://github.com/ucb-eecs251b/sp22-project-ito-nguyen/blob/main/models.py): contains machine learning models.
   * [`methods.py`](https://github.com/ucb-eecs251b/sp22-project-ito-nguyen/blob/main/methods.py): contains methods for train and evaluate the machine learning models.
   * [`utils.py`](https://github.com/ucb-eecs251b/sp22-project-ito-nguyen/blob/main/utils.py): contains useful helper functions.
   * [`plot_final_comparison.py`](https://github.com/ucb-eecs251b/sp22-project-ito-nguyen/blob/main/plot_final_comparison.py): to generate a plot comparing all machine learning models and all circuits.

* Config/Make/Tcl files:
    * [`sim-inputs.yml`](https://github.com/ucb-eecs251b/sp22-project-ito-nguyen/blob/main/sim-inputs.yml): Configs for running RTL simulation
    * [`Makefile`](https://github.com/ucb-eecs251b/sp22-project-ito-nguyen/blob/main/Makefile): Run sim (and convert to VCD format): `make vpd2vcd`; Run Joules: `make joules`
    * [`main.tcl`](https://github.com/ucb-eecs251b/sp22-project-ito-nguyen/blob/main/main.tcl): TCL commands for running Joules
    * [`module.sdc`](https://github.com/ucb-eecs251b/sp22-project-ito-nguyen/blob/main/module.sdc): timing configs/constraints

* Jupyter Notebook files: for testing machine learning methods

## Setup (First-time Users):
* SSH into EDA machine
* Create a work directory and clone chipyard:
```bash
mkdir /scratch/<your-username>
cd /scratch/<your-username>
git clone /home/ff/eecs251b/sp22-workspace/chipyard
```
* Set up the project repo and install necessary Python packages: 
```bash
cd chipyard/vlsi/
git clone https://github.com/ucb-eecs251b/sp22-project-ito-nguyen.git
cd sp22-project-ito-nguyen/
pip install --user -r requirements.txt
```

## Usages:
* SSH into EDA machine and go to the project directory
```bash
cd /scratch/<your-username>/chipyard/vlsi/sp22-project-ito-nguyen/
```

* Activate Hammer environments (every time opening a new terminal):
```bash
source source_env
```

* (Recommended) Activate `tmux`/`screen` (allows the process to keep running after we exit the shell):
    * `tmux` cheatsheet: https://tmuxcheatsheet.com/
    * `screen` cheatsheet: https://maojr.github.io/screencheatsheet/

* Run simulation and collect data:
```
python3 collect_training_data.py --circuit <circuit_name> --num_samples <number_of_samples_to_collect>
```

* Preprocess data:
```
python3 preprocess_data.py --circuit <circuit_name>
```

Notes:
* `<circuit_name>` (str): alu32, alu64, mux32, mux64, 7sg, etc.
* `<number_of_samples_to_collect>` (int): 1000, 2000, etc.
