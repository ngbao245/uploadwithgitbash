# TODO report timing? Just to make sure...
# TODO figure out how to use PVT corners

# User defined variables
set LIB_HOME /home/ff/eecs251b/sp22-workspace/asap7/asap7sc7p5t_27/LIB/NLDM
set TOP_MODULE MODULE
set DUT_INSTANCE /MODULE_tb/DUT
set RTL "
    $::env(PRJHOME)/vlsi/sp22-project-ito-nguyen/src/MODULE.v
"
set SYN_EFFORT medium
set LEAKAGE_EFFORT low
set CLOCK_BUFFERS "
    BUFx2_ASAP7_75t_R
    BUFx3_ASAP7_75t_R
    BUFx4_ASAP7_75t_R
    BUFx5_ASAP7_75t_R
    BUFx8_ASAP7_75t_R
    BUFx10_ASAP7_75t_R
    BUFx12_ASAP7_75t_R
    BUFx24_ASAP7_75t_R
    BUFx2_ASAP7_75t_L
    BUFx3_ASAP7_75t_L
    BUFx4_ASAP7_75t_L
    BUFx5_ASAP7_75t_L
    BUFx8_ASAP7_75t_L
    BUFx10_ASAP7_75t_L
    BUFx12_ASAP7_75t_L
    BUFx24_ASAP7_75t_L
    BUFx2_ASAP7_75t_SL
    BUFx3_ASAP7_75t_SL
    BUFx4_ASAP7_75t_SL
    BUFx5_ASAP7_75t_SL
    BUFx8_ASAP7_75t_SL
    BUFx10_ASAP7_75t_SL
    BUFx12_ASAP7_75t_SL
    BUFx24_ASAP7_75t_SL
"
set STIM_FILE $::env(PRJHOME)/vlsi/sp22-project-ito-nguyen/build/sim/sim-rundir/sim.vcd
set STIM_TYPE vcd
set FRAME_SIZE 2ns
set STIM_START 0ns
set STIM_END 2ns

# Read library and create library domains
# TODO Add support for PVT corners
read_libs -node 28 -libs "
    $LIB_HOME/asap7sc7p5t_AO_RVT_TT_nldm_201020.lib.gz
    $LIB_HOME/asap7sc7p5t_INVBUF_RVT_TT_nldm_201020.lib.gz
    $LIB_HOME/asap7sc7p5t_OA_RVT_TT_nldm_201020.lib.gz
    $LIB_HOME/asap7sc7p5t_SEQ_RVT_TT_nldm_201020.lib.gz
    $LIB_HOME/asap7sc7p5t_SIMPLE_RVT_TT_nldm_201020.lib.gz
    $LIB_HOME/asap7sc7p5t_AO_LVT_TT_nldm_201020.lib.gz
    $LIB_HOME/asap7sc7p5t_INVBUF_LVT_TT_nldm_201020.lib.gz
    $LIB_HOME/asap7sc7p5t_OA_LVT_TT_nldm_201020.lib.gz
    $LIB_HOME/asap7sc7p5t_SEQ_LVT_TT_nldm_201020.lib.gz
    $LIB_HOME/asap7sc7p5t_SIMPLE_LVT_TT_nldm_201020.lib.gz
    $LIB_HOME/asap7sc7p5t_AO_SLVT_TT_nldm_201020.lib.gz
    $LIB_HOME/asap7sc7p5t_INVBUF_SLVT_TT_nldm_201020.lib.gz
    $LIB_HOME/asap7sc7p5t_OA_SLVT_TT_nldm_201020.lib.gz
    $LIB_HOME/asap7sc7p5t_SEQ_SLVT_TT_nldm_201020.lib.gz
    $LIB_HOME/asap7sc7p5t_SIMPLE_SLVT_TT_nldm_201020.lib.gz
"

# TODO Add support for PLE flow - i.e. read LEFs and captable

# Read RTL design and CPF power intent
read_hdl -sv $RTL
# TODO Add support for CPF flow
elaborate $TOP_MODULE
write_db -all_root_attributes -to_file build/joules/$TOP_MODULE.elab.db

# Read stimuli files and create Joules SDB
read_stimulus \
    -file $STIM_FILE \
    -format $STIM_TYPE \
    -dut_instance $DUT_INSTANCE \
    -start $STIM_START \
    -end $STIM_END \
    -interval_size $FRAME_SIZE
write_sdb -out build/joules/joules.sdb

# Synthesis the design and (optionally) insert DFT
read_sdc module.sdc
syn_power \
    -effort $SYN_EFFORT \
    -leakage_power_effort $LEAKAGE_EFFORT \
    -to_file build/joules/$TOP_MODULE.syn.db

# Compute and report time based power (also runs activity propagation)
compute_power -mode time_based
report_power -unit mW -out build/joules/power_report.time_based.csv

exit
