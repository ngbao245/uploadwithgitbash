create_clock clk -name clk -period 1000.0
set_clock_uncertainty 100.0 [get_clocks clk]
set_clock_groups -asynchronous  -group { clk }
set_load 1.0 [all_outputs]
