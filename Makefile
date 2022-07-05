# Show this help.
help:
	@awk '/^#/{c=substr($$0,3);next}c&&/^[[:alpha:]][[:alnum:]_-]+:/{print substr($$1,1,index($$1,":")),c}1{c=0}' $(MAKEFILE_LIST) | column -s: -t

# Deletes all generated files. 
clean:
	rm -rf build/ fv/ *.log

# Run simulation via hammer to generate .vcd and .saif files.
sim: clean
	cd ..; \
	./example-vlsi \
		sim -e \
		./inst-env.yml -p \
		./inst-asap7.yml \
		-p sp22-project-ito-nguyen/sim-inputs.yml \
		--obj_dir sp22-project-ito-nguyen/build/sim

# Converts generated vpd file to ascii vcd file.
vpd2vcd: sim
	vpd2vcd build/sim/sim-rundir/vcdplus.vpd | sed '1,6d' | tee build/sim/sim-rundir/sim.vcd

# Run joules using generated .vcd and .saif files.
joules: 
	joules -common_ui -overwrite -work build/joules -files main.tcl
	rm -rf fv/

.PHONY: sim default clean help
