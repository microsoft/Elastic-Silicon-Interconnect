build:
	cmake -S ../../../../ -B../../../../bin
	cmake --build ../../../../bin
	cmake --build ../../../../bin --target install
	verilator --cc --trace --top-module cosim_test Cosim_DpiPkg.sv Cosim_Endpoint.sv driver.sv --exe driver.cpp ../../../../../bin/src/core/cosim/cosim_dpi_server/libEsiCosimDpiServer.so
	make USER_CPPFLAGS="-DVL_USER_STOP" -j -C obj_dir -f Vcosim_test.mk Vcosim_test

run:
	LD_LIBRARY_PATH=../../../../bin/src/core/cosim/cosim_dpi_server/ ./obj_dir/Vcosim_test +trace

debug:
	LD_LIBRARY_PATH=../../../../bin/src/core/cosim/cosim_dpi_server/ gdb ./obj_dir/Vcosim_test +trace

questa:
	cmake -S ../../../../ -B../../../../../bin
	cmake --build ../../../../../bin
	vlog -sv Cosim_DpiPkg.sv Cosim_Endpoint.sv driver.sv +acc
	vopt cosim_test -o cosim_test_opt +acc

run_questa:
	vsim -sv_lib ../../../../../bin/bin/Debug/EsiCosimDpiServer cosim_test_opt -voptargs="+acc" -do wave.do -c