build:
	cmake -S ../../../../ -B/tmp/esi/bin
	cmake --build /tmp/esi/bin
	# cmake --build ../../../../bin --target install
	verilator --cc --trace --top-module cosim_test Cosim_DpiPkg.sv Cosim_Endpoint.sv driver.sv --exe driver.cpp /tmp/esi/bin/src/core/cosim/cosim_dpi_server/libEsiCosimDpiServer.so -LDFLAGS "-lpthread"
	make USER_CPPFLAGS="-DVL_USER_STOP -O0 -ggdb" -j -C obj_dir -f Vcosim_test.mk Vcosim_test

run:
	LD_LIBRARY_PATH=/tmp/esi/bin/src/core/cosim/cosim_dpi_server/ ./obj_dir/Vcosim_test +trace

valgrind:
	LD_LIBRARY_PATH=/tmp/esi/bin/src/core/cosim/cosim_dpi_server/ valgrind ./obj_dir/Vcosim_test +trace

debug:
	LD_LIBRARY_PATH=/tmp/esi/bin/src/core/cosim/cosim_dpi_server/ gdb ./obj_dir/Vcosim_test +trace

questa:
	cmake -S ../../../../ -B/tmp/esi/bin
	cmake --build /tmp/esi/bin
	vlog -sv Cosim_DpiPkg.sv Cosim_Endpoint.sv driver.sv +acc
	vopt cosim_test -o cosim_test_opt +acc

qrun:
	vsim -sv_lib /tmp/esi/bin/src/core/cosim/cosim_dpi_server/libEsiCosimDpiServer cosim_test_opt -voptargs="+acc" -do wave.do -c

qgui:
	vsim -sv_lib /tmp/esi/bin/src/core/cosim/cosim_dpi_server/libEsiCosimDpiServer cosim_test_opt -voptargs="+acc" -do wave.do