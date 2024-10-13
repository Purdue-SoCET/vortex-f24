# Verilated -*- Makefile -*-
# DESCRIPTION: Verilator output: Make include file with class lists
#
# This file lists generated Verilated files, for including in higher level makefiles.
# See VVortex.mk for the caller.

### Switches...
# C11 constructs required?  0/1 (always on now)
VM_C11 = 1
# Timing enabled?  0/1
VM_TIMING = 0
# Coverage output mode?  0/1 (from --coverage)
VM_COVERAGE = 0
# Parallel builds?  0/1 (from --output-split)
VM_PARALLEL_BUILDS = 1
# Tracing output mode?  0/1 (from --trace/--trace-fst)
VM_TRACE = 0
# Tracing output mode in VCD format?  0/1 (from --trace)
VM_TRACE_VCD = 0
# Tracing output mode in FST format?  0/1 (from --trace-fst)
VM_TRACE_FST = 0

### Object file lists...
# Generated module classes, fast-path, compile with highest optimization
VM_CLASSES_FAST += \
	VVortex \
	VVortex___024root__DepSet_h1c673058__0 \
	VVortex___024root__DepSet_h1c673058__1 \
	VVortex___024root__DepSet_h1c673058__2 \
	VVortex___024root__DepSet_h1c673058__3 \
	VVortex___024root__DepSet_h1c673058__4 \
	VVortex___024root__DepSet_hc3f83dbf__0 \
	VVortex___024root__DepSet_hc3f83dbf__1 \
	VVortex___024root__DepSet_hc3f83dbf__2 \
	VVortex___024unit__DepSet_h5c1e1864__0 \
	VVortex_VX_schedule_if__DepSet_h21f00a0b__0 \
	VVortex_VX_fetch_if__DepSet_h9a44b6fb__0 \
	VVortex_VX_trace_pkg__DepSet_h6661a6b0__0 \
	VVortex_VX_decode_if__DepSet_he272fd78__0 \
	VVortex_VX_warp_ctl_if__DepSet_hdd229499__0 \
	VVortex_VX_commit_if__DepSet_h515c5650__0 \
	VVortex_VX_writeback_if__DepSet_h9b79735c__0 \
	VVortex_VX_ibuffer_if__DepSet_h283232f8__0 \
	VVortex_VX_scoreboard_if__DepSet_h583683ba__0 \
	VVortex_VX_operands_if__DepSet_h5596ade6__0 \
	VVortex_VX_mem_bus_if__D40_T8__DepSet_h5e11b043__0 \
	VVortex_VX_mem_bus_if__D4_T3__DepSet_h160029bf__0 \
	VVortex_VX_mem_bus_if__D40_T4__DepSet_h3acc36d8__0 \
	VVortex_VX_mem_bus_if__D10_T3__DepSet_h4d08320b__0 \
	VVortex_VX_mem_bus_if__D40_T7__DepSet_hb17cc4a5__0 \
	VVortex_VX_lsu_mem_if__N4_D4_T2__DepSet_h66bbb8e0__0 \
	VVortex_VX_lsu_mem_if__D10_T3__DepSet_h5f6dfcf3__0 \
	VVortex_VX_mem_bus_if__D40_T6__DepSet_hf5f2f98a__0 \
	VVortex_VX_mem_bus_if__D4_T2__DepSet_hb990928b__0 \
	VVortex_VX_execute_if__N4__DepSet_hd0e0828a__0 \

# Generated module classes, non-fast-path, compile with low/medium optimization
VM_CLASSES_SLOW += \
	VVortex__ConstPool_0 \
	VVortex___024root__Slow \
	VVortex___024root__DepSet_h1c673058__0__Slow \
	VVortex___024root__DepSet_h1c673058__1__Slow \
	VVortex___024root__DepSet_h1c673058__2__Slow \
	VVortex___024root__DepSet_hc3f83dbf__0__Slow \
	VVortex___024unit__Slow \
	VVortex___024unit__DepSet_h83f155a3__0__Slow \
	VVortex_VX_schedule_if__Slow \
	VVortex_VX_schedule_if__DepSet_h21f00a0b__0__Slow \
	VVortex_VX_fetch_if__Slow \
	VVortex_VX_fetch_if__DepSet_h9a44b6fb__0__Slow \
	VVortex_VX_trace_pkg__Slow \
	VVortex_VX_trace_pkg__DepSet_h93620eaf__0__Slow \
	VVortex_VX_decode_if__Slow \
	VVortex_VX_decode_if__DepSet_he272fd78__0__Slow \
	VVortex_VX_warp_ctl_if__Slow \
	VVortex_VX_warp_ctl_if__DepSet_hdd229499__0__Slow \
	VVortex_VX_commit_if__Slow \
	VVortex_VX_commit_if__DepSet_h515c5650__0__Slow \
	VVortex_VX_writeback_if__Slow \
	VVortex_VX_writeback_if__DepSet_h9b79735c__0__Slow \
	VVortex_VX_ibuffer_if__Slow \
	VVortex_VX_ibuffer_if__DepSet_h283232f8__0__Slow \
	VVortex_VX_scoreboard_if__Slow \
	VVortex_VX_scoreboard_if__DepSet_h583683ba__0__Slow \
	VVortex_VX_operands_if__Slow \
	VVortex_VX_operands_if__DepSet_h5596ade6__0__Slow \
	VVortex_VX_mem_bus_if__D40_T8__Slow \
	VVortex_VX_mem_bus_if__D40_T8__DepSet_h5e11b043__0__Slow \
	VVortex_VX_mem_bus_if__D4_T3__Slow \
	VVortex_VX_mem_bus_if__D4_T3__DepSet_h160029bf__0__Slow \
	VVortex_VX_mem_bus_if__D40_T4__Slow \
	VVortex_VX_mem_bus_if__D40_T4__DepSet_h3acc36d8__0__Slow \
	VVortex_VX_mem_bus_if__D10_T3__Slow \
	VVortex_VX_mem_bus_if__D10_T3__DepSet_h4d08320b__0__Slow \
	VVortex_VX_mem_bus_if__D40_T7__Slow \
	VVortex_VX_mem_bus_if__D40_T7__DepSet_hb17cc4a5__0__Slow \
	VVortex_VX_lsu_mem_if__N4_D4_T2__Slow \
	VVortex_VX_lsu_mem_if__N4_D4_T2__DepSet_h66bbb8e0__0__Slow \
	VVortex_VX_lsu_mem_if__D10_T3__Slow \
	VVortex_VX_lsu_mem_if__D10_T3__DepSet_h5f6dfcf3__0__Slow \
	VVortex_VX_mem_bus_if__D40_T6__Slow \
	VVortex_VX_mem_bus_if__D40_T6__DepSet_hf5f2f98a__0__Slow \
	VVortex_VX_mem_bus_if__D4_T2__Slow \
	VVortex_VX_mem_bus_if__D4_T2__DepSet_hb990928b__0__Slow \
	VVortex_VX_execute_if__N4__Slow \
	VVortex_VX_execute_if__N4__DepSet_hd0e0828a__0__Slow \

# Generated support classes, fast-path, compile with highest optimization
VM_SUPPORT_FAST += \
	VVortex__Dpi \

# Generated support classes, non-fast-path, compile with low/medium optimization
VM_SUPPORT_SLOW += \
	VVortex__Syms \

# Global classes, need linked once per executable, fast-path, compile with highest optimization
VM_GLOBAL_FAST += \
	verilated \
	verilated_dpi \
	verilated_threads \

# Global classes, need linked once per executable, non-fast-path, compile with low/medium optimization
VM_GLOBAL_SLOW += \


# Verilated -*- Makefile -*-
