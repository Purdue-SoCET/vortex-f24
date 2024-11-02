# Verilated -*- Makefile -*-
# DESCRIPTION: Verilator output: Makefile for building Verilated archive or executable
#
# Execute this makefile from the object directory:
#    make -f VVortex.mk

default: /home/ecegridfs/a/socet137/socet/vortex-f24/runtime/rtlsim/../librtlsim.so

### Constants...
# Perl executable (from $PERL)
PERL = perl
# Path to Verilator kit (from $VERILATOR_ROOT)
VERILATOR_ROOT = /home/ecegridfs/a/socet137/socet/vortex-f24/tools/verilator/share/verilator
# SystemC include directory with systemc.h (from $SYSTEMC_INCLUDE)
SYSTEMC_INCLUDE ?= 
# SystemC library directory with libsystemc.a (from $SYSTEMC_LIBDIR)
SYSTEMC_LIBDIR ?= 

### Switches...
# C++ code coverage  0/1 (from --prof-c)
VM_PROFC = 0
# SystemC output mode?  0/1 (from --sc)
VM_SC = 0
# Legacy or SystemC output mode?  0/1 (from --sc)
VM_SP_OR_SC = $(VM_SC)
# Deprecated
VM_PCLI = 1
# Deprecated: SystemC architecture to find link library path (from $SYSTEMC_ARCH)
VM_SC_TARGET_ARCH = linux

### Vars...
# Design prefix (from --prefix)
VM_PREFIX = VVortex
# Module prefix (from --prefix)
VM_MODPREFIX = VVortex
# User CFLAGS (from -CFLAGS on Verilator command line)
VM_USER_CFLAGS = \
	-std=c++17 -Wall -Wextra -Wfatal-errors -Wno-array-bounds -fPIC -Wno-maybe-uninitialized -I/home/ecegridfs/a/socet137/socet/vortex-f24/hw -I/home/ecegridfs/a/socet137/socet/vortex-f24/sim/common -I/home/ecegridfs/a/socet137/socet/vortex-f24/third_party/softfloat/source/include -I/home/ecegridfs/a/socet137/socet/vortex-f24/third_party/ramulator/ext/spdlog/include -I/home/ecegridfs/a/socet137/socet/vortex-f24/third_party/ramulator/ext/yaml-cpp/include -I/home/ecegridfs/a/socet137/socet/vortex-f24/third_party/ramulator/src -DXLEN_32  -O2 -DNDEBUG \

# User LDLIBS (from -LDFLAGS on Verilator command line)
VM_USER_LDLIBS = \
	-shared /home/ecegridfs/a/socet137/socet/vortex-f24/third_party/softfloat/build/Linux-x86_64-GCC/softfloat.a -Wl,-rpath,/home/ecegridfs/a/socet137/socet/vortex-f24/third_party/ramulator  -L/home/ecegridfs/a/socet137/socet/vortex-f24/third_party/ramulator -lramulator \

# User .cpp files (from .cpp's on Verilator command line)
VM_USER_CLASSES = \
	float_dpi \
	util_dpi \
	dram_sim \
	mem \
	rvfloats \
	util \
	processor \

# User .cpp directories (from .cpp's on Verilator command line)
VM_USER_DIR = \
	/home/ecegridfs/a/socet137/socet/vortex-f24/hw/dpi \
	/home/ecegridfs/a/socet137/socet/vortex-f24/sim/common \
	/home/ecegridfs/a/socet137/socet/vortex-f24/sim/rtlsim \


### Default rules...
# Include list of all generated classes
include VVortex_classes.mk
# Include global rules
include $(VERILATOR_ROOT)/include/verilated.mk

### Executable rules... (from --exe)
VPATH += $(VM_USER_DIR)

float_dpi.o: /home/ecegridfs/a/socet137/socet/vortex-f24/hw/dpi/float_dpi.cpp
	$(OBJCACHE) $(CXX) $(CXXFLAGS) $(CPPFLAGS) $(OPT_FAST) -c -o $@ $<
util_dpi.o: /home/ecegridfs/a/socet137/socet/vortex-f24/hw/dpi/util_dpi.cpp
	$(OBJCACHE) $(CXX) $(CXXFLAGS) $(CPPFLAGS) $(OPT_FAST) -c -o $@ $<
dram_sim.o: /home/ecegridfs/a/socet137/socet/vortex-f24/sim/common/dram_sim.cpp
	$(OBJCACHE) $(CXX) $(CXXFLAGS) $(CPPFLAGS) $(OPT_FAST) -c -o $@ $<
mem.o: /home/ecegridfs/a/socet137/socet/vortex-f24/sim/common/mem.cpp
	$(OBJCACHE) $(CXX) $(CXXFLAGS) $(CPPFLAGS) $(OPT_FAST) -c -o $@ $<
rvfloats.o: /home/ecegridfs/a/socet137/socet/vortex-f24/sim/common/rvfloats.cpp
	$(OBJCACHE) $(CXX) $(CXXFLAGS) $(CPPFLAGS) $(OPT_FAST) -c -o $@ $<
util.o: /home/ecegridfs/a/socet137/socet/vortex-f24/sim/common/util.cpp
	$(OBJCACHE) $(CXX) $(CXXFLAGS) $(CPPFLAGS) $(OPT_FAST) -c -o $@ $<
processor.o: /home/ecegridfs/a/socet137/socet/vortex-f24/sim/rtlsim/processor.cpp
	$(OBJCACHE) $(CXX) $(CXXFLAGS) $(CPPFLAGS) $(OPT_FAST) -c -o $@ $<

### Link rules... (from --exe)
/home/ecegridfs/a/socet137/socet/vortex-f24/runtime/rtlsim/../librtlsim.so: $(VK_USER_OBJS) $(VK_GLOBAL_OBJS) $(VM_PREFIX)__ALL.a $(VM_HIER_LIBS)
	$(LINK) $(LDFLAGS) $^ $(LOADLIBES) $(LDLIBS) $(LIBS) $(SC_LIBS) -o $@


# Verilated -*- Makefile -*-
