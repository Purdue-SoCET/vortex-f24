// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See VVortex.h for the primary calling header

#include "VVortex__pch.h"
#include "VVortex_VX_mem_bus_if__D40_T8.h"
#include "VVortex__Syms.h"

void VVortex_VX_mem_bus_if__D40_T8___ctor_var_reset(VVortex_VX_mem_bus_if__D40_T8* vlSelf);

VVortex_VX_mem_bus_if__D40_T8::VVortex_VX_mem_bus_if__D40_T8(VVortex__Syms* symsp, const char* v__name)
    : VerilatedModule{v__name}
    , vlSymsp{symsp}
 {
    // Reset structure values
    VVortex_VX_mem_bus_if__D40_T8___ctor_var_reset(this);
}

void VVortex_VX_mem_bus_if__D40_T8::__Vconfigure(bool first) {
    (void)first;  // Prevent unused variable warning
}

VVortex_VX_mem_bus_if__D40_T8::~VVortex_VX_mem_bus_if__D40_T8() {
}
