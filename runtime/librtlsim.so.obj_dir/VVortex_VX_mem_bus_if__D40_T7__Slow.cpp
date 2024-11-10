// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See VVortex.h for the primary calling header

#include "VVortex__pch.h"
#include "VVortex_VX_mem_bus_if__D40_T7.h"
#include "VVortex__Syms.h"

void VVortex_VX_mem_bus_if__D40_T7___ctor_var_reset(VVortex_VX_mem_bus_if__D40_T7* vlSelf);

VVortex_VX_mem_bus_if__D40_T7::VVortex_VX_mem_bus_if__D40_T7(VVortex__Syms* symsp, const char* v__name)
    : VerilatedModule{v__name}
    , vlSymsp{symsp}
 {
    // Reset structure values
    VVortex_VX_mem_bus_if__D40_T7___ctor_var_reset(this);
}

void VVortex_VX_mem_bus_if__D40_T7::__Vconfigure(bool first) {
    (void)first;  // Prevent unused variable warning
}

VVortex_VX_mem_bus_if__D40_T7::~VVortex_VX_mem_bus_if__D40_T7() {
}
