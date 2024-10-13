// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See VVortex.h for the primary calling header

#include "VVortex__pch.h"
#include "VVortex_VX_warp_ctl_if.h"
#include "VVortex__Syms.h"

void VVortex_VX_warp_ctl_if___ctor_var_reset(VVortex_VX_warp_ctl_if* vlSelf);

VVortex_VX_warp_ctl_if::VVortex_VX_warp_ctl_if(VVortex__Syms* symsp, const char* v__name)
    : VerilatedModule{v__name}
    , vlSymsp{symsp}
 {
    // Reset structure values
    VVortex_VX_warp_ctl_if___ctor_var_reset(this);
}

void VVortex_VX_warp_ctl_if::__Vconfigure(bool first) {
    (void)first;  // Prevent unused variable warning
}

VVortex_VX_warp_ctl_if::~VVortex_VX_warp_ctl_if() {
}
