// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See VVortex.h for the primary calling header

#include "VVortex__pch.h"
#include "VVortex_VX_ibuffer_if.h"

VL_ATTR_COLD void VVortex_VX_ibuffer_if___ctor_var_reset(VVortex_VX_ibuffer_if* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    VVortex__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+                    VVortex_VX_ibuffer_if___ctor_var_reset\n"); );
    // Body
    vlSelf->ready = VL_RAND_RESET_I(1);
}
