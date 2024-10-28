// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See VVortex.h for the primary calling header

#include "VVortex__pch.h"
#include "VVortex_VX_mem_bus_if__D40_T7.h"

VL_ATTR_COLD void VVortex_VX_mem_bus_if__D40_T7___ctor_var_reset(VVortex_VX_mem_bus_if__D40_T7* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    VVortex__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+                    VVortex_VX_mem_bus_if__D40_T7___ctor_var_reset\n"); );
    // Body
    VL_RAND_RESET_W(613, vlSelf->req_data);
    VL_RAND_RESET_W(519, vlSelf->rsp_data);
}
