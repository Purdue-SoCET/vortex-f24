// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See VVortex.h for the primary calling header

#include "VVortex__pch.h"
#include "VVortex_VX_lsu_mem_if__N4_D4_T2.h"

VL_ATTR_COLD void VVortex_VX_lsu_mem_if__N4_D4_T2___ctor_var_reset(VVortex_VX_lsu_mem_if__N4_D4_T2* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    VVortex__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+                    VVortex_VX_lsu_mem_if__N4_D4_T2___ctor_var_reset\n"); );
    // Body
    VL_RAND_RESET_W(283, vlSelf->req_data);
    vlSelf->req_ready = VL_RAND_RESET_I(1);
    VL_RAND_RESET_W(134, vlSelf->rsp_data);
}
