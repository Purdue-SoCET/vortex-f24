// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See VVortex.h for the primary calling header

#include "VVortex__pch.h"
#include "VVortex__Syms.h"
#include "VVortex___024root.h"

VL_ATTR_COLD void VVortex___024root___stl_comb__TOP__71(VVortex___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    VVortex__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVortex___024root___stl_comb__TOP__71\n"); );
    // Init
    CData/*3:0*/ Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard__DOT__arb_ready_in;
    Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard__DOT__arb_ready_in = 0;
    // Body
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard__DOT__out_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__fair_arbiter__DOT__genblk1__DOT__priority_arbiter__DOT__genblk1__DOT__priority_encoder__DOT__genblk2__DOT__scan__DOT__t 
        = ((0xf00U & ((0xffffff00U & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard__DOT__out_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__fair_arbiter__DOT__genblk1__DOT__priority_arbiter__DOT__genblk1__DOT__priority_encoder__DOT__genblk2__DOT__scan__DOT__t) 
                                      << 4U)) | ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard__DOT__out_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__fair_arbiter__DOT__genblk1__DOT__priority_arbiter__DOT__genblk1__DOT__priority_encoder__DOT__genblk2__DOT__scan__DOT__genblk2__DOT__genblk1__BRA__1__KET____DOT__shifted) 
                                                 << 8U))) 
           | ((0xf0U & (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard__DOT__out_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__fair_arbiter__DOT__genblk1__DOT__priority_arbiter__DOT__genblk1__DOT__priority_encoder__DOT__genblk2__DOT__scan__DOT__t) 
                         | (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard__DOT__out_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__fair_arbiter__DOT__genblk1__DOT__priority_arbiter__DOT__genblk1__DOT__priority_encoder__DOT__genblk2__DOT__scan__DOT__genblk2__DOT__genblk1__BRA__0__KET____DOT__shifted)) 
                        << 4U)) | VL_STREAML_FAST_III(4, (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard__DOT__out_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__fair_arbiter__DOT__genblk1__DOT__requests_qual), 0)));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard__DOT__out_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot 
        = (((8U & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard__DOT__out_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__fair_arbiter__DOT__genblk1__DOT__priority_arbiter__DOT__genblk1__DOT__priority_encoder__DOT__genblk2__DOT__scan__DOT__t) 
                   >> 5U)) | (IData)(vlSelf->__VdfgRegularize_hd87f99a1_95_0)) 
           & (1U | (0xeU & ((~ (IData)(vlSelf->__VdfgRegularize_hd87f99a1_95_0)) 
                            << 1U))));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard__DOT__out_arb__DOT__genblk1__DOT__genblk1__DOT__out_buf__DOT__genblk1__DOT__push 
        = (1U & ((~ (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard__DOT__out_arb__DOT__genblk1__DOT__genblk1__DOT__out_buf__DOT__genblk1__DOT__fifo_queue__DOT__genblk6__DOT__full_r)) 
                 & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard__DOT__out_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__fair_arbiter__DOT__genblk1__DOT__priority_arbiter__DOT__genblk1__DOT__priority_encoder__DOT__genblk2__DOT__scan__DOT__t) 
                    >> 8U)));
    Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard__DOT__arb_ready_in 
        = ((((~ (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard__DOT__out_arb__DOT__genblk1__DOT__genblk1__DOT__out_buf__DOT__genblk1__DOT__fifo_queue__DOT__genblk6__DOT__full_r)) 
             & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard__DOT__out_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                >> 3U)) << 3U) | ((4U & (((~ (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard__DOT__out_arb__DOT__genblk1__DOT__genblk1__DOT__out_buf__DOT__genblk1__DOT__fifo_queue__DOT__genblk6__DOT__full_r)) 
                                          << 2U) & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard__DOT__out_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot))) 
                                  | (3U & ((- (IData)(
                                                      (1U 
                                                       & (~ (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard__DOT__out_arb__DOT__genblk1__DOT__genblk1__DOT__out_buf__DOT__genblk1__DOT__fifo_queue__DOT__genblk6__DOT__full_r))))) 
                                           & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard__DOT__out_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot)))));
    vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard__DOT__staging_if__BRA__0__KET__.ready 
        = (1U & ((IData)(Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard__DOT__arb_ready_in) 
                 & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard__DOT__operands_ready)));
    vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard__DOT__staging_if__BRA__1__KET__.ready 
        = (1U & (((IData)(Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard__DOT__arb_ready_in) 
                  & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard__DOT__operands_ready)) 
                 >> 1U));
    vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard__DOT__staging_if__BRA__2__KET__.ready 
        = (1U & (((IData)(Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard__DOT__arb_ready_in) 
                  & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard__DOT__operands_ready)) 
                 >> 2U));
    vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard__DOT__staging_if__BRA__3__KET__.ready 
        = (1U & (((IData)(Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard__DOT__arb_ready_in) 
                  & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard__DOT__operands_ready)) 
                 >> 3U));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard__DOT__genblk2__BRA__0__KET____DOT__staging_fire 
        = ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard__DOT__genblk1__BRA__0__KET____DOT__stanging_buf__DOT__genblk1__DOT__pipe_buffer__DOT__genblk1__DOT__genblk1__BRA__0__KET____DOT__pipe_register__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_r) 
           & (IData)(vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard__DOT__staging_if__BRA__0__KET__.ready));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard__DOT__genblk2__BRA__1__KET____DOT__staging_fire 
        = ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard__DOT__genblk1__BRA__1__KET____DOT__stanging_buf__DOT__genblk1__DOT__pipe_buffer__DOT__genblk1__DOT__genblk1__BRA__0__KET____DOT__pipe_register__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_r) 
           & (IData)(vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard__DOT__staging_if__BRA__1__KET__.ready));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard__DOT__genblk2__BRA__2__KET____DOT__staging_fire 
        = ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard__DOT__genblk1__BRA__2__KET____DOT__stanging_buf__DOT__genblk1__DOT__pipe_buffer__DOT__genblk1__DOT__genblk1__BRA__0__KET____DOT__pipe_register__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_r) 
           & (IData)(vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard__DOT__staging_if__BRA__2__KET__.ready));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard__DOT__genblk2__BRA__3__KET____DOT__staging_fire 
        = ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard__DOT__genblk1__BRA__3__KET____DOT__stanging_buf__DOT__genblk1__DOT__pipe_buffer__DOT__genblk1__DOT__genblk1__BRA__0__KET____DOT__pipe_register__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_r) 
           & (IData)(vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard__DOT__staging_if__BRA__3__KET__.ready));
}

VL_ATTR_COLD void VVortex___024root___stl_comb__TOP__76(VVortex___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    VVortex__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVortex___024root___stl_comb__TOP__76\n"); );
    // Init
    VlWide<4>/*127:0*/ __Vtemp_2;
    // Body
    __Vtemp_2[0U] = (IData)((((QData)((IData)((0xfffffffU 
                                               & ((vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lsu_dcache_if__BRA__0__KET__.req_data[6U] 
                                                   << 0x12U) 
                                                  | (vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lsu_dcache_if__BRA__0__KET__.req_data[5U] 
                                                     >> 0xeU))))) 
                              << 0x1cU) | (QData)((IData)(
                                                          (0xfffffffU 
                                                           & ((vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lsu_dcache_if__BRA__0__KET__.req_data[5U] 
                                                               << 0x10U) 
                                                              | (vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lsu_dcache_if__BRA__0__KET__.req_data[4U] 
                                                                 >> 0x10U)))))));
    __Vtemp_2[1U] = ((0xff000000U & (vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lsu_dcache_if__BRA__0__KET__.req_data[6U] 
                                     << 0xcU)) | (IData)(
                                                         ((((QData)((IData)(
                                                                            (0xfffffffU 
                                                                             & ((vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lsu_dcache_if__BRA__0__KET__.req_data[6U] 
                                                                                << 0x12U) 
                                                                                | (vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lsu_dcache_if__BRA__0__KET__.req_data[5U] 
                                                                                >> 0xeU))))) 
                                                            << 0x1cU) 
                                                           | (QData)((IData)(
                                                                             (0xfffffffU 
                                                                              & ((vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lsu_dcache_if__BRA__0__KET__.req_data[5U] 
                                                                                << 0x10U) 
                                                                                | (vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lsu_dcache_if__BRA__0__KET__.req_data[4U] 
                                                                                >> 0x10U)))))) 
                                                          >> 0x20U)));
    __Vtemp_2[2U] = ((0xfff00000U & (vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lsu_dcache_if__BRA__0__KET__.req_data[7U] 
                                     << 0xaU)) | (0xfffffU 
                                                  & ((0xfff000U 
                                                      & (vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lsu_dcache_if__BRA__0__KET__.req_data[7U] 
                                                         << 0xcU)) 
                                                     | (vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lsu_dcache_if__BRA__0__KET__.req_data[6U] 
                                                        >> 0x14U))));
    __Vtemp_2[3U] = (0xffffU & ((0xffc00U & (vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lsu_dcache_if__BRA__0__KET__.req_data[8U] 
                                             << 0xaU)) 
                                | (vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lsu_dcache_if__BRA__0__KET__.req_data[7U] 
                                   >> 0x16U)));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__seed_addr_n 
        = (0xfffffffU & ((0x6fU >= (0x7fU & ((IData)(0x1cU) 
                                             * (3U 
                                                & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__genblk4__BRA__0__KET____DOT__priority_encoder__DOT__genblk2__DOT__lzc__DOT__genblk1__DOT__find_first__DOT__d_n)))))
                          ? (((0U == (0x1fU & ((IData)(0x1cU) 
                                               * (3U 
                                                  & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__genblk4__BRA__0__KET____DOT__priority_encoder__DOT__genblk2__DOT__lzc__DOT__genblk1__DOT__find_first__DOT__d_n)))))
                               ? 0U : (__Vtemp_2[(((IData)(0x1bU) 
                                                   + 
                                                   (0x7fU 
                                                    & ((IData)(0x1cU) 
                                                       * 
                                                       (3U 
                                                        & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__genblk4__BRA__0__KET____DOT__priority_encoder__DOT__genblk2__DOT__lzc__DOT__genblk1__DOT__find_first__DOT__d_n))))) 
                                                  >> 5U)] 
                                       << ((IData)(0x20U) 
                                           - (0x1fU 
                                              & ((IData)(0x1cU) 
                                                 * 
                                                 (3U 
                                                  & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__genblk4__BRA__0__KET____DOT__priority_encoder__DOT__genblk2__DOT__lzc__DOT__genblk1__DOT__find_first__DOT__d_n))))))) 
                             | (__Vtemp_2[(3U & (((IData)(0x1cU) 
                                                  * 
                                                  (3U 
                                                   & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__genblk4__BRA__0__KET____DOT__priority_encoder__DOT__genblk2__DOT__lzc__DOT__genblk1__DOT__find_first__DOT__d_n))) 
                                                 >> 5U))] 
                                >> (0x1fU & ((IData)(0x1cU) 
                                             * (3U 
                                                & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__genblk4__BRA__0__KET____DOT__priority_encoder__DOT__genblk2__DOT__lzc__DOT__genblk1__DOT__find_first__DOT__d_n))))))
                          : vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT____Vxrand_h39f4912a__0));
}
