// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See VVortex.h for the primary calling header

#include "VVortex__pch.h"
#include "VVortex__Syms.h"
#include "VVortex___024root.h"

void VVortex___024unit____Vdpiimwrap_dpi_fdiv_TOP____024unit(CData/*0:0*/ enable, IData/*31:0*/ dst_fmt, QData/*63:0*/ a, QData/*63:0*/ b, CData/*2:0*/ frm, QData/*63:0*/ &result, CData/*4:0*/ &fflags);
void VVortex___024unit____Vdpiimwrap_dpi_fsqrt_TOP____024unit(CData/*0:0*/ enable, IData/*31:0*/ dst_fmt, QData/*63:0*/ a, CData/*2:0*/ frm, QData/*63:0*/ &result, CData/*4:0*/ &fflags);

VL_INLINE_OPT void VVortex___024root___nba_comb__TOP__19(VVortex___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    VVortex__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVortex___024root___nba_comb__TOP__19\n"); );
    // Init
    VlWide<9>/*287:0*/ __Vtemp_8;
    VlWide<5>/*159:0*/ __Vtemp_11;
    VlWide<17>/*543:0*/ __Vtemp_25;
    VlWide<5>/*159:0*/ __Vtemp_29;
    // Body
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__div_sqrt_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot 
        = (((- (IData)((1U & (~ (IData)((0U != (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__div_sqrt_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__rr_arbiter__DOT__genblk1__DOT__req_masked))))))) 
            & ((~ (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__div_sqrt_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__rr_arbiter__DOT__genblk1__DOT__unmask_higher_pri_regs)) 
               & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT____Vcellinp__div_sqrt_arb__valid_in))) 
           | ((~ (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__div_sqrt_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__rr_arbiter__DOT__genblk1__DOT__mask_higher_pri_regs)) 
              & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__div_sqrt_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__rr_arbiter__DOT__genblk1__DOT__req_masked)));
    __Vtemp_8[0U] = (0x40U | ((0xffffff80U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__shift_reg__DOT__genblk1__DOT__entries[0U] 
                                              << 2U)) 
                              | ((0x3eU & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__shift_reg__DOT__genblk1__DOT__entries[0U] 
                                           << 1U)) 
                                 | (1U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__shift_reg__DOT__genblk1__DOT__entries[4U] 
                                          >> 5U)))));
    __Vtemp_8[1U] = (((0x7cU & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__shift_reg__DOT__genblk1__DOT__entries[1U] 
                                << 2U)) | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__shift_reg__DOT__genblk1__DOT__entries[0U] 
                                           >> 0x1eU)) 
                     | (0xffffff80U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__shift_reg__DOT__genblk1__DOT__entries[1U] 
                                       << 2U)));
    __Vtemp_8[2U] = (((0x7cU & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__shift_reg__DOT__genblk1__DOT__entries[2U] 
                                << 2U)) | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__shift_reg__DOT__genblk1__DOT__entries[1U] 
                                           >> 0x1eU)) 
                     | (0xffffff80U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__shift_reg__DOT__genblk1__DOT__entries[2U] 
                                       << 2U)));
    __Vtemp_8[3U] = (((0x7cU & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__shift_reg__DOT__genblk1__DOT__entries[3U] 
                                << 2U)) | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__shift_reg__DOT__genblk1__DOT__entries[2U] 
                                           >> 0x1eU)) 
                     | (0xffffff80U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__shift_reg__DOT__genblk1__DOT__entries[3U] 
                                       << 2U)));
    __Vtemp_8[4U] = (0x2000U | ((0xffffc000U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__shift_reg__DOT__genblk1__DOT__entries[0U] 
                                                << 9U)) 
                                | ((0x1f00U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__shift_reg__DOT__genblk1__DOT__entries[0U] 
                                               << 8U)) 
                                   | ((0x80U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__shift_reg__DOT__genblk1__DOT__entries[4U] 
                                                << 2U)) 
                                      | ((0x7cU & (
                                                   vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__shift_reg__DOT__genblk1__DOT__entries[4U] 
                                                   << 2U)) 
                                         | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__shift_reg__DOT__genblk1__DOT__entries[3U] 
                                            >> 0x1eU))))));
    __Vtemp_8[5U] = (((0x3e00U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__shift_reg__DOT__genblk1__DOT__entries[1U] 
                                  << 9U)) | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__shift_reg__DOT__genblk1__DOT__entries[0U] 
                                             >> 0x17U)) 
                     | (0xffffc000U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__shift_reg__DOT__genblk1__DOT__entries[1U] 
                                       << 9U)));
    __Vtemp_8[6U] = (((0x3e00U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__shift_reg__DOT__genblk1__DOT__entries[2U] 
                                  << 9U)) | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__shift_reg__DOT__genblk1__DOT__entries[1U] 
                                             >> 0x17U)) 
                     | (0xffffc000U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__shift_reg__DOT__genblk1__DOT__entries[2U] 
                                       << 9U)));
    __Vtemp_8[7U] = (((0x3e00U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__shift_reg__DOT__genblk1__DOT__entries[3U] 
                                  << 9U)) | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__shift_reg__DOT__genblk1__DOT__entries[2U] 
                                             >> 0x17U)) 
                     | (0xffffc000U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__shift_reg__DOT__genblk1__DOT__entries[3U] 
                                       << 9U)));
    __Vtemp_8[8U] = ((0x3e00U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__shift_reg__DOT__genblk1__DOT__entries[4U] 
                                 << 9U)) | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__shift_reg__DOT__genblk1__DOT__entries[3U] 
                                            >> 0x17U));
    if ((0x10dU >= (0x1ffU & ((IData)(0x87U) * (1U 
                                                & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__div_sqrt_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                                   >> 1U)))))) {
        __Vtemp_11[0U] = (((0U == (0x1fU & ((IData)(0x87U) 
                                            * (1U & 
                                               ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__div_sqrt_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                                >> 1U)))))
                            ? 0U : (__Vtemp_8[((IData)(1U) 
                                               + (0xfU 
                                                  & (((IData)(0x87U) 
                                                      * 
                                                      (1U 
                                                       & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__div_sqrt_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                                          >> 1U))) 
                                                     >> 5U)))] 
                                    << ((IData)(0x20U) 
                                        - (0x1fU & 
                                           ((IData)(0x87U) 
                                            * (1U & 
                                               ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__div_sqrt_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                                >> 1U))))))) 
                          | (__Vtemp_8[(0xfU & (((IData)(0x87U) 
                                                 * 
                                                 (1U 
                                                  & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__div_sqrt_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                                     >> 1U))) 
                                                >> 5U))] 
                             >> (0x1fU & ((IData)(0x87U) 
                                          * (1U & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__div_sqrt_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                                   >> 1U))))));
        __Vtemp_11[1U] = (((0U == (0x1fU & ((IData)(0x87U) 
                                            * (1U & 
                                               ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__div_sqrt_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                                >> 1U)))))
                            ? 0U : (__Vtemp_8[((IData)(2U) 
                                               + (0xfU 
                                                  & (((IData)(0x87U) 
                                                      * 
                                                      (1U 
                                                       & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__div_sqrt_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                                          >> 1U))) 
                                                     >> 5U)))] 
                                    << ((IData)(0x20U) 
                                        - (0x1fU & 
                                           ((IData)(0x87U) 
                                            * (1U & 
                                               ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__div_sqrt_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                                >> 1U))))))) 
                          | (__Vtemp_8[((IData)(1U) 
                                        + (0xfU & (
                                                   ((IData)(0x87U) 
                                                    * 
                                                    (1U 
                                                     & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__div_sqrt_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                                        >> 1U))) 
                                                   >> 5U)))] 
                             >> (0x1fU & ((IData)(0x87U) 
                                          * (1U & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__div_sqrt_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                                   >> 1U))))));
        __Vtemp_11[2U] = (((0U == (0x1fU & ((IData)(0x87U) 
                                            * (1U & 
                                               ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__div_sqrt_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                                >> 1U)))))
                            ? 0U : (__Vtemp_8[((IData)(3U) 
                                               + (0xfU 
                                                  & (((IData)(0x87U) 
                                                      * 
                                                      (1U 
                                                       & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__div_sqrt_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                                          >> 1U))) 
                                                     >> 5U)))] 
                                    << ((IData)(0x20U) 
                                        - (0x1fU & 
                                           ((IData)(0x87U) 
                                            * (1U & 
                                               ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__div_sqrt_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                                >> 1U))))))) 
                          | (__Vtemp_8[((IData)(2U) 
                                        + (0xfU & (
                                                   ((IData)(0x87U) 
                                                    * 
                                                    (1U 
                                                     & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__div_sqrt_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                                        >> 1U))) 
                                                   >> 5U)))] 
                             >> (0x1fU & ((IData)(0x87U) 
                                          * (1U & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__div_sqrt_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                                   >> 1U))))));
        __Vtemp_11[3U] = (((0U == (0x1fU & ((IData)(0x87U) 
                                            * (1U & 
                                               ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__div_sqrt_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                                >> 1U)))))
                            ? 0U : (__Vtemp_8[((IData)(4U) 
                                               + (0xfU 
                                                  & (((IData)(0x87U) 
                                                      * 
                                                      (1U 
                                                       & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__div_sqrt_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                                          >> 1U))) 
                                                     >> 5U)))] 
                                    << ((IData)(0x20U) 
                                        - (0x1fU & 
                                           ((IData)(0x87U) 
                                            * (1U & 
                                               ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__div_sqrt_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                                >> 1U))))))) 
                          | (__Vtemp_8[((IData)(3U) 
                                        + (0xfU & (
                                                   ((IData)(0x87U) 
                                                    * 
                                                    (1U 
                                                     & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__div_sqrt_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                                        >> 1U))) 
                                                   >> 5U)))] 
                             >> (0x1fU & ((IData)(0x87U) 
                                          * (1U & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__div_sqrt_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                                   >> 1U))))));
        __Vtemp_11[4U] = (((0U == (0x1fU & ((IData)(0x87U) 
                                            * (1U & 
                                               ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__div_sqrt_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                                >> 1U)))))
                            ? 0U : (__Vtemp_8[((IData)(5U) 
                                               + (0xfU 
                                                  & (((IData)(0x87U) 
                                                      * 
                                                      (1U 
                                                       & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__div_sqrt_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                                          >> 1U))) 
                                                     >> 5U)))] 
                                    << ((IData)(0x20U) 
                                        - (0x1fU & 
                                           ((IData)(0x87U) 
                                            * (1U & 
                                               ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__div_sqrt_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                                >> 1U))))))) 
                          | (__Vtemp_8[((IData)(4U) 
                                        + (0xfU & (
                                                   ((IData)(0x87U) 
                                                    * 
                                                    (1U 
                                                     & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__div_sqrt_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                                        >> 1U))) 
                                                   >> 5U)))] 
                             >> (0x1fU & ((IData)(0x87U) 
                                          * (1U & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__div_sqrt_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                                   >> 1U))))));
    } else {
        __Vtemp_11[0U] = vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__div_sqrt_arb__DOT____Vxrand_h8fdbc47e__0[0U];
        __Vtemp_11[1U] = vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__div_sqrt_arb__DOT____Vxrand_h8fdbc47e__0[1U];
        __Vtemp_11[2U] = vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__div_sqrt_arb__DOT____Vxrand_h8fdbc47e__0[2U];
        __Vtemp_11[3U] = vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__div_sqrt_arb__DOT____Vxrand_h8fdbc47e__0[3U];
        __Vtemp_11[4U] = vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__div_sqrt_arb__DOT____Vxrand_h8fdbc47e__0[4U];
    }
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT____Vcellout__div_sqrt_arb__data_out[0U] 
        = __Vtemp_11[0U];
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT____Vcellout__div_sqrt_arb__data_out[1U] 
        = __Vtemp_11[1U];
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT____Vcellout__div_sqrt_arb__data_out[2U] 
        = __Vtemp_11[2U];
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT____Vcellout__div_sqrt_arb__data_out[3U] 
        = __Vtemp_11[3U];
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT____Vcellout__div_sqrt_arb__data_out[4U] 
        = (0x7fU & __Vtemp_11[4U]);
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__fdiv_ready 
        = (1U & ((~ (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__shift_reg__DOT__genblk1__DOT__entries[4U] 
                     >> 6U)) | ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT____Vcellinp__div_sqrt_arb__ready_out) 
                                & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__div_sqrt_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot))));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__fsqrt_ready 
        = (1U & ((~ (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__shift_reg__DOT__genblk1__DOT__entries[4U] 
                     >> 6U)) | ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT____Vcellinp__div_sqrt_arb__ready_out) 
                                & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__div_sqrt_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                   >> 1U))));
    __Vtemp_25[0U] = (0x40U | ((0xffffff80U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fma__DOT__shift_reg__DOT__genblk1__DOT__entries[0U] 
                                               << 2U)) 
                               | ((0x3eU & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fma__DOT__shift_reg__DOT__genblk1__DOT__entries[0U] 
                                            << 1U)) 
                                  | (1U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fma__DOT__shift_reg__DOT__genblk1__DOT__entries[4U] 
                                           >> 5U)))));
    __Vtemp_25[1U] = (((0x7cU & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fma__DOT__shift_reg__DOT__genblk1__DOT__entries[1U] 
                                 << 2U)) | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fma__DOT__shift_reg__DOT__genblk1__DOT__entries[0U] 
                                            >> 0x1eU)) 
                      | (0xffffff80U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fma__DOT__shift_reg__DOT__genblk1__DOT__entries[1U] 
                                        << 2U)));
    __Vtemp_25[2U] = (((0x7cU & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fma__DOT__shift_reg__DOT__genblk1__DOT__entries[2U] 
                                 << 2U)) | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fma__DOT__shift_reg__DOT__genblk1__DOT__entries[1U] 
                                            >> 0x1eU)) 
                      | (0xffffff80U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fma__DOT__shift_reg__DOT__genblk1__DOT__entries[2U] 
                                        << 2U)));
    __Vtemp_25[3U] = (((0x7cU & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fma__DOT__shift_reg__DOT__genblk1__DOT__entries[3U] 
                                 << 2U)) | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fma__DOT__shift_reg__DOT__genblk1__DOT__entries[2U] 
                                            >> 0x1eU)) 
                      | (0xffffff80U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fma__DOT__shift_reg__DOT__genblk1__DOT__entries[3U] 
                                        << 2U)));
    __Vtemp_25[4U] = ((vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT____Vcellout__div_sqrt_arb__data_out[0U] 
                       << 7U) | ((0x7cU & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fma__DOT__shift_reg__DOT__genblk1__DOT__entries[4U] 
                                           << 2U)) 
                                 | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fma__DOT__shift_reg__DOT__genblk1__DOT__entries[3U] 
                                    >> 0x1eU)));
    __Vtemp_25[5U] = ((vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT____Vcellout__div_sqrt_arb__data_out[0U] 
                       >> 0x19U) | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT____Vcellout__div_sqrt_arb__data_out[1U] 
                                    << 7U));
    __Vtemp_25[6U] = ((vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT____Vcellout__div_sqrt_arb__data_out[1U] 
                       >> 0x19U) | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT____Vcellout__div_sqrt_arb__data_out[2U] 
                                    << 7U));
    __Vtemp_25[7U] = ((vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT____Vcellout__div_sqrt_arb__data_out[2U] 
                       >> 0x19U) | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT____Vcellout__div_sqrt_arb__data_out[3U] 
                                    << 7U));
    __Vtemp_25[8U] = (0x100000U | ((0xffe00000U & (
                                                   vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fcvt__DOT__shift_reg__DOT__genblk1__DOT__entries[0U] 
                                                   << 0x10U)) 
                                   | ((0xf8000U & (
                                                   vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fcvt__DOT__shift_reg__DOT__genblk1__DOT__entries[0U] 
                                                   << 0xfU)) 
                                      | ((0x4000U & 
                                          (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fcvt__DOT__shift_reg__DOT__genblk1__DOT__entries[4U] 
                                           << 9U)) 
                                         | ((vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT____Vcellout__div_sqrt_arb__data_out[3U] 
                                             >> 0x19U) 
                                            | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT____Vcellout__div_sqrt_arb__data_out[4U] 
                                               << 7U))))));
    __Vtemp_25[9U] = (((0x1f0000U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fcvt__DOT__shift_reg__DOT__genblk1__DOT__entries[1U] 
                                     << 0x10U)) | (
                                                   vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fcvt__DOT__shift_reg__DOT__genblk1__DOT__entries[0U] 
                                                   >> 0x10U)) 
                      | (0xffe00000U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fcvt__DOT__shift_reg__DOT__genblk1__DOT__entries[1U] 
                                        << 0x10U)));
    __Vtemp_25[0xaU] = (((0x1f0000U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fcvt__DOT__shift_reg__DOT__genblk1__DOT__entries[2U] 
                                       << 0x10U)) | 
                         (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fcvt__DOT__shift_reg__DOT__genblk1__DOT__entries[1U] 
                          >> 0x10U)) | (0xffe00000U 
                                        & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fcvt__DOT__shift_reg__DOT__genblk1__DOT__entries[2U] 
                                           << 0x10U)));
    __Vtemp_25[0xbU] = (((0x1f0000U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fcvt__DOT__shift_reg__DOT__genblk1__DOT__entries[3U] 
                                       << 0x10U)) | 
                         (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fcvt__DOT__shift_reg__DOT__genblk1__DOT__entries[2U] 
                          >> 0x10U)) | (0xffe00000U 
                                        & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fcvt__DOT__shift_reg__DOT__genblk1__DOT__entries[3U] 
                                           << 0x10U)));
    __Vtemp_25[0xcU] = ((0xf0000000U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fncp__DOT__shift_reg__DOT__genblk1__DOT__entries[0U] 
                                        << 0x17U)) 
                        | ((0x8000000U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fncp__DOT__shift_reg__DOT__genblk1__DOT__entries[4U] 
                                          << 0x16U)) 
                           | ((0x7c00000U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fncp__DOT__shift_reg__DOT__genblk1__DOT__entries[0U] 
                                             << 0x16U)) 
                              | ((0x200000U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fncp__DOT__shift_reg__DOT__genblk1__DOT__entries[4U] 
                                               << 0xfU)) 
                                 | ((0x1f0000U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fcvt__DOT__shift_reg__DOT__genblk1__DOT__entries[4U] 
                                                  << 0x10U)) 
                                    | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fcvt__DOT__shift_reg__DOT__genblk1__DOT__entries[3U] 
                                       >> 0x10U))))));
    __Vtemp_25[0xdU] = (((0xf800000U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fncp__DOT__shift_reg__DOT__genblk1__DOT__entries[1U] 
                                        << 0x17U)) 
                         | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fncp__DOT__shift_reg__DOT__genblk1__DOT__entries[0U] 
                            >> 9U)) | (0xf0000000U 
                                       & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fncp__DOT__shift_reg__DOT__genblk1__DOT__entries[1U] 
                                          << 0x17U)));
    __Vtemp_25[0xeU] = (((0xf800000U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fncp__DOT__shift_reg__DOT__genblk1__DOT__entries[2U] 
                                        << 0x17U)) 
                         | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fncp__DOT__shift_reg__DOT__genblk1__DOT__entries[1U] 
                            >> 9U)) | (0xf0000000U 
                                       & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fncp__DOT__shift_reg__DOT__genblk1__DOT__entries[2U] 
                                          << 0x17U)));
    __Vtemp_25[0xfU] = (((0xf800000U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fncp__DOT__shift_reg__DOT__genblk1__DOT__entries[3U] 
                                        << 0x17U)) 
                         | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fncp__DOT__shift_reg__DOT__genblk1__DOT__entries[2U] 
                            >> 9U)) | (0xf0000000U 
                                       & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fncp__DOT__shift_reg__DOT__genblk1__DOT__entries[3U] 
                                          << 0x17U)));
    __Vtemp_25[0x10U] = ((0xf800000U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fncp__DOT__shift_reg__DOT__genblk1__DOT__entries[4U] 
                                        << 0x17U)) 
                         | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fncp__DOT__shift_reg__DOT__genblk1__DOT__entries[3U] 
                            >> 9U));
    if ((0x21bU >= (0x3ffU & ((IData)(0x87U) * (3U 
                                                & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__fair_arbiter__DOT__genblk1__DOT__priority_arbiter__DOT__genblk1__DOT__priority_encoder__DOT__genblk2__DOT__lzc__DOT__genblk1__DOT__find_first__DOT__d_n)))))) {
        __Vtemp_29[1U] = (((0U == (0x1fU & ((IData)(0x87U) 
                                            * (3U & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__fair_arbiter__DOT__genblk1__DOT__priority_arbiter__DOT__genblk1__DOT__priority_encoder__DOT__genblk2__DOT__lzc__DOT__genblk1__DOT__find_first__DOT__d_n)))))
                            ? 0U : (__Vtemp_25[((IData)(2U) 
                                                + (0x1fU 
                                                   & (((IData)(0x87U) 
                                                       * 
                                                       (3U 
                                                        & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__fair_arbiter__DOT__genblk1__DOT__priority_arbiter__DOT__genblk1__DOT__priority_encoder__DOT__genblk2__DOT__lzc__DOT__genblk1__DOT__find_first__DOT__d_n))) 
                                                      >> 5U)))] 
                                    << ((IData)(0x20U) 
                                        - (0x1fU & 
                                           ((IData)(0x87U) 
                                            * (3U & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__fair_arbiter__DOT__genblk1__DOT__priority_arbiter__DOT__genblk1__DOT__priority_encoder__DOT__genblk2__DOT__lzc__DOT__genblk1__DOT__find_first__DOT__d_n))))))) 
                          | (__Vtemp_25[((IData)(1U) 
                                         + (0x1fU & 
                                            (((IData)(0x87U) 
                                              * (3U 
                                                 & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__fair_arbiter__DOT__genblk1__DOT__priority_arbiter__DOT__genblk1__DOT__priority_encoder__DOT__genblk2__DOT__lzc__DOT__genblk1__DOT__find_first__DOT__d_n))) 
                                             >> 5U)))] 
                             >> (0x1fU & ((IData)(0x87U) 
                                          * (3U & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__fair_arbiter__DOT__genblk1__DOT__priority_arbiter__DOT__genblk1__DOT__priority_encoder__DOT__genblk2__DOT__lzc__DOT__genblk1__DOT__find_first__DOT__d_n))))));
        __Vtemp_29[2U] = (((0U == (0x1fU & ((IData)(0x87U) 
                                            * (3U & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__fair_arbiter__DOT__genblk1__DOT__priority_arbiter__DOT__genblk1__DOT__priority_encoder__DOT__genblk2__DOT__lzc__DOT__genblk1__DOT__find_first__DOT__d_n)))))
                            ? 0U : (__Vtemp_25[((IData)(3U) 
                                                + (0x1fU 
                                                   & (((IData)(0x87U) 
                                                       * 
                                                       (3U 
                                                        & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__fair_arbiter__DOT__genblk1__DOT__priority_arbiter__DOT__genblk1__DOT__priority_encoder__DOT__genblk2__DOT__lzc__DOT__genblk1__DOT__find_first__DOT__d_n))) 
                                                      >> 5U)))] 
                                    << ((IData)(0x20U) 
                                        - (0x1fU & 
                                           ((IData)(0x87U) 
                                            * (3U & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__fair_arbiter__DOT__genblk1__DOT__priority_arbiter__DOT__genblk1__DOT__priority_encoder__DOT__genblk2__DOT__lzc__DOT__genblk1__DOT__find_first__DOT__d_n))))))) 
                          | (__Vtemp_25[((IData)(2U) 
                                         + (0x1fU & 
                                            (((IData)(0x87U) 
                                              * (3U 
                                                 & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__fair_arbiter__DOT__genblk1__DOT__priority_arbiter__DOT__genblk1__DOT__priority_encoder__DOT__genblk2__DOT__lzc__DOT__genblk1__DOT__find_first__DOT__d_n))) 
                                             >> 5U)))] 
                             >> (0x1fU & ((IData)(0x87U) 
                                          * (3U & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__fair_arbiter__DOT__genblk1__DOT__priority_arbiter__DOT__genblk1__DOT__priority_encoder__DOT__genblk2__DOT__lzc__DOT__genblk1__DOT__find_first__DOT__d_n))))));
        __Vtemp_29[3U] = (((0U == (0x1fU & ((IData)(0x87U) 
                                            * (3U & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__fair_arbiter__DOT__genblk1__DOT__priority_arbiter__DOT__genblk1__DOT__priority_encoder__DOT__genblk2__DOT__lzc__DOT__genblk1__DOT__find_first__DOT__d_n)))))
                            ? 0U : (__Vtemp_25[((IData)(4U) 
                                                + (0x1fU 
                                                   & (((IData)(0x87U) 
                                                       * 
                                                       (3U 
                                                        & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__fair_arbiter__DOT__genblk1__DOT__priority_arbiter__DOT__genblk1__DOT__priority_encoder__DOT__genblk2__DOT__lzc__DOT__genblk1__DOT__find_first__DOT__d_n))) 
                                                      >> 5U)))] 
                                    << ((IData)(0x20U) 
                                        - (0x1fU & 
                                           ((IData)(0x87U) 
                                            * (3U & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__fair_arbiter__DOT__genblk1__DOT__priority_arbiter__DOT__genblk1__DOT__priority_encoder__DOT__genblk2__DOT__lzc__DOT__genblk1__DOT__find_first__DOT__d_n))))))) 
                          | (__Vtemp_25[((IData)(3U) 
                                         + (0x1fU & 
                                            (((IData)(0x87U) 
                                              * (3U 
                                                 & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__fair_arbiter__DOT__genblk1__DOT__priority_arbiter__DOT__genblk1__DOT__priority_encoder__DOT__genblk2__DOT__lzc__DOT__genblk1__DOT__find_first__DOT__d_n))) 
                                             >> 5U)))] 
                             >> (0x1fU & ((IData)(0x87U) 
                                          * (3U & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__fair_arbiter__DOT__genblk1__DOT__priority_arbiter__DOT__genblk1__DOT__priority_encoder__DOT__genblk2__DOT__lzc__DOT__genblk1__DOT__find_first__DOT__d_n))))));
        __Vtemp_29[4U] = (0x7fU & (((0U == (0x1fU & 
                                            ((IData)(0x87U) 
                                             * (3U 
                                                & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__fair_arbiter__DOT__genblk1__DOT__priority_arbiter__DOT__genblk1__DOT__priority_encoder__DOT__genblk2__DOT__lzc__DOT__genblk1__DOT__find_first__DOT__d_n)))))
                                     ? 0U : (__Vtemp_25[
                                             ((IData)(5U) 
                                              + (0x1fU 
                                                 & (((IData)(0x87U) 
                                                     * 
                                                     (3U 
                                                      & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__fair_arbiter__DOT__genblk1__DOT__priority_arbiter__DOT__genblk1__DOT__priority_encoder__DOT__genblk2__DOT__lzc__DOT__genblk1__DOT__find_first__DOT__d_n))) 
                                                    >> 5U)))] 
                                             << ((IData)(0x20U) 
                                                 - 
                                                 (0x1fU 
                                                  & ((IData)(0x87U) 
                                                     * 
                                                     (3U 
                                                      & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__fair_arbiter__DOT__genblk1__DOT__priority_arbiter__DOT__genblk1__DOT__priority_encoder__DOT__genblk2__DOT__lzc__DOT__genblk1__DOT__find_first__DOT__d_n))))))) 
                                   | (__Vtemp_25[((IData)(4U) 
                                                  + 
                                                  (0x1fU 
                                                   & (((IData)(0x87U) 
                                                       * 
                                                       (3U 
                                                        & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__fair_arbiter__DOT__genblk1__DOT__priority_arbiter__DOT__genblk1__DOT__priority_encoder__DOT__genblk2__DOT__lzc__DOT__genblk1__DOT__find_first__DOT__d_n))) 
                                                      >> 5U)))] 
                                      >> (0x1fU & ((IData)(0x87U) 
                                                   * 
                                                   (3U 
                                                    & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__fair_arbiter__DOT__genblk1__DOT__priority_arbiter__DOT__genblk1__DOT__priority_encoder__DOT__genblk2__DOT__lzc__DOT__genblk1__DOT__find_first__DOT__d_n)))))));
        vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__rsp_arb__DOT____Vcellinp__genblk1__DOT__genblk1__DOT__out_buf__data_in[0U] 
            = (((0U == (0x1fU & ((IData)(0x87U) * (3U 
                                                   & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__fair_arbiter__DOT__genblk1__DOT__priority_arbiter__DOT__genblk1__DOT__priority_encoder__DOT__genblk2__DOT__lzc__DOT__genblk1__DOT__find_first__DOT__d_n)))))
                 ? 0U : (__Vtemp_25[((IData)(1U) + 
                                     (0x1fU & (((IData)(0x87U) 
                                                * (3U 
                                                   & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__fair_arbiter__DOT__genblk1__DOT__priority_arbiter__DOT__genblk1__DOT__priority_encoder__DOT__genblk2__DOT__lzc__DOT__genblk1__DOT__find_first__DOT__d_n))) 
                                               >> 5U)))] 
                         << ((IData)(0x20U) - (0x1fU 
                                               & ((IData)(0x87U) 
                                                  * 
                                                  (3U 
                                                   & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__fair_arbiter__DOT__genblk1__DOT__priority_arbiter__DOT__genblk1__DOT__priority_encoder__DOT__genblk2__DOT__lzc__DOT__genblk1__DOT__find_first__DOT__d_n))))))) 
               | (__Vtemp_25[(0x1fU & (((IData)(0x87U) 
                                        * (3U & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__fair_arbiter__DOT__genblk1__DOT__priority_arbiter__DOT__genblk1__DOT__priority_encoder__DOT__genblk2__DOT__lzc__DOT__genblk1__DOT__find_first__DOT__d_n))) 
                                       >> 5U))] >> 
                  (0x1fU & ((IData)(0x87U) * (3U & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__fair_arbiter__DOT__genblk1__DOT__priority_arbiter__DOT__genblk1__DOT__priority_encoder__DOT__genblk2__DOT__lzc__DOT__genblk1__DOT__find_first__DOT__d_n))))));
    } else {
        __Vtemp_29[1U] = vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__rsp_arb__DOT____Vxrand_h8fdbc47e__0[1U];
        __Vtemp_29[2U] = vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__rsp_arb__DOT____Vxrand_h8fdbc47e__0[2U];
        __Vtemp_29[3U] = vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__rsp_arb__DOT____Vxrand_h8fdbc47e__0[3U];
        __Vtemp_29[4U] = (0x7fU & vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__rsp_arb__DOT____Vxrand_h8fdbc47e__0[4U]);
        vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__rsp_arb__DOT____Vcellinp__genblk1__DOT__genblk1__DOT__out_buf__data_in[0U] 
            = vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__rsp_arb__DOT____Vxrand_h8fdbc47e__0[0U];
    }
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__rsp_arb__DOT____Vcellinp__genblk1__DOT__genblk1__DOT__out_buf__data_in[1U] 
        = __Vtemp_29[1U];
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__rsp_arb__DOT____Vcellinp__genblk1__DOT__genblk1__DOT__out_buf__data_in[2U] 
        = __Vtemp_29[2U];
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__rsp_arb__DOT____Vcellinp__genblk1__DOT__genblk1__DOT__out_buf__data_in[3U] 
        = __Vtemp_29[3U];
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__rsp_arb__DOT____Vcellinp__genblk1__DOT__genblk1__DOT__out_buf__data_in[4U] 
        = ((0x180U & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__fair_arbiter__DOT__genblk1__DOT__priority_arbiter__DOT__genblk1__DOT__priority_encoder__DOT__genblk2__DOT__lzc__DOT__genblk1__DOT__find_first__DOT__d_n) 
                      << 7U)) | __Vtemp_29[4U]);
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__fdiv_fire 
        = ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__fdiv_valid) 
           & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__fdiv_ready));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__fsqrt_fire 
        = ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__fsqrt_valid) 
           & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__fsqrt_ready));
    vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__per_block_execute_if__BRA__0__KET__.ready 
        = (1U & (((((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fncp__DOT__fncp_ready) 
                    << 3U) | (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fcvt__DOT__fcvt_ready) 
                               << 2U) | ((((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__is_div)
                                            ? (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__fdiv_ready)
                                            : (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__fsqrt_ready)) 
                                          << 1U) | (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fma__DOT__fma_ready)))) 
                  >> (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__core_select)) 
                 & (~ (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__tag_store__DOT__allocator__DOT__full_r))));
    VVortex___024unit____Vdpiimwrap_dpi_fdiv_TOP____024unit(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__fdiv_fire, (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__dst_fmt), 
                                                            (((QData)((IData)(
                                                                              vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__operands
                                                                              [0U][1U])) 
                                                              << 0x20U) 
                                                             | (QData)((IData)(
                                                                               vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__operands
                                                                               [0U][0U]))), 
                                                            (((QData)((IData)(
                                                                              vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__operands
                                                                              [1U][1U])) 
                                                              << 0x20U) 
                                                             | (QData)((IData)(
                                                                               vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__operands
                                                                               [1U][0U]))), (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_req_frm), vlSelf->__Vtask_dpi_fdiv__20__result, vlSelf->__Vtask_dpi_fdiv__20__fflags);
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__result_fdiv[0U] 
        = (IData)(vlSelf->__Vtask_dpi_fdiv__20__result);
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__result_fdiv[1U] 
        = (IData)((vlSelf->__Vtask_dpi_fdiv__20__result 
                   >> 0x20U));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT____Vlvbound_ha9bec532__0 
        = vlSelf->__Vtask_dpi_fdiv__20__fflags;
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__fflags_fdiv 
        = ((0xfffe0U & vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__fflags_fdiv) 
           | (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT____Vlvbound_ha9bec532__0));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__result_fdiv_r[0U] 
        = vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__result_fdiv[0U];
    VVortex___024unit____Vdpiimwrap_dpi_fdiv_TOP____024unit(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__fdiv_fire, (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__dst_fmt), 
                                                            (((QData)((IData)(
                                                                              vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__operands
                                                                              [0U][3U])) 
                                                              << 0x20U) 
                                                             | (QData)((IData)(
                                                                               vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__operands
                                                                               [0U][2U]))), 
                                                            (((QData)((IData)(
                                                                              vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__operands
                                                                              [1U][3U])) 
                                                              << 0x20U) 
                                                             | (QData)((IData)(
                                                                               vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__operands
                                                                               [1U][2U]))), (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_req_frm), vlSelf->__Vtask_dpi_fdiv__20__result, vlSelf->__Vtask_dpi_fdiv__20__fflags);
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__result_fdiv[2U] 
        = (IData)(vlSelf->__Vtask_dpi_fdiv__20__result);
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__result_fdiv[3U] 
        = (IData)((vlSelf->__Vtask_dpi_fdiv__20__result 
                   >> 0x20U));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT____Vlvbound_ha9bec532__0 
        = vlSelf->__Vtask_dpi_fdiv__20__fflags;
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__fflags_fdiv 
        = ((0xffc1fU & vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__fflags_fdiv) 
           | ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT____Vlvbound_ha9bec532__0) 
              << 5U));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__result_fdiv_r[1U] 
        = vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__result_fdiv[2U];
    VVortex___024unit____Vdpiimwrap_dpi_fdiv_TOP____024unit(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__fdiv_fire, (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__dst_fmt), 
                                                            (((QData)((IData)(
                                                                              vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__operands
                                                                              [0U][5U])) 
                                                              << 0x20U) 
                                                             | (QData)((IData)(
                                                                               vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__operands
                                                                               [0U][4U]))), 
                                                            (((QData)((IData)(
                                                                              vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__operands
                                                                              [1U][5U])) 
                                                              << 0x20U) 
                                                             | (QData)((IData)(
                                                                               vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__operands
                                                                               [1U][4U]))), (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_req_frm), vlSelf->__Vtask_dpi_fdiv__20__result, vlSelf->__Vtask_dpi_fdiv__20__fflags);
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__result_fdiv[4U] 
        = (IData)(vlSelf->__Vtask_dpi_fdiv__20__result);
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__result_fdiv[5U] 
        = (IData)((vlSelf->__Vtask_dpi_fdiv__20__result 
                   >> 0x20U));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT____Vlvbound_ha9bec532__0 
        = vlSelf->__Vtask_dpi_fdiv__20__fflags;
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__fflags_fdiv 
        = ((0xf83ffU & vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__fflags_fdiv) 
           | ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT____Vlvbound_ha9bec532__0) 
              << 0xaU));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__result_fdiv_r[2U] 
        = vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__result_fdiv[4U];
    VVortex___024unit____Vdpiimwrap_dpi_fdiv_TOP____024unit(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__fdiv_fire, (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__dst_fmt), 
                                                            (((QData)((IData)(
                                                                              vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__operands
                                                                              [0U][7U])) 
                                                              << 0x20U) 
                                                             | (QData)((IData)(
                                                                               vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__operands
                                                                               [0U][6U]))), 
                                                            (((QData)((IData)(
                                                                              vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__operands
                                                                              [1U][7U])) 
                                                              << 0x20U) 
                                                             | (QData)((IData)(
                                                                               vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__operands
                                                                               [1U][6U]))), (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_req_frm), vlSelf->__Vtask_dpi_fdiv__20__result, vlSelf->__Vtask_dpi_fdiv__20__fflags);
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__result_fdiv[6U] 
        = (IData)(vlSelf->__Vtask_dpi_fdiv__20__result);
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__result_fdiv[7U] 
        = (IData)((vlSelf->__Vtask_dpi_fdiv__20__result 
                   >> 0x20U));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT____Vlvbound_ha9bec532__0 
        = vlSelf->__Vtask_dpi_fdiv__20__fflags;
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__fflags_fdiv 
        = ((0x7fffU & vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__fflags_fdiv) 
           | ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT____Vlvbound_ha9bec532__0) 
              << 0xfU));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__result_fdiv_r[3U] 
        = vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__result_fdiv[6U];
    VVortex___024unit____Vdpiimwrap_dpi_fsqrt_TOP____024unit(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__fsqrt_fire, (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__dst_fmt), 
                                                             (((QData)((IData)(
                                                                               vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__operands
                                                                               [0U][1U])) 
                                                               << 0x20U) 
                                                              | (QData)((IData)(
                                                                                vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__operands
                                                                                [0U][0U]))), (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_req_frm), vlSelf->__Vtask_dpi_fsqrt__21__result, vlSelf->__Vtask_dpi_fsqrt__21__fflags);
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__result_fsqrt[0U] 
        = (IData)(vlSelf->__Vtask_dpi_fsqrt__21__result);
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__result_fsqrt[1U] 
        = (IData)((vlSelf->__Vtask_dpi_fsqrt__21__result 
                   >> 0x20U));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT____Vlvbound_hb0d77104__0 
        = vlSelf->__Vtask_dpi_fsqrt__21__fflags;
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__fflags_fsqrt 
        = ((0xfffe0U & vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__fflags_fsqrt) 
           | (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT____Vlvbound_hb0d77104__0));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__result_fsqrt_r[0U] 
        = vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__result_fsqrt[0U];
    VVortex___024unit____Vdpiimwrap_dpi_fsqrt_TOP____024unit(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__fsqrt_fire, (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__dst_fmt), 
                                                             (((QData)((IData)(
                                                                               vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__operands
                                                                               [0U][3U])) 
                                                               << 0x20U) 
                                                              | (QData)((IData)(
                                                                                vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__operands
                                                                                [0U][2U]))), (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_req_frm), vlSelf->__Vtask_dpi_fsqrt__21__result, vlSelf->__Vtask_dpi_fsqrt__21__fflags);
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__result_fsqrt[2U] 
        = (IData)(vlSelf->__Vtask_dpi_fsqrt__21__result);
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__result_fsqrt[3U] 
        = (IData)((vlSelf->__Vtask_dpi_fsqrt__21__result 
                   >> 0x20U));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT____Vlvbound_hb0d77104__0 
        = vlSelf->__Vtask_dpi_fsqrt__21__fflags;
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__fflags_fsqrt 
        = ((0xffc1fU & vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__fflags_fsqrt) 
           | ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT____Vlvbound_hb0d77104__0) 
              << 5U));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__result_fsqrt_r[1U] 
        = vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__result_fsqrt[2U];
    VVortex___024unit____Vdpiimwrap_dpi_fsqrt_TOP____024unit(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__fsqrt_fire, (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__dst_fmt), 
                                                             (((QData)((IData)(
                                                                               vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__operands
                                                                               [0U][5U])) 
                                                               << 0x20U) 
                                                              | (QData)((IData)(
                                                                                vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__operands
                                                                                [0U][4U]))), (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_req_frm), vlSelf->__Vtask_dpi_fsqrt__21__result, vlSelf->__Vtask_dpi_fsqrt__21__fflags);
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__result_fsqrt[4U] 
        = (IData)(vlSelf->__Vtask_dpi_fsqrt__21__result);
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__result_fsqrt[5U] 
        = (IData)((vlSelf->__Vtask_dpi_fsqrt__21__result 
                   >> 0x20U));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT____Vlvbound_hb0d77104__0 
        = vlSelf->__Vtask_dpi_fsqrt__21__fflags;
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__fflags_fsqrt 
        = ((0xf83ffU & vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__fflags_fsqrt) 
           | ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT____Vlvbound_hb0d77104__0) 
              << 0xaU));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__result_fsqrt_r[2U] 
        = vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__result_fsqrt[4U];
    VVortex___024unit____Vdpiimwrap_dpi_fsqrt_TOP____024unit(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__fsqrt_fire, (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__dst_fmt), 
                                                             (((QData)((IData)(
                                                                               vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__operands
                                                                               [0U][7U])) 
                                                               << 0x20U) 
                                                              | (QData)((IData)(
                                                                                vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__operands
                                                                                [0U][6U]))), (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_req_frm), vlSelf->__Vtask_dpi_fsqrt__21__result, vlSelf->__Vtask_dpi_fsqrt__21__fflags);
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__result_fsqrt[6U] 
        = (IData)(vlSelf->__Vtask_dpi_fsqrt__21__result);
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__result_fsqrt[7U] 
        = (IData)((vlSelf->__Vtask_dpi_fsqrt__21__result 
                   >> 0x20U));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT____Vlvbound_hb0d77104__0 
        = vlSelf->__Vtask_dpi_fsqrt__21__fflags;
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__fflags_fsqrt 
        = ((0x7fffU & vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__fflags_fsqrt) 
           | ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT____Vlvbound_hb0d77104__0) 
              << 0xfU));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__result_fsqrt_r[3U] 
        = vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__result_fsqrt[6U];
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__execute_fire 
        = ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__dispatch__DOT__genblk2__BRA__3__KET____DOT__buffer__DOT__genblk1__DOT__out_buf__DOT__genblk1__DOT__genblk1__BRA__0__KET____DOT__pipe_register__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_r) 
           & (IData)(vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__per_block_execute_if__BRA__0__KET__.ready));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__dispatch_unit__DOT____Vlvbound_h3b4d06f5__0 
        = vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__per_block_execute_if__BRA__0__KET__.ready;
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__dispatch_unit__DOT__ready_in 
        = vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__dispatch_unit__DOT____Vlvbound_h3b4d06f5__0;
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged = 0U;
    if ((0x100000U & vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__dispatch_unit__DOT____Vcellinp__genblk5__BRA__0__KET____DOT__buf_out__data_in[0xeU])) {
        vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged 
            = ((0x1eU & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged)) 
               | (1U & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged) 
                        | vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__fflags_fdiv)));
        vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged 
            = ((0x1dU & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged)) 
               | (2U & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged) 
                        | vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__fflags_fdiv)));
        vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged 
            = ((0x1bU & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged)) 
               | (4U & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged) 
                        | vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__fflags_fdiv)));
        vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged 
            = ((0x17U & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged)) 
               | (8U & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged) 
                        | vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__fflags_fdiv)));
        vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged 
            = ((0xfU & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged)) 
               | (0x10U & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged) 
                           | vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__fflags_fdiv)));
    }
    if ((0x200000U & vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__dispatch_unit__DOT____Vcellinp__genblk5__BRA__0__KET____DOT__buf_out__data_in[0xeU])) {
        vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged 
            = ((0x1eU & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged)) 
               | (1U & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged) 
                        | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__fflags_fdiv 
                           >> 5U))));
        vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged 
            = ((0x1dU & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged)) 
               | (2U & ((0xfffffffeU & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged)) 
                        | (0x7fffffeU & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__fflags_fdiv 
                                         >> 5U)))));
        vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged 
            = ((0x1bU & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged)) 
               | (4U & ((0xfffffffcU & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged)) 
                        | (0x7fffffcU & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__fflags_fdiv 
                                         >> 5U)))));
        vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged 
            = ((0x17U & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged)) 
               | (8U & ((0xfffffff8U & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged)) 
                        | (0x7fffff8U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__fflags_fdiv 
                                         >> 5U)))));
        vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged 
            = ((0xfU & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged)) 
               | (0x10U & ((0xfffffff0U & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged)) 
                           | (0x7fffff0U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__fflags_fdiv 
                                            >> 5U)))));
    }
    if ((0x400000U & vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__dispatch_unit__DOT____Vcellinp__genblk5__BRA__0__KET____DOT__buf_out__data_in[0xeU])) {
        vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged 
            = ((0x1eU & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged)) 
               | (1U & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged) 
                        | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__fflags_fdiv 
                           >> 0xaU))));
        vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged 
            = ((0x1dU & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged)) 
               | (2U & ((0xfffffffeU & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged)) 
                        | (0x3ffffeU & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__fflags_fdiv 
                                        >> 0xaU)))));
        vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged 
            = ((0x1bU & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged)) 
               | (4U & ((0xfffffffcU & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged)) 
                        | (0x3ffffcU & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__fflags_fdiv 
                                        >> 0xaU)))));
        vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged 
            = ((0x17U & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged)) 
               | (8U & ((0xfffffff8U & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged)) 
                        | (0x3ffff8U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__fflags_fdiv 
                                        >> 0xaU)))));
        vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged 
            = ((0xfU & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged)) 
               | (0x10U & ((0xfffffff0U & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged)) 
                           | (0x3ffff0U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__fflags_fdiv 
                                           >> 0xaU)))));
    }
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged = 0U;
    if ((0x100000U & vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__dispatch_unit__DOT____Vcellinp__genblk5__BRA__0__KET____DOT__buf_out__data_in[0xeU])) {
        vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged 
            = ((0x1eU & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged)) 
               | (1U & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged) 
                        | vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__fflags_fsqrt)));
        vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged 
            = ((0x1dU & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged)) 
               | (2U & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged) 
                        | vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__fflags_fsqrt)));
        vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged 
            = ((0x1bU & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged)) 
               | (4U & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged) 
                        | vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__fflags_fsqrt)));
        vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged 
            = ((0x17U & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged)) 
               | (8U & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged) 
                        | vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__fflags_fsqrt)));
        vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged 
            = ((0xfU & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged)) 
               | (0x10U & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged) 
                           | vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__fflags_fsqrt)));
    }
    if ((0x200000U & vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__dispatch_unit__DOT____Vcellinp__genblk5__BRA__0__KET____DOT__buf_out__data_in[0xeU])) {
        vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged 
            = ((0x1eU & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged)) 
               | (1U & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged) 
                        | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__fflags_fsqrt 
                           >> 5U))));
        vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged 
            = ((0x1dU & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged)) 
               | (2U & ((0xfffffffeU & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged)) 
                        | (0x7fffffeU & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__fflags_fsqrt 
                                         >> 5U)))));
        vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged 
            = ((0x1bU & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged)) 
               | (4U & ((0xfffffffcU & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged)) 
                        | (0x7fffffcU & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__fflags_fsqrt 
                                         >> 5U)))));
        vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged 
            = ((0x17U & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged)) 
               | (8U & ((0xfffffff8U & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged)) 
                        | (0x7fffff8U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__fflags_fsqrt 
                                         >> 5U)))));
        vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged 
            = ((0xfU & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged)) 
               | (0x10U & ((0xfffffff0U & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged)) 
                           | (0x7fffff0U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__fflags_fsqrt 
                                            >> 5U)))));
    }
    if ((0x400000U & vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__dispatch_unit__DOT____Vcellinp__genblk5__BRA__0__KET____DOT__buf_out__data_in[0xeU])) {
        vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged 
            = ((0x1eU & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged)) 
               | (1U & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged) 
                        | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__fflags_fsqrt 
                           >> 0xaU))));
        vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged 
            = ((0x1dU & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged)) 
               | (2U & ((0xfffffffeU & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged)) 
                        | (0x3ffffeU & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__fflags_fsqrt 
                                        >> 0xaU)))));
        vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged 
            = ((0x1bU & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged)) 
               | (4U & ((0xfffffffcU & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged)) 
                        | (0x3ffffcU & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__fflags_fsqrt 
                                        >> 0xaU)))));
        vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged 
            = ((0x17U & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged)) 
               | (8U & ((0xfffffff8U & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged)) 
                        | (0x3ffff8U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__fflags_fsqrt 
                                        >> 0xaU)))));
        vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged 
            = ((0xfU & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged)) 
               | (0x10U & ((0xfffffff0U & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged)) 
                           | (0x3ffff0U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__fflags_fsqrt 
                                           >> 0xaU)))));
    }
    if ((0x800000U & vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__dispatch_unit__DOT____Vcellinp__genblk5__BRA__0__KET____DOT__buf_out__data_in[0xeU])) {
        vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged 
            = ((0x1eU & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged)) 
               | (1U & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged) 
                        | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__fflags_fdiv 
                           >> 0xfU))));
        vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged 
            = ((0x1dU & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged)) 
               | (2U & ((0xfffffffeU & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged)) 
                        | (0x1fffeU & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__fflags_fdiv 
                                       >> 0xfU)))));
        vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged 
            = ((0x1bU & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged)) 
               | (4U & ((0xfffffffcU & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged)) 
                        | (0x1fffcU & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__fflags_fdiv 
                                       >> 0xfU)))));
        vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged 
            = ((0x17U & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged)) 
               | (8U & ((0xfffffff8U & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged)) 
                        | (0x1fff8U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__fflags_fdiv 
                                       >> 0xfU)))));
        vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged 
            = ((0xfU & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged)) 
               | ((IData)((((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT_____05Ffflags_merged) 
                            >> 4U) | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fdiv__DOT__fflags_fdiv 
                                      >> 0x13U))) << 4U));
        vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged 
            = ((0x1eU & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged)) 
               | (1U & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged) 
                        | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__fflags_fsqrt 
                           >> 0xfU))));
        vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged 
            = ((0x1dU & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged)) 
               | (2U & ((0xfffffffeU & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged)) 
                        | (0x1fffeU & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__fflags_fsqrt 
                                       >> 0xfU)))));
        vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged 
            = ((0x1bU & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged)) 
               | (4U & ((0xfffffffcU & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged)) 
                        | (0x1fffcU & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__fflags_fsqrt 
                                       >> 0xfU)))));
        vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged 
            = ((0x17U & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged)) 
               | (8U & ((0xfffffff8U & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged)) 
                        | (0x1fff8U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__fflags_fsqrt 
                                       >> 0xfU)))));
        vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged 
            = ((0xfU & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged)) 
               | ((IData)((((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT_____05Ffflags_merged) 
                            >> 4U) | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__fpu_unit__DOT__genblk1__BRA__0__KET____DOT__fpu_dpi__DOT__fsqrt__DOT__fflags_fsqrt 
                                      >> 0x13U))) << 4U));
    }
}

VL_INLINE_OPT void VVortex___024root___nba_comb__TOP__26(VVortex___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    VVortex__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVortex___024root___nba_comb__TOP__26\n"); );
    // Init
    VlWide<6>/*191:0*/ __Vtemp_6;
    VlWide<7>/*223:0*/ __Vtemp_13;
    // Body
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__genblk4__BRA__0__KET____DOT__priority_encoder__DOT__genblk2__DOT__scan__DOT__t 
        = ((0xf00U & ((0xffffff00U & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__genblk4__BRA__0__KET____DOT__priority_encoder__DOT__genblk2__DOT__scan__DOT__t) 
                                      << 4U)) | ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__genblk4__BRA__0__KET____DOT__priority_encoder__DOT__genblk2__DOT__scan__DOT__genblk2__DOT__genblk1__BRA__1__KET____DOT__shifted) 
                                                 << 8U))) 
           | ((0xf0U & (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__genblk4__BRA__0__KET____DOT__priority_encoder__DOT__genblk2__DOT__scan__DOT__t) 
                         | (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__genblk4__BRA__0__KET____DOT__priority_encoder__DOT__genblk2__DOT__scan__DOT__genblk2__DOT__genblk1__BRA__0__KET____DOT__shifted)) 
                        << 4U)) | VL_STREAML_FAST_III(4, (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__genblk4__BRA__0__KET____DOT__batch_mask), 0)));
    __Vtemp_6[5U] = ((0x1c0000U & (((0xbU >= (0xfU 
                                              & ((IData)(3U) 
                                                 * 
                                                 (3U 
                                                  & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__genblk4__BRA__0__KET____DOT__priority_encoder__DOT__genblk2__DOT__lzc__DOT__genblk1__DOT__find_first__DOT__d_n)))))
                                     ? (((0U == (0x1fU 
                                                 & ((IData)(0x82U) 
                                                    + 
                                                    (0xfU 
                                                     & ((IData)(3U) 
                                                        * 
                                                        (3U 
                                                         & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__genblk4__BRA__0__KET____DOT__priority_encoder__DOT__genblk2__DOT__lzc__DOT__genblk1__DOT__find_first__DOT__d_n)))))))
                                          ? 0U : (vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lsu_dcache_if__BRA__0__KET__.req_data[
                                                  (((IData)(0x84U) 
                                                    + 
                                                    (0xfU 
                                                     & ((IData)(3U) 
                                                        * 
                                                        (3U 
                                                         & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__genblk4__BRA__0__KET____DOT__priority_encoder__DOT__genblk2__DOT__lzc__DOT__genblk1__DOT__find_first__DOT__d_n))))) 
                                                   >> 5U)] 
                                                  << 
                                                  ((IData)(0x20U) 
                                                   - 
                                                   (0x1fU 
                                                    & ((IData)(0x82U) 
                                                       + 
                                                       (0xfU 
                                                        & ((IData)(3U) 
                                                           * 
                                                           (3U 
                                                            & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__genblk4__BRA__0__KET____DOT__priority_encoder__DOT__genblk2__DOT__lzc__DOT__genblk1__DOT__find_first__DOT__d_n))))))))) 
                                        | (vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lsu_dcache_if__BRA__0__KET__.req_data[
                                           (((IData)(0x82U) 
                                             + (0xfU 
                                                & ((IData)(3U) 
                                                   * 
                                                   (3U 
                                                    & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__genblk4__BRA__0__KET____DOT__priority_encoder__DOT__genblk2__DOT__lzc__DOT__genblk1__DOT__find_first__DOT__d_n))))) 
                                            >> 5U)] 
                                           >> (0x1fU 
                                               & ((IData)(0x82U) 
                                                  + 
                                                  (0xfU 
                                                   & ((IData)(3U) 
                                                      * 
                                                      (3U 
                                                       & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__genblk4__BRA__0__KET____DOT__priority_encoder__DOT__genblk2__DOT__lzc__DOT__genblk1__DOT__find_first__DOT__d_n))))))))
                                     : (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT____Vxrand_h8df70ee3__0)) 
                                   << 0x12U)) | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__out_req_addr_n 
                                                 >> 0xaU));
    __Vtemp_13[6U] = ((((0xfffffffU & ((vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lsu_dcache_if__BRA__0__KET__.req_data[8U] 
                                        << 0x16U) | 
                                       (vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lsu_dcache_if__BRA__0__KET__.req_data[7U] 
                                        >> 0xaU))) 
                        == vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__seed_addr_n) 
                       << 0x16U) | ((((0xfffffffU & 
                                       ((vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lsu_dcache_if__BRA__0__KET__.req_data[7U] 
                                         << 0x14U) 
                                        | (vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lsu_dcache_if__BRA__0__KET__.req_data[6U] 
                                           >> 0xcU))) 
                                      == vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__seed_addr_n) 
                                     << 0x15U) | ((
                                                   ((0xfffffffU 
                                                     & ((vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lsu_dcache_if__BRA__0__KET__.req_data[6U] 
                                                         << 0x12U) 
                                                        | (vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lsu_dcache_if__BRA__0__KET__.req_data[5U] 
                                                           >> 0xeU))) 
                                                    == vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__seed_addr_n) 
                                                   << 0x14U) 
                                                  | ((((0xfffffffU 
                                                        & ((vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lsu_dcache_if__BRA__0__KET__.req_data[5U] 
                                                            << 0x10U) 
                                                           | (vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lsu_dcache_if__BRA__0__KET__.req_data[4U] 
                                                              >> 0x10U))) 
                                                       == vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__seed_addr_n) 
                                                      << 0x13U) 
                                                     | ((0x40000U 
                                                         & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__genblk4__BRA__0__KET____DOT__priority_encoder__DOT__genblk2__DOT__scan__DOT__t) 
                                                            << 0xaU)) 
                                                        | (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__out_req_mask_n) 
                                                            << 0x11U) 
                                                           | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__seed_addr_n 
                                                              >> 0xbU)))))));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT____Vcellinp__pipe_reg__data_in[0U] 
        = ((vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__out_req_data_n[0U] 
            << 3U) | (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__out_req_tag_n));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT____Vcellinp__pipe_reg__data_in[1U] 
        = ((vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__out_req_data_n[0U] 
            >> 0x1dU) | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__out_req_data_n[1U] 
                         << 3U));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT____Vcellinp__pipe_reg__data_in[2U] 
        = ((vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__out_req_data_n[1U] 
            >> 0x1dU) | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__out_req_data_n[2U] 
                         << 3U));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT____Vcellinp__pipe_reg__data_in[3U] 
        = ((vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__out_req_data_n[2U] 
            >> 0x1dU) | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__out_req_data_n[3U] 
                         << 3U));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT____Vcellinp__pipe_reg__data_in[4U] 
        = ((vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__out_req_addr_n 
            << 0x16U) | (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__out_req_atype_n) 
                          << 0x13U) | (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__out_req_byteen_n) 
                                        << 3U) | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__out_req_data_n[3U] 
                                                  >> 0x1dU))));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT____Vcellinp__pipe_reg__data_in[5U] 
        = ((vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__seed_addr_n 
            << 0x15U) | __Vtemp_6[5U]);
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT____Vcellinp__pipe_reg__data_in[6U] 
        = (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__state_n) 
            << 0x1dU) | (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__processed_mask_n) 
                          << 0x19U) | (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__out_req_valid_n) 
                                        << 0x18U) | 
                                       (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__out_req_rw_n) 
                                         << 0x17U) 
                                        | __Vtemp_13[6U]))));
}

VL_INLINE_OPT void VVortex___024root___nba_comb__TOP__55(VVortex___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    VVortex__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVortex___024root___nba_comb__TOP__55\n"); );
    // Body
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__lsu_unit__DOT__dispatch_unit__DOT__genblk5__BRA__0__KET____DOT__buf_out__DOT__genblk1__DOT__pipe_buffer__DOT__genblk1__DOT__ready 
        = (((IData)(vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__lsu_unit__DOT__per_block_execute_if__BRA__0__KET__.ready) 
            << 1U) | (1U & (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__lsu_unit__DOT__dispatch_unit__DOT__genblk5__BRA__0__KET____DOT__buf_out__DOT__genblk1__DOT__pipe_buffer__DOT__genblk1__DOT__ready) 
                             >> 1U) | (~ (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__execute__DOT__lsu_unit__DOT__dispatch_unit__DOT__genblk5__BRA__0__KET____DOT__buf_out__DOT__genblk1__DOT__pipe_buffer__DOT__genblk1__DOT__genblk1__BRA__0__KET____DOT__pipe_register__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_r)))));
}

VL_INLINE_OPT void VVortex___024root___nba_comb__TOP__86(VVortex___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    VVortex__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVortex___024root___nba_comb__TOP__86\n"); );
    // Init
    VlWide<9>/*287:0*/ __Vtemp_4;
    VlWide<5>/*159:0*/ __Vtemp_7;
    // Body
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk3__BRA__0__KET____DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot 
        = (((- (IData)((1U & (~ (IData)((0U != (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk3__BRA__0__KET____DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__rr_arbiter__DOT__genblk1__DOT__req_masked))))))) 
            & ((~ (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk3__BRA__0__KET____DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__rr_arbiter__DOT__genblk1__DOT__unmask_higher_pri_regs)) 
               & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT____Vcellinp__genblk3__BRA__0__KET____DOT__rsp_arb__valid_in))) 
           | ((~ (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk3__BRA__0__KET____DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__rr_arbiter__DOT__genblk1__DOT__mask_higher_pri_regs)) 
              & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk3__BRA__0__KET____DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__rr_arbiter__DOT__genblk1__DOT__req_masked)));
    __Vtemp_4[0U] = vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lsu_dcache_if__BRA__0__KET__.rsp_data[0U];
    __Vtemp_4[1U] = vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lsu_dcache_if__BRA__0__KET__.rsp_data[1U];
    __Vtemp_4[2U] = vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lsu_dcache_if__BRA__0__KET__.rsp_data[2U];
    __Vtemp_4[3U] = vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lsu_dcache_if__BRA__0__KET__.rsp_data[3U];
    __Vtemp_4[4U] = ((vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk4__BRA__0__KET____DOT__lsu_adapter__DOT__rsp_data_out[0U] 
                      << 8U) | (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk4__BRA__0__KET____DOT__lsu_adapter__DOT____Vcellout__stream_pack__tag_out) 
                                 << 6U) | vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lsu_dcache_if__BRA__0__KET__.rsp_data[4U]));
    __Vtemp_4[5U] = ((vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk4__BRA__0__KET____DOT__lsu_adapter__DOT__rsp_data_out[0U] 
                      >> 0x18U) | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk4__BRA__0__KET____DOT__lsu_adapter__DOT__rsp_data_out[1U] 
                                   << 8U));
    __Vtemp_4[6U] = ((vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk4__BRA__0__KET____DOT__lsu_adapter__DOT__rsp_data_out[1U] 
                      >> 0x18U) | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk4__BRA__0__KET____DOT__lsu_adapter__DOT__rsp_data_out[2U] 
                                   << 8U));
    __Vtemp_4[7U] = ((vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk4__BRA__0__KET____DOT__lsu_adapter__DOT__rsp_data_out[2U] 
                      >> 0x18U) | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk4__BRA__0__KET____DOT__lsu_adapter__DOT__rsp_data_out[3U] 
                                   << 8U));
    __Vtemp_4[8U] = (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk4__BRA__0__KET____DOT__lsu_adapter__DOT____Vcellout__stream_pack__mask_out) 
                      << 8U) | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk4__BRA__0__KET____DOT__lsu_adapter__DOT__rsp_data_out[3U] 
                                >> 0x18U));
    if ((0x10bU >= (0x1ffU & ((IData)(0x86U) * (1U 
                                                & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk3__BRA__0__KET____DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                                   >> 1U)))))) {
        __Vtemp_7[0U] = (((0U == (0x1fU & ((IData)(0x86U) 
                                           * (1U & 
                                              ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk3__BRA__0__KET____DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                               >> 1U)))))
                           ? 0U : (__Vtemp_4[((IData)(1U) 
                                              + (0xfU 
                                                 & (((IData)(0x86U) 
                                                     * 
                                                     (1U 
                                                      & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk3__BRA__0__KET____DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                                         >> 1U))) 
                                                    >> 5U)))] 
                                   << ((IData)(0x20U) 
                                       - (0x1fU & ((IData)(0x86U) 
                                                   * 
                                                   (1U 
                                                    & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk3__BRA__0__KET____DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                                       >> 1U))))))) 
                         | (__Vtemp_4[(0xfU & (((IData)(0x86U) 
                                                * (1U 
                                                   & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk3__BRA__0__KET____DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                                      >> 1U))) 
                                               >> 5U))] 
                            >> (0x1fU & ((IData)(0x86U) 
                                         * (1U & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk3__BRA__0__KET____DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                                  >> 1U))))));
        __Vtemp_7[1U] = (((0U == (0x1fU & ((IData)(0x86U) 
                                           * (1U & 
                                              ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk3__BRA__0__KET____DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                               >> 1U)))))
                           ? 0U : (__Vtemp_4[((IData)(2U) 
                                              + (0xfU 
                                                 & (((IData)(0x86U) 
                                                     * 
                                                     (1U 
                                                      & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk3__BRA__0__KET____DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                                         >> 1U))) 
                                                    >> 5U)))] 
                                   << ((IData)(0x20U) 
                                       - (0x1fU & ((IData)(0x86U) 
                                                   * 
                                                   (1U 
                                                    & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk3__BRA__0__KET____DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                                       >> 1U))))))) 
                         | (__Vtemp_4[((IData)(1U) 
                                       + (0xfU & (((IData)(0x86U) 
                                                   * 
                                                   (1U 
                                                    & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk3__BRA__0__KET____DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                                       >> 1U))) 
                                                  >> 5U)))] 
                            >> (0x1fU & ((IData)(0x86U) 
                                         * (1U & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk3__BRA__0__KET____DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                                  >> 1U))))));
        __Vtemp_7[2U] = (((0U == (0x1fU & ((IData)(0x86U) 
                                           * (1U & 
                                              ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk3__BRA__0__KET____DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                               >> 1U)))))
                           ? 0U : (__Vtemp_4[((IData)(3U) 
                                              + (0xfU 
                                                 & (((IData)(0x86U) 
                                                     * 
                                                     (1U 
                                                      & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk3__BRA__0__KET____DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                                         >> 1U))) 
                                                    >> 5U)))] 
                                   << ((IData)(0x20U) 
                                       - (0x1fU & ((IData)(0x86U) 
                                                   * 
                                                   (1U 
                                                    & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk3__BRA__0__KET____DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                                       >> 1U))))))) 
                         | (__Vtemp_4[((IData)(2U) 
                                       + (0xfU & (((IData)(0x86U) 
                                                   * 
                                                   (1U 
                                                    & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk3__BRA__0__KET____DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                                       >> 1U))) 
                                                  >> 5U)))] 
                            >> (0x1fU & ((IData)(0x86U) 
                                         * (1U & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk3__BRA__0__KET____DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                                  >> 1U))))));
        __Vtemp_7[3U] = (((0U == (0x1fU & ((IData)(0x86U) 
                                           * (1U & 
                                              ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk3__BRA__0__KET____DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                               >> 1U)))))
                           ? 0U : (__Vtemp_4[((IData)(4U) 
                                              + (0xfU 
                                                 & (((IData)(0x86U) 
                                                     * 
                                                     (1U 
                                                      & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk3__BRA__0__KET____DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                                         >> 1U))) 
                                                    >> 5U)))] 
                                   << ((IData)(0x20U) 
                                       - (0x1fU & ((IData)(0x86U) 
                                                   * 
                                                   (1U 
                                                    & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk3__BRA__0__KET____DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                                       >> 1U))))))) 
                         | (__Vtemp_4[((IData)(3U) 
                                       + (0xfU & (((IData)(0x86U) 
                                                   * 
                                                   (1U 
                                                    & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk3__BRA__0__KET____DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                                       >> 1U))) 
                                                  >> 5U)))] 
                            >> (0x1fU & ((IData)(0x86U) 
                                         * (1U & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk3__BRA__0__KET____DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                                  >> 1U))))));
        __Vtemp_7[4U] = (((0U == (0x1fU & ((IData)(0x86U) 
                                           * (1U & 
                                              ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk3__BRA__0__KET____DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                               >> 1U)))))
                           ? 0U : (__Vtemp_4[((IData)(5U) 
                                              + (0xfU 
                                                 & (((IData)(0x86U) 
                                                     * 
                                                     (1U 
                                                      & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk3__BRA__0__KET____DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                                         >> 1U))) 
                                                    >> 5U)))] 
                                   << ((IData)(0x20U) 
                                       - (0x1fU & ((IData)(0x86U) 
                                                   * 
                                                   (1U 
                                                    & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk3__BRA__0__KET____DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                                       >> 1U))))))) 
                         | (__Vtemp_4[((IData)(4U) 
                                       + (0xfU & (((IData)(0x86U) 
                                                   * 
                                                   (1U 
                                                    & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk3__BRA__0__KET____DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                                       >> 1U))) 
                                                  >> 5U)))] 
                            >> (0x1fU & ((IData)(0x86U) 
                                         * (1U & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk3__BRA__0__KET____DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                                  >> 1U))))));
    } else {
        __Vtemp_7[0U] = vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk3__BRA__0__KET____DOT__rsp_arb__DOT____Vxrand_he19c0b17__0[0U];
        __Vtemp_7[1U] = vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk3__BRA__0__KET____DOT__rsp_arb__DOT____Vxrand_he19c0b17__0[1U];
        __Vtemp_7[2U] = vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk3__BRA__0__KET____DOT__rsp_arb__DOT____Vxrand_he19c0b17__0[2U];
        __Vtemp_7[3U] = vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk3__BRA__0__KET____DOT__rsp_arb__DOT____Vxrand_he19c0b17__0[3U];
        __Vtemp_7[4U] = vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk3__BRA__0__KET____DOT__rsp_arb__DOT____Vxrand_he19c0b17__0[4U];
    }
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk3__BRA__0__KET____DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__out_buf__DOT__genblk1__DOT__pipe_buffer__DOT____Vcellinp__genblk1__DOT__genblk1__BRA__0__KET____DOT__pipe_register__data_in[0U] 
        = __Vtemp_7[0U];
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk3__BRA__0__KET____DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__out_buf__DOT__genblk1__DOT__pipe_buffer__DOT____Vcellinp__genblk1__DOT__genblk1__BRA__0__KET____DOT__pipe_register__data_in[1U] 
        = __Vtemp_7[1U];
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk3__BRA__0__KET____DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__out_buf__DOT__genblk1__DOT__pipe_buffer__DOT____Vcellinp__genblk1__DOT__genblk1__BRA__0__KET____DOT__pipe_register__data_in[2U] 
        = __Vtemp_7[2U];
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk3__BRA__0__KET____DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__out_buf__DOT__genblk1__DOT__pipe_buffer__DOT____Vcellinp__genblk1__DOT__genblk1__BRA__0__KET____DOT__pipe_register__data_in[3U] 
        = __Vtemp_7[3U];
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk3__BRA__0__KET____DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__out_buf__DOT__genblk1__DOT__pipe_buffer__DOT____Vcellinp__genblk1__DOT__genblk1__BRA__0__KET____DOT__pipe_register__data_in[4U] 
        = (((IData)((0U != (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT____Vcellinp__genblk3__BRA__0__KET____DOT__rsp_arb__valid_in))) 
            << 7U) | ((0x40U & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk3__BRA__0__KET____DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                << 5U)) | (0x3fU & 
                                           __Vtemp_7[4U])));
}

VL_INLINE_OPT void VVortex___024root___nba_comb__TOP__94(VVortex___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    VVortex__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVortex___024root___nba_comb__TOP__94\n"); );
    // Init
    SData/*11:0*/ Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__req_xbar__DOT__genblk3__DOT__genblk1__DOT__per_output_ready_in;
    Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__req_xbar__DOT__genblk3__DOT__genblk1__DOT__per_output_ready_in = 0;
    VlWide<3>/*95:0*/ __Vtemp_1;
    VlWide<3>/*95:0*/ __Vtemp_2;
    // Body
    Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__req_xbar__DOT__genblk3__DOT__genblk1__DOT__per_output_ready_in 
        = (((0xfffff800U & (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__req_xbar__DOT____Vcellinp__genblk3__DOT__genblk1__DOT__genblk1__BRA__0__KET____DOT__xbar_arb__ready_out) 
                             << 0xbU) & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__req_xbar__DOT__genblk3__DOT__genblk1__DOT__genblk1__BRA__3__KET____DOT__xbar_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                         << 9U))) | 
            (0x600U & (((- (IData)((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__req_xbar__DOT____Vcellinp__genblk3__DOT__genblk1__DOT__genblk1__BRA__0__KET____DOT__xbar_arb__ready_out))) 
                        & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__req_xbar__DOT__genblk3__DOT__genblk1__DOT__genblk1__BRA__3__KET____DOT__xbar_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot)) 
                       << 9U))) | (((0xffffff00U & 
                                     (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__req_xbar__DOT____Vcellinp__genblk3__DOT__genblk1__DOT__genblk1__BRA__0__KET____DOT__xbar_arb__ready_out) 
                                       << 8U) & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__req_xbar__DOT__genblk3__DOT__genblk1__DOT__genblk1__BRA__2__KET____DOT__xbar_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                                 << 6U))) 
                                    | (0xc0U & (((- (IData)((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__req_xbar__DOT____Vcellinp__genblk3__DOT__genblk1__DOT__genblk1__BRA__0__KET____DOT__xbar_arb__ready_out))) 
                                                 & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__req_xbar__DOT__genblk3__DOT__genblk1__DOT__genblk1__BRA__2__KET____DOT__xbar_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot)) 
                                                << 6U))) 
                                   | (((0xffffffe0U 
                                        & (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__req_xbar__DOT____Vcellinp__genblk3__DOT__genblk1__DOT__genblk1__BRA__0__KET____DOT__xbar_arb__ready_out) 
                                            << 5U) 
                                           & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__req_xbar__DOT__genblk3__DOT__genblk1__DOT__genblk1__BRA__1__KET____DOT__xbar_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                                              << 3U))) 
                                       | (0x18U & (
                                                   ((- (IData)((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__req_xbar__DOT____Vcellinp__genblk3__DOT__genblk1__DOT__genblk1__BRA__0__KET____DOT__xbar_arb__ready_out))) 
                                                    & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__req_xbar__DOT__genblk3__DOT__genblk1__DOT__genblk1__BRA__1__KET____DOT__xbar_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot)) 
                                                   << 3U))) 
                                      | ((0xfffffffcU 
                                          & (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__req_xbar__DOT____Vcellinp__genblk3__DOT__genblk1__DOT__genblk1__BRA__0__KET____DOT__xbar_arb__ready_out) 
                                              << 2U) 
                                             & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__req_xbar__DOT__genblk3__DOT__genblk1__DOT__genblk1__BRA__0__KET____DOT__xbar_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot))) 
                                         | (3U & ((- (IData)((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__req_xbar__DOT____Vcellinp__genblk3__DOT__genblk1__DOT__genblk1__BRA__0__KET____DOT__xbar_arb__ready_out))) 
                                                  & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__req_xbar__DOT__genblk3__DOT__genblk1__DOT__genblk1__BRA__0__KET____DOT__xbar_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot)))))));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__req_in_ready 
        = ((4U & (((0xbU >= (0xfU & ((IData)(2U) + 
                                     ((IData)(3U) * 
                                      (3U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__src_regs 
                                             >> 0xcU))))))
                    ? ((IData)(Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__req_xbar__DOT__genblk3__DOT__genblk1__DOT__per_output_ready_in) 
                       >> (0xfU & ((IData)(2U) + ((IData)(3U) 
                                                  * 
                                                  (3U 
                                                   & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__src_regs 
                                                      >> 0xcU))))))
                    : (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__req_xbar__DOT____Vxrand_h8df6f50b__2)) 
                  << 2U)) | ((2U & (((0xbU >= (0xfU 
                                               & ((IData)(1U) 
                                                  + 
                                                  ((IData)(3U) 
                                                   * 
                                                   (3U 
                                                    & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__src_regs 
                                                       >> 6U))))))
                                      ? ((IData)(Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__req_xbar__DOT__genblk3__DOT__genblk1__DOT__per_output_ready_in) 
                                         >> (0xfU & 
                                             ((IData)(1U) 
                                              + ((IData)(3U) 
                                                 * 
                                                 (3U 
                                                  & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__src_regs 
                                                     >> 6U))))))
                                      : (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__req_xbar__DOT____Vxrand_h8df6f50b__1)) 
                                    << 1U)) | (1U & 
                                               ((0xbU 
                                                 >= 
                                                 (0xfU 
                                                  & ((IData)(3U) 
                                                     * 
                                                     (3U 
                                                      & vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__src_regs))))
                                                 ? 
                                                ((IData)(Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__req_xbar__DOT__genblk3__DOT__genblk1__DOT__per_output_ready_in) 
                                                 >> 
                                                 (0xfU 
                                                  & ((IData)(3U) 
                                                     * 
                                                     (3U 
                                                      & vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__src_regs))))
                                                 : (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__req_xbar__DOT____Vxrand_h8df6f50b__0)))));
    __Vtemp_1[0U] = (IData)((((QData)((IData)((1U & 
                                               (vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard_if.data[0U] 
                                                >> 0x18U)))) 
                              << 0x32U) | (((QData)((IData)(
                                                            (3U 
                                                             & (vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard_if.data[2U] 
                                                                >> 2U)))) 
                                            << 0x30U) 
                                           | (((QData)((IData)(
                                                               (0xfU 
                                                                & ((vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard_if.data[2U] 
                                                                    << 2U) 
                                                                   | (vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard_if.data[1U] 
                                                                      >> 0x1eU))))) 
                                               << 0x2cU) 
                                              | ((0xfffffffff80ULL 
                                                  & (((QData)((IData)(
                                                                      vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard_if.data[1U])) 
                                                      << 0xeU) 
                                                     | (0x3fffffffff80ULL 
                                                        & ((QData)((IData)(
                                                                           vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard_if.data[0U])) 
                                                           >> 0x12U)))) 
                                                 | (QData)((IData)(
                                                                   ((0x7eU 
                                                                     & (vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard_if.data[0U] 
                                                                        >> 0x11U)) 
                                                                    | (1U 
                                                                       & (vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard_if.data[3U] 
                                                                          >> 9U))))))))));
    __Vtemp_1[1U] = (((IData)((0x1fffffffffULL & (((QData)((IData)(
                                                                   vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard_if.data[3U])) 
                                                   << 0x1cU) 
                                                  | ((QData)((IData)(
                                                                     vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard_if.data[2U])) 
                                                     >> 4U)))) 
                      << 0x13U) | (IData)(((((QData)((IData)(
                                                             (1U 
                                                              & (vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard_if.data[0U] 
                                                                 >> 0x18U)))) 
                                             << 0x32U) 
                                            | (((QData)((IData)(
                                                                (3U 
                                                                 & (vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard_if.data[2U] 
                                                                    >> 2U)))) 
                                                << 0x30U) 
                                               | (((QData)((IData)(
                                                                   (0xfU 
                                                                    & ((vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard_if.data[2U] 
                                                                        << 2U) 
                                                                       | (vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard_if.data[1U] 
                                                                          >> 0x1eU))))) 
                                                   << 0x2cU) 
                                                  | ((0xfffffffff80ULL 
                                                      & (((QData)((IData)(
                                                                          vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard_if.data[1U])) 
                                                          << 0xeU) 
                                                         | (0x3fffffffff80ULL 
                                                            & ((QData)((IData)(
                                                                               vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard_if.data[0U])) 
                                                               >> 0x12U)))) 
                                                     | (QData)((IData)(
                                                                       ((0x7eU 
                                                                         & (vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard_if.data[0U] 
                                                                            >> 0x11U)) 
                                                                        | (1U 
                                                                           & (vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard_if.data[3U] 
                                                                              >> 9U))))))))) 
                                           >> 0x20U)));
    __Vtemp_2[2U] = ((0xff000000U & ((0x8000000U & 
                                      ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__req_xbar__DOT__genblk3__DOT__genblk1__DOT__genblk1__BRA__3__KET____DOT__xbar_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__priority_arbiter__DOT__genblk1__DOT__priority_encoder__DOT__genblk2__DOT__scan__DOT__t) 
                                       << 0x15U)) | 
                                     ((0x4000000U & 
                                       ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__req_xbar__DOT__genblk3__DOT__genblk1__DOT__genblk1__BRA__2__KET____DOT__xbar_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__priority_arbiter__DOT__genblk1__DOT__priority_encoder__DOT__genblk2__DOT__scan__DOT__t) 
                                        << 0x14U)) 
                                      | ((0x2000000U 
                                          & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__req_xbar__DOT__genblk3__DOT__genblk1__DOT__genblk1__BRA__1__KET____DOT__xbar_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__priority_arbiter__DOT__genblk1__DOT__priority_encoder__DOT__genblk2__DOT__scan__DOT__t) 
                                             << 0x13U)) 
                                         | (0x1000000U 
                                            & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__req_xbar__DOT__genblk3__DOT__genblk1__DOT__genblk1__BRA__0__KET____DOT__xbar_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__priority_arbiter__DOT__genblk1__DOT__priority_encoder__DOT__genblk2__DOT__scan__DOT__t) 
                                               << 0x12U)))))) 
                     | (((IData)((0x1fffffffffULL & 
                                  (((QData)((IData)(
                                                    vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard_if.data[3U])) 
                                    << 0x1cU) | ((QData)((IData)(
                                                                 vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard_if.data[2U])) 
                                                 >> 4U)))) 
                         >> 0xdU) | ((IData)(((0x1fffffffffULL 
                                               & (((QData)((IData)(
                                                                   vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard_if.data[3U])) 
                                                   << 0x1cU) 
                                                  | ((QData)((IData)(
                                                                     vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard_if.data[2U])) 
                                                     >> 4U))) 
                                              >> 0x20U)) 
                                     << 0x13U)));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT____Vcellinp__pipe_reg1__data_in[0U] 
        = (IData)((((QData)((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__has_collision_n)) 
                    << 0x20U) | (QData)((IData)(((((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__req_xbar__DOT____Vcellout__genblk3__DOT__genblk1__DOT__genblk1__BRA__3__KET____DOT__xbar_arb__data_out) 
                                                   << 0x1aU) 
                                                  | (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__req_xbar__DOT____Vcellout__genblk3__DOT__genblk1__DOT__genblk1__BRA__2__KET____DOT__xbar_arb__data_out) 
                                                      << 0x14U) 
                                                     | (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__req_xbar__DOT____Vcellout__genblk3__DOT__genblk1__DOT__genblk1__BRA__1__KET____DOT__xbar_arb__data_out) 
                                                         << 0xeU) 
                                                        | ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__req_xbar__DOT____Vcellout__genblk3__DOT__genblk1__DOT__genblk1__BRA__0__KET____DOT__xbar_arb__data_out) 
                                                           << 8U)))) 
                                                 | ((0xc0U 
                                                     & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__req_xbar__DOT__genblk3__DOT__genblk1__DOT__genblk1__BRA__3__KET____DOT__xbar_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__priority_arbiter__DOT__genblk1__DOT__priority_encoder__DOT__genblk2__DOT__lzc__DOT__genblk1__DOT__find_first__DOT__d_n) 
                                                        << 6U)) 
                                                    | ((0x30U 
                                                        & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__req_xbar__DOT__genblk3__DOT__genblk1__DOT__genblk1__BRA__2__KET____DOT__xbar_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__priority_arbiter__DOT__genblk1__DOT__priority_encoder__DOT__genblk2__DOT__lzc__DOT__genblk1__DOT__find_first__DOT__d_n) 
                                                           << 4U)) 
                                                       | ((0xcU 
                                                           & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__req_xbar__DOT__genblk3__DOT__genblk1__DOT__genblk1__BRA__1__KET____DOT__xbar_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__priority_arbiter__DOT__genblk1__DOT__priority_encoder__DOT__genblk2__DOT__lzc__DOT__genblk1__DOT__find_first__DOT__d_n) 
                                                              << 2U)) 
                                                          | (3U 
                                                             & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__req_xbar__DOT__genblk3__DOT__genblk1__DOT__genblk1__BRA__0__KET____DOT__xbar_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__priority_arbiter__DOT__genblk1__DOT__priority_encoder__DOT__genblk2__DOT__lzc__DOT__genblk1__DOT__find_first__DOT__d_n))))))))));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT____Vcellinp__pipe_reg1__data_in[1U] 
        = ((__Vtemp_1[0U] << 1U) | (IData)(((((QData)((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__has_collision_n)) 
                                              << 0x20U) 
                                             | (QData)((IData)(
                                                               ((((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__req_xbar__DOT____Vcellout__genblk3__DOT__genblk1__DOT__genblk1__BRA__3__KET____DOT__xbar_arb__data_out) 
                                                                  << 0x1aU) 
                                                                 | (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__req_xbar__DOT____Vcellout__genblk3__DOT__genblk1__DOT__genblk1__BRA__2__KET____DOT__xbar_arb__data_out) 
                                                                     << 0x14U) 
                                                                    | (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__req_xbar__DOT____Vcellout__genblk3__DOT__genblk1__DOT__genblk1__BRA__1__KET____DOT__xbar_arb__data_out) 
                                                                        << 0xeU) 
                                                                       | ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__req_xbar__DOT____Vcellout__genblk3__DOT__genblk1__DOT__genblk1__BRA__0__KET____DOT__xbar_arb__data_out) 
                                                                          << 8U)))) 
                                                                | ((0xc0U 
                                                                    & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__req_xbar__DOT__genblk3__DOT__genblk1__DOT__genblk1__BRA__3__KET____DOT__xbar_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__priority_arbiter__DOT__genblk1__DOT__priority_encoder__DOT__genblk2__DOT__lzc__DOT__genblk1__DOT__find_first__DOT__d_n) 
                                                                       << 6U)) 
                                                                   | ((0x30U 
                                                                       & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__req_xbar__DOT__genblk3__DOT__genblk1__DOT__genblk1__BRA__2__KET____DOT__xbar_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__priority_arbiter__DOT__genblk1__DOT__priority_encoder__DOT__genblk2__DOT__lzc__DOT__genblk1__DOT__find_first__DOT__d_n) 
                                                                          << 4U)) 
                                                                      | ((0xcU 
                                                                          & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__req_xbar__DOT__genblk3__DOT__genblk1__DOT__genblk1__BRA__1__KET____DOT__xbar_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__priority_arbiter__DOT__genblk1__DOT__priority_encoder__DOT__genblk2__DOT__lzc__DOT__genblk1__DOT__find_first__DOT__d_n) 
                                                                             << 2U)) 
                                                                         | (3U 
                                                                            & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__req_xbar__DOT__genblk3__DOT__genblk1__DOT__genblk1__BRA__0__KET____DOT__xbar_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__priority_arbiter__DOT__genblk1__DOT__priority_encoder__DOT__genblk2__DOT__lzc__DOT__genblk1__DOT__find_first__DOT__d_n))))))))) 
                                            >> 0x20U)));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT____Vcellinp__pipe_reg1__data_in[2U] 
        = ((__Vtemp_1[0U] >> 0x1fU) | (__Vtemp_1[1U] 
                                       << 1U));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT____Vcellinp__pipe_reg1__data_in[3U] 
        = ((__Vtemp_1[1U] >> 0x1fU) | ((((IData)(vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard_if.ready)
                                          ? 0U : (7U 
                                                  & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__pipe_reg1__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_r) 
                                                     | (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__req_in_ready)))) 
                                        << 0x1dU) | 
                                       (__Vtemp_2[2U] 
                                        << 1U)));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT____Vcellinp__pipe_reg1__data_in[4U] 
        = ((1U & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard__DOT__out_arb__DOT__genblk1__DOT__genblk1__DOT__out_buf__DOT__genblk1__DOT__out_buf__DOT__genblk1__DOT__genblk1__BRA__0__KET____DOT__pipe_register__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_r)) 
           | ((1U & (((IData)(vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__scoreboard_if.ready)
                       ? 0U : (7U & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__pipe_reg1__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_r) 
                                     | (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__issue__DOT__issue_slices__BRA__0__KET____DOT__issue_slice__DOT__operands__DOT__req_in_ready)))) 
                     >> 3U)) | (__Vtemp_2[2U] >> 0x1fU)));
}

VL_INLINE_OPT void VVortex___024root___nba_comb__TOP__99(VVortex___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    VVortex__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVortex___024root___nba_comb__TOP__99\n"); );
    // Init
    CData/*3:0*/ Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk4__BRA__0__KET____DOT__lsu_adapter__DOT__rsp_ready_out;
    Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk4__BRA__0__KET____DOT__lsu_adapter__DOT__rsp_ready_out = 0;
    CData/*0:0*/ Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__out_rsp_fire;
    Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__out_rsp_fire = 0;
    // Body
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT____Vcellout__genblk3__BRA__0__KET____DOT__rsp_arb__ready_in 
        = ((((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk3__BRA__0__KET____DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__out_buf__DOT__genblk1__DOT__pipe_buffer__DOT__genblk1__DOT__ready) 
             & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk3__BRA__0__KET____DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot) 
                >> 1U)) << 1U) | (1U & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk3__BRA__0__KET____DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__out_buf__DOT__genblk1__DOT__pipe_buffer__DOT__genblk1__DOT__ready) 
                                        & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk3__BRA__0__KET____DOT__rsp_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot))));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk2__DOT__cache_bypass__DOT__genblk14__BRA__0__KET____DOT__core_rsp_buf__DOT__genblk1__DOT__skid_buffer__DOT__genblk1__DOT__stream_buffer__DOT__genblk1__DOT__genblk1__DOT__pop 
        = ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT____Vcellout__genblk3__BRA__0__KET____DOT__rsp_arb__ready_in) 
           & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk2__DOT__cache_bypass__DOT__genblk14__BRA__0__KET____DOT__core_rsp_buf__DOT__genblk1__DOT__skid_buffer__DOT__genblk1__DOT__stream_buffer__DOT__genblk1__DOT__genblk1__DOT__valid_out_r));
    Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__out_rsp_fire 
        = ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT____Vcellout__genblk3__BRA__0__KET____DOT__rsp_arb__ready_in) 
           & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk2__DOT__cache_bypass__DOT__genblk14__BRA__0__KET____DOT__core_rsp_buf__DOT__genblk1__DOT__skid_buffer__DOT__genblk1__DOT__stream_buffer__DOT__genblk1__DOT__genblk1__DOT__valid_out_r));
    Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk4__BRA__0__KET____DOT__lsu_adapter__DOT__rsp_ready_out 
        = ((0xfffffff8U & (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT____Vcellout__genblk3__BRA__0__KET____DOT__rsp_arb__ready_in) 
                            << 2U) & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk4__BRA__0__KET____DOT__lsu_adapter__DOT__stream_pack__DOT____VdfgRegularize_h305aea41_0_4) 
                                      << 3U))) | ((0xfffffffcU 
                                                   & (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT____Vcellout__genblk3__BRA__0__KET____DOT__rsp_arb__ready_in) 
                                                       << 1U) 
                                                      & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk4__BRA__0__KET____DOT__lsu_adapter__DOT__stream_pack__DOT____VdfgRegularize_h305aea41_0_3) 
                                                         << 2U))) 
                                                  | ((0xfffffffeU 
                                                      & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT____Vcellout__genblk3__BRA__0__KET____DOT__rsp_arb__ready_in) 
                                                         & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk4__BRA__0__KET____DOT__lsu_adapter__DOT__stream_pack__DOT____VdfgRegularize_h305aea41_0_2) 
                                                            << 1U))) 
                                                     | (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT____Vcellout__genblk3__BRA__0__KET____DOT__rsp_arb__ready_in) 
                                                         >> 1U) 
                                                        & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk4__BRA__0__KET____DOT__lsu_adapter__DOT__stream_pack__DOT____VdfgRegularize_h305aea41_0_1)))));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__ibuf_pop 
        = Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__out_rsp_fire;
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__local_mem__DOT__rsp_ready_out 
        = Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__genblk4__BRA__0__KET____DOT__lsu_adapter__DOT__rsp_ready_out;
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__req_ibuf__DOT__allocator__DOT__free_slots_n 
        = vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__req_ibuf__DOT__allocator__DOT__free_slots;
    if (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__ibuf_pop) {
        vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__req_ibuf__DOT__allocator__DOT__free_slots_n 
            = ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__req_ibuf__DOT__allocator__DOT__free_slots_n) 
               | (0xfU & ((IData)(1U) << (3U & vlSymsp->TOP__Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__genblk2__BRA__0__KET____DOT__core_bus_tmp_if__BRA__0__KET__.rsp_data[0U]))));
    }
    if (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__ibuf_push) {
        vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__req_ibuf__DOT__allocator__DOT__free_slots_n 
            = ((~ ((IData)(1U) << (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__req_ibuf__DOT__allocator__DOT__acquire_addr_r))) 
               & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__genblk1__BRA__0__KET____DOT__genblk1__DOT__mem_coalescer__DOT__req_ibuf__DOT__allocator__DOT__free_slots_n));
    }
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__local_mem__DOT__rsp_xbar__DOT__genblk3__DOT__genblk1__DOT__genblk1__BRA__0__KET____DOT__xbar_arb__DOT__genblk1__DOT__genblk1__DOT__out_buf__DOT__genblk1__DOT__skid_buffer__DOT__genblk1__DOT__stream_buffer__DOT__genblk1__DOT__genblk1__DOT__stall_out 
        = ((~ (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__local_mem__DOT__rsp_ready_out)) 
           & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__local_mem__DOT__rsp_xbar__DOT__genblk3__DOT__genblk1__DOT__genblk1__BRA__0__KET____DOT__xbar_arb__DOT__genblk1__DOT__genblk1__DOT__out_buf__DOT__genblk1__DOT__skid_buffer__DOT__genblk1__DOT__stream_buffer__DOT__genblk1__DOT__genblk1__DOT__valid_out_r));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__local_mem__DOT__rsp_xbar__DOT__genblk3__DOT__genblk1__DOT__genblk1__BRA__1__KET____DOT__xbar_arb__DOT__genblk1__DOT__genblk1__DOT__out_buf__DOT__genblk1__DOT__skid_buffer__DOT__genblk1__DOT__stream_buffer__DOT__genblk1__DOT__genblk1__DOT__stall_out 
        = ((~ ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__local_mem__DOT__rsp_ready_out) 
               >> 1U)) & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__local_mem__DOT__rsp_xbar__DOT__genblk3__DOT__genblk1__DOT__genblk1__BRA__1__KET____DOT__xbar_arb__DOT__genblk1__DOT__genblk1__DOT__out_buf__DOT__genblk1__DOT__skid_buffer__DOT__genblk1__DOT__stream_buffer__DOT__genblk1__DOT__genblk1__DOT__valid_out_r));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__local_mem__DOT__rsp_xbar__DOT__genblk3__DOT__genblk1__DOT__genblk1__BRA__2__KET____DOT__xbar_arb__DOT__genblk1__DOT__genblk1__DOT__out_buf__DOT__genblk1__DOT__skid_buffer__DOT__genblk1__DOT__stream_buffer__DOT__genblk1__DOT__genblk1__DOT__stall_out 
        = ((~ ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__local_mem__DOT__rsp_ready_out) 
               >> 2U)) & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__local_mem__DOT__rsp_xbar__DOT__genblk3__DOT__genblk1__DOT__genblk1__BRA__2__KET____DOT__xbar_arb__DOT__genblk1__DOT__genblk1__DOT__out_buf__DOT__genblk1__DOT__skid_buffer__DOT__genblk1__DOT__stream_buffer__DOT__genblk1__DOT__genblk1__DOT__valid_out_r));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__local_mem__DOT__rsp_xbar__DOT__genblk3__DOT__genblk1__DOT__genblk1__BRA__3__KET____DOT__xbar_arb__DOT__genblk1__DOT__genblk1__DOT__out_buf__DOT__genblk1__DOT__skid_buffer__DOT__genblk1__DOT__stream_buffer__DOT__genblk1__DOT__genblk1__DOT__stall_out 
        = ((~ ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__local_mem__DOT__rsp_ready_out) 
               >> 3U)) & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__lmem_unit__DOT__local_mem__DOT__rsp_xbar__DOT__genblk3__DOT__genblk1__DOT__genblk1__BRA__3__KET____DOT__xbar_arb__DOT__genblk1__DOT__genblk1__DOT__out_buf__DOT__genblk1__DOT__skid_buffer__DOT__genblk1__DOT__stream_buffer__DOT__genblk1__DOT__genblk1__DOT__valid_out_r));
}
