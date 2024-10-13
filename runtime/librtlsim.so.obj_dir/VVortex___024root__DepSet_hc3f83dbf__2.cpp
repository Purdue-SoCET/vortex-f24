// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See VVortex.h for the primary calling header

#include "VVortex__pch.h"
#include "VVortex___024root.h"

VL_INLINE_OPT void VVortex___024root___nba_comb__TOP__79(VVortex___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    VVortex__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVortex___024root___nba_comb__TOP__79\n"); );
    // Init
    VlWide<18>/*575:0*/ __Vtemp_2;
    // Body
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[0U] 
        = ((0xfffe0000U & vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[0U]) 
           | (((IData)((0U != (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__genblk1__BRA__1__KET____DOT__genblk1__BRA__0__KET____DOT__vs))) 
               << 0x10U) | ((~ (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__next_table_x)) 
                            & (((IData)(vlSelf->__VdfgRegularize_hd87f99a1_43_4) 
                                << 0xfU) | (((IData)(vlSelf->__VdfgRegularize_hd87f99a1_43_5) 
                                             << 0xeU) 
                                            | (((IData)(vlSelf->__VdfgRegularize_hd87f99a1_43_6) 
                                                << 0xdU) 
                                               | (((IData)(vlSelf->__VdfgRegularize_hd87f99a1_43_7) 
                                                   << 0xcU) 
                                                  | (((IData)(vlSelf->__VdfgRegularize_hd87f99a1_43_8) 
                                                      << 0xbU) 
                                                     | (((IData)(vlSelf->__VdfgRegularize_hd87f99a1_43_9) 
                                                         << 0xaU) 
                                                        | (((IData)(vlSelf->__VdfgRegularize_hd87f99a1_43_10) 
                                                            << 9U) 
                                                           | (((IData)(vlSelf->__VdfgRegularize_hd87f99a1_43_11) 
                                                               << 8U) 
                                                              | (((IData)(vlSelf->__VdfgRegularize_hd87f99a1_43_12) 
                                                                  << 7U) 
                                                                 | (((IData)(vlSelf->__VdfgRegularize_hd87f99a1_43_13) 
                                                                     << 6U) 
                                                                    | (((IData)(vlSelf->__VdfgRegularize_hd87f99a1_43_14) 
                                                                        << 5U) 
                                                                       | (((IData)(vlSelf->__VdfgRegularize_hd87f99a1_43_15) 
                                                                           << 4U) 
                                                                          | (((IData)(vlSelf->__VdfgRegularize_hd87f99a1_43_16) 
                                                                              << 3U) 
                                                                             | (((IData)(vlSelf->__VdfgRegularize_hd87f99a1_43_17) 
                                                                                << 2U) 
                                                                                | (((IData)(vlSelf->__VdfgRegularize_hd87f99a1_43_18) 
                                                                                << 1U) 
                                                                                | (IData)(vlSelf->__VdfgRegularize_hd87f99a1_43_19)))))))))))))))))));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__genblk1__BRA__1__KET____DOT__genblk1__BRA__0__KET____DOT__vs 
        = (3U & vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[0U]);
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__genblk1__BRA__1__KET____DOT__genblk1__BRA__1__KET____DOT__vs 
        = (3U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[0U] 
                 >> 2U));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__genblk1__BRA__1__KET____DOT__genblk1__BRA__2__KET____DOT__vs 
        = (3U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[0U] 
                 >> 4U));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__genblk1__BRA__1__KET____DOT__genblk1__BRA__3__KET____DOT__vs 
        = (3U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[0U] 
                 >> 6U));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__genblk1__BRA__1__KET____DOT__genblk1__BRA__4__KET____DOT__vs 
        = (3U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[0U] 
                 >> 8U));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__genblk1__BRA__1__KET____DOT__genblk1__BRA__5__KET____DOT__vs 
        = (3U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[0U] 
                 >> 0xaU));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__genblk1__BRA__1__KET____DOT__genblk1__BRA__6__KET____DOT__vs 
        = (3U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[0U] 
                 >> 0xcU));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__genblk1__BRA__1__KET____DOT__genblk1__BRA__7__KET____DOT__vs 
        = (3U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[0U] 
                 >> 0xeU));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__genblk1__BRA__2__KET____DOT__genblk1__BRA__0__KET____DOT__vs 
        = ((2U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[0U] 
                  >> 0x11U)) | (1U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[0U] 
                                      >> 0x10U)));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__genblk1__BRA__2__KET____DOT__genblk1__BRA__1__KET____DOT__vs 
        = ((2U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[0U] 
                  >> 0x15U)) | (1U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[0U] 
                                      >> 0x14U)));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__genblk1__BRA__2__KET____DOT__genblk1__BRA__2__KET____DOT__vs 
        = ((2U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[0U] 
                  >> 0x19U)) | (1U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[0U] 
                                      >> 0x18U)));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__genblk1__BRA__2__KET____DOT__genblk1__BRA__3__KET____DOT__vs 
        = ((2U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[0U] 
                  >> 0x1dU)) | (1U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[0U] 
                                      >> 0x1cU)));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__genblk1__BRA__3__KET____DOT__genblk1__BRA__0__KET____DOT__vs 
        = ((2U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[1U] 
                  >> 3U)) | (1U & vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[1U]));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__genblk1__BRA__3__KET____DOT__genblk1__BRA__1__KET____DOT__vs 
        = ((2U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[1U] 
                  >> 0xbU)) | (1U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[1U] 
                                     >> 8U)));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__genblk1__BRA__4__KET____DOT__genblk1__BRA__0__KET____DOT__vs 
        = ((2U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[1U] 
                  >> 0x17U)) | (1U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[1U] 
                                      >> 0x10U)));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__addr 
        = ((0xfffffffffffffffeULL & vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__addr) 
           | (IData)((IData)((1U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[0U] 
                                    >> 1U)))));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__addr 
        = ((0xfffffffffffffffbULL & vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__addr) 
           | ((QData)((IData)((1U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[0U] 
                                     >> 3U)))) << 2U));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__addr 
        = ((0xffffffffffffffefULL & vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__addr) 
           | ((QData)((IData)((1U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[0U] 
                                     >> 5U)))) << 4U));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__addr 
        = ((0xffffffffffffffbfULL & vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__addr) 
           | ((QData)((IData)((1U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[0U] 
                                     >> 7U)))) << 6U));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__addr 
        = ((0xfffffffffffffeffULL & vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__addr) 
           | ((QData)((IData)((1U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[0U] 
                                     >> 9U)))) << 8U));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__addr 
        = ((0xfffffffffffffbffULL & vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__addr) 
           | ((QData)((IData)((1U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[0U] 
                                     >> 0xbU)))) << 0xaU));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__addr 
        = ((0xffffffffffffefffULL & vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__addr) 
           | ((QData)((IData)((1U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[0U] 
                                     >> 0xdU)))) << 0xcU));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__addr 
        = ((0xffffffffffffbfffULL & vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__addr) 
           | ((QData)((IData)((1U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[0U] 
                                     >> 0xfU)))) << 0xeU));
    __Vtemp_2[0U] = (IData)((((QData)((IData)((0x3fffffU 
                                               & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0U] 
                                                  >> 4U)))) 
                              << 0xbU) | (QData)((IData)(
                                                         ((0x780U 
                                                           & (((0x200000U 
                                                                & vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0x11U])
                                                                ? (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__allocate_id_r)
                                                                : 
                                                               vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0U]) 
                                                              << 7U)) 
                                                          | ((0x78U 
                                                              & ((IData)(
                                                                         (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__addr 
                                                                          >> 0x30U)) 
                                                                 << 3U)) 
                                                             | ((((1U 
                                                                   & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0x11U] 
                                                                      >> 0x17U)) 
                                                                  || ((vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_tags__DOT__genblk3__BRA__0__KET____DOT__line_rdata 
                                                                       >> 0x12U) 
                                                                      & ((0x3ffffU 
                                                                          & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0x11U] 
                                                                             >> 1U)) 
                                                                         == 
                                                                         (0x3ffffU 
                                                                          & vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_tags__DOT__genblk3__BRA__0__KET____DOT__line_rdata)))) 
                                                                 << 2U) 
                                                                | (((~ 
                                                                     (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__write_table) 
                                                                       >> 0xfU) 
                                                                      | (0xfU 
                                                                         == (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__allocate_id_r)))) 
                                                                    & (IData)(vlSelf->__VdfgRegularize_hd87f99a1_43_4)) 
                                                                   | (((~ 
                                                                        (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__write_table) 
                                                                          >> 0xeU) 
                                                                         | (0xeU 
                                                                            == (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__allocate_id_r)))) 
                                                                       & (IData)(vlSelf->__VdfgRegularize_hd87f99a1_43_5)) 
                                                                      | (((~ 
                                                                           (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__write_table) 
                                                                             >> 0xdU) 
                                                                            | (0xdU 
                                                                               == (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__allocate_id_r)))) 
                                                                          & (IData)(vlSelf->__VdfgRegularize_hd87f99a1_43_6)) 
                                                                         | (((~ 
                                                                              (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__write_table) 
                                                                                >> 0xcU) 
                                                                               | (0xcU 
                                                                                == (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__allocate_id_r)))) 
                                                                             & (IData)(vlSelf->__VdfgRegularize_hd87f99a1_43_7)) 
                                                                            | (((~ 
                                                                                (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__write_table) 
                                                                                >> 0xbU) 
                                                                                | (0xbU 
                                                                                == (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__allocate_id_r)))) 
                                                                                & (IData)(vlSelf->__VdfgRegularize_hd87f99a1_43_8)) 
                                                                               | (((~ 
                                                                                (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__write_table) 
                                                                                >> 0xaU) 
                                                                                | (0xaU 
                                                                                == (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__allocate_id_r)))) 
                                                                                & (IData)(vlSelf->__VdfgRegularize_hd87f99a1_43_9)) 
                                                                                | (((~ 
                                                                                (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__write_table) 
                                                                                >> 9U) 
                                                                                | (9U 
                                                                                == (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__allocate_id_r)))) 
                                                                                & (IData)(vlSelf->__VdfgRegularize_hd87f99a1_43_10)) 
                                                                                | (((~ 
                                                                                (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__write_table) 
                                                                                >> 8U) 
                                                                                | (8U 
                                                                                == (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__allocate_id_r)))) 
                                                                                & (IData)(vlSelf->__VdfgRegularize_hd87f99a1_43_11)) 
                                                                                | (((~ 
                                                                                (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__write_table) 
                                                                                >> 7U) 
                                                                                | (7U 
                                                                                == (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__allocate_id_r)))) 
                                                                                & (IData)(vlSelf->__VdfgRegularize_hd87f99a1_43_12)) 
                                                                                | (((~ 
                                                                                (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__write_table) 
                                                                                >> 6U) 
                                                                                | (6U 
                                                                                == (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__allocate_id_r)))) 
                                                                                & (IData)(vlSelf->__VdfgRegularize_hd87f99a1_43_13)) 
                                                                                | (((~ 
                                                                                (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__write_table) 
                                                                                >> 5U) 
                                                                                | (5U 
                                                                                == (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__allocate_id_r)))) 
                                                                                & (IData)(vlSelf->__VdfgRegularize_hd87f99a1_43_14)) 
                                                                                | (((~ 
                                                                                (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__write_table) 
                                                                                >> 4U) 
                                                                                | (4U 
                                                                                == (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__allocate_id_r)))) 
                                                                                & (IData)(vlSelf->__VdfgRegularize_hd87f99a1_43_15)) 
                                                                                | (((~ 
                                                                                (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__write_table) 
                                                                                >> 3U) 
                                                                                | (3U 
                                                                                == (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__allocate_id_r)))) 
                                                                                & (IData)(vlSelf->__VdfgRegularize_hd87f99a1_43_16)) 
                                                                                | (((~ 
                                                                                (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__write_table) 
                                                                                >> 2U) 
                                                                                | (2U 
                                                                                == (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__allocate_id_r)))) 
                                                                                & (IData)(vlSelf->__VdfgRegularize_hd87f99a1_43_17)) 
                                                                                | (((~ 
                                                                                (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__write_table) 
                                                                                >> 1U) 
                                                                                | (1U 
                                                                                == (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__allocate_id_r)))) 
                                                                                & (IData)(vlSelf->__VdfgRegularize_hd87f99a1_43_18)) 
                                                                                | ((~ 
                                                                                ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__write_table) 
                                                                                | (0U 
                                                                                == (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__allocate_id_r)))) 
                                                                                & (IData)(vlSelf->__VdfgRegularize_hd87f99a1_43_19))))))))))))))))))))))));
    __Vtemp_2[1U] = (((vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[1U] 
                       << 6U) | (0x3eU & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0U] 
                                          >> 0x1aU))) 
                     | (IData)(((((QData)((IData)((0x3fffffU 
                                                   & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0U] 
                                                      >> 4U)))) 
                                  << 0xbU) | (QData)((IData)(
                                                             ((0x780U 
                                                               & (((0x200000U 
                                                                    & vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0x11U])
                                                                    ? (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__allocate_id_r)
                                                                    : 
                                                                   vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0U]) 
                                                                  << 7U)) 
                                                              | ((0x78U 
                                                                  & ((IData)(
                                                                             (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__addr 
                                                                              >> 0x30U)) 
                                                                     << 3U)) 
                                                                 | ((((1U 
                                                                       & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0x11U] 
                                                                          >> 0x17U)) 
                                                                      || ((vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_tags__DOT__genblk3__BRA__0__KET____DOT__line_rdata 
                                                                           >> 0x12U) 
                                                                          & ((0x3ffffU 
                                                                              & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0x11U] 
                                                                                >> 1U)) 
                                                                             == 
                                                                             (0x3ffffU 
                                                                              & vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_tags__DOT__genblk3__BRA__0__KET____DOT__line_rdata)))) 
                                                                     << 2U) 
                                                                    | (((~ 
                                                                         (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__write_table) 
                                                                           >> 0xfU) 
                                                                          | (0xfU 
                                                                             == (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__allocate_id_r)))) 
                                                                        & (IData)(vlSelf->__VdfgRegularize_hd87f99a1_43_4)) 
                                                                       | (((~ 
                                                                            (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__write_table) 
                                                                              >> 0xeU) 
                                                                             | (0xeU 
                                                                                == (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__allocate_id_r)))) 
                                                                           & (IData)(vlSelf->__VdfgRegularize_hd87f99a1_43_5)) 
                                                                          | (((~ 
                                                                               (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__write_table) 
                                                                                >> 0xdU) 
                                                                                | (0xdU 
                                                                                == (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__allocate_id_r)))) 
                                                                              & (IData)(vlSelf->__VdfgRegularize_hd87f99a1_43_6)) 
                                                                             | (((~ 
                                                                                (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__write_table) 
                                                                                >> 0xcU) 
                                                                                | (0xcU 
                                                                                == (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__allocate_id_r)))) 
                                                                                & (IData)(vlSelf->__VdfgRegularize_hd87f99a1_43_7)) 
                                                                                | (((~ 
                                                                                (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__write_table) 
                                                                                >> 0xbU) 
                                                                                | (0xbU 
                                                                                == (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__allocate_id_r)))) 
                                                                                & (IData)(vlSelf->__VdfgRegularize_hd87f99a1_43_8)) 
                                                                                | (((~ 
                                                                                (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__write_table) 
                                                                                >> 0xaU) 
                                                                                | (0xaU 
                                                                                == (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__allocate_id_r)))) 
                                                                                & (IData)(vlSelf->__VdfgRegularize_hd87f99a1_43_9)) 
                                                                                | (((~ 
                                                                                (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__write_table) 
                                                                                >> 9U) 
                                                                                | (9U 
                                                                                == (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__allocate_id_r)))) 
                                                                                & (IData)(vlSelf->__VdfgRegularize_hd87f99a1_43_10)) 
                                                                                | (((~ 
                                                                                (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__write_table) 
                                                                                >> 8U) 
                                                                                | (8U 
                                                                                == (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__allocate_id_r)))) 
                                                                                & (IData)(vlSelf->__VdfgRegularize_hd87f99a1_43_11)) 
                                                                                | (((~ 
                                                                                (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__write_table) 
                                                                                >> 7U) 
                                                                                | (7U 
                                                                                == (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__allocate_id_r)))) 
                                                                                & (IData)(vlSelf->__VdfgRegularize_hd87f99a1_43_12)) 
                                                                                | (((~ 
                                                                                (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__write_table) 
                                                                                >> 6U) 
                                                                                | (6U 
                                                                                == (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__allocate_id_r)))) 
                                                                                & (IData)(vlSelf->__VdfgRegularize_hd87f99a1_43_13)) 
                                                                                | (((~ 
                                                                                (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__write_table) 
                                                                                >> 5U) 
                                                                                | (5U 
                                                                                == (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__allocate_id_r)))) 
                                                                                & (IData)(vlSelf->__VdfgRegularize_hd87f99a1_43_14)) 
                                                                                | (((~ 
                                                                                (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__write_table) 
                                                                                >> 4U) 
                                                                                | (4U 
                                                                                == (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__allocate_id_r)))) 
                                                                                & (IData)(vlSelf->__VdfgRegularize_hd87f99a1_43_15)) 
                                                                                | (((~ 
                                                                                (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__write_table) 
                                                                                >> 3U) 
                                                                                | (3U 
                                                                                == (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__allocate_id_r)))) 
                                                                                & (IData)(vlSelf->__VdfgRegularize_hd87f99a1_43_16)) 
                                                                                | (((~ 
                                                                                (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__write_table) 
                                                                                >> 2U) 
                                                                                | (2U 
                                                                                == (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__allocate_id_r)))) 
                                                                                & (IData)(vlSelf->__VdfgRegularize_hd87f99a1_43_17)) 
                                                                                | (((~ 
                                                                                (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__write_table) 
                                                                                >> 1U) 
                                                                                | (1U 
                                                                                == (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__allocate_id_r)))) 
                                                                                & (IData)(vlSelf->__VdfgRegularize_hd87f99a1_43_18)) 
                                                                                | ((~ 
                                                                                ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__write_table) 
                                                                                | (0U 
                                                                                == (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__allocate_id_r)))) 
                                                                                & (IData)(vlSelf->__VdfgRegularize_hd87f99a1_43_19))))))))))))))))))))))) 
                                >> 0x20U)));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT____Vcellinp__pipe_reg1__data_in[0U] 
        = __Vtemp_2[0U];
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT____Vcellinp__pipe_reg1__data_in[1U] 
        = __Vtemp_2[1U];
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT____Vcellinp__pipe_reg1__data_in[2U] 
        = ((1U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[1U] 
                  >> 0x1aU)) | ((vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[2U] 
                                 << 6U) | (0x3eU & 
                                           (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[1U] 
                                            >> 0x1aU))));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT____Vcellinp__pipe_reg1__data_in[3U] 
        = ((1U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[2U] 
                  >> 0x1aU)) | ((vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[3U] 
                                 << 6U) | (0x3eU & 
                                           (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[2U] 
                                            >> 0x1aU))));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT____Vcellinp__pipe_reg1__data_in[4U] 
        = ((1U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[3U] 
                  >> 0x1aU)) | ((vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[4U] 
                                 << 6U) | (0x3eU & 
                                           (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[3U] 
                                            >> 0x1aU))));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT____Vcellinp__pipe_reg1__data_in[5U] 
        = ((1U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[4U] 
                  >> 0x1aU)) | ((vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[5U] 
                                 << 6U) | (0x3eU & 
                                           (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[4U] 
                                            >> 0x1aU))));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT____Vcellinp__pipe_reg1__data_in[6U] 
        = ((1U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[5U] 
                  >> 0x1aU)) | ((vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[6U] 
                                 << 6U) | (0x3eU & 
                                           (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[5U] 
                                            >> 0x1aU))));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT____Vcellinp__pipe_reg1__data_in[7U] 
        = ((1U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[6U] 
                  >> 0x1aU)) | ((vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[7U] 
                                 << 6U) | (0x3eU & 
                                           (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[6U] 
                                            >> 0x1aU))));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT____Vcellinp__pipe_reg1__data_in[8U] 
        = ((1U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[7U] 
                  >> 0x1aU)) | ((vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[8U] 
                                 << 6U) | (0x3eU & 
                                           (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[7U] 
                                            >> 0x1aU))));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT____Vcellinp__pipe_reg1__data_in[9U] 
        = ((1U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[8U] 
                  >> 0x1aU)) | ((vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[9U] 
                                 << 6U) | (0x3eU & 
                                           (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[8U] 
                                            >> 0x1aU))));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT____Vcellinp__pipe_reg1__data_in[0xaU] 
        = ((1U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[9U] 
                  >> 0x1aU)) | ((vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0xaU] 
                                 << 6U) | (0x3eU & 
                                           (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[9U] 
                                            >> 0x1aU))));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT____Vcellinp__pipe_reg1__data_in[0xbU] 
        = ((1U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0xaU] 
                  >> 0x1aU)) | ((vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0xbU] 
                                 << 6U) | (0x3eU & 
                                           (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0xaU] 
                                            >> 0x1aU))));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT____Vcellinp__pipe_reg1__data_in[0xcU] 
        = ((1U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0xbU] 
                  >> 0x1aU)) | ((vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0xcU] 
                                 << 6U) | (0x3eU & 
                                           (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0xbU] 
                                            >> 0x1aU))));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT____Vcellinp__pipe_reg1__data_in[0xdU] 
        = ((1U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0xcU] 
                  >> 0x1aU)) | ((vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0xdU] 
                                 << 6U) | (0x3eU & 
                                           (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0xcU] 
                                            >> 0x1aU))));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT____Vcellinp__pipe_reg1__data_in[0xeU] 
        = ((1U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0xdU] 
                  >> 0x1aU)) | ((vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0xeU] 
                                 << 6U) | (0x3eU & 
                                           (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0xdU] 
                                            >> 0x1aU))));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT____Vcellinp__pipe_reg1__data_in[0xfU] 
        = ((1U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0xeU] 
                  >> 0x1aU)) | ((vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0xfU] 
                                 << 6U) | (0x3eU & 
                                           (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0xeU] 
                                            >> 0x1aU))));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT____Vcellinp__pipe_reg1__data_in[0x10U] 
        = ((1U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0xfU] 
                  >> 0x1aU)) | ((vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0x10U] 
                                 << 6U) | (0x3eU & 
                                           (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0xfU] 
                                            >> 0x1aU))));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT____Vcellinp__pipe_reg1__data_in[0x11U] 
        = ((0xe0000000U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0x11U] 
                           << 6U)) | ((0xc000000U & 
                                       (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0x11U] 
                                        << 6U)) | (
                                                   (0x2000000U 
                                                    & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0U] 
                                                       >> 1U)) 
                                                   | ((0x1fffffeU 
                                                       & (((0x800000U 
                                                            & vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0x11U])
                                                            ? 
                                                           ((0xffffc0U 
                                                             & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_tags__DOT__genblk3__BRA__0__KET____DOT__line_rdata 
                                                                << 6U)) 
                                                            | (0x3fU 
                                                               & ((vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0x11U] 
                                                                   << 5U) 
                                                                  | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0x10U] 
                                                                     >> 0x1bU))))
                                                            : 
                                                           ((vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0x11U] 
                                                             << 5U) 
                                                            | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0x10U] 
                                                               >> 0x1bU))) 
                                                          << 1U)) 
                                                      | (1U 
                                                         & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0x10U] 
                                                            >> 0x1aU))))));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT____Vcellinp__pipe_reg1__data_in[0x12U] 
        = vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__dcache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_r;
}

VL_INLINE_OPT void VVortex___024root___nba_comb__TOP__81(VVortex___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    VVortex__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVortex___024root___nba_comb__TOP__81\n"); );
    // Init
    VlWide<17>/*543:0*/ __Vtemp_2;
    // Body
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[0U] 
        = ((0xfffe0000U & vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[0U]) 
           | (((IData)((0U != (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__genblk1__BRA__1__KET____DOT__genblk1__BRA__0__KET____DOT__vs))) 
               << 0x10U) | ((~ (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__next_table_x)) 
                            & (((IData)(vlSelf->__VdfgRegularize_hd87f99a1_14_0) 
                                << 0xfU) | (((IData)(vlSelf->__VdfgRegularize_hd87f99a1_14_1) 
                                             << 0xeU) 
                                            | (((IData)(vlSelf->__VdfgRegularize_hd87f99a1_14_2) 
                                                << 0xdU) 
                                               | (((IData)(vlSelf->__VdfgRegularize_hd87f99a1_14_3) 
                                                   << 0xcU) 
                                                  | (((IData)(vlSelf->__VdfgRegularize_hd87f99a1_14_4) 
                                                      << 0xbU) 
                                                     | (((IData)(vlSelf->__VdfgRegularize_hd87f99a1_14_5) 
                                                         << 0xaU) 
                                                        | (((IData)(vlSelf->__VdfgRegularize_hd87f99a1_14_6) 
                                                            << 9U) 
                                                           | (((IData)(vlSelf->__VdfgRegularize_hd87f99a1_14_7) 
                                                               << 8U) 
                                                              | (((IData)(vlSelf->__VdfgRegularize_hd87f99a1_14_8) 
                                                                  << 7U) 
                                                                 | (((IData)(vlSelf->__VdfgRegularize_hd87f99a1_14_9) 
                                                                     << 6U) 
                                                                    | (((IData)(vlSelf->__VdfgRegularize_hd87f99a1_14_10) 
                                                                        << 5U) 
                                                                       | (((IData)(vlSelf->__VdfgRegularize_hd87f99a1_14_11) 
                                                                           << 4U) 
                                                                          | (((IData)(vlSelf->__VdfgRegularize_hd87f99a1_14_12) 
                                                                              << 3U) 
                                                                             | (((IData)(vlSelf->__VdfgRegularize_hd87f99a1_14_13) 
                                                                                << 2U) 
                                                                                | (((IData)(vlSelf->__VdfgRegularize_hd87f99a1_14_14) 
                                                                                << 1U) 
                                                                                | (IData)(vlSelf->__VdfgRegularize_hd87f99a1_14_15)))))))))))))))))));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__genblk1__BRA__1__KET____DOT__genblk1__BRA__0__KET____DOT__vs 
        = (3U & vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[0U]);
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__genblk1__BRA__1__KET____DOT__genblk1__BRA__1__KET____DOT__vs 
        = (3U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[0U] 
                 >> 2U));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__genblk1__BRA__1__KET____DOT__genblk1__BRA__2__KET____DOT__vs 
        = (3U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[0U] 
                 >> 4U));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__genblk1__BRA__1__KET____DOT__genblk1__BRA__3__KET____DOT__vs 
        = (3U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[0U] 
                 >> 6U));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__genblk1__BRA__1__KET____DOT__genblk1__BRA__4__KET____DOT__vs 
        = (3U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[0U] 
                 >> 8U));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__genblk1__BRA__1__KET____DOT__genblk1__BRA__5__KET____DOT__vs 
        = (3U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[0U] 
                 >> 0xaU));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__genblk1__BRA__1__KET____DOT__genblk1__BRA__6__KET____DOT__vs 
        = (3U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[0U] 
                 >> 0xcU));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__genblk1__BRA__1__KET____DOT__genblk1__BRA__7__KET____DOT__vs 
        = (3U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[0U] 
                 >> 0xeU));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__genblk1__BRA__2__KET____DOT__genblk1__BRA__0__KET____DOT__vs 
        = ((2U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[0U] 
                  >> 0x11U)) | (1U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[0U] 
                                      >> 0x10U)));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__genblk1__BRA__2__KET____DOT__genblk1__BRA__1__KET____DOT__vs 
        = ((2U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[0U] 
                  >> 0x15U)) | (1U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[0U] 
                                      >> 0x14U)));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__genblk1__BRA__2__KET____DOT__genblk1__BRA__2__KET____DOT__vs 
        = ((2U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[0U] 
                  >> 0x19U)) | (1U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[0U] 
                                      >> 0x18U)));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__genblk1__BRA__2__KET____DOT__genblk1__BRA__3__KET____DOT__vs 
        = ((2U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[0U] 
                  >> 0x1dU)) | (1U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[0U] 
                                      >> 0x1cU)));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__genblk1__BRA__3__KET____DOT__genblk1__BRA__0__KET____DOT__vs 
        = ((2U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[1U] 
                  >> 3U)) | (1U & vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[1U]));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__genblk1__BRA__3__KET____DOT__genblk1__BRA__1__KET____DOT__vs 
        = ((2U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[1U] 
                  >> 0xbU)) | (1U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[1U] 
                                     >> 8U)));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__genblk1__BRA__4__KET____DOT__genblk1__BRA__0__KET____DOT__vs 
        = ((2U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[1U] 
                  >> 0x17U)) | (1U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[1U] 
                                      >> 0x10U)));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__addr 
        = ((0xfffffffffffffffeULL & vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__addr) 
           | (IData)((IData)((1U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[0U] 
                                    >> 1U)))));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__addr 
        = ((0xfffffffffffffffbULL & vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__addr) 
           | ((QData)((IData)((1U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[0U] 
                                     >> 3U)))) << 2U));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__addr 
        = ((0xffffffffffffffefULL & vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__addr) 
           | ((QData)((IData)((1U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[0U] 
                                     >> 5U)))) << 4U));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__addr 
        = ((0xffffffffffffffbfULL & vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__addr) 
           | ((QData)((IData)((1U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[0U] 
                                     >> 7U)))) << 6U));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__addr 
        = ((0xfffffffffffffeffULL & vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__addr) 
           | ((QData)((IData)((1U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[0U] 
                                     >> 9U)))) << 8U));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__addr 
        = ((0xfffffffffffffbffULL & vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__addr) 
           | ((QData)((IData)((1U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[0U] 
                                     >> 0xbU)))) << 0xaU));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__addr 
        = ((0xffffffffffffefffULL & vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__addr) 
           | ((QData)((IData)((1U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[0U] 
                                     >> 0xdU)))) << 0xcU));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__addr 
        = ((0xffffffffffffbfffULL & vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__addr) 
           | ((QData)((IData)((1U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__v[0U] 
                                     >> 0xfU)))) << 0xeU));
    __Vtemp_2[0U] = ((0xff800000U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0U] 
                                     << 6U)) | ((0x7ff800U 
                                                 & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0U] 
                                                    << 7U)) 
                                                | ((0x780U 
                                                    & (((0x2000U 
                                                         & vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0x11U])
                                                         ? (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__allocate_id_r)
                                                         : 
                                                        vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0U]) 
                                                       << 7U)) 
                                                   | ((0x78U 
                                                       & ((IData)(
                                                                  (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__prev_sel__DOT__genblk1__DOT__addr 
                                                                   >> 0x30U)) 
                                                          << 3U)) 
                                                      | ((((1U 
                                                            & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0x11U] 
                                                               >> 0xfU)) 
                                                           || ((vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_tags__DOT__genblk3__BRA__0__KET____DOT__line_rdata 
                                                                >> 0x12U) 
                                                               & ((0x3ffffU 
                                                                   & ((vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0x11U] 
                                                                       << 7U) 
                                                                      | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0x10U] 
                                                                         >> 0x19U))) 
                                                                  == 
                                                                  (0x3ffffU 
                                                                   & vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_tags__DOT__genblk3__BRA__0__KET____DOT__line_rdata)))) 
                                                          << 2U) 
                                                         | (((~ 
                                                              (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__write_table) 
                                                                >> 0xfU) 
                                                               | (0xfU 
                                                                  == (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__allocate_id_r)))) 
                                                             & (IData)(vlSelf->__VdfgRegularize_hd87f99a1_14_0)) 
                                                            | (((~ 
                                                                 (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__write_table) 
                                                                   >> 0xeU) 
                                                                  | (0xeU 
                                                                     == (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__allocate_id_r)))) 
                                                                & (IData)(vlSelf->__VdfgRegularize_hd87f99a1_14_1)) 
                                                               | (((~ 
                                                                    (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__write_table) 
                                                                      >> 0xdU) 
                                                                     | (0xdU 
                                                                        == (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__allocate_id_r)))) 
                                                                   & (IData)(vlSelf->__VdfgRegularize_hd87f99a1_14_2)) 
                                                                  | (((~ 
                                                                       (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__write_table) 
                                                                         >> 0xcU) 
                                                                        | (0xcU 
                                                                           == (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__allocate_id_r)))) 
                                                                      & (IData)(vlSelf->__VdfgRegularize_hd87f99a1_14_3)) 
                                                                     | (((~ 
                                                                          (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__write_table) 
                                                                            >> 0xbU) 
                                                                           | (0xbU 
                                                                              == (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__allocate_id_r)))) 
                                                                         & (IData)(vlSelf->__VdfgRegularize_hd87f99a1_14_4)) 
                                                                        | (((~ 
                                                                             (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__write_table) 
                                                                               >> 0xaU) 
                                                                              | (0xaU 
                                                                                == (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__allocate_id_r)))) 
                                                                            & (IData)(vlSelf->__VdfgRegularize_hd87f99a1_14_5)) 
                                                                           | (((~ 
                                                                                (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__write_table) 
                                                                                >> 9U) 
                                                                                | (9U 
                                                                                == (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__allocate_id_r)))) 
                                                                               & (IData)(vlSelf->__VdfgRegularize_hd87f99a1_14_6)) 
                                                                              | (((~ 
                                                                                (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__write_table) 
                                                                                >> 8U) 
                                                                                | (8U 
                                                                                == (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__allocate_id_r)))) 
                                                                                & (IData)(vlSelf->__VdfgRegularize_hd87f99a1_14_7)) 
                                                                                | (((~ 
                                                                                (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__write_table) 
                                                                                >> 7U) 
                                                                                | (7U 
                                                                                == (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__allocate_id_r)))) 
                                                                                & (IData)(vlSelf->__VdfgRegularize_hd87f99a1_14_8)) 
                                                                                | (((~ 
                                                                                (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__write_table) 
                                                                                >> 6U) 
                                                                                | (6U 
                                                                                == (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__allocate_id_r)))) 
                                                                                & (IData)(vlSelf->__VdfgRegularize_hd87f99a1_14_9)) 
                                                                                | (((~ 
                                                                                (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__write_table) 
                                                                                >> 5U) 
                                                                                | (5U 
                                                                                == (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__allocate_id_r)))) 
                                                                                & (IData)(vlSelf->__VdfgRegularize_hd87f99a1_14_10)) 
                                                                                | (((~ 
                                                                                (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__write_table) 
                                                                                >> 4U) 
                                                                                | (4U 
                                                                                == (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__allocate_id_r)))) 
                                                                                & (IData)(vlSelf->__VdfgRegularize_hd87f99a1_14_11)) 
                                                                                | (((~ 
                                                                                (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__write_table) 
                                                                                >> 3U) 
                                                                                | (3U 
                                                                                == (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__allocate_id_r)))) 
                                                                                & (IData)(vlSelf->__VdfgRegularize_hd87f99a1_14_12)) 
                                                                                | (((~ 
                                                                                (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__write_table) 
                                                                                >> 2U) 
                                                                                | (2U 
                                                                                == (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__allocate_id_r)))) 
                                                                                & (IData)(vlSelf->__VdfgRegularize_hd87f99a1_14_13)) 
                                                                                | (((~ 
                                                                                (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__write_table) 
                                                                                >> 1U) 
                                                                                | (1U 
                                                                                == (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__allocate_id_r)))) 
                                                                                & (IData)(vlSelf->__VdfgRegularize_hd87f99a1_14_14)) 
                                                                                | ((~ 
                                                                                ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__write_table) 
                                                                                | (0U 
                                                                                == (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_mshr__DOT__allocate_id_r)))) 
                                                                                & (IData)(vlSelf->__VdfgRegularize_hd87f99a1_14_15))))))))))))))))))))));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT____Vcellinp__pipe_reg1__data_in[0U] 
        = __Vtemp_2[0U];
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT____Vcellinp__pipe_reg1__data_in[1U] 
        = (((0x7fffc0U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[1U] 
                          << 6U)) | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0U] 
                                     >> 0x1aU)) | (0xff800000U 
                                                   & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[1U] 
                                                      << 6U)));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT____Vcellinp__pipe_reg1__data_in[2U] 
        = (((0x7fffc0U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[2U] 
                          << 6U)) | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[1U] 
                                     >> 0x1aU)) | (0xff800000U 
                                                   & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[2U] 
                                                      << 6U)));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT____Vcellinp__pipe_reg1__data_in[3U] 
        = (((0x7fffc0U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[3U] 
                          << 6U)) | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[2U] 
                                     >> 0x1aU)) | (0xff800000U 
                                                   & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[3U] 
                                                      << 6U)));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT____Vcellinp__pipe_reg1__data_in[4U] 
        = (((0x7fffc0U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[4U] 
                          << 6U)) | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[3U] 
                                     >> 0x1aU)) | (0xff800000U 
                                                   & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[4U] 
                                                      << 6U)));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT____Vcellinp__pipe_reg1__data_in[5U] 
        = (((0x7fffc0U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[5U] 
                          << 6U)) | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[4U] 
                                     >> 0x1aU)) | (0xff800000U 
                                                   & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[5U] 
                                                      << 6U)));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT____Vcellinp__pipe_reg1__data_in[6U] 
        = (((0x7fffc0U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[6U] 
                          << 6U)) | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[5U] 
                                     >> 0x1aU)) | (0xff800000U 
                                                   & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[6U] 
                                                      << 6U)));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT____Vcellinp__pipe_reg1__data_in[7U] 
        = (((0x7fffc0U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[7U] 
                          << 6U)) | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[6U] 
                                     >> 0x1aU)) | (0xff800000U 
                                                   & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[7U] 
                                                      << 6U)));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT____Vcellinp__pipe_reg1__data_in[8U] 
        = (((0x7fffc0U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[8U] 
                          << 6U)) | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[7U] 
                                     >> 0x1aU)) | (0xff800000U 
                                                   & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[8U] 
                                                      << 6U)));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT____Vcellinp__pipe_reg1__data_in[9U] 
        = (((0x7fffc0U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[9U] 
                          << 6U)) | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[8U] 
                                     >> 0x1aU)) | (0xff800000U 
                                                   & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[9U] 
                                                      << 6U)));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT____Vcellinp__pipe_reg1__data_in[0xaU] 
        = (((0x7fffc0U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0xaU] 
                          << 6U)) | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[9U] 
                                     >> 0x1aU)) | (0xff800000U 
                                                   & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0xaU] 
                                                      << 6U)));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT____Vcellinp__pipe_reg1__data_in[0xbU] 
        = (((0x7fffc0U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0xbU] 
                          << 6U)) | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0xaU] 
                                     >> 0x1aU)) | (0xff800000U 
                                                   & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0xbU] 
                                                      << 6U)));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT____Vcellinp__pipe_reg1__data_in[0xcU] 
        = (((0x7fffc0U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0xcU] 
                          << 6U)) | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0xbU] 
                                     >> 0x1aU)) | (0xff800000U 
                                                   & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0xcU] 
                                                      << 6U)));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT____Vcellinp__pipe_reg1__data_in[0xdU] 
        = (((0x7fffc0U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0xdU] 
                          << 6U)) | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0xcU] 
                                     >> 0x1aU)) | (0xff800000U 
                                                   & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0xdU] 
                                                      << 6U)));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT____Vcellinp__pipe_reg1__data_in[0xeU] 
        = (((0x7fffc0U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0xeU] 
                          << 6U)) | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0xdU] 
                                     >> 0x1aU)) | (0xff800000U 
                                                   & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0xeU] 
                                                      << 6U)));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT____Vcellinp__pipe_reg1__data_in[0xfU] 
        = (((0x7fffc0U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0xfU] 
                          << 6U)) | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0xeU] 
                                     >> 0x1aU)) | (0xff800000U 
                                                   & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0xfU] 
                                                      << 6U)));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT____Vcellinp__pipe_reg1__data_in[0x10U] 
        = ((((0x8000U & vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0x11U])
              ? ((0x3ffff00U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_tags__DOT__genblk3__BRA__0__KET____DOT__line_rdata 
                                << 8U)) | (0xffU & 
                                           (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0x10U] 
                                            >> 0x11U)))
              : ((vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0x11U] 
                  << 0xfU) | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0x10U] 
                              >> 0x11U))) << 0x17U) 
           | ((0x7fffc0U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0x10U] 
                            << 6U)) | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0xfU] 
                                       >> 0x1aU)));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT____Vcellinp__pipe_reg1__data_in[0x11U] 
        = (((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_r) 
            << 0x18U) | ((0xe00000U & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0x11U] 
                                       << 6U)) | ((0xc0000U 
                                                   & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0x11U] 
                                                      << 6U)) 
                                                  | ((0x20000U 
                                                      & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0U] 
                                                         << 1U)) 
                                                     | (0x1ffffU 
                                                        & (((0x8000U 
                                                             & vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0x11U])
                                                             ? 
                                                            ((0x3ffff00U 
                                                              & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__cache_tags__DOT__genblk3__BRA__0__KET____DOT__line_rdata 
                                                                 << 8U)) 
                                                             | (0xffU 
                                                                & (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0x10U] 
                                                                   >> 0x11U)))
                                                             : 
                                                            ((vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0x11U] 
                                                              << 0xfU) 
                                                             | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__icache__DOT__caches__BRA__0__KET____DOT__cache_wrap__DOT__genblk3__DOT__cache__DOT__banks__BRA__0__KET____DOT__bank__DOT__pipe_reg0__DOT__genblk1__DOT__genblk1__DOT__genblk1__DOT__value_d[0x10U] 
                                                                >> 0x11U))) 
                                                           >> 9U))))));
}

VL_INLINE_OPT void VVortex___024root___nba_comb__TOP__97(VVortex___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    VVortex__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVortex___024root___nba_comb__TOP__97\n"); );
    // Init
    VlWide<6>/*191:0*/ __Vtemp_4;
    // Body
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__rr_arbiter__DOT__genblk1__DOT__onehot_encoder__DOT__genblk1__DOT__v 
        = ((0xfe0U & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__rr_arbiter__DOT__genblk1__DOT__onehot_encoder__DOT__genblk1__DOT__v)) 
           | (((IData)((0U != (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__rr_arbiter__DOT__genblk1__DOT__onehot_encoder__DOT__genblk1__DOT__genblk1__BRA__1__KET____DOT__genblk1__BRA__0__KET____DOT__vs))) 
               << 4U) | (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT__genblk1__DOT__genblk1__DOT__arb_onehot)));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__rr_arbiter__DOT__genblk1__DOT__onehot_encoder__DOT__genblk1__DOT__genblk1__BRA__1__KET____DOT__genblk1__BRA__0__KET____DOT__vs 
        = (3U & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__rr_arbiter__DOT__genblk1__DOT__onehot_encoder__DOT__genblk1__DOT__v));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__rr_arbiter__DOT__genblk1__DOT__onehot_encoder__DOT__genblk1__DOT__genblk1__BRA__1__KET____DOT__genblk1__BRA__1__KET____DOT__vs 
        = (3U & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__rr_arbiter__DOT__genblk1__DOT__onehot_encoder__DOT__genblk1__DOT__v) 
                 >> 2U));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__rr_arbiter__DOT__genblk1__DOT__onehot_encoder__DOT__genblk1__DOT__genblk1__BRA__2__KET____DOT__genblk1__BRA__0__KET____DOT__vs 
        = ((2U & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__rr_arbiter__DOT__genblk1__DOT__onehot_encoder__DOT__genblk1__DOT__v) 
                  >> 5U)) | (1U & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__rr_arbiter__DOT__genblk1__DOT__onehot_encoder__DOT__genblk1__DOT__v) 
                                   >> 4U)));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__rr_arbiter__DOT__genblk1__DOT__onehot_encoder__DOT__genblk1__DOT__addr 
        = ((0xfeU & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__rr_arbiter__DOT__genblk1__DOT__onehot_encoder__DOT__genblk1__DOT__addr)) 
           | (1U & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__rr_arbiter__DOT__genblk1__DOT__onehot_encoder__DOT__genblk1__DOT__v) 
                    >> 1U)));
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__rr_arbiter__DOT__genblk1__DOT__onehot_encoder__DOT__genblk1__DOT__addr 
        = ((0xfbU & (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__rr_arbiter__DOT__genblk1__DOT__onehot_encoder__DOT__genblk1__DOT__addr)) 
           | (4U & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__rr_arbiter__DOT__genblk1__DOT__onehot_encoder__DOT__genblk1__DOT__v) 
                    >> 1U)));
    if ((0x2bfU >= (0x3ffU & ((IData)(0xb0U) * (3U 
                                                & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__rr_arbiter__DOT__genblk1__DOT__onehot_encoder__DOT__genblk1__DOT__addr) 
                                                   >> 4U)))))) {
        __Vtemp_4[0U] = (((0U == (0x1fU & ((IData)(0xb0U) 
                                           * (3U & 
                                              ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__rr_arbiter__DOT__genblk1__DOT__onehot_encoder__DOT__genblk1__DOT__addr) 
                                               >> 4U)))))
                           ? 0U : (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__data_in[
                                   ((IData)(1U) + (0x1fU 
                                                   & (((IData)(0xb0U) 
                                                       * 
                                                       (3U 
                                                        & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__rr_arbiter__DOT__genblk1__DOT__onehot_encoder__DOT__genblk1__DOT__addr) 
                                                           >> 4U))) 
                                                      >> 5U)))] 
                                   << ((IData)(0x20U) 
                                       - (0x1fU & ((IData)(0xb0U) 
                                                   * 
                                                   (3U 
                                                    & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__rr_arbiter__DOT__genblk1__DOT__onehot_encoder__DOT__genblk1__DOT__addr) 
                                                       >> 4U))))))) 
                         | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__data_in[
                            (0x1fU & (((IData)(0xb0U) 
                                       * (3U & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__rr_arbiter__DOT__genblk1__DOT__onehot_encoder__DOT__genblk1__DOT__addr) 
                                                >> 4U))) 
                                      >> 5U))] >> (0x1fU 
                                                   & ((IData)(0xb0U) 
                                                      * 
                                                      (3U 
                                                       & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__rr_arbiter__DOT__genblk1__DOT__onehot_encoder__DOT__genblk1__DOT__addr) 
                                                          >> 4U))))));
        __Vtemp_4[1U] = (((0U == (0x1fU & ((IData)(0xb0U) 
                                           * (3U & 
                                              ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__rr_arbiter__DOT__genblk1__DOT__onehot_encoder__DOT__genblk1__DOT__addr) 
                                               >> 4U)))))
                           ? 0U : (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__data_in[
                                   ((IData)(2U) + (0x1fU 
                                                   & (((IData)(0xb0U) 
                                                       * 
                                                       (3U 
                                                        & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__rr_arbiter__DOT__genblk1__DOT__onehot_encoder__DOT__genblk1__DOT__addr) 
                                                           >> 4U))) 
                                                      >> 5U)))] 
                                   << ((IData)(0x20U) 
                                       - (0x1fU & ((IData)(0xb0U) 
                                                   * 
                                                   (3U 
                                                    & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__rr_arbiter__DOT__genblk1__DOT__onehot_encoder__DOT__genblk1__DOT__addr) 
                                                       >> 4U))))))) 
                         | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__data_in[
                            ((IData)(1U) + (0x1fU & 
                                            (((IData)(0xb0U) 
                                              * (3U 
                                                 & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__rr_arbiter__DOT__genblk1__DOT__onehot_encoder__DOT__genblk1__DOT__addr) 
                                                    >> 4U))) 
                                             >> 5U)))] 
                            >> (0x1fU & ((IData)(0xb0U) 
                                         * (3U & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__rr_arbiter__DOT__genblk1__DOT__onehot_encoder__DOT__genblk1__DOT__addr) 
                                                  >> 4U))))));
        __Vtemp_4[2U] = (((0U == (0x1fU & ((IData)(0xb0U) 
                                           * (3U & 
                                              ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__rr_arbiter__DOT__genblk1__DOT__onehot_encoder__DOT__genblk1__DOT__addr) 
                                               >> 4U)))))
                           ? 0U : (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__data_in[
                                   ((IData)(3U) + (0x1fU 
                                                   & (((IData)(0xb0U) 
                                                       * 
                                                       (3U 
                                                        & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__rr_arbiter__DOT__genblk1__DOT__onehot_encoder__DOT__genblk1__DOT__addr) 
                                                           >> 4U))) 
                                                      >> 5U)))] 
                                   << ((IData)(0x20U) 
                                       - (0x1fU & ((IData)(0xb0U) 
                                                   * 
                                                   (3U 
                                                    & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__rr_arbiter__DOT__genblk1__DOT__onehot_encoder__DOT__genblk1__DOT__addr) 
                                                       >> 4U))))))) 
                         | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__data_in[
                            ((IData)(2U) + (0x1fU & 
                                            (((IData)(0xb0U) 
                                              * (3U 
                                                 & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__rr_arbiter__DOT__genblk1__DOT__onehot_encoder__DOT__genblk1__DOT__addr) 
                                                    >> 4U))) 
                                             >> 5U)))] 
                            >> (0x1fU & ((IData)(0xb0U) 
                                         * (3U & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__rr_arbiter__DOT__genblk1__DOT__onehot_encoder__DOT__genblk1__DOT__addr) 
                                                  >> 4U))))));
        __Vtemp_4[3U] = (((0U == (0x1fU & ((IData)(0xb0U) 
                                           * (3U & 
                                              ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__rr_arbiter__DOT__genblk1__DOT__onehot_encoder__DOT__genblk1__DOT__addr) 
                                               >> 4U)))))
                           ? 0U : (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__data_in[
                                   ((IData)(4U) + (0x1fU 
                                                   & (((IData)(0xb0U) 
                                                       * 
                                                       (3U 
                                                        & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__rr_arbiter__DOT__genblk1__DOT__onehot_encoder__DOT__genblk1__DOT__addr) 
                                                           >> 4U))) 
                                                      >> 5U)))] 
                                   << ((IData)(0x20U) 
                                       - (0x1fU & ((IData)(0xb0U) 
                                                   * 
                                                   (3U 
                                                    & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__rr_arbiter__DOT__genblk1__DOT__onehot_encoder__DOT__genblk1__DOT__addr) 
                                                       >> 4U))))))) 
                         | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__data_in[
                            ((IData)(3U) + (0x1fU & 
                                            (((IData)(0xb0U) 
                                              * (3U 
                                                 & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__rr_arbiter__DOT__genblk1__DOT__onehot_encoder__DOT__genblk1__DOT__addr) 
                                                    >> 4U))) 
                                             >> 5U)))] 
                            >> (0x1fU & ((IData)(0xb0U) 
                                         * (3U & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__rr_arbiter__DOT__genblk1__DOT__onehot_encoder__DOT__genblk1__DOT__addr) 
                                                  >> 4U))))));
        __Vtemp_4[4U] = (((0U == (0x1fU & ((IData)(0xb0U) 
                                           * (3U & 
                                              ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__rr_arbiter__DOT__genblk1__DOT__onehot_encoder__DOT__genblk1__DOT__addr) 
                                               >> 4U)))))
                           ? 0U : (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__data_in[
                                   ((IData)(5U) + (0x1fU 
                                                   & (((IData)(0xb0U) 
                                                       * 
                                                       (3U 
                                                        & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__rr_arbiter__DOT__genblk1__DOT__onehot_encoder__DOT__genblk1__DOT__addr) 
                                                           >> 4U))) 
                                                      >> 5U)))] 
                                   << ((IData)(0x20U) 
                                       - (0x1fU & ((IData)(0xb0U) 
                                                   * 
                                                   (3U 
                                                    & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__rr_arbiter__DOT__genblk1__DOT__onehot_encoder__DOT__genblk1__DOT__addr) 
                                                       >> 4U))))))) 
                         | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__data_in[
                            ((IData)(4U) + (0x1fU & 
                                            (((IData)(0xb0U) 
                                              * (3U 
                                                 & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__rr_arbiter__DOT__genblk1__DOT__onehot_encoder__DOT__genblk1__DOT__addr) 
                                                    >> 4U))) 
                                             >> 5U)))] 
                            >> (0x1fU & ((IData)(0xb0U) 
                                         * (3U & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__rr_arbiter__DOT__genblk1__DOT__onehot_encoder__DOT__genblk1__DOT__addr) 
                                                  >> 4U))))));
        __Vtemp_4[5U] = (((0U == (0x1fU & ((IData)(0xb0U) 
                                           * (3U & 
                                              ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__rr_arbiter__DOT__genblk1__DOT__onehot_encoder__DOT__genblk1__DOT__addr) 
                                               >> 4U)))))
                           ? 0U : (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__data_in[
                                   ((IData)(6U) + (0x1fU 
                                                   & (((IData)(0xb0U) 
                                                       * 
                                                       (3U 
                                                        & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__rr_arbiter__DOT__genblk1__DOT__onehot_encoder__DOT__genblk1__DOT__addr) 
                                                           >> 4U))) 
                                                      >> 5U)))] 
                                   << ((IData)(0x20U) 
                                       - (0x1fU & ((IData)(0xb0U) 
                                                   * 
                                                   (3U 
                                                    & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__rr_arbiter__DOT__genblk1__DOT__onehot_encoder__DOT__genblk1__DOT__addr) 
                                                       >> 4U))))))) 
                         | (vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__data_in[
                            ((IData)(5U) + (0x1fU & 
                                            (((IData)(0xb0U) 
                                              * (3U 
                                                 & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__rr_arbiter__DOT__genblk1__DOT__onehot_encoder__DOT__genblk1__DOT__addr) 
                                                    >> 4U))) 
                                             >> 5U)))] 
                            >> (0x1fU & ((IData)(0xb0U) 
                                         * (3U & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__rr_arbiter__DOT__genblk1__DOT__onehot_encoder__DOT__genblk1__DOT__addr) 
                                                  >> 4U))))));
    } else {
        __Vtemp_4[0U] = vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT____Vxrand_h519368a0__0[0U];
        __Vtemp_4[1U] = vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT____Vxrand_h519368a0__0[1U];
        __Vtemp_4[2U] = vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT____Vxrand_h519368a0__0[2U];
        __Vtemp_4[3U] = vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT____Vxrand_h519368a0__0[3U];
        __Vtemp_4[4U] = vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT____Vxrand_h519368a0__0[4U];
        __Vtemp_4[5U] = vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT____Vxrand_h519368a0__0[5U];
    }
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT__genblk1__DOT__genblk1__DOT__out_buf__DOT__genblk1__DOT__pipe_buffer__DOT____Vcellinp__genblk1__DOT__genblk1__BRA__0__KET____DOT__pipe_register__data_in[0U] 
        = __Vtemp_4[0U];
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT__genblk1__DOT__genblk1__DOT__out_buf__DOT__genblk1__DOT__pipe_buffer__DOT____Vcellinp__genblk1__DOT__genblk1__BRA__0__KET____DOT__pipe_register__data_in[1U] 
        = __Vtemp_4[1U];
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT__genblk1__DOT__genblk1__DOT__out_buf__DOT__genblk1__DOT__pipe_buffer__DOT____Vcellinp__genblk1__DOT__genblk1__BRA__0__KET____DOT__pipe_register__data_in[2U] 
        = __Vtemp_4[2U];
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT__genblk1__DOT__genblk1__DOT__out_buf__DOT__genblk1__DOT__pipe_buffer__DOT____Vcellinp__genblk1__DOT__genblk1__BRA__0__KET____DOT__pipe_register__data_in[3U] 
        = __Vtemp_4[3U];
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT__genblk1__DOT__genblk1__DOT__out_buf__DOT__genblk1__DOT__pipe_buffer__DOT____Vcellinp__genblk1__DOT__genblk1__BRA__0__KET____DOT__pipe_register__data_in[4U] 
        = __Vtemp_4[4U];
    vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT__genblk1__DOT__genblk1__DOT__out_buf__DOT__genblk1__DOT__pipe_buffer__DOT____Vcellinp__genblk1__DOT__genblk1__BRA__0__KET____DOT__pipe_register__data_in[5U] 
        = (((IData)((0U != (IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__valid_in))) 
            << 0x12U) | ((0x30000U & ((IData)(vlSelf->Vortex__DOT__clusters__BRA__0__KET____DOT__cluster__DOT__sockets__BRA__0__KET____DOT__socket__DOT__cores__BRA__0__KET____DOT__core__DOT__commit__DOT__genblk1__BRA__0__KET____DOT__commit_arb__DOT__genblk1__DOT__genblk1__DOT__arbiter__DOT__genblk1__DOT__rr_arbiter__DOT__genblk1__DOT__onehot_encoder__DOT__genblk1__DOT__addr) 
                                      << 0xcU)) | (0xffffU 
                                                   & __Vtemp_4[5U])));
}

void VVortex___024root___eval_triggers__act(VVortex___024root* vlSelf);
void VVortex___024root___eval_act(VVortex___024root* vlSelf);

bool VVortex___024root___eval_phase__act(VVortex___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    VVortex__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVortex___024root___eval_phase__act\n"); );
    // Init
    VlTriggerVec<229> __VpreTriggered;
    CData/*0:0*/ __VactExecute;
    // Body
    VVortex___024root___eval_triggers__act(vlSelf);
    __VactExecute = vlSelf->__VactTriggered.any();
    if (__VactExecute) {
        __VpreTriggered.andNot(vlSelf->__VactTriggered, vlSelf->__VnbaTriggered);
        vlSelf->__VnbaTriggered.thisOr(vlSelf->__VactTriggered);
        VVortex___024root___eval_act(vlSelf);
    }
    return (__VactExecute);
}

void VVortex___024root___eval_nba(VVortex___024root* vlSelf);

bool VVortex___024root___eval_phase__nba(VVortex___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    VVortex__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVortex___024root___eval_phase__nba\n"); );
    // Init
    CData/*0:0*/ __VnbaExecute;
    // Body
    __VnbaExecute = vlSelf->__VnbaTriggered.any();
    if (__VnbaExecute) {
        VVortex___024root___eval_nba(vlSelf);
        vlSelf->__VnbaTriggered.clear();
    }
    return (__VnbaExecute);
}

#ifdef VL_DEBUG
VL_ATTR_COLD void VVortex___024root___dump_triggers__ico(VVortex___024root* vlSelf);
#endif  // VL_DEBUG
bool VVortex___024root___eval_phase__ico(VVortex___024root* vlSelf);
#ifdef VL_DEBUG
VL_ATTR_COLD void VVortex___024root___dump_triggers__nba(VVortex___024root* vlSelf);
#endif  // VL_DEBUG
#ifdef VL_DEBUG
VL_ATTR_COLD void VVortex___024root___dump_triggers__act(VVortex___024root* vlSelf);
#endif  // VL_DEBUG

void VVortex___024root___eval(VVortex___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    VVortex__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVortex___024root___eval\n"); );
    // Init
    IData/*31:0*/ __VicoIterCount;
    CData/*0:0*/ __VicoContinue;
    IData/*31:0*/ __VnbaIterCount;
    CData/*0:0*/ __VnbaContinue;
    // Body
    __VicoIterCount = 0U;
    vlSelf->__VicoFirstIteration = 1U;
    __VicoContinue = 1U;
    while (__VicoContinue) {
        if (VL_UNLIKELY((0x64U < __VicoIterCount))) {
#ifdef VL_DEBUG
            VVortex___024root___dump_triggers__ico(vlSelf);
#endif
            VL_FATAL_MT("/home/ecegridfs/a/socet39/socet/vortex-f24/hw/rtl/Vortex.sv", 16, "", "Input combinational region did not converge.");
        }
        __VicoIterCount = ((IData)(1U) + __VicoIterCount);
        __VicoContinue = 0U;
        if (VVortex___024root___eval_phase__ico(vlSelf)) {
            __VicoContinue = 1U;
        }
        vlSelf->__VicoFirstIteration = 0U;
    }
    __VnbaIterCount = 0U;
    __VnbaContinue = 1U;
    while (__VnbaContinue) {
        if (VL_UNLIKELY((0x64U < __VnbaIterCount))) {
#ifdef VL_DEBUG
            VVortex___024root___dump_triggers__nba(vlSelf);
#endif
            VL_FATAL_MT("/home/ecegridfs/a/socet39/socet/vortex-f24/hw/rtl/Vortex.sv", 16, "", "NBA region did not converge.");
        }
        __VnbaIterCount = ((IData)(1U) + __VnbaIterCount);
        __VnbaContinue = 0U;
        vlSelf->__VactIterCount = 0U;
        vlSelf->__VactContinue = 1U;
        while (vlSelf->__VactContinue) {
            if (VL_UNLIKELY((0x64U < vlSelf->__VactIterCount))) {
#ifdef VL_DEBUG
                VVortex___024root___dump_triggers__act(vlSelf);
#endif
                VL_FATAL_MT("/home/ecegridfs/a/socet39/socet/vortex-f24/hw/rtl/Vortex.sv", 16, "", "Active region did not converge.");
            }
            vlSelf->__VactIterCount = ((IData)(1U) 
                                       + vlSelf->__VactIterCount);
            vlSelf->__VactContinue = 0U;
            if (VVortex___024root___eval_phase__act(vlSelf)) {
                vlSelf->__VactContinue = 1U;
            }
        }
        if (VVortex___024root___eval_phase__nba(vlSelf)) {
            __VnbaContinue = 1U;
        }
    }
}

#ifdef VL_DEBUG
void VVortex___024root___eval_debug_assertions(VVortex___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    VVortex__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVortex___024root___eval_debug_assertions\n"); );
    // Body
    if (VL_UNLIKELY((vlSelf->clk & 0xfeU))) {
        Verilated::overWidthError("clk");}
    if (VL_UNLIKELY((vlSelf->reset & 0xfeU))) {
        Verilated::overWidthError("reset");}
    if (VL_UNLIKELY((vlSelf->mem_req_ready & 0xfeU))) {
        Verilated::overWidthError("mem_req_ready");}
    if (VL_UNLIKELY((vlSelf->mem_rsp_valid & 0xfeU))) {
        Verilated::overWidthError("mem_rsp_valid");}
    if (VL_UNLIKELY((vlSelf->dcr_wr_valid & 0xfeU))) {
        Verilated::overWidthError("dcr_wr_valid");}
    if (VL_UNLIKELY((vlSelf->dcr_wr_addr & 0xf000U))) {
        Verilated::overWidthError("dcr_wr_addr");}
}
#endif  // VL_DEBUG
