// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See VVortex.h for the primary calling header

#include "VVortex__pch.h"
#include "VVortex__Syms.h"
#include "VVortex___024root.h"

void VVortex___024root___ctor_var_reset(VVortex___024root* vlSelf);

VVortex___024root::VVortex___024root(VVortex__Syms* symsp, const char* v__name)
    : VerilatedModule{v__name}
    , vlSymsp{symsp}
 {
    // Reset structure values
    VVortex___024root___ctor_var_reset(this);
}

void VVortex___024root::__Vconfigure(bool first) {
    (void)first;  // Prevent unused variable warning
}

VVortex___024root::~VVortex___024root() {
}
