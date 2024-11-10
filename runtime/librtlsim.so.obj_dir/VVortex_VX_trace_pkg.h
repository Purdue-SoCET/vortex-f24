// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See VVortex.h for the primary calling header

#ifndef VERILATED_VVORTEX_VX_TRACE_PKG_H_
#define VERILATED_VVORTEX_VX_TRACE_PKG_H_  // guard

#include "verilated.h"


class VVortex__Syms;

class alignas(VL_CACHE_LINE_BYTES) VVortex_VX_trace_pkg final : public VerilatedModule {
  public:

    // INTERNAL VARIABLES
    VVortex__Syms* const vlSymsp;

    // CONSTRUCTORS
    VVortex_VX_trace_pkg(VVortex__Syms* symsp, const char* v__name);
    ~VVortex_VX_trace_pkg();
    VL_UNCOPYABLE(VVortex_VX_trace_pkg);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
};


#endif  // guard
