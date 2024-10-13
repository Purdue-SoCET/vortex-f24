// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See VVortex.h for the primary calling header

#ifndef VERILATED_VVORTEX_VX_FETCH_IF_H_
#define VERILATED_VVORTEX_VX_FETCH_IF_H_  // guard

#include "verilated.h"


class VVortex__Syms;

class alignas(VL_CACHE_LINE_BYTES) VVortex_VX_fetch_if final : public VerilatedModule {
  public:

    // DESIGN SPECIFIC STATE
    VlWide<3>/*69:0*/ data;

    // INTERNAL VARIABLES
    VVortex__Syms* const vlSymsp;

    // CONSTRUCTORS
    VVortex_VX_fetch_if(VVortex__Syms* symsp, const char* v__name);
    ~VVortex_VX_fetch_if();
    VL_UNCOPYABLE(VVortex_VX_fetch_if);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
};

std::string VL_TO_STRING(const VVortex_VX_fetch_if* obj);

#endif  // guard
