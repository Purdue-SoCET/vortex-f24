// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See VVortex.h for the primary calling header

#ifndef VERILATED_VVORTEX_VX_EXECUTE_IF__N4_H_
#define VERILATED_VVORTEX_VX_EXECUTE_IF__N4_H_  // guard

#include "verilated.h"


class VVortex__Syms;

class alignas(VL_CACHE_LINE_BYTES) VVortex_VX_execute_if__N4 final : public VerilatedModule {
  public:

    // DESIGN SPECIFIC STATE
    CData/*0:0*/ valid;
    CData/*0:0*/ ready;

    // INTERNAL VARIABLES
    VVortex__Syms* const vlSymsp;

    // CONSTRUCTORS
    VVortex_VX_execute_if__N4(VVortex__Syms* symsp, const char* v__name);
    ~VVortex_VX_execute_if__N4();
    VL_UNCOPYABLE(VVortex_VX_execute_if__N4);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
};

std::string VL_TO_STRING(const VVortex_VX_execute_if__N4* obj);

#endif  // guard
