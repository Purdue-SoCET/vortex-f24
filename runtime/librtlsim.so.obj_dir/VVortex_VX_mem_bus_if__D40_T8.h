// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See VVortex.h for the primary calling header

#ifndef VERILATED_VVORTEX_VX_MEM_BUS_IF__D40_T8_H_
#define VERILATED_VVORTEX_VX_MEM_BUS_IF__D40_T8_H_  // guard

#include "verilated.h"


class VVortex__Syms;

class alignas(VL_CACHE_LINE_BYTES) VVortex_VX_mem_bus_if__D40_T8 final : public VerilatedModule {
  public:

    // DESIGN SPECIFIC STATE
    VlWide<17>/*519:0*/ rsp_data;

    // INTERNAL VARIABLES
    VVortex__Syms* const vlSymsp;

    // CONSTRUCTORS
    VVortex_VX_mem_bus_if__D40_T8(VVortex__Syms* symsp, const char* v__name);
    ~VVortex_VX_mem_bus_if__D40_T8();
    VL_UNCOPYABLE(VVortex_VX_mem_bus_if__D40_T8);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
};

std::string VL_TO_STRING(const VVortex_VX_mem_bus_if__D40_T8* obj);

#endif  // guard
