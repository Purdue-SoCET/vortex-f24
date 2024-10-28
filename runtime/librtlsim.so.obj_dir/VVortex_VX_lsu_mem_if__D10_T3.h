// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See VVortex.h for the primary calling header

#ifndef VERILATED_VVORTEX_VX_LSU_MEM_IF__D10_T3_H_
#define VERILATED_VVORTEX_VX_LSU_MEM_IF__D10_T3_H_  // guard

#include "verilated.h"


class VVortex__Syms;

class alignas(VL_CACHE_LINE_BYTES) VVortex_VX_lsu_mem_if__D10_T3 final : public VerilatedModule {
  public:

    // DESIGN SPECIFIC STATE
    VlWide<6>/*179:0*/ req_data;

    // INTERNAL VARIABLES
    VVortex__Syms* const vlSymsp;

    // CONSTRUCTORS
    VVortex_VX_lsu_mem_if__D10_T3(VVortex__Syms* symsp, const char* v__name);
    ~VVortex_VX_lsu_mem_if__D10_T3();
    VL_UNCOPYABLE(VVortex_VX_lsu_mem_if__D10_T3);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
};

std::string VL_TO_STRING(const VVortex_VX_lsu_mem_if__D10_T3* obj);

#endif  // guard
