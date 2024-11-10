// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See VVortex.h for the primary calling header

#include "VVortex__pch.h"
#include "VVortex_VX_schedule_if.h"

std::string VL_TO_STRING(const VVortex_VX_schedule_if* obj) {
    VL_DEBUG_IF(VL_DBG_MSGF("+                VVortex_VX_schedule_if::VL_TO_STRING\n"); );
    // Body
    return (obj ? obj->name() : "null");
}
