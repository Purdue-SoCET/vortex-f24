// auto-generated by gen_config.py. DO NOT EDIT
// Generated at 2024-10-27 13:37:06.166121

// Translated from /home/ecegridfs/a/socet143/vortex-f24/hw/rtl/VX_types.vh:

// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef VX_TYPES_VH
#define VX_TYPES_VH

// Device configuration registers /////////////////////////////////////////////

#define VX_CSR_ADDR_BITS                12
#define VX_DCR_ADDR_BITS                12

#define VX_DCR_BASE_STATE_BEGIN         0x001
#define VX_DCR_BASE_STARTUP_ADDR0       0x001
#define VX_DCR_BASE_STARTUP_ADDR1       0x002
#define VX_DCR_BASE_STARTUP_ARG0        0x003
#define VX_DCR_BASE_STARTUP_ARG1        0x004
#define VX_DCR_BASE_MPM_CLASS           0x005
#define VX_DCR_BASE_STATE_END           0x006

#define VX_DCR_BASE_STATE(addr)         ((addr) - VX_DCR_BASE_STATE_BEGIN)
#define VX_DCR_BASE_STATE_COUNT         (VX_DCR_BASE_STATE_END-VX_DCR_BASE_STATE_BEGIN)

// Machine Performance-monitoring counters classes ////////////////////////////

#define VX_DCR_MPM_CLASS_NONE           0
#define VX_DCR_MPM_CLASS_CORE           1
#define VX_DCR_MPM_CLASS_MEM            2

// User Floating-Point CSRs ///////////////////////////////////////////////////

#define VX_CSR_FFLAGS                   0x001
#define VX_CSR_FRM                      0x002
#define VX_CSR_FCSR                     0x003

#define VX_CSR_SATP                     0x180

#define VX_CSR_PMPCFG0                  0x3A0
#define VX_CSR_PMPADDR0                 0x3B0

#define VX_CSR_MSTATUS                  0x300
#define VX_CSR_MISA                     0x301
#define VX_CSR_MEDELEG                  0x302
#define VX_CSR_MIDELEG                  0x303
#define VX_CSR_MIE                      0x304
#define VX_CSR_MTVEC                    0x305

#define VX_CSR_MSCRATCH                 0x340
#define VX_CSR_MEPC                     0x341
#define VX_CSR_MCAUSE                   0x342

#define VX_CSR_MNSTATUS                 0x744

#define VX_CSR_MPM_BASE                 0xB00
#define VX_CSR_MPM_BASE_H               0xB80
#define VX_CSR_MPM_USER                 0xB03
#define VX_CSR_MPM_USER_H               0xB83

// Machine Performance-monitoring core counters (Standard) ////////////////////

#define VX_CSR_MCYCLE                   0xB00
#define VX_CSR_MCYCLE_H                 0xB80
#define VX_CSR_MPM_RESERVED             0xB01
#define VX_CSR_MPM_RESERVED_H           0xB81
#define VX_CSR_MINSTRET                 0xB02
#define VX_CSR_MINSTRET_H               0xB82

// Machine Performance-monitoring core counters (class 1) /////////////////////

// PERF: pipeline
#define VX_CSR_MPM_SCHED_ID             0xB03
#define VX_CSR_MPM_SCHED_ID_H           0xB83
#define VX_CSR_MPM_SCHED_ST             0xB04
#define VX_CSR_MPM_SCHED_ST_H           0xB84
#define VX_CSR_MPM_IBUF_ST              0xB05
#define VX_CSR_MPM_IBUF_ST_H            0xB85
#define VX_CSR_MPM_SCRB_ST              0xB06
#define VX_CSR_MPM_SCRB_ST_H            0xB86
#define VX_CSR_MPM_OPDS_ST              0xB07
#define VX_CSR_MPM_OPDS_ST_H            0xB87
#define VX_CSR_MPM_SCRB_ALU             0xB08
#define VX_CSR_MPM_SCRB_ALU_H           0xB88
#define VX_CSR_MPM_SCRB_FPU             0xB09
#define VX_CSR_MPM_SCRB_FPU_H           0xB89
#define VX_CSR_MPM_SCRB_LSU             0xB0A
#define VX_CSR_MPM_SCRB_LSU_H           0xB8A
#define VX_CSR_MPM_SCRB_SFU             0xB0B
#define VX_CSR_MPM_SCRB_SFU_H           0xB8B
#define VX_CSR_MPM_SCRB_CSRS            0xB0C
#define VX_CSR_MPM_SCRB_CSRS_H          0xB8C
#define VX_CSR_MPM_SCRB_WCTL            0xB0D
#define VX_CSR_MPM_SCRB_WCTL_H          0xB8D
// PERF: memory
#define VX_CSR_MPM_IFETCHES             0xB0E
#define VX_CSR_MPM_IFETCHES_H           0xB8E
#define VX_CSR_MPM_LOADS                0xB0F
#define VX_CSR_MPM_LOADS_H              0xB8F
#define VX_CSR_MPM_STORES               0xB10
#define VX_CSR_MPM_STORES_H             0xB90
#define VX_CSR_MPM_IFETCH_LT            0xB11
#define VX_CSR_MPM_IFETCH_LT_H          0xB91
#define VX_CSR_MPM_LOAD_LT              0xB12
#define VX_CSR_MPM_LOAD_LT_H            0xB92

// Machine Performance-monitoring memory counters (class 2) ///////////////////

// PERF: icache
#define VX_CSR_MPM_ICACHE_READS         0xB03     // total reads
#define VX_CSR_MPM_ICACHE_READS_H       0xB83
#define VX_CSR_MPM_ICACHE_MISS_R        0xB04     // read misses
#define VX_CSR_MPM_ICACHE_MISS_R_H      0xB84
#define VX_CSR_MPM_ICACHE_MSHR_ST       0xB05     // MSHR stalls
#define VX_CSR_MPM_ICACHE_MSHR_ST_H     0xB85
// PERF: dcache
#define VX_CSR_MPM_DCACHE_READS         0xB06     // total reads
#define VX_CSR_MPM_DCACHE_READS_H       0xB86
#define VX_CSR_MPM_DCACHE_WRITES        0xB07     // total writes
#define VX_CSR_MPM_DCACHE_WRITES_H      0xB87
#define VX_CSR_MPM_DCACHE_MISS_R        0xB08     // read misses
#define VX_CSR_MPM_DCACHE_MISS_R_H      0xB88
#define VX_CSR_MPM_DCACHE_MISS_W        0xB09     // write misses
#define VX_CSR_MPM_DCACHE_MISS_W_H      0xB89
#define VX_CSR_MPM_DCACHE_BANK_ST       0xB0A     // bank conflicts
#define VX_CSR_MPM_DCACHE_BANK_ST_H     0xB8A
#define VX_CSR_MPM_DCACHE_MSHR_ST       0xB0B     // MSHR stalls
#define VX_CSR_MPM_DCACHE_MSHR_ST_H     0xB8B
// PERF: l2cache
#define VX_CSR_MPM_L2CACHE_READS        0xB0C     // total reads
#define VX_CSR_MPM_L2CACHE_READS_H      0xB8C
#define VX_CSR_MPM_L2CACHE_WRITES       0xB0D     // total writes
#define VX_CSR_MPM_L2CACHE_WRITES_H     0xB8D
#define VX_CSR_MPM_L2CACHE_MISS_R       0xB0E     // read misses
#define VX_CSR_MPM_L2CACHE_MISS_R_H     0xB8E
#define VX_CSR_MPM_L2CACHE_MISS_W       0xB0F     // write misses
#define VX_CSR_MPM_L2CACHE_MISS_W_H     0xB8F
#define VX_CSR_MPM_L2CACHE_BANK_ST      0xB10     // bank conflicts
#define VX_CSR_MPM_L2CACHE_BANK_ST_H    0xB90
#define VX_CSR_MPM_L2CACHE_MSHR_ST      0xB11     // MSHR stalls
#define VX_CSR_MPM_L2CACHE_MSHR_ST_H    0xB91
// PERF: l3cache
#define VX_CSR_MPM_L3CACHE_READS        0xB12     // total reads
#define VX_CSR_MPM_L3CACHE_READS_H      0xB92
#define VX_CSR_MPM_L3CACHE_WRITES       0xB13     // total writes
#define VX_CSR_MPM_L3CACHE_WRITES_H     0xB93
#define VX_CSR_MPM_L3CACHE_MISS_R       0xB14     // read misses
#define VX_CSR_MPM_L3CACHE_MISS_R_H     0xB94
#define VX_CSR_MPM_L3CACHE_MISS_W       0xB15     // write misses
#define VX_CSR_MPM_L3CACHE_MISS_W_H     0xB95
#define VX_CSR_MPM_L3CACHE_BANK_ST      0xB16     // bank conflicts
#define VX_CSR_MPM_L3CACHE_BANK_ST_H    0xB96
#define VX_CSR_MPM_L3CACHE_MSHR_ST      0xB17     // MSHR stalls
#define VX_CSR_MPM_L3CACHE_MSHR_ST_H    0xB97
// PERF: memory
#define VX_CSR_MPM_MEM_READS            0xB18     // total reads
#define VX_CSR_MPM_MEM_READS_H          0xB98
#define VX_CSR_MPM_MEM_WRITES           0xB19     // total writes
#define VX_CSR_MPM_MEM_WRITES_H         0xB99
#define VX_CSR_MPM_MEM_LT               0xB1A     // memory latency
#define VX_CSR_MPM_MEM_LT_H             0xB9A
#define VX_CSR_MPM_MEM_BANK_CNTR        0xB1E     // memory bank requests
#define VX_CSR_MPM_MEM_BANK_CNTR_H      0xB9E
#define VX_CSR_MPM_MEM_BANK_TICK        0xB1F     // memory ticks
#define VX_CSR_MPM_MEM_BANK_TICK_H      0xB9F
// PERF: lmem
#define VX_CSR_MPM_LMEM_READS           0xB1B     // memory reads
#define VX_CSR_MPM_LMEM_READS_H         0xB9B
#define VX_CSR_MPM_LMEM_WRITES          0xB1C     // memory writes
#define VX_CSR_MPM_LMEM_WRITES_H        0xB9C
#define VX_CSR_MPM_LMEM_BANK_ST         0xB1D     // bank conflicts
#define VX_CSR_MPM_LMEM_BANK_ST_H       0xB9D

// Machine Performance-monitoring memory counters (class 3) ///////////////////
// <Add your own counters: use addresses hB03..B1F, hB83..hB9F>

// Machine Information Registers //////////////////////////////////////////////

#define VX_CSR_MVENDORID                0xF11
#define VX_CSR_MARCHID                  0xF12
#define VX_CSR_MIMPID                   0xF13
#define VX_CSR_MHARTID                  0xF14

// GPGU CSRs

#define VX_CSR_THREAD_ID                0xCC0
#define VX_CSR_WARP_ID                  0xCC1
#define VX_CSR_CORE_ID                  0xCC2
#define VX_CSR_ACTIVE_WARPS             0xCC3
#define VX_CSR_ACTIVE_THREADS           0xCC4     // warning! this value is also used in LLVM

#define VX_CSR_NUM_THREADS              0xFC0
#define VX_CSR_NUM_WARPS                0xFC1
#define VX_CSR_NUM_CORES                0xFC2
#define VX_CSR_LOCAL_MEM_BASE           0xFC3

#endif // VX_TYPES_VH
