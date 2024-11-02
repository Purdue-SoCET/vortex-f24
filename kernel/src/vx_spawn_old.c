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

#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include <vx_print.h>

#define NUM_CORES_MAX 1024

#ifdef __cplusplus
extern "C" {
#endif

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

//////////////////// vx_kernel_func_cb -> vx_spawn_tasks_cb ? /////////////////////////////

__thread dim3_t blockIdx;
__thread dim3_t threadIdx;
dim3_t gridDim;
dim3_t blockDim;

__thread uint32_t __local_group_id;
uint32_t __warps_per_group;

typedef struct {
	vx_kernel_func_cb callback;
	const void* arg;
	uint32_t group_offset;
	uint32_t warp_batches;
	uint32_t remaining_warps;
  uint32_t warps_per_group;
  uint32_t groups_per_core;
  uint32_t remaining_mask;
} wspawn_groups_args_t;

typedef struct {
	vx_kernel_func_cb callback;
	const void* arg;
	uint32_t all_tasks_offset;
  uint32_t remain_tasks_offset;
	uint32_t warp_batches;
	uint32_t remaining_warps;
} wspawn_threads_args_t;

// FROM OLD SCLR --------------------------------------

typedef struct {
    vx_spawn_tasks_cb callback;
    void *arg;
    int offset;  // task offset
    int NWs;     // number of NW batches where NW=<total warps per core>.
    int RWs;     // number of remaining warps in the core
    int fWindex; // nth rotation of a full warp
} wspawn_tasks_args_t;

typedef struct {
    context_t *ctx;
    vx_spawn_kernel_cb callback;
    void *arg;
    int offset; // task offset
    int NWs;    // number of NW batches where NW=<total warps per core>.
    int RWs;    // number of remaining warps in the core
    char isXYpow2;
    char log2XY;
    char log2X;
} wspawn_kernel_args_t;

// ----------------------------------------------------------

void (*return_handler_ptr)();
void (*interrupt_simt_handler_ptr)();
void *g_wspawn_args[NUM_CORES_MAX];


inline char is_log2(int x) {
  return ((x & (x - 1)) == 0);
}


inline int fast_log2(int x) {
  float f = x;
  return (*(int *)(&f) >> 23) - 127;
}


static void __attribute__((noinline)) spawn_tasks_all_stub() {
}

static void __attribute__((noinline)) spawn_tasks_rem_stub() {
  int core_id_tasks_rem_stub    = vx_core_id();
  int thread_id_tasks_rem_stub  = vx_thread_id();

  wspawn_tasks_args_t *p_wspawn_args = (wspawn_tasks_args_t *) g_wspawn_args[core_id_tasks_rem_stub];
  int task_id = p_wspawn_args->offset + thread_id_tasks_rem_stub;
  (p_wspawn_args->callback)(task_id, p_wspawn_args->arg);
}

static void __attribute__((noinline)) spawn_tasks_all_cb() {
  // activate all threads
  vx_tmc(-1);

  // call stub routine
  spawn_tasks_all_stub();

  // disable warp
  vx_tmc_zero();
}

void vx_spawn_tasks(int num_tasks, vx_spawn_tasks_cb callback, void *arg) {
  int num_cores_total_spawn_tasks = vx_num_cores();
  int num_cores_spawn_tasks       = num_cores_total_spawn_tasks / 2;
  int num_warps_spawn_tasks       = vx_num_warps();
  int num_threads_spawn_tasks     = vx_num_threads();

  // current core id
  int core_id = vx_core_id();

  // assign non-priority tasks only to the first half cores
  if (core_id >= (num_cores_total_spawn_tasks / 2)) {
    // 2
    vx_printf("Vx_spawn_tasks core_id too high, so returning core_id:%d, total cores=%d\n",
              core_id, num_cores_total_spawn_tasks);
    return;
  }

  vx_printf("VXspawn starting spawn,  core_id=%d\n", core_id);
  // calculate necessary active cores
  int WT = num_warps_spawn_tasks * num_threads_spawn_tasks;
  int nC1 = (num_tasks > WT) ? (num_tasks / WT) : 1;
  int nc = MIN(nC1, num_cores_total_spawn_tasks / 2);
  int nCoreIDMax = nc - 1;
  if (core_id > nCoreIDMax) {
    vx_printf("VXspawn returning coz core_id=%d >= nc=%d nCoreIDMax=%d\n "
              "(nC1=%d, num_cores_total_spawn_tasks/2=%d)",
              core_id, nc, nCoreIDMax, nC1, num_cores_total_spawn_tasks / 2);
    return; // terminate extra cores
  }

  // number of tasks per core
  int tasks_per_core = num_tasks / nc;
  int tasks_per_core_n1 = tasks_per_core;
  if (core_id == (nc - 1)) {
    int rem = num_tasks - (nc * tasks_per_core);
    tasks_per_core_n1 += rem; // last core also executes remaining tasks
  }

  // number of tasks per warp
  int TW = tasks_per_core_n1 / num_threads_spawn_tasks;      // occupied warps
  int rT = tasks_per_core_n1 - TW * num_threads_spawn_tasks; // remaining threads
  // TW = tasks_per_core_n1;
  // rT = 0;
  int fW = 1, rW = 0;
  if (TW >= num_warps_spawn_tasks) {
    fW = TW / num_warps_spawn_tasks;      // full warps iterations
    rW = TW - fW * num_warps_spawn_tasks; // remaining warps
  }

  wspawn_tasks_args_t wspawn_args = {callback, arg, core_id * tasks_per_core, fW, rW, 0};
  g_wspawn_args[core_id] = &wspawn_args;
  int nw = MIN(TW, num_warps_spawn_tasks);
  vx_printf("VXSpawn: core_id=%d num_tasks=%d num_cores_spawn_tasks=%d num_warps_spawn_tasks=%d num_threads_spawn_tasks=%d WT=%d nC1=%d nc=%d tasks_per_core_n1=%d TW=%d rT=%d fW=%d rW=%d offset=%d nw=%d\n", core_id, num_tasks, num_cores_spawn_tasks, num_warps_spawn_tasks, num_threads_spawn_tasks, WT, nC1, nc, tasks_per_core_n1, TW, rT, fW, rW, core_id * tasks_per_core, nw);
  if (TW >= 1) {
    for (int i = 0; i < fW; i++) {
      // execute callback on other warps
      wspawn_args.fWindex = i;
      // go off right here.
      vx_wspawn(nw, spawn_tasks_all_cb);

      // activate all threads
      vx_tmc(-1);

      // vx_tmc_one();

      // call stub routine
      spawn_tasks_all_stub();

      // back to single-threaded
      vx_tmc_one();

      // wait for spawn warps to terminate
      vx_wspawn_wait();
    }

    if (rW > 0) {
      // execute callback on other warps
      wspawn_args.fWindex = fW;
      vx_wspawn(rW, spawn_tasks_all_cb);

      // activate all threads
      vx_tmc(-1);

      // vx_tmc_one();

      // call stub routine
      spawn_tasks_all_stub();

      // back to single-threaded
      vx_tmc_one();

      // wait for spawn warps to terminate
      vx_wspawn_wait();
    }
  }

  vx_printf("VXSpawn: I am done with the for loop\n");
  if (rT != 0) {
    // adjust offset
    wspawn_args.offset += (tasks_per_core_n1 - rT);

    // activate remaining threads
    int tmask = (1 << rT) - 1;
    vx_tmc(tmask);

    // call stub routine
    spawn_tasks_rem_stub();

    // back to single-threaded
    vx_tmc_one();
  }
}

void vx_spawn_priority_tasks(int num_tasks, int priority_tasks_offset,
                              vx_spawn_tasks_cb callback, void *arg) {

  vx_printf("VXPSpawn: Priority thread scheduler on scalar core has begun");

  int priority_threads[16] = {0, 3, 6, 8, 9, 10, 12, 1, 2, 4, 5, 7, 11, 13, 15, 14};

  asm volatile(".insn r %0, 1, 0, x0, %1, %2" ::"i"(RISCV_CUSTOM0), "r"("x0"), "r"("x0")); // this should result in a wspawn call to 1 warp, and pc 0.
  vx_wspawn_wait();                                                                        // we really have to pray that this does not touch the stack!
}

///////////////////////////////////////////////////////////////////////////////

static void __attribute__((noinline)) spawn_kernel_all_stub() {
  int num_threads_spawn_tasks = vx_num_threads();
  int cid = vx_core_id();
  int wid = vx_warp_id();
  int tid = vx_thread_id();

  wspawn_kernel_args_t *p_wspawn_args = (wspawn_kernel_args_t *)g_wspawn_args[cid];

  int wK = (p_wspawn_args->NWs * wid) + MIN(p_wspawn_args->RWs, wid);
  int tK = p_wspawn_args->NWs + (wid < p_wspawn_args->RWs);
  int offset = p_wspawn_args->offset + (wK * num_threads_spawn_tasks) + (tid * tK);

  int X = p_wspawn_args->ctx->num_groups[0];
  int Y = p_wspawn_args->ctx->num_groups[1];
  int XY = X * Y;

  if (p_wspawn_args->isXYpow2) {
    for (int wg_id = offset, N = wg_id + tK; wg_id < N; ++wg_id) {
      int k = wg_id >> p_wspawn_args->log2XY;
      int wg_2d = wg_id - k * XY;
      int j = wg_2d >> p_wspawn_args->log2X;
      int i = wg_2d - j * X;
      (p_wspawn_args->callback)(p_wspawn_args->arg, p_wspawn_args->ctx, i, j, k);
    }
  }
  else {
    for (int wg_id = offset, N = wg_id + tK; wg_id < N; ++wg_id) {
      int k = wg_id / XY;
      int wg_2d = wg_id - k * XY;
      int j = wg_2d / X;
      int i = wg_2d - j * X;
      (p_wspawn_args->callback)(p_wspawn_args->arg, p_wspawn_args->ctx, i, j, k);
    }
  }
}

static void __attribute__((noinline)) spawn_kernel_rem_stub() {
  int cid = vx_core_id();
  int tid = vx_thread_id();

  wspawn_kernel_args_t *p_wspawn_args = (wspawn_kernel_args_t *)g_wspawn_args[cid];

  int wg_id = p_wspawn_args->offset + tid;

  int X = p_wspawn_args->ctx->num_groups[0];
  int Y = p_wspawn_args->ctx->num_groups[1];
  int XY = X * Y;

  if (p_wspawn_args->isXYpow2) {
    int k = wg_id >> p_wspawn_args->log2XY;
    int wg_2d = wg_id - k * XY;
    int j = wg_2d >> p_wspawn_args->log2X;
    int i = wg_2d - j * X;
    (p_wspawn_args->callback)(p_wspawn_args->arg, p_wspawn_args->ctx, i, j, k);
  }
  else {
    int k = wg_id / XY;
    int wg_2d = wg_id - k * XY;
    int j = wg_2d / X;
    int i = wg_2d - j * X;
    (p_wspawn_args->callback)(p_wspawn_args->arg, p_wspawn_args->ctx, i, j, k);
  }
}

static void __attribute__((noinline)) spawn_kernel_all_cb() {
  // activate all threads
  vx_tmc(-1);

  // call stub routine
  spawn_kernel_all_stub();

  // disable warp
  vx_tmc_zero();
}

void vx_spawn_kernel(context_t *ctx, vx_spawn_kernel_cb callback, void *arg) {
  // total number of WGs
  int X = ctx->num_groups[0];
  int Y = ctx->num_groups[1];
  int Z = ctx->num_groups[2];
  int XY = X * Y;
  int num_tasks = XY * Z;

  // device specs
  int num_cores_spawn_tasks = vx_num_cores();
  int num_warps_spawn_tasks = vx_num_warps();
  int num_threads_spawn_tasks = vx_num_threads();

  // current core id
  int core_id = vx_core_id();
  if (core_id >= NUM_CORES_MAX)
    return;

  // calculate necessary active cores
  int WT = num_warps_spawn_tasks * num_threads_spawn_tasks;
  int nC = (num_tasks > WT) ? (num_tasks / WT) : 1;
  int nc = MIN(nC, num_cores_spawn_tasks);
  if (core_id >= nc)
    return; // terminate extra cores

  // number of tasks per core
  int tasks_per_core = num_tasks / nc;
  int tasks_per_core_n1 = tasks_per_core;
  if (core_id == (nc - 1)) {
      int rem = num_tasks - (nc * tasks_per_core);
      tasks_per_core_n1 += rem; // last core also executes remaining WGs
  }

  // number of tasks per warp
  int TW = tasks_per_core_n1 / num_threads_spawn_tasks;      // occupied warps
  int rT = tasks_per_core_n1 - TW * num_threads_spawn_tasks; // remaining threads
  int fW = 1, rW = 0;
  if (TW >= num_warps_spawn_tasks) {
    fW = TW / num_warps_spawn_tasks;      // full warps iterations
    rW = TW - fW * num_warps_spawn_tasks; // remaining warps
  }

  // fast path handling
  char isXYpow2 = is_log2(XY);
  char log2XY = fast_log2(XY);
  char log2X = fast_log2(X);

  wspawn_kernel_args_t wspawn_args = {
      ctx, callback, arg, core_id * tasks_per_core, fW, rW, isXYpow2, log2XY, log2X};
  g_wspawn_args[core_id] = &wspawn_args;

  if (TW >= 1) {
    // execute callback on other warps
    int nw = MIN(TW, num_warps_spawn_tasks);
    vx_wspawn(nw, spawn_kernel_all_cb);

    // activate all threads
    vx_tmc(-1);

    // call stub routine
    asm volatile("" ::: "memory");
    spawn_kernel_all_stub();

    // back to single-threaded
    vx_tmc_one();

    // wait for spawn warps to terminate
    vx_wspawn_wait();
  }

  if (rT != 0) {
    // adjust offset
    wspawn_args.offset += (tasks_per_core_n1 - rT);

    // activate remaining threads
    int tmask = (1 << rT) - 1;
    vx_tmc(tmask);

    // call stub routine
    spawn_kernel_rem_stub();

    // back to single-threaded
    vx_tmc_one();
  }
}


static void __attribute__ ((noinline)) process_threads() {
  wspawn_threads_args_t* targs = (wspawn_threads_args_t*)csr_read(VX_CSR_MSCRATCH);

  uint32_t threads_per_warp = vx_num_threads();
  uint32_t warp_id = vx_warp_id();
  uint32_t thread_id = vx_thread_id();

  uint32_t start_warp = (warp_id * targs->warp_batches) + MIN(warp_id, targs->remaining_warps);
  uint32_t iterations = targs->warp_batches + (warp_id < targs->remaining_warps);

  uint32_t start_task_id = targs->all_tasks_offset + (start_warp * threads_per_warp) + thread_id;
  uint32_t end_task_id = start_task_id + iterations * threads_per_warp;

  __local_group_id = 0;
  threadIdx.x = 0;
  threadIdx.y = 0;
  threadIdx.z = 0;

  vx_kernel_func_cb callback = targs->callback;
  const void* arg = targs->arg;

  for (uint32_t task_id = start_task_id; task_id < end_task_id; task_id += threads_per_warp) {
    blockIdx.x = task_id % gridDim.x;
    blockIdx.y = (task_id / gridDim.x) % gridDim.y;
    blockIdx.z = task_id / (gridDim.x * gridDim.y);
    callback((void*)arg);
  }
}

static void __attribute__ ((noinline)) process_remaining_threads() {
  wspawn_threads_args_t* targs = (wspawn_threads_args_t*)csr_read(VX_CSR_MSCRATCH);

  uint32_t thread_id = vx_thread_id();
  uint32_t task_id = targs->remain_tasks_offset + thread_id;

  (targs->callback)((void*)targs->arg);
}

static void __attribute__ ((noinline)) process_threads_stub() {
  // activate all threads
  vx_tmc(-1);

  // process all tasks
  process_threads();

  // disable warp
  vx_tmc_zero();
}

static void __attribute__ ((noinline)) process_thread_groups() {
  wspawn_groups_args_t* targs = (wspawn_groups_args_t*)csr_read(VX_CSR_MSCRATCH);

  uint32_t threads_per_warp = vx_num_threads();
  uint32_t warp_id = vx_warp_id();
  uint32_t thread_id = vx_thread_id();

  uint32_t warps_per_group = targs->warps_per_group;
  uint32_t groups_per_core = targs->groups_per_core;

  uint32_t iterations = targs->warp_batches + (warp_id < targs->remaining_warps);

  uint32_t local_group_id = warp_id / warps_per_group;
  uint32_t group_warp_id = warp_id - local_group_id * warps_per_group;
  uint32_t local_task_id = group_warp_id * threads_per_warp + thread_id;

  uint32_t start_group = targs->group_offset + local_group_id;
  uint32_t end_group = start_group + iterations * groups_per_core;

  __local_group_id = local_group_id;

  threadIdx.x = local_task_id % blockDim.x;
  threadIdx.y = (local_task_id / blockDim.x) % blockDim.y;
  threadIdx.z = local_task_id / (blockDim.x * blockDim.y);

  vx_kernel_func_cb callback = targs->callback;
  const void* arg = targs->arg;

  for (uint32_t group_id = start_group; group_id < end_group; group_id += groups_per_core) {
    blockIdx.x = group_id % gridDim.x;
    blockIdx.y = (group_id / gridDim.x) % gridDim.y;
    blockIdx.z = group_id / (gridDim.x * gridDim.y);
    callback((void*)arg);
  }
}

static void __attribute__ ((noinline)) process_thread_groups_stub() {
  // Reads the value of the machine-mode scratch register (VX_CSR_MSCRATCH
  wspawn_groups_args_t* targs = (wspawn_groups_args_t*)csr_read(VX_CSR_MSCRATCH);
  uint32_t warps_per_group = targs->warps_per_group;
  uint32_t remaining_mask = targs->remaining_mask;
  uint32_t warp_id = vx_warp_id();
  uint32_t group_warp_id = warp_id % warps_per_group;

  uint32_t threads_mask = (group_warp_id == warps_per_group-1) ? remaining_mask : -1;

  // activate threads
  vx_tmc(threads_mask);

  // process thread groups
  process_thread_groups();

  // disable all warps except warp0
  vx_tmc(0 == vx_warp_id());
}

int vx_spawn_threads(uint32_t dimension,
                     const uint32_t* grid_dim,
                     const uint32_t * block_dim,
                     vx_kernel_func_cb kernel_func,
                     const void* arg) {
  // calculate number of groups and group size
  uint32_t num_groups = 1;
  uint32_t group_size = 1;
  for (uint32_t i = 0; i < 3; ++i) {
    uint32_t ith_grid_dim   = (grid_dim  && (i < dimension)) ? grid_dim[i]  : 1;
    uint32_t ith_block_dim  = (block_dim && (i < dimension)) ? block_dim[i] : 1;
    
    num_groups *= ith_grid_dim;
    group_size *= ith_block_dim;
    
    gridDim.m[i]  = ith_grid_dim;
    blockDim.m[i] = ith_block_dim;
  }

  // device specifications
  uint32_t num_cores        = vx_num_cores();
  uint32_t warps_per_core   = vx_num_warps();
  uint32_t threads_per_warp = vx_num_threads();
  uint32_t core_id          = vx_core_id();

  // check group size
  uint32_t threads_per_core = warps_per_core * threads_per_warp;
  if (threads_per_core < group_size) {
    vx_printf("error: group_size > threads_per_core (%d,%d)\n", group_size, threads_per_core);
    return -1;
  }

  if (group_size > 1) {
    // calculate number of warps per group
    uint32_t warps_per_group = group_size / threads_per_warp;
    uint32_t remaining_threads = group_size - warps_per_group * threads_per_warp;
    uint32_t remaining_mask = -1;
    if (remaining_threads != 0) {
      remaining_mask = (1 << remaining_threads) - 1;
      ++warps_per_group;
    }

    // calculate necessary active cores
    uint32_t needed_warps = num_groups * warps_per_group;
    uint32_t needed_cores = (needed_warps + warps_per_core-1) / warps_per_core;
    uint32_t active_cores = MIN(needed_cores, num_cores);
    // Change active cores to point to SCALAR and SIMT here?

    // only active cores participate
    if (core_id >= active_cores)
      return 0;

    // total number of groups per core
    uint32_t total_groups_per_core = num_groups / active_cores;
    uint32_t remaining_groups_per_core = num_groups - active_cores * total_groups_per_core;
    if (core_id < remaining_groups_per_core)
      ++total_groups_per_core;

    // calculate number of warps to activate
    uint32_t groups_per_core = warps_per_core / warps_per_group;
    uint32_t total_warps_per_core = total_groups_per_core * warps_per_group;
    uint32_t active_warps = total_warps_per_core;
    uint32_t warp_batches = 1, remaining_warps = 0;
    if (active_warps > warps_per_core) {
      active_warps = groups_per_core * warps_per_group;
      warp_batches = total_warps_per_core / active_warps;
      remaining_warps = total_warps_per_core - warp_batches * active_warps;
    }

    // calculate offsets for group distribution
    uint32_t group_offset = core_id * total_groups_per_core + MIN(core_id, remaining_groups_per_core);

    // set scheduler arguments
    wspawn_groups_args_t wspawn_args = {
      kernel_func,
      arg,
      group_offset,
      warp_batches,
      remaining_warps,
      warps_per_group,
      groups_per_core,
      remaining_mask
    };
    csr_write(VX_CSR_MSCRATCH, &wspawn_args);
    // set global variables
    __warps_per_group = warps_per_group;

    // execute callback on other warps
    vx_wspawn(active_warps, process_thread_groups_stub);

    // execute callback on warp0
    process_thread_groups_stub();
  } else {
    uint32_t num_tasks = num_groups;
    __warps_per_group = 0;

    // calculate necessary active cores
    uint32_t needed_cores = (num_tasks + threads_per_core - 1) / threads_per_core;
    uint32_t active_cores = MIN(needed_cores, num_cores);

    // only active cores participate
    if (core_id >= active_cores)
      return 0;

    // number of tasks per core
    uint32_t tasks_per_core = num_tasks / active_cores;
    uint32_t remaining_tasks_per_core = num_tasks - tasks_per_core * active_cores;
    if (core_id < remaining_tasks_per_core)
      ++tasks_per_core;

    // calculate number of warps to activate
    uint32_t total_warps_per_core = tasks_per_core / threads_per_warp;
    uint32_t remaining_tasks = tasks_per_core - total_warps_per_core * threads_per_warp;
    uint32_t active_warps = total_warps_per_core;
    uint32_t warp_batches = 1, remaining_warps = 0;
    if (active_warps > warps_per_core) {
      active_warps = warps_per_core;
      warp_batches = total_warps_per_core / active_warps;
      remaining_warps = total_warps_per_core - warp_batches * active_warps;
    }

    // calculate offsets for task distribution
    uint32_t all_tasks_offset = core_id * tasks_per_core + MIN(core_id, remaining_tasks_per_core);
    uint32_t remain_tasks_offset = all_tasks_offset + (tasks_per_core - remaining_tasks);

    // prepare scheduler arguments
    wspawn_threads_args_t wspawn_args = {
      kernel_func,
      arg,
      all_tasks_offset,
      remain_tasks_offset,
      warp_batches,
      remaining_warps
    };
    csr_write(VX_CSR_MSCRATCH, &wspawn_args);

    if (active_warps >= 1) {
      // execute callback on other warps
      vx_wspawn(active_warps, process_threads_stub);

      // activate all threads
      vx_tmc(-1);

      // process threads
      process_threads();

      // back to single-threaded
      vx_tmc_one();
    }

    if (remaining_tasks != 0) {
      // activate remaining threads
      uint32_t tmask = (1 << remaining_tasks) - 1;
      vx_tmc(tmask);

      // process remaining threads
      process_remaining_threads();

      // back to single-threaded
      vx_tmc_one();
    }
  }

  // wait for spawned warps to complete
  vx_wspawn(1, 0); // -> spawn warps
  // Passed in a null pointer -> create a nop?

  return 0;
}

#ifdef __cplusplus
}
#endif