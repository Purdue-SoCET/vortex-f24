// Copyright Â© 2019-2023
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

#ifdef __cplusplus
extern "C" {
#endif

#define NUM_CORES_MAX 1024
void* g_wspawn_args[NUM_CORES_MAX];

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

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
	vx_spawn_tasks_cb callback;
	const void* arg;
	uint32_t all_tasks_offset;
  uint32_t remain_tasks_offset;
	uint32_t warp_batches;
	uint32_t remaining_warps;
} wspawn_threads_args_t;

// Stub Function
static void __attribute__ ((noinline)) spawn_tasks_all_stub() {
  int NT  = vx_num_threads();
  int NW = vx_num_warps();
  int cid = vx_core_id();
  int wid = vx_warp_id();
  int tid = vx_thread_id();

  // Changed struct 
  wspawn_threads_args_t* p_wspawn_args = (wspawn_threads_args_t*)g_wspawn_args[cid];

  // Change type for callback
  vx_kernel_func_cb callback = p_wspawn_args->callback;
  void* arg = p_wspawn_args->arg;

  int warp_gid = (p_wspawn_args->warp_batches * NW) + wid; 
  int thread_gid = warp_gid * NT + tid + p_wspawn_args->all_tasks_offset; 
  // vx_printf("VXSpawn: cid=%d, wid=%d, tid=%d, wK=%d, tK=%d, offset=%d, taskids=%d-%d, fWindex=%d, warp_gid=%d, thread_gid=%d\n",cid, wid, tid, wK, tK, offset, (offset), (offset+tK-1),p_wspawn_args->fWindex,warp_gid,thread_gid);
  vx_printf("VXSpawn: cid=%d, wid=%d, tid=%d, fWindex=%d, offset= %d, warp_gid=%d, thread_gid=%d\n",cid, wid, tid, p_wspawn_args->warp_batches,p_wspawn_args->all_tasks_offset,warp_gid,thread_gid);
  callback(thread_gid);
}
static void __attribute__ ((noinline)) spawn_tasks_rem_stub() {
  int cid = vx_core_id();
  int tid = vx_thread_id();
  
  wspawn_threads_args_t* p_wspawn_args = (wspawn_threads_args_t*)g_wspawn_args[cid];
  int task_id = p_wspawn_args->all_tasks_offset + tid;
  (p_wspawn_args->callback)(task_id, p_wspawn_args->arg);
}

static void __attribute__ ((noinline)) spawn_tasks_all_cb() {  
  // activate all threads
  vx_tmc(-1);

  // vx_tmc_one();

  // call stub routine
  spawn_tasks_all_stub();

  // disable warp
  vx_tmc_zero();
}

//---------------------------------------------------------------------------------------------

// First Function
//---------------------------------------------------------------------------------------------
void vx_spawn_tasks(uint32_t num_tasks, vx_spawn_tasks_cb callback , void * arg) {
	// device specs
  
  int NC_total = vx_num_cores();
  int NC = NC_total/2;
  int NW = vx_num_warps();
  int NT = vx_num_threads();

  // current core id
  int core_id = vx_core_id();
  // assign non-priority tasks only to the first half cores
  if (core_id >= (NC_total/2)) ///2
  {
    vx_printf("Vx_spawn_tasks core_id too high, so returning core_id:%d, total cores=%d\n", core_id, NC_total);
    return;
  }

  vx_printf("VXspawn starting spawn,  core_id=%d\n",core_id);
  // calculate necessary active cores
  int WT = NW * NT;
  int nC1 = (num_tasks > WT) ? (num_tasks / WT) : 1;
  int nc = MIN(nC1, NC_total/2);
  int nCoreIDMax = nc-1;
  if (core_id > nCoreIDMax)
  {
    vx_printf("VXspawn returning coz core_id=%d >= nc=%d nCoreIDMax=%d\n (nC1=%d, NC_total/2=%d)",core_id,nc,nCoreIDMax, nC1, NC_total/2);
    return; // terminate extra cores
  }
    

  // number of tasks per core
  int tasks_per_core = num_tasks / nc;
  int tasks_per_core_n1 = tasks_per_core;  
  if (core_id == (nc-1)) {    
    int rem = num_tasks - (nc * tasks_per_core); 
    tasks_per_core_n1 += rem; // last core also executes remaining tasks
  }

  // number of tasks per warp
  int TW = tasks_per_core_n1 / NT;      // occupied warps
  int rT = tasks_per_core_n1 - TW * NT; // remaining threads
  // TW = tasks_per_core_n1;
  // rT = 0;
  int fW = 1, rW = 0;
  if (TW >= NW) {
    fW = TW / NW;			                  // full warps iterations
    rW = TW - fW * NW;                  // remaining warps
  }
  
  wspawn_threads_args_t wspawn_args = { callback, arg, core_id * tasks_per_core, fW, rW,0 };
  g_wspawn_args[core_id] = &wspawn_args;
  int nw = MIN(TW, NW);
  vx_printf("VXSpawn: core_id=%d num_tasks=%d NC=%d NW=%d NT=%d WT=%d nC1=%d nc=%d tasks_per_core_n1=%d TW=%d rT=%d fW=%d rW=%d offset=%d nw=%d\n", core_id, num_tasks,NC, NW, NT,WT, nC1, nc, tasks_per_core_n1, TW, rT, fW, rW, core_id*tasks_per_core, nw);
	if(TW>=1)
  {
  for (int i=0; i<fW; i++)
    {
      // execute callback on other warps
      wspawn_args.warp_batches = i;
      vx_wspawn(nw, spawn_tasks_all_cb);

      // activate all threads
      vx_tmc(-1);

      // vx_tmc_one();

      // call stub routine
      spawn_tasks_all_stub();
    
      // back to single-threaded
      vx_tmc_one();
      
      // wait for spawn warps to terminate
      // vx_wspawn_wait(); 
    }
    if(rW>0)
    {
      // execute callback on other warps
      wspawn_args.warp_batches = fW;
      vx_wspawn(rW, spawn_tasks_all_cb);

      // activate all threads
      vx_tmc(-1);

      // vx_tmc_one();

      // call stub routine
      spawn_tasks_all_stub();
    
      // back to single-threaded
      vx_tmc_one();
      
      // wait for spawn warps to terminate
      // vx_wspawn_wait(); 
    }
  }

  vx_printf("VXSpawn: I am done with the for loop\n");
  if (rT != 0) {
    // adjust offset
    wspawn_args.remain_tasks_offset += (tasks_per_core_n1 - rT);
    
    // activate remaining threads  
    int tmask = (1 << rT) - 1;
    vx_tmc(tmask);

    // call stub routine
    spawn_tasks_rem_stub();

    // back to single-threaded
    vx_tmc_one();
  }

}

static void __attribute__ ((noinline)) spawn_priority_tasks_all_stub() {
  int NT  = 1; //vx_num_threads();
  int NW = vx_num_warps();
  int cid = vx_core_id();
  int wid = vx_warp_id();
  int tid = vx_thread_id();
  
  wspawn_threads_args_t* p_wspawn_args = (wspawn_threads_args_t*)g_wspawn_args[cid]; 

  vx_kernel_func_cb callback = p_wspawn_args->callback;
  void* arg = p_wspawn_args->arg; 

  int warp_gid = (p_wspawn_args->warp_batches * NW) + wid; 
  int thread_gid = warp_gid * NT + tid + p_wspawn_args->all_tasks_offset;

  vx_printf("VXPSpawn: cid=%d, wid=%d, tid=%d, fWindex=%d, offset= %d, warp_gid=%d, thread_gid=%d\n",cid, wid, tid, p_wspawn_args->warp_batches,p_wspawn_args->all_tasks_offset,warp_gid,thread_gid);
  callback(thread_gid);
}

static void __attribute__ ((noinline)) spawn_priority_tasks_all_cb() {  
  // activate the 1 priority thread
  // vx_tmc(-1);

  vx_tmc_one();

  // call stub routine
  spawn_priority_tasks_all_stub();

  // disable warp
  vx_tmc_zero();
}

//---------------------------------------------------------------------------------------------

// Second Function
//---------------------------------------------------------------------------------------------

void vx_spawn_priority_tasks(uint32_t num_tasks, int priority_tasks_offset, vx_spawn_tasks_cb callback , void * arg) {
	// device specs
  int NC_total = vx_num_cores();
  int NC = NC_total/2;
  int NW = vx_num_warps(); 
  int NT = 1; //vx_num_threads(); //priority warps are made of only 1 thread, will be run on scalar core 

  // current core id
  int core_id = vx_core_id();
  int core_second = (NC_total/2);
  // vx_printf("VXPspawn where are we skipping? ,  core_boundary=%d\n",core_second);
  
  if (core_id >= NUM_CORES_MAX) 
    return;

  // assign priority tasks only to second half cores
  if(core_id >= (NC_total/2))
  {
    vx_printf("VXPspawn starting spawn,  core_id=%d\n",core_id);
    // calculate necessary active cores
    int WT = NW * NT;
    int nC1 = (num_tasks > WT) ? (num_tasks / WT) : 1;
    int nc = MIN(nC1, NC_total/2);
    int nCoreIDMax = (nc+ (NC_total/2)-1);
    if (core_id > nCoreIDMax )
    {
      vx_printf("VXPspawn returning coz core_id=%d >= nc=%d nCoreIDMax=%d\n (nC1=%d, NC_total/2=%d)",core_id,nc,nCoreIDMax, nC1, NC_total/2);
      return; // terminate extra cores
    }
      

    // number of tasks per core
    int tasks_per_core = (num_tasks / nc);
    int tasks_per_core_n1 = tasks_per_core;  
    if (core_id == (nc-1)) {    
      int rem = num_tasks - (nc * tasks_per_core); 
      tasks_per_core_n1 += rem; // last core also executes remaining tasks
    }

    // number of tasks per warp
    // int TW = tasks_per_core_n1 / NT ;      // occupied warps
    // int rT = tasks_per_core_n1 - TW; // remaining threads
    int TW = tasks_per_core_n1;
    int rT = 0;
    int fW = 1, rW = 0;
    if (TW >= NW) {
      fW = TW / NW;			                  // full warps iterations
      rW = TW - fW * NW;                  // remaining warps
    }

    wspawn_threads_args_t wspawn_args = { callback, arg, priority_tasks_offset + ((core_id - core_second) * tasks_per_core), fW, rW,0 };
    g_wspawn_args[core_id] = &wspawn_args;
    int nw = MIN(TW, NW);
    vx_printf("VXPSpawn: core_id=%d num_tasks=%d NC=%d NW=%d NT=%d WT=%d nC1=%d nc=%d tasks_per_core_n1=%d TW=%d rT=%d fW=%d rW=%d offset=%d nw=%d\n", core_id, num_tasks,NC, NW, NT,WT, nC1, nc, tasks_per_core_n1, TW, rT, fW, rW, core_id*tasks_per_core, nw);
    if (TW >= 1)	{
      for(int i=0; i<fW; i++)
      {
      // execute callback on other warps
      wspawn_args.warp_batches = i;
      vx_wspawn(nw, spawn_priority_tasks_all_cb);

      // activate all threads
      // vx_tmc(-1);

      vx_tmc_one();

      // call stub routine
      spawn_priority_tasks_all_stub();

      // back to single-threaded
      vx_tmc_one();
      
      // wait for spawn warps to terminate
      // vx_wspawn_wait();
      }
      if(rW>0)
      {
        // execute callback on other warps
        wspawn_args.warp_batches = fW;
        vx_wspawn(rW, spawn_priority_tasks_all_cb);

        // activate all threads
        // vx_tmc(-1);

        vx_tmc_one();

        // call stub routine
        spawn_priority_tasks_all_stub();

        // back to single-threaded
        vx_tmc_one();
        
        // wait for spawn warps to terminate
        // vx_wspawn_wait();
      }
    }  
  }
  else
  {
    vx_printf("VXPspawn skipping spawning core id too low,  core_id=%d\n",core_id);
  }
}

///////////////////////////////////////////////////////////////////////////////
// spawn_kernel_all_stub
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

  (targs->callback)(thread_id, (void*)targs->arg);
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
    uint32_t gd = (grid_dim && (i < dimension)) ? grid_dim[i] : 1;
    uint32_t bd = (block_dim && (i < dimension)) ? block_dim[i] : 1;
    num_groups *= gd;
    group_size *= bd;
    gridDim.m[i] = gd;
    blockDim.m[i] = bd;
  }

  // device specifications
  uint32_t num_cores = vx_num_cores();
  uint32_t warps_per_core = vx_num_warps();
  uint32_t threads_per_warp = vx_num_threads();
  uint32_t core_id = vx_core_id();

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
  vx_wspawn(1, 0);

  return 0;
}

#ifdef __cplusplus
}
#endif
