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

#ifndef __VX_SPAWN_H__
#define __VX_SPAWN_H__

#include <vx_intrinsics.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef union {
  struct {
    uint32_t x;
    uint32_t y;
    uint32_t z;
  };
  uint32_t m[3];
} dim3_t;

extern __thread dim3_t blockIdx;
extern __thread dim3_t threadIdx;
extern dim3_t gridDim;
extern dim3_t blockDim;

extern __thread uint32_t __local_group_id;
extern uint32_t __warps_per_group;

typedef struct {
  uint32_t num_groups[3];
  uint32_t global_offset[3];
  uint32_t local_size[3];
  char * printf_buffer;
  uint32_t *printf_buffer_position;
  uint32_t printf_buffer_capacity;
  uint32_t work_dim;
} context_t;

typedef void (*vx_kernel_func_cb)(void *arg);
typedef void (*vx_spawn_tasks_cb)(int task_id, void *arg);
typedef void (*vx_spawn_kernel_cb) (
  const void * /* arg */,
	const context_t * /* context */,
	uint32_t /* group_x */,
	uint32_t /* group_y */,
	uint32_t /* group_z */
);

typedef void (*vx_serial_cb)(void *arg);

#define __local_mem(size) \
  (void*)((int8_t*)csr_read(VX_CSR_LOCAL_MEM_BASE) + __local_group_id * size)

#define __syncthreads() \
  vx_barrier(__local_group_id, __warps_per_group)

// launch a kernel function with a grid of blocks and block of threads
int vx_spawn_threads(uint32_t dimension,
                     const uint32_t* grid_dim,
                     const uint32_t* block_dim,
                     vx_kernel_func_cb kernel_func,
                     const void* arg);

// function call serialization
void vx_serial(vx_serial_cb callback, const void * arg);

void vx_spawn_priority_tasks(int num_tasks, int priority_tasks_offset,
                              vx_spawn_tasks_cb callback, void *arg);

void vx_spawn_tasks(int num_tasks, vx_spawn_tasks_cb callback, void *arg);

void vx_wspawn_wait();

#ifdef __cplusplus
}
#endif

#endif // __VX_SPAWN_H__
