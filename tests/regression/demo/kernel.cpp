#include <vx_intrinsics.h>
#include <vx_print.h>
#include <vx_spawn.h>
#include "common.h"
#include "stdint.h"
#include "stdlib.h"

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
	auto src0_ptr = reinterpret_cast<TYPE*>(arg->src0_addr);
	auto src1_ptr = reinterpret_cast<TYPE*>(arg->src1_addr);
	auto dst_ptr = reinterpret_cast<TYPE*>(arg->dst_addr);

	uint32_t count = arg->task_size;
	uint32_t offset = blockIdx.x * count;

	for (uint32_t i = 0; i < count; ++i) {
		dst_ptr[offset+i] = src0_ptr[offset+i] + src1_ptr[offset+i];
	}
}

int main() {
	// kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
	// return vx_spawn_threads(1, &arg->num_tasks, nullptr, (vx_kernel_func_cb)kernel_body, arg);

	kernel_arg_t* arg = (kernel_arg_t*) csr_read(VX_CSR_MSCRATCH);
    // kernel_arg_t* arg = (kernel_arg_t*)KERNEL_ARG_DEV_MEM_ADDR2;
    vx_printf("Calling VXSpawn1\n");
    vx_spawn_tasks(arg->num_tasks_nonpriority, (vx_spawn_tasks_cb)kernel_body, arg);
    // vx_printf("Calling VXPSpawn2\n");
    // vx_spawn_priority_tasks(arg->num_tasks_priority,arg->num_tasks_nonpriority, (vx_spawn_tasks_cb)kernel_body, arg);
	return 0;
}
