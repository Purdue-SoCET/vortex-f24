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

#include "socket.h"
#include "cluster.h"

#define SCALAR_ID_OFFSET 10

using namespace vortex;

Socket::Socket(const SimContext& ctx,
                uint32_t socket_id,
                Cluster* cluster,
                Arch &arch,
                const DCRS &dcrs)
  : SimObject(ctx, "socket")
  , icache_mem_req_port(this)
  , icache_mem_rsp_port(this)
  , dcache_mem_req_port(this)
  , dcache_mem_rsp_port(this)
  , socket_id_(socket_id)
  , cluster_(cluster)
  , cores_(arch.socket_size())
  , scalarcores_(arch.socket_size())
{
  auto cores_per_socket = cores_.size();

  char sname[100];
  snprintf(sname, 100, "socket%d-icaches", socket_id);
  // Need double the inputs for double the cores
  icaches_ = CacheCluster::Create(sname, cores_per_socket, NUM_ICACHES, 1, CacheSim::Config{
    !ICACHE_ENABLED,
    log2ceil(ICACHE_SIZE),  // C
    log2ceil(L1_LINE_SIZE), // L
    log2ceil(sizeof(uint32_t)), // W
    log2ceil(ICACHE_NUM_WAYS),// A
    1,                      // B
    XLEN,                   // address bits
    1,                      // number of ports
    1,                      // number of inputs
    false,                  // write-back
    false,                  // write response
    (uint8_t)arch.num_warps(), // mshr size
    2,                      // pipeline latency
  });

  icaches_->MemReqPort.bind(&icache_mem_req_port);
  icache_mem_rsp_port.bind(&icaches_->MemRspPort);

  snprintf(sname, 100, "socket%d-dcaches", socket_id);
  // Need double the inputs for double the cores
  dcaches_ = CacheCluster::Create(sname, cores_per_socket, NUM_DCACHES, DCACHE_NUM_REQS, CacheSim::Config{
    !DCACHE_ENABLED,
    log2ceil(DCACHE_SIZE),  // C
    log2ceil(L1_LINE_SIZE), // L
    log2ceil(DCACHE_WORD_SIZE), // W
    log2ceil(DCACHE_NUM_WAYS),// A
    log2ceil(DCACHE_NUM_BANKS), // B
    XLEN,                   // address bits
    1,                      // number of ports
    DCACHE_NUM_REQS,        // number of inputs
    DCACHE_WRITEBACK,       // write-back
    false,                  // write response
    DCACHE_MSHR_SIZE,       // mshr size
    2,                      // pipeline latency
  });

  dcaches_->MemReqPort.bind(&dcache_mem_req_port);
  dcache_mem_rsp_port.bind(&dcaches_->MemRspPort);

  // create cores

  // Ex: I set cores = 1
  // 1 SIMT core is created in the first loop
  // 1 scalar core is created in the 2nd loop.
  // Core 0 is SIMT, core 1 is scalar 

  // Create the SIMT cores
  for (uint32_t i = 0; i < cores_per_socket; ++i) {
    uint32_t core_id = socket_id * cores_per_socket + i;
    cores_.at(i) = Core::Create(core_id, this, arch, dcrs);

    cores_.at(i)->icache_req_ports.at(0).bind(&icaches_->CoreReqPorts.at(i).at(0));
    icaches_->CoreRspPorts.at(i).at(0).bind(&cores_.at(i)->icache_rsp_ports.at(0));

    for (uint32_t j = 0; j < DCACHE_NUM_REQS; ++j) {
      cores_.at(i)->dcache_req_ports.at(j).bind(&dcaches_->CoreReqPorts.at(i).at(j));
      dcaches_->CoreRspPorts.at(i).at(j).bind(&cores_.at(i)->dcache_rsp_ports.at(j));
    }
  }

  // Create the scalar cores
  // ID is an offset from its SIMT pair. Make sure they don't conflict at large # of cores. 
  // Arch scalar_arch(1, 1, cores_.size()); //Scalar core arch is 1 thread, 1 warp, and however many cores
  // for (uint32_t i = 0; i < cores_per_socket; ++i) {
  //   uint32_t scalar_core_id = socket_id * cores_per_socket + i;
  //   scalarcores_.at(i) = Core::Create(scalar_core_id, this, scalar_arch, dcrs);

  //   scalarcores_.at(i)->icache_req_ports.at(0).bind(&icaches_->CoreReqPorts.at(i).at(0));
  //   icaches_->CoreRspPorts.at(i).at(0).bind(&scalarcores_.at(i)->icache_rsp_ports.at(0));

  //   for (uint32_t j = 0; j < DCACHE_NUM_REQS; ++j) {
  //     scalarcores_.at(i)->dcache_req_ports.at(j).bind(&dcaches_->CoreReqPorts.at(i).at(j));
  //     dcaches_->CoreRspPorts.at(i).at(j).bind(&cores_.at(i)->dcache_rsp_ports.at(j));
  //   }
  // }
}

Socket::~Socket() {
  //--
}

void Socket::reset() {
  //--
}

void Socket::tick() {
  //--
}

void Socket::attach_ram(RAM* ram) {
  for (auto core : cores_) {
    core->attach_ram(ram);
  }
}

#ifdef VM_ENABLE
void Socket::set_satp(uint64_t satp) {
  for (auto core : cores_) {
    core->set_satp(satp);
  }
}
#endif

bool Socket::running() const {
  for (auto& core : cores_) {
    if (core->running())
      return true;
  }
  return false;
}

int Socket::get_exitcode() const {
  int exitcode = 0;
  for (auto& core : cores_) {
    exitcode |= core->get_exitcode();
  }
  return exitcode;
}

void Socket::barrier(uint32_t bar_id, uint32_t count, uint32_t core_id) {
  cluster_->barrier(bar_id, count, socket_id_ * cores_.size() + core_id);
}

void Socket::resume(uint32_t core_index) {
  cores_.at(core_index)->resume(-1);
}

Socket::PerfStats Socket::perf_stats() const {
  PerfStats perf_stats;
  perf_stats.icache = icaches_->perf_stats();
  perf_stats.dcache = dcaches_->perf_stats();
  return perf_stats;
}