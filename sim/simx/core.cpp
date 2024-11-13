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

#include <iostream>
#include <iomanip>
#include <string.h>
#include <assert.h>
#include <util.h>
#include "types.h"
#include "arch.h"
#include "mem.h"
#include "core.h"
#include "debug.h"
#include "constants.h"

using namespace vortex;

#define STORAGE_MULTIPLIER 100

Core::Core(const SimContext& ctx,
           uint32_t core_id,
           Socket* socket,
           Arch &arch,
           const DCRS &dcrs)
  : SimObject(ctx, "core")
  , icache_req_ports(10*1, this) //larger vector for port?
  , icache_rsp_ports(10*1, this)
  , dcache_req_ports(DCACHE_NUM_REQS, this)
  , dcache_rsp_ports(DCACHE_NUM_REQS, this)
  , core_id_(core_id)
  , socket_(socket)
  , arch_(arch)
  , emulator_(arch, dcrs, this)
  , ibuffers_(arch.num_warps(), IBUF_SIZE) //larger ibuffer?
  , scoreboard_(arch_)
  , operands_(ISSUE_WIDTH)
  , dispatchers_((uint32_t)FUType::Count)
  , func_units_((uint32_t)FUType::Count)
  , lsu_demux_(NUM_LSU_BLOCKS)
  , mem_coalescers_(NUM_LSU_BLOCKS)
  , lsu_dcache_adapter_(NUM_LSU_BLOCKS)
  , lsu_lmem_adapter_(NUM_LSU_BLOCKS)
  , pending_icache_(100*arch_.num_warps()) //Make pending_icache larger?
  , commit_arbs_(ISSUE_WIDTH)
{
  char sname[100];
  for (uint32_t i = 0; i < ISSUE_WIDTH; ++i) {
    operands_.at(i) = SimPlatform::instance().create_object<Operand>();
  }

  // create the memory coalescer
  for (uint32_t i = 0; i < NUM_LSU_BLOCKS; ++i) {
    snprintf(sname, 100, "core%d-coalescer%d", core_id, i);
    mem_coalescers_.at(i) = MemCoalescer::Create(sname, LSU_CHANNELS, DCACHE_CHANNELS, DCACHE_WORD_SIZE, LSUQ_OUT_SIZE, 1);
  }

  // create local memory
  snprintf(sname, 100, "core%d-local_mem", core_id);
  local_mem_ = LocalMem::Create(sname, LocalMem::Config{
    (1 << LMEM_LOG_SIZE),
    LSU_WORD_SIZE,
    LSU_NUM_REQS,
    log2ceil(LMEM_NUM_BANKS),
    false
  });

  // create lsu demux
  for (uint32_t i = 0; i < NUM_LSU_BLOCKS; ++i) {
    snprintf(sname, 100, "core%d-lsu_demux%d", core_id, i);
    lsu_demux_.at(i) = LocalMemDemux::Create(sname, 1);
  }

  // create lsu dcache adapter
  for (uint32_t i = 0; i < NUM_LSU_BLOCKS; ++i) {
    snprintf(sname, 100, "core%d-lsu_dcache_adapter%d", core_id, i);
    lsu_dcache_adapter_.at(i) = LsuMemAdapter::Create(sname, DCACHE_CHANNELS, 1);
  }

  // create lsu lmem adapter
  for (uint32_t i = 0; i < NUM_LSU_BLOCKS; ++i) {
    snprintf(sname, 100, "core%d-lsu_lmem_adapter%d", core_id, i);
    lsu_lmem_adapter_.at(i) = LsuMemAdapter::Create(sname, LSU_CHANNELS, 1);
  }

  // connect lsu demux
  for (uint32_t b = 0; b < NUM_LSU_BLOCKS; ++b) {
    lsu_demux_.at(b)->ReqDC.bind(&mem_coalescers_.at(b)->ReqIn);
    mem_coalescers_.at(b)->RspIn.bind(&lsu_demux_.at(b)->RspDC);

    lsu_demux_.at(b)->ReqLmem.bind(&lsu_lmem_adapter_.at(b)->ReqIn);
    lsu_lmem_adapter_.at(b)->RspIn.bind(&lsu_demux_.at(b)->RspLmem);
  }

  // connect coalescer-adapter
  for (uint32_t b = 0; b < NUM_LSU_BLOCKS; ++b) {
    mem_coalescers_.at(b)->ReqOut.bind(&lsu_dcache_adapter_.at(b)->ReqIn);
    lsu_dcache_adapter_.at(b)->RspIn.bind(&mem_coalescers_.at(b)->RspOut);
  }

  // connect adapter-dcache
  for (uint32_t b = 0; b < NUM_LSU_BLOCKS; ++b) {
    for (uint32_t c = 0; c < DCACHE_CHANNELS; ++c) {
      uint32_t i = b * DCACHE_CHANNELS + c;
      lsu_dcache_adapter_.at(b)->ReqOut.at(c).bind(&dcache_req_ports.at(i));
      dcache_rsp_ports.at(i).bind(&lsu_dcache_adapter_.at(b)->RspOut.at(c));
    }
  }

  // connect adapter-lmem
  for (uint32_t b = 0; b < NUM_LSU_BLOCKS; ++b) {
    for (uint32_t c = 0; c < LSU_CHANNELS; ++c) {
      uint32_t i = b * LSU_CHANNELS + c;
      lsu_lmem_adapter_.at(b)->ReqOut.at(c).bind(&local_mem_->Inputs.at(i));
      local_mem_->Outputs.at(i).bind(&lsu_lmem_adapter_.at(b)->RspOut.at(c));
    }
  }

  // initialize dispatchers
  dispatchers_.at((int)FUType::ALU) = SimPlatform::instance().create_object<Dispatcher>(arch, 2, NUM_ALU_BLOCKS, NUM_ALU_LANES);
  dispatchers_.at((int)FUType::FPU) = SimPlatform::instance().create_object<Dispatcher>(arch, 2, NUM_FPU_BLOCKS, NUM_FPU_LANES);
  dispatchers_.at((int)FUType::LSU) = SimPlatform::instance().create_object<Dispatcher>(arch, 2, NUM_LSU_BLOCKS, NUM_LSU_LANES);
  dispatchers_.at((int)FUType::SFU) = SimPlatform::instance().create_object<Dispatcher>(arch, 2, NUM_SFU_BLOCKS, NUM_SFU_LANES);

  // initialize execute units
  func_units_.at((int)FUType::ALU) = SimPlatform::instance().create_object<AluUnit>(this);
  func_units_.at((int)FUType::FPU) = SimPlatform::instance().create_object<FpuUnit>(this);
  func_units_.at((int)FUType::LSU) = SimPlatform::instance().create_object<LsuUnit>(this);
  func_units_.at((int)FUType::SFU) = SimPlatform::instance().create_object<SfuUnit>(this);

  // bind commit arbiters
  for (uint32_t i = 0; i < ISSUE_WIDTH; ++i) {
    snprintf(sname, 100, "core%d-commit-arb%d", core_id, i);
    auto arbiter = TraceSwitch::Create(sname, ArbiterType::RoundRobin, (uint32_t)FUType::Count, 1);
    for (uint32_t j = 0; j < (uint32_t)FUType::Count; ++j) {
      func_units_.at(j)->Outputs.at(i).bind(&arbiter->Inputs.at(j));
    }
    commit_arbs_.at(i) = arbiter;
  }

  // Print the core ID and thread count to run.log
  std::cout << "Core ID: " << this->core_id_ << " has " << this->arch_.num_threads() << " thread(s) and " << this->arch_.num_warps() << " warp(s)." << std::endl;

  this->reset();
}

Core::~Core() {
  std::cout << "Core ID: " << this->core_id_ << " has " << this->arch_.num_threads() << " thread(s) and " << this->arch_.num_warps() << " warp(s)." << std::endl;
  if (BRANCH_PRED) {
    double branch_pred_accuracy = (1.0 - (double)perf_stats_.wrong_pred / (double) perf_stats_.total_branches) * 100.0;
    std::cout << "Wrong Pred: " << perf_stats_.wrong_pred << std::endl;
    std::cout << "Correct Pred: " << (perf_stats_.total_branches - perf_stats_.wrong_pred) << std::endl;
    std::cout << "Total Branches: " << perf_stats_.total_branches << std::endl;
    std::cout << "Core ID: " << this->core_id_ << " has branch pred accuracy of " << branch_pred_accuracy << "%" << std::endl;
  }
}

void Core::reset() {
  emulator_.clear();

  for (auto& commit_arb : commit_arbs_) {
    commit_arb->reset();
  }

  for (auto& ibuf : ibuffers_) {
    ibuf.clear();
  }

  scoreboard_.clear();
  fetch_latch_.clear();
  decode_latch_.clear();
  pending_icache_.clear();

  ibuffer_idx_ = 0;
  pending_instrs_ = 0;
  pending_ifetches_ = 0;

  perf_stats_ = PerfStats();

  this->branch_mispred_flush = false;
  this->squash_in_progress = false;
  this->num_exec_inflight = 0;
  this->draining = false;   
}

void Core::tick() {
  this->branch_mispred_flush = 0; //Reset any flush from the previous cycle
  if (arch_.num_threads() == 1 && BRANCH_PRED) {
    bool active; 
    active |= this->scalar_commit();
    active |= this->scalar_execute();
    active |= this->scalar_issue();
    active |= this->scalar_decode();
    active |= this->scalar_fetch();
    active |= this->scalar_schedule();
    if (active) {
      // DT(3, "In Flight: " << this->num_exec_inflight); 
      DT(3, "----------------------");
    }
  } else {
    this->commit();
    this->execute();
    this->issue();
    this->decode();
    this->fetch();
    this->schedule();
  }

  ++perf_stats_.cycles;
  DPN(2, std::flush);
}

void Core::schedule() {
  auto trace = emulator_.step();
  if (trace == nullptr) {
    ++perf_stats_.sched_idle;
    return;
  }

  // suspend warp until decode
  emulator_.suspend(trace->wid);

  DT(3, "pipeline-schedule: " << *trace);

  // advance to fetch stage
  fetch_latch_.push(trace);
  ++pending_instrs_;
}

bool Core::scalar_schedule() {
  bool active = false;
  if (this->branch_mispred_flush) { // On a flush, clear the latch
    DT(3, "Flushing pipeline-schedule"); 
    fetch_latch_.clear();
    // emulator_.resume(0); // Fetch cannot clear the suspended warp bc it got flushed 
    return active; 
  }

  auto trace = emulator_.step_schedule();
  if (trace == nullptr) {
    ++perf_stats_.sched_idle;
    return active;
  }
  active = true; 

  // Only suspend the warp until instruction finished fetching
  emulator_.suspend(trace->wid); 


  DT(3, "pipeline-schedule: " << *trace);

  // advance to fetch stage
  fetch_latch_.push(trace);
  ++pending_instrs_;
  return active; 
}

void Core::fetch() {
  perf_stats_.ifetch_latency += pending_ifetches_;

  // handle icache response
  auto& icache_rsp_port = icache_rsp_ports.at(0);
  if (!icache_rsp_port.empty()){
    auto& mem_rsp = icache_rsp_port.front();
    auto trace = pending_icache_.at(mem_rsp.tag);
    decode_latch_.push(trace);
    DT(3, "icache-rsp: addr=0x" << std::hex << trace->PC << ", tag=0x" << mem_rsp.tag << std::dec << ", " << *trace);
    pending_icache_.release(mem_rsp.tag);
    icache_rsp_port.pop();
    --pending_ifetches_;
  }

  // send icache request
  if (fetch_latch_.empty())
    return;
  auto trace = fetch_latch_.front();
  MemReq mem_req;
  mem_req.addr  = trace->PC;
  mem_req.write = false;
  mem_req.tag   = pending_icache_.allocate(trace);
  mem_req.cid   = trace->cid;
  mem_req.uuid  = trace->uuid;
  icache_req_ports.at(0).push(mem_req, 2);
  DT(3, "icache-req: addr=0x" << std::hex << mem_req.addr << ", tag=0x" << mem_req.tag << std::dec << ", " << *trace);
  fetch_latch_.pop();
  ++perf_stats_.ifetches;
  ++pending_ifetches_;
}

bool Core::scalar_fetch() {
  bool active = false; 
  perf_stats_.ifetch_latency += pending_ifetches_;

  // handle icache response
  auto& icache_rsp_port = icache_rsp_ports.at(0);

  this->squash_in_progress = this->squash_in_progress ? (pending_ifetches_ != 0) : this->branch_mispred_flush; 
  // DT(3, this->squash_in_progress << " (" << pending_ifetches_ << " pending ifetches)");

  // On a pipeline flush, it is possible icache-req was already sent when not needed. So need to squash the response when it's received. 
  // Start the squash on a flush. Clear the squash once no more pending_ifetches
  // Squash has 
  if (this->squash_in_progress) {
     if (!icache_rsp_port.empty()) {
      active = true; 
      DT(3, "Squashing icache-rsp in pipeline-fetch (" << pending_ifetches_ << " pending ifetches)");
      icache_rsp_port.pop(); 
      --pending_ifetches_;
     }
    if (pending_ifetches_ == 0) {
      emulator_.resume(0); // Resume once all squashing is done
    }
  }

  if (this->branch_mispred_flush) { // On a flush, push a NOP instr and don't send icache request
    DT(3, "Flushing pipeline-fetch"); 
    // pending_ifetches_ -= pending_icache_.size(); 
    fetch_latch_.clear();
    pending_icache_.clear();
    active = true;
    return active; 
  }
  
  if (!icache_rsp_port.empty()) { // Process the response
    active = true;
    auto& mem_rsp = icache_rsp_port.front();
    DT(3, "Mem rsp: " << std::hex << "tag=0x" << mem_rsp.tag);
    auto trace = pending_icache_.at(mem_rsp.tag);
    emulator_.resume(trace->wid);
    decode_latch_.push(trace);
    DT(3, "icache-rsp: addr=0x" << std::hex << trace->PC << ", tag=0x" << mem_rsp.tag << std::dec << ", " << *trace);
    pending_icache_.release(mem_rsp.tag);
    icache_rsp_port.pop();
    // Response valid so decrement pending_ifetches
      --pending_ifetches_;
  }

  if (fetch_latch_.empty()) {
    active = false;
    return active;
  }

  // send icache request
  active = true; 
  auto trace = fetch_latch_.front();
  emulator_.step_fetch(trace); 
  MemReq mem_req;
  mem_req.addr  = trace->PC;
  mem_req.write = false;
  mem_req.tag   = pending_icache_.allocate(trace);
  mem_req.cid   = trace->cid;
  mem_req.uuid  = trace->uuid;
  icache_req_ports.at(0).push(mem_req, 2);
  DT(3, "icache-req: addr=0x" << std::hex << mem_req.addr << ", tag=0x" << mem_req.tag << std::dec << ", " << *trace);
  fetch_latch_.pop();
  ++perf_stats_.ifetches;
  ++pending_ifetches_;
  return active; 
}

void Core::decode() {
  if (decode_latch_.empty())
    return;

  auto trace = decode_latch_.front();

  // check ibuffer capacity
  auto& ibuffer = ibuffers_.at(trace->wid);
  if (ibuffer.full()) {
    if (!trace->log_once(true)) {
      DT(4, "*** ibuffer-stall: " << *trace);
    }
    ++perf_stats_.ibuf_stalls;
    return;
  } else {
    trace->log_once(false);
  }

  // release warp
  if (!trace->fetch_stall) {
    emulator_.resume(trace->wid);
  }

  DT(3, "pipeline-decode: " << *trace);

  // insert to ibuffer
  ibuffer.push(trace);

  decode_latch_.pop();
}

bool Core::scalar_decode() {
  bool active = false;
  if (this->branch_mispred_flush) { //Flush before any stalling on mispredict
    DT(3, "Flushing pipeline-decode");
    // --pending_instrs_; 
    decode_latch_.clear(); 
    return active;
  }

  if (decode_latch_.empty())
    return active;

  active = true; 
  auto trace = decode_latch_.front();

  // check ibuffer capacity
  auto& ibuffer = ibuffers_.at(trace->wid);



  if (ibuffer.full()) {
    if (!trace->log_once(true)) {
      DT(4, "*** ibuffer-stall: " << *trace);
    }
    ++perf_stats_.ibuf_stalls;
    return active;
  } else {
    trace->log_once(false);
  }

  emulator_.step_decode(trace); 

  DT(3, "pipeline-decode: " << *trace);

  // insert to ibuffer
  ibuffer.push(trace);
  decode_latch_.pop();
  return active; 
}

void Core::issue() {
  // operands to dispatchers
  for (uint32_t i = 0; i < ISSUE_WIDTH; ++i) {
    auto& operand = operands_.at(i);
    if (operand->Output.empty())
      continue;
    auto trace = operand->Output.front();
    if (dispatchers_.at((int)trace->fu_type)->push(i, trace)) {
      operand->Output.pop();
      trace->log_once(false);
    } else {
      if (!trace->log_once(true)) {
        DT(4, "*** dispatch-stall: " << *trace);
      }
    }
  }

  // issue ibuffer instructions
  for (uint32_t i = 0; i < ISSUE_WIDTH; ++i) {
    bool has_instrs = false;
    bool found_match = false;
    for (uint32_t w = 0; w < PER_ISSUE_WARPS; ++w) {
      uint32_t kk = (ibuffer_idx_ + w) % PER_ISSUE_WARPS;
      uint32_t ii = kk * ISSUE_WIDTH + i;
      auto& ibuffer = ibuffers_.at(ii);
      if (ibuffer.empty())
        continue;
      // check scoreboard
      has_instrs = true;
      auto trace = ibuffer.top();
      if (scoreboard_.in_use(trace)) {
        auto uses = scoreboard_.get_uses(trace);
        if (!trace->log_once(true)) {
          DTH(4, "*** scoreboard-stall: dependents={");
          for (uint32_t j = 0, n = uses.size(); j < n; ++j) {
            auto& use = uses.at(j);
            __unused (use);
            if (j) DTN(4, ", ");
            DTN(4, use.reg_type << use.reg_id << "(#" << use.uuid << ")");
          }
          DTN(4, "}, " << *trace << std::endl);
        }
        for (uint32_t j = 0, n = uses.size(); j < n; ++j) {
          auto& use = uses.at(j);
          switch (use.fu_type) {
          case FUType::ALU: ++perf_stats_.scrb_alu; break;
          case FUType::FPU: ++perf_stats_.scrb_fpu; break;
          case FUType::LSU: ++perf_stats_.scrb_lsu; break;
          case FUType::SFU: {
            ++perf_stats_.scrb_sfu;
            switch (use.sfu_type) {
            case SfuType::TMC:
            case SfuType::WSPAWN:
            case SfuType::SPLIT:
            case SfuType::JOIN:
            case SfuType::BAR:
            case SfuType::PRED: ++perf_stats_.scrb_wctl; break;
            case SfuType::CSRRW:
            case SfuType::CSRRS:
            case SfuType::CSRRC: ++perf_stats_.scrb_csrs; break;
            default: assert(false);
            }
          } break;
          default: assert(false);
          }
        }
      } else {
        trace->log_once(false);
        // update scoreboard
        DT(3, "pipeline-scoreboard: " << *trace);
        if (trace->wb) {
          scoreboard_.reserve(trace);
        }
        // to operand stage
        operands_.at(i)->Input.push(trace, 2);
        ibuffer.pop();
        found_match = true;
        break;
      }
    }
    if (has_instrs && !found_match) {
      ++perf_stats_.scrb_stalls;
    }
  }
  ++ibuffer_idx_;
}

bool Core::scalar_issue() {
  bool active = false; 
  // operands to dispatchers
  for (uint32_t i = 0; i < ISSUE_WIDTH; ++i) {
    auto& operand = operands_.at(i);
    // if (this->branch_mispred_flush) {
    //   DT(3, "Flushing pipeline-issue (operand collectors)");
    //   // operand->clear(); 
    // }

    if (operand->Output.empty())
      continue;
    active = true; 
    auto trace = operand->Output.front();
    if (dispatchers_.at((int)trace->fu_type)->push(i, trace)) {
      operand->Output.pop();
      trace->log_once(false);
    } else {
      if (!trace->log_once(true)) {
        DT(4, "*** dispatch-stall: " << *trace);
      }
    }
  }

  // issue ibuffer instructions (check scoreboard and put into appropriate operand collector bank)
  for (uint32_t i = 0; i < ISSUE_WIDTH; ++i) {
    bool has_instrs = false;
    bool found_match = false;
    for (uint32_t w = 0; w < PER_ISSUE_WARPS; ++w) {
      uint32_t kk = (ibuffer_idx_ + w) % PER_ISSUE_WARPS;
      uint32_t ii = kk * ISSUE_WIDTH + i;
      auto& ibuffer = ibuffers_.at(ii);
      if (this->branch_mispred_flush) {
          DT(3, "Flushing pipeline-issue (ibuffer and scoreboard)"); 
          scoreboard_.clear();
          ibuffer.clear();
          // --pending_instrs_; 
          break; 
      }

      if (ibuffer.empty())
        continue;
      // check scoreboard
      active = true; 
      has_instrs = true;
      auto trace = ibuffer.top();

      if (scoreboard_.in_use(trace)) {
        auto uses = scoreboard_.get_uses(trace);
        if (!trace->log_once(true)) {
          DTH(4, "*** scoreboard-stall: dependents={");
          for (uint32_t j = 0, n = uses.size(); j < n; ++j) {
            auto& use = uses.at(j);
            __unused (use);
            if (j) DTN(4, ", ");
            DTN(4, use.reg_type << use.reg_id << "(#" << use.uuid << ")");
          }
          DTN(4, "}, " << *trace << std::endl);
        }
        for (uint32_t j = 0, n = uses.size(); j < n; ++j) {
          auto& use = uses.at(j);
          switch (use.fu_type) {
          case FUType::ALU: ++perf_stats_.scrb_alu; break;
          case FUType::FPU: ++perf_stats_.scrb_fpu; break;
          case FUType::LSU: ++perf_stats_.scrb_lsu; break;
          case FUType::SFU: {
            ++perf_stats_.scrb_sfu;
            switch (use.sfu_type) {
            case SfuType::TMC:
            case SfuType::WSPAWN:
            case SfuType::SPLIT:
            case SfuType::JOIN:
            case SfuType::BAR:
            case SfuType::PRED: ++perf_stats_.scrb_wctl; break;
            case SfuType::CSRRW:
            case SfuType::CSRRS:
            case SfuType::CSRRC: ++perf_stats_.scrb_csrs; break;
            default: assert(false);
            }
          } break;
          default: assert(false);
          }
        }
      } else { //instr not stalled by scoreboard
        trace->log_once(false);
        if (trace->alu_type == AluType::BRANCH && trace->fu_type == FUType::ALU) {
          if (this->num_exec_inflight != 0) { // Wait for execute to empty before jump/branch enters
            DT(4, "*** execute-stage-empty-stall: " << *trace);
            continue; 
          }
          // Nothing is in the execute stage now, so stop other instructions from entering
        // update scoreboard
          DT(3, "pipeline-through-scoreboard: " << *trace);
          if (trace->wb) {
            scoreboard_.reserve(trace);
          }
          DT(4, "Entering execute: " << *trace); 
          this->draining = true; 
          operands_.at(i)->Input.push(trace, 2);
          this->num_exec_inflight++; 
          ibuffer.pop();
          emulator_.step_issue(); // Core.cpp handled everything
          found_match = true;
          continue;
        }
        if (this->draining) {
          DT(4, "*** execute-stage-stalling: " << *trace);
          continue; 
        }
        // update scoreboard
        DT(3, "pipeline-through-scoreboard: " << *trace);
        if (trace->wb) {
          scoreboard_.reserve(trace);
        }
        // to operand stage
        operands_.at(i)->Input.push(trace, 2);
        this->num_exec_inflight++; 
        ibuffer.pop();
        emulator_.step_issue(); // Core.cpp handled everything
        found_match = true;
        continue;
      }
    }
    if (has_instrs && !found_match) {
      ++perf_stats_.scrb_stalls;
    }
  }
  ++ibuffer_idx_;
  return active; 
}

void Core::execute() {
  for (uint32_t i = 0; i < (uint32_t)FUType::Count; ++i) {
    auto& dispatch = dispatchers_.at(i);
    auto& func_unit = func_units_.at(i);
    for (uint32_t j = 0; j < ISSUE_WIDTH; ++j) {
      if (dispatch->Outputs.at(j).empty())
        continue;
      auto trace = dispatch->Outputs.at(j).front();
      func_unit->Inputs.at(j).push(trace, 2);
      dispatch->Outputs.at(j).pop();
    }
  }
}

bool Core::scalar_execute() {
  bool active = false; 

  for (uint32_t i = 0; i < (uint32_t)FUType::Count; ++i) {
    auto& dispatch = dispatchers_.at(i);
    auto& func_unit = func_units_.at(i);
    for (uint32_t j = 0; j < ISSUE_WIDTH; ++j) {
      if (dispatch->Outputs.at(j).empty())
        continue;
      auto trace = dispatch->Outputs.at(j).front();

      DT(3, "Trace to func_unit stage: " << *trace);
      active = true; 

      if ((trace->alu_type == AluType::BRANCH) & (i == (uint32_t)FUType::ALU)) { //Check that instruction in ALU is a branch/jump (IDLE to DRAIN)
        // Check that execute stage is empty (DRAIN to FIRE)
        perf_stats_.total_branches++; 
        emulator_.step_execute(trace); 
        if (trace->halt) { // If "tmc 0" (only thread tries to kill itself), mark the scheduled warp as inactive and flush the pipeline
          this->branch_mispred_flush = true;
        }
        // Fire the branch/jump into the execute stage and flush if it is a mispredict
        if (trace->branch_mispred_flush) {
          DT(3, "Branch Mispredicted! Flushing pipeling! Branch Instr: " << *trace);
          this->branch_mispred_flush = true;
          DT(3, "Flushing pipeline-execute (dispatcher)");             
          perf_stats_.wrong_pred++; 
          pending_instrs_ = 0;
          this->num_exec_inflight = 1; //Only the jump/branch is in flight now
        }
        func_unit->Inputs.at(j).push(trace, 2);
        dispatch->Outputs.at(j).pop();
      } else {
        // Otherwise can send it to a func_unit as normal
        emulator_.step_execute(trace); 
        func_unit->Inputs.at(j).push(trace, 2);
        dispatch->Outputs.at(j).pop();
      }
    }
  }

  return active; 
}

void Core::commit() {
  // process completed instructions
  for (uint32_t i = 0; i < ISSUE_WIDTH; ++i) {
    auto& commit_arb = commit_arbs_.at(i);
    if (commit_arb->Outputs.at(0).empty())
      continue;
    auto trace = commit_arb->Outputs.at(0).front();

    // advance to commit stage
    DT(3, "pipeline-commit: " << *trace);
    assert(trace->cid == core_id_);

    // update scoreboard
    if (trace->eop) {
      if (trace->wb) {
        scoreboard_.release(trace);
      }

      --pending_instrs_;

      perf_stats_.instrs += trace->tmask.count();
    }

    perf_stats_.opds_stalls = 0;
    for (uint32_t i = 0; i < ISSUE_WIDTH; ++i) {
      perf_stats_.opds_stalls += operands_.at(i)->total_stalls();
    }

    commit_arb->Outputs.at(0).pop();

    // delete the trace
    delete trace;
  }
}

bool Core::scalar_commit() {
  bool active = false;

  // process completed instructions
  for (uint32_t i = 0; i < ISSUE_WIDTH; ++i) {
    auto& commit_arb = commit_arbs_.at(i);
    if (commit_arb->Outputs.at(0).empty())
      continue;
    active = true;
    auto trace = commit_arb->Outputs.at(0).front();

    // advance to commit stage
    DT(3, "pipeline-commit: " << *trace);
    assert(trace->cid == core_id_);

    this->num_exec_inflight--; 
    if (trace->alu_type == AluType::BRANCH) {
      this->draining = false;
    }

    emulator_.step_commit(trace); 

    // update scoreboard
    if (trace->eop) {
      // Since scoreboard might have been flushed, only need to release if it is still in the scoreboard
      if (trace->wb && scoreboard_.in_use(trace)) { 
        scoreboard_.release(trace);
      }

      --pending_instrs_;

      perf_stats_.instrs += trace->tmask.count();
    }

    perf_stats_.opds_stalls = 0;
    for (uint32_t i = 0; i < ISSUE_WIDTH; ++i) {
      perf_stats_.opds_stalls += operands_.at(i)->total_stalls();
    }

    commit_arb->Outputs.at(0).pop();


    // delete the trace
    delete trace;

  }
  return active; 
}

int Core::get_exitcode() const {
  return emulator_.get_exitcode();
}

bool Core::running() const {
  // DT(3, "Pending Instrs: " << pending_instrs_);
  return emulator_.running() || (pending_instrs_ != 0);
}

void Core::resume(uint32_t wid) {
  emulator_.resume(wid);
}

bool Core::barrier(uint32_t bar_id, uint32_t count, uint32_t wid) {
  return emulator_.barrier(bar_id, count, wid);
}

bool Core::wspawn(uint32_t num_warps, Word nextPC) {
  return emulator_.wspawn(num_warps, nextPC);
}

void Core::attach_ram(RAM* ram) {
  emulator_.attach_ram(ram);
}

#ifdef VM_ENABLE
void Core::set_satp(uint64_t satp) {
  emulator_.set_satp(satp); //JAEWON wit, tid???
  // emulator_.set_csr(VX_CSR_SATP,satp,0,0); //JAEWON wit, tid???
}
#endif