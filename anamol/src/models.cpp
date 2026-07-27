#include "models.h"

#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "instr.h"
#include "params.h"
#include "resource_registry.h"  // RESOURCE_REGISTRY (generated)

namespace analytical {

using std::vector;

const uint64_t CACHE_LINE_SIZE = 64;
const uint64_t LARGE_CONSTANT = 1000000000ULL;

// Sentinel throughput returned when a resource is not a bottleneck for this
// window (no relevant instruction type present, or zero issue width).
// Swap the active return to experiment:
//   static_cast<double>(window.size())        natural window ceiling (current)
//   std::numeric_limits<double>::infinity()   true "infinity"
//   0.0                                       treat as fully blocking
inline double unbottlenecked_thr(size_t window_size) {
  return static_cast<double>(window_size);
}

////////////////////////////////////////////////////////////////////////////
// Base calculations
////////////////////////////////////////////////////////////////////////////

/* Implements Algorithm 1 from the paper. */
unsigned resp_cycle(
    unsigned req_cycle,  // request cycle for the instruction
    const Instr& instr,  // instruction
    std::map<unsigned, unsigned>&
        last_req_cycles,  // last request cycles for each cache line
    std::map<unsigned, unsigned>&
        last_resp_cycles  // last response cycles for each cache line
) {
  uint64_t resp_cycle;
  if (instr.is_load) {
    /* Improve on the in-order cache simulation's memory model
       for load instructions. */
    if (instr.read_address == 0)
      return req_cycle + instr.exe_latency;
    uint64_t cache_line = instr.read_address / CACHE_LINE_SIZE;
    auto it = last_req_cycles.find(cache_line);
    if (it == last_req_cycles.end()) {
      /* If there isn't already an entry for the cache line,
         we need to make a request, so set entries for the
         cache line in all state variables. */
      last_req_cycles[cache_line] = 0;
      last_resp_cycles[cache_line] = 0;
    }
    uint64_t prev_resp_cycle = last_resp_cycles[cache_line];
    resp_cycle = std::max(static_cast<uint64_t>(req_cycle + instr.exe_latency),
                          prev_resp_cycle);

    // update state variables
    last_resp_cycles[cache_line] = resp_cycle;
    last_req_cycles[cache_line] = req_cycle;
  } else {
    /* Constant latency for non-load instructions. */
    resp_cycle = instr.exe_latency + req_cycle;
  }

  return resp_cycle;
}

// 1. ROB (Reorder Buffer) Throughput
double get_thr_rob(const vector<Instr>& window, uint16_t rob_size) {
  instr_id_t k = window.size();
  instr_id_t firstID = window[0].id;

  vector<unsigned> arrival(k);      // a_i
  vector<unsigned> start_cycle(k);  // s_i
  vector<unsigned> finish(k);       // f_i
  vector<unsigned> commit(k);       // c_i

  std::map<unsigned, unsigned> last_req_cycles;
  std::map<unsigned, unsigned> last_resp_cycles;

  for (unsigned i = 0; i < k; ++i) {
    const auto& instr = window[i];

    // a_i = c_{i-ROB}
    // For first ROB instructions, arrival time is 0
    if (i < rob_size) {
      arrival[i] = 0;
    } else {
      arrival[i] = commit[i - rob_size];
    }

    // s_i = max(a_i, max{f_d | d in Dep(i)})
    instr_id_t max_dep_finish = arrival[i];
    for (instr_id_t dep_id : instr.deps) {
      int dep_idx = dep_id - firstID;
      if (dep_idx >= 0 && dep_idx < (int)i) {
        // Dependency is in the current window and already processed
        max_dep_finish = std::max(max_dep_finish, finish[dep_idx]);
      }
      // Dependencies outside the window or from future instructions are ignored
      // (assumes they're either complete or will be handled by dependency
      // tracking)
    }
    start_cycle[i] = max_dep_finish;

    // f_i = RespCycle(s_i, instr_i)
    finish[i] =
        resp_cycle(start_cycle[i], instr, last_req_cycles, last_resp_cycles);

    // c_i = max(f_i, c_{i-1})
    // Instructions must commit in order
    if (i == 0) {
      commit[i] = finish[i];
    } else {
      commit[i] = std::max(finish[i], commit[i - 1]);
    }
  }

  // throughput = k / c_{k-1}
  // Total cycles is the commit time of the last instruction
  uint32_t total_cycles = commit[k - 1];

  // Edge case: if all instructions complete at cycle 0 (very unlikely)
  // Return a very high throughput rather than dividing by zero
  if (total_cycles == 0) {
    return k;
  }

  return (double)k / total_cycles;
}

// 2. Load Queue Throughput
double get_thr_load_queue(const vector<Instr>& window,
                          uint16_t load_queue_size) {
  // Filter to only load instructions
  vector<const Instr*> loads;
  for (const auto& instr : window) {
    if (instr.is_load) {
      loads.push_back(&instr);
    }
  }

  // If no loads, this resource is not a bottleneck
  if (loads.empty()) return unbottlenecked_thr(window.size());

  uint32_t k = window.size();
  uint32_t n_loads = loads.size();

  vector<unsigned> arrival(n_loads);      // a_i
  vector<unsigned> start_cycle(n_loads);  // s_i
  vector<unsigned> finish(n_loads);       // f_i
  vector<unsigned> commit(n_loads);       // c_i

  std::map<unsigned, unsigned> last_req_cycles;
  std::map<unsigned, unsigned> last_resp_cycles;

  for (int i = 0; i < (int)n_loads; ++i) {
    const auto& instr = *loads[i];

    // a_i = c_{i-LQ}
    // For first LQ loads, arrival time is 0
    if (i < (int)load_queue_size) {
      arrival[i] = 0;
    } else {
      arrival[i] = commit[i - load_queue_size];
    }

    // s_i = a_i (no dependency constraints for Load Queue model)
    start_cycle[i] = arrival[i];

    // f_i = RespCycle(s_i, instr_i)
    finish[i] =
        resp_cycle(start_cycle[i], instr, last_req_cycles, last_resp_cycles);

    // c_i = max(f_i, c_{i-1})
    // Loads must commit in program order
    if (i == 0) {
      commit[i] = finish[i];
    } else {
      commit[i] = std::max(finish[i], commit[i - 1]);
    }
  }

  // throughput = k / c_{n_loads-1}
  // Total cycles based on last load's commit time
  uint32_t total_cycles = commit[n_loads - 1];

  // Edge case: if all loads complete at cycle 0
  if (total_cycles == 0) return unbottlenecked_thr(window.size());

  return (double)k / total_cycles;
}

// 3. Store Queue Throughput
double get_thr_store_queue(const vector<Instr>& window,
                           uint16_t store_queue_size) {
  // Filter to only store instructions
  vector<const Instr*> stores;
  for (const auto& instr : window) {
    if (instr.is_store) {
      stores.push_back(&instr);
    }
  }

  // If no stores, this resource is not a bottleneck
  if (stores.empty()) return unbottlenecked_thr(window.size());

  uint32_t k = window.size();
  uint32_t n_stores = stores.size();

  vector<unsigned> arrival(n_stores);      // a_i
  vector<unsigned> start_cycle(n_stores);  // s_i
  vector<unsigned> finish(n_stores);       // f_i
  vector<unsigned> commit(n_stores);       // c_i

  std::map<unsigned, unsigned> last_req_cycles;
  std::map<unsigned, unsigned> last_resp_cycles;

  for (int i = 0; i < (int)n_stores; ++i) {
    const auto& instr = *stores[i];

    // a_i = c_{i-SQ}
    // For first SQ stores, arrival time is 0
    if (i < (int)store_queue_size) {
      arrival[i] = 0;
    } else {
      arrival[i] = commit[i - store_queue_size];
    }

    // s_i = a_i (no dependency constraints for Store Queue model)
    start_cycle[i] = arrival[i];

    // f_i = RespCycle(s_i, instr_i)
    finish[i] =
        resp_cycle(start_cycle[i], instr, last_req_cycles, last_resp_cycles);

    // c_i = max(f_i, c_{i-1})
    // Stores must commit in program order
    if (i == 0) {
      commit[i] = finish[i];
    } else {
      commit[i] = std::max(finish[i], commit[i - 1]);
    }
  }

  // throughput = k / c_{n_stores-1}
  // Total cycles based on last store's commit time
  uint32_t total_cycles = commit[n_stores - 1];

  // Edge case: if all stores complete at cycle 0
  if (total_cycles == 0) return unbottlenecked_thr(window.size());

  return (double)k / total_cycles;
}

// 4. ALU Issue Width Throughput
double get_thr_alu_issue(const vector<Instr>& window,
                         uint16_t alu_issue_width) {
  // Count ALU instructions in the window
  int n_alu = 0;
  for (const auto& instr : window) {
    n_alu += instr.is_alu;
  }

  // Handle edge case: no ALU instructions
  if (n_alu == 0) return unbottlenecked_thr(window.size());

  // Protect against zero issue width and ensure at least 1 cycle
  // if (alu_issue_width == 0) return window.size();
  uint32_t k = window.size();
  double cycles_needed = (double)n_alu / alu_issue_width;
  if (cycles_needed < 1.0) cycles_needed = 1.0;

  return k / cycles_needed;
}

// 4a. ALU Multiply/Divide Issue Width Throughput
double get_thr_alu_mult_div_issue(const vector<Instr>& window,
                                  uint16_t alu_mult_div_issue_width) {
  // Count ALU multiply and divide instructions in the window
  int n_mult_div = 0;
  for (const auto& instr : window) {
    n_mult_div += instr.is_alu_mult_div;
  }

  // Handle edge case: no MUL/DIV instructions
  if (n_mult_div == 0) return unbottlenecked_thr(window.size());

  // Protect against zero issue width and ensure at least 1 cycle
  // if (alu_mult_div_issue_width == 0) return window.size();
  uint32_t k = window.size();
  double cycles_needed = (double)n_mult_div / alu_mult_div_issue_width;
  if (cycles_needed < 1.0) cycles_needed = 1.0;

  return k / cycles_needed;
}

// 5. Floating-Point Issue Width Throughput
double get_thr_fp_issue(const vector<Instr>& window, uint16_t fp_issue_width) {
  // Count FP instructions in the window
  int n_fp = 0;
  for (const auto& instr : window) {
    n_fp += instr.is_fp;
  }

  // Handle edge case: no FP instructions
  if (n_fp == 0) return unbottlenecked_thr(window.size());

  // Protect against zero issue width and ensure at least 1 cycle
  // if (fp_issue_width == 0) return window.size();
  uint32_t k = window.size();
  double cycles_needed = (double)n_fp / fp_issue_width;
  if (cycles_needed < 1.0) cycles_needed = 1.0;

  return k / cycles_needed;
}

// 5a. FP Multiply/Divide Issue Width Throughput
double get_thr_fp_mult_div_issue(const vector<Instr>& window,
                                 uint16_t fp_mult_div_issue_width) {
  // Count FP multiply/divide/sqrt/FMA instructions in the window
  int n_fp_mult_div = 0;
  for (const auto& instr : window) {
    n_fp_mult_div += instr.is_fp_mult_div;
  }

  // Handle edge case: no FP mult/div instructions
  if (n_fp_mult_div == 0) return unbottlenecked_thr(window.size());

  // Protect against zero issue width and ensure at least 1 cycle
  // if (fp_mult_div_issue_width == 0) return window.size();
  uint32_t k = window.size();
  double cycles_needed = (double)n_fp_mult_div / fp_mult_div_issue_width;
  if (cycles_needed < 1.0) cycles_needed = 1.0;

  return k / cycles_needed;
}

// 6. Load-Store Issue Width Throughput
double get_thr_ls_issue(const vector<Instr>& window, uint16_t ls_issue_width) {
  // Count Load/Store instructions in the window
  int n_ls = 0;
  for (const auto& instr : window) {
    n_ls += instr.is_load + instr.is_store;
  }

  // Handle edge case: no Load/Store instructions
  if (n_ls == 0) return unbottlenecked_thr(window.size());

  // Protect against zero issue width and ensure at least 1 cycle
  // if (ls_issue_width == 0) return window.size();
  uint32_t k = window.size();
  double cycles_needed = (double)n_ls / ls_issue_width;
  if (cycles_needed < 1.0) cycles_needed = 1.0;

  return k / cycles_needed;
}

// 7. Load/Load-Store Pipes Lower Bound Throughput
double get_thr_load_ls_pipes_lower(const vector<Instr>& window,
                                   uint16_t num_ls_pipes,
                                   uint16_t num_load_pipes) {
  // Based on worst-case allocation scenario
  int n_load = 0, n_store = 0;
  for (const auto& instr : window) {
    n_load += instr.is_load;
    n_store += instr.is_store;
  }

  if (n_load == 0 && n_store == 0) return unbottlenecked_thr(window.size());

  double t_max = (double)n_load / (num_ls_pipes + num_load_pipes) +
                 (double)n_store / num_ls_pipes;

  return window.size() / t_max;
}

// 8. Load/Load-Store Pipes Upper Bound Throughput
double get_thr_load_ls_pipes_upper(const vector<Instr>& window,
                                   uint16_t num_ls_pipes,
                                   uint16_t num_load_pipes) {
  // Based on best-case allocation scenario
  int n_load = 0, n_store = 0;
  for (const auto& instr : window) {
    n_load += instr.is_load;
    n_store += instr.is_store;
  }

  if (n_load == 0 && n_store == 0) return unbottlenecked_thr(window.size());

  // Best-case: stores use LSPs while loads use LPs concurrently
  // Time to complete all stores using LS pipes
  double t_store = (double)n_store / num_ls_pipes;

  // During t_store cycles, load-only pipes can process loads
  double loads_via_load_pipes = t_store * num_load_pipes;

  // Remaining loads after concurrent execution
  double n_remaining_load = std::max(0.0, n_load - loads_via_load_pipes);

  // Remaining loads share LS pipes after stores complete
  double t_min = t_store + n_remaining_load / (num_ls_pipes + num_load_pipes);

  return window.size() / t_min;
}

// 9. I-Cache Fills Throughput
double get_thr_icache_fills(const vector<Instr>& window,
                            uint16_t max_icache_fills) {
  if (window.empty()) {
    return 0.0;
  }

  /* Stores the state related to in-flight requests. Keys are the
     cache lines, values are the cycles at which the requests
     complete. */
  std::map<uint64_t, uint64_t> in_flight_requests;

  /* Cycle at which the previous instruction was issued. For this
     simulation, we assume in-order issue. */
  uint64_t prev_inst_ready_cycle = 0;

  for (const auto& instr : window) {
    uint64_t cache_line = instr.IP / CACHE_LINE_SIZE;
    uint64_t fill_latency = instr.fetch_latency;

    auto it = in_flight_requests.find(cache_line);

    if (it != in_flight_requests.end()) {
      /* Request for this instruction's cache line is already in flight. */
      uint64_t completion_cycle = it->second;

      /* Enforce in-order constraint: even if the cache line arrives, the
         instruction isn't ready until the previous one is. */
      prev_inst_ready_cycle = std::max(prev_inst_ready_cycle, completion_cycle);
    } else {
      /* We need to issue an icache request for this instruction's cache line.
       */
      uint64_t next_available_slot_cycle = prev_inst_ready_cycle;

      while (in_flight_requests.size() >= max_icache_fills) {
        uint64_t earliest_finish_time = LARGE_CONSTANT;
        uint64_t earliest_cache_line = 0;

        for (const auto& pair : in_flight_requests) {
          if (pair.second < earliest_finish_time) {
            earliest_finish_time = pair.second;
            earliest_cache_line = pair.first;
          }
        }

        next_available_slot_cycle = earliest_finish_time;
        in_flight_requests.erase(earliest_cache_line);
      }

      uint64_t request_start_cycle =
          std::max(prev_inst_ready_cycle, next_available_slot_cycle);
      uint64_t completion_cycle = request_start_cycle + fill_latency;

      in_flight_requests[cache_line] = completion_cycle;
      prev_inst_ready_cycle = completion_cycle;
    }
  }

  uint64_t total_cycles = prev_inst_ready_cycle;
  size_t total_instructions = window.size();

  if (total_cycles == 0) {
    return static_cast<double>(total_instructions);
  }

  return static_cast<double>(total_instructions) /
         static_cast<double>(total_cycles);
}

// 10. Fetch Buffers Throughput
double get_thr_fetch_buffers(const vector<Instr>& window,
                             uint16_t num_fetch_buffers) {
  if (window.empty()) {
    return 0.0;
  }

  /* Tracks the filled state of the fetch buffer. */
  std::map<instr_id_t, uint64_t> buffer_state;

  uint64_t final_cycle = 0;
  instr_id_t finishing_instr;

  for (const auto& instr : window) {
    if (buffer_state.size() == num_fetch_buffers) {
      /* We need to wait for space to become free. */
      uint64_t earliest_finish_time = LARGE_CONSTANT;
      for (const auto& pair : buffer_state) {
        if (pair.second < earliest_finish_time) {
          earliest_finish_time = pair.second;
          finishing_instr = pair.first;
        }
      }
      buffer_state.erase(finishing_instr);
      final_cycle = earliest_finish_time + instr.fetch_latency;
    } else {
      final_cycle = instr.fetch_latency;
    }
    buffer_state[instr.id] = final_cycle;
  }

  uint64_t total_cycles = final_cycle;
  size_t total_instructions = window.size();

  if (total_cycles == 0) {
    return static_cast<double>(total_instructions);
  }

  return static_cast<double>(total_instructions) /
         static_cast<double>(total_cycles);
}

// main entry
PerResThrVecs get_throughput(vector<Instr> instr_trace, int window_size,
                             std::optional<bool> latency_dep_filter) {
  if (instr_trace.empty()) return {};

  PerResThrVecs PER_RES_THR_VECS;

  size_t num_windows = (instr_trace.size() + window_size - 1) / window_size;

  // Phase 1 (serial): build a flat list of (resource, param-combo) work units
  // and pre-populate PER_RES_THR_VECS with correct sizes and metadata.
  // Pre-allocation here means Phase 2 threads never resize any container,
  // so writes to different (res_idx, p_idx) slots are race-free.
  struct ParamUnit {
    size_t res_idx;
    size_t p_idx;
    ThrFunc func;
    std::vector<uint16_t> params;
  };
  std::vector<ParamUnit> param_units;

  for (const auto& entry : RESOURCE_REGISTRY) {
    if (!entry.enabled) continue;
    if (latency_dep_filter.has_value() &&
        entry.latency_dependent != latency_dep_filter.value())
      continue;

    size_t res_idx = static_cast<size_t>(entry.resource);
    auto& vecs = PER_RES_THR_VECS[res_idx];

    size_t p_idx = 0;
    for (const auto& params : entry.sweep) {
      if (vecs.size() <= p_idx) vecs.resize(p_idx + 1);

      ThrVec& tv = vecs[p_idx];
      tv.double_params = (params.size() > 1);
      tv.p0 = params.size() > 0 ? params[0] : 0;
      tv.p1 = params.size() > 1 ? params[1] : 0;
      tv.data.assign(num_windows, 0.0);

      param_units.push_back({res_idx, p_idx, entry.func, params});
      ++p_idx;
    }
  }

  // Phase 2 (parallel): one thread per (resource, param-combo) work unit.
  // Each unit owns a unique (res_idx, p_idx) slot → no races, no critical
  // sections, no thread-local storage needed.
#pragma omp parallel for schedule(dynamic)
  for (int pu_idx = 0; pu_idx < (int)param_units.size(); ++pu_idx) {
    const auto& pu = param_units[pu_idx];
    ThrVec& tv = PER_RES_THR_VECS[pu.res_idx][pu.p_idx];

    for (int win_idx = 0; win_idx < (int)num_windows; ++win_idx) {
      size_t start_idx = (size_t)win_idx * window_size;
      size_t end_idx =
          std::min(start_idx + (size_t)window_size, instr_trace.size());

      vector<Instr> window(instr_trace.begin() + start_idx,
                           instr_trace.begin() + end_idx);

      tv.data[win_idx] = pu.func(window, pu.params);
    }
  }

  return PER_RES_THR_VECS;
}

// Single-config throughput: computes throughput for exactly one set of param
// values.
PerResThrVecs get_throughput_single_config(
    vector<Instr> instr_trace, int window_size,
    const std::map<std::string, uint16_t>& config,
    std::optional<bool> latency_dep_filter) {
  if (instr_trace.empty()) return {};

  PerResThrVecs PER_RES_THR_VECS;

  size_t num_windows = (instr_trace.size() + window_size - 1) / window_size;

  struct ParamUnit {
    size_t res_idx;
    ThrFunc func;
    std::vector<uint16_t> params;
  };
  std::vector<ParamUnit> param_units;

  for (const auto& entry : RESOURCE_REGISTRY) {
    if (!entry.enabled) continue;
    if (latency_dep_filter.has_value() &&
        entry.latency_dependent != latency_dep_filter.value())
      continue;

    // Build param vector from config using entry.param_names
    std::vector<uint16_t> params;
    for (const auto& pname : entry.param_names) {
      auto it = config.find(pname);
      if (it != config.end()) {
        params.push_back(it->second);
      } else {
        std::cerr << "Warning: param '" << pname
                  << "' not found in config for resource '" << entry.name
                  << "', using 0\n";
        params.push_back(0);
      }
    }

    size_t res_idx = static_cast<size_t>(entry.resource);
    auto& vecs = PER_RES_THR_VECS[res_idx];
    vecs.resize(1);

    ThrVec& tv = vecs[0];
    tv.double_params = (params.size() > 1);
    tv.p0 = params.size() > 0 ? params[0] : 0;
    tv.p1 = params.size() > 1 ? params[1] : 0;
    tv.data.assign(num_windows, 0.0);

    param_units.push_back({res_idx, entry.func, params});
  }

#pragma omp parallel for schedule(dynamic)
  for (int pu_idx = 0; pu_idx < (int)param_units.size(); ++pu_idx) {
    const auto& pu = param_units[pu_idx];
    ThrVec& tv = PER_RES_THR_VECS[pu.res_idx][0];

    for (int win_idx = 0; win_idx < (int)num_windows; ++win_idx) {
      size_t start_idx = (size_t)win_idx * window_size;
      size_t end_idx =
          std::min(start_idx + (size_t)window_size, instr_trace.size());

      vector<Instr> window(instr_trace.begin() + start_idx,
                           instr_trace.begin() + end_idx);

      tv.data[win_idx] = pu.func(window, pu.params);
    }
  }

  return PER_RES_THR_VECS;
}

// ROB Latency Analysis Implementation
std::vector<RobLatencyData> get_rob_latency_analysis(
    const std::vector<Instr>& instr_trace,
    const std::vector<uint16_t>& rob_sizes_in) {
  if (instr_trace.empty()) return {};

  std::vector<RobLatencyData> results;

  // Default sweep: {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}
  std::vector<uint16_t> rob_sizes;
  if (rob_sizes_in.empty()) {
    for (uint16_t size = 1; size <= 1024; size *= 2) {
      rob_sizes.push_back(size);
    }
  } else {
    rob_sizes = rob_sizes_in;
  }

  std::cout << "\nRunning ROB latency analysis for " << rob_sizes.size()
            << " configurations...\n";

  for (uint16_t rob_size : rob_sizes) {
    RobLatencyData data;
    data.rob_size = rob_size;

    instr_id_t k = instr_trace.size();
    instr_id_t firstID = instr_trace[0].id;

    std::vector<unsigned> arrival(k);      // a_i
    std::vector<unsigned> start_cycle(k);  // s_i
    std::vector<unsigned> finish(k);       // f_i
    std::vector<unsigned> commit(k);       // c_i

    std::map<unsigned, unsigned> last_req_cycles;
    std::map<unsigned, unsigned> last_resp_cycles;

    // Run ROB simulation for entire trace
    for (unsigned i = 0; i < k; ++i) {
      const auto& instr = instr_trace[i];

      // a_i = c_{i-ROB}
      if (i < rob_size) {
        arrival[i] = 0;
      } else {
        arrival[i] = commit[i - rob_size];
      }

      // s_i = max(a_i, max{f_d | d in Dep(i)})
      instr_id_t max_dep_finish = arrival[i];
      for (instr_id_t dep_id : instr.deps) {
        int dep_idx = dep_id - firstID;
        if (dep_idx >= 0 && dep_idx < (int)i) {
          max_dep_finish = std::max(max_dep_finish, finish[dep_idx]);
        }
      }
      start_cycle[i] = max_dep_finish;

      // f_i = RespCycle(s_i, instr_i)
      finish[i] =
          resp_cycle(start_cycle[i], instr, last_req_cycles, last_resp_cycles);

      // c_i = max(f_i, c_{i-1})
      if (i == 0) {
        commit[i] = finish[i];
      } else {
        commit[i] = std::max(finish[i], commit[i - 1]);
      }
    }

    // Compute overall throughput
    uint32_t final_commit = commit[k - 1];
    if (final_commit == 0) {
      data.overall_throughput = k;
    } else {
      data.overall_throughput = (double)k / final_commit;
    }

    // Collect latency values
    data.issue_latencies.reserve(k);
    data.commit_latencies.reserve(k);
    data.exec_latencies.reserve(k);

    for (unsigned i = 0; i < k; ++i) {
      data.issue_latencies.push_back(start_cycle[i] - arrival[i]);
      data.commit_latencies.push_back(commit[i] - finish[i]);
      data.exec_latencies.push_back(finish[i] - start_cycle[i]);
    }

    results.push_back(data);

    std::cout << "  ROB size " << rob_size
              << ": throughput = " << data.overall_throughput << " IPC\n";
  }

  std::cout << "ROB latency analysis complete.\n";
  return results;
}

// Minimal NumPy .npy v1.0 writer for 2D float64 arrays in C-order.
// shape: (rows, cols), data: rows*cols doubles, row-major.
static void write_npy_2d_float64(const std::string& filename, size_t rows,
                                 size_t cols, const std::vector<double>& data) {
  // Magic string: \x93NUMPY
  const unsigned char magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
  const uint8_t major = 1;
  const uint8_t minor = 0;

  // Build the header dict as a Python literal string.
  // We use little-endian float64: "<f8", Fortran order: False, and given shape.
  std::ostringstream header_ss;
  header_ss << "{'descr': '<f8', 'fortran_order': False, 'shape': (" << rows;
  if (cols == 1) {
    header_ss << ",)";  // 1D shape (rows,)
  } else {
    header_ss << ", " << cols << "),";
  }
  header_ss << "}";

  std::string header = header_ss.str();

  // Pad with spaces and newline so that (magic + 2 + 2 + header) % 64 == 0.
  // Layout: magic(6) + major(1) + minor(1) + header_len_le(2) + header_bytes
  const size_t header_len = header.size();
  // Compute total header + prefix length (not including magic+version+len
  // field) In v1.0 the 2-byte little-endian length is just the header length.
  size_t pre_header = sizeof(magic) + 2 /*major+minor*/ + 2 /*len field*/;
  size_t total = pre_header + header_len;
  size_t pad = 64 - (total % 64);
  if (pad == 64) pad = 0;

  header.append(pad, ' ');
  // Last char must be newline
  if (header.empty() || header.back() != '\n') {
    if (!header.empty())
      header.back() = '\n';
    else
      header.push_back('\n');
  }

  uint16_t header_len_le = static_cast<uint16_t>(header.size());

  std::ofstream ofs(filename, std::ios::binary);
  if (!ofs) {
    throw std::runtime_error("Failed to open file for writing: " + filename);
  }

  ofs.write(reinterpret_cast<const char*>(magic), sizeof(magic));
  ofs.put(static_cast<char>(major));
  ofs.put(static_cast<char>(minor));
  ofs.write(reinterpret_cast<const char*>(&header_len_le),
            sizeof(header_len_le));
  ofs.write(header.data(), header.size());

  // Data is already in row-major order.
  ofs.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size() * sizeof(double)));
  ofs.close();
}

void export_throughputs(PerResThrVecs PER_RES_THR_VECS,
                        const std::string& output_dir) {
  // Ensure output directory exists.
  std::system(("mkdir -p " + output_dir).c_str());

  for (const auto& entry : RESOURCE_REGISTRY) {
    if (!entry.enabled) continue;

    size_t r = static_cast<size_t>(entry.resource);
    const auto& thr_vecs = PER_RES_THR_VECS[r];

    // Skip resources with no data
    if (thr_vecs.empty()) continue;

    // Infer number of windows from first ThrVec
    const size_t num_windows = thr_vecs[0].data.size();
    if (num_windows == 0) continue;

    const bool double_params = thr_vecs[0].double_params;
    // col 0: param count (1 or 2)
    // col 1: p0
    // col 2: p1 (if double_params)
    const size_t param_cols = double_params ? 3 : 2;
    const size_t rows = thr_vecs.size();
    const size_t cols = param_cols + num_windows;

    std::vector<double> array(rows * cols, 0.0);

    for (size_t i = 0; i < rows; ++i) {
      const ThrVec& tv = thr_vecs[i];
      size_t base = i * cols;

      array[base + 0] = double_params ? 2.0 : 1.0;
      array[base + 1] = static_cast<double>(tv.p0);
      if (double_params) {
        array[base + 2] = static_cast<double>(tv.p1);
      }

      const size_t w = std::min(num_windows, tv.data.size());
      for (size_t j = 0; j < w; ++j) {
        array[base + param_cols + j] = tv.data[j];
      }
    }

    // Filename uses the canonical resource name from the registry.
    std::string fname = output_dir + "/thr_" + entry.name + ".npy";
    write_npy_2d_float64(fname, rows, cols, array);
  }
}

void export_latency_analysis(const std::vector<RobLatencyData>& latency_data,
                             const std::string& output_dir) {
  if (latency_data.empty()) return;

  std::cout << "\nExporting latency analysis to " << output_dir << " ...\n";

  // Ensure output directory exists
  std::system(("mkdir -p " + output_dir).c_str());

  size_t num_rob_sizes = latency_data.size();
  size_t num_instructions = latency_data[0].issue_latencies.size();

  // 1. Export overall throughput: shape (11, 2) [rob_size, throughput]
  {
    std::vector<double> thr_data(num_rob_sizes * 2);
    for (size_t i = 0; i < num_rob_sizes; ++i) {
      thr_data[i * 2 + 0] = latency_data[i].rob_size;
      thr_data[i * 2 + 1] = latency_data[i].overall_throughput;
    }
    write_npy_2d_float64(output_dir + "/rob_latency_overall_thr.npy",
                         num_rob_sizes, 2, thr_data);
    std::cout << "  Wrote rob_latency_overall_thr.npy (" << num_rob_sizes
              << " x 2)\n";
  }

  // 2. Export issue latencies: shape (11, k)
  {
    std::vector<double> issue_data(num_rob_sizes * num_instructions);
    for (size_t i = 0; i < num_rob_sizes; ++i) {
      for (size_t j = 0; j < num_instructions; ++j) {
        issue_data[i * num_instructions + j] =
            latency_data[i].issue_latencies[j];
      }
    }
    write_npy_2d_float64(output_dir + "/rob_latency_issue.npy", num_rob_sizes,
                         num_instructions, issue_data);
    std::cout << "  Wrote rob_latency_issue.npy (" << num_rob_sizes << " x "
              << num_instructions << ")\n";
  }

  // 3. Export commit latencies: shape (11, k)
  {
    std::vector<double> commit_data(num_rob_sizes * num_instructions);
    for (size_t i = 0; i < num_rob_sizes; ++i) {
      for (size_t j = 0; j < num_instructions; ++j) {
        commit_data[i * num_instructions + j] =
            latency_data[i].commit_latencies[j];
      }
    }
    write_npy_2d_float64(output_dir + "/rob_latency_commit.npy", num_rob_sizes,
                         num_instructions, commit_data);
    std::cout << "  Wrote rob_latency_commit.npy (" << num_rob_sizes << " x "
              << num_instructions << ")\n";
  }

  // 4. Export exec latencies: shape (11, k) - one row per ROB size
  //    (even though they might be similar across ROB sizes)
  {
    std::vector<double> exec_data(num_rob_sizes * num_instructions);
    for (size_t i = 0; i < num_rob_sizes; ++i) {
      for (size_t j = 0; j < num_instructions; ++j) {
        exec_data[i * num_instructions + j] = latency_data[i].exec_latencies[j];
      }
    }
    write_npy_2d_float64(output_dir + "/rob_latency_exec.npy", num_rob_sizes,
                         num_instructions, exec_data);
    std::cout << "  Wrote rob_latency_exec.npy (" << num_rob_sizes << " x "
              << num_instructions << ")\n";
  }

  std::cout << "Latency analysis export complete.\n";
}

}  // namespace analytical
