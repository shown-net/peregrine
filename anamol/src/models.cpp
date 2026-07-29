#include "models.h"

#include <cstddef>
#include <map>
#include <vector>

#include "instr.h"

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
inline double unbottlenecked_thr(std::size_t window_size) {
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
  std::size_t total_instructions = window.size();

  if (total_cycles == 0) {
    return static_cast<double>(total_instructions);
  }

  return static_cast<double>(total_instructions) /
         static_cast<double>(total_cycles);
}

}  // namespace analytical
