#ifndef TYPES_H
#define TYPES_H

#include <cstdint>
#include <vector>

using latency_t = uint16_t;

namespace analytical {

using instr_id_t = uint32_t;

enum class branch_t : uint8_t { DIRECT_COND, DIRECT_UNCOND, INDIRECT };

struct Instr {
  uint64_t IP;
  instr_id_t id;
  latency_t exe_latency;
  latency_t fetch_latency;

  bool is_alu;
  bool is_alu_mult_div;
  bool is_simd;
  bool is_fp;
  bool is_fp_mult_div;
  bool is_load;
  bool is_store;
  bool is_isb;
  bool is_branch;

  branch_t branch_type;
  bool branch_taken;
  uint64_t branch_target_addr;
  std::vector<uint32_t> deps;
  uint64_t read_address;
  uint64_t write_address;
};

}  // namespace analytical

#endif  // TYPES_H
