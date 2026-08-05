#ifndef TYPES_H
#define TYPES_H

#include <cstdint>
#include <vector>

using latency_t = uint16_t;

namespace analytical {

using instr_id_t = uint32_t;

enum class branch_t : uint8_t { DIRECT_COND, DIRECT_UNCOND, INDIRECT };

struct MemoryAccess {
  uint64_t address;
  uint32_t size;
};

struct MicroOp {
  instr_id_t id;
  latency_t exe_latency;
  uint32_t class_flags;
  std::vector<instr_id_t> deps;
  std::vector<MemoryAccess> reads;
  std::vector<MemoryAccess> writes;

  bool is_alu() const { return class_flags & 1U; }
  bool is_alu_mult_div() const { return class_flags & 2U; }
  bool is_fp() const { return class_flags & 8U; }
  bool is_fp_mult_div() const { return class_flags & 16U; }
  bool is_load() const { return class_flags & 32U; }
  bool is_store() const { return class_flags & 64U; }
};

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
  std::vector<MicroOp> micro_ops;
};

}  // namespace analytical

#endif  // TYPES_H
