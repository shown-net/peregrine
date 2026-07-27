#include "parser.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "csv.h"
#include "opcode_categories.h"
#include "peregrine_trace.pb.h"
#include <zstd.h>

namespace analytical {

namespace {

class ZstdFrameReader {
 public:
  explicit ZstdFrameReader(const std::string& path)
      : file(path, std::ios::binary), stream(ZSTD_createDStream()) {
    if (!file) throw std::runtime_error("cannot open Anamol trace: " + path);
    if (!stream) throw std::runtime_error("cannot allocate zstd decoder");
    check(ZSTD_initDStream(stream.get()));
    compressed.resize(ZSTD_DStreamInSize());
    decompressed.resize(ZSTD_DStreamOutSize());
  }

  void readExact(void* destination, size_t size) {
    auto* out = static_cast<uint8_t*>(destination);
    while (size) {
      if (output_pos == output_size) refill();
      const size_t count = std::min(size, output_size - output_pos);
      std::memcpy(out, decompressed.data() + output_pos, count);
      out += count;
      size -= count;
      output_pos += count;
    }
  }

  bool readDelimited(google::protobuf::MessageLite& message) {
    uint32_t size = 0;
    unsigned shift = 0;
    for (; shift < 35; shift += 7) {
      uint8_t byte = 0;
      if (!tryReadByte(byte)) {
        if (shift == 0) return false;
        throw std::runtime_error("trace ends inside protobuf length");
      }
      if (shift == 28 && byte > 0x0f)
        throw std::runtime_error("invalid protobuf length varint");
      size |= uint32_t(byte & 0x7f) << shift;
      if (!(byte & 0x80)) {
        std::vector<uint8_t> payload(size);
        readExact(payload.data(), payload.size());
        if (!message.ParseFromArray(payload.data(), int(payload.size())) ||
            !message.IsInitialized())
          throw std::runtime_error("invalid delimited protobuf message");
        return true;
      }
    }
    throw std::runtime_error("invalid protobuf length varint");
  }

 private:
  struct Deleter {
    void operator()(ZSTD_DStream* value) const { ZSTD_freeDStream(value); }
  };

  void check(size_t code) const {
    if (ZSTD_isError(code)) throw std::runtime_error(ZSTD_getErrorName(code));
  }

  bool tryReadByte(uint8_t& value) {
    if (output_pos == output_size) {
      if (finished) return false;
      refill();
    }
    value = decompressed[output_pos++];
    return true;
  }

  void refill() {
    output_pos = output_size = 0;
    while (!output_size) {
      if (finished) throw std::runtime_error("unexpected end of Anamol trace");
      if (input.pos == input.size) {
        file.read(reinterpret_cast<char*>(compressed.data()), compressed.size());
        input = {compressed.data(), size_t(file.gcount()), 0};
        if (!input.size) throw std::runtime_error("truncated zstd frame");
      }
      ZSTD_outBuffer output{decompressed.data(), decompressed.size(), 0};
      const size_t remaining = ZSTD_decompressStream(stream.get(), &output, &input);
      check(remaining);
      output_size = output.pos;
      if (remaining == 0) {
        if (input.pos != input.size || file.peek() != EOF)
          throw std::runtime_error("Anamol trace must contain one zstd frame");
        finished = true;
      }
    }
  }

  std::ifstream file;
  std::unique_ptr<ZSTD_DStream, Deleter> stream;
  std::vector<uint8_t> compressed;
  std::vector<uint8_t> decompressed;
  ZSTD_inBuffer input{nullptr, 0, 0};
  size_t output_pos = 0;
  size_t output_size = 0;
  bool finished = false;
};

branch_t branch_from_wire(uint32_t value) {
  switch (value) {
    case 1:
      return branch_t::DIRECT_COND;
    case 3:
      return branch_t::INDIRECT;
    default:
      return branch_t::DIRECT_UNCOND;
  }
}

void validate_offsets(const google::protobuf::RepeatedField<uint32_t>& offsets,
                      int records, int flat_size) {
  if (offsets.size() != records + 1 || offsets.Get(0) != 0 ||
      offsets.Get(records) != uint32_t(flat_size))
    throw std::runtime_error("invalid dependency offsets");
  for (int index = 1; index <= records; ++index)
    if (offsets.Get(index) < offsets.Get(index - 1))
      throw std::runtime_error("nonmonotonic dependency offsets");
}

}  // namespace

// Helper to parse hex string to unsigned long
unsigned long parse_hex(const std::string& hex_str) {
  if (hex_str.empty()) return 0;
  return std::stoull(hex_str, nullptr, 16);
}

// Helper to parse memory access from "addr(size)" format
tracing::mem_access_t parse_mem_access(const std::string& str) {
  if (str.empty()) return {0, 0};

  size_t paren_pos = str.find('(');
  if (paren_pos == std::string::npos) {
    return {parse_hex(str), 0};
  }

  unsigned long addr = parse_hex(str.substr(0, paren_pos));
  unsigned int size =
      std::stoul(str.substr(paren_pos + 1, str.find(')') - paren_pos - 1));
  return {addr, size};
}

// Helper to split semicolon-separated list
std::vector<std::string> split_semicolon(const std::string& str) {
  std::vector<std::string> result;
  if (str.empty()) return result;

  std::stringstream ss(str);
  std::string item;
  while (std::getline(ss, item, ';')) {
    if (!item.empty()) {
      result.push_back(item);
    }
  }
  return result;
}

// Helper to parse semicolon-separated hex IPs
std::vector<unsigned long> parse_ip_list(const std::string& str) {
  std::vector<unsigned long> result;
  auto items = split_semicolon(str);
  for (const auto& item : items) {
    result.push_back(parse_hex(item));
  }
  return result;
}

// Helper to parse semicolon-separated memory accesses
std::vector<tracing::mem_access_t> parse_mem_access_list(
    const std::string& str) {
  std::vector<tracing::mem_access_t> result;
  auto items = split_semicolon(str);
  for (const auto& item : items) {
    result.push_back(parse_mem_access(item));
  }
  return result;
}

std::vector<tracing::instr_trace_t> parse_csv(const std::string& csv_path) {
  std::vector<tracing::instr_trace_t> instructions;

  try {
    io::CSVReader<16, io::trim_chars<' ', '\t'>,
                  io::double_quote_escape<',', '"'>>
        in(csv_path);

    in.read_header(
        io::ignore_extra_column | io::ignore_missing_column,
        "IP", "Assembly", "Category", "Opcode",
        "Branch Type", "Branch Taken", "Branch Target Address",
        "Instruction Sync", "Read Registers", "Write Registers",
        "Register Dependent IPs", "Read Addresses", "Write Addresses",
        "Memory Dependent IPs", "Fetch Latency", "Execution Latency");

    std::string ip_str, assembly, category, opcode;
    std::string branch_type, branch_taken, branch_target_addr;
    std::string inst_sync_str;
    std::string read_regs_str, write_regs_str, reg_deps_str;
    std::string read_addrs_str, write_addrs_str, mem_deps_str;
    // Initialize to "0" so missing columns default to zero latency.
    // Latencies will be overridden per cache config when --latencies-npy is used.
    std::string fetch_latency_str = "0", exec_latency_str = "0";

    while (in.read_row(ip_str, assembly, category, opcode, branch_type,
                       branch_taken, branch_target_addr, inst_sync_str,
                       read_regs_str, write_regs_str, reg_deps_str,
                       read_addrs_str, write_addrs_str, mem_deps_str,
                       fetch_latency_str, exec_latency_str)) {
      tracing::instr_trace_t inst;

      inst.ip = parse_hex(ip_str);
      inst.assembly = assembly;
      inst.category = category;
      inst.opcode = opcode;
      inst.branch_type = branch_type;
      inst.branch_taken = (branch_taken == "true" || branch_taken == "True");
      inst.branch_target_addr = parse_hex(branch_target_addr);
      inst.inst_sync = (inst_sync_str == "true" || inst_sync_str == "True");

      inst.read_registers = split_semicolon(read_regs_str);
      inst.write_registers = split_semicolon(write_regs_str);
      inst.reg_dependent_ips = parse_ip_list(reg_deps_str);
      inst.read_addresses = parse_mem_access_list(read_addrs_str);
      inst.write_addresses = parse_mem_access_list(write_addrs_str);
      inst.mem_dependent_ips = parse_ip_list(mem_deps_str);

      inst.fetch_latency =
          fetch_latency_str.empty() ? 0 : std::stoul(fetch_latency_str);
      inst.exe_latency =
          exec_latency_str.empty() ? 0 : std::stoul(exec_latency_str);

      instructions.push_back(inst);
    }
  } catch (const io::error::base& e) {
    throw std::runtime_error(std::string("CSV parsing error: ") + e.what());
  }

  return instructions;
}

Instr convert_to_instr(const tracing::instr_trace_t& inst, instr_id_t id) {
  Instr result;

  result.IP = inst.ip;
  result.id = id;
  result.exe_latency = inst.exe_latency;
  result.fetch_latency = inst.fetch_latency;

  // Determine instruction type from opcode lookup table
  uint8_t opcat = get_opcode_categories(inst.opcode);
  result.is_alu = (opcat & OPCAT_ALU) != 0;
  result.is_alu_mult_div = (opcat & OPCAT_ALU_MULT_DIV) != 0;
  result.is_simd = (opcat & OPCAT_SIMD) != 0;
  result.is_fp = (opcat & OPCAT_FP) != 0;
  result.is_fp_mult_div = (opcat & OPCAT_FP_MULT_DIV) != 0;

  result.is_load = !inst.read_addresses.empty();
  result.is_store = !inst.write_addresses.empty();

  // if the instruction was a load, store its read address
  if (result.is_load) {
    result.read_address = inst.read_addresses.front().addr;
  } else {
    result.read_address = 0;
  }

  result.is_isb = inst.inst_sync;

  // Determine branch type
  result.is_branch =
      (inst.category == "COND_BR" || inst.category == "UNCOND_BR" ||
       inst.category == "CALL" || inst.category == "RET");

  if (result.is_branch) {
    if (inst.branch_type.find("conditional") != std::string::npos) {
      result.branch_type = branch_t::DIRECT_COND;
    } else if (inst.branch_type.find("indirect") != std::string::npos) {
      result.branch_type = branch_t::INDIRECT;
    } else {
      result.branch_type = branch_t::DIRECT_UNCOND;
    }

    result.branch_taken = inst.branch_taken;
    result.branch_target_addr = inst.branch_target_addr;
  }

  return result;
}

std::vector<Instr> convert_trace(
    const std::vector<tracing::instr_trace_t>& instructions) {
  std::vector<Instr> result;
  result.reserve(instructions.size());

  // Map from IP to instruction ID for dependency resolution
  std::unordered_map<uint64_t, instr_id_t> ip_to_id;

  // First pass: convert instructions
  for (size_t i = 0; i < instructions.size(); ++i) {
    Instr instr = convert_to_instr(instructions[i], static_cast<instr_id_t>(i));
    ip_to_id[instr.IP] = instr.id;
    result.push_back(instr);
  }

  // Second pass: resolve dependencies
  for (size_t i = 0; i < instructions.size(); ++i) {
    const auto& trace_inst = instructions[i];
    auto& instr = result[i];

    // Add register dependencies
    for (uint64_t dep_ip : trace_inst.reg_dependent_ips) {
      auto it = ip_to_id.find(dep_ip);
      if (it != ip_to_id.end() && it->second < instr.id) {
        instr.deps.push_back(it->second);
      }
    }

    // Add memory dependencies
    for (uint64_t dep_ip : trace_inst.mem_dependent_ips) {
      auto it = ip_to_id.find(dep_ip);
      if (it != ip_to_id.end() && it->second < instr.id) {
        instr.deps.push_back(it->second);
      }
    }

    // Remove duplicates and sort
    std::sort(instr.deps.begin(), instr.deps.end());
    instr.deps.erase(std::unique(instr.deps.begin(), instr.deps.end()),
                     instr.deps.end());
  }

  return result;
}

std::vector<Instr> parse_and_convert(const std::string& csv_path) {
  auto trace = parse_csv(csv_path);
  return convert_trace(trace);
}

void stream_proto_region(
    const std::string& proto_path,
    const RegionConsumer& consume) {
  ZstdFrameReader reader(proto_path);

  std::vector<Instr> section;
  ProtoMessage::AnamolTraceChunk chunk;
  while (reader.readDelimited(chunk)) {
    const int records = chunk.ids_size();
    if (records <= 0 || chunk.ids_size() != records ||
        chunk.ips_size() != records || chunk.class_flags_size() != records ||
        chunk.branch_types_size() != records ||
        chunk.branch_taken_size() != records ||
        chunk.branch_target_addrs_size() != records ||
        (chunk.fixed_execution_latencies_size() != 0 &&
         chunk.fixed_execution_latencies_size() != records) ||
        chunk.read_sizes_size() != chunk.read_addresses_size() ||
        chunk.write_sizes_size() != chunk.write_addresses_size())
      throw std::runtime_error("invalid Anamol trace chunk lengths");
    validate_offsets(chunk.dep_offsets(), records, chunk.dep_ids_size());
    validate_offsets(chunk.read_offsets(), records, chunk.read_addresses_size());
    validate_offsets(chunk.write_offsets(), records, chunk.write_addresses_size());

    for (int index = 0; index < records; ++index) {
      Instr instr{};
      instr.id = static_cast<instr_id_t>(section.size());
      instr.IP = chunk.ips(index);
      instr.fetch_latency = 0;
      instr.exe_latency =
          chunk.fixed_execution_latencies_size()
              ? static_cast<latency_t>(chunk.fixed_execution_latencies(index))
              : 0;
      const uint32_t flags = chunk.class_flags(index);
      instr.is_alu = flags & 1U;
      instr.is_alu_mult_div = flags & 2U;
      instr.is_simd = flags & 4U;
      instr.is_fp = flags & 8U;
      instr.is_fp_mult_div = flags & 16U;
      instr.is_load = flags & 32U;
      instr.is_store = flags & 64U;
      instr.is_branch = flags & 128U;
      instr.is_isb = flags & 256U;
      instr.branch_type = branch_from_wire(chunk.branch_types(index));
      instr.branch_taken = chunk.branch_taken(index);
      instr.branch_target_addr = chunk.branch_target_addrs(index);
      for (uint32_t flat = chunk.dep_offsets(index);
           flat < chunk.dep_offsets(index + 1); ++flat)
        instr.deps.push_back(chunk.dep_ids(int(flat)));
      for (uint32_t flat = chunk.read_offsets(index);
           flat < chunk.read_offsets(index + 1); ++flat) {
        tracing::mem_access_t access{
            chunk.read_addresses(int(flat)),
            chunk.read_sizes(int(flat))};
        if (instr.read_address == 0)
          instr.read_address = access.addr;
      }
      for (uint32_t flat = chunk.write_offsets(index);
           flat < chunk.write_offsets(index + 1); ++flat) {
        (void)flat;
      }
      section.push_back(std::move(instr));
    }
    chunk.Clear();
  }
  if (section.empty())
    throw std::runtime_error("Anamol trace contains no instructions");
  consume(std::move(section));
}

}  // namespace analytical

#ifdef STANDALONE_PARSER

#include <iomanip>
#include <iostream>

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <csv_file>" << std::endl;
    return 1;
  }

  try {
    std::cout << "Parsing CSV file: " << argv[1] << std::endl;

    auto trace = analytical::parse_csv(argv[1]);
    auto instructions = analytical::convert_trace(trace);

    std::cout << "\nSuccessfully parsed " << instructions.size()
              << " instructions\n"
              << std::endl;

    // Print first 10 instructions as sample
    size_t print_count = std::min(size_t(10), instructions.size());
    std::cout << "First " << print_count << " instructions:\n" << std::endl;

    for (size_t i = 0; i < print_count; ++i) {
      const auto& instr = instructions[i];

      std::cout << "Instruction " << i << ":" << std::endl;
      std::cout << "  IP: 0x" << std::hex << instr.IP << std::dec << std::endl;
      std::cout << "  ID: " << instr.id << std::endl;
      std::cout << "  Latencies: exe=" << instr.exe_latency
                << " fetch=" << instr.fetch_latency << std::endl;
      std::cout << "  Type: ";
      if (instr.is_alu) std::cout << "ALU ";
      if (instr.is_fp) std::cout << "FP ";
      if (instr.is_load) std::cout << "LOAD ";
      if (instr.is_store) std::cout << "STORE ";
      if (instr.is_branch) std::cout << "BRANCH ";
      if (instr.is_isb) std::cout << "ISB ";
      std::cout << std::endl;

      if (instr.is_branch) {
        std::cout << "  Branch Type: ";
        switch (instr.branch_type) {
          case analytical::branch_t::DIRECT_COND:
            std::cout << "Direct Conditional";
            break;
          case analytical::branch_t::DIRECT_UNCOND:
            std::cout << "Direct Unconditional";
            break;
          case analytical::branch_t::INDIRECT:
            std::cout << "Indirect";
            break;
        }
        std::cout << std::endl;
      }

      std::cout << "  Dependencies (" << instr.deps.size() << "): ";
      for (size_t j = 0; j < instr.deps.size(); ++j) {
        if (j > 0) std::cout << ", ";
        std::cout << instr.deps[j];
      }
      std::cout << std::endl;
    }

    // Print statistics
    size_t alu_count = 0, fp_count = 0, load_count = 0, store_count = 0;
    size_t branch_count = 0, isb_count = 0;
    size_t total_deps = 0;

    for (const auto& instr : instructions) {
      if (instr.is_alu) alu_count++;
      if (instr.is_fp) fp_count++;
      if (instr.is_load) load_count++;
      if (instr.is_store) store_count++;
      if (instr.is_branch) branch_count++;
      if (instr.is_isb) isb_count++;
      total_deps += instr.deps.size();
    }

    std::cout << "\nStatistics:" << std::endl;
    std::cout << "  ALU instructions: " << alu_count << std::endl;
    std::cout << "  FP instructions: " << fp_count << std::endl;
    std::cout << "  Load instructions: " << load_count << std::endl;
    std::cout << "  Store instructions: " << store_count << std::endl;
    std::cout << "  Branch instructions: " << branch_count << std::endl;
    std::cout << "  ISB instructions: " << isb_count << std::endl;
    std::cout << "  Average dependencies per instruction: "
              << (instructions.empty()
                      ? 0.0
                      : double(total_deps) / instructions.size())
              << std::endl;

    return 0;

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}

#endif  // STANDALONE_PARSER
