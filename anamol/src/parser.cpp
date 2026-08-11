#include "parser.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>

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

void validate_micro_ops(const ProtoMessage::AnamolTraceChunk& chunk, int records) {
  const int micro_ops = chunk.micro_op_ids_size();
  if (micro_ops <= 0 || chunk.micro_op_offsets_size() != records + 1 ||
      chunk.micro_op_class_flags_size() != micro_ops ||
      chunk.micro_op_fixed_execution_latencies_size() != micro_ops ||
      chunk.micro_op_read_sizes_size() != chunk.micro_op_read_addresses_size() ||
      chunk.micro_op_write_sizes_size() != chunk.micro_op_write_addresses_size())
    throw std::runtime_error("invalid Anamol micro-op trace chunk lengths");
  validate_offsets(chunk.micro_op_offsets(), records, micro_ops);
  validate_offsets(chunk.micro_op_dep_offsets(), micro_ops,
                   chunk.micro_op_dep_ids_size());
  validate_offsets(chunk.micro_op_read_offsets(), micro_ops,
                   chunk.micro_op_read_addresses_size());
  validate_offsets(chunk.micro_op_write_offsets(), micro_ops,
                   chunk.micro_op_write_addresses_size());
}

}  // namespace

void stream_proto_chunks(
    const std::string& proto_path,
    const ChunkConsumer& consume) {
  ZstdFrameReader reader(proto_path);
  ProtoMessage::AnamolTraceChunk chunk;
  bool observed = false;
  while (reader.readDelimited(chunk)) {
    const int records = chunk.ids_size();
    if (records <= 0 || chunk.ips_size() != records ||
        chunk.class_flags_size() != records ||
        chunk.branch_types_size() != records ||
        chunk.branch_taken_size() != records ||
        chunk.branch_target_addrs_size() != records)
      throw std::runtime_error("invalid Anamol trace chunk lengths");
    validate_offsets(chunk.dep_offsets(), records, chunk.dep_ids_size());
    validate_offsets(chunk.read_offsets(), records, chunk.read_addresses_size());
    validate_offsets(chunk.write_offsets(), records, chunk.write_addresses_size());
    validate_micro_ops(chunk, records);
    if (!consume(chunk)) return;
    observed = true;
    chunk.Clear();
  }
  if (!observed)
    throw std::runtime_error("Anamol trace contains no instructions");
}

void stream_proto_instructions(
    const std::string& proto_path,
    const InstructionConsumer& consume, size_t maximum_instructions) {
  size_t instruction_index = 0;
  instr_id_t previous_micro_id = 0;
  bool have_micro_id = false;
  stream_proto_chunks(proto_path, [&](const ProtoMessage::AnamolTraceChunk& chunk) {
    const int records = chunk.ids_size();
    for (int index = 0; index < records; ++index) {
      if (instruction_index == maximum_instructions) return false;
      Instr instr{};
      instr.id = static_cast<instr_id_t>(instruction_index++);
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
        if (instr.read_address == 0)
          instr.read_address = chunk.read_addresses(int(flat));
      }
      for (uint32_t flat = chunk.write_offsets(index);
           flat < chunk.write_offsets(index + 1); ++flat) {
        if (instr.write_address == 0)
          instr.write_address = chunk.write_addresses(int(flat));
      }
      for (uint32_t micro = chunk.micro_op_offsets(index);
           micro < chunk.micro_op_offsets(index + 1); ++micro) {
        MicroOp op{};
        op.id = chunk.micro_op_ids(int(micro));
        if (have_micro_id && op.id <= previous_micro_id)
          throw std::runtime_error("Anamol micro-op IDs must be globally increasing");
        op.class_flags = chunk.micro_op_class_flags(int(micro));
        const uint32_t latency = chunk.micro_op_fixed_execution_latencies(int(micro));
        if (latency > UINT16_MAX)
          throw std::runtime_error("Anamol micro-op latency exceeds supported range");
        op.exe_latency = static_cast<latency_t>(latency);
        for (uint32_t flat = chunk.micro_op_dep_offsets(int(micro));
             flat < chunk.micro_op_dep_offsets(int(micro + 1)); ++flat)
          op.deps.push_back(chunk.micro_op_dep_ids(int(flat)));
        for (const instr_id_t dependency : op.deps) {
          if (dependency >= op.id)
            throw std::runtime_error("Anamol micro-op dependency must be causal");
        }
        for (uint32_t flat = chunk.micro_op_read_offsets(int(micro));
             flat < chunk.micro_op_read_offsets(int(micro + 1)); ++flat)
          op.reads.push_back({chunk.micro_op_read_addresses(int(flat)),
                              chunk.micro_op_read_sizes(int(flat))});
        for (uint32_t flat = chunk.micro_op_write_offsets(int(micro));
             flat < chunk.micro_op_write_offsets(int(micro + 1)); ++flat)
          op.writes.push_back({chunk.micro_op_write_addresses(int(flat)),
                               chunk.micro_op_write_sizes(int(flat))});
        instr.micro_ops.push_back(std::move(op));
        previous_micro_id = chunk.micro_op_ids(int(micro));
        have_micro_id = true;
      }
      consume(std::move(instr));
    }
    return true;
  });
  if (instruction_index == 0)
    throw std::runtime_error("Anamol trace contains no instructions");
}

void stream_proto_region(
    const std::string& proto_path,
    const RegionConsumer& consume) {
  std::vector<Instr> section;
  stream_proto_instructions(proto_path, [&](Instr&& instruction) {
    section.push_back(std::move(instruction));
  });
  consume(std::move(section));
}

}  // namespace analytical
