#ifndef PARSER_H
#define PARSER_H

#pragma once

#include <string>
#include <vector>
#include <functional>
#include <limits>

#include "instr.h"

namespace ProtoMessage {
class AnamolTraceChunk;
}

namespace analytical {

using RegionConsumer = std::function<void(std::vector<Instr>&& region)>;
using InstructionConsumer = std::function<void(Instr&& instruction)>;
using ChunkConsumer = std::function<bool(const ProtoMessage::AnamolTraceChunk& chunk)>;

void stream_proto_chunks(
    const std::string& proto_path,
    const ChunkConsumer& consume);

void stream_proto_instructions(
    const std::string& proto_path,
    const InstructionConsumer& consume,
    size_t maximum_instructions = std::numeric_limits<size_t>::max());

void stream_proto_region(
    const std::string& proto_path,
    const RegionConsumer& consume);

}  // namespace analytical

#endif  // PARSER_H
