#ifndef PARSER_H
#define PARSER_H

#pragma once

#include <string>
#include <vector>
#include <functional>

#include "instr.h"

namespace analytical {

// Step 1: CSV → instr_trace_t
std::vector<tracing::instr_trace_t> parse_csv(const std::string& csv_path);

// Step 2: instr_trace_t → Instr
Instr convert_to_instr(const tracing::instr_trace_t& inst, instr_id_t id);

// Helper: Convert entire trace
std::vector<Instr> convert_trace(
    const std::vector<tracing::instr_trace_t>& instructions);

// Complete pipeline: CSV → Instr
std::vector<Instr> parse_and_convert(const std::string& csv_path);

using RegionConsumer = std::function<void(std::vector<Instr>&& region)>;

void stream_proto_region(
    const std::string& proto_path,
    const RegionConsumer& consume);

}  // namespace analytical

#endif  // PARSER_H
