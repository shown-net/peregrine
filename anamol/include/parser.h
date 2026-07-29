#ifndef PARSER_H
#define PARSER_H

#pragma once

#include <string>
#include <vector>
#include <functional>

#include "instr.h"

namespace analytical {

using RegionConsumer = std::function<void(std::vector<Instr>&& region)>;

void stream_proto_region(
    const std::string& proto_path,
    const RegionConsumer& consume);

}  // namespace analytical

#endif  // PARSER_H
