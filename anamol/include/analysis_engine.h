#ifndef ANALYSIS_ENGINE_H
#define ANALYSIS_ENGINE_H

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "instr.h"

namespace analytical {

struct MechanismBinding {
  std::string name;
  std::string model;
  std::vector<std::string> params;
};

using ConfigValues = std::map<std::string, double>;

size_t feature_count(const std::vector<MechanismBinding>& mechanisms);

std::vector<double> analyze_trace_windows(
    const std::vector<Instr>& instrs,
    int full_roi_window_size,
    int analysis_window_size,
    size_t window_count,
    const std::vector<ConfigValues>& configs,
    const std::vector<MechanismBinding>& mechanisms,
    int config_threads);

std::vector<double> analyze_trace_file_windows(
    const std::string& trace_path,
    int full_roi_window_size,
    int analysis_window_size,
    size_t window_count,
    const std::vector<ConfigValues>& configs,
    const std::vector<MechanismBinding>& mechanisms);

}  // namespace analytical

#endif  // ANALYSIS_ENGINE_H
