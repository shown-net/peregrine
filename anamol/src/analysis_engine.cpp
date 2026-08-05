#include "analysis_engine.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace analytical {
namespace {

constexpr size_t kFeaturesPerDistribution = 101;

uint16_t require_u16(const ConfigValues& config, const std::string& name) {
  const auto found = config.find(name);
  if (found == config.end() || !std::isfinite(found->second) ||
      found->second <= 0.0 || found->second > UINT16_MAX ||
      std::floor(found->second) != found->second)
    throw std::runtime_error("Anamol parameter must be a positive integer: " + name);
  return static_cast<uint16_t>(found->second);
}

uint64_t require_u64(const ConfigValues& config, const std::string& name) {
  return require_u16(config, name);
}

struct CacheConfig { uint64_t line_bytes, sets, associativity, latency; };

class LruCache {
 public:
  explicit LruCache(CacheConfig config) : config_(config) {}
  bool access(uint64_t line) {
    const uint64_t set = line % config_.sets;
    auto& lines = sets_[set];
    const auto found = std::find(lines.begin(), lines.end(), line);
    if (found != lines.end()) {
      std::rotate(lines.begin(), found, found + 1);
      return true;
    }
    lines.insert(lines.begin(), line);
    if (lines.size() > config_.associativity) lines.pop_back();
    return false;
  }
  uint64_t latency() const { return config_.latency; }
 private:
  CacheConfig config_;
  std::unordered_map<uint64_t, std::vector<uint64_t>> sets_;
};

CacheConfig cache_config(const ConfigValues& config, const std::string& size,
                         const std::string& assoc, const std::string& latency) {
  const uint64_t line_bytes = require_u64(config, "line_bytes");
  const uint64_t associativity = require_u64(config, assoc);
  const uint64_t bytes = require_u64(config, size) * 1024ULL;
  if (bytes < line_bytes * associativity)
    throw std::runtime_error("Anamol cache size is smaller than one set: " + size);
  return {line_bytes, std::max<uint64_t>(1, bytes / (line_bytes * associativity)),
          associativity, require_u64(config, latency)};
}

struct AccessResult { uint64_t latency; bool l1_miss; bool l2_miss; uint64_t line; };

class CacheHierarchy {
 public:
  explicit CacheHierarchy(const ConfigValues& config)
      : line_bytes_(require_u64(config, "line_bytes")),
        l1i_(cache_config(config, "l1i_size", "l1_associativity", "l1i_data_latency")),
        l1d_(cache_config(config, "l1d_size", "l1_associativity", "l1d_data_latency")),
        l2_(cache_config(config, "l2_size", "l2_associativity", "l2_data_latency")),
        dram_(require_u64(config, "dram_latency_cycles")) {}
  AccessResult instruction(uint64_t address) { return access(address, l1i_); }
  AccessResult data(uint64_t address) { return access(address, l1d_); }
  uint64_t line_bytes() const { return line_bytes_; }
 private:
  AccessResult access(uint64_t address, LruCache& l1) {
    const uint64_t line = address / line_bytes_;
    const uint64_t latency = l1.latency();
    if (l1.access(line)) return {latency, false, false, line};
    uint64_t total = latency + l2_.latency();
    const bool l2_miss = !l2_.access(line);
    if (l2_miss) total += dram_;
    return {total, true, l2_miss, line};
  }
  uint64_t line_bytes_;
  LruCache l1i_, l1d_, l2_;
  uint64_t dram_;
};

std::vector<double> distribution_features(std::vector<double> values) {
  if (values.empty()) values.push_back(0.0);
  std::sort(values.begin(), values.end());
  const auto percentile = [&](double point) {
    const double rank = (values.size() - 1) * point;
    const size_t lower = static_cast<size_t>(std::floor(rank));
    const size_t upper = static_cast<size_t>(std::ceil(rank));
    return values[lower] + (rank - lower) * (values[upper] - values[lower]);
  };
  std::vector<double> out;
  out.reserve(kFeaturesPerDistribution);
  for (size_t index = 0; index < 50; ++index)
    out.push_back(percentile((static_cast<double>(index) * 98.0 / 49.0 + 1.0) / 100.0));
  double weight = 0.0;
  for (double value : values) weight += std::max(0.0, value);
  for (size_t index = 0; index < 50; ++index) {
    const double point = (static_cast<double>(index) * 98.0 / 49.0 + 1.0) / 100.0;
    if (weight == 0.0) { out.push_back(percentile(point)); continue; }
    const double target = point * weight;
    double cumulative = 0.0;
    size_t selected = values.size() - 1;
    for (size_t value = 0; value < values.size(); ++value) {
      cumulative += std::max(0.0, values[value]);
      if (cumulative >= target) { selected = value; break; }
    }
    out.push_back(values[selected]);
  }
  out.push_back(std::accumulate(values.begin(), values.end(), 0.0) / values.size());
  return out;
}

struct WindowCounts { uint64_t int_alu=0, int_mult_div=0, fp=0, fp_mult_div=0, load=0, store=0, micro_ops=0; };

class ProgressState {
 public:
  double readout(size_t macro_count) {
    const uint64_t delta = progress_ - observed_;
    observed_ = progress_;
    return delta == 0 ? static_cast<double>(macro_count)
                      : static_cast<double>(macro_count) / delta;
  }
 protected:
  uint64_t progress_ = 0, observed_ = 0;
};

class QueueState : public ProgressState {
 public:
  explicit QueueState(uint16_t entries) : commits_(entries, 0) {}
  void process(const MicroOp& op, uint64_t latency, const std::vector<std::pair<uint64_t, uint64_t>>& lines,
               bool dependencies) {
    const uint64_t arrival = count_ < commits_.size() ? 0 : commits_[count_ % commits_.size()];
    uint64_t start = arrival;
    if (dependencies) {
      for (const instr_id_t dependency : op.deps) {
        const auto found = finishes_.find(dependency);
        if (found != finishes_.end()) start = std::max(start, found->second);
      }
    }
    uint64_t finish = start + latency;
    for (const auto& [line, line_latency] : lines) {
      const uint64_t response = std::max(start + line_latency, responses_[line]);
      responses_[line] = response;
      finish = std::max(finish, response);
    }
    const uint64_t commit = std::max(finish, previous_commit_);
    commits_[count_ % commits_.size()] = commit;
    previous_commit_ = commit;
    ++count_;
    progress_ = commit;
    if (dependencies) finishes_[op.id] = finish;
    retire(arrival);
  }
 private:
  void retire(uint64_t frontier) {
    for (auto it = finishes_.begin(); it != finishes_.end();) {
      if (it->second <= frontier) it = finishes_.erase(it); else ++it;
    }
    for (auto it = responses_.begin(); it != responses_.end();) {
      if (it->second <= frontier) it = responses_.erase(it); else ++it;
    }
  }
  std::vector<uint64_t> commits_;
  uint64_t count_ = 0, previous_commit_ = 0;
  std::unordered_map<instr_id_t, uint64_t> finishes_;
  std::unordered_map<uint64_t, uint64_t> responses_;
};

class IcacheState : public ProgressState {
 public:
  explicit IcacheState(uint16_t slots) : slots_(slots) {}
  void process(const AccessResult& access) {
    retire();
    if (!access.l1_miss) return;
    const auto existing = in_flight_.find(access.line);
    if (existing != in_flight_.end()) { progress_ = std::max(progress_, existing->second); return; }
    while (in_flight_.size() >= slots_) {
      const auto earliest = std::min_element(in_flight_.begin(), in_flight_.end(),
          [](const auto& left, const auto& right) { return left.second < right.second; });
      request_clock_ = std::max(request_clock_, earliest->second);
      in_flight_.erase(earliest);
      retire();
    }
    const uint64_t done = request_clock_ + access.latency;
    in_flight_[access.line] = done;
    progress_ = std::max(progress_, done);
  }
 private:
  void retire() {
    for (auto it = in_flight_.begin(); it != in_flight_.end();) {
      if (it->second <= request_clock_) it = in_flight_.erase(it); else ++it;
    }
  }
  uint16_t slots_;
  uint64_t request_clock_ = 0;
  std::unordered_map<uint64_t, uint64_t> in_flight_;
};

class WidthState : public ProgressState {
 public:
  explicit WidthState(uint16_t width) : width_(width) {}
  void add(uint64_t micro_ops) { total_ += micro_ops; progress_ = (total_ + width_ - 1) / width_; }
 private:
  uint16_t width_; uint64_t total_ = 0;
};

class LoadMissPressureState {
 public:
  explicit LoadMissPressureState(bool l2) : l2_(l2) {}
  void observe(const AccessResult& access) { misses_ += l2_ ? access.l2_miss : access.l1_miss; }
  void finish_macro() { ++macros_; }
  double readout() {
    const double value = macros_ == 0 ? 0.0 : 1000.0 * static_cast<double>(misses_) / macros_;
    misses_ = 0;
    macros_ = 0;
    return value;
  }
 private:
  bool l2_;
  uint64_t misses_ = 0;
  uint64_t macros_ = 0;
};

double count_bound(size_t macros, uint64_t count, uint16_t width) {
  if (count == 0) return static_cast<double>(macros);
  return static_cast<double>(macros) / std::max(1.0, static_cast<double>(count) / width);
}

double port_bound(size_t macros, const WindowCounts& counts, uint16_t rdwr, uint16_t read, bool lower) {
  if (counts.load == 0 && counts.store == 0) return static_cast<double>(macros);
  const double store_cycles = static_cast<double>(counts.store) / rdwr;
  const double cycles = lower
      ? static_cast<double>(counts.load) / (rdwr + read) + store_cycles
      : store_cycles + std::max(0.0, static_cast<double>(counts.load) - store_cycles * read) / (rdwr + read);
  return static_cast<double>(macros) / std::max(1.0, cycles);
}

enum class Model { ROB, LQ, SQ, ISSUE, PORT_LOWER, PORT_UPPER, WIDTH, ICACHE, L1D_MISS, L2_MISS };
struct Binding { Model model; std::string parameter; std::string second_parameter; };

Binding parse_binding(const MechanismBinding& binding) {
  if (binding.model == "rob_capacity_latency_bound") return {Model::ROB, binding.params.at(0), {}};
  if (binding.model == "load_queue_capacity_latency_bound") return {Model::LQ, binding.params.at(0), {}};
  if (binding.model == "store_queue_capacity_latency_bound") return {Model::SQ, binding.params.at(0), {}};
  if (binding.model == "issue_width_count_bound") return {Model::ISSUE, binding.params.at(0), {}};
  if (binding.model == "load_store_port_lower_bound") return {Model::PORT_LOWER, binding.params.at(0), binding.params.at(1)};
  if (binding.model == "load_store_port_upper_bound") return {Model::PORT_UPPER, binding.params.at(0), binding.params.at(1)};
  if (binding.model == "micro_op_width_bound") return {Model::WIDTH, binding.params.at(0), {}};
  if (binding.model == "icache_fill_slots_bound") return {Model::ICACHE, binding.params.at(0), {}};
  if (binding.model == "l1d_load_miss_pressure") return {Model::L1D_MISS, {}, {}};
  if (binding.model == "l2_load_miss_pressure") return {Model::L2_MISS, {}, {}};
  throw std::runtime_error("unknown Anamol mechanism model: " + binding.model);
}

class ConfigAnalyzer {
 public:
  ConfigAnalyzer(const ConfigValues& config, const std::vector<MechanismBinding>& mechanisms)
      : config_(config), caches_(config), rob_(require_u16(config, parameter(mechanisms, Model::ROB))),
        lq_(require_u16(config, parameter(mechanisms, Model::LQ))),
        sq_(require_u16(config, parameter(mechanisms, Model::SQ))),
        icache_(require_u16(config, parameter(mechanisms, Model::ICACHE))),
        l1d_miss_(false), l2_miss_(true),
        decode_(require_u16(config, parameter(mechanisms, Model::WIDTH, "decode_width"))),
        rename_(require_u16(config, parameter(mechanisms, Model::WIDTH, "rename_width"))),
        commit_(require_u16(config, parameter(mechanisms, Model::WIDTH, "commit_width"))) {}
  void process(const Instr& macro) {
    icache_.process(caches_.instruction(macro.IP));
    for (const auto& op : macro.micro_ops) {
      std::vector<std::pair<uint64_t, uint64_t>> lines;
      uint64_t memory_latency = 0;
      const auto access = [&](const MemoryAccess& memory, bool load_read) {
        if (memory.size == 0) return;
        const uint64_t first = memory.address / caches_.line_bytes();
        const uint64_t last = (memory.address + memory.size - 1) / caches_.line_bytes();
        for (uint64_t line = first; line <= last; ++line) {
          const auto result = caches_.data(line * caches_.line_bytes());
          lines.push_back({result.line, result.latency});
          memory_latency = std::max(memory_latency, result.latency);
          if (load_read) {
            l1d_miss_.observe(result);
            l2_miss_.observe(result);
          }
        }
      };
      for (const auto& memory : op.reads) access(memory, op.is_load());
      for (const auto& memory : op.writes) access(memory, false);
      const uint64_t latency = op.exe_latency + memory_latency;
      rob_.process(op, latency, lines, true);
      if (op.is_load()) lq_.process(op, latency, lines, false);
      if (op.is_store()) sq_.process(op, latency, lines, false);
      counts_.int_alu += op.is_alu(); counts_.int_mult_div += op.is_alu_mult_div();
      counts_.fp += op.is_fp(); counts_.fp_mult_div += op.is_fp_mult_div();
      counts_.load += op.is_load(); counts_.store += op.is_store(); ++counts_.micro_ops;
    }
    l1d_miss_.finish_macro();
    l2_miss_.finish_macro();
    decode_.add(macro.micro_ops.size()); rename_.add(macro.micro_ops.size()); commit_.add(macro.micro_ops.size());
    ++macros_;
  }
  std::vector<double> snapshot(const std::vector<MechanismBinding>& mechanisms) {
    std::vector<double> values;
    values.reserve(mechanisms.size());
    for (const auto& mechanism : mechanisms) {
      const Binding binding = parse_binding(mechanism);
      const uint16_t width = binding.parameter.empty() ? 0 : require_u16(config_, binding.parameter);
      switch (binding.model) {
        case Model::ROB: values.push_back(rob_.readout(macros_)); break;
        case Model::LQ: values.push_back(lq_.readout(macros_)); break;
        case Model::SQ: values.push_back(sq_.readout(macros_)); break;
        case Model::ICACHE: values.push_back(icache_.readout(macros_)); break;
        case Model::L1D_MISS: values.push_back(l1d_miss_.readout()); break;
        case Model::L2_MISS: values.push_back(l2_miss_.readout()); break;
        case Model::WIDTH:
          if (binding.parameter == "decode_width") values.push_back(decode_.readout(macros_));
          else if (binding.parameter == "rename_width") values.push_back(rename_.readout(macros_));
          else values.push_back(commit_.readout(macros_));
          break;
        case Model::ISSUE:
          values.push_back(binding.parameter == "int_reg_issue_width" ? count_bound(macros_, counts_.int_alu, width) :
              binding.parameter == "int_mult_div_issue_width" ? count_bound(macros_, counts_.int_mult_div, width) :
              binding.parameter == "fp_reg_issue_width" ? count_bound(macros_, counts_.fp, width) :
              count_bound(macros_, counts_.fp_mult_div, width));
          break;
        case Model::PORT_LOWER: values.push_back(port_bound(macros_, counts_, width, require_u16(config_, binding.second_parameter), true)); break;
        case Model::PORT_UPPER: values.push_back(port_bound(macros_, counts_, width, require_u16(config_, binding.second_parameter), false)); break;
      }
    }
    macros_ = 0; counts_ = {};
    return values;
  }
 private:
  static std::string parameter(const std::vector<MechanismBinding>& mechanisms, Model wanted, const std::string& fallback = {}) {
    for (const auto& mechanism : mechanisms) {
      const Binding binding = parse_binding(mechanism);
      if (binding.model == wanted && (fallback.empty() || binding.parameter == fallback)) return binding.parameter;
    }
    throw std::runtime_error("Anamol canonical mechanism is missing");
  }
  const ConfigValues& config_; CacheHierarchy caches_; QueueState rob_, lq_, sq_; IcacheState icache_; LoadMissPressureState l1d_miss_, l2_miss_; WidthState decode_, rename_, commit_; WindowCounts counts_; size_t macros_ = 0;
};

}  // namespace

size_t feature_count(const std::vector<MechanismBinding>& mechanisms) { return mechanisms.size() * kFeaturesPerDistribution; }

std::vector<double> analyze_trace_windows(const std::vector<Instr>& instrs, int full_roi_window_size,
                                          int analysis_window_size, size_t requested_window_count,
                                          const std::vector<ConfigValues>& configs,
                                          const std::vector<MechanismBinding>& mechanisms) {
  if (full_roi_window_size <= 0 || analysis_window_size <= 0 || configs.empty() || mechanisms.empty())
    throw std::runtime_error("invalid Anamol full-ROI analysis arguments");
  const size_t full = static_cast<size_t>(full_roi_window_size), analysis = static_cast<size_t>(analysis_window_size);
  if (full % analysis != 0 || requested_window_count < 2 || requested_window_count > instrs.size() / full)
    throw std::runtime_error("Anamol requires at least two complete, evenly divisible full-ROI windows");
  const size_t feature_columns = feature_count(mechanisms);
  std::vector<double> output(configs.size() * (requested_window_count - 1) * feature_columns);
  for (size_t config_index = 0; config_index < configs.size(); ++config_index) {
    ConfigAnalyzer analyzer(configs[config_index], mechanisms);
    std::vector<std::vector<double>> samples(mechanisms.size());
    size_t write_window = 0;
    for (size_t index = 0; index < requested_window_count * full; ++index) {
      analyzer.process(instrs[index]);
      if ((index + 1) % analysis == 0) {
        const auto snapshot = analyzer.snapshot(mechanisms);
        for (size_t mechanism = 0; mechanism < mechanisms.size(); ++mechanism) samples[mechanism].push_back(snapshot[mechanism]);
      }
      if ((index + 1) % full == 0) {
        const size_t raw_window = (index + 1) / full - 1;
        if (raw_window != 0) {
          size_t offset = (config_index * (requested_window_count - 1) + write_window++) * feature_columns;
          for (auto& component : samples) {
            const auto features = distribution_features(std::move(component));
            std::copy(features.begin(), features.end(), output.begin() + offset);
            offset += features.size();
          }
        }
        samples.assign(mechanisms.size(), {});
      }
    }
  }
  return output;
}

}  // namespace analytical
