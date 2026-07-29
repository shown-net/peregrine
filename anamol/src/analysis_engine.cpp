#include "analysis_engine.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

#include "models.h"

namespace analytical {
namespace {

constexpr size_t kFeaturesPerDistribution = 101;

struct Window {
  size_t start;
  size_t end;
};

struct WindowCounts {
  uint32_t int_alu = 0;
  uint32_t int_mult_div = 0;
  uint32_t fp = 0;
  uint32_t fp_mult_div = 0;
  uint32_t load = 0;
  uint32_t store = 0;
};

struct PreparedTrace {
  std::vector<Window> windows;
  std::vector<WindowCounts> counts;
};

struct LatencyOverlay {
  std::vector<latency_t> fetch;
  std::vector<latency_t> exe;
};

struct ActiveTrace {
  const std::vector<Instr>& instrs;
  const LatencyOverlay* latencies = nullptr;

  const Instr& instr(size_t index) const {
    return instrs[index];
  }

  latency_t fetch_latency(size_t index) const {
    return latencies ? latencies->fetch[index] : instrs[index].fetch_latency;
  }

  latency_t exe_latency(size_t index) const {
    return latencies ? latencies->exe[index] : instrs[index].exe_latency;
  }
};

uint16_t require_u16(const ConfigValues& config, const std::string& name) {
  const auto it = config.find(name);
  if (it == config.end())
    throw std::runtime_error("Anamol config is missing parameter: " + name);
  const double value = it->second;
  if (!std::isfinite(value) || value <= 0.0 ||
      value > static_cast<double>(UINT16_MAX) ||
      std::floor(value) != value)
    throw std::runtime_error("Anamol parameter must be a positive integer: " +
                             name);
  return static_cast<uint16_t>(value);
}

uint64_t require_u64(const ConfigValues& config, const std::string& name) {
  return static_cast<uint64_t>(require_u16(config, name));
}

struct CacheConfig {
  uint64_t line_bytes;
  uint64_t sets;
  uint64_t associativity;
  uint64_t latency;
};

class LruCache {
 public:
  explicit LruCache(CacheConfig config) : config_(config) {
    if (config_.line_bytes == 0 || config_.sets == 0 ||
        config_.associativity == 0 || config_.latency == 0)
      throw std::runtime_error("invalid Anamol cache configuration");
  }

  bool access(uint64_t address) {
    const uint64_t line = address / config_.line_bytes;
    const uint64_t set = line % config_.sets;
    auto& lines = sets_[set];
    auto found = std::find(lines.begin(), lines.end(), line);
    if (found != lines.end()) {
      std::rotate(lines.begin(), found, found + 1);
      return true;
    }
    lines.insert(lines.begin(), line);
    if (lines.size() > config_.associativity) {
      lines.pop_back();
    }
    return false;
  }

  uint64_t latency() const {
    return config_.latency;
  }

 private:
  CacheConfig config_;
  std::unordered_map<uint64_t, std::vector<uint64_t>> sets_;
};

struct CacheHierarchy {
  LruCache l1i;
  LruCache l1d;
  LruCache l2;
  uint64_t dram_latency;

  uint64_t access_instruction(uint64_t address) {
    return access_l1(address, l1i);
  }

  uint64_t access_data(uint64_t address) {
    return access_l1(address, l1d);
  }

 private:
  uint64_t access_l1(uint64_t address, LruCache& l1) {
    uint64_t latency = l1.latency();
    if (l1.access(address))
      return latency;
    latency += l2.latency();
    if (l2.access(address))
      return latency;
    return latency + dram_latency;
  }
};

CacheConfig cache_config(const ConfigValues& config, const std::string& size_key,
                         const std::string& assoc_key,
                         const std::string& latency_key) {
  const uint64_t line_bytes = require_u64(config, "line_bytes");
  const uint64_t size_kb = require_u64(config, size_key);
  const uint64_t associativity = require_u64(config, assoc_key);
  const uint64_t bytes = size_kb * 1024ULL;
  if (bytes < line_bytes * associativity)
    throw std::runtime_error("Anamol cache size is smaller than one set: " +
                             size_key);
  return CacheConfig{
      line_bytes,
      std::max<uint64_t>(1, bytes / (line_bytes * associativity)),
      associativity,
      require_u64(config, latency_key),
  };
}

LatencyOverlay annotate_cache_latencies(
    const std::vector<Instr>& instrs,
    const ConfigValues& config) {
  LatencyOverlay out;
  out.fetch.resize(instrs.size());
  out.exe.resize(instrs.size());
  CacheHierarchy caches{
      LruCache(cache_config(
          config, "l1i_size", "l1_associativity", "l1i_data_latency")),
      LruCache(cache_config(
          config, "l1d_size", "l1_associativity", "l1d_data_latency")),
      LruCache(cache_config(
          config, "l2_size", "l2_associativity", "l2_data_latency")),
      require_u64(config, "dram_latency_cycles"),
  };

  for (size_t index = 0; index < instrs.size(); ++index) {
    const auto& instr = instrs[index];
    out.fetch[index] = static_cast<latency_t>(
        std::min<uint64_t>(UINT16_MAX, caches.access_instruction(instr.IP)));
    uint64_t memory_latency = 0;
    if (instr.is_load && instr.read_address != 0) {
      memory_latency = caches.access_data(instr.read_address);
    } else if (instr.is_store && instr.write_address != 0) {
      memory_latency = caches.access_data(instr.write_address);
    }
    out.exe[index] = static_cast<latency_t>(
        std::min<uint64_t>(UINT16_MAX,
                           static_cast<uint64_t>(instr.exe_latency) +
                               memory_latency));
  }
  return out;
}

PreparedTrace prepare_trace(const std::vector<Instr>& instrs, int window_size) {
  PreparedTrace prepared;
  const size_t count = instrs.size();
  const size_t step = static_cast<size_t>(window_size);
  const size_t num_windows = (count + step - 1) / step;
  prepared.windows.reserve(num_windows);
  prepared.counts.reserve(num_windows);
  for (size_t start = 0; start < count; start += step) {
    const size_t end = std::min(start + step, count);
    WindowCounts counts;
    for (size_t index = start; index < end; ++index) {
      const auto& instr = instrs[index];
      counts.int_alu += instr.is_alu;
      counts.int_mult_div += instr.is_alu_mult_div;
      counts.fp += instr.is_fp;
      counts.fp_mult_div += instr.is_fp_mult_div;
      counts.load += instr.is_load;
      counts.store += instr.is_store;
    }
    prepared.windows.push_back({start, end});
    prepared.counts.push_back(counts);
  }
  return prepared;
}

double count_bound(size_t window_size, uint32_t count, uint16_t width) {
  if (count == 0)
    return static_cast<double>(window_size);
  double cycles_needed = static_cast<double>(count) / width;
  if (cycles_needed < 1.0)
    cycles_needed = 1.0;
  return static_cast<double>(window_size) / cycles_needed;
}

double width_bound(size_t window_size, uint16_t width) {
  (void)window_size;
  return static_cast<double>(width);
}

double load_store_ports_bound(size_t window_size, const WindowCounts& counts,
                              uint16_t rdwr, uint16_t read) {
  if (counts.load == 0 && counts.store == 0)
    return static_cast<double>(window_size);
  const double total_cycles =
      static_cast<double>(counts.load + counts.store) / (rdwr + read);
  const double lower_cycles =
      static_cast<double>(counts.load) / (rdwr + read) +
      static_cast<double>(counts.store) / rdwr;
  const double store_cycles = static_cast<double>(counts.store) / rdwr;
  const double loads_via_read = store_cycles * read;
  const double remaining_loads =
      std::max(0.0, static_cast<double>(counts.load) - loads_via_read);
  const double upper_cycles = store_cycles + remaining_loads / (rdwr + read);
  const double cycles = (total_cycles + lower_cycles + upper_cycles) / 3.0;
  return static_cast<double>(window_size) / cycles;
}

uint64_t resp_cycle_range(uint64_t req_cycle, const ActiveTrace& trace,
                          size_t instr_index,
                          std::unordered_map<uint64_t, uint64_t>& last_req,
                          std::unordered_map<uint64_t, uint64_t>& last_resp) {
  const auto& instr = trace.instr(instr_index);
  if (!instr.is_load) {
    return req_cycle + trace.exe_latency(instr_index);
  }
  if (instr.read_address == 0) {
    return req_cycle + trace.exe_latency(instr_index);
  }
  const uint64_t cache_line = instr.read_address / 64;
  const uint64_t previous = last_resp[cache_line];
  const uint64_t response =
      std::max(req_cycle + static_cast<uint64_t>(trace.exe_latency(instr_index)), previous);
  last_req[cache_line] = req_cycle;
  last_resp[cache_line] = response;
  return response;
}

double rob_range(const ActiveTrace& trace, const Window& window,
                 uint16_t rob_size) {
  const size_t count = window.end - window.start;
  std::vector<uint64_t> arrival(count);
  std::vector<uint64_t> finish(count);
  std::vector<uint64_t> commit(count);
  std::unordered_map<uint64_t, uint64_t> last_req;
  std::unordered_map<uint64_t, uint64_t> last_resp;
  const instr_id_t first = trace.instr(window.start).id;
  for (size_t offset = 0; offset < count; ++offset) {
    const size_t instr_index = window.start + offset;
    const auto& instr = trace.instr(instr_index);
    arrival[offset] = offset < rob_size ? 0 : commit[offset - rob_size];
    uint64_t start_cycle = arrival[offset];
    for (instr_id_t dep : instr.deps) {
      const int dep_index = static_cast<int>(dep - first);
      if (dep_index >= 0 && dep_index < static_cast<int>(offset)) {
        start_cycle = std::max(start_cycle, finish[dep_index]);
      }
    }
    finish[offset] = resp_cycle_range(start_cycle, trace, instr_index, last_req, last_resp);
    commit[offset] =
        offset == 0 ? finish[offset] : std::max(finish[offset], commit[offset - 1]);
  }
  return commit.back() == 0 ? static_cast<double>(count)
                            : static_cast<double>(count) / commit.back();
}

template <typename Predicate>
double queue_range(const ActiveTrace& trace, const Window& window,
                   uint16_t entries, Predicate predicate) {
  std::vector<size_t> items;
  items.reserve(window.end - window.start);
  for (size_t index = window.start; index < window.end; ++index) {
    if (predicate(trace.instr(index))) {
      items.push_back(index);
    }
  }
  if (items.empty()) {
    return static_cast<double>(window.end - window.start);
  }
  std::vector<uint64_t> arrival(items.size());
  std::vector<uint64_t> finish(items.size());
  std::vector<uint64_t> commit(items.size());
  std::unordered_map<uint64_t, uint64_t> last_req;
  std::unordered_map<uint64_t, uint64_t> last_resp;
  for (size_t index = 0; index < items.size(); ++index) {
    arrival[index] = index < entries ? 0 : commit[index - entries];
    finish[index] = resp_cycle_range(arrival[index], trace, items[index], last_req, last_resp);
    commit[index] =
        index == 0 ? finish[index] : std::max(finish[index], commit[index - 1]);
  }
  return commit.back() == 0
             ? static_cast<double>(window.end - window.start)
             : static_cast<double>(window.end - window.start) / commit.back();
}

double icache_range(const ActiveTrace& trace, const Window& window,
                    uint16_t max_icache_fills) {
  std::unordered_map<uint64_t, uint64_t> in_flight;
  uint64_t previous_ready = 0;
  for (size_t index = window.start; index < window.end; ++index) {
    const auto& instr = trace.instr(index);
    const uint64_t cache_line = instr.IP / 64;
    const auto found = in_flight.find(cache_line);
    if (found != in_flight.end()) {
      previous_ready = std::max(previous_ready, found->second);
      continue;
    }
    uint64_t next_available = previous_ready;
    while (in_flight.size() >= max_icache_fills) {
      auto earliest = std::min_element(
          in_flight.begin(), in_flight.end(),
          [](const auto& lhs, const auto& rhs) { return lhs.second < rhs.second; });
      next_available = earliest->second;
      in_flight.erase(earliest);
    }
    const uint64_t start = std::max(previous_ready, next_available);
    const uint64_t done = start + trace.fetch_latency(index);
    in_flight[cache_line] = done;
    previous_ready = done;
  }
  const size_t count = window.end - window.start;
  return previous_ready == 0 ? static_cast<double>(count)
                             : static_cast<double>(count) / previous_ready;
}

using ModelEvaluator = double (*)(
    const PreparedTrace& prepared,
    const ActiveTrace& active_trace,
    const Window& window,
    size_t window_index,
    const ConfigValues& config,
    const MechanismBinding& binding);

struct ModelSpec {
  const char* name;
  size_t param_count;
  bool requires_cache_annotation;
  ModelEvaluator evaluate;
};

double eval_issue_width(const PreparedTrace& prepared,
                        const ActiveTrace& active_trace,
                        const Window& window, size_t window_index,
                        const ConfigValues& config,
                        const MechanismBinding& binding) {
  (void)active_trace;
  const auto& param = binding.params[0];
  const uint16_t width = require_u16(config, param);
  const auto& counts = prepared.counts[window_index];
  const size_t window_size = window.end - window.start;
  if (param == "int_reg_issue_width")
    return count_bound(window_size, counts.int_alu, width);
  if (param == "int_mult_div_issue_width")
    return count_bound(window_size, counts.int_mult_div, width);
  if (param == "fp_reg_issue_width")
    return count_bound(window_size, counts.fp, width);
  if (param == "fp_mult_div_issue_width")
    return count_bound(window_size, counts.fp_mult_div, width);
  throw std::runtime_error("unknown issue-width Anamol parameter: " + param);
}

double eval_load_store_ports(const PreparedTrace& prepared,
                             const ActiveTrace& active_trace,
                             const Window& window, size_t window_index,
                             const ConfigValues& config,
                             const MechanismBinding& binding) {
  (void)active_trace;
  return load_store_ports_bound(
      window.end - window.start,
      prepared.counts[window_index],
      require_u16(config, binding.params[0]),
      require_u16(config, binding.params[1]));
}

double eval_width(const PreparedTrace& prepared,
                  const ActiveTrace& active_trace, const Window& window,
                  size_t window_index, const ConfigValues& config,
                  const MechanismBinding& binding) {
  (void)prepared;
  (void)active_trace;
  (void)window_index;
  return width_bound(window.end - window.start,
                     require_u16(config, binding.params[0]));
}

double eval_rob(const PreparedTrace& prepared,
                const ActiveTrace& active_trace, const Window& window,
                size_t window_index, const ConfigValues& config,
                const MechanismBinding& binding) {
  (void)prepared;
  (void)window_index;
  return rob_range(active_trace, window, require_u16(config, binding.params[0]));
}

double eval_load_queue(const PreparedTrace& prepared,
                       const ActiveTrace& active_trace,
                       const Window& window, size_t window_index,
                       const ConfigValues& config,
                       const MechanismBinding& binding) {
  (void)prepared;
  (void)window_index;
  return queue_range(active_trace, window, require_u16(config, binding.params[0]),
                     [](const Instr& instr) { return instr.is_load; });
}

double eval_store_queue(const PreparedTrace& prepared,
                        const ActiveTrace& active_trace,
                        const Window& window, size_t window_index,
                        const ConfigValues& config,
                        const MechanismBinding& binding) {
  (void)prepared;
  (void)window_index;
  return queue_range(active_trace, window, require_u16(config, binding.params[0]),
                     [](const Instr& instr) { return instr.is_store; });
}

double eval_icache_fills(const PreparedTrace& prepared,
                         const ActiveTrace& active_trace,
                         const Window& window, size_t window_index,
                         const ConfigValues& config,
                         const MechanismBinding& binding) {
  (void)prepared;
  (void)window_index;
  return icache_range(active_trace, window, require_u16(config, binding.params[0]));
}

const std::vector<ModelSpec>& model_specs() {
  static const std::vector<ModelSpec> specs = {
      {"rob_capacity_latency_bound", 1, true, eval_rob},
      {"load_queue_capacity_latency_bound", 1, true, eval_load_queue},
      {"store_queue_capacity_latency_bound", 1, true, eval_store_queue},
      {"issue_width_count_bound", 1, false, eval_issue_width},
      {"load_store_port_combined_bound", 2, false, eval_load_store_ports},
      {"width_bound", 1, false, eval_width},
      {"icache_fill_slots_bound", 1, true, eval_icache_fills},
  };
  return specs;
}

const ModelSpec& model_spec(const MechanismBinding& binding) {
  const auto& specs = model_specs();
  const auto found = std::find_if(
      specs.begin(), specs.end(),
      [&](const ModelSpec& spec) { return binding.model == spec.name; });
  if (found == specs.end())
    throw std::runtime_error("unknown Anamol mechanism model: " + binding.model);
  if (binding.params.size() != found->param_count)
    throw std::runtime_error(binding.model + " requires " +
                             std::to_string(found->param_count) +
                             " parameter(s): " + binding.name);
  return *found;
}

std::vector<double> component_samples(
    const PreparedTrace& prepared,
    const ActiveTrace& active_trace,
    const MechanismBinding& binding,
    const ConfigValues& config) {
  const ModelSpec& spec = model_spec(binding);
  std::vector<double> samples;
  samples.reserve(prepared.windows.size());
  for (size_t index = 0; index < prepared.windows.size(); ++index) {
    const auto& window = prepared.windows[index];
    samples.push_back(
        spec.evaluate(prepared, active_trace, window, index, config, binding));
  }
  return samples;
}

std::vector<double> distribution_features(std::vector<double> values) {
  if (values.empty()) values.push_back(0.0);
  std::sort(values.begin(), values.end());
  auto percentile = [&](double point) {
    const double rank = (values.size() - 1) * point;
    const size_t lower = static_cast<size_t>(std::floor(rank));
    const size_t upper = static_cast<size_t>(std::ceil(rank));
    return values[lower] + (rank - lower) * (values[upper] - values[lower]);
  };

  std::vector<double> output;
  output.reserve(kFeaturesPerDistribution);
  for (size_t index = 0; index < 50; ++index) {
    const double point =
        (static_cast<double>(index) * 98.0 / 49.0 + 1.0) / 100.0;
    output.push_back(percentile(point));
  }

  double total_weight = 0.0;
  for (double value : values) total_weight += std::max(0.0, value);
  if (total_weight == 0.0) {
    output.insert(output.end(), output.begin(), output.begin() + 50);
  } else {
    std::vector<double> cumulative;
    std::vector<double> weighted_values;
    double running = 0.0;
    for (double value : values) {
      const double weight = std::max(0.0, value);
      if (weight == 0.0) continue;
      running += weight / total_weight;
      cumulative.push_back(running);
      weighted_values.push_back(value);
    }
    for (size_t index = 0; index < 50; ++index) {
      const double target =
          (static_cast<double>(index) * 98.0 / 49.0 + 1.0) / 100.0;
      const auto upper =
          std::lower_bound(cumulative.begin(), cumulative.end(), target);
      if (upper == cumulative.begin()) {
        output.push_back(weighted_values.front());
      } else if (upper == cumulative.end()) {
        output.push_back(weighted_values.back());
      } else {
        const size_t right = static_cast<size_t>(upper - cumulative.begin());
        const size_t left = right - 1;
        const double span = cumulative[right] - cumulative[left];
        const double alpha =
            span == 0.0 ? 0.0 : (target - cumulative[left]) / span;
        output.push_back(weighted_values[left] +
                         alpha * (weighted_values[right] -
                                  weighted_values[left]));
      }
    }
  }

  const double sum = std::accumulate(values.begin(), values.end(), 0.0);
  output.push_back(sum / static_cast<double>(values.size()));
  return output;
}

void validate_binding_params(const MechanismBinding& binding,
                             const ConfigValues& config) {
  (void)model_spec(binding);
  for (const auto& param : binding.params) {
    (void)require_u16(config, param);
  }
}

}  // namespace

size_t feature_count(const std::vector<MechanismBinding>& mechanisms) {
  return mechanisms.size() * kFeaturesPerDistribution;
}

std::vector<double> analyze_trace(
    const std::vector<Instr>& instrs,
    int window_size,
    const std::vector<ConfigValues>& configs,
    const std::vector<MechanismBinding>& mechanisms) {
  if (instrs.empty())
    throw std::runtime_error("Anamol trace contains no instructions");
  if (window_size <= 0)
    throw std::runtime_error("Anamol window size must be positive");
  if (configs.empty())
    throw std::runtime_error("Anamol configs must not be empty");
  if (mechanisms.empty())
    throw std::runtime_error("Anamol mechanisms must not be empty");

  bool annotate = false;
  for (const auto& binding : mechanisms) {
    annotate = annotate || model_spec(binding).requires_cache_annotation;
  }
  const PreparedTrace prepared = prepare_trace(instrs, window_size);
  std::vector<double> matrix;
  matrix.reserve(configs.size() * feature_count(mechanisms));

  for (const auto& config : configs) {
    LatencyOverlay overlay;
    const LatencyOverlay* active_overlay = nullptr;
    if (annotate) {
      overlay = annotate_cache_latencies(instrs, config);
      active_overlay = &overlay;
    }
    const ActiveTrace active_trace{instrs, active_overlay};
    for (size_t mechanism_index = 0; mechanism_index < mechanisms.size();
         ++mechanism_index) {
      validate_binding_params(mechanisms[mechanism_index], config);
      auto features = distribution_features(component_samples(
          prepared, active_trace, mechanisms[mechanism_index], config));
      matrix.insert(matrix.end(), features.begin(), features.end());
    }
  }
  return matrix;
}

}  // namespace analytical
