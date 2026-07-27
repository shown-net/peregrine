#include <getopt.h>

#include <algorithm>
#include <bit>
#include <cstdlib>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "instr.h"
#include "models.h"
#include "npy_reader.h"
#include "parser.h"
#include "resource_registry.h"

static void print_usage(const char* prog) {
  std::cout << "Usage: " << prog
            << " [-w WINDOW_SIZE|--window WINDOW_SIZE] "
               "[-t TRACE|--tracefile TRACE] "
               "[-o OUTPUT_DIR|--output-dir OUTPUT_DIR] "
               "[-l LATENCIES_NPY|--latencies-npy LATENCIES_NPY] "
               "[-c CONFIG_JSON|--config-json CONFIG_JSON]\n"
               "Defaults: WINDOW_SIZE=400, TRACE=trace.csv\n"
               "          OUTPUT_DIR=output/<trace_stem>\n"
               "\n"
               "Without --latencies-npy: single run using latencies from TRACE.\n"
               "With    --latencies-npy: runs latency-independent resources once,\n"
               "  then iterates over every cache config in the .npy, writing\n"
               "  latency-dependent results to OUTPUT_DIR/config_NNNN/.\n"
               "\n"
               "Without --config-json: sweeps all param combinations (default).\n"
               "With    --config-json: computes only for the specified config.\n"
               "  JSON format: {\"rob_size\": 128, \"load_queue_size\": 64, ...}\n";
}

// Minimal JSON parser for a flat {string: integer} object.
static std::map<std::string, uint16_t> parse_config_json(
    const std::string& json) {
  std::map<std::string, uint16_t> config;
  std::string s = json;

  // Strip whitespace
  s.erase(std::remove_if(s.begin(), s.end(), ::isspace), s.end());

  if (s.empty() || s[0] != '{' || s.back() != '}')
    throw std::runtime_error("config-json must be a JSON object {...}");
  s = s.substr(1, s.size() - 2);

  if (s.empty()) return config;

  std::istringstream iss(s);
  std::string token;
  while (std::getline(iss, token, ',')) {
    if (token.empty()) continue;
    auto colon = token.find(':');
    if (colon == std::string::npos)
      throw std::runtime_error("Malformed JSON pair (no ':'): " + token);

    std::string key = token.substr(0, colon);
    std::string val_str = token.substr(colon + 1);

    // Remove surrounding quotes from key
    auto q1 = key.find('"');
    auto q2 = key.rfind('"');
    if (q1 != std::string::npos && q2 != q1)
      key = key.substr(q1 + 1, q2 - q1 - 1);

    config[key] = static_cast<uint16_t>(std::stoi(val_str));
  }
  return config;
}

static std::vector<std::map<std::string, uint16_t>> read_configs_stdin() {
  std::vector<std::map<std::string, uint16_t>> configs;
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
    configs.push_back(parse_config_json(line));
  }
  if (configs.empty())
    throw std::runtime_error("stdin contains no configurations");
  return configs;
}

static std::vector<double> cdf_summary(std::vector<double> values) {
  if (values.empty()) values.push_back(0.0);
  std::sort(values.begin(), values.end());
  auto percentile = [&](double point) {
    const double rank = (values.size() - 1) * point;
    const size_t lower = static_cast<size_t>(std::floor(rank));
    const size_t upper = static_cast<size_t>(std::ceil(rank));
    return values[lower] + (rank - lower) * (values[upper] - values[lower]);
  };
  std::vector<double> output;
  output.reserve(101);
  for (size_t index = 0; index < 50; ++index) {
    const double point = (static_cast<double>(index) * 98.0 / 49.0 + 1.0) / 100.0;
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
      const double target = (static_cast<double>(index) * 98.0 / 49.0 + 1.0) / 100.0;
      const auto upper = std::lower_bound(cumulative.begin(), cumulative.end(), target);
      if (upper == cumulative.begin()) {
        output.push_back(weighted_values.front());
      } else if (upper == cumulative.end()) {
        output.push_back(weighted_values.back());
      } else {
        const size_t right = static_cast<size_t>(upper - cumulative.begin());
        const size_t left = right - 1;
        const double span = cumulative[right] - cumulative[left];
        const double alpha = span == 0.0 ? 0.0 : (target - cumulative[left]) / span;
        output.push_back(weighted_values[left] + alpha * (weighted_values[right] - weighted_values[left]));
      }
    }
  }
  const double sum = std::accumulate(values.begin(), values.end(), 0.0);
  output.push_back(sum / static_cast<double>(values.size()));
  return output;
}

static std::vector<double> feature_row(
    const std::vector<analytical::Instr>& instrs,
    int window_size,
    const std::map<std::string, uint16_t>& config) {
  const auto throughputs =
      analytical::get_throughput_single_config(instrs, window_size, config);
  std::vector<double> row;
  for (const auto& entry : analytical::RESOURCE_REGISTRY) {
    if (!entry.enabled) continue;
    const auto& values = throughputs[static_cast<size_t>(entry.resource)];
    std::vector<double> samples;
    if (!values.empty()) samples = values.front().data;
    const auto summary = cdf_summary(std::move(samples));
    row.insert(row.end(), summary.begin(), summary.end());
  }
  return row;
}

static void write_region_rows(
    const std::vector<analytical::Instr>& instrs,
    int window_size,
    const std::vector<std::map<std::string, uint16_t>>& configs) {
  static_assert(std::endian::native == std::endian::little,
                "Anamol feature stream requires little-endian host");
  for (const auto& config : configs) {
    const auto row = feature_row(instrs, window_size, config);
    std::cout.write(reinterpret_cast<const char*>(row.data()),
                    static_cast<std::streamsize>(row.size() * sizeof(double)));
  }
  if (!std::cout)
    throw std::runtime_error("cannot write Anamol feature stream");
}

static void write_trace_region(size_t instruction_count) {
  static_assert(std::endian::native == std::endian::little,
                "Anamol validation stream requires little-endian host");
  const uint64_t value = instruction_count;
  std::cout.write(reinterpret_cast<const char*>(&value), sizeof(value));
  if (!std::cout)
    throw std::runtime_error("cannot write Anamol validation stream");
}

// Return the filename stem (no directory, no extension) of a path.
// e.g. "traces/collatz_trace_with_latency.csv" -> "collatz_trace_with_latency"
static std::string stem_of(const std::string& path) {
  std::string base = path;
  auto slash = base.rfind('/');
  if (slash != std::string::npos) base = base.substr(slash + 1);
  auto dot = base.rfind('.');
  if (dot != std::string::npos) base = base.substr(0, dot);
  return base;
}

static std::string zero_pad(size_t n, int width) {
  std::ostringstream ss;
  ss << std::setfill('0') << std::setw(width) << n;
  return ss.str();
}

int main(int argc, char* argv[]) {
  std::string csv_file = "trace.csv";
  std::string proto_file;
  int window_size = 400;
  std::string output_dir;       // empty = auto-derive from csv_file
  std::string latencies_npy;    // empty = single-run mode
  std::string config_json;      // empty = full sweep (default)
  bool configs_stdin = false;
  bool validate_trace = false;

  const char* short_opts = "hw:t:o:l:c:p:sv";
  const option long_opts[] = {
      {"help",          no_argument,       nullptr, 'h'},
      {"window",        required_argument, nullptr, 'w'},
      {"tracefile",     required_argument, nullptr, 't'},
      {"output-dir",    required_argument, nullptr, 'o'},
      {"latencies-npy", required_argument, nullptr, 'l'},
      {"config-json",   required_argument, nullptr, 'c'},
      {"trace-proto",   required_argument, nullptr, 'p'},
      {"configs-stdin", no_argument,       nullptr, 's'},
      {"validate-trace", no_argument,      nullptr, 'v'},
      {nullptr, 0, nullptr, 0},
  };

  while (true) {
    int opt = getopt_long(argc, argv, short_opts, long_opts, nullptr);
    if (opt == -1) break;

    switch (opt) {
      case 'h':
        print_usage(argv[0]);
        return 0;
      case 'w':
        window_size = std::stoi(optarg);
        break;
      case 't':
        csv_file = optarg;
        break;
      case 'o':
        output_dir = optarg;
        break;
      case 'l':
        latencies_npy = optarg;
        break;
      case 'c':
        config_json = optarg;
        break;
      case 'p':
        proto_file = optarg;
        break;
      case 's':
        configs_stdin = true;
        break;
      case 'v':
        validate_trace = true;
        break;
      case '?':
      default:
        print_usage(argv[0]);
        return 1;
    }
  }

  if (!proto_file.empty()) {
    try {
      if (validate_trace == configs_stdin)
        throw std::runtime_error("protobuf mode requires exactly one of --validate-trace or --configs-stdin");
      if (validate_trace) {
        analytical::stream_proto_region(
            proto_file,
            [](std::vector<analytical::Instr>&& region) {
              write_trace_region(region.size());
            });
      } else {
        const auto configs = read_configs_stdin();
        analytical::stream_proto_region(
            proto_file,
            [&](std::vector<analytical::Instr>&& region) {
              write_region_rows(region, window_size, configs);
            });
      }
      return 0;
    } catch (const std::exception& e) {
      std::cerr << "Anamol error: " << e.what() << "\n";
      return 1;
    }
  }

  // Derive output directory from trace stem if not provided.
  if (output_dir.empty()) {
    output_dir = "output/" + stem_of(csv_file);
  }

  std::cout << "Analytical model driver\n";
  std::cout << "Trace file : " << csv_file << "\n";
  std::cout << "Window size: " << window_size << "\n";
  std::cout << "Output dir : " << output_dir << "\n";
  if (!latencies_npy.empty())
    std::cout << "Latencies  : " << latencies_npy << "\n";
  if (!config_json.empty())
    std::cout << "Config     : " << config_json << " (single-config mode)\n";

  // Parse single config if provided
  bool single_config_mode = !config_json.empty();
  std::map<std::string, uint16_t> single_config;
  if (single_config_mode) {
    try {
      single_config = parse_config_json(config_json);
    } catch (const std::exception& e) {
      std::cerr << "Error parsing --config-json: " << e.what() << "\n";
      return 1;
    }
  }

  // ROB latency analysis is always run for the default 11-size sweep
  // ({1,2,4,...,1024}). Downstream models consume per-size features
  // (rob1_issue_*, rob2_issue_*, ...); narrowing to the config's rob_size
  // would break that feature set. Leave this empty so the C++ side falls
  // back to its built-in default list.
  std::vector<uint16_t> rob_sizes_for_analysis;

  std::cout << "\nParsing and converting trace...\n";
  std::vector<analytical::Instr> instrs =
      analytical::parse_and_convert(csv_file);
  std::cout << "Converted " << instrs.size() << " instructions\n";

  if (instrs.empty()) {
    std::cerr << "Error: No instructions parsed from " << csv_file << "\n";
    return 1;
  }

  std::size_t num_windows =
      (instrs.size() + static_cast<std::size_t>(window_size) - 1) /
      static_cast<std::size_t>(window_size);
  std::cout << "Number of windows: " << num_windows << "\n";

  if (latencies_npy.empty()) {
    // ── Single-run mode ──────────────────────────────────────────────────────
    std::cout << "\nCalculating throughput...\n";
    analytical::PerResThrVecs per_res_thr_vecs =
        single_config_mode
            ? analytical::get_throughput_single_config(instrs, window_size,
                                                        single_config)
            : analytical::get_throughput(instrs, window_size);

    std::cout << "Exporting throughputs to " << output_dir << " ...\n";
    analytical::export_throughputs(per_res_thr_vecs, output_dir);
    std::cout << "Done.\n";

    std::cout << "\nCalculating ROB latency analysis...\n";
    std::vector<analytical::RobLatencyData> latency_data =
        analytical::get_rob_latency_analysis(instrs, rob_sizes_for_analysis);

    std::cout << "\nExporting latency analysis to " << output_dir << " ...\n";
    analytical::export_latency_analysis(latency_data, output_dir);
    std::cout << "Done.\n";

  } else {
    // ── Per-cache-config mode ────────────────────────────────────────────────
    auto npy = analytical::open_npy_latencies(latencies_npy, instrs.size());
    std::cout << "Loaded latencies: " << npy.n_configs << " configs × "
              << npy.n_instrs << " instrs\n";

    // Latency-independent resources: compute once with the trace's own latencies
    std::cout << "\nCalculating latency-independent throughputs...\n";
    analytical::PerResThrVecs thr_indep =
        single_config_mode
            ? analytical::get_throughput_single_config(instrs, window_size,
                                                        single_config, false)
            : analytical::get_throughput(instrs, window_size, false);
    analytical::export_throughputs(thr_indep, output_dir);
    std::cout << "Exported to " << output_dir << "\n";

    // Latency-dependent resources + ROB latency analysis: once per cache config
    std::cout << "\nRunning " << npy.n_configs
              << " cache configs (latency-dependent resources + ROB analysis)...\n";

    for (size_t cfg = 0; cfg < npy.n_configs; ++cfg) {
      auto slice = npy.slice(cfg);
      for (size_t i = 0; i < instrs.size(); ++i) {
        instrs[i].fetch_latency = slice[i * 2];
        instrs[i].exe_latency   = slice[i * 2 + 1];
      }

      std::string cfg_dir = output_dir + "/config_" + zero_pad(cfg, 4);

      analytical::PerResThrVecs thr_dep =
          single_config_mode
              ? analytical::get_throughput_single_config(instrs, window_size,
                                                          single_config, true)
              : analytical::get_throughput(instrs, window_size, true);
      analytical::export_throughputs(thr_dep, cfg_dir);

      std::vector<analytical::RobLatencyData> lat =
          analytical::get_rob_latency_analysis(instrs, rob_sizes_for_analysis);
      analytical::export_latency_analysis(lat, cfg_dir);

      std::cout << "  [" << (cfg + 1) << "/" << npy.n_configs << "] config_"
                << zero_pad(cfg, 4) << "\n";
    }

    std::cout << "Done.\n";
  }

  return 0;
}
