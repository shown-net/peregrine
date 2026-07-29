#include <getopt.h>

#include <algorithm>
#include <cctype>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "analysis_engine.h"
#include "parser.h"

namespace {

void print_usage(const char* prog) {
  std::cout << "Usage: " << prog
            << " --trace-proto TRACE.pb.zst --configs-json '[{...}]'"
               " --mechanisms-json '[{\"name\":...,\"model\":...,\"params\":[...]}]'"
               " [--window N]\n";
}

void skip_ws(const std::string& text, size_t& pos) {
  while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])))
    ++pos;
}

void expect(const std::string& text, size_t& pos, char value) {
  skip_ws(text, pos);
  if (pos >= text.size() || text[pos] != value)
    throw std::runtime_error(std::string("expected '") + value + "'");
  ++pos;
}

std::string parse_string(const std::string& text, size_t& pos) {
  skip_ws(text, pos);
  expect(text, pos, '"');
  std::string out;
  while (pos < text.size()) {
    char c = text[pos++];
    if (c == '"') return out;
    if (c == '\\') {
      if (pos >= text.size()) throw std::runtime_error("unterminated escape");
      out.push_back(text[pos++]);
    } else {
      out.push_back(c);
    }
  }
  throw std::runtime_error("unterminated string");
}

double parse_number(const std::string& text, size_t& pos) {
  skip_ws(text, pos);
  const size_t start = pos;
  while (pos < text.size() &&
         (std::isdigit(static_cast<unsigned char>(text[pos])) || text[pos] == '.' ||
          text[pos] == '-' || text[pos] == '+'))
    ++pos;
  if (start == pos) throw std::runtime_error("expected number");
  return std::stod(text.substr(start, pos - start));
}

std::vector<analytical::ConfigValues> parse_configs(const std::string& text) {
  size_t pos = 0;
  std::vector<analytical::ConfigValues> out;
  expect(text, pos, '[');
  skip_ws(text, pos);
  while (pos < text.size() && text[pos] != ']') {
    analytical::ConfigValues row;
    expect(text, pos, '{');
    skip_ws(text, pos);
    while (pos < text.size() && text[pos] != '}') {
      auto key = parse_string(text, pos);
      expect(text, pos, ':');
      row[key] = parse_number(text, pos);
      skip_ws(text, pos);
      if (pos < text.size() && text[pos] == ',') ++pos;
    }
    expect(text, pos, '}');
    out.push_back(std::move(row));
    skip_ws(text, pos);
    if (pos < text.size() && text[pos] == ',') ++pos;
  }
  expect(text, pos, ']');
  if (out.empty()) throw std::runtime_error("configs-json must not be empty");
  return out;
}

std::vector<std::string> parse_string_array(const std::string& text, size_t& pos) {
  std::vector<std::string> out;
  expect(text, pos, '[');
  skip_ws(text, pos);
  while (pos < text.size() && text[pos] != ']') {
    out.push_back(parse_string(text, pos));
    skip_ws(text, pos);
    if (pos < text.size() && text[pos] == ',') ++pos;
  }
  expect(text, pos, ']');
  return out;
}

std::vector<analytical::MechanismBinding> parse_mechanisms(const std::string& text) {
  size_t pos = 0;
  std::vector<analytical::MechanismBinding> out;
  expect(text, pos, '[');
  skip_ws(text, pos);
  while (pos < text.size() && text[pos] != ']') {
    analytical::MechanismBinding binding;
    expect(text, pos, '{');
    skip_ws(text, pos);
    while (pos < text.size() && text[pos] != '}') {
      auto key = parse_string(text, pos);
      expect(text, pos, ':');
      if (key == "name") {
        binding.name = parse_string(text, pos);
      } else if (key == "model") {
        binding.model = parse_string(text, pos);
      } else if (key == "params") {
        binding.params = parse_string_array(text, pos);
      } else {
        throw std::runtime_error("unknown mechanism key: " + key);
      }
      skip_ws(text, pos);
      if (pos < text.size() && text[pos] == ',') ++pos;
    }
    expect(text, pos, '}');
    if (binding.name.empty()) throw std::runtime_error("mechanism name is required");
    if (binding.model.empty()) throw std::runtime_error("mechanism model is required");
    out.push_back(std::move(binding));
    skip_ws(text, pos);
    if (pos < text.size() && text[pos] == ',') ++pos;
  }
  expect(text, pos, ']');
  if (out.empty()) throw std::runtime_error("mechanisms-json must not be empty");
  return out;
}

std::vector<analytical::Instr> read_trace(const std::string& trace_path) {
  std::vector<analytical::Instr> region;
  analytical::stream_proto_region(trace_path, [&](std::vector<analytical::Instr>&& parsed) {
    if (!region.empty()) throw std::runtime_error("trace must contain one region");
    region = std::move(parsed);
  });
  return region;
}

}  // namespace

int main(int argc, char* argv[]) {
  std::string trace_proto;
  std::string configs_json;
  std::string mechanisms_json;
  int window_size = 400;

  const option long_opts[] = {
      {"help", no_argument, nullptr, 'h'},
      {"trace-proto", required_argument, nullptr, 'p'},
      {"configs-json", required_argument, nullptr, 'c'},
      {"mechanisms-json", required_argument, nullptr, 'm'},
      {"window", required_argument, nullptr, 'w'},
      {nullptr, 0, nullptr, 0},
  };
  while (true) {
    int opt = getopt_long(argc, argv, "hp:c:m:w:", long_opts, nullptr);
    if (opt == -1) break;
    switch (opt) {
      case 'h': print_usage(argv[0]); return 0;
      case 'p': trace_proto = optarg; break;
      case 'c': configs_json = optarg; break;
      case 'm': mechanisms_json = optarg; break;
      case 'w': window_size = std::stoi(optarg); break;
      default: print_usage(argv[0]); return 1;
    }
  }

  try {
    if (trace_proto.empty() || configs_json.empty() || mechanisms_json.empty()) {
      print_usage(argv[0]);
      return 1;
    }
    auto instrs = read_trace(trace_proto);
    auto configs = parse_configs(configs_json);
    auto mechanisms = parse_mechanisms(mechanisms_json);
    auto values = analytical::analyze_trace(instrs, window_size, configs, mechanisms);
    const size_t rows = configs.size();
    const size_t cols = analytical::feature_count(mechanisms);
    std::cout << "rows=" << rows << " cols=" << cols
              << " instructions=" << instrs.size() << "\n";
    for (size_t row = 0; row < rows; ++row) {
      double sum = 0.0;
      for (size_t col = 0; col < cols; ++col)
        sum += values[row * cols + col];
      std::cout << "row=" << row << " mean=" << (sum / cols) << "\n";
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Anamol debug error: " << error.what() << "\n";
    return 1;
  }
}
