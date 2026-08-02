#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "analysis_engine.h"
#include "parser.h"

namespace py = pybind11;

namespace {

std::vector<analytical::ConfigValues> parse_configs(const py::list& configs) {
  std::vector<analytical::ConfigValues> out;
  out.reserve(static_cast<size_t>(py::len(configs)));
  for (const auto& item : configs) {
    py::dict dict = py::reinterpret_borrow<py::dict>(item);
    analytical::ConfigValues row;
    for (const auto& pair : dict) {
      if (py::isinstance<py::bool_>(pair.second) ||
          py::isinstance<py::int_>(pair.second) ||
          py::isinstance<py::float_>(pair.second)) {
        row.emplace(py::str(pair.first), py::cast<double>(pair.second));
      }
    }
    out.push_back(std::move(row));
  }
  return out;
}

std::vector<analytical::MechanismBinding> parse_mechanisms(
    const py::list& mechanisms) {
  std::vector<analytical::MechanismBinding> out;
  out.reserve(static_cast<size_t>(py::len(mechanisms)));
  for (const auto& item : mechanisms) {
    py::dict dict = py::reinterpret_borrow<py::dict>(item);
    analytical::MechanismBinding binding;
    binding.name = py::cast<std::string>(dict["name"]);
    binding.model = py::cast<std::string>(dict["model"]);
    for (const auto& param : py::reinterpret_borrow<py::tuple>(dict["params"]))
      binding.params.push_back(py::cast<std::string>(param));
    out.push_back(std::move(binding));
  }
  return out;
}

std::vector<analytical::Instr> parse_single_region(const std::string& path) {
  std::vector<analytical::Instr> region;
  analytical::stream_proto_region(
      path,
      [&](std::vector<analytical::Instr>&& parsed) {
        if (!region.empty())
          throw std::runtime_error("Anamol trace file must contain one region");
        region = std::move(parsed);
      });
  return region;
}

}  // namespace

PYBIND11_MODULE(_analysis, module) {
  module.doc() = "In-process Anamol analytical engine";

  module.def("trace_instruction_count", [](const std::string& trace_path) {
    return parse_single_region(trace_path).size();
  });

  module.def("feature_count_for_bindings", [](const py::list& mechanisms) {
    auto bindings = parse_mechanisms(mechanisms);
    return analytical::feature_count(bindings);
  });

  module.def(
      "analyze_trace",
      [](const std::string& trace_path, int window_size, const py::list& configs,
         const py::list& mechanisms) {
        auto rows = parse_configs(configs);
        auto bindings = parse_mechanisms(mechanisms);
        std::vector<analytical::Instr> parsed;
        std::vector<double> flat;
        {
          py::gil_scoped_release release;
          parsed = parse_single_region(trace_path);
          flat = analytical::analyze_trace(parsed, window_size, rows, bindings);
        }
        const size_t row_count = rows.size();
        const size_t column_count = analytical::feature_count(bindings);
        py::array_t<double> output({row_count, column_count});
        std::copy(flat.begin(), flat.end(), output.mutable_data());
        return output;
      });

  module.def(
      "analyze_trace_windows",
      [](const std::string& trace_path, int full_roi_window_size,
         int analysis_window_size, size_t window_count, const py::list& configs,
         const py::list& mechanisms) {
        auto rows = parse_configs(configs);
        auto bindings = parse_mechanisms(mechanisms);
        std::vector<analytical::Instr> parsed;
        std::vector<double> flat;
        {
          py::gil_scoped_release release;
          parsed = parse_single_region(trace_path);
          flat = analytical::analyze_trace_windows(
              parsed, full_roi_window_size, analysis_window_size, window_count, rows,
              bindings);
        }
        const size_t column_count = analytical::feature_count(bindings);
        const size_t config_count = rows.size();
        if (config_count == 0 || column_count == 0 ||
            flat.size() % (config_count * column_count) != 0)
          throw std::runtime_error("invalid Anamol window feature dimensions");
        const size_t actual_window_count = flat.size() / (config_count * column_count);
        py::array_t<double> output({config_count, actual_window_count, column_count});
        std::copy(flat.begin(), flat.end(), output.mutable_data());
        return output;
      });
}
