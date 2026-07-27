#include <iomanip>
#include <iostream>
#include <vector>

#include "instr.h"
#include "models.h"
#include "parser.h"

static const char* resource_name(analytical::Resource res) {
  using analytical::Resource;
  switch (res) {
    case Resource::ROB:
      return "ROB";
    case Resource::LOAD_QUEUE:
      return "LOAD_QUEUE";
    case Resource::STORE_QUEUE:
      return "STORE_QUEUE";
    case Resource::ALU_ISSUE:
      return "ALU_ISSUE";
    case Resource::FP_ISSUE:
      return "FP_ISSUE";
    case Resource::LS_ISSUE:
      return "LS_ISSUE";
    case Resource::LOAD_LS_PIPES_LOWER:
      return "LOAD_LS_PIPES_LOWER";
    case Resource::LOAD_LS_PIPES_UPPER:
      return "LOAD_LS_PIPES_UPPER";
    case Resource::ICACHE_FILLS:
      return "ICACHE_FILLS";
    case Resource::FETCH_BUFFERS:
      return "FETCH_BUFFERS";
    default:
      return "UNKNOWN";
  }
}

static void print_throughput_results(
    const analytical::PerResThrVecs& per_res_thr_vecs,
    std::size_t expected_num_windows) {
  using namespace analytical;

  std::cout << "\n========================================\n";
  std::cout << "Throughput Results (summary)\n";
  std::cout << "========================================\n";

  const std::size_t num_resources = static_cast<std::size_t>(Resource::COUNT);

  for (std::size_t r = 0; r < num_resources; ++r) {
    Resource res = static_cast<Resource>(r);
    const auto& thr_vecs = per_res_thr_vecs[r];

    const char* res_name = resource_name(res);
    std::cout << "\n" << res_name << ":\n";
    std::cout << "  num_param_combos: " << thr_vecs.size() << "\n";

    if (thr_vecs.empty()) {
      std::cout << "  no data\n";
      continue;
    }

    for (std::size_t i = 0; i < thr_vecs.size(); ++i) {
      const ThrVec& tv = thr_vecs[i];

      std::size_t num_windows = tv.data.size();
      std::cout << "    combo " << i << ": ";
      if (tv.double_params) {
        std::cout << "p0=" << tv.p0 << ", p1=" << tv.p1;
      } else {
        std::cout << "p0=" << tv.p0;
      }
      std::cout << "\n";
      std::cout << "      windows: " << num_windows << " (expected "
                << expected_num_windows << ")\n";

      if (num_windows == 0) {
        std::cout << "      no window data\n";
        continue;
      }

      double sum = 0.0;
      double min_thr = tv.data[0];
      double max_thr = tv.data[0];

      for (double thr : tv.data) {
        sum += thr;
        if (thr < min_thr) min_thr = thr;
        if (thr > max_thr) max_thr = thr;
      }

      double avg_thr = sum / static_cast<double>(num_windows);

      std::cout << "      Avg Throughput: " << std::fixed
                << std::setprecision(4) << avg_thr << " IPC\n";
      std::cout << "      Min Throughput: " << min_thr << " IPC\n";
      std::cout << "      Max Throughput: " << max_thr << " IPC\n";
    }
  }
}

int main(int argc, char* argv[]) {
  std::string csv_file = "trace.csv";
  int window_size = 400;

  if (argc > 1) {
    csv_file = argv[1];
  }
  if (argc > 2) {
    window_size = std::stoi(argv[2]);
  }

  std::cout << "Testing get_throughput()\n";
  std::cout << "CSV file: " << csv_file << "\n";
  std::cout << "Window size: " << window_size << "\n";
  std::cout << "========================================\n";

  // Parse and convert trace
  std::cout << "\nParsing and converting trace...\n";
  std::vector<analytical::Instr> instrs =
      analytical::parse_and_convert(csv_file);
  std::cout << "Converted " << instrs.size() << " instructions\n";

  if (instrs.empty()) {
    std::cerr << "Error: No instructions parsed\n";
    return 1;
  }

  std::size_t num_windows =
      (instrs.size() + static_cast<std::size_t>(window_size) - 1) /
      static_cast<std::size_t>(window_size);
  std::cout << "Number of windows: " << num_windows << "\n";

  // Calculate throughput and get results
  std::cout << "\nCalculating throughput...\n";
  analytical::PerResThrVecs per_res_thr_vecs =
      analytical::get_throughput(instrs, window_size);

  // Print results from returned PerResThrVecs
  print_throughput_results(per_res_thr_vecs, num_windows);

  std::cout << "\n========================================\n";
  std::cout << "Results will be written to .npy files in a later step.\n";
  std::cout << "You can already inspect the summary printed above.\n";
  std::cout << "========================================\n";
  std::cout << "Test completed successfully!\n";
  std::cout << "Use Python/NumPy to load and analyze the .npy files once\n"
               "the writing logic is implemented.\n";

  return 0;
}
