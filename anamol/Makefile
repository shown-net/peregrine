CXX := g++
CXXFLAGS := -std=c++20 -Wall -Wextra
GEM5_ROOT ?= ../../gem5
PROTOC ?= protoc
INCLUDE := -I include -I src -I build/proto
LDLIBS := -lprotobuf -lzstd

# OpenMP usage: set OMP=0 to disable OpenMP (e.g., make OMP=0)
OMP ?= 1
ifeq ($(OMP),1)
OMP_FLAG := -fopenmp
else
OMP_FLAG :=
endif

# Python interpreter
PYTHON := python3

SRC_DIR := src
TESTS_DIR := tests

BUILD := build
OBJDIR := $(BUILD)/obj

# ---- Parameters ----
WINDOW_SIZE    ?= 400
RANDOM_CONFIGS ?= 1000
SEED           ?=
# Specific config override (JSON string). When set, --config-json is used instead of --random-configs.
# Wrap the assignment in single quotes on the command line: 'CONFIG_JSON={"rob_size":256}'
CONFIG_JSON    ?=

# Optional: path to latency .npy produced by gen_cache_latencies.py (enables per-cache-config sweep)
LATENCIES_NPY  ?=
# Optional: path to configs JSON produced alongside the .npy (enables latency-dependent lookup)
CONFIGS_JSON   ?=

# Pass TRACE_DIR=traces/foo to derive all three paths from the trace folder at once.
# Expects: <folder>/trace.csv, <folder>/trace_latencies.npy, <folder>/trace_configs.json
# Output goes to output/<folder-name>/  (e.g. output/foo/)
TRACE_DIR      ?=
ifneq ($(TRACE_DIR),)
# Strip trailing slash so notdir works: traces/foo/ → traces/foo
_TRACE_DIR     := $(patsubst %/,%,$(TRACE_DIR))
TRACE_CSV      := $(_TRACE_DIR)/trace.csv
LATENCIES_NPY  := $(_TRACE_DIR)/trace_latencies.npy
CONFIGS_JSON   := $(_TRACE_DIR)/trace_configs.json
TRACE_NAME     := $(notdir $(_TRACE_DIR))
endif

# Trace file (used when TRACE_DIR is not set)
TRACE_CSV   ?= traces/trace.csv
# Derive trace name from CSV stem (e.g. collatz_trace_with_latency)
TRACE_NAME  ?= $(basename $(notdir $(TRACE_CSV)))
# Per-trace output directory
OUTPUT_DIR  := output/$(TRACE_NAME)
# Throughput lookup table for this trace
LOOKUP_FILE := $(OUTPUT_DIR)/throughput_lookup.pkl
# Per-trace training data output
TRAINING_DIR  ?= training
TRAINING_OUT  := $(TRAINING_DIR)/$(TRACE_NAME).pkl

# ---- Registry code generation ----
# Edit registry.yaml to add/remove/enable/disable resources and params, then run make.
REGISTRY_CONFIG ?= registry.yaml
GENERATED_HEADERS := include/resources_gen.h include/params_gen.h include/resource_registry.h include/models_decl_gen.h
GENERATED_PY := python/config_gen.py

$(GENERATED_HEADERS) $(GENERATED_PY): $(REGISTRY_CONFIG) python/gen_registry.py
	$(PYTHON) python/gen_registry.py --config $(REGISTRY_CONFIG)

# ---- Common sources ----

# Serial-only models (no OpenMP), used for benchmark no-OMP binary
SRCS_MODELS_SERIAL := $(SRC_DIR)/models.cpp
OBJS_MODELS_SERIAL := $(patsubst $(SRC_DIR)/%.cpp,$(OBJDIR)/serial_%.o,$(SRCS_MODELS_SERIAL))

# RapidCSV-based parsing
SRCS_PARSER := \
  $(SRC_DIR)/parser.cpp

PROTO_BUILD := $(BUILD)/proto
PROTO_SRC := $(PROTO_BUILD)/peregrine_trace.pb.cc
PROTO_OBJ := $(OBJDIR)/proto/peregrine_trace.pb.o

# ---- Executables ----
SRCS_TEST_PARAMS := \
  $(TESTS_DIR)/test_params.cpp

SRCS_TEST_MODELS := \
  $(TESTS_DIR)/test_models.cpp \
  $(SRCS_PARSER) \
  $(SRC_DIR)/models.cpp

SRCS_MAIN := \
  $(SRC_DIR)/main.cpp \
  $(SRCS_PARSER) \
  $(SRC_DIR)/models.cpp

# ---- Objects ----
OBJS_MAIN := $(patsubst $(SRC_DIR)/%.cpp,$(OBJDIR)/%.o,$(filter $(SRC_DIR)/%,$(SRCS_MAIN))) \
             $(PROTO_OBJ)

OBJS_TEST_PARAMS := $(patsubst $(TESTS_DIR)/%.cpp,$(OBJDIR)/tests/%.o,$(SRCS_TEST_PARAMS))

OBJS_TEST_MODELS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJDIR)/%.o,$(filter $(SRC_DIR)/%,$(SRCS_TEST_MODELS))) \
					$(patsubst $(TESTS_DIR)/%.cpp,$(OBJDIR)/tests/%.o,$(filter $(TESTS_DIR)/%,$(SRCS_TEST_MODELS))) \
					$(PROTO_OBJ)

OBJS_PARSER := $(patsubst $(SRC_DIR)/%.cpp,$(OBJDIR)/standalone_%.o,$(SRCS_PARSER)) \
               $(PROTO_OBJ)

# ---- Default ----
all: $(BUILD)/main $(BUILD)/test_params $(BUILD)/test_models_omp $(BUILD)/test_models_noomp $(BUILD)/parser

# Add a named target for the main executable
anamol: $(BUILD)/main
	@cp $(BUILD)/main anamol || true

# Ensure build directories exist
prep:
	@mkdir -p $(BUILD) $(OBJDIR) $(OBJDIR)/tests

$(PROTO_SRC): $(GEM5_ROOT)/src/proto/peregrine_trace.proto | prep
	@mkdir -p $(PROTO_BUILD)
	$(PROTOC) --cpp_out=$(PROTO_BUILD) -I$(GEM5_ROOT)/src/proto $<

# ---- Compile rules ----
# All objects depend on generated headers so a registry.yaml change triggers a rebuild.
$(OBJDIR)/%.o: $(SRC_DIR)/%.cpp $(GENERATED_HEADERS) $(PROTO_SRC) | prep
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(OMP_FLAG) -c $< -o $@

$(OBJDIR)/proto/%.o: $(PROTO_BUILD)/%.cc | prep
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $@

$(OBJDIR)/tests/%.o: $(TESTS_DIR)/%.cpp $(GENERATED_HEADERS) | prep
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(OMP_FLAG) -c $< -o $@

# Standalone parser executable with STANDALONE_PARSER flag
$(OBJDIR)/standalone_%.o: $(SRC_DIR)/%.cpp $(GENERATED_HEADERS) | prep
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(OMP_FLAG) -DSTANDALONE_PARSER -c $< -o $@

# Serial (no OpenMP) objects for benchmark-only binary
$(OBJDIR)/serial_%.o: $(SRC_DIR)/%.cpp $(GENERATED_HEADERS) | prep
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $@

# ---- Link executables ----
$(BUILD)/main: $(OBJS_MAIN)
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(OMP_FLAG) $^ $(LDLIBS) -o $@

$(BUILD)/parser: $(OBJS_PARSER)
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(OMP_FLAG) $^ $(LDLIBS) -o $@

$(BUILD)/test_params: $(OBJS_TEST_PARAMS)
	$(CXX) $(CXXFLAGS) $(INCLUDE) $^ -o $@

# OpenMP version (parallel)
$(BUILD)/test_models_omp: $(OBJS_TEST_MODELS)
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(OMP_FLAG) $^ $(LDLIBS) -o $@

# Non-OpenMP version (serial, for benchmark)
$(BUILD)/test_models_noomp: $(filter-out $(OBJDIR)/models.o,$(OBJS_TEST_MODELS)) $(OBJS_MODELS_SERIAL)
	$(CXX) $(CXXFLAGS) $(INCLUDE) $^ $(LDLIBS) -o $@

# Keep backward compatibility
$(BUILD)/test_models: $(BUILD)/test_models_omp
	cp $< $@

# Paths to trace-annotation scripts at the repo root (one level up from anamol/)
GEN_CACHE_LATENCIES ?= ../gen_cache_latencies.py
GEN_BP_RATES        ?= ../gen_bp_rates.py

# ---- Run targets ----

# Generate per-cache-config latency .npy and configs JSON for a trace
gen_latencies.run:
	$(PYTHON) $(GEN_CACHE_LATENCIES) $(TRACE_CSV)

# Generate BP misprediction rates JSON for a trace
gen_bp.run:
	$(PYTHON) $(GEN_BP_RATES) $(TRACE_CSV)

main.run: $(BUILD)/main
	$< -t $(TRACE_CSV) -w $(WINDOW_SIZE) -o $(OUTPUT_DIR) \
		$(if $(LATENCIES_NPY),-l $(LATENCIES_NPY),)

# Build throughput lookup table for a specific trace
lookup.run:
	$(PYTHON) python/build_throughput_lookup.py \
		-i $(OUTPUT_DIR) -o $(LOOKUP_FILE) \
		$(if $(CONFIGS_JSON),--configs-json $(CONFIGS_JSON),)

# Generate per-trace training data.
# Default: random configs. Override with 'CONFIG_JSON={"rob_size":256}' for a specific config.
training.run:
	@mkdir -p $(TRAINING_DIR)
	$(PYTHON) python/gen_training_data.py \
		--lookup $(LOOKUP_FILE) \
		--trace $(TRACE_CSV) \
		--window-size $(WINDOW_SIZE) \
		$(if $(CONFIG_JSON),--config-json '$(CONFIG_JSON)',--random-configs $(RANDOM_CONFIGS)) \
		$(if $(SEED),--seed $(SEED),) \
		-o $(TRAINING_OUT)

# Full pipeline for one trace: simulate → lookup → training
pipeline.run: main.run lookup.run training.run

# Full pipeline for all *_with_latency.csv traces
all_traces.run: $(BUILD)/main
	@for csv in $(wildcard traces/*_with_latency.csv); do \
		$(MAKE) pipeline.run TRACE_CSV=$$csv; \
	done

# Consolidate all per-trace training files into a single matrix
consolidate.run:
	$(PYTHON) python/consolidate_training.py \
		-i $(TRAINING_DIR) -o $(TRAINING_DIR)/all_traces.pkl

# ── Sweep-to-training pipeline ───────────────────────────────────────────────
# Generate training data directly from a gem5 sweep results CSV.
# Each row in the CSV becomes one training row (no lookup table required).
#
# Parameters:
#   SWEEP_CSV      path to sweep_results.csv (default: most recent sim_sweep dir)
#   BENCHMARKS     comma-separated benchmark whitelist, e.g. "collatz,sparse"
#                  (default: empty = all benchmarks found in CSV)
#   SWEEP_DO_SWEEP set to 1 to run full param sweeps once per benchmark
#                  instead of single-config runs per row (builds lookup tables)
#   SWEEP_FORMAT   output format: pkl (default) or csv
#   SWEEP_OUT      output file path (default: training/sweep_training.<format>)
#
# Examples:
#   make sweep_training.run BENCHMARKS=collatz
#   make sweep_training.run SWEEP_DO_SWEEP=1 SWEEP_FORMAT=csv
SWEEP_CSV              ?= sim_sweeps/sim_sweep_02_24_2026/sweep_results.csv
BENCHMARKS             ?=
SWEEP_DO_SWEEP         ?= 0
SWEEP_FORMAT           ?= pkl
SWEEP_OUT              ?= $(TRAINING_DIR)/sweep_training.$(SWEEP_FORMAT)
# Set SWEEP_PRECOMPUTED_DIR to an existing anamol output base directory
# (organized as <dir>/<benchmark>/) to skip re-running the binary.
# Example: make sweep_training.run SWEEP_PRECOMPUTED_DIR=output BENCHMARKS=collatz
SWEEP_PRECOMPUTED_DIR  ?=
# Number of parallel worker threads for sweep_training.run (default: os.cpu_count())
SWEEP_WORKERS          ?=
# Output verbosity: quiet=warnings only, normal=1 line per row, verbose=full detail
SWEEP_LOG_LEVEL        ?= verbose
# Directory containing benchmark folders with trace CSV files (default: traces/)
SWEEP_TRACES_DIR       ?= traces
# Intermediate output directory for sweep_training.run (default: .cache/sweep_out)
SWEEP_CACHE_DIR        ?= .cache/sweep_out

sweep_training.run:
	@mkdir -p $(TRAINING_DIR)
	$(PYTHON) python/sweep_to_training.py \
		--sweep-csv $(SWEEP_CSV) \
		--traces-dir $(SWEEP_TRACES_DIR) \
		--window-size $(WINDOW_SIZE) \
		--output-dir $(SWEEP_CACHE_DIR) \
		--format $(SWEEP_FORMAT) \
		-o $(SWEEP_OUT) \
		$(if $(BENCHMARKS),--benchmarks $(BENCHMARKS),) \
		$(if $(filter 1,$(SWEEP_DO_SWEEP)),--do-sweep,) \
		$(if $(SWEEP_PRECOMPUTED_DIR),--precomputed-dir $(SWEEP_PRECOMPUTED_DIR),) \
		$(if $(SWEEP_WORKERS),--workers $(SWEEP_WORKERS),) \
		$(if $(filter-out verbose,$(SWEEP_LOG_LEVEL)),--log-level $(SWEEP_LOG_LEVEL),)

# Convenience run target for anamol
anamol.run: main.run visualize.run
	@true

# Visualizer run target
visualize.run:
	$(PYTHON) python/visualize.py --output-dir $(OUTPUT_DIR)

parser.run: $(BUILD)/parser
	$< $(TRACE_CSV)

test_parser.run: $(BUILD)/parser
	$< $(TRACE_CSV)

test_params.run: $(BUILD)/test_params
	$<

test_models.run: $(BUILD)/test_models
	$< $(TRACE_CSV) $(WINDOW_SIZE)

test_models_omp.run: $(BUILD)/test_models_omp
	$< $(TRACE_CSV) $(WINDOW_SIZE)

test_models_noomp.run: $(BUILD)/test_models_noomp
	$< $(TRACE_CSV) $(WINDOW_SIZE)

# ---- Performance comparison ----
benchmark: $(BUILD)/test_models_omp $(BUILD)/test_models_noomp
	@echo "========================================="
	@echo "Performance Comparison"
	@echo "Trace: $(TRACE_CSV)"
	@echo "Window size: $(WINDOW_SIZE)"
	@echo "========================================="
	@echo ""
	@echo "Running SERIAL (no OpenMP)..."
	@/usr/bin/time -l $(BUILD)/test_models_noomp $(TRACE_CSV) $(WINDOW_SIZE) 2>&1 | grep -E "real|user"
	@echo ""
	@echo "Running PARALLEL with 1 thread..."
	@OMP_NUM_THREADS=1 /usr/bin/time -l $(BUILD)/test_models_omp $(TRACE_CSV) $(WINDOW_SIZE) 2>&1 | grep -E "real|user"
	@echo ""
	@echo "Running PARALLEL with 4 threads..."
	@OMP_NUM_THREADS=4 /usr/bin/time -l $(BUILD)/test_models_omp $(TRACE_CSV) $(WINDOW_SIZE) 2>&1 | grep -E "real|user"
	@echo ""
	@echo "Running PARALLEL with 8 threads..."
	@OMP_NUM_THREADS=8 /usr/bin/time -l $(BUILD)/test_models_omp $(TRACE_CSV) $(WINDOW_SIZE) 2>&1 | grep -E "real|user"
	@echo ""
	@echo "========================================="

# ---- Clean ----
clean:
	rm -rf $(BUILD)

clean-gen:
	rm -f $(GENERATED_HEADERS) $(GENERATED_PY)

.PHONY: all clean clean-gen prep gen_latencies.run gen_bp.run main.run parser.run test_parser.run \
		test_params.run test_models.run test_models_omp.run test_models_noomp.run \
		benchmark anamol anamol.run visualize.run \
		lookup.run training.run pipeline.run all_traces.run consolidate.run \
		sweep_training.run
