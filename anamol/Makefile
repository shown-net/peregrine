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
PYBIND11_INCLUDES := $(shell $(PYTHON) -m pybind11 --includes 2>/dev/null)
PY_EXT_SUFFIX := $(shell $(PYTHON) -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')

SRC_DIR := src
TESTS_DIR := tests

BUILD := build
OBJDIR := $(BUILD)/obj

# ---- Common sources ----

SRCS_PARSER := \
  $(SRC_DIR)/parser.cpp

PROTO_BUILD := $(BUILD)/proto
PROTO_SRC := $(PROTO_BUILD)/peregrine_trace.pb.cc
PROTO_OBJ := $(OBJDIR)/proto/peregrine_trace.pb.o
PROTO_PYEXT_OBJ := $(OBJDIR)/pyext_proto/peregrine_trace.pb.o
ANAMOL_ROOT_PYEXT := _analysis$(PY_EXT_SUFFIX)

SRCS_PYEXT := \
  $(SRC_DIR)/python_extension.cpp \
  $(SRC_DIR)/analysis_engine.cpp \
  $(SRCS_PARSER)

# ---- Objects ----
OBJS_PYEXT := $(patsubst $(SRC_DIR)/%.cpp,$(OBJDIR)/pyext_%.o,$(filter $(SRC_DIR)/%,$(SRCS_PYEXT))) \
              $(PROTO_PYEXT_OBJ)

# ---- Default ----
all: $(ANAMOL_ROOT_PYEXT)

python-extension: $(ANAMOL_ROOT_PYEXT)

# Ensure build directories exist
prep:
	@mkdir -p $(BUILD) $(OBJDIR) $(OBJDIR)/tests

$(PROTO_SRC): $(GEM5_ROOT)/src/proto/peregrine_trace.proto | prep
	@mkdir -p $(PROTO_BUILD)
	$(PROTOC) --cpp_out=$(PROTO_BUILD) -I$(GEM5_ROOT)/src/proto $<

# ---- Compile rules ----
$(OBJDIR)/%.o: $(SRC_DIR)/%.cpp $(PROTO_SRC) | prep
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(OMP_FLAG) -c $< -o $@

$(OBJDIR)/proto/%.o: $(PROTO_BUILD)/%.cc | prep
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $@

$(OBJDIR)/pyext_proto/%.o: $(PROTO_BUILD)/%.cc | prep
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -fPIC $(INCLUDE) -c $< -o $@

$(OBJDIR)/pyext_%.o: $(SRC_DIR)/%.cpp $(PROTO_SRC) | prep
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -fPIC $(INCLUDE) $(PYBIND11_INCLUDES) $(OMP_FLAG) -c $< -o $@

$(ANAMOL_ROOT_PYEXT): $(OBJS_PYEXT)
	$(CXX) $(CXXFLAGS) -shared $(OMP_FLAG) $^ $(LDLIBS) -o $@

# ---- Clean ----
clean:
	rm -rf $(BUILD)

clean-gen:
	rm -f include/resources_gen.h include/params_gen.h include/resource_registry.h include/models_decl_gen.h python/config_gen.py

.PHONY: all clean clean-gen prep python-extension
