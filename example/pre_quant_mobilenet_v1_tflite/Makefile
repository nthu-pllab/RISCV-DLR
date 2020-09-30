# Dependency install directories
RISCV_INSTALL     := /path/to/riscv-gnu-toolchain/install #[TODO]
DLR_INSTALL       := /path/to/DLR-normal-build/install #[TODO]
DLR_RISCV_INSTALL := /path/to/DLR-riscv-build/install #[TODO]
LLVM_INSTALL      := /path/to/llvm-rvp/build #[TODO]

# required executable path
DUMP_KERNEL := $(DLR_INSTALL)/bin/DumpKernel
CXX         := $(RISCV_INSTALL)/bin/riscv64-unknown-elf-g++
OBJDUMP     := $(RISCV_INSTALL)/bin/riscv64-unknown-elf-objdump
SPIKE       := $(RISCV_INSTALL)/bin/spike
PK          := $(RISCV_INSTALL)/riscv64-unknown-elf/bin/pk
LLC         := $(LLVM_INSTALL)/bin/llc

DLR_INCLUDE_DIR := $(DLR_RISCV_INSTALL)/include
DLR_STATIC_LIB  := $(DLR_RISCV_INSTALL)/lib/libDLR.a

LLC_FLAGS := --march="riscv64" --mattr=+F,+M,+P

CXX_FLAGS := -I$(DLR_RISCV_INSTALL)/include
CXX_FLAGS += -Wl,--whole-archive $(DLR_STATIC_LIB) -Wl,--no-whole-archive
CXX_FLAGS += -std=c++11
CXX_FLAGS += -g

MODEL_NAME := mobilenet
MODEL_GRAPH := $(MODEL_NAME).graph
MODEL_LIB   := $(MODEL_NAME).ll
MODEL_PARAM := $(MODEL_NAME).params
MODEL_ASM   := $(MODEL_NAME).s
EXECUTABLE  := $(MODEL_NAME)
EXE_DUMP    := $(MODEL_NAME).dump

BUILD_MODULE_PY := build_module.py
HOST_CODE       := host.cpp
KERNEL_INC      := kernel.inc

LOGS_DIR         := logs
BUILD_MODULE_LOG := $(LOGS_DIR)/build-module.log
BUILD_MODULE_ERR := $(LOGS_DIR)/build-module.err
RUN_LOG := $(LOGS_DIR)/run.log
RUN_ERR := $(LOGS_DIR)/run.err

TARGETS :=
TARGETS += $(MODEL_GRAPH) $(MODEL_LIB) $(MODEL_PARAM)
TARGETS += $(KERNEL_INC)
TARGETS += $(MODEL_ASM)
TARGETS += $(EXECUTABLE) $(EXE_DUMP)

.PHONY: all
all: $(LOGS_DIR) $(TARGETS)

.PHONY: run
run: $(TARGETS)
	$(SPIKE) $(PK) $(EXECUTABLE) $(MODEL_GRAPH) $(MODEL_PARAM) $(MODEL_LIB) > $(RUN_LOG) 2> $(RUN_ERR)

.PHONY: clean
clean:
	$(RM) $(TARGETS)

$(MODEL_GRAPH) $(MODEL_LIB) $(MODEL_PARAM) &: $(BUILD_MODULE_PY)
	python $< > $(BUILD_MODULE_LOG) 2> $(BUILD_MODULE_ERR)

$(KERNEL_INC): $(MODEL_GRAPH) $(MODEL_PARAM) $(MODEL_LIB)
	$(DUMP_KERNEL) $^ > $@

$(MODEL_ASM): $(MODEL_LIB)
	$(LLC) $(LLC_FLAGS) $< -o $@

$(EXECUTABLE): $(HOST_CODE) $(MODEL_ASM) $(KERNEL_INC)
	$(CXX) $(CXX_FLAGS) $(HOST_CODE) $(MODEL_ASM) -o $@

$(EXE_DUMP): $(EXECUTABLE)
	$(OBJDUMP) -dC $< > $@

$(LOGS_DIR):
	@mkdir -p $(LOGS_DIR)
