CC ?= clang
CFLAGS ?= -O3 -ffp-contract=off -Wall -Wextra -Iinclude
LDFLAGS ?= -lm

ifeq ($(OS),Windows_NT)
    TARGET = vitna-anchor.exe
    ALIAS = vitna-engine.exe
    TESTS = vitna-anchor-tests.exe
    LDFLAGS += -lws2_32
    CP = cp
else
    TARGET = vitna-anchor
    ALIAS = vitna-engine
    TESTS = vitna-anchor-tests
    LDFLAGS += -lpthread
    CP = cp -f
endif

SRCS = $(sort $(wildcard src/*.c))
LIB_SRCS = $(filter-out src/main.c,$(SRCS))

all: $(TARGET) $(ALIAS)

ifeq ($(VITNA_CUDA),1)
# The CUDA path, for --device cuda. The C sources compile as below with
# VITNA_CUDA defined, nvcc compiles src/model_cuda.cu for CUDA_ARCH (by
# default this machine's GPU; name one, such as sm_86, where there is none),
# and nvcc links, which brings in the CUDA runtime statically.
NVCC ?= nvcc
CUDA_ARCH ?= native
NVCCFLAGS ?= -O3
CUDA_OBJ = cuda-obj
OBJS = $(patsubst src/%.c,$(CUDA_OBJ)/%.o,$(SRCS)) $(CUDA_OBJ)/model_cuda.o
LIB_OBJS = $(filter-out $(CUDA_OBJ)/main.o,$(OBJS))

$(CUDA_OBJ):
	mkdir -p $(CUDA_OBJ)

$(CUDA_OBJ)/%.o: src/%.c | $(CUDA_OBJ)
	$(CC) $(CFLAGS) -DVITNA_CUDA=1 -c $< -o $@

$(CUDA_OBJ)/unit_tests.o: tests/unit_tests.c | $(CUDA_OBJ)
	$(CC) $(CFLAGS) -DVITNA_CUDA=1 -c $< -o $@

$(CUDA_OBJ)/model_cuda.o: src/model_cuda.cu include/gpu_compat.h include/model_cuda.h include/model.h include/expert_stream.h | $(CUDA_OBJ)
	$(NVCC) $(NVCCFLAGS) -arch=$(CUDA_ARCH) -Iinclude -c $< -o $@

$(TARGET): $(OBJS)
	$(NVCC) $(OBJS) -o $(TARGET) $(LDFLAGS)

$(TESTS): $(LIB_OBJS) $(CUDA_OBJ)/unit_tests.o
	$(NVCC) $(LIB_OBJS) $(CUDA_OBJ)/unit_tests.o -o $(TESTS) $(LDFLAGS)
else ifeq ($(VITNA_HIP),1)
# The AMD ROCm HIP path, compiling model_cuda.cu via hipcc with gpu_compat.h.
# Name an architecture with HIP_ARCH (for example gfx1100 or gfx1201).
HIPCC ?= hipcc
HIP_ARCH ?=
HIPFLAGS ?= -O3
HIP_OBJ = hip-obj
HIP_ARCH_FLAG = $(if $(HIP_ARCH),--offload-arch=$(HIP_ARCH),)
OBJS = $(patsubst src/%.c,$(HIP_OBJ)/%.o,$(SRCS)) $(HIP_OBJ)/model_hip.o
LIB_OBJS = $(filter-out $(HIP_OBJ)/main.o,$(OBJS))

$(HIP_OBJ):
	mkdir -p $(HIP_OBJ)

$(HIP_OBJ)/%.o: src/%.c | $(HIP_OBJ)
	$(CC) $(CFLAGS) -DVITNA_CUDA=1 -DVITNA_HIP=1 -c $< -o $@

$(HIP_OBJ)/unit_tests.o: tests/unit_tests.c | $(HIP_OBJ)
	$(CC) $(CFLAGS) -DVITNA_CUDA=1 -DVITNA_HIP=1 -c $< -o $@

$(HIP_OBJ)/model_hip.o: src/model_cuda.cu include/gpu_compat.h include/model_cuda.h include/model.h include/expert_stream.h | $(HIP_OBJ)
	$(HIPCC) $(HIPFLAGS) $(HIP_ARCH_FLAG) -x hip -DVITNA_CUDA=1 -DVITNA_HIP=1 -Iinclude -c $< -o $@

$(TARGET): $(OBJS)
	$(HIPCC) $(OBJS) -o $(TARGET) $(LDFLAGS)

$(TESTS): $(LIB_OBJS) $(HIP_OBJ)/unit_tests.o
	$(HIPCC) $(LIB_OBJS) $(HIP_OBJ)/unit_tests.o -o $(TESTS) $(LDFLAGS)
else
$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS)

$(TESTS): $(LIB_SRCS) tests/unit_tests.c
	$(CC) $(CFLAGS) $(LIB_SRCS) tests/unit_tests.c -o $(TESTS) $(LDFLAGS)
endif

$(ALIAS): $(TARGET)
	$(CP) $(TARGET) $(ALIAS)

test: $(TESTS)
	./$(TESTS)

clean:
	rm -f $(TARGET) $(ALIAS) $(TESTS) src/*.o
	rm -rf cuda-obj hip-obj

.PHONY: all clean test
