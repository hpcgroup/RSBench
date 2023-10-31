#===============================================================================
# User Options
#===============================================================================

RAJA_PATH ?= 
Umpire_PATH ?=
camp_PATH ?=
EXECUTION_POLICY ?= 
SM_VERSION ?=
OFFLOAD_ARCH ?=

OPTIMIZE ?= yes
DEBUG ?= no
PROFILE ?= no
MPI ?= no

RAJA_INCLUDE_DIR = $(RAJA_PATH)/include
RAJA_LIB_DIR = $(RAJA_PATH)/lib

Umpire_INCLUDE_DIR = $(Umpire_PATH)/include
Umpire_LIB_DIR = $(Umpire_PATH)/lib

camp_INCLUDE_DIR ?= $(camp_PATH)/include
camp_LIB_DIR = $(camp_PATH)/lib

#===============================================================================
# Program name & source code list
#===============================================================================

program = RSBench

source = \
main.cpp \
io.cpp \
simulation.cpp \
init.cpp \
utils.cpp \
material.cpp \

obj = $(source:.cpp=.o)

CXXFLAGS := -std=c++14

# Linker Flags
LDFLAGS = -lm

# Debug Flags
ifeq ($(DEBUG),yes)
  CXXFLAGS += -g -G
  LDFLAGS += -g -G
endif

# Profiling Flags
ifeq ($(PROFILE),yes)
  CXXFLAGS += -pg
  LDFLAGS  += -pg
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CXXFLAGS += -O3
endif

ifeq ($(EXECUTION_POLICY),CUDA)
  CXX = nvcc
  CUDAFLAGS = -x cu --expt-relaxed-constexpr -forward-unknown-to-host-compiler -extended-lambda -restrict -keep -arch=sm_$(SM_VERSION) 
  CXXFLAGS += $(CUDAFLAGS) -DCUDA
endif  

ifeq ($(EXECUTION_POLICY), OpenMP)
  CXXFLAGS += -fopenmp -DOpenMP
endif

default: $(program)

CXXFLAGS += -I./ -I$(RAJA_INCLUDE_DIR) -I$(Umpire_INCLUDE_DIR) -I$(camp_INCLUDE_DIR)
LDFLAGS += -lRAJA -lumpire -lcamp
LDFLAGS += -L$(RAJA_LIB_DIR) -L$(Umpire_LIB_DIR) -L$(camp_LIB_DIR)
LDFLAGS += -forward-unknown-to-host-compiler -Wl,-rpath=$(RAJA_LIB_DIR)

CXXFLAGS += -w

$(program): $(obj) rsbench.h Makefile
	$(CXX) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp rsbench.h Makefile
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf rsbench $(obj)

edit:
	vim -p $(source) rsbench.h

run:
	./rsbench

