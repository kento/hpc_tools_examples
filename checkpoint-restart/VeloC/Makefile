# ==================== Variables ======================

# MPI C++ compiler
CXX 		= mpicc -fPIC
CC		= $(CXX)

# Passed to compiler
OPT_LEVEL	= -O2
DEBUG_LEVEL	= -g 
EXTRA_CCFLAGS	=  -fopenmp -fno-inline -Wno-deprecated 
CXXFLAGS	= $(OPT_LEVEL) $(DEBUG_LEVEL) $(EXTRA_CCFLAGS) 
CFLAGS		= $(CXXFLAGS) 

VELOC_HOME	= /home/kento/opt/veloc

# Passed to the C pre-processor
CPPFLAGS        = -I./ -I$(VELOC_HOME)/include

# Passed to the linker
LDFLAGS         = -L./ -L$(VELOC_HOME)/lib64

# Link with these libraries
LDLIBS          = -lveloc-client

# Installation path
prefix		= 

# =================================================

SRC_DIR=.

#===== edit "pi_mpi_with_veloc" to your binary name ===========
pi_mpi_with_veloc_PROGRAM=pi_mpi_with_veloc
pi_mpi_with_veloc_SRCS = $(SRC_DIR)/pi_mpi_with_veloc.c 
pi_mpi_with_veloc_OBJS = $(pi_mpi_with_veloc_SRCS:%.c=%.o) 
pi_mpi_with_veloc_DEPS = $(pi_mpi_with_veloc_SRCS:%.c=%.d) 
#===================================================

PROGRAM = $(pi_mpi_with_veloc_PROGRAM)
OBJS = $(pi_mpi_with_veloc_OBJS)
DEPS = $(pi_mpi_with_veloc_DEPS)

all: $(PROGRAM) $(LIBS)

-include $(DEPS)

$(pi_mpi_with_veloc_PROGRAM): $(pi_mpi_with_veloc_OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) $(LDLIBS) -o $@ $^

.SUFFIXES: .c .o
.c.o: 
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ -c -MMD -MP $< 

.SUFFIXES: .cpp .o
.cpp.o:
	$(CC) $(CXXFLAGS) $(CPPFLAGS) -o $@ -c -MMD -MP $<

.PHONY: clean
clean:
	-rm -rf $(PROGRAM) $(OBJS) $(DEPS) $(LIBS)

.PHONY: cleanx
clean_core:
	-rm -rf *.core



