# ==================== Variables ======================

# MPI C++ compiler
CXX 		= mpicc -fPIC
CC		= $(CXX)

# Passed to compiler
OPT_LEVEL	= -O2
DEBUG_LEVEL	= -g 
EXTRA_CCFLAGS	=  -fopenmp -fno-inline -Wno-deprecated # "-fno-omit-frame-pointer" will be automatically used due to -O0 and -g option
CXXFLAGS	= $(OPT_LEVEL) $(DEBUG_LEVEL) $(EXTRA_CCFLAGS) 
CFLAGS		= $(CXXFLAGS) 

# Passed to the C pre-processor
CPPFLAGS        = -I./ 

# Passed to the linker
LDFLAGS         = -L./

# Link with these libraries
LDLIBS          = 

# Installation path
prefix		= 

# =================================================

SRC_DIR=.

#===== edit "pi_mpi_with_cr" to your binary name ===========
pi_mpi_with_cr_PROGRAM=pi_mpi_with_cr
pi_mpi_with_cr_SRCS = $(SRC_DIR)/pi_mpi_with_cr.c 
pi_mpi_with_cr_OBJS = $(pi_mpi_with_cr_SRCS:%.c=%.o) 
pi_mpi_with_cr_DEPS = $(pi_mpi_with_cr_SRCS:%.c=%.d) 
#pi_mpi_with_cr_a_LIBS  = ./libpi_mpi_with_cr.a 
#pi_mpi_with_cr_so_LIBS = ./libpi_mpi_with_cr.so  
#===================================================

PROGRAM = $(pi_mpi_with_cr_PROGRAM)
OBJS = $(pi_mpi_with_cr_OBJS)
DEPS = $(pi_mpi_with_cr_DEPS)
#LIBS = $(pi_mpi_with_cr_a_LIBS) $(pi_mpi_with_cr_so_LIBS) 

all: $(PROGRAM) $(LIBS)

-include $(DEPS)

$(pi_mpi_with_cr_PROGRAM): $(pi_mpi_with_cr_OBJS)
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

.PHONY: cleanckpt
clean_ckpt:
	-rm -rf *.ckpt



