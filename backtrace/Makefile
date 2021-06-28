# ==================== Variables ======================

# MPI C++ compiler
CXX 		= gcc -fPIC
CC		= $(CXX)

# Passed to compiler
OPT_LEVEL	= -O0
DEBUG_LEVEL	= -g 
EXTRA_CCFLAGS	= 
CXXFLAGS	= $(OPT_LEVEL) $(DEBUG_LEVEL) $(EXTRA_CCFLAGS) 
CFLAGS		= $(CXXFLAGS) 

# Passed to the C pre-processor
CPPFLAGS        = -I./ 

# Passed to the linker
LDFLAGS         = -L./

# Link with these libraries
LDLIBS          = -lstdc++ 

# Installation path
prefix		= 

# =================================================

SRC_DIR=.

#===== edit "ex1" to your binary name ===========
ex1_PROGRAM=ex1
ex1_SRCS = $(SRC_DIR)/ex1.c 
ex1_OBJS = $(ex1_SRCS:%.c=%.o) 
ex1_DEPS = $(ex1_SRCS:%.c=%.d) 
#===================================================

PROGRAM = $(ex1_PROGRAM)
OBJS = $(ex1_OBJS)
DEPS = $(ex1_DEPS)

all: $(PROGRAM) $(LIBS)

-include $(DEPS)

$(ex1_PROGRAM): $(ex1_OBJS)
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



