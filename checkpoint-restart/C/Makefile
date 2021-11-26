# ==================== Variables ======================

# MPI C++ compiler
CXX 		= g++ -fPIC
CC		= gcc -fPIC

# Passed to compiler
OPT_LEVEL	= -O2
DEBUG_LEVEL	= -g 
EXTRA_CCFLAGS	= 
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

#===== edit "pi" to your binary name ===========
pi_PROGRAM=pi
pi_SRCS = $(SRC_DIR)/pi.c 
pi_OBJS = $(pi_SRCS:%.c=%.o) 
pi_DEPS = $(pi_SRCS:%.c=%.d) 
#===================================================

#===== edit "pi_with_cr" to your binary name ===========
pi_with_cr_PROGRAM=pi_with_cr
pi_with_cr_SRCS = $(SRC_DIR)/pi_with_cr.c 
pi_with_cr_OBJS = $(pi_with_cr_SRCS:%.c=%.o) 
pi_with_cr_DEPS = $(pi_with_cr_SRCS:%.c=%.d) 
#===================================================

PROGRAM = $(pi_PROGRAM) $(pi_with_cr_PROGRAM)
OBJS = $(pi_OBJS) $(pi_with_cr_OBJS)
DEPS = $(pi_DEPS) $(pi_with_cr_DEPS)


all: $(PROGRAM) $(LIBS)

-include $(DEPS)

$(pi_PROGRAM): $(pi_OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) $(LDLIBS) -o $@ $^

$(pi_with_cr_PROGRAM): $(pi_with_cr_OBJS)
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

.PHONY: clean_ckpt
clean_ckpt:
	-rm -rf pi_count.*.ckpt



