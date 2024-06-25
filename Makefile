ARCH= -gencode arch=compute_100,code=sm_100

VPATH=./src/
SLIB=lissdn.so
ALIB=lissdn.a
EXEC=ssdn
OBJDIR=./obj/

CC=gcc
CPP=g++
NVCC=nvcc
AR=ar
ARFLAGS=rcs
LDFLAGS= -lm -pthread -L/usr/local/lib -lstdc++ -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand -lcudnn
COMMON= -Iinclude/ -Isrc/ -DGPU -I/usr/local/cuda/include/ -DCUDNN
CFLAGS=-Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC -fopenmp -Ofast -DGPU -DCUDNN

OBJ=cuda.o utils.o

EXECOBJA=ssdn.o

EXECOBJ = $(addprefix $(OBJDIR), $(EXECOBJA))
OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile include/ssdn.h

all: obj $(SLIB) $(ALIB) $(EXEC)

$(EXEC): $(EXECOBJ) $(ALIB)
	$(CPP) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(ALIB)

$(ALIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $^

$(SLIB): $(OBJS)
	$(CC) $(CFLAGS) -shared $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CPP) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(COMMON) --compiler-options "$(CFLAGS)" -c $< -o $@

obj:
	mkdir -p obj

.PHONY: clean

clean:
	rm -rf $(OBJS) $(SLIB) $(ALIB) $(EXEC) $(EXECOBJ) $(OBJDIR)/*