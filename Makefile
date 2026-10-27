ARCH= -gencode arch=compute_75,code=sm_75

VPATH=./src/:./app/
SLIB=libssdn.so
ALIB=libssdn.a
EXEC=ssdn
OBJDIR=./obj/

CPP=g++
NVCC=nvcc
AR=ar
ARFLAGS=rcs
LDFLAGS= -lm -pthread -L/usr/local/lib -lstdc++ -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand -lcudnn
COMMON= -Iinclude/ -Isrc/ -DGPU -I/usr/local/cuda/include/ -DCUDNN
CFLAGS=-Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC -fopenmp -Ofast -DGPU -DCUDNN

OBJ=cuda.o utils.o blas.o blas_kernels.o image_kernels.o

EXECOBJA=malm_idag_m3_2x.o malm_edsr_2x.o draft.o ssdn.o

EXECOBJ = $(addprefix $(OBJDIR), $(EXECOBJA))
OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile include/ssdn.h

all: obj data $(SLIB) $(ALIB) $(EXEC)

$(EXEC): $(EXECOBJ) $(ALIB)
	$(CPP) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(ALIB) -O3

$(ALIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $^

$(SLIB): $(OBJS)
	$(CPP) $(CFLAGS) -shared $^ -o $@ $(LDFLAGS) -O3

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CPP) $(COMMON) $(CFLAGS) -c $< -o $@ -O3

$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(COMMON) --compiler-options "$(CFLAGS)" -c $< -o $@ -O3

obj:
	mkdir -p obj
data:
	mkdir -p data

.PHONY: clean

clean:
	rm -rf $(OBJS) $(SLIB) $(ALIB) $(EXEC) $(EXECOBJ) $(OBJDIR)/*