
CUDA_PATH ?= /usr/local/cuda
HOST_COMPILER ?= cc
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)
NVFLAGS  := -I. -I$(CUDA_PATH)/include -I$(CUDA_PATH)/samples/common/inc

NVLIBS :=  -L$(CUDA_PATH)/lib64 -lcudart

NVFLAGS_RELEASE  := -m64  -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_60,code=compute_60
NVFLAGS_DEBUG  := -g -G -m64  -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_60,code=compute_60

CC := cc
CXX := cc

# # uncomment to disable OpenGL functionality
# NO_OPENGL := true

CXXFLAGS := -Idependencies/include -I. -I./inc 
ifdef NO_OPENGL
	CXXFLAGS := $(CXXFLAGS) -DNO_OPENGL
endif

CXXFLAGS := $(CXXFLAGS) -DOUTPUT_TXT
NVFLAGS := $(NVFLAGS) -DOUTPUT_TXT

CXXFLAGS_DEBUG := -D_DEBUG -Wall -g -Wno-sign-compare #-fopenmp
CXXFLAGS_RELEASE := -O3 -Wreturn-type #fopenmp

LDFLAGS := -L/opt/local/lib -lgomp -lstdc++ -lm

#-Ldependencies/lib -L/opt/local/lib -L/usr/lib/x86_64-linux-gnu -lpng -lz -llapack -lblas -lboost_filesystem -lboost_system -lboost_thread -lstdc++ -ljsoncpp -lgomp -lalglib -lstdc++

#LDFLAGS := -Ldependencies/lib -L/opt/local/lib -lpng -lz -ltaucs -llapack -lblas -lboost_filesystem -lboost_system -lboost_thread -ljson -lgomp -lalglib
#LDFLAGS := -Ldependencies/lib -L/opt/local/lib -lpng -lz -ltaucs -llapack -lblas -lboost_filesystem-mt -lboost_system-mt -lboost_thread-mt -ljson -lgomp -lalglib
ifndef NO_OPENGL
	LDFLAGS := $(LDFLAGS) -lglut -lGLU -lGL
endif


OBJ := \
	ccd.o \
	cmesh.o\
	cmodel.o\
	collid.o\
	cubic.o\
	tmbvh.o\
	\
	Collision.o\
	\
	collision.o

.PHONY: all debug release tags clean

all: debug release ctags

debug: bin/objcdd.a ctags

release: bin/objcd.a ctags

bin/objcdd.a: $(addprefix build/debug/,$(OBJ))
#	$(CXX) $^ -o $@ $(LDFLAGS) $(LDLIBS)
#	$(NVCC) $(NVFLAGS) $(NVFLAGS_RELEASE) $^ -o $@ $(LDFLAGS) $(LDLIBS) $(NVLIBS)
	$(AR) -r $@ $^

bin/objcd.a: $(addprefix build/release/,$(OBJ))
#	$(CXX) $^ -o $@ $(LDFLAGS) $(LDLIBS)
#	$(NVCC) $(NVFLAGS) $(NVFLAGS_RELEASE) $^ -o $@ $(LDFLAGS) $(LDLIBS) $(NVLIBS)
	$(AR) -r $@ $^

build/debug/%.o: %.cpp
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) $(NVFLAGS) $(CXXFLAGS_DEBUG) $< -o $@

build/debug/%.o: %.cu
	$(NVCC) -c $(NVFLAGS) $(NVFLAGS_DEBUG) $< -o $@

build/release/%.o: %.cpp
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) $(NVFLAGS) $(CXXFLAGS_RELEASE) $< -o $@

build/release/%.o: %.cu
	$(NVCC) -c $(NVFLAGS) $(NVFLAGS_RELEASE) $< -o $@

# Nicked from http://www.gnu.org/software/make/manual/make.html#Automatic-Prerequisites
ctags:
	ctags -w *.?pp
	etags *.?pp

clean:
	rm -rf bin/* build/debug/* build/release/*
