#pragma once
#include"cuda_runtime.h"
#include"device_launch_parameters.h"

#define max_threads 32

inline bool thread_allocate(int tot_thread, dim3 &grid_size, dim3 &block_size);

#define BLOCKSIZE 64
#define NUMBLOCK(x) (((x)-1)/BLOCKSIZE+1)