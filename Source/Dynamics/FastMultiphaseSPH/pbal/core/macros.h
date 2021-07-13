#pragma once

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) < (b) ? (b) : (a))

#include <limits.h>

#if defined(_WIN32) || defined(_WIN64)
#define FLUID_WINDOWS
#elif defined(__APPLE__)
#define FLUID_APPLE
#ifndef FLUID_IOS
#define FLUID_MACOSX
#endif
#elif defined(linux) || defined(__linux__)
#define FLUID_LINUX
#endif

// Host vs. device
#ifdef __CUDACC__
#define FLUID_CUDA_DEVICE __device__
#define FLUID_CUDA_HOST __host__
#else
#define FLUID_CUDA_DEVICE
#define FLUID_CUDA_HOST
#endif  // __CUDACC__
#define FLUID_CUDA_HOST_DEVICE FLUID_CUDA_HOST FLUID_CUDA_DEVICE

// Alignment
#ifdef __CUDACC__  // NVCC
#define FLUID_CUDA_ALIGN(n) __align__(n)
#elif defined(__GNUC__)  // GCC
#define FLUID_CUDA_ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER)  // MSVC
#define FLUID_CUDA_ALIGN(n) __declspec(align(n))
#else
#error "Don't know how to handle FLUID_CUDA_ALIGN"
#endif  // __CUDACC__

// Exception
#define _FLUID_CUDA_CHECK(result, msg, file, line)                                                                                                     \
    if (result != cudaSuccess)                                                                                                                         \
    {                                                                                                                                                  \
        fprintf(stderr, "CUDA error at %s:%d code=%d (%s) \"%s\" \n", file, line, static_cast<unsigned int>(result), cudaGetErrorString(result), msg); \
        cudaDeviceReset();                                                                                                                             \
        exit(EXIT_FAILURE);                                                                                                                            \
    }

#define FLUID_CUDA_CHECK(expression) \
    _FLUID_CUDA_CHECK((expression), #expression, __FILE__, __LINE__)

#define FLUID_CUDA_CHECK_LAST_ERROR(msg) \
    _FLUID_CUDA_CHECK(cudaGetLastError(), msg, __FILE__, __LINE__)
