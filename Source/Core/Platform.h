#pragma once
#define PHYSIKA_VERSION 2.2.1
#define PHYSIKA_VERSION_MAJOR 2
#define PHYSIKA_VERSION_MINOR 2
#define PHYSIKA_VERSION_PATCH 1


#if ((defined _WIN32) || (defined(__MINGW32__) || defined(__CYGWIN__))) && defined(_DLL)
#if !defined(PHYSIKA_DLL) && !defined(PHYSIKA_STATIC)
#define PHYSIKA_DLL
#endif
#endif

#if ((defined _WIN32) || (defined(__MINGW32__) || defined(__CYGWIN__))) && defined(PHYSIKA_DLL)
#define PHYSIKA_EXPORT __declspec(dllexport)
#define PHYSIKA_IMPORT __declspec(dllimport)
#else
#define PHYSIKA_EXPORT
#define PHYSIKA_IMPORT
#endif

#if defined(PHYSIKA_API_COMPILE)
#define PHYSIKAApi PHYSIKA_EXPORT
#else
#define PHYSIKAAPI PHYSIKA_IMPORT
#endif

#define PHYSIKA_COMPILER_CUDA

#if(defined(PHYSIKA_COMPILER_CUDA))
#include <cuda_runtime.h>
#	define COMM_FUNC __device__ __host__ 
#	define GPU_FUNC __device__ 
#	define CPU_FUNC __host__ 
#else
#	define COMM_FUNC
#	define GPU_FUNC 
#	define CPU_FUNC 
#endif

enum DeviceType
{
	CPU,
	GPU,
	UNDEFINED
};

#define PRECISION_FLOAT

#ifdef PRECISION_FLOAT
typedef float Real;
#else
typedef double Real;
#endif

//#define SIMULATION2D
