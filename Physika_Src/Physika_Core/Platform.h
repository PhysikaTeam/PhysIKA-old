#pragma once
#define NEOPHYSICS_VERSION "0.1.0"
#define NEOPHYSICS_VERSION_MAJOR "0"
#define NEOPHYSICS_VERSION_MINOR "1"
#define NEOPHYSICS_VERSION_PATCH "0"


#if ((defined _WIN32) || (defined(__MINGW32__) || defined(__CYGWIN__))) && defined(_DLL)
#if !defined(NEO_DLL) && !defined(NEO_STATIC)
#define NEO_DLL
#endif
#endif

#if ((defined _WIN32) || (defined(__MINGW32__) || defined(__CYGWIN__))) && defined(NEO_DLL)
#define NEO_EXPORT __declspec(dllexport)
#define NEO_IMPORT __declspec(dllimport)
#else
#define NEO_EXPORT
#define NEO_IMPORT
#endif

#if defined(NEO_API_COMPILE)
#define NEOApi NEO_EXPORT
#else
#define NEOAPI NEO_IMPORT
#endif

#define NEO_COMPILER_CUDA

#if(defined(NEO_COMPILER_CUDA))
#	define HYBRID_FUNC __device__ __host__ 
#	define GPU_FUNC __device__ 
#	define CPU_FUNC __host__ 
#else
#	define HYBRID_FUNC
#	define GPU_FUNC 
#	define CPU_FUNC 
#endif

enum DeviceType
{
	CPU,
	GPU,
	UNDEFINED
};
// 
#ifdef PRECISION_FLOAT
typedef float Real;
#else
typedef double Real;
#endif

//#define SIMULATION2D