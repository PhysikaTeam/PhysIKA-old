#pragma once
#include <cuda_runtime.h>
//#include <helper_cuda.h>
//#include <helper_math.h>

#include <assert.h>
#include <omp.h>
#include <string>

//// error handler
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T result, char const* const func, const char* const file,
	int const line) {
	if (result) {
		fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
			static_cast<unsigned int>(result), cudaGetErrorName(result), func);
		exit(EXIT_FAILURE);
	}
}

#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)
inline void __getLastCudaError(const char* errorMessage, const char* file,
	const int line) {
	cudaError_t err = cudaGetLastError();

	if (cudaSuccess != err) {
		fprintf(stderr,
			"%s(%i) : getLastCudaError() CUDA error :"
			" %s : (%d) %s.\n",
			file, line, errorMessage, static_cast<int>(err),
			cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}


#define VLST_BEGIN(lstIdx, lstData, idd) \
	{int vst = (idd == 0) ? 0 : lstIdx[idd-1];\
	int vnum = lstIdx[idd] - vst;\
	for (int vi=0; vi<vnum; vi++) {\
		int vid = lstData[vi+vst];\

#define VLST_END }}

#define FLST_BEGIN(lstIdx, lstData, idd) \
	{int fst = (idd == 0) ? 0 : lstIdx[idd-1];\
	int fnum = lstIdx[idd] - fst;\
	for (int fi=0; fi<fnum; fi++) {\
		int fid = lstData[fi+fst];\

#define FLST_END }}

void  reportMemory(char*);

///////////////////////////////////////////////////////
// show memory usage of GPU

#define BLOCK_DIM 64

inline int BPG(int N, int TPB)
{
	int blocksPerGrid = (N + TPB - 1) / (TPB);
	if (blocksPerGrid > 65536) {
		printf("TM: blocksPerGrid is larger than 65536, aborting ... (N=%d, TPB=%d, BPG=%d)\n", N, TPB, blocksPerGrid);
		exit(0);
	}

	return blocksPerGrid;
}

inline int BPG(int N, int TPB, int& stride)
{
	int blocksPerGrid = 0;

	do {
		blocksPerGrid = (N + TPB * stride - 1) / (TPB * stride);
		if (blocksPerGrid <= 65536)
			return blocksPerGrid;

		stride *= 2;
	} while (1);

	assert(0);
	return 0;
}

#include "cuda_occupancy.h"
extern cudaDeviceProp deviceProp;

inline int evalOptimalBlockSize(cudaFuncAttributes attribs, cudaFuncCache cachePreference, size_t smemBytes) {
	cudaOccDeviceProp prop = deviceProp;
	cudaOccFuncAttributes occAttribs = attribs;
	cudaOccDeviceState occCache;

	switch (cachePreference) {
	case cudaFuncCachePreferNone:
		occCache.cacheConfig = CACHE_PREFER_NONE;
		break;
	case cudaFuncCachePreferShared:
		occCache.cacheConfig = CACHE_PREFER_SHARED;
		break;
	case cudaFuncCachePreferL1:
		occCache.cacheConfig = CACHE_PREFER_L1;
		break;
	case cudaFuncCachePreferEqual:
		occCache.cacheConfig = CACHE_PREFER_EQUAL;
		break;
	default:
		;	///< should throw error
	}

	int minGridSize, blockSize;
	cudaOccMaxPotentialOccupancyBlockSize(
		&minGridSize, &blockSize, &prop, &occAttribs, &occCache, 0, smemBytes);
	return blockSize;
}

#define LEN_CHK(l) \
    int idx = blockDim.x * blockIdx.x + threadIdx.x;\
	if (idx >= l) return;

#define BLK_PAR(l) \
   int T = BLOCK_DIM; \
    int B = BPG(l, T);

#define BLK_PAR2(l, s) \
   int T = BLOCK_DIM; \
    int B = BPG(l, T, s);

#define BLK_PAR3(l, s, n) \
   int T = n; \
    int B = BPG(l, T, s);

#define cutilSafeCall checkCudaErrors

#define M_PI       3.14159265358979323846
#define M_SQRT2    1.41421356237309504880

#include <map>

typedef std::map<void*, int> FUNC_INT_MAP;
static  FUNC_INT_MAP blkSizeTable;

inline int getBlkSize(void* func)
{
	FUNC_INT_MAP::iterator it;

	it = blkSizeTable.find(func);
	if (it == blkSizeTable.end()) {
		cudaFuncAttributes attr;
		cudaFuncGetAttributes(&attr, func);
		int num = evalOptimalBlockSize(attr, cudaFuncCachePreferL1, 0);
		blkSizeTable[func] = num;
		return num;
	}
	else {
		return it->second;
	}
}