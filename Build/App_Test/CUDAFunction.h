#ifndef __CUDAFUNCTION_H__
#define __CUDAFUNCTION_H__

#include "Physika_Core/Cuda_Array/Array.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "Physika_Core/Utilities/cuda_utilities.h"

using namespace Physika;
class CUDAFunction
{
public:
	CUDAFunction() {};
	~CUDAFunction() {};

	static void CopyData(float3* dst, float4* color, Array<float3> posArr, Array<float> indexArr);
};

#endif