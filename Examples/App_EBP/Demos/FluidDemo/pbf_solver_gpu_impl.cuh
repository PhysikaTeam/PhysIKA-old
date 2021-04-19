#ifndef PBF_SOLVER_GPU_IMPL_
#define PBF_SOLVER_GPU_IMPL_

#include <stdio.h>
#include <math.h>
#include "Demos\FluidDemo\helper_math.h"

#include "thrust\device_ptr.h"
#include "thrust\for_each.h"
#include "thrust\for_each.h"
#include "thrust\iterator\zip_iterator.h"
#include "thrust\sort.h"

#include "pbf_solver_gpu_kernel.cuh"

__constant__ SimParams params;

struct integrate_functor
{
	float deltaTime;
	
	__host__ __device__
		integrate_functor(float deltaTime) : deltaTime(deltaTime) {}

	template<typename Tuple>
	__device__ void operator()(Tuple t)
	{
		float3 posData = thrust::get<0>(t);
		float3 velData = thrust::get<1>(t);

		velData = velData + 0.01f;// +params.acceleation * deltaTime;
		posData = posData + velData * deltaTime;

		thrust::get<0>(t) = posData;
		thrust::get<1>(t) = velData;
	}
};
#endif // !PBF_SOLVER_GPU_IMPL_
