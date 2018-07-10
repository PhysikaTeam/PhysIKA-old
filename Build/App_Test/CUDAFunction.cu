#include "CUDAFunction.h"


__global__ void K_SetupRendering(float3* dst, float4* color, Array<float3> posArr, Array<float> indexArr)
{
	int pId = threadIdx.x + (blockIdx.x * blockDim.x);
	if (pId >= posArr.Size()) return;

	dst[pId] = posArr[pId];

	float a = 0.0f;
	float b = 0.0f;
	if (indexArr[pId] > 0.0f)
		a = indexArr[pId] / 1000.0f;
	if (a > 1.0f)
		a = 1.0f;

	if (indexArr[pId] < 0.0f)
		b = -indexArr[pId] / 1000.0f;
	if (b > 1.0f)
		b = 1.0f;
	color[pId] = indexArr[pId] > 0.0f ? make_float4(1.0 - a, 1.0f, 0.0f, 1.0f) : make_float4(1.0f, 1.0 - b, 0.0f, 1.0f);

	//		color[pId] = colorIndex[pId] > 0.0f ? make_float4(0.0f, 1.0f, 0.0f, 1.0f) : make_float4(1.0f, 0.0f, 0.0f,1.0f);

	//		printf("%f \n", colorIndex[pId]);
}

void CUDAFunction::CopyData(float3* dst, float4* color, Array<float3> posArr, Array<float> indexArr)
{
	dim3 pDims = int(ceil(posArr.Size() / BLOCK_SIZE + 0.5f));
	K_SetupRendering << <pDims, BLOCK_SIZE >> > (dst, color, posArr, indexArr);
}

