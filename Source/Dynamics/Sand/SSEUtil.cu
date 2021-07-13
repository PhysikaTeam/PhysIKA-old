//#include "SSEUtil.h"
//
//
//__host__ __device__ void  getSphereAngleInfo(float3 relp, float& cosphi, float& sinphi, float& costheta, float& sintheta)
//{
//	float r = sqrtf(relp.x*relp.x + relp.y*relp.y + relp.z*relp.z);
//	float rxy = sqrtf(relp.x*relp.x + relp.y * relp.y);
//	sinphi = rxy < 1e-6 ? 0 : relp.y / rxy;
//	cosphi = rxy < 1e-6 ? 1 : relp.x / rxy;
//	sintheta = r < 1e-6 ? 0 : rxy / r;
//	costheta = r < 1e-6 ? 1 : relp.z / r;
//}
//
//__global__ void gpuSumation(float* resv, float * vdata, int ngroup, int groupsize)
//{
//	int tid = threadIdx.x + blockIdx.x * blockDim.x;
//	int nthread = blockIdx.x * blockDim.x;
//	__shared__ float sharedData[1024];
//
//	for (int gi = 0; gi < groupsize;++gi)
//	{
//		int i = tid;
//		sharedData[threadIdx.x] = 0;
//
//		while (i < ngroup)
//		{
//			sharedData[threadIdx.x] += vdata[i*groupsize + gi];
//			i += nthread;
//		}
//
//		__syncthreads();
//
//		int curn = blockDim.x;
//		int halfn = (curn + 1) >> 1;
//		while (curn>1)
//		{
//			if (threadIdx.x < halfn && threadIdx.x + halfn < curn)
//			{
//				sharedData[threadIdx.x] += sharedData[threadIdx.x + halfn];
//			}
//			__syncthreads();
//			curn = halfn;
//			halfn = (curn + 1) >> 1;
//		}
//
//		if (threadIdx.x == 0)
//		{
//			resv[blockIdx.x] = sharedData[0];
//		}
//		__syncthreads();
//	}
//
//}
//
//__global__ void gpuSimpleMul(float * resv, float * mat1, float * mat2, int n, int k, int m)
//{
//	int tidx = threadIdx.x;
//	int tidy = threadIdx.y;
//	if (tidx < n && tidy < m)
//	{
//		float val = 0;
//		for (int i = 0; i < k; ++i)
//			val += mat1[tidx * k + i] * mat2[i * m + tidy];
//		resv[tidx * m + tidy] = val;
//	 }
//}
//
//__host__ __device__ float SHfunc00(float sinphi, float cosphi, float sintheta, float costheta)
//{
//	return 0.28209479f; // sqrt(1/4/pi);
//}
//
//__host__ __device__ float SHfunc11_(float sinphi, float cosphi, float sintheta, float costheta)
//{
//	// sqrt(3/4/pi) *sin(phi) * sin(theta)
//	return 0.48860251f * sinphi * sintheta;
//}
//
//__host__ __device__ float SHfunc10(float sinphi, float cosphi, float sintheta, float costheta)
//{
//	// sqrt(3/4/pi) * cos(theta)
//	return 0.48860251f * costheta;
//}
//
//__host__ __device__ float SHfunc11(float sinphi, float cosphi, float sintheta, float costheta)
//{
//	// sqrt(3/4/pi) *sin(phi) * cos(theta)
//	return 0.48860251f * cosphi * sintheta;
//}
//
//__host__ __device__ float SHfunc22_(float sinphi, float cosphi, float sintheta, float costheta)
//{
//	// sqrt(15/4/pi) * sin(phi)*cos(phi) * sin(theta)*sin(theta)
//	return 1.0925484f * sinphi*cosphi * sintheta*sintheta;
//}
//
//__host__ __device__ float SHfunc21_(float sinphi, float cosphi, float sintheta, float costheta)
//{
//	// sqrt(15/4/pi) * sin(phi)* sin(theta)*cos(theta)
//	return 1.0925484f * sinphi * sintheta*costheta;
//}
//
//__host__ __device__ float SHfunc20(float sinphi, float cosphi, float sintheta, float costheta)
//{
//	// sqrt(5/16/pi) * (3*cos(theta)*cos(theta)-1)
//	return 0.3153915f *(3 * costheta*costheta - 1);
//}
//
//__host__ __device__ float SHfunc21(float sinphi, float cosphi, float sintheta, float costheta)
//{
//	// sqrt(15/4/pi) * cos(phi)* sin(theta)*cos(theta)
//	return 1.0925484f * cosphi * sintheta*costheta;
//}
//
//__host__ __device__ float SHfunc22(float sinphi, float cosphi, float sintheta, float costheta)
//{
//	// sqrt(15/16/pi) * (cos(phi) *cos(phi) - sin(phi)*sin(phi))* sin(theta)*sin(theta)
//	return 1.0925484f *0.5f * (cosphi *cosphi - sinphi*sinphi)* sintheta*sintheta;
//}