#pragma once
#include <cstdlib>
#include <cstdio>
#include <string.h>

#include <Windows.h> // 避免<Gl/gl.h>提示错误 GL/gl.h(1190): error : variable "WINGDIAPI" has already been defined

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "device_launch_parameters.h"
#include "cuda.h"

#include "thrust\device_ptr.h"
#include "thrust\for_each.h"
#include "thrust\for_each.h"
#include "thrust\iterator\zip_iterator.h"
#include "thrust\sort.h"

//#include "helper_math.h"
#include "Demos\FluidDemo\helper_math.h"
//#include "pbf_solver_gpu_kernel.cuh"
#include "pbf_solver_gpu_impl.cuh"
#include "helper_cuda.h"

//include the function of kernel
//#include "PositionBasedDynamics\SPHKernels.h"
#include "Demos\FluidDemo\pbf_SPH_Kernel_W.cuh"
//#include "build\Demos\FluidDemo\kernel.cuh"
// describe about the gridParticleHash, gridParticleIndex, cellStart, cellEnd
//
//example (particleNum = 6, particle index is |0 1 2 3 4 5|
//         cellNum = 16, cell index(hash value) is |0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15|
//         the no. of cells containing particles:|4,6,9|)
// original ---> gridParticleHash:|9 6 6 4 6 4|, sorted by hash-> gridParticleHash:|4 4 6 6 6 9|
// original --->gridParticleIndex:|0 1 2 3 4 5|, sorted by hash->gridParticleIndex:|3 5 1 2 4 0|
// cellStart[4] = 0, cellstart[6] = 2, cellSatrt[9] = 5 , orthers cellStart[0,1,2,3,5,7,8,10,11,12,13,14,15] = 0xffffffff
//   cellEnd[4] = 2,   cellEnd[6] = 5,   cellEnd[9] = 6
//
// 获取粒子属于哪个网格，|cell_sz = 2 * particleRadius|

//向量点积运算
__device__ float DotProduct(float3 v1, float3 v2)
{
	float result;
	result = v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
	return result;
}

__device__ float Sigmoid(float x)
{
	float result;
	result = 1.0 / (1.0 + exp(-x));
	return result;
}
__device__ int3 GetCell(float3 pos, float cell_sz)
{
	int3 cellPos;
	//const float cell_sz_recpr = 1.0 / cell_sz;
	const float cell_sz_recpr = 1.0/params.cellSize;
	cellPos.x = (floor)((pos.x-(params.gridOrigin.x))*cell_sz_recpr);
	cellPos.y = (floor)((pos.y-(params.gridOrigin.y))*cell_sz_recpr);
	cellPos.z = (floor)((pos.z-(params.gridOrigin.z))*cell_sz_recpr);
	return cellPos;
}

__device__ bool IsCellInRange(int3 cell)
{
	return ((0 <= cell.x && cell.x < params.gridSize.x) &&
		(0 <= cell.y && cell.y < params.gridSize.y) &&
		(0 <= cell.z && cell.z < params.gridSize.z));
}

//返回指定cell的中心位置
__device__ float3 GetCellCenterPos(int3 cell)
{
	float3 centerPos;
	centerPos.x = (cell.x * params.cellSize + params.gridOrigin.x) + 0.5 * params.cellSize;
	centerPos.y = (cell.y * params.cellSize + params.gridOrigin.y) + 0.5 * params.cellSize;
	centerPos.z = (cell.z * params.cellSize + params.gridOrigin.z) + 0.5 * params.cellSize;
	return centerPos;
}
__device__ uint calcGridHash(int3 ptc_cell)
{
	uint hash;
	hash = params.gridSize.y*ptc_cell.z;
	hash = (hash + ptc_cell.y) * params.gridSize.x;
	hash = hash + ptc_cell.x;
	return hash;

}

//根据索引值返回cell的x,y,z坐标
__device__ int3 indexToXYZForCell(unsigned int index)
{
	int3 cell;
	cell.x = index % params.gridSize.x;
	cell.z = ((index - cell.x) / (params.gridSize.x * params.gridSize.y));
	cell.y = ((index - cell.x - params.gridSize.x * params.gridSize.y * cell.z) / params.gridSize.x);
	return cell;
}

//calculate the distance square between two particles
__device__ float DistanceSquare(float3 pos1, float3 pos2)
{
	float x = pos1.x - pos2.x;
	float y = pos1.y - pos2.y;
	float z = pos1.z - pos2.z;
	float result = x*x + y*y + z*z;
	return result;
}


//pbf:Eq(13),解决表面张力不稳定问题
__device__ float computeScorr(float3 &pos_i, float3 &pos_j)
{
	//设置参数，可以调节
	float k = 0.1;
	int n = 4;
	float deltQCoeff = 0.1; //0.1~0.3
	float x = poly6(pos_i, pos_j, params.kernelRadius) / poly6_h(deltQCoeff*params.kernelRadius, params.kernelRadius);
	//float x = W(pos_i, pos_j, params.kernelRadius) / poly6_h(deltQCoeff*params.kernelRadius, params.kernelRadius);
	float result = -1 * k * pow(x, n);
}

//检查给定的点是否与给定cell中的所有的点相交,|如果相交，返回true|
//|gridPos:要检查的单元格|pos_i:被测粒子位置|
__device__ bool collidDetection(int3 gridPos, float3 pos_i, float3 *partilcePos, unsigned int *cellStart, unsigned int *cellEnd, unsigned int *gridParticleIndex)
{
	unsigned int gridHash = calcGridHash(gridPos);
	unsigned int startIndex = cellStart[gridHash];
	bool flag = true;
	if (startIndex != 0xffffffff)
	{
		float collidDist = 2.0 * params.particleRadius;
		unsigned int endIndex = cellEnd[gridHash];
		for (unsigned int j = startIndex; j < endIndex; j++)
		{
			////get the position of particle in the cell
			float3 pos2 = partilcePos[gridParticleIndex[j]];
			float distance = sqrt(DistanceSquare(pos_i, pos2)); //the distance between two particles
			
			if (distance < collidDist)
			{	
				flag = true; 
				break;
			}
		}
	}
	return flag;
}

//检测给的的粒子是否与所有邻域中的粒子有碰撞，如果有返回true
//|detect_pos: 需要检测的粒子|detect_cell: 被测粒子所在的cell|
__device__ bool collidDetectionForNeighborCells(float3 detect_pos, int3 detect_cell, float3 *pos, unsigned int *cellStart, unsigned int *cellEnd, unsigned int *gridParticleIndex)
{
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 cell_nb = detect_cell + make_int3(x, y, z);
				if (IsCellInRange(cell_nb))
				{
					//与给定cell所包含的粒子进行碰撞检测
					bool collidFlag = collidDetection(cell_nb, detect_pos, pos, cellStart, cellEnd, gridParticleIndex);
					if (collidFlag)
					{
						return true;
					}
				}	
			}
		}
	}
	return false;
}

//在给定的cell中找到一个与邻域粒子都不碰撞且在给定粒子pos_i的核范围内的位置
//若找到合适位置则返回，若找不到则返回整个网格最小坐标减1的位置
//pos_i:给定的粒子位置
__device__ float3 findPositionInCell(int3 cell, float3 pos_i,float3 *pos, unsigned int *cellStart, unsigned int *cellEnd, unsigned int *gridParticleIndex)
{
	float3 cellCenter = GetCellCenterPos(cell);
	float3 addPos;
	float step = params.particleRadius / 2; //设置位置搜索的步长
	

	for(float z = cellCenter.z-params.particleRadius; z<=cellCenter.z+params.particleRadius; z+=step)
		for(float y = cellCenter.y-params.particleRadius; y<=cellCenter.y+params.particleRadius; y+=step)
			for (float x = cellCenter.x - params.particleRadius; x <= cellCenter.x + params.particleRadius; x += step)
			{
				addPos.x = x;
				addPos.y = y;
				addPos.z = z;
				float3 vector; 
				vector.x = x - pos_i.x;
				vector.y = y - pos_i.y;
				vector.z = z - pos_i.z;
				float dist = vector.x * vector.x + vector.y * vector.y + vector.z * vector.z;
				/*float dist = sqrt(DistanceSquare(pos_i, addPos));*/
				if (dist > params.kernelRadius * params.kernelRadius) continue;
				return addPos;

				//进行冲突检测
				/*bool flag = collidDetectionForNeighborCells(addPos, cell, pos, cellStart, cellEnd, gridParticleIndex);
				if (!flag)
				{
					return addPos;
				}*/
			}
	//没有找到合适位置的情况
	return make_float3(params.gridOrigin.x - 1, params.gridOrigin.y - 1, params.gridOrigin.z - 1);
}

//计算权重函数，参考Target Particle Control of Smoke Simulation，eq.2
//---------------zzl--------------2019-1-23
__device__ float weightForParticleControl(float3 p1, float3 p2, float r)
{
	float w = 0.0;
	float d = DistanceSquare(p1, p2);
	float w1 = 0.0;
	if (sqrt(d) < r)
	{
		w1 = (r*r - d);
		w = w1 * w1 * w1;
	}
	return w;
}

// rearrange particle data into sorted order
//find the start of each cell in the sorted hash array
__global__ void reorderDataAndFindCellStartKernel(uint *cellSatrt,        //output: cell start index
												  uint *cellEnd,          //output: cell end index , cellEnd-cellSatrt = cell中的粒子个数
												  float3 *sortedPos,    //output: sorted positions
												  float3 *sortedVel,    //output: sorted velocities
	                                              uint *gridParticleHash,  //input: sorted grid hashes
	                                              uint *gridParticleIndex, //input: sorted particle indices
	                                              float3 *oldPos,        //
	                                              float3 *oldVel,
	                                              int numParticles)
{
	extern __shared__ uint sharedHash[]; //共享内存空间，可以由同一个线程块中的线程共享访问
	uint index = blockDim.x * blockIdx.x + threadIdx.x;
	uint hash;
	if (index < numParticles)
	{
		hash = gridParticleHash[index];
		//load has data into shared memory so that we can look 
		//at neighboring  particle's hash value without loading 
		//two hash values per thread
		//gridParticleHash[index] --> sharedHash[index + 1]
		sharedHash[threadIdx.x + 1] = hash;

		if (index > 0 && threadIdx.x == 0)
		{
			// first thread in block must load neighbor particle (in the previous block) hash
			sharedHash[0] = gridParticleHash[index - 1];
		}
	}
	__syncthreads(); //进行多线程间的同步

	if (index < numParticles)
	{
		//index ==0 : the first particle
		// hash != sharedHash[threadIdx.x] : 连续两个粒子所对应的hash值不同，表面这两个粒子存储在不同的cell中
		if (index ==0 || hash != sharedHash[threadIdx.x]) 
		{
			// set the cellStart
			cellSatrt[hash] = index;
			// set the cellEnd
			if (index > 0)
			{
				cellEnd[sharedHash[threadIdx.x]] = index; // 最后的索引值为下一个新hash值得开始位置
			}
		}
		//处理最后一个粒子
		if (index == numParticles - 1)
		{
			cellEnd[hash] = index + 1;
		}

		////use the sorted index to reorder the pos and vel data
		uint sortedIndex = gridParticleIndex[index];
		sortedPos[index] = oldPos[sortedIndex];
		sortedVel[index] = oldVel[sortedIndex];

	}


}

//calculate the hash for each particel(也就是粒子所在的cell编号)
//gridParticleHash[粒子的编号] = 粒子所在网格的编号（hash值）
//gridParticleIndex[粒子的编号] = 粒子的编号（在根据hash值排序时，配合gridParticleHash，获得粒子的编号）
__global__ void calHashKernel(uint *gridParticleHash, uint *gridParticleIndex, float3 *pos, int numParticles)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (index >= numParticles) return;
	float3 p_i = pos[index];
	int3 ptc_cell = GetCell(p_i, params.cellSize);
	uint hash = calcGridHash(ptc_cell);
	if (hash >= 0 && hash < params.numGridCells)
	{
		gridParticleHash[index] = hash;
	}
	//for test whether the value of Hash is larger than the numGridCells
	else
	{
		//printf("%d  ,%d  ,%d   ,%d   ",ptc_cell.x,ptc_cell.y,ptc_cell.z,hash);
	}
	gridParticleIndex[index] = index;
}

//查找邻域的核函数 |Positions: 原始粒子位置| neighbors: 存放邻域粒子编号 | cellGridSize:网格间距长度 (2*粒子半径)|numParticles: 总粒子数|
//|radius: 粒子半径|kernelRadius: 核函数的支持域半径|num_cell_dim：网格的维数|
__global__ void GpuNeighborSearchKernel(float3 *Positions, unsigned int *numNeighbors, unsigned int * neighbors, int numParticles, float radius,float kernelRadius, uint *cellStart, uint *cellEnd, uint *gridParticleIndex)
{
	const int ptc_i = (blockIdx.x * blockDim.x) + threadIdx.x; //粒子的索引号0~(numParticles-1)
	if (ptc_i >= numParticles) return;

	int3 ptc_cell = GetCell(Positions[ptc_i], params.cellSize);
	unsigned int neighborNum = 0; // record the amount of neighbors for particle_i
	float3 pos_i = Positions[ptc_i]; //get the posiont of particle_i

	int numCellInKernel = ceil(kernelRadius/params.cellSize); //计算在支持域半径内包含多个单元格
	float dist2 = kernelRadius*kernelRadius;

	for(int cz = -1*numCellInKernel; cz <=numCellInKernel; cz++)
		for(int cy = -1*numCellInKernel; cy <=numCellInKernel; cy++)
			for (int cx = -1*numCellInKernel; cx <= numCellInKernel; cx++)
			{
				int3 nb_cell; //邻域单元格位置
				nb_cell.x = ptc_cell.x + cx;
				nb_cell.y = ptc_cell.y + cy;
				nb_cell.z = ptc_cell.z + cz;

				if (!IsCellInRange(nb_cell)) continue; //判断网格索引是否在有效范围内

				uint hash = calcGridHash(nb_cell); //获取cell的hash值
				uint startIndex = cellStart[hash]; //根据hash值，得到cell中存储的第一个粒子的序号

				if (startIndex != 0xffffffff)
				{
					uint endIndex = cellEnd[hash];
					for (uint i = startIndex; i < endIndex; i++)
					{
						uint originalIndex = gridParticleIndex[i];
						float3 pos_n = Positions[originalIndex];
						float dist = DistanceSquare(pos_i, pos_n);
						if (dist < dist2 && neighborNum < params.maxNumOfNeighbors && originalIndex != ptc_i)
						{
							neighbors[ptc_i * params.maxNumOfNeighbors + neighborNum] = originalIndex;
							neighborNum++;
						}

					}//for(uint i = startIndex; i < endIndex; i++)
				}
			}// endfor endfor endfor
	numNeighbors[ptc_i] = neighborNum;
}

__global__ void computeDensityKernel(double *density, //output
	float3 * position, //input
	double * mass,//input
	unsigned int * numNeighbors,//input 
	unsigned int *neighbors, //input
	unsigned int numParticles) //input
{
	unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index < numParticles)
	{
		float temDensity = 0.0f;
		float3 posI = position[index];
		double massI = mass[index];
		
		//temDensity = massI*poly6_new(posI, posI, params.kernelRadius);
		temDensity = massI*W(posI, posI, params.kernelRadius);
		
		unsigned int neIndex; //邻域粒子的索引
		for (int i = 0; i < numNeighbors[index]; i++)
		{
			neIndex = neighbors[index*params.maxNumOfNeighbors + i];
			//temDensity = temDensity + mass[neIndex] * poly6_new(position[index], position[neIndex],params.kernelRadius);
			temDensity = temDensity + mass[neIndex] * W(position[index], position[neIndex], params.kernelRadius);
		}
		//printf("(%f)  ", temDensity);
		density[index] = temDensity;
	}
}

// compute pressure
__global__ void computePressureKernel(double *pressure,double *density, unsigned int numParticles)
{
	unsigned int index;
	index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < numParticles)
	{
		pressure[index] = params.gasConstantK * (density[index] - params.density0);
	}
}

//compute pressure force
__global__ void computePressureForceKernel(float3 *pressureForce, //output
											double *pressure, //input
											double *mass, //input
											double *density,
											float3 *position, //input
											unsigned int numParticles,//input 
											unsigned int *numNeighbor, //input
											unsigned int *neighbors)//input
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < numParticles)
	{
		float3 tempPressureForce = make_float3(0.0f, 0.0f, 0.0f);
		for (int i = 0; i < numNeighbor[index]; i++)
		{
			unsigned int nbIndex = neighbors[index * params.maxNumOfNeighbors + i];
			float m_k1 = (pressure[index] + pressure[nbIndex]) / (2 * density[nbIndex]); // (p_i+p_j)/2 density_j
			float m_k2 = -1 * mass[nbIndex] * m_k1; //-m_j*m_k1
			//float3 m_k3 = grad_poly6(position[index], position[nbIndex], params.kernelRadius);
			float3 m_k3 = gradW(position[index], position[nbIndex], params.kernelRadius);
			float3 tempResult = make_float3(m_k2 * m_k3.x, m_k2 * m_k3.y, m_k2 * m_k3.z);
			tempPressureForce = make_float3(tempPressureForce.x + tempResult.x,
											tempPressureForce.y + tempResult.y,
											tempPressureForce.z + tempResult.z);			
		}
		pressureForce[index] = tempPressureForce;
	}
	
}

//根据与环境温度差异计算浮力
//F浮 = k * (T - T0)/T0
__global__ void computeBuoyanceByTemperature(float3 *buoyance, double *temperature, float3 *pos, unsigned int numParticle)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	float T_amb = 3.0; //最底部环境温度，为可调节参数
	float T_amb_lapseRate = 0.02;//环境温度随高度的变化率，为可调节参数
	float kb = 1.0; //浮力常数，为调节参数
	if (index < numParticle)
	{
		float3 pos_i = pos[index];
		float h = pos_i.y - params.gridOrigin.y;
		float T0 = T_amb * (1 - h * T_amb_lapseRate);
		float Ti = temperature[index];
		float bu = kb * (Ti - T0) / T0;
		float3 j = make_float3(0, -1, 0); //y轴为垂直向上方向
		buoyance[index] = bu * j;
	}
}

//根据温度梯度计算浮力
//reference: <Modeling and Characterization of Cloud Dynamics>
__global__ void computeBuoyanceByGradientOfTemperatureKernel(float3 *buoyance, //output
																double *temperature, //input
																float3 *position,//input
																double *mass,
																double *density,
																unsigned int numParticles,//input
																unsigned int *numNeighbors,//input
																unsigned int *neighbors)//input
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < numParticles)
	{
		float3 result;
		float3 tempGradinetOfTemperature = make_float3(0.0f,0.0f,0.0f);
		float c = 0.00002;//浮力控制参数
		float3 m_k3 = make_float3(1.0,1.0,1.0);
		for (int i = 0; i < numNeighbors[index]; i++)
		{
			unsigned int nbIndex = neighbors[index * params.maxNumOfNeighbors + i];
			float m_k1 = (temperature[index] + temperature[nbIndex]) / 2.0f;
			float m_k2 = mass[nbIndex] / density[nbIndex];
			//m_k3 = grad_spiky_new(position[index], position[nbIndex], params.kernelRadius);
			m_k3 = gradW(position[index], position[nbIndex], params.kernelRadius);
			//printf("(%f %f %f)", m_k3.x,m_k3.y,m_k3.z);
			float3 tempF = make_float3(m_k1 * m_k2 * m_k3.x, m_k1 * m_k2 * m_k3.y, m_k1 * m_k2 * m_k3.z);
			tempGradinetOfTemperature = make_float3(tempGradinetOfTemperature.x + tempF.x,
													tempGradinetOfTemperature.y + tempF.y,
													tempGradinetOfTemperature.z + tempF.z);
		}
		//printf("(%f %f %f)", tempGradinetOfTemperature.x, tempGradinetOfTemperature.y, tempGradinetOfTemperature.z);
		buoyance[index] = c * tempGradinetOfTemperature;
	}
}

__global__ void updateAccelerationKernel(float3 *acceleration, float3 *buoyance,double *mass, unsigned int numParticles)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < numParticles)
	{
		////考虑重力和浮力
		//float3 graviety = make_float3(0, 0.002f, 0); //重力系数可以修改,(0,0.002,0),
		//float3 sumForce = make_float3(graviety.x + buoyance[index].x, graviety.y + buoyance[index].y, graviety.z + buoyance[index].z);

		//只考虑反馈力
		float3 sumForce = buoyance[index];
		//printf("(%f,%f,%f)  ", buoyance[index].x, buoyance[index].y, buoyance[index].z);
		float a_x = sumForce.x / mass[index];
		float a_y = sumForce.y / mass[index];
		float a_z = sumForce.z / mass[index];
		acceleration[index] = make_float3(a_x, a_y, a_z);
		//printf("(%f,%f,%f)", acceleration[index].x, acceleration[index].y, acceleration[index].z);
	}
}

//相变函数
__global__ void phaseTransitionKernel(double *newCloud, double *newVapor, double *newTemperature, double *temperatureChange, double *oldCloud, double *oldVapor, double *oldTemperature, unsigned int numParticles)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < numParticles)
	{
		//这些参数可以调节 |A|B|C|alpha|Q|
		const double A = 100.0;
		const double B = 3.0;
		const double C = -2.3;
		const double alpha = 0.2; //phase transition rate
		const double Q = 0.2; //latent heat coefficient

		const double vapor_old = oldVapor[index];
		const double cloud_old = oldCloud[index];
		const double temperature_old = oldTemperature[index];

		//计算饱和蒸汽密度
		double satQv; //包含蒸汽密度
		double w1 = A*exp((-1 * B) / (temperature_old + C));
		double w2 = vapor_old + cloud_old;

		if (w1 < w2)
		{
			satQv = w1;
		}
		else
		{
			satQv = w2;
		}

		//更新vapor, cloud, temperature
		double deltaC = alpha * (vapor_old - satQv);
		newCloud[index] = cloud_old + deltaC;
		newVapor[index] = vapor_old - deltaC;

		temperatureChange[index] = -1 * Q * deltaC;
		newTemperature[index] = oldTemperature[index] - Q * deltaC;
	}
}

__global__ void integrateSystemForCloudKernel(float3 *newPos, float3 *newVel, float3 *oldPos, float3 *oldVel, float3 *acceleration, float deltaTime, uint numParticle)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < numParticle)
	{
		float v_x = oldVel[index].x + acceleration[index].x * deltaTime;
		float v_y = oldVel[index].y + acceleration[index].y * deltaTime;
		float v_z = oldVel[index].z + acceleration[index].z * deltaTime;
		
		float p_x = oldPos[index].x + v_x*deltaTime;
		float p_y = oldPos[index].y + v_y*deltaTime;
		float p_z = oldPos[index].z + v_z*deltaTime;

		newPos[index] = make_float3(p_x, p_y, p_z);
		newVel[index] = make_float3(v_x, v_y, v_z);
 	}
}

//compute lambda for each particel (eq.11)
__global__ void computeLambdaKernel(double *lambda, double *density, double *mass, float3 *pos, unsigned int *numNeighbor, unsigned int *neighbor, unsigned int numParticle)
{
	unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index < numParticle)
	{
		double C_i = density[index] / params.density0 - 1.0f;
		if (C_i < 0.0)
			C_i = 0.0;
		float sumGradient = 0.0f;
		float3 gradientI = make_float3(0.0f, 0.0f, 0.0);
		float3 positionI = pos[index];
		for (int i = 0; i < numNeighbor[index]; i++)
		{
			unsigned int nbIndex = neighbor[index * params.maxNumOfNeighbors + i];
			float3 positionJ = pos[nbIndex];
			//float3 gradientJ = -1.0 *mass[nbIndex]/params.density0 * grad_spiky(positionI, positionJ, params.kernelRadius);
			float3 gradientJ = -1.0 *mass[nbIndex] / params.density0 * gradW(positionI, positionJ, params.kernelRadius);
			sumGradient = sumGradient + gradientJ.x*gradientJ.x + gradientJ.y*gradientJ.y + gradientJ.z*gradientJ.z;

			gradientI.x = gradientI.x - gradientJ.x;
			gradientI.y = gradientI.y - gradientJ.y;
			gradientI.z = gradientI.z - gradientJ.z;
		}
		sumGradient = sumGradient + gradientI.x*gradientI.x+ gradientI.y*gradientI.y+ gradientI.z*gradientI.z;
		double epsilon = 1.0e-6;
		double lambdaI = -1.0f * C_i / (sumGradient + epsilon);
		if (C_i != 0.0)
		{
			lambda[index] = lambdaI;
		}
		else
		{
			lambda[index] = 0.0;
		}
		
	}
	
}

//pbf: Eq(12)
__global__ void computeDeltaPositionKernel(float3 *deltaPos, float3 *pos, double *lambda, double *mass, double *density, unsigned int *numNeighbor, unsigned int *neighbor, unsigned int numParticle)
{
	unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index < numParticle)
	{
		double lambda_i = lambda[index];
		float3 pos_i = pos[index];
		float3 delt_pos = make_float3(0.0f, 0.0f, 0.0f);
		for (int i = 0; i < numNeighbor[index]; i++)
		{
			unsigned int nbIndex = neighbor[index*params.maxNumOfNeighbors + i];
			double lambda_j = lambda[nbIndex];
			float3 pos_j = pos[nbIndex];
			//delt_pos = delt_pos + (lambda_i+lambda_j)* (mass[nbIndex]/density[nbIndex]) *grad_spiky(pos_i, pos_j, params.kernelRadius);
			delt_pos = delt_pos + (lambda_i + lambda_j)* (mass[nbIndex] / params.density0) *gradW(pos_i, pos_j, params.kernelRadius);
		}
		//delt_pos = delt_pos / params.density0;
		deltaPos[index] = delt_pos;
	}
}

//update position according to delta_Position
__global__ void updatePositionKernel(float3 *pos, float3 *deltaPos, unsigned int numParticle)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < numParticle)
	{
		float3 temPos = pos[index] + deltaPos[index];
		pos[index] = temPos;
	}
}

//update velocity
__global__ void updateVelocityKernel(float3 *vel, float3 *newPos, float3 *oldPos, double deltime, unsigned int numParticle)
{
	unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index < numParticle)
	{
		float3 old_pos = oldPos[index];
		float3 new_pos = newPos[index];
		float3 new_vel = (new_pos - old_pos) / deltime;
		vel[index] = new_vel;
	}
}

//根据密度设置flage，|高于上限设置为1， 低于下限设置为-1， 其他设置为0|
__global__ void checkParticleByDensityKernel(int *flag, float3 *pos, double *density, double densityK, unsigned int numParticle)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < numParticle)
	{
		double density_i = density[index];
		//设置密度的上下限
		double upDensity = params.density0 * (1 + densityK);
		double lowDensity = params.density0 * (1 - densityK);
		
		if (density_i < lowDensity)
		{
			flag[index] = -1;
		}
		else if (density_i > upDensity)
		{
			flag[index] = 1;
		}
		else
		{
			flag[index] = 0;
		}
	}
}

//根据密度和位置设置flage，|高于上限设置为(from cloud to vapor)1， 低于下限设置为(from vapor to cloud)-1， 其他设置为0|
__global__ void checkParticleByDensityAndLocationKernel(int *flag, float3 *pos, double *density, double densityK, double *singDistance, unsigned int *numNeighbor, unsigned int numParticle)
{
	unsigned int index = blockDim.x*blockIdx.x + threadIdx.x;
	int minN = 50;
	int maxN = 3;
	if (index < numParticle)
	{
		double density_i = density[index];
		double signDistance_i = singDistance[index];
		//设置密度的上下限
		double upDensity = params.density0 * (1 + densityK);
		double lowDensity = params.density0 * (1 - densityK);

		flag[index] = 0;
		//test for the phase transition from vapor to cloud
		//在目标模型内，同时密度低于rest density
		if (density_i < lowDensity && signDistance_i < -1*params.particleRadius*2 && numNeighbor[index] < minN && numNeighbor[index]>maxN)
		{
			flag[index] = -1;
		}

		//test for the phase transition from cloud to vapor
		//由于粒子的聚集，导致局部具有高密度，通过相变减少粒子数
		if (density_i>upDensity)  
		{
			flag[index] = 1;
		}
		//test for the phase transition from cloud to vapor
		//在仿真云的边界上，通过相变将边界上的粒子删除掉，实现形状的匹配
		if ((density_i < lowDensity && signDistance_i > 0+params.particleRadius*2)  || (signDistance_i > 0 + params.particleRadius * 2 && numNeighbor[index]<maxN))
		{
			flag[index] = 2;
		}	
	}
}

//add particles
__global__ void addParticleKernel(float3 *addPos, unsigned int *addNum, float3 *pos, int *flag, unsigned int *cellstart, unsigned int *cellEnd, unsigned int *gridParticleIndex, double *density, double *mass,unsigned int numParticle)
{
	unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < numParticle)
	{
		unsigned int add_num = 0; //添加粒子的个数
		double add_density = 0; //添加粒子后对粒子index的密度贡献总量，用于限制密度增加过大
		if (-1 == flag[index])
		{
			//通过相变而增加的粒子的位置和质量
			float3 add_Pos;
			double mass_add; //新增粒子的质量
			mass_add = mass[index]; //将新增粒子的质量设置为与中心粒子相同

			float3 pos_i = pos[index];
			int3 cell_i = GetCell(pos_i, params.cellSize);
			int numCellInKernel = ceil(params.kernelRadius / params.cellSize);
			double dist2 = params.kernelRadius * params.kernelRadius;
			for (int cz = -1 * numCellInKernel; cz <= numCellInKernel; cz++)
			{
				for (int cy = -1 * numCellInKernel; cy <= numCellInKernel; cy++)
				{
					for (int cx = -1 * numCellInKernel; cx <= numCellInKernel; cx++)
					{
						int3 cell_nb = make_int3(cell_i.x + cx, cell_i.y + cy, cell_i.z + cz);
						if (!IsCellInRange(cell_nb)) continue;

						uint hash = calcGridHash(cell_nb);
						uint startIndex = cellstart[hash];
						//在cell中存在粒子
						if (startIndex != 0xffffffff) 
						{
							//cell中只包含一个粒子时可以考虑其内部增加粒子
							//uint endIndex = cellEnd[hash];
							////只存在一个粒子的情况
							//if ((endIndex - startIndex) == 1)
							//{
							//	add_Pos = findPositionInCell(cell_nb, pos_i, pos, cellstart, cellEnd, gridParticleIndex);
							//	if (add_Pos.x >= params.gridOrigin.x && add_Pos.y >= params.gridOrigin.y && add_Pos.z >= params.gridOrigin.z)
							//	{	
							//		double tempDensity = poly6_new(pos_i, add_Pos, params.kernelRadius);
							//		if ((add_density + tempDensity)< (density[index] - params.density0))
							//		{
							//			addPos[index*params.maxNumOfNeighbors + add_num] = add_Pos;
							//			add_density += tempDensity;
							//			add_num++;
							//		}
							//	}
							//}
						}
						//在cell中不存在粒子时，增加粒子
						else
						{
							add_Pos = findPositionInCell(cell_nb, pos_i, pos, cellstart, cellEnd, gridParticleIndex);
							if (add_Pos.x >= params.gridOrigin.x && add_Pos.y >= params.gridOrigin.y && add_Pos.z >= params.gridOrigin.z && add_num<15)
							{	
								double tempDensity = poly6_new(pos_i, add_Pos, params.kernelRadius);
								if ((add_density + tempDensity)< abs((density[index] - params.density0)))
								{
									addPos[index*params.maxNumOfNeighbors + add_num] = add_Pos;
									add_density += tempDensity;
									add_num++;
								}
							}
						}
					}//end for cx
				}//end for cy
			}//end for cz
		}//end if -1 == flag[index]
		addNum[index] = add_num; //存储增添的粒子个数
	}//end if index
}

//delete particle
__global__ void deleteParticleKernel(uint *deleteFlag, int *flag, float3 *pos, double *mass, double *density, uint *neighbor, uint *neighborNum, uint numParticle)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	double deltaDis = 1.0e-5; 
	double testDis = params.particleRadius * 2 - deltaDis;//设置距离阈值，判断两个粒子是否距离太近，若太近需要删除掉其中一个
	if (index < numParticle)
	{
		if (1 == flag[index])
		{
			double deleteDensity = 0;
			float3 pos_i = pos[index];
			for (int i = 0; i < neighborNum[index]; i++)
			{
				if (1 == deleteFlag[neighbor[index * params.maxNumOfNeighbors + i]]) continue; //若该粒子已被删除则不进行任何操作

				float3 pos_1 = pos[neighbor[index * params.maxNumOfNeighbors + i]]; //取位置
				double mass_1 = mass[neighbor[index * params.maxNumOfNeighbors + i]]; //取质量
				
				for (int j = i + 1; i < neighborNum[index]; i++)
				{
					if (1 == deleteFlag[neighbor[index * params.maxNumOfNeighbors + j]]) continue; //若该粒子已被删除则不进行任何操作

					float3 pos_2 = pos[neighbor[index * params.maxNumOfNeighbors + j]];   //取位置
					double mass_2 = mass[neighbor[index * params.maxNumOfNeighbors + j]]; //取质量
					//compute the square distance between pos_1 and pos_2
					double dis12 = DistanceSquare(pos_1, pos_2);
					if (dis12 < testDis)
					{
						double deleteDensity_1 = mass_1 * poly6_new(pos_1, pos_i, params.kernelRadius);
						double deleteDensity_2 = mass_2 * poly6_new(pos_1, pos_i, params.kernelRadius);
						if ((deleteDensity + deleteDensity_1) < (params.density0 - density[index]))
						{
							deleteDensity = deleteDensity + deleteDensity_1;
							deleteFlag[neighbor[index * params.maxNumOfNeighbors + i]] = 1;
						}
						else if ((deleteDensity + deleteDensity_2) < (params.density0 - density[index]))
						{
							deleteDensity = deleteDensity + deleteDensity_2;
							deleteFlag[neighbor[index * params.maxNumOfNeighbors + j]] = 1;
						}
					}//end if (dis12 < testDis)
				}//end for j
			}//end for i
		} //end if (1 == flag[index])
	}//end if(index < numParticle)

}

// addPos：记录可以添加粒子的位置
// addPosFlag:记录是否可以添加粒子的标记， |1=可以添加|0=不可以添加|
__global__ void findPosForAddParticleKernel(float3 *addPos, uint *addPosFlag, float3 *pos, uint * cellStart, uint * cellEnd, uint *gridParticleIndex)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < params.numGridCells)
	{
		//初始化addPosFlag和addPos
		addPosFlag[index] = 0;
		addPos[index] = make_float3(params.gridOrigin.x - 1, params.gridOrigin.y - 1, params.gridOrigin.z - 1);
		int3 cell = indexToXYZForCell(index);
		float3 cellCenter = GetCellCenterPos(cell); //得到cell的中心坐标
		float3 addPos_i;
		float maxLength = 9999.0f;
		float step = params.particleRadius / 2; //设置位置搜索的步长
		for (float z = cellCenter.z - params.particleRadius; z <= cellCenter.z + params.particleRadius; z += step)
			for (float y = cellCenter.y - params.particleRadius; y <= cellCenter.y + params.particleRadius; y += step)
				for (float x = cellCenter.x - params.particleRadius; x <= cellCenter.x + params.particleRadius; x += step)
				{
					addPos_i.x = x;
					addPos_i.y = y;
					addPos_i.z = z;
					bool flag = collidDetectionForNeighborCells(addPos_i, cell, pos, cellStart, cellEnd, gridParticleIndex);
					float lengthToCellCenter_i = DistanceSquare(addPos_i, cellCenter);
					if (!flag && lengthToCellCenter_i < maxLength)
					{
						if (addPosFlag[index] == 0)
						{
							addPosFlag[index] = 1;
						}
						addPos[index] = addPos_i;
						maxLength = lengthToCellCenter_i;
					}
				}//end for
	}//end if (index < params.numGridCells)
}

/*计算color field 梯度的核函数， the gradient of the color field 可用于是否为表面粒子 */
__global__ void computeGradientOfColoFieldKernel(float3 *gradientOfColorField, double *mass, double *density, float3* pos, unsigned int numParticle, unsigned int *numNeighbor, uint *neighbor)
{
	unsigned int index = blockDim.x*blockIdx.x + threadIdx.x;
	if (index < numParticle)
	{
		float3 pos_i = pos[index];
		unsigned int neighborNum_i = numNeighbor[index];
		float3 tempN=make_float3(0,0,0);
		for (int i = 0; i < neighborNum_i; i++)
		{
			int index_j = neighbor[index*params.maxNumOfNeighbors + i];
			float3 pos_j = pos[index_j];
			double mass_j = mass[index_j];
			double density_j = density[index_j];
			//tempN += 1*(mass_j / density_j*grad_spiky_new(pos_i, pos_j, params.kernelRadius));
			tempN += 1 * (mass_j / density_j*gradW(pos_i, pos_j, params.kernelRadius));
		}
		gradientOfColorField[index] = -1.0*tempN;
	}

}

//计算反馈力
__global__ void computeFeedbackForceKernel(float3 *feedbackForce, double *density, double *signDistance, float3 *gradientOfSignDistance, float3 *gradienOfColorField, float3 *velocity, float3 *pos, unsigned int *numNeighbor, uint *neighbor,unsigned int numParticles)
{
	unsigned int index = blockDim.x*blockIdx.x + threadIdx.x;
	double epslionDistance = params.particleRadius;
	if (index < numParticles)
	{
		int s_i;
		double forceMag; //反馈力的大小
		double densityDifference;
		float3 velocity_i = velocity[index];
		double dotCG;// gradientOfColorField 和 gradientOfSignDistance点积
		double dotVS;//计算速度和距离场梯度的点积
		dotCG = DotProduct(gradienOfColorField[index], gradientOfSignDistance[index]);
		dotVS = DotProduct(velocity_i, gradientOfSignDistance[index]);
		//判断几何势场方向和粒子期望的运动趋势是否冲突
		if(dotCG >= 0.0)
			s_i = 1;
		else
			s_i = -1;

		if (abs(density[index] - params.density0) > 0.02*params.density0)
		{
			densityDifference = (density[index] - params.density0)*(density[index] - params.density0);
		}
		else
		{
			densityDifference = 0.0;
		}

		//zl-19-1-3-start
	/*	if (signDistance[index] > 10)
			signDistance[index] = 3;*/
		//zl-19-1-3-end

		forceMag = 0.16*(Sigmoid(signDistance[index] * signDistance[index] * densityDifference)-0.5); //驱动力根据符号距离和密度差确定,由于Sigmoid（）中的参量大于0，-0.5是为了Sigmoid（）的计算结果最小值为0；
		//forceMag =0.08; //驱动力设置为恒定力
		

		//计算邻域所受力的大小及方向
		double nebForceSumOut = 0.0; //位于目标模型外粒子所有力的合力
		double nebForceSumIn = 0.0;  //位于目标模型内粒子所有力的合力
		float3 nebForceDirectionSumOut = make_float3(0.0, 0.0, 0.0);
		float3 nebForceDirectionSumIn = make_float3(0.0, 0.0, 0.0);
		unsigned int nebNum = numNeighbor[index];
		double nebDensityDiff;
		int outNum = 0; //位于目标模型外粒子数
		int inNum = 0;  //位于目标模型内粒子数
		int largeDensity0In; //位于目标模型内部的大于density0的邻域粒子数
		for (int k = 0; k < nebNum; k++)
		{
			int index_k = neighbor[index*params.maxNumOfNeighbors + k];
			float3 pos_k = pos[index_k];
			double density_k = density[index_k];
			nebDensityDiff = (density_k - params.density0)*(density_k - params.density0);

			//方向
			if (signDistance[index_k] >= 0)
			{
				nebForceSumOut += 0.16*(Sigmoid(signDistance[index] * signDistance[index] * nebDensityDiff) - 0.5);
				nebForceDirectionSumOut = nebForceDirectionSumOut + make_float3(0 - pos_k.x, 0 - pos_k.y, 0 - pos_k.z);
				outNum++;
			}
			else
			{
				if (abs(density_k - params.density0) > 0.05*params.density0)
				{
					nebForceSumIn += 0.16*(Sigmoid(signDistance[index] * signDistance[index] * nebDensityDiff) - 0.5);
					nebForceDirectionSumIn = nebForceDirectionSumIn + gradientOfSignDistance[index_k];
					inNum++;
				}
				
				if (density_k > (1 + 0.03)*params.density0)
					largeDensity0In++;
			}
		}


		if (signDistance[index] >= 0)
		{
			float3 pos_i = pos[index];
			float3 direction = make_float3(0 - pos_i.x, 0 - pos_i.y, 0 - pos_i.z);
			direction = direction + nebForceDirectionSumOut; //加上邻域的方向
			double norm = sqrt(direction.x*direction.x + direction.y*direction.y + direction.z*direction.z);
			direction.x = direction.x / norm;
			direction.y = direction.y / norm;
			direction.z = direction.z / norm;
			//feedbackForce[index] = forceMag*direction;
			feedbackForce[index] = ((forceMag+nebForceSumOut)/(1.0+outNum))*direction;
		}
		else if(signDistance[index] < 0-epslionDistance && density[index]<params.density0)
		{
			//feedbackForce[index] = forceMag*gradientOfSignDistance[index];

			double temForce = (forceMag + nebForceSumIn) / (1.0 + inNum);
			float3 temDir = gradientOfSignDistance[index] + nebForceDirectionSumIn;
			double temNorm = sqrt(temDir.x*temDir.x + temDir.y*temDir.y + temDir.z*temDir.z);
			temDir.x = temDir.x / temNorm;
			temDir.y = temDir.y / temNorm;
			temDir.z = temDir.z / temNorm;
			feedbackForce[index] = temForce*temDir;
			
		}
		else if (signDistance[index] < 0 && density[index]>(1+0.03)*params.density0)
		{
			//通过第一个数字修改受力大小
			feedbackForce[index] = -4.5*forceMag*gradientOfSignDistance[index]; 

			/*if(dotVS>0)
				velocity[index] = velocity[index] * (-1);*/
			//printf("den=%f,force=%f,%f,%f\n", density[index], feedbackForce[index].x, feedbackForce[index].y, feedbackForce[index].z);
		}
		else
		{
			feedbackForce[index] = -0.01*gradientOfSignDistance[index]; //距离表面比较近时，用反馈力迫使粒子回到体内
		}

		//用于控制当粒子在目标体内部扩散到边界时，为保持形状匹配，需要将速度变为反方向
		//dotVS:是粒子速度和符号距离场梯度的点积，用于判断粒子是否是在目标内部做扩散运动|>0 表示粒子是从目标体外面运到了目标体内部|
		if (signDistance[index] < 0 && signDistance[index] > -epslionDistance && dotVS>0)
		{
			velocity[index] = velocity[index] * (-1);
		}
		
		//根据密度差，用于阻止体内粒子不断向边界运动的情况
		//当粒子密度与restDensity相近时，表示局部已充满了粒子，这时粒子应为静止状态，实现目标的稳定
		if (signDistance[index] < 0 && abs(density[index] - params.density0) < params.density0*0.05)
		{
			velocity[index] = make_float3(0.0,0.0,0.0);
		}
		//printf("(%f  %f  %f)  ", feedbackForce[index].x, feedbackForce[index].y, feedbackForce[index].z);

	}
}

/* 计算阻力，
*/
__global__ void computeDampingForceKernel(float3 *dampForce, float3 *velocity, float3 *feedbackForce, double *signDistance, unsigned int numParticle)
{
	unsigned int index = blockDim.x*blockIdx.x + threadIdx.x;
	double k = 0.5;
	if (index < numParticle)
	{
		double distance = signDistance[index];
		if (signDistance[index] < 0)
		{
			dampForce[index] = -1 * k*velocity[index];
		}
		else
		{
			dampForce[index] = make_float3(0.0, 0.0, 0.0);
		}
		
	}
}

//计算合力
__global__ void computeSumForceKernel(float3 *sumForce, float3 *feedbackForce, float3 *dampingForce, unsigned int numParticle)
{
	unsigned int index = blockDim.x*blockIdx.x + threadIdx.x;
	if (index < numParticle)
	{
		sumForce[index] = feedbackForce[index] + dampingForce[index];
	}
}

//根据SDF值更新粒子的质量
__global__ void updateMassForParAccordingSDFKernel(double *mass, double *signDistance, double massForParInShape, double massForParOnSurface, float3 *pos, unsigned int numParticle)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < numParticle)
	{
		float3 pos_i = pos[index];
		double sdf_i = signDistance[index];
		//当粒子为与目标形状内部，且不属于目标形状表面粒子时，更新粒子质量
		double distanceTest = -3 * 2 * params.particleRadius;
		if (sdf_i < distanceTest)
		{
			mass[index] = massForParInShape;
		}
		else
		{
			mass[index] = massForParOnSurface;
		}
	}
}


//-----------------控制粒子法实现演化控制的核函数-----zzl-----2019-1-16-------start------
__global__ void computeDensityAtTargetParticleKernel(double *densityAtTargetParticle, //output, 目标粒子位置的密度
													float3 *targetParticlesPos, //input, 目标粒子的位置
													unsigned int numTarParticles, //input, 目标粒子的数量
													float3 *pos, //input, 流体粒子的位置
													double *mass, //input, 流体粒子的质量
													unsigned int numParticle,//input, 流体粒子的数量
													uint *cellStart, //input, cell中存放的第一个粒子的索引
													uint *cellEnd, //input, cell中存放的最后一个粒子的索引
													uint *gridParticleIndex)//input, 按hash排序后的存放粒子初始编号的变量
{
	unsigned int index;
	index = blockDim.x*blockIdx.x + threadIdx.x;
	if (index < numTarParticles)
	{
		float3 posTar = targetParticlesPos[index];
		int3 ptc_cell = GetCell(posTar, params.cellSize); //计算控制粒子所在的cell编号
		double densitySum = 0.0;

		int numCellInKernel = ceil(params.kernelRadius / params.cellSize); //计算在支持域半径内包含多个单元格
		float dist2 = params.kernelRadius*params.kernelRadius;

		for (int cz = -1 * numCellInKernel; cz <= numCellInKernel; cz++)
			for (int cy = -1 * numCellInKernel; cy <= numCellInKernel; cy++)
				for (int cx = -1 * numCellInKernel; cx <= numCellInKernel; cx++)
				{
					int3 nb_cell; //邻域单元格位置
					nb_cell.x = ptc_cell.x + cx;
					nb_cell.y = ptc_cell.y + cy;
					nb_cell.z = ptc_cell.z + cz;

					if (!IsCellInRange(nb_cell)) continue; //判断网格索引是否在有效范围内

					uint hash = calcGridHash(nb_cell); //获取cell的hash值
					uint startIndex = cellStart[hash]; //根据hash值，得到cell中存储的第一个粒子的序号

					if (startIndex != 0xffffffff)
					{
						uint endIndex = cellEnd[hash];
						for (uint i = startIndex; i < endIndex; i++)
						{
							uint originalIndex = gridParticleIndex[i];
							float3 pos_n = pos[originalIndex];
							float dist = DistanceSquare(posTar, pos_n);
							if (dist < dist2 )
							{
								densitySum += mass[originalIndex] * W(posTar, pos_n, params.kernelRadius);
							}

						}//for(uint i = startIndex; i < endIndex; i++)
					}//end if(startIndex != 0xffffffff)
				}// endfor endfor endfor
		densityAtTargetParticle[index] = densitySum;
	}
}

//计算流体粒子受到周围目标粒子所产生的吸引力
__global__ void computeForceFromTargetParticleKernel (float3 *forceFromTargetParticle,  //output
														float3 *targetParticlePos,  //input
														double *densityAtTargetParticle, //input
														unsigned int numTargetParticles, //input
														double *signDistance, //input
														float3 *pos, //input
														unsigned int numParticles)//input
{
	unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < numParticles)
	{
		float3 pos_i = pos[index];
		double sdf_i = signDistance[index];
		float3 forceSum = make_float3(0.0, 0.0, 0.0);
		float kForInfluenceArea = 9.0;
		double disTest = kForInfluenceArea * sqrt(params.kernelRadius*params.kernelRadius); //用于测试两个粒子之间是否存在影响的距离阈值
		double densityK = 0.3; //给定的密度阈值
		double strengthOfForce = 0.8 * 0.05*(1.0 / 1000); // 0.05*(1.0 / 1000)使得force取值在0~1之间，0.8是缩放系数

		if (sdf_i < 0) //当位于目标内部时，计算目标粒子施加的力
		{
			for (int j = 0; j < numTargetParticles; j++)
			{
				double density_j = densityAtTargetParticle[j];
				float3 pos_j = targetParticlePos[j];
				double dis_i_j = sqrt(DistanceSquare(pos_i, pos_j));  //计算pos_i和pos_j之间的距离
																	  
			    //如果粒子的密度小于给定的阈值 && 与流体粒子之间的距离小于阈值， 则产生影响力 
				if (density_j < densityK*params.density0 && dis_i_j < disTest)
				{
					//ADD code
					float norm_i_j = sqrt(DistanceSquare(pos_i, pos_j));
					float3 direction = (pos_j - pos_i) / norm_i_j;
					float kernelWeight = W(pos_i, pos_j, kForInfluenceArea*params.kernelRadius);
					float densityWeight = exp(-1.0*density_j / params.density0); //e^(-x)
					forceSum = forceSum + direction*densityWeight*kernelWeight;
				}
			}
			forceFromTargetParticle[index] = strengthOfForce * forceSum;
		}
		else //当位于目标外部时，不考虑目标粒子施加的力
		{
			forceFromTargetParticle[index] = forceSum;
		}

		//printf(" (%f, %f, %f) ", forceFromTargetParticle[index].x, forceFromTargetParticle[index].y, forceFromTargetParticle[index].z);
	}
}

__global__ void computeSumForceForParticleControlKernel(float3 *sumForce, float3 *feedbackForce, float3 *dampingForce, float3 *targetParticleAttractionForce, unsigned int numParticle)
{
	unsigned int index = blockDim.x*blockIdx.x + threadIdx.x;
	if (index < numParticle)
	{
		float3 force1 = feedbackForce[index];
		float3 force2 = targetParticleAttractionForce[index];
		float dotResult = DotProduct(force1, force2);
		if (dotResult < 0)
		{
			sumForce[index] = dampingForce[index] + targetParticleAttractionForce[index];
		}
		else
		{
			sumForce[index] = feedbackForce[index] + dampingForce[index] + targetParticleAttractionForce[index];
		}
	}
}
//-------------------------------------------------zzl-----2019-1-16-------end---------


/*---------------控制粒子-目标粒子-对应控制法-----------------------------------------------*/
//-------------------zzl-----------------2019-1-23-start----------------------------------
__global__ void computeForceForControlParticleKernel(float3 *sumForce, 
													float3 *iniPos, 
													float3 *tarPos,
													unsigned int *conIndex, 
													unsigned int *tarIndex, 
													unsigned int particleNum,
													unsigned int controlParticleNum)
{
	unsigned int index = blockDim.x*blockIdx.x + threadIdx.x;
	if (index < particleNum)
	{
		float3 force = make_float3(0, 0, 0);
		int index_con = -1; //记录当前粒子在控制粒子列表中的位置
		unsigned int index_ini = -1; //记录控制粒子列表中第i个位置存放的是哪个初始形状的粒子
		//判断索引为index的粒子是否为控制粒子
		for (int i = 0; i < controlParticleNum; i++)
		{
			index_ini = conIndex[i];
			if (index == index_ini)
			{
				index_con = i;
				break;
			}
		}
		//是控制粒子
		if (index_con >= 0)
		{
			float3 pos_con = iniPos[index_ini];
			float3 pos_tar = tarPos[tarIndex[index_con]];
			float distance = sqrt(DistanceSquare(pos_con, pos_tar));
			//引力方向
			float3 direction = (pos_tar - pos_con) / distance;
			//引力大小
			float A = 0.0; //scaling parameters
			float B = 0.6; //scaling parameters
			float w_attraction = (distance - A)*B;
			//clamping the w_attraction
			if (w_attraction > 1)
				w_attraction = 1;
			if (w_attraction < 0)
				w_attraction = 0;

			float S = 1.0; //global attraciont strength parameter

			force = direction * w_attraction * S;
		}
		
		sumForce[index] = force;
	}
}

//速度插值核函数
__global__ void velocityInterpolationKernel(float3 *velocity, 
											float3 *pos, 
											unsigned int *conIndex, 
											unsigned int particleNum, 
											unsigned int controlParticleNum)
{
	unsigned int index = blockDim.x*blockIdx.x + threadIdx.x;
	if (index < particleNum)
	{
		float3 pos_i = pos[index]; //获取当前粒子的位置
		float3 vel_i;
		int index_ini = -1;
		int flag = 0; //1:控制粒子，0：非控制粒子
		//判断粒子是否为控制粒子
		for (int i = 0; i < controlParticleNum; i++)
		{
			index_ini = conIndex[i];
			if (index_ini == index)
			{
				flag = 1;
				break;
			}
		}
		//非控制粒子，需要进行速度插值
		float3 sumWeightU;
		float sumWeight = 0.0;
		float weight_j;
		float r = params.particleRadius * 2 * 5; //控制粒子影响区域半径
		if (flag == 0)
		{
			for (int j = 0; j < controlParticleNum; j++)
			{
				float3 pos_j = pos[conIndex[j]]; 
				weight_j = weightForParticleControl(pos_i, pos_j, r);
				float3 u_j = velocity[conIndex[j]];
				sumWeightU = sumWeightU + weight_j * u_j;
				sumWeight = sumWeight + weight_j;
			}
			vel_i = sumWeightU / sumWeight;
			velocity[index] = vel_i;
		}
	}
}

//根据加速更新速度
__global__ void computeVelocityForConTarMehtodKernel (float3 *newVel, float3 *oldVel, float3 *acceleration, float deltaTime, unsigned int numParticle)
{
	unsigned int index = blockDim.x*blockIdx.x + threadIdx.x;
	if (index < numParticle)
	{
		float3 velocity;
		float c = 0.5;
		/*newVel[index].x = oldVel[index].x + acceleration[index].x*deltaTime;
		newVel[index].y = oldVel[index].y + acceleration[index].y*deltaTime;
		newVel[index].z = oldVel[index].z + acceleration[index].z*deltaTime;*/
		velocity.x = oldVel[index].x + acceleration[index].x*deltaTime;
		velocity.y = oldVel[index].y + acceleration[index].y*deltaTime;
		velocity.z = oldVel[index].z + acceleration[index].z*deltaTime;

		newVel[index].x = velocity.x*(1 - c);
		newVel[index].y = velocity.y*(1 - c);
		newVel[index].z = velocity.z*(1 - c);
	}
}

//根据计算得到的新速度计算粒子的位置
__global__ void computePosForConTarMethodKernel(float3 *newPos, float3 *oldPos, float3 *newVel, float deltaTime, unsigned int numParticle)
{
	unsigned int index = blockDim.x*blockIdx.x + threadIdx.x;
	if (index < numParticle)
	{
		newPos[index].x = oldPos[index].x + newVel[index].x*deltaTime;
		newPos[index].y = oldPos[index].y + newVel[index].y*deltaTime;
		newPos[index].z = oldPos[index].z + newVel[index].z*deltaTime;
	}
}

//-------------------zzl-----------------2019-1-23-end----------------------------------


//////////////////////////////////////////////////////////////////////////////////
////---------------------------extern function----------------------------------
// Round a / b to nearest higher integer value
extern "C" uint iDivUp(uint a, uint b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

// 计算block（包含多少个thread：numThreads）大小和grid(包含多少个block：numBlock)大小
extern "C" void computeGridSize(uint n, uint blockSize, uint &numBlock, uint &numThreads)
{
	numThreads = min(n, blockSize);
	numBlock = iDivUp(n, numThreads);
}

//将全局参数从host 拷贝到device中
extern "C" void setParams(SimParams * hostParams)
{
	cudaMemcpyToSymbol(params, hostParams, sizeof(SimParams));
}


//计算粒子所在的网格|（cell_id，particle_id），例如（10,2）|
extern "C" void calcHash( uint *gridParticleHash, //output: 存cell_id
					 uint *gridParticleIndex, //output: 存particle_id
					 float *particlePos,   //input
	                 int particleNum) //input
{
	uint numBlocks, numThreads;
	computeGridSize(particleNum, 256, numBlocks, numThreads);
	
	//execute the kernel
	calHashKernel << <numBlocks, numThreads >> > (gridParticleHash, gridParticleIndex, (float3 *)particlePos, particleNum);

	/*cudaError_t error = cudaGetLastError();
	std::cout << "CUDA error:" << cudaGetErrorString(error);*/

	// check if kernel invocation generated an error
	getLastCudaError("calHashKernel execution failed");
}

// sort particle based on hash
//example for gridParticleHash and gridParticleIndex:(particleNum = 6, cellNum = 16, only three cells contain particles: cell no.:4,6,9)
// original ---> gridParticleHash:|9 6 6 4 6 4|, sorted by hash-> gridParticleHash:|4 4 6 6 6 9|
// original --->gridParticleIndex:|0 1 2 3 4 5|, sorted by hash->gridParticleIndex:|3 5 1 2 4 0|
// cellStart[4] = 0, cellstart[6] = 2, cellSatrt[9] = 5
//   cellEnd[4] = 2,   cellEnd[6] = 5,   cellEnd[9] = 6
extern "C" void sortParticlesByHash(uint *gridParticleHash, uint *gridParticleIndex, int particleNum)
{
	thrust::sort_by_key(thrust::device_ptr<uint>(gridParticleHash),
		thrust::device_ptr<uint>(gridParticleHash + particleNum),
		thrust::device_ptr<uint>(gridParticleIndex));
}


// find the cellStart and cellEnd
extern "C" void reorderDataAndFindCellStart(uint *cellStart, uint *cellEnd, float *sortedPos, float *sortedVel, uint *gridParticheHash, uint *gridParticleIndex, float *oldPos, float * oldVel, int numParticles, int numCells)
{
	uint numBlocks, numThreads;
	computeGridSize(numParticles, 256, numBlocks, numThreads);

	// ste all cells to empty
	checkCudaErrors(cudaMemset(cellStart, 0xffffffff, numCells * sizeof(uint)));

	uint shareMemSize = sizeof(uint)*(numThreads + 1);
	reorderDataAndFindCellStartKernel << <numBlocks, numThreads, shareMemSize >> > (
		cellStart, 
		cellEnd, 
		(float3 *)sortedPos, 
		(float3 *)sortedVel, 
		gridParticheHash, 
		gridParticleIndex, 
		(float3 *)oldPos, 
		(float3 *)oldVel, 
		numParticles);
	getLastCudaError("Kernel exectuion failed : reorderDataAndCellSatrt");
}


// 查找邻域
extern "C" void GpuNeighborSearch(float * Positions, //input 粒子位置
									unsigned int *numNeighbors, //output 存放每个粒子的邻域粒子数,大小为[numParticles]
									unsigned int *neighbors, //output 存放邻域粒子编号,大小为[numParticles]*[maxNeighborsNum]
									int numParticles, //input 粒子数
									int maxNeighborsNum, //input 最大邻域数
									float radius, //input 粒子半径
									float kernelRadius, //input 支持域半径
									uint *cellStart, //input 
									uint *cellEnd, //input
									uint *gridParticleIndex)//input
{
	//float3 *cudaPositions; //存放粒子位置信息
	//unsigned int * cudaNeighbors; //存放每个粒子的邻域信息
	//unsigned int * cudaNumNeighborsForParticle; //存放每个粒子的邻域个数
	////在device上分配存储空间
	//unsigned int positionSize = sizeof(float3) * numParticles; //
	//unsigned int neighborSize = sizeof(unsigned int) * numParticles * maxNeighborsNum; //存放邻域信息所需的总空间量
	//unsigned int neighborNumSize = sizeof(unsigned int) * numParticles; //存放粒子邻域个数所需的总空间
	//cudaMalloc((void **)&cudaPositions, positionSize);
	//cudaMalloc((void **)&cudaNeighbors, neighborSize);
	//cudaMalloc((void**)&cudaNumNeighborsForParticle, neighborNumSize);
	//
	uint numBlocks, numThreads;
	computeGridSize(numParticles, 256, numBlocks, numThreads);
	// 数据拷贝host->device
	/*cudaMemcpy(Positions, Positions, positionSize, cudaMemcpyHostToDevice);*/
	GpuNeighborSearchKernel << <numBlocks, numThreads >> > ((float3 *)Positions, numNeighbors, neighbors, numParticles, radius, kernelRadius, cellStart, cellEnd, gridParticleIndex);
	getLastCudaError("kernel executon failed: neighborSearch");
}

// update position and velocity
extern "C" void integrateSystem(float *pos, float *vel, float deltaTime, uint numParticle)
{
	thrust::device_ptr<float3> d_pos((float3*)pos);
	thrust::device_ptr<float3> d_vel((float3*)vel);

	thrust::for_each(
		thrust::make_zip_iterator(thrust::make_tuple(d_pos, d_vel)),
		thrust::make_zip_iterator(thrust::make_tuple(d_pos + numParticle, d_vel + numParticle)),
		integrate_functor(deltaTime));
}

extern "C" void _integrateSystemForCloud(float *newPos, //output
										float *newVel, //output
										float *oldPos, //input
										float *oldVel, //input
										float *acceleration, //input
										float deltaTime, //input
										uint numParticle) //input
{
	unsigned int numBlocks, numThreads;
	computeGridSize(numParticle, 256, numBlocks, numThreads);
	integrateSystemForCloudKernel << <numBlocks, numThreads >> > ((float3 *)newPos, (float3 *)newVel, (float3 *)oldPos, (float3 *)oldVel, (float3 *)acceleration, deltaTime, numParticle);
	cudaDeviceSynchronize();
	getLastCudaError("kernel exection failed: integrateSystemForCloud");
}

extern "C" void _computeDensity(double * density,  //output: density[i] for particle_i
	float * position,  //input: position for particles
	double * mass,//input: mass for particle
	unsigned int * numNeighbors, //input: numNeighbors[i]:the total of neighbors of the particle_i
	unsigned int *neighbors, //input
	unsigned int numParticles) //input
{
	unsigned int numBlocks, numThreads;
	computeGridSize(numParticles, 256, numBlocks, numThreads);
	computeDensityKernel << <numBlocks, numThreads >> > (density, (float3 *)position, mass, numNeighbors, neighbors, numParticles);
	cudaDeviceSynchronize();
	getLastCudaError("Kernel exectuion failed : computeDensity");
}

//compute pressure
extern "C" void  _computePressure(double *pressure,double *density,unsigned int numParticles)
{
	unsigned int numBlocks, numThreads;
	computeGridSize(numParticles, 256, numBlocks, numThreads);
	computePressureKernel << <numBlocks, numThreads >> > (pressure, density, numParticles);
	cudaDeviceSynchronize();
	getLastCudaError("Kernel exectuion failed :computePressure");
}

//compute pressure force
extern "C" void _computePressureForce(float *pressureForce, //output
										double *pressure,  //input
										double *mass, //input
										double *density,
										float *position, //input
										unsigned int numParticles,//input
										unsigned int *numNeighbors,//input
										unsigned int *neighbors)//input
{
	unsigned int numBlocks, numThreads;
	computeGridSize(numParticles, 256, numBlocks, numThreads);
	computePressureForceKernel << <numBlocks, numThreads >> > ((float3 *)pressureForce, 
																pressure, 
																mass, 
																density,
																(float3 *)position, 
																numParticles, 
																numNeighbors, 
																neighbors);
	cudaDeviceSynchronize();
	getLastCudaError("Kernel execution failed: computePressureForce");
}

//compute buoyance according to the gradient of temperature
// f=c*(lapsin T) | ▽T = sum(m_j/density_j)*(T_i+T_j)/2 * (▽W）|
extern "C" void _computeBuoyanceByGradientOfTemperature(float *buoyance, //output
														double *temperature, //input
														float *position, //input
														double *density,//input
														double *mass,
														unsigned int numParticles, //input
														unsigned int *numNeighbors,//input 
														unsigned int *neighbors)//input
{
	unsigned int numBlocks, numThreads;
	computeGridSize(numParticles, 256, numBlocks, numThreads);
	computeBuoyanceByGradientOfTemperatureKernel << <numBlocks, numThreads >> > ((float3 *)buoyance, 
																				 temperature, 
																				 (float3 *)position, 
																				 mass, 
																				 density, 
																				 numParticles, 
																				 numNeighbors, 
																				 neighbors);
	cudaDeviceSynchronize();
	getLastCudaError("Kernel execution failed: computeBuoyanceForce");
}

//根据计算的外力，更新加速度： a = f/m
extern "C" void _updateAccelerations(float *acceleration, //output: 加速度
									float *buoyanceForce, //input
									double *mass, //input
									unsigned int numParticles) //input
{
	unsigned int numBlocks, numThreads;
	computeGridSize(numParticles, 256, numBlocks, numThreads);
	updateAccelerationKernel << <numBlocks, numThreads >> > ((float3 *)acceleration, (float3 *)buoyanceForce, mass, numParticles);
	cudaDeviceSynchronize();
	getLastCudaError("kernel execution failed: updateAcceleration");
}

//相变： △Qc = k(qv-qs)| △Qv = -k(qv-qs) | △T=m*(k(qv-qs)) |
extern "C" void _phaseTransition(double *newCloud, //output
								double *newVapor, //output
								double *newTemperature, //output
								double *temperatureChange, //output
								double *oldCloud, //input
								double *oldVapor, //input
							    double *temperature, //input
								unsigned int numParticles) //input
{
	unsigned int numBlocks, numThreads;
	computeGridSize(numParticles, 256, numBlocks, numThreads);
	phaseTransitionKernel << <numBlocks, numThreads >> > (newCloud, newVapor, newTemperature, temperatureChange, oldCloud, oldVapor, temperature, numParticles);
	cudaDeviceSynchronize();
	getLastCudaError("kernel exectuion failed: phaseTransition");
}

//compute lambda for updating position
extern "C" void _computeLambda(double *lambda, //output
								double *density, //input
								double *mass, //input
								float *pos, //input
								unsigned int *numNeighbor,//input 
								unsigned int *neighbor, //input
								unsigned int numParticle)//input
{
	unsigned int numBlocks, numThreads;
	computeGridSize(numParticle, 256, numBlocks, numThreads);
	computeLambdaKernel << <numBlocks, numThreads >> > (lambda, density, mass, (float3 *)pos, numNeighbor, neighbor, numParticle);
	cudaDeviceSynchronize();
	getLastCudaError("Kernel execution failed: computeLambda");
}

//compute △p
extern "C" void _computeDeltaPositionByLambda(float *deltaPos, //output
										float *pos, //input
										double *lambda, //input
										double *mass, //input
										double *density, //input
										unsigned int *numNeighbor, //input 
										unsigned int *neighbor, //input
										unsigned int numParticle) //input
{
	unsigned int numBlocks, numThreads;
	computeGridSize(numParticle, 256, numBlocks, numThreads);
	computeDeltaPositionKernel<<<numBlocks,numThreads>>>((float3 *)deltaPos, (float3 *)pos, lambda, mass, density, numNeighbor, neighbor, numParticle);
	cudaDeviceSynchronize();
	getLastCudaError("Kernel execution failed: computeDeltaPosition");
}

//update postion according to △p
extern "C" void _updatePosition(float* newPos, float* deltaPos, unsigned int numParticle)
{
	unsigned int numBlocks, numThreads;
	computeGridSize(numParticle, 256, numBlocks, numThreads);
	updatePositionKernel << <numBlocks, numThreads >> > ((float3 *)newPos, (float3 *)deltaPos, numParticle);
	cudaDeviceSynchronize();
	getLastCudaError("Kernel execution failed: updatePosition");
}

//update velocity
extern "C" void _updateVelocity(float *newVel, float *oldPos, float *newPos, double deltime, unsigned int numParticle)
{
	unsigned int numBlocks, numThreads;
	computeGridSize(numParticle, 256, numBlocks, numThreads);
	updateVelocityKernel << <numBlocks, numThreads >> > ((float3 *)newVel, (float3 *)newPos, (float3 *)oldPos, deltime, numParticle);
	cudaDeviceSynchronize();
	getLastCudaError("Kernel execution failed: updateVelocity");
}

//判断哪些位置需要增加粒子或删除粒子
//|densityK:密度阈值参数，下限密度为（1-densityK）* density0; 上线密度为（1+densityK）* density0;
extern "C" void _checkParticleByDensity(int *flag, float *pos, double *density, double densityK, unsigned int numParticle)
{
	unsigned int numBlocks, numThreads;
	computeGridSize(numParticle, 256, numBlocks, numThreads);
	checkParticleByDensityKernel << <numBlocks, numThreads >> > (flag, (float3 *)pos, density, densityK, numParticle);
	getLastCudaError("kernel execution failed: checkParticleByDensity");
}

//增加粒子
extern "C" void _addParticles(float *addPos,          //output: 新增粒子的位置 size = numParticle * maxNeighborNum
								unsigned int *addNum, //output: 新增的粒子数量
								float *pos,           //input: 原始粒子位置
								int *flag,            //input: 标记是否密度< density0
								unsigned int *cellstart, //input
								unsigned int *cellEnd,   //input
								unsigned int *gridParticleIndex, //input
								double *density,      //input
								double *mass,         //input
								unsigned int numParticle) //input
{
	unsigned int numBlocks, numThreads;
	computeGridSize(numParticle, 256, numBlocks, numThreads);
	addParticleKernel << <numBlocks, numThreads >> >((float3 *)addPos, addNum, (float3 *)pos, flag, cellstart, cellEnd, gridParticleIndex, density, mass, numParticle);
	cudaDeviceSynchronize();
	getLastCudaError("kernel execution failed: addParticelKernel");
}

extern "C" void _deleteParticle(unsigned int *deleteFlag,  //output: 记录粒子是否被删除，1：删除，0未删除
								int *flag,                 //input: 记录粒子是否需要更新， 1：增加，0：不变，-1：删除
								float *pos,                //input
								double *mass,              //input
								double *density,           //input
								uint *neighbor,            //input
								uint *neighborNum,         //input
								unsigned int numParticle)  //input
{
	unsigned int numBlocks, numThreads;
	computeGridSize(numParticle, 256, numBlocks, numThreads);
	//cudaMemset(m_dDeleteFlag, 0, sizeof(unsigned int)*m_numParticles);
	deleteParticleKernel << <numBlocks, numThreads >> > (deleteFlag, flag, (float3 *)pos, mass, density, neighbor, neighborNum, numParticle);
	cudaDeviceSynchronize();
	getLastCudaError("kernel execution failed: deleteParticle");
}

extern "C" void _compactFlag(int *flag, float *pos, unsigned int numParticle)
{
	//add your code
}


//计算color field 梯度
extern "C" void _computeGradientOfColorField(float *gradientOfColorField, //output, gradient of color field
											double *mass, //input
											float *pos, //input
											double *density, //input
											unsigned int numParticle, //input
											unsigned int *numNeighbor, //input
											unsigned int *neighbor)//input
{
	unsigned int numBlocks, numThreads;
	computeGridSize(numParticle, 256, numBlocks, numThreads);
	computeGradientOfColoFieldKernel << <numBlocks, numThreads >> > ((float3 *)gradientOfColorField, mass, density, (float3*)pos, numParticle, numNeighbor, neighbor);
	getLastCudaError("kernel execution failed: computeGradientOfColorFieldKernel");
}

extern "C" void _computeFeedbackForce(float *feedbackForce, //output 
										double *density,  //input
										double *signDistance,  //input
										float *gradientOfSignDistance, //input
										float *gradienOfColorField,
										float *velocity,
										float *pos,
										unsigned int *numNeighbor,
										unsigned int *neighbor,
										unsigned int numParticle) //input
{
	unsigned int numBlocks, numThreads;
	computeGridSize(numParticle, 256, numBlocks, numThreads);
	computeFeedbackForceKernel << <numBlocks, numThreads >> > ((float3 *)feedbackForce, density, signDistance, (float3 *)gradientOfSignDistance, (float3 *)gradienOfColorField, (float3 *) velocity,(float3 *)pos, numNeighbor, neighbor,numParticle);
	getLastCudaError("kernel execution failed: computeFeedbackForceKernel");
}


//for control phase transition
//根据密度和符号距离场值确定是否需要相变
extern "C" void _checkParticleByDensityAndLocation(int *flag,  //output
													float *pos, 
													double *density, 
													double densityK, 
													double* signDistance,
													unsigned int *numNeighbor,
													unsigned int numParticle)
{
	unsigned int numBlocks, numThreads;
	computeGridSize(numParticle, 256, numBlocks, numThreads);
	checkParticleByDensityAndLocationKernel << <numBlocks, numThreads >> > (flag, (float3 *)pos, density, densityK, signDistance, numNeighbor,numParticle);
	cudaDeviceSynchronize();
	getLastCudaError("kernel execution failed: checkParticleByDensityAndLocationKernel");
}

extern "C" void _computeDampingForce(float * dampForce, //output, 阻力
									float *velocity,  //input
									float *feedbackForce, //input
									double *signDistance, //input
									double *density, //inout
									unsigned int numParticle) //input
{
	unsigned int numBlocks, numThreads;
	computeGridSize(numParticle, 256, numBlocks, numThreads);
	computeDampingForceKernel << <numBlocks, numThreads >> > ((float3 *)dampForce, (float3 *)velocity, (float3 *)feedbackForce, signDistance, numParticle);
	getLastCudaError("kernel execution failed: computeDampingKernel");
}

extern "C" void _computeSumForce(float *sumForce, float *feedbackForce, float *dampingForce, unsigned int numParticle)
{
	unsigned int numBlocks, numThreads;
	computeGridSize(numParticle, 256, numBlocks, numThreads);
	computeSumForceKernel << <numBlocks, numThreads >> > ((float3 *)sumForce, (float3 *)feedbackForce, (float3 *)dampingForce, numParticle);
	getLastCudaError("kernel execution failed: computeSumForceKernel");
}

extern "C" void _updateMassForParticleAccordingSDF(double *mass, //output
													double *signDistance, //input
													double massForParInShape, //input 内部粒子的质量
													double massForParOnSurface,//input 非内部粒子的质量
													float *pos, //input
													unsigned int numParticle) //input
{
	unsigned int numBlocks, numThreads;
	computeGridSize(numParticle, 256, numBlocks, numThreads);
	updateMassForParAccordingSDFKernel << <numBlocks, numThreads >> > (mass, signDistance, massForParInShape, massForParOnSurface,(float3 *)pos, numParticle);
	getLastCudaError("kernel execution failed: updatMassForParAccordingSDFKerenl");
}

//---------------控制粒子法所需功能--------zzl-------2019-1-16-------start-------------

//计算控制粒子所在位置的密度值
extern "C" void _computeDensityAtTargetParticle(double *densityAtTargetParticle, //output,密度输出
												float *targetParticlesPos, //input,目标粒子位置
												unsigned int numTarParticles, //input,目标粒子的数量
												float *pos,		//input,粒子位置
												double *mass, //input,粒子质量
												unsigned int numParticle, //input,粒子数量
												uint *cellStart, //input, cell中存放的第一个粒子的索引
												uint *cellEnd, //input, cell中存放的最后一个粒子的索引
												uint *gridParticleIndex)//input, 按hash排序后的存放粒子初始编号的变量
{
	unsigned int numBlocks, numThreads;
	computeGridSize(numTarParticles, 256, numBlocks, numThreads);
	computeDensityAtTargetParticleKernel << <numBlocks, numThreads >> > (densityAtTargetParticle, (float3 *)targetParticlesPos, numTarParticles, (float3 *)pos, mass, numParticle,cellStart,cellEnd,gridParticleIndex);
	getLastCudaError("kernel execution failed: computeDensityAtTargetParticleKernel");
}

//根据目标粒子的密度，及目标粒子与流体粒子间的距离计算流体粒子所受的力
extern "C" void _computeForceFromTargetParticle(float *forceFromTargetParticle, //output, 目标粒子对流体粒子所产生的力
												float *targetParticlePos, //input
												double *densityAtTargetParticle, //input
												unsigned int numTargetParticles,
												double *signDistance, //input
												float *pos,//input
												unsigned int numParticles)//input
{
	unsigned int numBlocks, numThreads;
	computeGridSize(numParticles, 256, numBlocks, numThreads);
	computeForceFromTargetParticleKernel << <numBlocks, numThreads >> > ((float3 *)forceFromTargetParticle, (float3 *)targetParticlePos, densityAtTargetParticle, numTargetParticles,signDistance, (float3 *)pos, numParticles);
	getLastCudaError("kernel execution failed: computeForceFromTargetParticleKernel");
}

extern "C" void _computeSumForceForParticleControl(float *sumForce, 
													float *feedbackForce, 
													float *dampingForce,
													float *targetParticleAttractionForce,
													unsigned int numParticle)
{
	unsigned int numBlocks, numThreads;
	computeGridSize(numParticle, 256, numBlocks, numThreads);
	computeSumForceForParticleControlKernel << <numBlocks, numThreads >> > ((float3 *)sumForce, (float3 *)feedbackForce, (float3 *)dampingForce, (float3 *) targetParticleAttractionForce, numParticle);
	getLastCudaError("kernel execution failed: computeSumForceForParticleControlKernel");
}
//---------------控制粒子法所需功能--------zzl-------2019-1-16-------end----------------

/*-----------------------------------控制粒子-目标粒子-对应控制方法-----------------------------*/
//参考论文：Target Particle Control of Smoke Simulation
//-----------------------------------------zzl---------2019-1-23-----------------------------*/

extern "C" void _computeForceForControlParticel(float *sumForce,  //output, 存储控制粒子受到的力，非控制粒子所受力设为0
												float *iniPos, //初始形状粒子位置
												float *tarPos, //目标形状的粒子位置
												unsigned int *conIndex, //控制粒子索引
												unsigned int *tarIndex, //目标粒子索引
												unsigned int particleNum, //粒子数
												unsigned int controlParticleNum) //控制粒子数
{
	unsigned int numBlocks, numThreads;
	computeGridSize(particleNum, 256, numBlocks, numThreads);
	computeForceForControlParticleKernel<<<numBlocks,numThreads>>>((float3 *)sumForce, (float3 *)iniPos, (float3 *)tarPos, conIndex, tarIndex, particleNum, controlParticleNum);
	getLastCudaError("kernel execution failed: computeForceForControlParticleKernel");
}

extern "C" void _velocityInterpolation(float *velocity, //input,output
										float *pos, //input 
										unsigned int *conIndex, //input
										unsigned int particleNum, //input
										unsigned int controlParticleNum)//input
{
	unsigned int numBlocks, numThreads;
	computeGridSize(particleNum, 256, numBlocks, numThreads);
	velocityInterpolationKernel << <numBlocks, numThreads >> > ((float3 *)velocity, (float3 *)pos, conIndex, particleNum, controlParticleNum);
	getLastCudaError("kernel execution failed: velocityInterpolationKernel");
}

//根据加速度计算控制粒子的速度
extern "C" void _computeVelocityForControlTargetMethod(float *newVel, //output
														float *oldVel, //input
														float *acceleration, //input
														float deltaTime, //input
														uint numParticle) //input
{
	unsigned int numBlocks, numThreads;
	computeGridSize(numParticle, 256, numBlocks, numThreads);
	computeVelocityForConTarMehtodKernel << <numBlocks, numThreads >> > ((float3 *)newVel, (float3 *)oldVel, (float3 *)acceleration, deltaTime, numParticle);
	getLastCudaError("kernel execution failed:computeVelocityForConTarMehtodKernel");
}

extern "C" void _computePosForConTarMethod(float* newPos, float *oldPos, float *newVel, float deltaTime,unsigned int numParticle)
{
	unsigned int numBlocks, numThreads;
	computeGridSize(numParticle, 256, numBlocks, numThreads);
	computePosForConTarMethodKernel << <numBlocks, numThreads >> > ((float3 *)newPos, (float3 *)oldPos, (float3 *)newVel, deltaTime, numParticle);
	getLastCudaError("kernel execution failed:computePosForConTarMethodKernel");
	
}
//-----------------------------------------zzl---------2019-1-23-----------------------------*/



__device__ float computeDistance(float3 p1, float3 p2)
{
	float distance;
	distance = (p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y) + (p1.z - p2.z)*(p1.z - p2.z);
	distance = sqrt(distance);
	return distance;
}


__device__ int3 computeGridCell(uint index, uint xResolution, uint yResolution, uint zResoution)
{
	int3 cell;
	cell.x = index % xResolution;
	cell.z = index / (xResolution*yResolution);
	cell.y = (index - cell.z*(xResolution*yResolution)) / xResolution;
	return cell;
}


__device__ float3 computeGridPos(int3 cell, float step, float min_x, float min_y, float min_z)
{
	float3 gridPos;
	gridPos.x = min_x + cell.x*step;
	gridPos.y = min_y + cell.y*step;
	gridPos.z = min_z + cell.z*step;
	return gridPos;

}


__global__ void computeGridDensityKernel(float *gridDensity, //output
	float* density,
	float* mass,
	float3* pos,
	float xResolution,
	float yResolution,
	float zResolution,
	float domain_max_x,
	float domain_max_y,
	float domain_max_z,
	float domain_min_x,
	float domain_min_y,
	float domain_min_z,
	float step,
	float h,
	unsigned int numParticle)
{
	uint index;
	uint gridNum = xResolution * yResolution*zResolution;
	index = blockDim.x*blockIdx.x + threadIdx.x;
	if (index < gridNum)
	{
		//���������ż�����������
		int3 cell = computeGridCell(index, xResolution, yResolution, zResolution); //��������
		float3 cellPos = computeGridPos(cell, step, domain_min_x, domain_min_y, domain_min_z); //��������
		float sumDensity = 0;
		float3 pos_i;
		float mass_i;
		float dis;
		int flagOfInVol = -1; //���������Ƿ�����������
		for (int i = 0; i < numParticle; i++)
		{
			pos_i = pos[i];
			mass_i = mass[i];
			dis = computeDistance(pos_i, cellPos);
			if (flagOfInVol == -1 && dis < (step*0.9))
			{
				flagOfInVol = 1;//��ʾ����������ڷ���������
			}
			if (dis < h)
			{
				sumDensity += mass_i * W(cellPos, pos_i, h);
			}
		}//endFor

		if (flagOfInVol == 1)
		{
			gridDensity[index] = sumDensity;
		}
		else
		{
			gridDensity[index] = 0.0;
		}
	}
}


extern "C" void _transformParIntoGrid(float *gridDensity, //output
	float *density,
	float *mass,
	float *pos,
	float xResolution,
	float yResolution,
	float zResolution,
	float domain_max_x,
	float domain_max_y,
	float domain_max_z,
	float domain_min_x,
	float domain_min_y,
	float domain_min_z,
	float step,
	float h,
	unsigned int numParticle)
{
	uint numBlocks, numThreads;
	uint gridNum = xResolution * yResolution*zResolution;
	computeGridSize(gridNum, 256, numBlocks, numThreads);


	//���ñ���
	float* d_GridDensity;
	float* d_density;
	float* d_pos;
	float* d_mass;

	//allocate memory
	cudaMalloc((void**)&d_GridDensity, sizeof(float)*gridNum);
	cudaMalloc((void**)&d_density, sizeof(float)*numParticle);
	cudaMalloc((void**)&d_pos, sizeof(float3)*numParticle);
	cudaMalloc((void**)&d_mass, sizeof(float)*numParticle);

	//���ݿ���
	cudaMemcpy(d_density, density, sizeof(float)*numParticle, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pos, pos, sizeof(float3)*numParticle, cudaMemcpyHostToDevice);
	cudaMemcpy(d_mass, mass, sizeof(float)*numParticle, cudaMemcpyHostToDevice);

	//�����ܶ�
	computeGridDensityKernel << <numBlocks, numThreads >> > (d_GridDensity, d_density, d_mass, (float3*)d_pos, xResolution, yResolution, zResolution, domain_max_x, domain_max_y, domain_max_z, domain_min_x, domain_min_y, domain_min_z, step, h, numParticle);

	//�������
	cudaMemcpy(gridDensity, d_GridDensity, sizeof(float)*gridNum, cudaMemcpyDeviceToHost);

	//�ͷſռ�
	cudaFree(d_GridDensity);
	cudaFree(d_mass);
	cudaFree(d_pos);
	cudaFree(d_density);
}