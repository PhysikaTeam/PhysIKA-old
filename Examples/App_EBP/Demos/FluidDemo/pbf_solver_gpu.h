//#pragma once
#ifndef pbf_solver_gpu_h
#define pbf_solver_gpu_h


#include "Common\Common.h"
#include <memory>
#include <thrust\reduce.h>
#include <thrust\device_vector.h>

#include "pbf_solver_gpu_kernel.cuh"
#include "Demos\FluidDemo\FluidModel.h"

#include "helper_math.h"
#include <iostream>

//for file process
#include <io.h>
#include <direct.h>
typedef double Real;
typedef unsigned int uint;

//add .cu function
extern "C" void setParams(SimParams * hostParams);
extern "C" void calcHash(uint *gridParticleHash, uint *gridParticleIndex, float *particlePos, int particleNum);
extern "C" void sortParticlesByHash(uint *gridParticleHash, uint *gridParticleIndex, int particleNum);
extern "C" void reorderDataAndFindCellStart(uint *cellStart, uint *cellEnd, float *sortedPos, float *sortedVel, uint *gridParticheHash, uint *gridParticleIndex, float *oldPos, float * oldVel, int numParticles, int numCells);
extern "C" void GpuNeighborSearch(float * Positions, unsigned int *numNeighbors, unsigned int *neighbors, int numParticles, int maxNeighborsNum, float radius, float kernelRadius, uint *cellStart, uint *cellEnd, uint *gridParticleIndex);
extern "C" void integrateSystem(float *pos, float *vel, float deltaTime, uint numParticle);
extern "C" void _computeDensity(double * density, float * position, double * mass, unsigned int * numNeighbors, unsigned int *neighbors, unsigned int numParticles);
extern "C" void  _computePressure(double *pressure, double *density, unsigned int numParticles);
extern "C" void _computePressureForce(float *pressureForce, double *pressure, double *mass, double *density, float *position, unsigned int numParticles, unsigned int *numNeighbors, unsigned int *neighbors);
extern "C" void _computeBuoyanceByGradientOfTemperature(float *buoyance, double *temperature, float *position, double *density, double *mass, unsigned int numParticles, unsigned int *numNeighbors, unsigned int *neighbors);
extern "C" void _updateAccelerations(float *acceleration, float *buoyanceForce, double *mass, unsigned int numParticles);
extern "C" void _phaseTransition(double *newCloud, double *newVapor, double *newTemperature, double *temperatureChange, double *oldCloud, double *oldVapor, double *temperature, unsigned int numParticles);
extern "C" void _integrateSystemForCloud(float *newPos, float *newVel, float *oldPos, float *oldVel, float *acceleration, float deltaTime, uint numParticle);
extern "C" void _computeLambda(double *lambda, double *density, double *mass, float *pos, unsigned int *numNeighbor, unsigned int *neighbor, unsigned int numParticle);
extern "C" void _updatePosition(float* newPos, float* deltaPos, unsigned int numParticle);
extern "C" void _computeDeltaPositionByLambda(float *deltaPos, float *oldPos, double *lambda, double *mass, double *density, unsigned int *numNeighbor, unsigned int *neighbor, unsigned int numParticle);
extern "C" void _updateVelocity(float *newVel, float *oldPos, float *newPos, double deltime, unsigned int numParticle);
extern "C" void _checkParticleByDensity(int *flag, float *pos, double *density, double densityK, unsigned int numParticle);
extern "C" void _compactFlag(int *flag, float *pos, unsigned int numParticle);
extern "C" void _addParticles(float *addPos, unsigned int *addNum, float *pos, int *flag, unsigned int *cellstart, unsigned int *cellEnd, unsigned int *gridParticleIndex, double *density, double *mass, unsigned int numParticle);
extern "C" void _deleteParticle(unsigned int *deleteFlag, int *flag, float *pos, double *mass, double *density, uint *neighbor, uint *neighborNum, unsigned int numParticle);
extern "C" void _computeGradientOfColorField(float *gradientOfColorField, double *mass, float *pos, double *density, unsigned int numParticle, unsigned int *numNeighbor, unsigned int *neighbor);
extern "C" void _computeFeedbackForce(float *feedbackForce, double *density, double *signDistance, float *gradientOfSignDistance, float *gradienOfColorField, float *velocity, float *pos, unsigned int *numNeighbor,
	unsigned int *neighbor, unsigned int numParticle);

extern "C" void _checkParticleByDensityAndLocation(int *flag, float *pos, double *density, double densityK, double *signDistance, unsigned int *numNeighbor,unsigned int numParticle);
extern "C" void _computeDampingForce(float * dampForce, float *velocity, float *feedbackForce, double *signDistance, double *density, unsigned int numParticle);
extern "C" void _computeSumForce(float *sumForce, float *feedbackForce, float *dampingForce, unsigned int numParticle);
extern "C" void _updateMassForParticleAccordingSDF(double *mass, double *signDistance, double massForParInShape, double massForParOnSurface, float *pos, unsigned int numParticle);
extern "C" void _computeDensityAtTargetParticle(double *densityAtTargetParticle, float *targetParticlesPos, unsigned int numTarParticles, float *pos, double *mass, unsigned int numParticle, uint *cellStart, uint *cellEnd, uint *gridParticleIndex);
extern "C" void _computeForceFromTargetParticle(float *forceFromTargetParticle, float *targetParticlePos, double *densityAtTargetParticle, unsigned int numTargetParticles, double *signDistance, float *pos, unsigned int numParticles);
extern "C" void _computeSumForceForParticleControl(float *sumForce, float *feedbackForce, float *dampingForce, float *targetParticleAttractionForce, unsigned int numParticle);
extern "C" void _computeForceForControlParticel(float *sumForce, float *iniPos, float *tarPos, unsigned int *conIndex, unsigned int *tarIndex, unsigned int particleNum, unsigned int controlParticleNum);
extern "C" void _velocityInterpolation(float *velocity, float *pos, unsigned int *conIndex, unsigned int particleNum, unsigned int controlParticleNum);

extern "C" void _computeVelocityForControlTargetMethod(float *newVel, float *oldVel, float *acceleration, float deltaTime, uint numParticle);
extern "C" void _computePosForConTarMethod(float* newPos, float *oldPos, float *newVel, float deltaTime, unsigned int numParticle);

namespace pbf {
	class GpuParticleSolver
	{
	private:
		unsigned int m_numParticles;
		unsigned int m_maxNeighbors;
		unsigned int m_maxParticlesPerCell;
		unsigned int **m_neighbors; //存放每个粒子的邻域粒子的编号
		unsigned int *m_neighborsForOneDimStore;//1维方式存储邻域编号，zize= m_numParticles*m_maxNeighbors
		unsigned int *m_numNeighbors; //存放每个粒子的邻域个数
		Real m_particleRadius;
		Real m_cellGridSize; //网格单位长度
		Real m_radius2;
		Real m_kernelRadius;//核函数的支持半径
		int3 m_gridNumEachDim;//每一个维度上的网格数
		uint m_numGridCeslls; //总网格数
		unsigned int m_currentTimestamp; //时间戳
		
		float m_deltaTime;
		float3 *m_pos; //position
		float3 *m_vel; //velocity
		double *m_mass; //mass
		double *m_density;//density
		double *m_pressure; //pressure
		float3 *m_pressureForce; // force generated by pressure
		float3 *m_buoyanceForce; //force generated by buoyance
		double *m_temperature; //temperature
		double *m_cloud; //cloud
		double *m_vapor; //vapor
		double *m_lambda;

		double *m_signDistance;
		float3 *m_gradientOfSignDistance;

		unsigned int *m_deleteFlag; //存储当前粒子是否被删除, 1:delete
		unsigned int *m_addNum; //存储在当前粒子邻域内增加的粒子数量
		float3 *m_addPos; //存储在当前粒子邻域内增加的粒子位置

		//add & delete particle
		int *m_flag; //标记粒子需要的操作：|1：增加，0：不变，-1：删除|

		double m_standMassForPar; //记录粒子的标准质量
		unsigned int m_iniParSum;
		unsigned int m_iniSurParSum;
		unsigned int m_tarParSum;
		unsigned int m_tarSurParSum;
		unsigned int m_constParSum;
		//记录动量和及平均动量
		//std::vector<float2> m_momentum;

		//基于控制粒子法的演化控制
		//zzl----2019-1-16---start---
		float3* m_targetParticles;
		unsigned int m_numTargetParticles;
		//zzl----2019-1-16---end-----

		//设置控制粒子索引和目标粒子索引，相同位置为对应的两个粒子
		unsigned int *m_controlParticleIndex;
		unsigned int *m_targetParticleIndex;
		unsigned int m_controlParticleNum;
	protected:
		
		//GPU Data
		//particle properties
		double *m_dMass; //mass
		double *m_dDensity; //density
		float *m_dPose; //GPU存储位置
		float *m_dVel; //Gpu存储速度
		float *m_dAcceleration;//GPU存储加速度
		float *m_dNewPos;//GPU存储新的位置
		float *m_dNewVel;//GPU存储新的速度

		double *m_dPressure;//GPU存储压力
		float *m_dPressureForce;//GPU存储压力产生的力
		float *m_dBuoyanceForce;//GPU存储浮力
		double *m_dTemperature;//Gpu存储温度
		double *m_dNewTemperature; //GPU存储相变后或增加潜热后的新温度
		double *m_dTemperatureChange;//GPU存储相变引起的文度变化量
		double *m_dCloud;//GPU存储云的含量
		double *m_dNewCloud;//GPU存储相变后新的云的含量
		double *m_dVapor;//GPU存储蒸汽的含量
		double *m_dNewVapor;//GPU存储相变后新的蒸汽的含量
		double *m_dLambda;//GPU存储lambda
		float *m_dDeltaPos;//GPU存储△P

		double *m_dSignDistance;//GPU存储符号距离
		float *m_dGradientOfSignDistance;//GPU存储符号距离梯度
		float *m_dGradientOfColorField;//GPU存储color field梯度
		float *m_dFeedabckForce;//GPU存储反馈力
		float *m_dDampingForce;//GPU存储阻力
		float *m_dSumForce;//GPU存储合力

		uint *m_dGridParticleHash; //存储粒子所在的cell编号，大小为 m_numParticles;
		uint *m_dGridParticleIndex; //
		
		uint *m_dCellStart;
		uint *m_dCellEnd;

		float *m_dSortedPos;
		float *m_dSortedVel;

		//粒子数量更新
		int *m_dFlag; //GPU存储粒子的密度情况：1：>density0, 0=density0, -1:<density0
		unsigned int *m_dDeleteFlag; //GPU存储当前粒子是否被删除, 1:delete
		unsigned int *m_dAddNum; //GPU存储在当前粒子邻域内增加的粒子数量
		float *m_dAddPos; //GPU存储在当前粒子邻域内增加的粒子位置
		

		uint *m_dNeighborNum; //存储粒子的邻域粒子数
		uint *m_dNeighborForOneDimStore; //1维方式存储邻域编号，zize= m_numParticles*m_maxNeighbors
		
		//params
		SimParams m_params;

		//基于控制粒子法的演化控制
		//zzl----2019-1-16---start---
		float* m_dTargetParticles; //gpu 存储目标粒子位置
		double* m_dDensityAtTargetParticles; //目标粒子位置的密度值
		float* m_dForceFromTargetParticles;
		//zzl----2019-1-16---end-----

		//设置存放控制粒子索引的变量--zzl--2019-1-23--
		unsigned int *m_dControlParticleIndex;
		unsigned int *m_dTargetParticleIndex;

	public:

		GpuParticleSolver(unsigned int num_particles, double cellGridSize,double particleRadius,double kernelRadius,int grdiNum_x,int gridNum_y,int gridNum_z, Vector3r* particlePos, Vector3r* particleVel,PBD::FluidModel &model);
		~GpuParticleSolver();
		void initialize(Vector3r* particlePos, Vector3r* particleVel,Vector3r* pressureForce,Vector3r* buoyanceForce, Vector3r *gradientOfSignDistance);

		void initializeForParticleControl(PBD::FluidModel &model);//初始化控制粒子，将控制粒子拷贝到m_targetParticles
		
		
		//void cleanup();
		//void neighborhoodSearch(Vector3r *x);
		void neighborhoodSearch();
		//void update();
		unsigned int **getNeighbors() const;
		unsigned int *getNumNeighbors() const;
		const unsigned int getMaxNeighbors() const { return m_maxNeighbors; }

		unsigned int getNumParticles() const;
		void setRadius(const Real radius);
		Real getRadius() const;

		void finalize(); //cuda free
		void finalizeForParticleControl();//cuda Free

		void translateOneDim2TwoDimForNeighbors();

		void translateVector3rToFloat3(Vector3r * vData, float3 * fData, unsigned int dataSize); //将vector3r转换成float3
		std::vector<Vector3r> translateFloat3ToVector3r(float3 * data, unsigned int dataSize);

		void timeIntegrator(PBD::FluidModel &model); //时间积分

		void copyDataToModel(PBD::FluidModel &model);//将数据存储到model

		void computeDensity(); //计算密度
		void computePressure(); //计算压力
		void computePressureForce(); //计算压力产生的力
		void computeBuoyanceForce(); //计算浮力
		void updateAcceleration(); //更新加速度
		void updatePhase();//相变
		void timeIntegratorForCloud();//引入浮力后的时间积分
		void computeLambda();//计算lambda
		void computeDeltaPosition();//计算△p
		void updatePosition();//根据deltaPosition更新位置
		void updateVelocity();//根据更新后的位置更新速度
		void undatePostionAfterUpdateVelocity();//速度更新后，将m_dNewPos 赋值给m_dPos
		void enforecIncompressibility();//完成密度约束，实现不可压缩性
		void computeGradientOfColorField();//计算color field 梯度
		void computeFeedbackForce();//计算反馈力
		// 将数据从一个位置复制到另外一个位置
		
		void addOrDeleteParticle(); //根据密度情况进行粒子的增加和删除
		void cpuCompactFlag();

		void deleteParticle(PBD::FluidModel &model); //从model中删除粒子
		void addParticle(PBD::FluidModel &model);//在model中增加粒子

		float2 computeMomentum();//计算所有粒子的动量和及平均值

		// 存储数据
		void save(std::string &rootPath, std::string &fOriName1, std::string &fTarName2, double time, double timeStep);

		//存储演化过程的粒子数据用于渲染（.bin二进制格式、文本格式），平均动量和平均密度变化（.txt文本格式）--2019-3-5
		void saveForCompare(std::string &rootPath, std::string &iniName, std::string &tarName, double time, double timeStep);

		//计算平均密度
		double pbf::GpuParticleSolver::computeAvgDensity();

		//根据粒子位置更新质量
		void updateMassForParticle();

		//----------控制粒子-目标粒子法---------zzl---------------------
		void initializeControlAndTargetMethod(PBD::FluidModel &model); //初始化，为控制粒子-目标粒子法
		void updateAccelerationForControlTargetMethod();
		void timeIntergatorForControlTargetMethod();
		void simulateForConTarMethod(PBD::FluidModel &model);
		//---------------------------------------------------------------

		/*float3 * transVector3r2Float3(Vector3r * oldData, int num);*/
		FORCE_INLINE unsigned int n_neighbors(unsigned int i) const
		{
			return m_numNeighbors[i];
		}
		FORCE_INLINE unsigned int neighbor(unsigned int i, unsigned int k) const
		{
			return m_neighbors[i][k];
		}
	};
}
#endif // !pbf_solver_gpu_h
