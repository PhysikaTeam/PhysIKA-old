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
		unsigned int **m_neighbors; //���ÿ�����ӵ��������ӵı��
		unsigned int *m_neighborsForOneDimStore;//1ά��ʽ�洢�����ţ�zize= m_numParticles*m_maxNeighbors
		unsigned int *m_numNeighbors; //���ÿ�����ӵ��������
		Real m_particleRadius;
		Real m_cellGridSize; //����λ����
		Real m_radius2;
		Real m_kernelRadius;//�˺�����֧�ְ뾶
		int3 m_gridNumEachDim;//ÿһ��ά���ϵ�������
		uint m_numGridCeslls; //��������
		unsigned int m_currentTimestamp; //ʱ���
		
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

		unsigned int *m_deleteFlag; //�洢��ǰ�����Ƿ�ɾ��, 1:delete
		unsigned int *m_addNum; //�洢�ڵ�ǰ�������������ӵ���������
		float3 *m_addPos; //�洢�ڵ�ǰ�������������ӵ�����λ��

		//add & delete particle
		int *m_flag; //���������Ҫ�Ĳ�����|1�����ӣ�0�����䣬-1��ɾ��|

		double m_standMassForPar; //��¼���ӵı�׼����
		unsigned int m_iniParSum;
		unsigned int m_iniSurParSum;
		unsigned int m_tarParSum;
		unsigned int m_tarSurParSum;
		unsigned int m_constParSum;
		//��¼�����ͼ�ƽ������
		//std::vector<float2> m_momentum;

		//���ڿ������ӷ����ݻ�����
		//zzl----2019-1-16---start---
		float3* m_targetParticles;
		unsigned int m_numTargetParticles;
		//zzl----2019-1-16---end-----

		//���ÿ�������������Ŀ��������������ͬλ��Ϊ��Ӧ����������
		unsigned int *m_controlParticleIndex;
		unsigned int *m_targetParticleIndex;
		unsigned int m_controlParticleNum;
	protected:
		
		//GPU Data
		//particle properties
		double *m_dMass; //mass
		double *m_dDensity; //density
		float *m_dPose; //GPU�洢λ��
		float *m_dVel; //Gpu�洢�ٶ�
		float *m_dAcceleration;//GPU�洢���ٶ�
		float *m_dNewPos;//GPU�洢�µ�λ��
		float *m_dNewVel;//GPU�洢�µ��ٶ�

		double *m_dPressure;//GPU�洢ѹ��
		float *m_dPressureForce;//GPU�洢ѹ����������
		float *m_dBuoyanceForce;//GPU�洢����
		double *m_dTemperature;//Gpu�洢�¶�
		double *m_dNewTemperature; //GPU�洢���������Ǳ�Ⱥ�����¶�
		double *m_dTemperatureChange;//GPU�洢���������Ķȱ仯��
		double *m_dCloud;//GPU�洢�Ƶĺ���
		double *m_dNewCloud;//GPU�洢�����µ��Ƶĺ���
		double *m_dVapor;//GPU�洢�����ĺ���
		double *m_dNewVapor;//GPU�洢�����µ������ĺ���
		double *m_dLambda;//GPU�洢lambda
		float *m_dDeltaPos;//GPU�洢��P

		double *m_dSignDistance;//GPU�洢���ž���
		float *m_dGradientOfSignDistance;//GPU�洢���ž����ݶ�
		float *m_dGradientOfColorField;//GPU�洢color field�ݶ�
		float *m_dFeedabckForce;//GPU�洢������
		float *m_dDampingForce;//GPU�洢����
		float *m_dSumForce;//GPU�洢����

		uint *m_dGridParticleHash; //�洢�������ڵ�cell��ţ���СΪ m_numParticles;
		uint *m_dGridParticleIndex; //
		
		uint *m_dCellStart;
		uint *m_dCellEnd;

		float *m_dSortedPos;
		float *m_dSortedVel;

		//������������
		int *m_dFlag; //GPU�洢���ӵ��ܶ������1��>density0, 0=density0, -1:<density0
		unsigned int *m_dDeleteFlag; //GPU�洢��ǰ�����Ƿ�ɾ��, 1:delete
		unsigned int *m_dAddNum; //GPU�洢�ڵ�ǰ�������������ӵ���������
		float *m_dAddPos; //GPU�洢�ڵ�ǰ�������������ӵ�����λ��
		

		uint *m_dNeighborNum; //�洢���ӵ�����������
		uint *m_dNeighborForOneDimStore; //1ά��ʽ�洢�����ţ�zize= m_numParticles*m_maxNeighbors
		
		//params
		SimParams m_params;

		//���ڿ������ӷ����ݻ�����
		//zzl----2019-1-16---start---
		float* m_dTargetParticles; //gpu �洢Ŀ������λ��
		double* m_dDensityAtTargetParticles; //Ŀ������λ�õ��ܶ�ֵ
		float* m_dForceFromTargetParticles;
		//zzl----2019-1-16---end-----

		//���ô�ſ������������ı���--zzl--2019-1-23--
		unsigned int *m_dControlParticleIndex;
		unsigned int *m_dTargetParticleIndex;

	public:

		GpuParticleSolver(unsigned int num_particles, double cellGridSize,double particleRadius,double kernelRadius,int grdiNum_x,int gridNum_y,int gridNum_z, Vector3r* particlePos, Vector3r* particleVel,PBD::FluidModel &model);
		~GpuParticleSolver();
		void initialize(Vector3r* particlePos, Vector3r* particleVel,Vector3r* pressureForce,Vector3r* buoyanceForce, Vector3r *gradientOfSignDistance);

		void initializeForParticleControl(PBD::FluidModel &model);//��ʼ���������ӣ����������ӿ�����m_targetParticles
		
		
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

		void translateVector3rToFloat3(Vector3r * vData, float3 * fData, unsigned int dataSize); //��vector3rת����float3
		std::vector<Vector3r> translateFloat3ToVector3r(float3 * data, unsigned int dataSize);

		void timeIntegrator(PBD::FluidModel &model); //ʱ�����

		void copyDataToModel(PBD::FluidModel &model);//�����ݴ洢��model

		void computeDensity(); //�����ܶ�
		void computePressure(); //����ѹ��
		void computePressureForce(); //����ѹ����������
		void computeBuoyanceForce(); //���㸡��
		void updateAcceleration(); //���¼��ٶ�
		void updatePhase();//���
		void timeIntegratorForCloud();//���븡�����ʱ�����
		void computeLambda();//����lambda
		void computeDeltaPosition();//�����p
		void updatePosition();//����deltaPosition����λ��
		void updateVelocity();//���ݸ��º��λ�ø����ٶ�
		void undatePostionAfterUpdateVelocity();//�ٶȸ��º󣬽�m_dNewPos ��ֵ��m_dPos
		void enforecIncompressibility();//����ܶ�Լ����ʵ�ֲ���ѹ����
		void computeGradientOfColorField();//����color field �ݶ�
		void computeFeedbackForce();//���㷴����
		// �����ݴ�һ��λ�ø��Ƶ�����һ��λ��
		
		void addOrDeleteParticle(); //�����ܶ�����������ӵ����Ӻ�ɾ��
		void cpuCompactFlag();

		void deleteParticle(PBD::FluidModel &model); //��model��ɾ������
		void addParticle(PBD::FluidModel &model);//��model����������

		float2 computeMomentum();//�����������ӵĶ����ͼ�ƽ��ֵ

		// �洢����
		void save(std::string &rootPath, std::string &fOriName1, std::string &fTarName2, double time, double timeStep);

		//�洢�ݻ����̵���������������Ⱦ��.bin�����Ƹ�ʽ���ı���ʽ����ƽ��������ƽ���ܶȱ仯��.txt�ı���ʽ��--2019-3-5
		void saveForCompare(std::string &rootPath, std::string &iniName, std::string &tarName, double time, double timeStep);

		//����ƽ���ܶ�
		double pbf::GpuParticleSolver::computeAvgDensity();

		//��������λ�ø�������
		void updateMassForParticle();

		//----------��������-Ŀ�����ӷ�---------zzl---------------------
		void initializeControlAndTargetMethod(PBD::FluidModel &model); //��ʼ����Ϊ��������-Ŀ�����ӷ�
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
