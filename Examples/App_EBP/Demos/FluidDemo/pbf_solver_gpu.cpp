#include "pbf_solver_gpu.h"
#include "cusolverSp.h"
#include "ConvertParticlesToVoxel.h"
#include "vector_types.h"

//constructor
pbf::GpuParticleSolver::GpuParticleSolver(unsigned int num_particles, double cellGridSize, double particleRadius, double kernelRadius, int gridNum_x, int gridNum_y, int gridNum_z, Vector3r* particlePos, Vector3r* particleVel,PBD::FluidModel &model)
{
	m_deltaTime = 0.05; //Ĭ��ʱ����
	
	//initialize particles
	m_numParticles = num_particles;
	m_particleRadius = particleRadius;

	m_mass =&model.getParticles().getMass(0); //��ȡmodel�е�mass
	m_density = &model.getDensity(0); //��ȡmodel�е�density

	//----------zzl-------2019-1-14------start--------------
	//Ϊʵ���������̶�ǰ���£���ͬλ�����ӵ�������������
	m_standMassForPar = model.getStandMassForPar();
	m_iniParSum = model.getIniParSum();
	m_iniSurParSum = model.getIniSurParSum();
	m_tarParSum = model.getTarParSum();
	m_tarSurParSum = model.getTarSurParSum();
	m_constParSum = model.getConstParSum();
	//----------zzl-------2019-1-14------end----------------

	m_pressure = &model.getPressure(0); //��ȡmodel�е�pressure
	m_temperature = &model.getParticles().getTemperature(0); //��ȡmodel�е��¶���Ϣ

	m_cloud = &model.getParticles().getCloud(0); //��ȡģ���е�cloud
	m_vapor = &model.getParticles().getVapor(0);//��ȡģ���е�vapor

	m_signDistance = &model.getSignDistance(0); //��ȡģ���е�signDistance
	

	m_maxNeighbors = 40U; //test,����������

	m_flag = (int *)malloc(sizeof(int)*m_numParticles);

	m_lambda = (double *)malloc(sizeof(double)*m_numParticles); //���ڴ洢lambdaֵ

	m_deleteFlag = (unsigned int *)malloc(sizeof(unsigned int)*m_numParticles); //ΪdeleteFlag����ռ�
	m_addNum = (unsigned int *)malloc(sizeof(unsigned int)*m_numParticles);//ΪaddNum����ռ�
	m_addPos = (float3 *)malloc(sizeof(float3)*m_numParticles*m_maxNeighbors);//ΪaddPos����ռ�
	
    //initialize the grid
	m_cellGridSize = cellGridSize;
	m_radius2 = m_cellGridSize * m_cellGridSize;

	//���ݴ��ݵĲ�����������ռ��С
	/*m_gridNumEachDim.x = gridNum_x;
	m_gridNumEachDim.y = gridNum_y;
	m_gridNumEachDim.z = gridNum_z;*/

	
	//initialize the kernel radius
	m_kernelRadius = kernelRadius;
	
	//initialize the neighbor information
	if (num_particles != 0)
	{
		m_numNeighbors = new unsigned int[m_numParticles];
		m_neighborsForOneDimStore = new unsigned int[m_numParticles*m_maxNeighbors];
		m_neighbors = new unsigned int*[m_numParticles];
		
		#pragma omp parallel default(shared)
		{
			#pragma omp for schedule(static) 
			for (unsigned int i = 0; i < m_numParticles; i++)
			{
				m_neighbors[i] = new unsigned int[m_maxNeighbors];
				//��ʼ��ÿ�����ӵ��������Ϊ0
				m_numNeighbors[i] = 0;
			}
		}
	}
	
	//initialize the params
	m_params.maxNumOfNeighbors = m_maxNeighbors; //���������
	m_params.particleRadius = m_particleRadius;  //���Ӱ뾶

	m_params.cellSize = m_cellGridSize; //the length of cell = 2 * particleRadius

	//m_params.density0 = model.getDensity0();
	m_params.density0 = 1000.0; //����ʵ��������е���

	m_params.gasConstantK = 0.8f;

	//����ռ�ĸ���ά�ȵ���С����
	m_params.gridOrigin.x = -2; //-2~2
	m_params.gridOrigin.y = -2; //-2~4
	m_params.gridOrigin.z = -2; //-2~2
	//����ռ����ά�Ȱ���cell�ĸ���
	m_gridNumEachDim.x = ceil((2-(-2)) / m_params.cellSize);
	m_gridNumEachDim.y = ceil((4 - (-2)) / m_params.cellSize);
	m_gridNumEachDim.z = ceil((2 - (-2)) / m_params.cellSize);

	m_params.gridSize = m_gridNumEachDim;
	m_numGridCeslls = m_gridNumEachDim.x * m_gridNumEachDim.y*m_gridNumEachDim.z; //����ռ��cell���ܸ���
	m_params.numGridCells = m_numGridCeslls;
	
	m_params.kernelRadius = m_kernelRadius;

	m_params.acceleation.x = 0.0f;
	m_params.acceleation.y = 0.0003f;
	m_params.acceleation.z = 0.0f;

	//���Vector3r���͵����ݽ��д���
	// GPU����ռ�
	Vector3r *tempPressureForce = &model.getPressureForce(0);
	Vector3r *tempBuoyanceForce = &model.getBuoyant(0);
	Vector3r *tempGradientOfSignDistance = &model.getGradientOfSignDistance(0);

	//for (int i = 0; i <= 10; i++)
	//{
	//	//std::cout << tempGradientOfSignDistance[i].x() << std::endl;
	//	//std::cout << tempGradientOfSignDistance[i].y() << std::endl;
	//	//std::cout << tempGradientOfSignDistance[i].z() << std::endl;
	//	std::cout << m_signDistance[i] << std::endl;
	//}

	/*for (int i = 0; i < m_numParticles; i++)
	{
		m_mass[i] = 1.0;
	}*/
	//std::cout << m_mass[1] << std::endl;
	

	if (m_numParticles > 0)
	{
		initialize(particlePos,particleVel,tempPressureForce,tempBuoyanceForce,tempGradientOfSignDistance);
	}

	
}

//Ϊʵ�ֻ��ڿ������ӵ�������ƶ����еĳ�ʼ�����������������ӿ�������ǰ���󣬲�������Ӧ�ռ�
void pbf::GpuParticleSolver::initializeForParticleControl(PBD::FluidModel &model)
{
	Vector3r *targetParticles = &model.getTargetParticles(0);
	unsigned int sizeOfTragetParticles = model.getSizeOfTargetParticles();
	m_numTargetParticles = sizeOfTragetParticles;
	m_targetParticles = (float3*)malloc(sizeof(float3)*sizeOfTragetParticles);
	translateVector3rToFloat3(targetParticles, m_targetParticles, sizeOfTragetParticles);
	
	cudaMalloc((void**)&m_dTargetParticles, sizeof(float3)*sizeOfTragetParticles);
	cudaMalloc((void**)&m_dDensityAtTargetParticles, sizeof(double)*sizeOfTragetParticles);
	cudaMemcpy(m_dTargetParticles, m_targetParticles, sizeof(float3)*sizeOfTragetParticles, cudaMemcpyHostToDevice);

	//ÿ�����Ӵ�Ŀ����������õ���
	cudaMalloc((void**)&m_dForceFromTargetParticles, sizeof(float3)*m_numParticles);
}

//ʵ�ֻ��ڿ������Ӻ�Ŀ�����Ӷ�Ӧ��������״�ݻ���������ʼ��Ŀ������λ�ã���������������Ŀ����������
void pbf::GpuParticleSolver::initializeControlAndTargetMethod(PBD::FluidModel &model)
{
	Vector3r *targetParticles = &model.getTargetParticles(0);
	unsigned int sizeOfTragetParticles = model.getSizeOfTargetParticles();
	m_numTargetParticles = sizeOfTragetParticles;
	m_targetParticles = (float3*)malloc(sizeof(float3)*sizeOfTragetParticles);
	translateVector3rToFloat3(targetParticles, m_targetParticles, sizeOfTragetParticles);

	//���豸�з���ռ���Ŀ������λ�ã��������ݿ�����ȥ
	cudaMalloc((void**)&m_dTargetParticles, sizeof(float3)*sizeOfTragetParticles);
	cudaMemcpy(m_dTargetParticles, m_targetParticles, sizeof(float3)*sizeOfTragetParticles, cudaMemcpyHostToDevice);

	//��ȡģ���еĿ�������������Ŀ����������
	m_controlParticleIndex = &model.getConrolParticleIndex(0);
	m_targetParticleIndex = &model.getTargetParticleIndex(0);
	m_controlParticleNum = model.getSizeOfControlParticleIndex();

	//�ռ���估���ݿ���
	cudaMalloc((void**)&m_dControlParticleIndex, sizeof(unsigned int)*m_controlParticleNum);
	cudaMalloc((void**)&m_dTargetParticleIndex, sizeof(unsigned int)*m_controlParticleNum);
	cudaMemcpy(m_dControlParticleIndex, m_controlParticleIndex, sizeof(unsigned int)*m_controlParticleNum, cudaMemcpyHostToDevice);
	cudaMemcpy(m_dTargetParticleIndex, m_targetParticleIndex, sizeof(unsigned int)*m_controlParticleNum, cudaMemcpyHostToDevice);
}

pbf::GpuParticleSolver::~GpuParticleSolver()
{
	finalize();
	m_numParticles = 0;
}

void pbf::GpuParticleSolver::finalize()
{
	cudaFree(m_dCellEnd);
	cudaFree(m_dCellStart);

	cudaFree(m_dGridParticleHash);
	cudaFree(m_dGridParticleIndex);

	cudaFree(m_dNeighborNum);
	cudaFree(m_dNeighborForOneDimStore);

	cudaFree(m_dPose);
	cudaFree(m_dVel);

	cudaFree(m_dSortedPos);
	cudaFree(m_dSortedVel);

	cudaFree(m_dFlag);
	cudaFree(m_dDeleteFlag);

	cudaFree(m_dAddPos);
	cudaFree(m_dAddNum);

	cudaFree(m_dSignDistance);
	cudaFree(m_dGradientOfSignDistance);
	cudaFree(m_dGradientOfColorField);
	cudaFree(m_dFeedabckForce);
	
	cudaDeviceSynchronize();


}

//�ͷ�Ϊʵ�ֿ������ӽ�����״�ݻ�������Ŀռ�
void pbf::GpuParticleSolver::finalizeForParticleControl()
{
	//cudaFree(m_dTargetParticles);
	//cudaFree(m_dDensityAtTargetParticles);
	//cudaFree(m_dForceFromTargetParticles);

	//control-target-particle method
	cudaFree(m_dTargetParticles);
	cudaFree(m_dControlParticleIndex);
	cudaFree(m_dTargetParticleIndex);
}
void pbf::GpuParticleSolver::translateOneDim2TwoDimForNeighbors()
{
	#pragma omp parallel default(shared)
	{
		#pragma omp for schedule(static)
		for (int i = 0; i < m_numParticles; i++)
		{
			for (int j = 0; i < m_numNeighbors[i]; j++)
			{
				m_neighbors[i][j] = m_neighborsForOneDimStore[i*m_maxNeighbors + j];
			}
		}
	}
}

void pbf::GpuParticleSolver::translateVector3rToFloat3(Vector3r * vData, float3 * fData, unsigned int dataSize)
{

	#pragma omp parallel default(shared)
	{
		#pragma omp for schedule(static)
		for (unsigned int i = 0; i < dataSize; i++)
		{
			fData[i].x = vData[i].x();
			fData[i].y = vData[i].y();
			fData[i].z = vData[i].z();
		}
	}
}

//��float3����ת����std:vector<Vecotr3r>
std::vector<Vector3r> pbf::GpuParticleSolver::translateFloat3ToVector3r(float3 * data, unsigned int dataSize)
{
	std::vector<Vector3r> result;
	result.resize(dataSize);

	#pragma omp parallel default(shared)
	{
		#pragma omp for schedule(static)
		for (int i = 0; i < dataSize; i++)
		{
			result[i].x() = data[i].x;
			result[i].y() = data[i].y;
			result[i].z() = data[i].z;
		}
	}
	return result;
}

unsigned int ** pbf::GpuParticleSolver::getNeighbors() const
{
	return m_neighbors;
}

unsigned int * pbf::GpuParticleSolver::getNumNeighbors() const
{
	return m_numNeighbors;
}

unsigned int pbf::GpuParticleSolver::getNumParticles() const
{
	return m_numParticles;
}

void pbf::GpuParticleSolver::setRadius(const Real radius)
{
	m_cellGridSize = radius;
	m_radius2 = radius * radius;
}

Real pbf::GpuParticleSolver::getRadius() const
{
	return m_cellGridSize;
}

//float3 * pbf::GpuParticleNeighbors::transVector3r2Float3(Vector3r * oldData, int num)
//{
//	float3* newData;
//
//	return nullptr;
//}



void pbf::GpuParticleSolver::initialize(Vector3r* particlePos, Vector3r* particleVel, Vector3r* pressureForce, Vector3r* buoyanceForce,Vector3r *gradientOfSignDistance)
{
	/*--------------------------------------
	//            ����ռ�
	----------------------------------------*/
	//allocate cpu memory for Vector3r���͵�����
	m_pos = (float3 *)malloc(m_numParticles * sizeof(float3));
	m_vel = (float3 *)malloc(m_numParticles * sizeof(float3));

	m_gradientOfSignDistance = (float3 *)malloc(m_numParticles * sizeof(float3));

	//���Ƶ��ݻ������в���Ҫ��������
	/*m_pressureForce = (float3 *)malloc(m_numParticles * sizeof(float3));
	m_buoyanceForce = (float3 *)malloc(m_numParticles * sizeof(float3));*/

	translateVector3rToFloat3(particlePos, m_pos, m_numParticles);
	translateVector3rToFloat3(particleVel, m_vel, m_numParticles);
	//translateVector3rToFloat3(pressureForce, m_pressureForce, m_numParticles);
	//translateVector3rToFloat3(buoyanceForce, m_buoyanceForce, m_numParticles);
	translateVector3rToFloat3(gradientOfSignDistance, m_gradientOfSignDistance, m_numParticles);

	//for particle properties: |mass|density|pressure|pressureForce|temperature|buoyanceForce|accelerations
	cudaMalloc((void **)&m_dMass, sizeof(double)*m_numParticles);
	cudaMalloc((void **)&m_dDensity, sizeof(double)*m_numParticles);
	
	cudaMalloc((void **)&m_dPressure, sizeof(double)*m_numParticles);
	cudaMalloc((void **)&m_dPressureForce, sizeof(float3)*m_numParticles);
	
	cudaMalloc((void **)&m_dTemperature, sizeof(double)*m_numParticles);
	cudaMalloc((void **)&m_dBuoyanceForce, sizeof(float3)*m_numParticles);

	cudaMalloc((void **)&m_dAcceleration, sizeof(float3)*m_numParticles);
	//for particle properties: |cloud|newCloud|vapor|newVapor|newTemperature|temperatureChange|
	cudaMalloc((void**)&m_dCloud, sizeof(double)*m_numParticles);
	cudaMalloc((void **)&m_dNewCloud, sizeof(double)*m_numParticles);

	cudaMalloc((void **)&m_dVapor, sizeof(double)*m_numParticles);
	cudaMalloc((void **)&m_dNewVapor, sizeof(double)*m_numParticles);

	cudaMalloc((void **)&m_dNewTemperature, sizeof(double)*m_numParticles);
	cudaMalloc((void **)&m_dTemperatureChange, sizeof(double)*m_numParticles);

	//for particle properties: |newPos|newVel|lmbda|deltaPos|
	cudaMalloc((void **)&m_dNewPos, sizeof(float3)*m_numParticles);
	cudaMalloc((void **)&m_dNewVel, sizeof(float3)*m_numParticles);

	cudaMalloc((void **)&m_dLambda, sizeof(double)*m_numParticles);
	cudaMalloc((void **)&m_dDeltaPos, sizeof(float3)*m_numParticles);

	//for particle properties:|position|velocity|
	cudaMalloc((void**)&m_dPose, m_numParticles * sizeof(float3));
	cudaMalloc((void**)&m_dVel, m_numParticles * sizeof(float3));


	//for particle hash and index
	cudaMalloc((void **)&m_dGridParticleHash, m_numParticles * sizeof(uint));
	cudaMalloc((void**)&m_dGridParticleIndex, m_numParticles * sizeof(uint));
	//for gird cellStart & cellEnd
	cudaMalloc((void**)&m_dCellStart, m_numGridCeslls * sizeof(uint));
	cudaMalloc((void**)&m_dCellEnd, m_numGridCeslls * sizeof(uint));
	//for sorted position & velocity
	cudaMalloc((void**)&m_dSortedPos, m_numParticles * sizeof(float3));
	cudaMalloc((void**)&m_dSortedVel, m_numParticles * sizeof(float3));
	//for neighbor
	cudaMalloc((void**)&m_dNeighborForOneDimStore, m_numParticles * m_maxNeighbors*sizeof(uint));
	cudaMalloc((void**)&m_dNeighborNum, m_numParticles * sizeof(uint));

	//for add or delete particles
	//for |flag|deleteFlag|
	cudaMalloc((void **)&m_dFlag, sizeof(int)*m_numParticles);
	cudaMalloc((void **)&m_dDeleteFlag, sizeof(unsigned int)*m_numParticles);
	
	////for |addNum|addPos|
	cudaMalloc((void **)&m_dAddNum, sizeof(unsigned int)*m_numParticles);
	cudaMalloc((void **)&m_dAddPos, sizeof(float3)*m_numParticles*m_maxNeighbors);

	//for |signDistance|gradientOfSignDistacne|gradientOfColorField|feedbackForce|dampingForce|sumForce
	cudaMalloc((void **)&m_dSignDistance, sizeof(double)*m_numParticles);
	cudaMalloc((void **)&m_dGradientOfSignDistance, sizeof(float3)*m_numParticles);
	cudaMalloc((void **)&m_dGradientOfColorField, sizeof(float3)*m_numParticles);
	cudaMalloc((void **)&m_dFeedabckForce, sizeof(float3)*m_numParticles);
	cudaMalloc((void **)&m_dDampingForce, sizeof(float3)*m_numParticles);
	cudaMalloc((void **)&m_dSumForce, sizeof(float3)*m_numParticles);
	/*--------------------------------------
	//            ��������
	----------------------------------------*/
	// �����ݼ��ص��Դ�
	cudaMemcpy(m_dPose, m_pos, m_numParticles * sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpy(m_dVel, m_vel, m_numParticles * sizeof(float3), cudaMemcpyHostToDevice);

	//cudaMemcpy(m_dDensity, m_density, m_numParticles * sizeof(double), cudaMemcpyHostToDevice);

	
	//test---��������--2018-11-11
	//std::cout << "capacity test" << std::endl;
	//double * dTest;
	//double * test;
	//test = (double *)malloc(sizeof(double) * 600000);
	//cudaMalloc((void **)&dTest, sizeof(double) * 600000);
	//cudaMemcpy(dTest, test, 60000 * sizeof(double), cudaMemcpyHostToDevice);
	//end---��������--2018-11-11

	cudaMemcpy(m_dMass, m_mass, m_numParticles * sizeof(double), cudaMemcpyHostToDevice);

	//cudaMemcpy(m_dPressure, m_pressure, m_numParticles * sizeof(double), cudaMemcpyHostToDevice);
	//cudaMemcpy(m_dPressureForce, m_pressureForce, m_numParticles * sizeof(float3), cudaMemcpyHostToDevice);

	//cudaMemcpy(m_dTemperature, m_temperature, m_numParticles * sizeof(double), cudaMemcpyHostToDevice);
	//cudaMemcpy(m_dBuoyanceForce, m_buoyanceForce, m_numParticles * sizeof(float3), cudaMemcpyHostToDevice);


	

	cudaMemcpy(m_dCloud, m_cloud, m_numParticles * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(m_dVapor, m_vapor, m_numParticles * sizeof(double), cudaMemcpyHostToDevice);

	cudaMemcpy(m_dSignDistance, m_signDistance, m_numParticles * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(m_dGradientOfSignDistance, m_gradientOfSignDistance, m_numParticles * sizeof(float3), cudaMemcpyHostToDevice);



	//set params
	setParams(&m_params);
}

void pbf::GpuParticleSolver::neighborhoodSearch()
{
	//copyȫ�ֲ���params
	//setParams(&m_params);
	//calculate hash value
	//std::cout << "=>=>=>=>=>=>=>=>=>" << m_numParticles << std::endl;
	

	calcHash(m_dGridParticleHash, m_dGridParticleIndex, m_dPose, m_numParticles);
	/*unsigned int * tempHash;
	tempHash = (unsigned int *)malloc(sizeof(unsigned int)*m_numGridCeslls);
	cudaMemcpy(tempHash, m_dGridParticleHash, sizeof(unsigned int)*m_numGridCeslls, cudaMemcpyDeviceToHost);
	std::cout << tempHash[1230] << std::endl;*/

	
	//std::cout << "==================================" << std::endl;
	//sort particle based on hash
	sortParticlesByHash(m_dGridParticleHash, m_dGridParticleIndex, m_numParticles);
	
	//fine start and end of each cell
	reorderDataAndFindCellStart(m_dCellStart, m_dCellEnd, m_dSortedPos, m_dSortedVel, m_dGridParticleHash, m_dGridParticleIndex, m_dPose, m_dVel, m_numParticles, m_numGridCeslls);

	//find neighbors for each particle
	GpuNeighborSearch(m_dPose, m_dNeighborNum, m_dNeighborForOneDimStore, m_numParticles, m_maxNeighbors, m_particleRadius, m_kernelRadius, m_dCellStart, m_dCellEnd, m_dGridParticleIndex);

	// copy neighbors information 
	//cudaMemcpy((float *)x, m_dPose, sizeof(float3)*m_numParticles, cudaMemcpyDeviceToHost);
	cudaMemcpy(m_numNeighbors, m_dNeighborNum, sizeof(uint)*m_numParticles,cudaMemcpyDeviceToHost);
	cudaMemcpy(m_neighborsForOneDimStore, m_dNeighborForOneDimStore, sizeof(uint)*m_numParticles*m_maxNeighbors, cudaMemcpyDeviceToHost);

	//std::cout << "nieghbor=====ok" << std::endl;
}

void pbf::GpuParticleSolver::timeIntegrator(PBD::FluidModel &model)
{
	//cudaMemcpy(m_dPose, m_pos, m_numParticles * sizeof(float3), cudaMemcpyHostToDevice);
	//cudaMemcpy(m_dVel, m_vel, m_numParticles * sizeof(float3), cudaMemcpyHostToDevice);
	
	//for test
	//std::cout << m_numParticles << std::endl;
	//std::cout << "p----0--" << m_vel[100].y << std::endl;

	integrateSystem(m_dPose, m_dVel, m_deltaTime, m_numParticles);
	
	cudaMemcpy(m_pos, m_dPose, sizeof(float3)*m_numParticles, cudaMemcpyDeviceToHost);
	cudaMemcpy(m_vel, m_dVel, sizeof(float3)*m_numParticles, cudaMemcpyDeviceToHost);
	//std::cout << "p----1--" << m_vel[100].y << std::endl;

	//std::cout << "==========" << std::endl;
}



void pbf::GpuParticleSolver::copyDataToModel(PBD::FluidModel & model)
{
	////for test: output the original postion and velocity 
	//std::cout << "before update: p=" << model.getParticles().getPosition(10) << std::endl;
	//std::cout << "before update: v=" << model.getParticles().getVelocity(10) << std::endl;
	
	//copy position and velocity
	model.getParticles().getAllPostion() = translateFloat3ToVector3r(m_pos, m_numParticles);
	model.getParticles().getAllVel() = translateFloat3ToVector3r(m_vel, m_numParticles);
	
	////for test: output the positon and velocity after implementing GPU_Solver
	//std::cout << "after update: p=" << model.getParticles().getPosition(10) << std::endl;
	//std::cout << "after update: v=" << model.getParticles().getVelocity(10) << std::endl;

	//copy neighborsNum and neighobrs

	/*unsigned int *tempNeighborsNum = model.getNeighborhoodSearch()->getNumNeighbors();
	unsigned int **tempNeighbors = model.getNeighborhoodSearch()->getNeighbors();

	std::cout << tempNeighborsNum[m_numParticles] << std::endl;
	#pragma omp parallel default(shared)
	{
		#pragma omp for schedule(static) 
		for (int i = 0; i < m_numParticles; i++)
		{
			tempNeighborsNum[i] = m_numNeighbors[i];
			for (int j = 0; j < m_numNeighbors[i]; j++)
			{
				tempNeighbors[i][j] = m_neighborsForOneDimStore[i*m_maxNeighbors + j];
			}
		}
	}*/
}

void pbf::GpuParticleSolver::computeDensity()
{
	_computeDensity(m_dDensity, m_dPose, m_dMass, m_dNeighborNum, m_dNeighborForOneDimStore, m_numParticles);
	cudaMemcpy(m_density, m_dDensity, sizeof(double)*m_numParticles, cudaMemcpyDeviceToHost);
	double sumDen = 0.0;
	double maxD = -10;
	double minD = 100;
	for (int i = 0; i < m_numParticles; i++)
	{
		sumDen += m_density[i];
		if (m_density[i] < minD)
			minD = m_density[i];
		if (m_density[i] > maxD)
			maxD = m_density[i];
	}

	std::cout << "avgDensity=" << sumDen/m_numParticles<<std::endl;
	std::cout << "maxD=" << maxD << "    minD=" << minD << std::endl;
}

// compute pressure 
void pbf::GpuParticleSolver::computePressure()
{
	_computePressure(m_dPressure, m_dMass, m_numParticles);
}

// compute pressure force
void pbf::GpuParticleSolver::computePressureForce()
{
	_computePressureForce(m_dPressureForce, m_dPressure, m_dMass, m_dDensity, m_dPose, m_numParticles, m_dNeighborNum, m_dNeighborForOneDimStore);
}

void pbf::GpuParticleSolver::computeBuoyanceForce()
{
	_computeBuoyanceByGradientOfTemperature(m_dBuoyanceForce, m_dTemperature, m_dPose, m_dDensity, m_dMass, m_numParticles, m_dNeighborNum, m_dNeighborForOneDimStore);
}

void pbf::GpuParticleSolver::updateAcceleration()
{
	//���Ǹ������˶�
	//_updateAccelerations(m_dAcceleration, m_dBuoyanceForce, m_dMass, m_numParticles);

	//������״�������������������

	_computeSumForce(m_dSumForce, m_dFeedabckForce, m_dDampingForce, m_numParticles);
	
	//����Ŀ������������
	//_computeSumForceForParticleControl(m_dSumForce, m_dFeedabckForce, m_dDampingForce, m_dForceFromTargetParticles, m_numParticles);
	_updateAccelerations(m_dAcceleration, m_dSumForce, m_dMass, m_numParticles);

	//ֻ������״�������������������������
	//_updateAccelerations(m_dAcceleration, m_dFeedabckForce, m_dMass, m_numParticles);
}
	
//---------��������-Ŀ�����ӷ�------zzl------2019-1-12---start-----------------------------------
//------���ݿ����������������¼��ٶ�----------------------------
void pbf::GpuParticleSolver::updateAccelerationForControlTargetMethod()
{
	//�����������������
	_computeForceForControlParticel(m_dSumForce, m_dPose, m_dTargetParticles, m_dControlParticleIndex, m_dTargetParticleIndex, m_numParticles, m_controlParticleNum);
	//�������������¼��ٶ�
	_updateAccelerations(m_dAcceleration, m_dSumForce, m_dMass, m_numParticles);
}


//ͨ��ʱ����ָ���λ��
void pbf::GpuParticleSolver::timeIntergatorForControlTargetMethod()
{
	//���ݿ����������ܵ�������������ӵ����ٶ�
	_computeVelocityForControlTargetMethod(m_dNewVel, m_dVel, m_dAcceleration, m_deltaTime, m_numParticles);
	//ͨ����ֵ��÷ǿ������ӵ����ٶ�
	_velocityInterpolation(m_dNewVel, m_dPose, m_dControlParticleIndex, m_numParticles, m_controlParticleNum);
	//�������ٶȼ������ӵ�λ��
	_computePosForConTarMethod(m_dNewPos, m_dPose, m_dNewVel, m_deltaTime, m_numParticles);

	cudaMemcpy(m_dPose, m_dNewPos, sizeof(float3)*m_numParticles, cudaMemcpyDeviceToDevice);
	cudaMemcpy(m_dVel, m_dNewVel, sizeof(float3)*m_numParticles, cudaMemcpyDeviceToDevice);
}


void pbf::GpuParticleSolver::simulateForConTarMethod(PBD::FluidModel &model)
{
	initializeControlAndTargetMethod(model);
	//���������ܶȼ���
	neighborhoodSearch();
	computeDensity();
	//�����ٶȺ�λ��
	updateAccelerationForControlTargetMethod(); //�����������¼��ٶ�
	timeIntergatorForControlTargetMethod(); //�������ٶȲ�����λ��
	//����Ѱ������
	neighborhoodSearch();
	computeDensity();
	//ʵʩ����ѹ����
	enforecIncompressibility();
	//ʵʩ����ѹ��֮����ٶȸ��£� v=(newPos-oldPos)/timeStep
	updateVelocity();
	//����λ�ø�ֵ���洢λ�õı�����oldPos=newPos
	undatePostionAfterUpdateVelocity();

	copyDataToModel(model);

	//�ͷ���Դ
	//finalize();
	//finalizeForParticleControl();
}
//-----------------------------------zzl---------------2019-1-24-----end------------------

void pbf::GpuParticleSolver::updatePhase()
{
	_phaseTransition(m_dNewCloud, m_dNewVapor, m_dNewTemperature, m_dTemperatureChange, m_dCloud, m_dVapor, m_dTemperature, m_numParticles);
}


//newV[i] = oldV[i] + ��t*acceleration[i]
//newPos[i] = oldPos[i] + ��t*v[i]
void pbf::GpuParticleSolver::timeIntegratorForCloud()
{
	_integrateSystemForCloud(m_dNewPos, m_dNewVel, m_dPose, m_dVel, m_dAcceleration, m_deltaTime, m_numParticles);
	cudaMemcpy(m_dPose, m_dNewPos, sizeof(float3)*m_numParticles, cudaMemcpyDeviceToDevice);
	cudaMemcpy(m_dVel, m_dNewVel, sizeof(float3)*m_numParticles, cudaMemcpyDeviceToDevice);
	//����λ�õ����ڴ�
	//cudaMemcpy(m_vel, m_dNewVel, sizeof(float3)*m_numParticles, cudaMemcpyDeviceToHost);
	//std::cout << "ok---vel" << std::endl;
	//cudaMemcpy(m_pos, m_dNewPos, sizeof(float3)*m_numParticles, cudaMemcpyDeviceToHost);
	//
	//std::cout << "ok---pos" << std::endl;
}



//compute lambda_i according to newPos[i] 
void pbf::GpuParticleSolver::computeLambda()
{
	_computeLambda(m_dLambda, m_dDensity, m_dMass, m_dNewPos, m_dNeighborNum, m_dNeighborForOneDimStore, m_numParticles);

	/*double *m_lambda;
	m_lambda = (double *)malloc(sizeof(double)*m_numParticles);
	cudaMemcpy(m_lambda, m_dDeltaPos, sizeof(double)*m_numParticles, cudaMemcpyDeviceToHost);
	cudaMemcpy(m_density, m_dDensity, sizeof(double)*m_numParticles, cudaMemcpyDeviceToHost);
	for (int i = 0; i < m_numParticles; i++)
	{
		double d;
		d = m_lambda[i];
		if (d < -3 || d>3)
		{
			std::cout << i << "= " << d << std::endl;
			std::cout << i << ": density= " << m_density[i] << std::endl;
		}
			
	}*/
}

//compute delatPosition_i according lambda[i] and newPos[i]
void pbf::GpuParticleSolver::computeDeltaPosition()
{
	_computeDeltaPositionByLambda(m_dDeltaPos, m_dNewPos, m_dLambda, m_dMass, m_dDensity, m_dNeighborNum, m_dNeighborForOneDimStore, m_numParticles);
	
	//����deltaPos��ֵ�Ƿ�ܴ�
	/*float3 *m_delPos;
	m_delPos = (float3 *)malloc(sizeof(float3)*m_numParticles);
	cudaMemcpy(m_delPos, m_dDeltaPos, sizeof(float3)*m_numParticles, cudaMemcpyDeviceToHost);
	for (int i = 0; i < m_numParticles; i++)
	{
		float x, y, z,d;
		x = m_delPos[i].x;
		y = m_delPos[i].y;
		z = m_delPos[i].z;
		d = sqrt(x*x + y*y + z*z);
		if (d > 2)
			std::cout << i << "= " << d << std::endl;
	}*/
}

//update positon according to deltaPosition[i]
void pbf::GpuParticleSolver::updatePosition()
{
	_updatePosition(m_dNewPos, m_dDeltaPos, m_numParticles);

}

//newV = (newPos[i] - oldPos[i])/��t
void pbf::GpuParticleSolver::updateVelocity()
{
	_updateVelocity(m_dVel, m_dPose, m_dNewPos, m_deltaTime, m_numParticles);
	cudaMemcpy(m_vel, m_dNewVel, sizeof(float3)*m_numParticles, cudaMemcpyDeviceToHost);
	cudaMemcpy(m_dVel, m_dNewVel, sizeof(float3)*m_numParticles, cudaMemcpyDeviceToDevice);
}

//
void pbf::GpuParticleSolver::undatePostionAfterUpdateVelocity()
{
	cudaMemcpy(m_dPose, m_dNewPos, sizeof(float3)*m_numParticles, cudaMemcpyDeviceToDevice);
	cudaMemcpy(m_pos, m_dNewPos, sizeof(float3)*m_numParticles, cudaMemcpyDeviceToHost);
}

void pbf::GpuParticleSolver::enforecIncompressibility()
{
	int maxIter = 8; //���õ�������
	for (int i = 0; i < maxIter; i++)
	{
		computeDensity();
		computeLambda();
		
	}
	computeDeltaPosition();
	updatePosition();
	
}

/*�����ܶȼ���color field�ݶ�*/
void pbf::GpuParticleSolver::computeGradientOfColorField()
{
	_computeGradientOfColorField(m_dGradientOfColorField, m_dMass, m_dPose, m_dDensity, m_numParticles, m_dNeighborNum, m_dNeighborForOneDimStore);

	//std::cout << "colorfield=====ok" << std::endl;
}

/*���ݼ����Ƴ��ݶȡ��ܶȲgradient of color field���㷴���� 
*/
void pbf::GpuParticleSolver::computeFeedbackForce()
{
	//����������
	_computeFeedbackForce(m_dFeedabckForce, m_dDensity, m_dSignDistance, m_dGradientOfSignDistance, m_dGradientOfColorField, m_dVel, m_dPose,m_dNeighborNum,m_dNeighborForOneDimStore, m_numParticles);

	
	//--------------ʵ�ֿ������ӷ�-----zzl-------2019-1-17--------start-----------
	//�ܶȼ���
	//_computeDensityAtTargetParticle(m_dDensityAtTargetParticles, m_dTargetParticles, m_numTargetParticles, m_dPose, m_dMass, m_numParticles, m_dCellStart, m_dCellEnd, m_dGridParticleIndex);
	
	//�������ӵ�����������
	//_computeForceFromTargetParticle(m_dForceFromTargetParticles, m_dTargetParticles, m_dDensityAtTargetParticles, m_numTargetParticles, m_dSignDistance, m_dPose, m_numParticles);
	//--------------ʵ�ֿ������ӷ�-----zzl-------2019-1-17--------end-----------

	//��������
	//std::cout << "x=" <<m_vel[1000].x << "  y=" << m_vel[1000].y << "  z=" << m_vel[1000].z << std::endl;
	_computeDampingForce(m_dDampingForce, m_dVel, m_dFeedabckForce, m_dSignDistance, m_dDensity, m_numParticles);
	//std::cout << "feedbackForce=====ok" << std::endl;
}

void pbf::GpuParticleSolver::addOrDeleteParticle()
{
	float densityK = 0.1; //�ܶȲ��Կ��Ʋ����� $\zeta$
	//test density
	//_checkParticleByDensity(m_dFlag, m_dPose, m_dDensity, densityK, m_numParticles); //���������ܶȽ��в���
	_checkParticleByDensityAndLocation(m_dFlag, m_dPose, m_dDensity, densityK, m_dSignDistance, m_dNeighborNum ,m_numParticles);
	//compact flag
	//_compactFlag(m_dFlag, m_dPose, m_numParticles);
	cpuCompactFlag();
	//delete particles

	_deleteParticle(m_dDeleteFlag, m_dFlag, m_dPose, m_dMass, m_dDensity,m_dNeighborForOneDimStore, m_dNeighborNum, m_numParticles);
	//add particles
	
	_addParticles(m_dAddPos, m_dAddNum, m_dPose, m_dFlag, m_dCellStart, m_dCellEnd, m_dGridParticleIndex, m_dDensity, m_dMass, m_numParticles);
}


void pbf::GpuParticleSolver::cpuCompactFlag()
{
	cudaMemcpy(m_flag, m_dFlag, sizeof(int)*m_numParticles, cudaMemcpyDeviceToHost);
	#pragma omp parallel default(shared)
	{
		#pragma omp for schedule(static)  
		for (int i = 0; i < m_numParticles; i++)
		{
			if (0 == m_flag[i]) continue;
			float3 pos_i = m_pos[i];
			if (1 == m_flag[i])
			{
				for (int j = i + 1; j < m_numParticles; j++)
				{
					if (1 != m_flag[1]) continue;
					float3 pos_j = m_pos[j];
					float3 pos = pos_i - pos_j;
					float dist_ij;
					dist_ij = length(pos);
					if (dist_ij < 2 * m_particleRadius)
					{
						m_flag[j] = 0;
					}
				}
			}//end if (1 == flag[i])
			else // flag[i] == -1
			{
				for (int k = i + 1; k < m_numParticles; k++)
				{
					if (-1 != m_flag[k]) continue;
					float3 pos_k = m_pos[k];
					float3 pos = pos_i - pos_k;
					float dist_ik;
					dist_ik = length(pos);
					if (dist_ik < 2 * m_particleRadius)
					{
						m_flag[k] = 0;
					}
				}
			}//end else
		}//end for (int i = 0; i < numParticle; i++)
	}
	cudaMemcpy(m_dFlag, m_flag, sizeof(int)*m_numParticles, cudaMemcpyHostToDevice);
}

void pbf::GpuParticleSolver::deleteParticle(PBD::FluidModel & model)
{

	cudaMemcpy(m_deleteFlag, m_dDeleteFlag, sizeof(unsigned int)*m_numParticles, cudaMemcpyDeviceToHost);
	//ɾ�����ӣ���Ҫ�Ӻ���ǰɾ��
	for (int i = m_numParticles-1; i >= 0; i--)
	{
		if(m_deleteFlag[i]==1)
			model.getParticles().deleteVertex(i); //ɾ������
	}
	
}

//���������������ӵ�ģ����
void pbf::GpuParticleSolver::addParticle(PBD::FluidModel & model)
{
	cudaMemcpy(m_addNum, m_dAddNum, sizeof(unsigned int)*m_numParticles, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cudaMemcpy(m_addPos, m_dAddPos, sizeof(float3)*m_numParticles*m_maxNeighbors, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	
	float percentOfNum;
	percentOfNum = 0.2; //����ÿ������������ӵı���������任̫����
	
	unsigned int maxAddNum =(unsigned int) ceil(percentOfNum * m_numParticles);

	int addNum = 0;
	//if (m_numParticles < 5000)
	{
		//std::cout << m_addPos[10].x << std::endl;
		#pragma omp parallel default(shared)
		{
			#pragma omp for schedule(static)  
			for (int i = 0; i < m_numParticles; i++)
			{
				if (m_addNum[i] > 0)
				{
					//std::cout << m_addNum[i] << std::endl;
					for (int j = 0; j < m_addNum[i]; j++)
					{
						float3 pos = m_addPos[i*m_maxNeighbors + j];
						Vector3r posV;
						posV.x() = pos.x;
						posV.y() = pos.y;
						posV.z() = pos.z;
						model.getParticles().addVertex(posV);
						addNum++;
						//std::cout << model.getParticles().size() << std::endl;
					}
				}
				if (addNum > maxAddNum)
				{
					break;
				}
			}
		}
	}
	
	//std::cout << model.getParticles().size() << std::endl;
	//std::cout << c << std::endl;
}

//���㶯��
float2 pbf::GpuParticleSolver::computeMomentum()
{
	float2 momentum; //momentum.x�洢�ܶ�����||mv||����momentum.y�洢ƽ������||mv||/N
	momentum.x = 0.0; //��¼�ܵĶ���
	momentum.y = 0.0; //��¼ƽ������
	float e = 1e-6;
	for (int i = 0; i < m_numParticles; i++)
	{
		float3 vel = m_vel[i];
		float mag_vel; //�ٶȵĴ�С
		mag_vel = sqrt(vel.x*vel.x + vel.y*vel.y + vel.z*vel.z);
		momentum.x += mag_vel*m_mass[i];
	}
	momentum.y = momentum.x / (m_numParticles+e);
	//std::cout << "momenturm:  " << momentum.x <<"  " << momentum.y << std::endl;
	return momentum;
}

/* ������������ļ�
fOriName:��ʼ��״���ļ���
fTarName:Ŀ����״���ļ���
time: ִ�е�ʱ��
timeStep:ʱ�䲽��
*/
void pbf::GpuParticleSolver::save(std::string &rootPath, std::string &fOriName, std::string &fTarName, double time, double timeStep)
{
	int nFrame; //��ǰ�ǵڼ�֡
	nFrame = int(time / timeStep); 

	char fileName[256];
	char velocityName[256];
	std::string fullFileName;
	std::string fullGTName;
	//std::string path = "DataForLearning\\";
	std::string path("");
	path = rootPath + std::string("\\");
	
	//����nFrame�����ļ���(�磺000001.bin)
	sprintf_s(fileName, "%06d.bin", nFrame);
	//����nFrame�����ٶȵ�GT�ļ�
	sprintf_s(velocityName, "%06dVGT.bin", nFrame);
	//��·�����ļ����ϲ����������ļ���
	fullFileName = path +fOriName+"_"+fTarName+"\\"+ fileName;  //���ÿ֡������Ϣ
	fullGTName = path + fOriName + "_" + fTarName + "\\" + velocityName; //ֻ����ٶ���Ϣ����Ϊground truth
	//����·��
	char *tag = new char[256];
	strcpy(tag, fullFileName.c_str());
	for (; *tag; tag++)
	{
		if (*tag == '\\')
		{
			char buf[100], path[100];
			strcpy(buf, fullFileName.c_str());
			buf[strlen(fullFileName.c_str()) - strlen(tag) + 1] = NULL;
			strcpy(path, buf);
			if (_access(path, 6) == -1)
			{
				_mkdir(path);
			}
		}
	}

	//д�ļ�
	FILE *fp,*fGT;
	if ((fp = fopen(fullFileName.c_str(), "wb")) == NULL)
	{
		std::cout << "open file failed" << std::endl;
		return;
	}
	if ((fGT = fopen(fullGTName.c_str(), "wb")) == NULL)
	{
		std::cout << "open GT-file failed" << std::endl;
		return;
	}
	/* ---------------ÿ֡������Ϣ-�ļ���ʽ˵��---------------------------------------
	��������N
	���������
	��һ�����ӣ�����λ��x��y,z �����ܶ�\rho ���ӵķ��ž���sdf ���ӵ��ٶ�u,v,w ���ӵ�������n  �������ӵ���Ϣ
	�ڶ������ӣ�����λ��x��y,z �����ܶ�\rho ���ӵķ��ž���sdf ���ӵ��ٶ�u,v,w ���ӵ�������n  �������ӵ���Ϣ
	...	...	...	...	...	...
	��N�����ӣ�����λ��x��y,z �����ܶ�\rho ���ӵķ��ž���sdf ���ӵ��ٶ�u,v,w ���ӵ�������n  �������ӵ���Ϣ
	-------------------------------------------------------------------*/

	//д������ --unsigned int
	fwrite(&m_numParticles, sizeof(m_numParticles), 1, fp);
	fwrite(&m_numParticles, sizeof(m_numNeighbors), 1, fGT);
	//д���������-- unsigned int
	fwrite(&m_maxNeighbors, sizeof(m_maxNeighbors), 1, fp);
	
	//if (nFrame == 0)
	//{
	//	for (int i = 0; i < m_numParticles; i++)
	//	{
	//		if(m_numNeighbors[i]>20)
	//			std::cout << m_numNeighbors[i] << std::endl;
	//	}
	//}

	for (int i = 0; i < m_numParticles; i++)
	{
		// write position-- float
		
		fwrite(&m_pos[i].x, sizeof(m_pos[i].x), 1, fp);
		fwrite(&m_pos[i].y, sizeof(m_pos[i].y), 1, fp);
		fwrite(&m_pos[i].z, sizeof(m_pos[i].z), 1, fp);
		//write density-- double
		fwrite(&m_density[i], sizeof(m_density[i]), 1, fp);
		//write sdf-- double
		//std::cout << i<<"--: "<<m_signDistance[i]  << std::endl; //for test
		fwrite(&m_signDistance[i], sizeof(m_signDistance[i]), 1, fp);
		//write velocity--float
		fwrite(&m_vel[i].x, sizeof(m_vel[i].x), 1, fp);
		fwrite(&m_vel[i].y, sizeof(m_vel[i].y), 1, fp);
		fwrite(&m_vel[i].z, sizeof(m_vel[i].z), 1, fp);

		fwrite(&m_vel[i].x, sizeof(m_vel[i].x), 1, fGT);
		fwrite(&m_vel[i].y, sizeof(m_vel[i].y), 1, fGT);
		fwrite(&m_vel[i].z, sizeof(m_vel[i].z), 1, fGT);
		//write velocity to GTFile--float
		//write the num of neighbor--unsigned int
		fwrite(&m_numNeighbors[i], sizeof(m_numNeighbors[i]), 1, fp);
		
		//write neighobr information
		int j;
		for (j = 0; j < m_numNeighbors[i]; j++)
		{
			//get the index of the jth neighbor
			unsigned int nborIndex = m_neighborsForOneDimStore[i*m_maxNeighbors + j];

			// write position-- float
			fwrite(&m_pos[nborIndex].x, sizeof(m_pos[nborIndex].x), 1, fp);
			fwrite(&m_pos[nborIndex].y, sizeof(m_pos[nborIndex].y), 1, fp);
			fwrite(&m_pos[nborIndex].z, sizeof(m_pos[nborIndex].z), 1, fp);
			//write density-- double
			fwrite(&m_density[nborIndex], sizeof(m_density[nborIndex]), 1, fp);
			//write sdf-- double
			fwrite(&m_signDistance[nborIndex], sizeof(m_signDistance[nborIndex]), 1, fp);
			//write velocity--float
			fwrite(&m_vel[nborIndex].x, sizeof(m_vel[nborIndex].x), 1, fp);
			fwrite(&m_vel[nborIndex].y, sizeof(m_vel[nborIndex].y), 1, fp);
			fwrite(&m_vel[nborIndex].z, sizeof(m_vel[nborIndex].z), 1, fp);
		}
		//��������С�����������ʱ��д��0
		if (j < m_maxNeighbors)
		{
			for (; j < m_maxNeighbors; j++)
			{
				float fPlaceholder = 0.0; //float ��ռλ��
				double dPlaceholder = 0.0; //double ��ռλ��
				//write position
				fwrite(&fPlaceholder, sizeof(float), 3, fp);
				//write density
				fwrite(&dPlaceholder, sizeof(double), 1, fp);
				//write sdf
				fwrite(&dPlaceholder, sizeof(double), 1, fp);
				//write velocity
				fwrite(&fPlaceholder, sizeof(float), 3, fp);
			}
		}//end if(j<m_maxNeighbors)
	}//end for(i)

	fclose(fp);
	fclose(fGT);
}

//�洢���ݵ��ļ�
void pbf::GpuParticleSolver::saveForCompare(std::string &rootPath, std::string &iniName, std::string & tarName, double time, double timeStep)
{
	int nFrame; //��ǰ�ǵڼ�֡
	nFrame = int(time / timeStep);

	std::string prefix("cloud_");
	std::string fullFileName;
	std::string fullGTName;
	std::string path("");
	path = rootPath + std::string("\\");
	
	//��·�����ļ����ϲ����������ļ���
	//fullFileName = path + iniName + "_" + tarName + "\\" + fileName;  //���ÿ֡������Ϣ/
	fullFileName = path + prefix + std::to_string(nFrame) + std::string(".vti");  //���ÿ֡������Ϣ/

	//float3* m_pos
	//double* m_density
	std::vector<float3> tmpPos(m_pos, m_pos + m_numParticles);
	std::vector<float> tmpDensity(m_density, m_density + m_numParticles);

	transformParticleIntoGrid(tmpPos, tmpDensity, fullFileName, 64, 0.02);

	////�������λ�õ��ı��ļ�
	//std::string txtDataFile;
	//sprintf_s(fileName, "%06d.txt", nFrame);
	//txtDataFile = path + iniName + "_" + tarName + "\\" + fileName;
	//std::ofstream writeTxtData(txtDataFile);
	//writeTxtData << m_numParticles << std::endl;
	//for (int k = 0; k < m_numParticles; k++)
	//{
	//	writeTxtData << m_pos[k].x << " " << m_pos[k].y << " " << m_pos[k].z << std::endl;
	//}
	//writeTxtData.close();

	////���ƽ���ܶȵ��ļ�
	//std::string densityName;
	//densityName = path + iniName + "_" + tarName + "\\" + "density.txt";
	//std::ofstream writeDensity(densityName, std::ios::app);
	//double temDen;
	//temDen = computeAvgDensity();
	//writeDensity << temDen << std::endl;
	//writeDensity.close();

	//
	////���ƽ���������ļ�
	//std::string momName;
	//momName = path + iniName + "_" + tarName + "\\" + "momentumAvg.txt";
	//std::ofstream writeMom(momName, std::ios::app);

	//float2 temMom;
	//temMom = computeMomentum();
	//writeMom << temMom.y << std::endl;
	//writeMom.close();

	////����ܶ������ļ�
	//std::string momSumName;
	//momSumName = path + iniName + "_" + tarName + "\\" + "momentumSum.txt";
	//std::ofstream writeSumMom(momSumName, std::ios::app);
	//writeSumMom << temMom.x << std::endl;
	//writeSumMom.close();
}

//�������ӵ�λ�ø�����������
void pbf::GpuParticleSolver::updateMassForParticle()
{
	double tarSumMass = m_standMassForPar * m_tarParSum; //ģ��������
	double tarSurfaceMass = m_standMassForPar * m_tarSurParSum; //ģ�ͱ������ӵ�������
	
	unsigned int numOfParInTarShape = m_constParSum - m_tarSurParSum; //ģ���ڲ������������
	double massForParInTarShape = (tarSumMass - tarSurfaceMass) / numOfParInTarShape; //ģ���ڲ�����Ӧ�߱�������
	_updateMassForParticleAccordingSDF(m_dMass, m_dSignDistance, massForParInTarShape, m_standMassForPar, m_dPose, m_numParticles);

}


double pbf::GpuParticleSolver::computeAvgDensity()
{
	double avgDensity;
	double sumDen = 0.0;
	for (int i = 0; i < m_numParticles; i++)
	{
		sumDen += m_density[i];
	}

	avgDensity = sumDen / m_numParticles;
	return avgDensity;
}


