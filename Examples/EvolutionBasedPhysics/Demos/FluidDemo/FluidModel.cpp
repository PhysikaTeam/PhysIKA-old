#include "FluidModel.h"
#include "PositionBasedDynamics/PositionBasedDynamics.h"
#include "PositionBasedDynamics/SPHKernels.h"

using namespace PBD;

FluidModel::FluidModel() :
	m_particles()
{	
	m_density0 = 1000.0;
	m_particleRadius = 0.025;
	viscosity = 0.02;
	m_neighborhoodSearch = NULL;
}

FluidModel::~FluidModel(void)
{
	cleanupModel();
}

void FluidModel::cleanupModel()
{
	m_particles.release();
	m_lambda.clear();
	m_density.clear();
	m_deltaX.clear();
	m_buoyant.clear();
	m_pressure.clear();
	m_pressureForce.clear();
	delete m_neighborhoodSearch;
}

void FluidModel::reset()
{
	const unsigned int nPoints = m_particles.size();
	
	for(unsigned int i=0; i < nPoints; i++)
	{
		const Vector3r& x0 = m_particles.getPosition0(i);
		m_particles.getPosition(i) = x0;
		m_particles.getLastPosition(i) = m_particles.getPosition(i);
		m_particles.getOldPosition(i) = m_particles.getPosition(i);
		m_particles.getVelocity(i).setZero();
		m_particles.getAcceleration(i).setZero();
		m_deltaX[i].setZero();
		m_lambda[i] = 0.0;
		m_density[i] = 0.0;
		// for clouid
		m_buoyant[i].setZero();
		m_pressureForce[i].setZero();
		m_pressure[i] = 0.0;
		
	}
}

ParticleData & PBD::FluidModel::getParticles()
{
	return m_particles;
}


/** 
*/
void FluidModel::initMasses()
{
	
	const int nParticles = (int) m_particles.size();
	const Real diam = 2.0*m_particleRadius;

	#pragma omp parallel default(shared)
	{
		#pragma omp for schedule(static)  
		for (int i = 0; i < nParticles; i++)
		{
			m_particles.setMass(i, 0.8 * diam*diam*diam * m_density0);		// each particle represents a cube with a side length of r		
																			// mass is slightly reduced to prevent pressure at the beginning of the simulation
		}
	}

	m_standMassForPar = 0.8 * diam*diam*diam * m_density0;
}


/** Resize the arrays containing the particle data.
*/
void FluidModel::resizeFluidParticles(const unsigned int newSize)
{
	m_particles.resize(newSize);
	m_lambda.resize(newSize);
	m_density.resize(newSize);
	m_deltaX.resize(newSize);
	//for cloud
	m_buoyant.resize(newSize);
	m_pressure.resize(newSize);
	m_pressureForce.resize(newSize);
	m_potentialForce.resize(newSize);

	m_signDistance.resize(newSize);
	m_gradientOfSignDistance.resize(newSize);
}


/** Release the arrays containing the particle data.
*/
void FluidModel::releaseFluidParticles()
{
	m_particles.release();
	m_lambda.clear();
	m_density.clear();
	m_deltaX.clear();

	//for cloud
	m_buoyant.clear();
	m_pressure.clear();
	m_pressureForce.clear();
	m_potentialForce.clear();

	m_signDistance.clear();
	m_gradientOfSignDistance.clear();
}

void FluidModel::initModel(const unsigned int nFluidParticles, Vector3r* fluidParticles, const unsigned int nBoundaryParticles, Vector3r* boundaryParticles)
{
	// ��ʼ��ģ�ͣ����ԭ����������
	releaseFluidParticles();
	resizeFluidParticles(nFluidParticles); //��ʼ����������

	// init kernel
	CubicKernel::setRadius(m_supportRadius);

	// copy fluid positions
	#pragma omp parallel default(shared)
	{
		#pragma omp for schedule(static)  
		for (int i = 0; i < (int)nFluidParticles; i++)
		{
			m_particles.getPosition0(i) = fluidParticles[i];
		}
	}

	m_boundaryX.resize(nBoundaryParticles);
	m_boundaryPsi.resize(nBoundaryParticles);

	// copy boundary positions
	#pragma omp parallel default(shared)
	{
		#pragma omp for schedule(static)  
		for (int i = 0; i < (int)nBoundaryParticles; i++)
		{
			m_boundaryX[i] = boundaryParticles[i];
		}
	}

	// initialize masses
	initMasses();

	/** ----------start : zzl-------------2018-5-18
	/* �������ӵ��������Խ��г�ʼ��
	*/

	initTemperature();
	initCloud();
	initVapor();

	//-------------end: zzl --------------2018-5-18


	//////////////////////////////////////////////////////////////////////////
	// Compute value psi for boundary particles (boundary handling)
	// (see Akinci et al. "Versatile rigid - fluid coupling for incompressible SPH", Siggraph 2012
	//////////////////////////////////////////////////////////////////////////

	// Search boundary neighborhood
	NeighborhoodSearchSpatialHashing neighborhoodSearchSH(nBoundaryParticles, m_supportRadius);
	neighborhoodSearchSH.neighborhoodSearch(&m_boundaryX[0]);
	 
	unsigned int **neighbors = neighborhoodSearchSH.getNeighbors();
	unsigned int *numNeighbors = neighborhoodSearchSH.getNumNeighbors();

	#pragma omp parallel default(shared)
	{
		#pragma omp for schedule(static)  
		for (int i = 0; i < (int) nBoundaryParticles; i++)
		{
			Real delta = CubicKernel::W_zero();
			for (unsigned int j = 0; j < numNeighbors[i]; j++)
			{
				const unsigned int neighborIndex = neighbors[i][j];
				delta += CubicKernel::W(m_boundaryX[i] - m_boundaryX[neighborIndex]);
			}
			const Real volume = 1.0 / delta;
			m_boundaryPsi[i] = m_density0 * volume;  //����߽����ӵ�psiֵ
		}
	}


	// Initialize neighborhood search
	if (m_neighborhoodSearch == NULL)
		m_neighborhoodSearch = new NeighborhoodSearchSpatialHashing(m_particles.size(), m_supportRadius);
	m_neighborhoodSearch->setRadius(m_supportRadius);

	reset();
}

//
void PBD::FluidModel::computeSignDistanceAndGradient(Discregrid::CubicLagrangeDiscreteGrid & sdf)
{
	const unsigned int pdNum = m_particles.size();
	#pragma omp parallel default(shared)
	{
		#pragma omp for schedule(static)
		for (int i = 0; i < pdNum; i++)
		{
			Eigen::Vector3d &pos = m_particles.getPosition(i);
			Real distance;
			Eigen::Vector3d gradient;
			distance = sdf.interpolate(0, pos, &gradient);
			m_signDistance[i] = distance;
			m_gradientOfSignDistance[i] = gradient;
			/*if (distance > 5)
			{
				distance = sqrt((0 - pos.x())*(0 - pos.x()) + (0 - pos.y())*(0 - pos.y()) + (0 - pos.z())*(0 - pos.z()));
			}*/
		}
	}
}

/* ���������ܶȣ�ͨ��ɾ����Χ���������
*/
void PBD::FluidModel::decreaseDensity(const unsigned int i)
{
	unsigned int * &neighbors_i = m_neighborhoodSearch->getNeighbors()[i]; //��ȡ��������
	unsigned int &neighborNum_i = m_neighborhoodSearch->getNumNeighbors()[i]; //��ȡ�������ӵ�����

	Eigen::Vector3d position_i = m_particles.getPosition(i);
	Real distanceThreshold = m_particleRadius * 2; //ָ���������ֵ�������ж��Ƿ���Ҫɾ������
	for (int i = 0; i < neighborNum_i; i++)
	{
		unsigned int neighborIndex = neighbors_i[i]; //��ȡ��i���������ӵı��
		Eigen::Vector3d neighborPosition = m_particles.getPosition(neighborIndex); //��ȡ���ӵ�λ��
		
		Real distance;
		distance = (position_i - neighborPosition).norm();
		if (distance < distanceThreshold)
		{
			deleteParticle(neighborIndex);
		}
	}
}

/* ��������ܶȣ�ͨ������Χ�������
*/
double PBD::FluidModel::increaseDensity(const unsigned int i, double constrainDensity, double theta, std::vector<Eigen::Vector3d> & positionOfSelectGrid, ParticleData & addParticlesByGrid)
{
	unsigned int * &neighbors_i = m_neighborhoodSearch->getNeighbors()[i]; //��ȡ��������
	unsigned int &neighborNum_i = m_neighborhoodSearch->getNumNeighbors()[i]; //��ȡ�������ӵ�����

	Eigen::Vector3d position_i = m_particles.getPosition(i);
	environmentGrid temGrid = m_environmentGrid;
	double dim = m_particleRadius * 2;
	double epsilon = 0.0001;
	for (int i = 1; i <= neighborNum_i; i++)
	{
		unsigned int neighborIndex = neighbors_i[i];
		Eigen::Vector3d position_n = m_particles.getPosition(neighborIndex);
		temGrid.setPotentialParticle(position_n, position_i, dim, m_supportRadius, epsilon, neighborIndex);
	}
	temGrid.setPotentialParticle(position_i, position_i, dim, m_supportRadius, epsilon, i); //��ӵ�ǰ���ӱ����Ӱ����

	//ȷ������i��֧���������ɵ�����Χ
	int min_x, max_x, min_y, max_y, min_z, max_z;
	min_x = floor((position_i.x() - m_supportRadius) / temGrid.xStep) + floor(temGrid.xGridNum / 2);
	max_x = floor((position_i.x() + m_supportRadius) / temGrid.xStep) + floor(temGrid.xGridNum / 2);
	min_y = floor((position_i.y() - m_supportRadius) / temGrid.yStep) + floor(temGrid.yGridNum / 2);
	max_y = floor((position_i.y() + m_supportRadius) / temGrid.yStep) + floor(temGrid.yGridNum / 2);
	min_z = floor((position_i.z() - m_supportRadius) / temGrid.zStep) + floor(temGrid.zGridNum / 2);
	max_z = floor((position_i.z() + m_supportRadius) / temGrid.zStep) + floor(temGrid.zGridNum / 2);
	if (min_x < 0)
	{
		min_x = 0;
	}
	if (max_x > temGrid.xGridNum)
	{
		max_x = temGrid.xGridNum;
	}
	if (min_y < 0)
	{
		min_y = 0;
	}
	if (max_y > temGrid.yGridNum)
	{
		max_y = temGrid.yGridNum;
	}
	if (min_z < 0)
	{
		min_z = 0;
	}
	if (max_z > temGrid.zGridNum)
	{
		max_z = temGrid.zGridNum;
	}

	//ȷ��ѡ����ЩǱ��λ�÷�������
	std::vector<Eigen::Vector3d> addParticles; //���ڴ������ӵ�����
	std::vector<Eigen::Vector3d> positionOfGrid; //�洢����������ӵ�����������
	std::vector<Eigen::Vector3i> indexOfGrid;
	std::vector<double> weightOfDensity;// ���Ǳ��λ�ö�Ӧ���ܶ�Ȩ��W
	
	double tempX, tempY, tempZ;
	Eigen::Vector3d tempPosition; //�������������
	Eigen::Vector3i tempIndex; //�������������
	#pragma omp parallel default(shared)
	{
		#pragma omp for schedule(static) 
		for (int i = min_x; i <= max_x; i++)
		{
			for (int j = min_y; j <= max_y; j++)
			{
				for (int k = min_z; k <= max_z; k++)
				{
					//��������λ�ö��ڵ�����ֵ
					tempX = i*temGrid.xStep;
					tempY = j*temGrid.yStep;
					tempZ = k*temGrid.zStep;
					//���㵱ǰ����㵽���ӵ�λ��
					double distance2CurrentParticle = sqrt((tempX - position_i.x())*(tempX - position_i.x()) + (tempY - position_i.y())*(tempY - position_i.y()) + (tempZ - position_i.z())*(tempZ - position_i.z()));
					if (distance2CurrentParticle < m_supportRadius)
					{
						if (temGrid.flag[i][j][k] != 2) //���������Է���,Ϊ2��ʾ���ɷ�������
						{
							tempPosition = Eigen::Vector3d(tempX, tempY, tempZ);
							tempIndex = Eigen::Vector3i(i, j, k);
							//���㵱ǰλ�õĺ˺�����ֵ
							Real weight = CubicKernel::W(tempPosition - position_i);
							positionOfGrid.push_back(tempPosition); //�洢��ǰ����������
							weightOfDensity.push_back(weight); //�洢��ǰ�����ĺ˺���ֵ
							indexOfGrid.push_back(tempIndex);  //�洢��ǰ����������
						}
					}
				}
			}
		}//end for
	}

	//����Ȩ�أ���Ǳ������λ�ý��н�������
	int size = weightOfDensity.size();
	double tempW; //�����ʱȨ��
	Eigen::Vector3d tempV;
	Eigen::Vector3i tempI;
	for (int i = 0; i < size - 1; i++)
	{
		for (int j = 0; j < size -1 - i; j++)
		{
			if (weightOfDensity[j] < weightOfDensity[j + 1])
			{
				//����Ȩ��
				tempW = weightOfDensity[j + 1];
				weightOfDensity[j + 1] = weightOfDensity[j];
				weightOfDensity[j] = tempW;
				//Ϊ��֤�洢λ�õ�˳����洢Ȩ�ص�˳��һ��
				tempV = positionOfGrid[j + 1];
				positionOfGrid[j + 1] = positionOfGrid[j];
				positionOfGrid[j] = tempV;
				//�޸�����˳��
				tempI = indexOfGrid[j + 1];
				indexOfGrid[j + 1] = indexOfGrid[j];
				indexOfGrid[j] = tempI;
			}
		}
	}

	//�ٽ���ǰ���λ������������ӣ��ҳ�����������������λ��
	double tempWeight = 0.0;
	std::vector<Eigen::Vector3i> indexOfSelectGrid;
	for (int i = 0; i < size; i++)
	{
		if (abs(weightOfDensity[i] + tempWeight - constrainDensity) >= theta) //|w[i] + tempW - potentialDensity| >=theta
		{
			tempWeight += weightOfDensity[i];
			positionOfSelectGrid.push_back(positionOfGrid[i]);
			indexOfSelectGrid.push_back(indexOfGrid[i]);
		}
	}

	//���ҵ�������������λ�÷������ӣ����洢��addParticlesByGrid��
	int addParticlesNum = positionOfSelectGrid.size();
	addParticlesByGrid.resize(addParticlesNum); //��ʼ���������Ӷ���
	for (int k = 0; k < addParticlesNum; k++)
	{
		temGrid.putParticle2Grid(m_particles, addParticlesByGrid, i, k, indexOfSelectGrid[k], positionOfSelectGrid[k]);
	}
	//�����������Ӻ������Լ��ֵ�ĲΪ����ʾ����ֵ��
	return tempW - constrainDensity;
}

/* ɾ��ָ��λ�õ�����
*/
void PBD::FluidModel::deleteParticle(const unsigned int i)
{
	m_deltaX.erase(m_deltaX.begin() + i);
	m_buoyant.erase(m_buoyant.begin() + i);
	m_density.erase(m_density.begin() + i);
	m_lambda.erase(m_lambda.begin() + i);
	m_pressure.erase(m_pressure.begin() + i);
	m_pressureForce.erase(m_pressureForce.begin() + i);
	m_potentialForce.erase(m_potentialForce.begin() + i);
	
	//ɾ������
	m_particles.deleteVertex(i);

}

/** ---------start: zzl ----------2018-5-18
/* ��ʼ�������е�����
*/
void FluidModel::initTemperature()
{
	const int nParticles = (int)m_particles.size();

	// compute tempreature 255.15 - 0.2 * y; delta y = 0.05
	#pragma omp parallel default(shared)
	{
		#pragma omp for schedule(static)  
		for (int i = 0; i < nParticles; i++)
		{
			Real temp = 255.15 - 0.2 * m_particles.getPosition0(i).y();
			m_particles.setTemperature(i, temp);
		}
	}
}

// ��ʼ��cloud-water= 0.802 + 0.01 * y;
void PBD::FluidModel::initCloud()
{
	ParticleData &pd = m_particles;
	unsigned int particleNum = pd.size();
	#pragma omp parallel default(shared)
	{
		#pragma omp for schedule(static)
		for(int i = 0; i < particleNum; i++)
		{
			Real &cloudWater = pd.getCloud(i);
			cloudWater = 0.802 + 0.01 * pd.getPosition0(i).y();
		}
	}
}

//��ʼ��vapor����cloud-vapor= 0.065 - 0.01 * y
void PBD::FluidModel::initVapor()
{
	ParticleData &pd = m_particles;
	unsigned int particleNum = pd.size();
	#pragma omp parallel default(shared)
	{
		#pragma omp for schedule(static)
		for(int i = 0; i < particleNum; i++)
		{
			Real &cloudVapor = pd.getVapor(i);
			cloudVapor = 0.065 - 0.01 * pd.getPosition0(i).y();
		}
	}
}
//-------------end: zzl ----------2018-5-18