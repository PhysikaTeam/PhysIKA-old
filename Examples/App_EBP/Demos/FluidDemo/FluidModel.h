#ifndef __FluidModel_h__
#define __FluidModel_h__

#include "Demos/Simulation/ParticleData.h"
#include <vector>
#include "Demos/Simulation/NeighborhoodSearchSpatialHashing.h"
#include "Discregrid/all"



namespace PBD 
{	
	struct environmentGrid
	{
		// ��Χ�д�С
		double max_x, min_x;
		double max_y, min_y;
		double max_z, min_z;
		// ����
		double xStep;
		double yStep;
		double zStep;
		// ÿ�������ϵ�������
		int xGridNum;
		int yGridNum;
		int zGridNum;
		
		std::vector<std::vector<std::vector<int>>> nearestParticleIndex; //��¼�����������������������,-1��ʾû�����������
		std::vector<std::vector<std::vector<double>>> nearestDistance; //��¼��ǰ�������������֮��ľ���
		std::vector <std::vector<std::vector<int>>> flag;  //��¼������״̬��0: air 1:potentialParticle, 2:notPotentialParticle

		void initGrid(int xHalfGridNum, int yHalfGridNum, int zHalfGridNum, double xstep, double ystep, double zstep)
		{
			xStep = xstep;
			yStep = ystep;
			zStep = zstep;
			min_x = -1 * xHalfGridNum*xstep;
			max_x = xHalfGridNum*xstep;
			min_y = -1 * yHalfGridNum*ystep;
			max_y = yHalfGridNum*ystep;
			min_z = -1 * zHalfGridNum*zstep;
			max_z = zHalfGridNum*zstep;
			xGridNum = 2 * xHalfGridNum + 1;
			yGridNum = 2 * yHalfGridNum + 1;
			zGridNum = 2 * zHalfGridNum + 1;

			flag.resize(xGridNum);
			nearestParticleIndex.resize(xGridNum);
			nearestDistance.resize(xGridNum);
			for (int i = 0; i < xGridNum; i++)
			{
				flag[i].resize(yGridNum);
				nearestParticleIndex[i].resize(yGridNum);
				nearestDistance[i].resize(yGridNum);
			}
			for (int i = 0; i<xGridNum; i++)
				for (int j = 0; j < yGridNum; j++)
				{
					flag[i][j].resize(zGridNum);
					nearestParticleIndex[i][j].resize(zGridNum);
					nearestDistance[i][j].resize(zGridNum);
				}
			for (int i = 0; i < xGridNum; i++)
			{
				for (int j = 0; j < yGridNum; j++)
				{
					for (int k = 0; k < zGridNum; k++)
					{
						flag[i][j][k] = 0;
						nearestParticleIndex[i][j][k] = -1;
						nearestDistance[i][j][k] = 9999.0;
					}
				}
			}
		}


		void setPotentialParticle(Eigen::Vector3d neighborParticlePosition, Eigen::Vector3d currentParticlePosition, double dim, double supportRadius, double epsilon, int particleIndex)
		{
			int halfx = floor(xGridNum / 2);
			int halfy = floor(yGridNum / 2);
			int halfz = floor(zGridNum / 2);

			int low_x, up_x, low_y, up_y, low_z, up_z;
			double x, y, z;
			//��ȡ�������ӵ�x��y��z����
			x = neighborParticlePosition.x();
			y = neighborParticlePosition.y();
			z = neighborParticlePosition.z();
			if (x >= min_x && x <= max_x && y >= min_y && y <= max_y && z >= min_z && z <= max_z)
			{
				//��ȡ�������ڵ�����λ��
				low_x = floor(x / xStep);
				up_x = ceil(x / xStep);
				low_y = floor(y / yStep);
				up_y = ceil(y / yStep);
				low_z = floor(z / zStep);
				up_z = ceil(z / zStep);

				double tempX, tempY, tempZ;
				for (int i = low_x; i <= up_x; i++)
				{
					for (int j = low_y; j <= up_y; j++)
					{
						for (int k = low_z; k <= up_z; k++)
						{
							//��������λ�ö��ڵ�����ֵ
							tempX = i*xStep;
							tempY = j*yStep;
							tempZ = k*zStep;
							//���㵽��������֮��ľ���
							double distance1 = sqrt((tempX - x)*(tempX - x) + (tempY - y)*(tempY - y) + (tempZ - z)*(tempZ - z));
							//���㵽��ǰ����֮��ľ��룬Ŀ����Ϊ���ж��Ƿ���֧������
							double distance2 = sqrt((tempX - currentParticlePosition.x())*(tempX - currentParticlePosition.x()) + (tempY - currentParticlePosition.y())*(tempY - currentParticlePosition.y()) + (tempZ - currentParticlePosition.z())*(tempZ - currentParticlePosition.z()));
							//��ǰ�������ھ����������̫�������ܷ������ӣ�����flagΪ2
							if (distance1 <= dim + epsilon)
							{
								flag[i + halfx][j + halfy][k + halfz] = 2;
							}
							//��ǰ����Զ��������ӣ���λ��֧�����ڣ����Է������ӣ�����flagΪ1
							if (distance1 > dim + epsilon && distance2 < supportRadius)
							{
								flag[i + halfx][j + halfy][k + halfz] = 1;
								if (nearestParticleIndex[i + halfx][j + halfy][k + halfz] == -1)
								{
									nearestParticleIndex[i + halfx][j + halfy][k + halfz] = particleIndex;
									nearestDistance[i + halfx][j + halfy][k + halfz] = distance1;
								}
								else
								{
									if (nearestDistance[i + halfx][j + halfy][k + halfz] > distance1)
									{
										nearestParticleIndex[i + halfx][j + halfy][k + halfz] = particleIndex;
										nearestDistance[i + halfx][j + halfy][k + halfz] = distance1;
									}
								}
								
							}

						}
					}
				}
			}
		}

		//�ڸ���������㴦������ӣ���ʼ�����ӵĸ�������
		void putParticle2Grid(ParticleData & oriParticleData, ParticleData & addParticleByGrid, const unsigned int oriIndex, const unsigned int addIndex,Eigen::Vector3i & indexOfGrid, Eigen::Vector3d & positionOfGrid)
		{
			int nearestIndex;
			nearestIndex = nearestParticleIndex[indexOfGrid.x()][indexOfGrid.y()][indexOfGrid.z()];
			if (nearestIndex >= 0)
			{
				//add location
				addParticleByGrid.setPosition(addIndex, positionOfGrid);
				addParticleByGrid.setOldPosition(addIndex, positionOfGrid);
				addParticleByGrid.setLastPosition(addIndex, positionOfGrid);
				//add mass
				addParticleByGrid.setMass(addIndex, oriParticleData.getMass(nearestIndex));
				//add m_a
				addParticleByGrid.setAcceleration(addIndex, oriParticleData.getAcceleration(nearestIndex));
				//add vapor
				addParticleByGrid.setVapor(addIndex, oriParticleData.getVapor(nearestIndex));
				//add cloud
				addParticleByGrid.setCloud(addIndex, oriParticleData.getCloud(nearestIndex));
				//add temperature
				addParticleByGrid.setTemperature(addIndex, oriParticleData.getTemperature(nearestIndex));
				// add velocity
				addParticleByGrid.setVelocity(addIndex, oriParticleData.getVelocity(nearestIndex));
			}
			else
			{
				//add location
				addParticleByGrid.setPosition(addIndex, positionOfGrid);
				addParticleByGrid.setOldPosition(addIndex, positionOfGrid);
				addParticleByGrid.setLastPosition(addIndex, positionOfGrid);
				//add mass
				addParticleByGrid.setMass(addIndex, oriParticleData.getMass(oriIndex));
				//add m_a
				addParticleByGrid.setAcceleration(addIndex, oriParticleData.getAcceleration(oriIndex));
				//add vapor
				addParticleByGrid.setVapor(addIndex, oriParticleData.getVapor(oriIndex));
				//add cloud
				addParticleByGrid.setCloud(addIndex, oriParticleData.getCloud(oriIndex));
				//add temperature
				addParticleByGrid.setTemperature(addIndex, oriParticleData.getTemperature(oriIndex));
				// add velocity
				addParticleByGrid.setVelocity(addIndex, oriParticleData.getVelocity(oriIndex));
			}
			
			
		}

	};

	class FluidModel 
	{
		public:
			FluidModel();
			virtual ~FluidModel();

		protected:	
			Real viscosity;
			Real m_density0;
			Real m_particleRadius;
			Real m_supportRadius;
			ParticleData m_particles;
			std::vector<Vector3r> m_boundaryX; //�洢��ʼ���߽����ӵ�λ��
			std::vector<Real> m_boundaryPsi;
			std::vector<Real> m_density;
			std::vector<Real> m_lambda;		
			std::vector<Vector3r> m_deltaX; //Ϊ�����ܶ�Լ�����õ������ӵľ���λ�� $\delta x$

			// for cloud's properties
			std::vector<Vector3r> m_buoyant;
			std::vector<Vector3r> m_pressureForce;
			std::vector<Real> m_pressure;
			std::vector<Eigen::Vector3d> m_potentialForce; //������

			//for shape control
			std::vector<Real> m_signDistance; //��������λ�õķ��ž���
			std::vector<Eigen::Vector3d> m_gradientOfSignDistance; //���ž����һ�����ݶ�


			// ��������
			environmentGrid m_environmentGrid;


			//Ϊʵ�����������̶�����£�ʹ�����������ģ�ͣ����ã��ڲ���������>��������������������
			//---------for learning----zzl-------2019-1-14--------start-------
			//ԭģ���������ͱ���������
			unsigned int m_iniParSum;
			unsigned int m_iniSurParSum;
			//Ŀ��ģ���������ͱ���������
			unsigned int m_tarParSum;
			unsigned int m_tarSurParSum;

			//����ģ��������������������������ģ�ʹ�С��ֻ�ܰ����������������
			unsigned int m_constParSum;
			Real m_standMassForPar;//�ڼ���ģ������ʱ�õı�׼�������������ӵ��������ڸ�ֵ�� ��initMass()�����н��г�ʼ��
			//---------for learning----zzl-------2019-1-14--------end-------

			/* ���±����Ķ���������ʵ��ͨ���������ӷ������״�ݻ�
			/--------------------------------zzl-----------2019-1-16-----------start-----------------
			-----------------------------------------------------------------------------------------*/
			std::vector<Vector3r> targetParticles; //���ڴ��Ŀ��������ӵ�λ��
			/*--------------------------------zzl-----------2019-1-16-----------end-----------------*/

			/*----------------�������ӷ�ʵ����״�ݻ�----------------*/
			/*-------��ӿ�����������-------zzl-2019-1-23--------------------------------*/
			std::vector<unsigned int> m_controlParticleIndex;
			std::vector<unsigned int> m_targetParticleIndex;
			/*----------------------------zzl-2019-1-23--------------------------------*/

			NeighborhoodSearchSpatialHashing *m_neighborhoodSearch;		


			void initMasses();

			void resizeFluidParticles(const unsigned int newSize);
			void releaseFluidParticles();
			//for cloud
			void initTemperature();
			void initCloud();
			void initVapor();

		public:
			void cleanupModel();
			virtual void reset();

			ParticleData &getParticles();

			void initModel(const unsigned int nFluidParticles, Vector3r* fluidParticles, const unsigned int nBoundaryParticles, Vector3r* boundaryParticles);

			const unsigned int numBoundaryParticles() const { return (unsigned int)m_boundaryX.size(); }
			Real getDensity0() const { return m_density0; }
			Real getSupportRadius() const { return m_supportRadius; }
			Real getParticleRadius() const { return m_particleRadius; }
			void setParticleRadius(Real val) { m_particleRadius = val; m_supportRadius = 4.0*m_particleRadius; }
			NeighborhoodSearchSpatialHashing* getNeighborhoodSearch() { return m_neighborhoodSearch; }

			Real getViscosity() const { return viscosity; }
			void setViscosity(Real val) { viscosity = val; }

			//���ó�ʼ��״�����������ͱ�����������
			void setIniShpaePar(unsigned int parSum, unsigned parSurCnt) 
			{ 
				m_iniParSum = parSum; 
				m_iniSurParSum = parSurCnt; 
			}
			//����Ŀ����״�����������ͱ�����������
			void setTarShpaePar(unsigned int parSum, unsigned parSurCnt)
			{
				m_tarParSum = parSum;
				m_tarSurParSum = parSurCnt;
			}
			//������״������������
			void setConstParCnt(unsigned int givenParCnt) { m_constParSum = givenParCnt; }
			//���س�ʼ��״������������������������Ŀ����״�����������ͱ���������
			unsigned int getIniParSum() { return m_iniParSum; }
			unsigned int getIniSurParSum() { return m_iniSurParSum; }
			unsigned int getTarParSum() { return m_tarParSum; }
			unsigned int getTarSurParSum() { return m_tarSurParSum; }
			unsigned int getConstParSum() { return m_constParSum; }
			Real getStandMassForPar() { return m_standMassForPar; }
			//�����������λ�õķ��ž��뼰���ݶ�
			void computeSignDistanceAndGradient(Discregrid::CubicLagrangeDiscreteGrid &sdf);

			void resizeSignDistanceAndGradient(unsigned int size)
			{
				m_signDistance.resize(size);
				m_gradientOfSignDistance.resize(size);
			}

			FORCE_INLINE const Vector3r& getBoundaryX(const unsigned int i) const
			{
				return m_boundaryX[i];
			}

			FORCE_INLINE Vector3r& getBoundaryX(const unsigned int i)
			{
				return m_boundaryX[i];
			}

			FORCE_INLINE void setBoundaryX(const unsigned int i, const Vector3r &val)
			{
				m_boundaryX[i] = val;
			}

			FORCE_INLINE const Real& getBoundaryPsi(const unsigned int i) const
			{
				return m_boundaryPsi[i];
			}

			FORCE_INLINE Real& getBoundaryPsi(const unsigned int i)
			{
				return m_boundaryPsi[i];
			}

			FORCE_INLINE void setBoundaryPsi(const unsigned int i, const Real &val)
			{
				m_boundaryPsi[i] = val;
			}

			FORCE_INLINE const Real& getLambda(const unsigned int i) const
			{
				return m_lambda[i];
			}

			FORCE_INLINE Real& getLambda(const unsigned int i)
			{
				return m_lambda[i];
			}

			FORCE_INLINE void setLambda(const unsigned int i, const Real &val)
			{
				m_lambda[i] = val;
			}

			FORCE_INLINE const Real& getDensity(const unsigned int i) const
			{
				return m_density[i];
			}

			FORCE_INLINE Real& getDensity(const unsigned int i)
			{
				return m_density[i];
			}

			FORCE_INLINE void setDensity(const unsigned int i, const Real &val)
			{
				m_density[i] = val;
			}

			FORCE_INLINE Vector3r &getDeltaX(const unsigned int i)
			{
				return m_deltaX[i];
			}

			FORCE_INLINE const Vector3r &getDeltaX(const unsigned int i) const
			{
				return m_deltaX[i];
			}

			FORCE_INLINE void setDeltaX(const unsigned int i, const Vector3r &val)
			{
				m_deltaX[i] = val;
			}

			// for cloud's properties
			FORCE_INLINE Vector3r & getBuoyant(const unsigned int i)
			{
				return m_buoyant[i];
			}

			FORCE_INLINE const Vector3r & getBuoyant(const unsigned int i) const
			{
				return m_buoyant[i];
			}

			FORCE_INLINE void setBuoyant(const unsigned int i, const Vector3r & val)
			{
				m_buoyant[i] = val;
			}

			FORCE_INLINE Real & getPressure(const unsigned int i)
			{
				return m_pressure[i];
			}

			FORCE_INLINE const Real & getPressure(const unsigned int i) const
			{
				return m_pressure[i];
			}

			FORCE_INLINE void setPressure(const unsigned int i, const Real & val)
			{
				m_pressure[i] = val;
			}

			FORCE_INLINE Vector3r & getPressureForce(const unsigned int i)
			{
				return m_pressureForce[i];
			}

			FORCE_INLINE const Vector3r & getPressureForce(const unsigned int i) const
			{
				return m_pressureForce[i];
			}

			FORCE_INLINE void setPressureForece(const unsigned int i, const Vector3r & val)
			{
				m_pressureForce[i] = val;
			}

			FORCE_INLINE  Eigen::Vector3d & getPotentialForce(const unsigned int i) 
			{
				return m_potentialForce[i];
			}

			FORCE_INLINE void setPotentialForce(const unsigned int i, const Vector3r &val)
			{
				m_potentialForce[i] = val;
			}


			FORCE_INLINE environmentGrid & getEnvironmentGrid()
			{
				return m_environmentGrid;
			}

			FORCE_INLINE void setSignDistance(const unsigned int i, Real &val)
			{
				m_signDistance[i] = val;
			}

			FORCE_INLINE Real & getSignDistance(const unsigned int i)
			{
				return m_signDistance[i];
			}

			FORCE_INLINE void setGradientOfSignDistance(const unsigned int i, Eigen::Vector3d &val)
			{
				m_gradientOfSignDistance[i] = val;
			}

			FORCE_INLINE Eigen::Vector3d & getGradientOfSignDistance(const unsigned int i)
			{
				return m_gradientOfSignDistance[i];
			}

			/* ���±����Ķ���������ʵ��ͨ���������ӷ������״�ݻ�
			/--------------------------------zzl-----------2019-1-16-----------start-----------------
			-----------------------------------------------------------------------------------------*/

			FORCE_INLINE Vector3r & getTargetParticles(unsigned int i)
			{
				return targetParticles[i];
			}

			FORCE_INLINE void setTargetParticles(unsigned int i, Vector3r val)
			{
				targetParticles[i] = val;
			}
			FORCE_INLINE void resizeTargetParticles(unsigned int size)
			{
				targetParticles.resize(size);
			}

			FORCE_INLINE unsigned int getSizeOfTargetParticles()
			{
				return targetParticles.size();
			}
		    /*--------------------------------zzl-----------2019-1-16-----------end-----------------*/

			//�Կ���������������m_controlParticleIndex�Ĳ���
			FORCE_INLINE void setControlParticleIndex(unsigned int i, unsigned int value)
			{
				m_controlParticleIndex[i] = value;
			}
			FORCE_INLINE unsigned int &getConrolParticleIndex(unsigned int i)
			{
				return m_controlParticleIndex[i];
			}
			FORCE_INLINE void setSizeOfControlParticleIndex(unsigned int size)
			{
				m_controlParticleIndex.resize(size);
			}
			FORCE_INLINE unsigned int getSizeOfControlParticleIndex()
			{
				return m_controlParticleIndex.size();
			}
			
			//��Ŀ���������������Ĳ�����m_targetParticleIndex
			FORCE_INLINE void setTargetParticleIndex(unsigned int i, unsigned int value)
			{
				m_targetParticleIndex[i] = value;
			}
			FORCE_INLINE unsigned int &getTargetParticleIndex(unsigned int i)
			{
				return m_targetParticleIndex[i];
			}
			FORCE_INLINE void setSizeOfTargetParticleIndex(unsigned int size)
			{
				m_targetParticleIndex.resize(size);
			}
			FORCE_INLINE unsigned int getSizeOfTargetParticleIndex()
			{
				return m_targetParticleIndex.size();
			}
			
			//-----zzl-------2018-7-18----start---
			// ���ָ������λ�õ��ܶȽ��͹��ܣ�ͨ��ɾ����Χ����ʵ�֣�
			void decreaseDensity(const unsigned int i);
			// ���ָ������λ�õ��ܶ����ӹ��ܣ�ͨ���������ʵ�֣�,����constrainDensity��ʾ��ӵ������ܵ��ܶȺ�Ҫ�������ֵ,����theta��ʾ��ӵ��������ܶ���Լ���ܶȵ��������ֵ,positionOfSelectGrid���ڴ��ѡ���������λ�����꣬ ��ΧֵΪ������ӵ��ܶȺ���constrainDensity�Ĳ�ֵ��
			double increaseDensity(const unsigned int i,double constrainDensity,double theta, std::vector<Eigen::Vector3d> & positionOfSelectGrid, ParticleData & addParticlesByGrid);

			//���ɾ��ָ��λ�õ�����
			void deleteParticle(const unsigned int i);

			//����������
			//-----zzl-------2018-7-18----end---
	}; 

}

#endif