


#include "Dynamics/Sand/PBDSandSolver.h"
#include <cuda_runtime.h>
#include "Dynamics/Sand/SSEUtil.h"
#include "Core/Utility/cuda_utilities.h"

#include "Core/Utility/CTimer.h"

#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

//#include <thrust/device_vector.h>
//#include <thrust/iterator/zip_iterator.h>
//#include <thrust/remove.h>

namespace PhysIKA
{

	/**
	* @brief Compute kernel weight(2D).
	*/
	__device__ inline double _PBD_weight2D( const Vector3d& p0, const Vector3d& p1,
		SpikyKernel2D<double> kern, double smoothLen)
	{
		Vector3d v = p0 - p1;
		v[1] = 0.0;
		return kern.Weight(v.norm(), smoothLen);
	}

	/**
	* @brief Compute gradient of kernel weight(2D).
	*/
	__device__ inline Vector3d _PBD_weightGrad2D(const Vector3d& p0, const Vector3d& p1,
		SpikyKernel2D<double> kern, double smoothLen)
	{
		Vector3d v = p0 - p1; 
		v[1] = 0.0;
		double r = v.norm();
		if (r > EPSILON)
		{
			v = kern.Gradient(r, smoothLen)*v * (1.0f / r);
		}
		else
		{
			v[0] = 0.0;  v[2] = 0.0;
		}
		return v;
	}


	/**
	* @brief Integrate sand force to particle, and update velocity.
	* @details Forces include: pressure gradient, Del(h); friction . 
	*/
	__global__ void PBD_integrateParticleForce(
		DeviceDArray<Vector3d> velArr,
		DeviceDArray<Vector3d> posArr,
		DeviceDArray<double> massArr,
		DeviceDArray<ParticleType> particleType,
		DeviceDArray<double> parRawRho2d,
		DeviceHeightField1d land,
		//DeviceHeightField1d staticHeight,
		NeighborList<int> neighbors,
		SpikyKernel2D<double> kern,
		double smoothingLength,
		double rho0,
		double gravity, double mu,
		double dt,
		double cfl,
		double hbar = 0.1,
		int* debNbsize = 0,
		double damp = 1.0
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size() || particleType[pId] != ParticleType::SAND) return;

		//if (pId % 100 ==0)
		//	printf(" %d :  neighbor size  \n", pId);

		Vector3d pos_i = posArr[pId];

		// Gradient of rho_2d.
		Vector3d rawgrad;
		Vector3d vec;
		Vector3d gradw;
		Vector3d rhoGrad;
		double weight = _PBD_weight2D(pos_i, pos_i, kern, smoothingLength);
		double smoothRho = weight * parRawRho2d[pId];
		int nbSize = neighbors.getNeighborSize(pId);

		//printf(" %d :  neighbor size %d \n", pId, nbSize);

		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors.getElement(pId, ne);
			
			if (particleType[j] != ParticleType::SAND&&particleType[j] != ParticleType::BOUNDARY)
				continue;
			rawgrad += _PBD_weightGrad2D(pos_i, posArr[j], kern, smoothingLength) *
				massArr[j];
			vec = _PBD_weightGrad2D(pos_i, posArr[j], kern, smoothingLength);
			double wj = _PBD_weight2D(pos_i, posArr[j], kern, smoothingLength);
			gradw += vec;
			rhoGrad += vec * parRawRho2d[j];
			weight += wj;
			smoothRho += wj * parRawRho2d[j];
		}

		double alpha = 0.0;
		vec = rhoGrad / weight - gradw * (smoothRho / (weight *weight));
		vec = vec * alpha + rawgrad * (1.0 - alpha);

		if (debNbsize)
			debNbsize[pId] = nbSize;

		// Del(rho) / rho0
		vec *= -1.0 / rho0;

		// Del(land)
		Vector3d gradh(0.0, 0.0, 0.0);
		land.gradient(pos_i[0], pos_i[2], gradh[0], gradh[2]);
		vec -= gradh;

		double curh = parRawRho2d[pId] / rho0;
		double surfaceh = hbar;
		surfaceh = surfaceh < curh ? surfaceh : curh;

		vec *= gravity;
		vec[1] = 0.0;
		vec = velArr[pId] + vec * dt * surfaceh /curh;


		// Apply friction.
		double vnorm = vec.norm();
		double fric = dt * mu * gravity * surfaceh / curh;// *rho2d / rho0 / massArr[pId];
		if (vnorm < /*mu * gravity * dt*/fric)
		{
			vec *= 0.0;
		}
		else
		{
			vec *= 1.0 - fric / vnorm;
		}

		vec *= damp;

		// debug
		double velThreshold = smoothingLength / dt/* / (nbSize + 1)*/* cfl;
		vnorm = vec.norm();
		if (vnorm > velThreshold)
		{
			vec *= velThreshold / vnorm;
		}

		velArr[pId] = vec;
	}


	/**
	* @brief Integrate particle velocity, update position.
	*/
	__global__ void PBD_integrateVelocity(
		DeviceDArray<Vector3d> posArr,
		DeviceDArray<Vector3d> velArr,
		DeviceDArray<ParticleType> particleType,
		double dt
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size() || particleType[pId] != ParticleType::SAND) return;

		posArr[pId] += velArr[pId] * dt;
	}

	

	bool PBDSandSolver::forwardSubStep(float dt)
	{

		auto& position = m_particlePos;






		//// Integrate force.
		//cuExecute(position.size(), PBD_integrateParticleForce,
		//	m_particleVel,
		//	position,
		//	m_particleMass,
		//	m_particleType,
		//	m_land,
		//	m_neighbors.getValue(),
		//	m_kernel,
		//	m_smoothingLength.getValue(),
		//	m_rho0, 9.8, dt
		//);

		//HostHeightField1d hostheight;
		//hostheight.resize(m_height.Nx(), m_height.Ny());
		//Function1Pt::copy(hostheight, m_height);

		//Function1Pt::copy(hostheight, m_staticHeight);

		//hostheight.Release();
		

		int bdim = 512;
		int gdim = (position.size() + bdim - 1) / bdim;
		PBD_integrateParticleForce << <gdim, bdim >> > (
			m_particleVel,
			position,
			m_particleMass,
			m_particleType,
			m_particleRho2D,
			m_land,
			//m_staticHeight,
			m_neighbors.getValue(),
			m_kernel,
			m_smoothingLength.getValue(),
			m_rho0, 9.8, m_mu, dt, m_CFL,
			m_flowingLayerHeight
			);
		cuSynchronize();

		// Integrate velocity.
		cuExecute(position.size(), PBD_integrateVelocity,
			position,
			m_particleVel,
			m_particleType,
			dt
		);

		// Neighbor query.
		this->_doNeighborDetection();
		this->_updateRawParticleRho();
		if (m_need3DPos)
			this->_updateParticlePos3D();

		return true;
	}


	bool PBDSandSolver::velocityUpdate(float dt)
	{
		if (m_particlePos.size() <= 0)
			return false;
		
		//// debug
		//DeviceArray<int> debNbsize;
		//debNbsize.resize(m_particlePos.size());
		//debNbsize.reset();

		int bdim = 512;
		int gdim = (m_particlePos.size() + bdim - 1) / bdim;
		PBD_integrateParticleForce << <gdim, bdim >> > (
			m_particleVel,
			m_particlePos,
			m_particleMass,
			m_particleType,
			m_particleRho2D,
			m_land,
			//m_staticHeight,
			m_neighbors.getValue(),
			m_kernel,
			m_smoothingLength.getValue(),
			m_rho0, 9.8, m_mu, dt, m_CFL,
			m_flowingLayerHeight
			//, debNbsize.begin()
			);
		cuSynchronize();

		//// debug
		//int nbsize = thrust::reduce(thrust::device, debNbsize.begin(), debNbsize.begin() + debNbsize.size(),
		//	(int)0, thrust::plus<int>());

		//float avgsize = (float)nbsize / debNbsize.size();
		//printf("      Average nb size: %f\n", avgsize);

		return true;
	}

	bool PBDSandSolver::positionUpdate(float dt)
	{
		// Integrate velocity.
		cuExecute(m_particlePos.size(), PBD_integrateVelocity,
			m_particlePos,
			m_particleVel,
			m_particleType,
			dt
		);
		return true;
	}

	bool PBDSandSolver::infoUpdate(float dt)
	{
		// Neighbor query.
		this->_doNeighborDetection();
		this->_updateRawParticleRho();
		if (m_need3DPos)
			this->_updateParticlePos3D();

		return true;
	}

	void PBDSandSolver::freeFlow(int steps)
	{
		if (m_particlePos.size() <= 0)
			return;

		double dt = 0.01;
		for (int i = 0; i < steps; ++i)
		{
			int bdim = 512;
			int gdim = (m_particlePos.size() + bdim - 1) / bdim;
			PBD_integrateParticleForce << <gdim, bdim >> > (
				m_particleVel,
				m_particlePos,
				m_particleMass,
				m_particleType,
				m_particleRho2D,
				m_land,
				//m_staticHeight,
				m_neighbors.getValue(),
				m_kernel,
				m_smoothingLength.getValue(),
				m_rho0, 9.8, 0, dt, m_CFL,
				m_flowingLayerHeight
				, 0,
				0.5
				);
			cuSynchronize();

			// Integrate velocity.
			auto& position = m_particlePos;
			cuExecute(position.size(), PBD_integrateVelocity,
				position,
				m_particleVel,
				m_particleType,
				dt
			);

			// Neighbor query.
			this->_doNeighborDetection();
			this->_updateRawParticleRho();
			//if (m_need3DPos)
			//	this->_updateParticlePos3D();

		}
		//return true;
	}

	
	PBDSandSolver::PBDSandSolver():
		m_smoothingLength(0.05),
		m_neighbors(1),
		m_position(1)
	{
		m_genEliCount.resize(1);

		//var_CFL.setValue(0.05);
	}

	bool PBDSandSolver::initialize()
	{
		int nx = m_SandInfo.nx, ny = m_SandInfo.ny;

		//m_height.resize(nx, ny);
		m_staticHeight.resize(nx, ny);
		m_staticHeight.setSpace(m_SandInfo.griddl, m_SandInfo.griddl);

		m_dStaticHeight.resize(nx, ny);
		m_dStaticHeight.setSpace(m_SandInfo.griddl, m_SandInfo.griddl);


		Vector3d lo = m_land.gridCenterPosition(-1, -1);
		Vector3d lu = m_land.gridCenterPosition(m_land.Nx(), m_land.Ny());
		Vector3f lof(lo[0], lo[1], lo[2]);
		Vector3f luf(lu[0], lu[1], lu[2]);

		m_gridParticleHash.setSpace(m_SandInfo.griddl, lof, luf);

		// Initialize NeighborQuery object.
		m_neighborQuery = std::make_shared<NeighborQuery<DataType3f>>(m_smoothingLength.getValue(),
			lof, luf);
		m_smoothingLength.connect(m_neighborQuery->inRadius());
		m_position.connect(m_neighborQuery->inPosition());
		m_neighborQuery->outNeighborhood()->connect(&(this->m_neighbors));

		this->_doNeighborDetection();
		this->_updateRawParticleRho();
		if (m_need3DPos)
			this->_updateParticlePos3D();

		//forwardSubStep(0.0001);

	
		// Init static height field.
		// Before initialization, grid height data should be prepared.
		this->computeSandStaticHeight();

		// debug
		Function1Pt::copy(m_staticHeight, m_height);

		//this->freeFlow(100);
		if (m_postInitFun)
		{
			m_postInitFun(this);
		}

		return true;
	}

	bool PBDSandSolver::stepSimulation(float dt)
	{

		//m_subStepNum = 2;
		double subdt = dt / (double)m_subStepNum;

		int val = -0.5;

		this->_doNeighborDetection();


		// Step simulation.
		for (int stepi = 0; stepi < m_subStepNum; ++stepi)
		{
			this->forwardSubStep(subdt);
		}


		//this->_generateAndEliminateParticle(dt);

		//this->_updateStaticHeightChange(dt);
		return true;
	}


	/**
	* @brief Update user render particles.
	* @details User point X,Z coordinate is directly simulation particle X,Z position.
	* @details User point Y = h + land.
	*/
	__global__ void PBD_updateUserParticle(
		DeviceArray<Vector3f> usePoints,
		DeviceDArray<Vector3d> posArr,
		DeviceDArray<double> massArr,
		DeviceDArray<double> rho2d,
		DeviceHeightField1d land,
		DeviceHeightField1d staticHeight,
		DeviceDArray<ParticleType> particleType,
		NeighborList<int> neighbors,
		SpikyKernel2D<double> kern,
		double smoothingLength,
		double rho0
	) 
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;

		if (particleType[pId] != ParticleType::SAND)
		{
			usePoints[pId] = Vector3f(0, 0, 0);
			return;
		}

		Vector3d pos_i = posArr[pId];

		// 2D density.
		double rhoi = 0.0;
		double weight = 0.0;
		int nbSize = neighbors.getNeighborSize(pId);
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors.getElement(pId, ne);

			rhoi += _PBD_weight2D(pos_i, posArr[j], kern, smoothingLength)
				* massArr[j];
			//double wj = _PBD_weight2D(pos_i, posArr[j], kern, smoothingLength);
			//rhoi += wj * rho2d[j];
			//weight += wj;
		}
		//rhoi /= weight;

		double hland = land.get(pos_i[0], pos_i[2])/* +staticHeight.get(pos_i[0], pos_i[2])*/;

		Vector3f upi(pos_i[0], rhoi / rho0 + hland, pos_i[2]);
		usePoints[pId] = upi;
	}

	void PBDSandSolver::updateUserParticle(DeviceArray<Vector3f>& userPoints)
	{
		this->_updateRawParticleRho();

		auto& pos = m_particlePos; // m_position.getValue();
		if (pos.size() != userPoints.size())
		{
			userPoints.resize(pos.size());
		}


		//// debug
		//HostHeightField1d hoststaticheight;
		//hoststaticheight.resize(m_staticHeight.Nx(), m_staticHeight.Ny());
		//Function1Pt::copy(hoststaticheight, m_staticHeight);

		//hoststaticheight.Release();

		cuExecute(pos.size(), PBD_updateUserParticle,
			userPoints,
			pos,
			m_particleMass,
			m_particleRho2D,
			m_land, m_staticHeight,
			m_particleType,
			m_neighbors.getValue(),
			m_kernel,
			m_smoothingLength.getValue(),
			m_rho0
		);

	}

	void PBDSandSolver::setParticles(Vector3d* pos, Vector3d* vel, double* mass,
		ParticleType* particleType, int num,
		double rho0, double m0, double smoothLen,
		double mu, double h0)
	{
		m_mu = mu;
		m_rho0 = rho0;
		m_smoothingLength.getValue() = smoothLen;
		m_m0 = m0;
		m_h0 = h0;

		auto& position = m_particlePos;// m_position.getValue();
		//m_position.getValue().resize(posit)

		position.resize(num);
		cudaMemcpy(position.begin(), pos, sizeof(Vector3d)* num, cudaMemcpyHostToDevice);

		m_particleVel.resize(num);
		cudaMemcpy(m_particleVel.begin(), vel, sizeof(Vector3d)*num, cudaMemcpyHostToDevice);

		m_particleMass.resize(num);
		cudaMemcpy(m_particleMass.begin(), mass, sizeof(double)*num, cudaMemcpyHostToDevice);

		m_particleType.resize(num);
		cudaMemcpy(m_particleType.begin(), particleType, sizeof(ParticleType) * num, cudaMemcpyHostToDevice);
		
	}

	void PBDSandSolver::setLand(HostHeightField1d & land)
	{
		m_land.resize(land.Nx(), land.Ny());
		m_land.setSpace(land.getDx(), land.getDz());
		Vector3d origin = land.getOrigin();
		m_land.setOrigin(origin[0], origin[1], origin[2]);
		Function1Pt::copy(m_land, land);
	}

	void PBDSandSolver::setHeight(HostHeightField1d & height)
	{
		m_height.resize(height.Nx(), height.Ny());
		m_height.setSpace(height.getDx(), height.getDz());
		Vector3d origin = height.getOrigin();
		m_height.setOrigin(origin[0], origin[1], origin[2]);

		Function1Pt::copy(m_height, height);
	}


	__global__ void PBD_computeSandHeight(
		DeviceHeightField1d height,
		DeviceHeightField1d staticHeight,
		DeviceDArray<Vector3d> posArr,
		DeviceDArray<double> massArr,
		DeviceDArray<ParticleType> particleType,
		//NeighborList<int> neighbors,
		GridHash<DataType3f> gridhash,
		SpikyKernel2D<double> kern,
		double smoothingLength,
		double rho0
	)
	{
		int pIdx = threadIdx.x + (blockIdx.x * blockDim.x);
		int pIdy = threadIdx.y + (blockIdx.y * blockDim.y);
		if (pIdx >= height.Nx() || pIdy >= height.Ny()) return;

		Vector3d posg = height.gridCenterPosition(pIdx, pIdy);
		Vector3f posgf(posg[0], posg[1], posg[2]);
		int3 id = gridhash.getIndex3(posgf);
		double rho2d = 0.0;

		for (int gi = 0; gi < 9; ++gi)
		{
			int hashid = gridhash.getIndex(id.x + gi / 3 - 1, id.y, id.z + gi % 3 - 1);
			int num = gridhash.getCounter(hashid);
			for (int i = 0; i < num; ++i)
			{
				int nbId = gridhash.getParticleId(hashid, i);
				if (particleType[nbId] != ParticleType::SAND) continue;

				rho2d += _PBD_weight2D(posg, posArr[nbId], kern, smoothingLength)
					* massArr[nbId];
			}
		}

		height(pIdx, pIdy) = staticHeight(pIdx, pIdy) + rho2d / rho0;
	}


	void PBDSandSolver::computeSandHeight()
	{
		uint3 totalsize = { m_SandInfo.nx, m_SandInfo.ny, 1 };
		cuExecute3D(totalsize, PBD_computeSandHeight,
			m_height, m_staticHeight,
			m_particlePos,
			m_particleMass,
			m_particleType,
			m_neighborQuery->getHash(),//m_gridParticleHash,
			m_kernel, m_smoothingLength.getValue(), m_rho0
		);


	}


	__global__ void PBD_computeSandStaticHeight(
		DeviceHeightField1d staticHeight,
		DeviceHeightField1d height,
		DeviceDArray<Vector3d> posArr,
		DeviceDArray<double> massArr,
		DeviceDArray<ParticleType> particleType,
		//NeighborList<int> neighbors,
		GridHash<DataType3f> gridhash,
		SpikyKernel2D<double> kern,
		double smoothingLength,
		double rho0
	)
	{
		int pIdx = threadIdx.x + (blockIdx.x * blockDim.x);
		int pIdy = threadIdx.y + (blockIdx.y * blockDim.y);
		if (pIdx >= height.Nx() || pIdy >= height.Ny()) return;

		Vector3d posg = height.gridCenterPosition(pIdx, pIdy);
		Vector3f posgf(posg[0], posg[1], posg[2]);
		int3 id = gridhash.getIndex3(posgf);
		double rho2d = 0.0;

		for (int gi = 0; gi < 9; ++gi)
		{
			//Vector3d posgi = height.gridCenterPosition(pIdx + gi / 3 - 1, pIdy + gi % 3 - 1);
			//Vector3f posgif(posgi[0], posgi[1], posgi[2]);
			//int hashid = gridhash.getIndex(posgif);
			int hashid = gridhash.getIndex(id.x + gi / 3 - 1, id.y, id.z + gi% 3 - 1);
			int num = gridhash.getCounter(hashid);

			//if (pIdx % 5 == 0 && pIdy % 5 == 0)
			//{
			//	printf(" %d %d, total height: %lf,  num: %d, id: %d  %d  %d \n", 
			//		pIdx, pIdy, rho2d / rho0, num, id.x, id.y, id.z);
			//}

			for (int i = 0; i < num; ++i)
			{
				int nbId = gridhash.getParticleId(hashid, i);
				if (particleType[nbId] != ParticleType::SAND) continue;

				//if (pIdx % 5 == 0 && pIdy % 5 == 0 /*&& (posg - posArr[nbId]).norm()< smoothingLength*/)
				//if(pIdx%5==0)
				//{
				//	printf(" %d %d, id: %d %d %d  Num: %d, nid: %d,  Type: %d \n",
				//		pIdx, pIdy,id.x, id.y, id.z, num, nbId, (int)(particleType[nbId]));
				//}
				//if (particleType[nbId] != ParticleType::SAND) continue;

				rho2d += _PBD_weight2D(posg, posArr[nbId], kern, smoothingLength)
					* massArr[nbId];
			}
		}

		//// debug
		//if (rho2d / rho0 > 0.15)
		//{
		//	printf("%d  %d, Pos: %lf %lf %lf,  rho: %lf \n", pIdx, pIdy, posg[0], posg[1], posg[2],
		//		rho2d);
		//}

		// debug
		/*if (pIdx % 5 == 0 && pIdy % 5 == 0)
		{
			printf(" %d %d, total height: %lf,  dyn height: %lf\n", pIdx, pIdy, height(pIdx, pIdy), rho2d / rho0);
		}*/

		staticHeight(pIdx, pIdy) = height(pIdx, pIdy) - rho2d / rho0;
	}
	void PBDSandSolver::computeSandStaticHeight()
	{

		//HostDArray<ParticleType> ptype;
		//ptype.resize(m_particleType.size());
		//Function1Pt::copy(ptype, m_particleType);

		//ptype.release();

		uint3 totalsize = { m_SandInfo.nx, m_SandInfo.ny, 1 };
		cuExecute3D(totalsize, PBD_computeSandStaticHeight,
			m_staticHeight, m_height,
			m_particlePos,
			m_particleMass,
			m_particleType,
			m_neighborQuery->getHash(),//m_gridParticleHash,
			m_kernel, m_smoothingLength.getValue(), m_rho0
		);
	}

	//__global__ void PBD_updateParticleMass(
	//	DeviceDArray<double> massArr,

	//	DeviceHeightField1d staticHeight,
	//	//DeviceHeightField1d height,
	//	DeviceDArray<Vector3d> posArr,
	//	DeviceDArray<ParticleType> particleType,
	//	//NeighborList<int> neighbors,
	//	GridHash<DataType3f> gridhash,
	//	SpikyKernel2D<double> kern,
	//	double smoothingLength,
	//	double rho0, double gama, double theta0,
	//	double dt
	//)
	//{
	//	int pIdx = threadIdx.x + (blockIdx.x * blockDim.x);
	//	int pIdy = threadIdx.y + (blockIdx.y * blockDim.y);
	//	if (pIdx >= staticHeight.Nx() || pIdy >= staticHeight.Ny()) return;

	//	Vector3d posg = staticHeight.gridCenterPosition(pIdx, pIdy);
	//	
	//	Vector3d grad;
	//	staticHeight.gradient(posg[0], posg[2], grad[0], grad[2]);
	//	double theta = atan(grad.norm());
	//	double dh = - dt * gama * (theta - theta0);
	//	double statich = staticHeight(pIdx, pIdy);
	//	dh = (-statich) < dh ? dh : (-statich);	// hconstraint<0: dh=max(dh, -hconstraint);  hconstraint>0: dh = max(dh, -hconstraint)
	//	double dmass = rho0 * dh * staticHeight.getDx() * staticHeight.getDz();

	//	int hashid = gridhash.getIndex(Vector3f(posg[0], posg[1], posg[2]));
	//	int num = gridhash.getCounter(hashid);
	//	for (int i = 0; i < num; ++i)
	//	{
	//		int phid = gridhash.getParticleId(hashid, i);
	//		if (particleType[phid] != ParticleType::SAND) continue;
	//		massArr[phid] += dmass / num;
	//		if (massArr[phid] < EPSILON)
	//			massArr[phid] = EPSILON;
	//	}
	//}

	//void PBDSandSolver::updateParticleMass(double dt)
	//{
	//	uint3 totalsize = { m_SandInfo.nx, m_SandInfo.ny, 1 };
	//	cuExecute3D(totalsize, PBD_updateParticleMass,
	//		m_particleMass,
	//		m_staticHeight,
	//		m_particlePos,
	//		m_particleType,
	//		m_gridParticleHash,
	//		m_kernel, m_smoothingLength.getValue(),
	//		m_rho0, 1.0, atan(m_mu), dt
	//	);
	//}


	/**
	* @brief Converte 3D position data from Vector3d to Vector3f.
	*/
	__global__ void PBD_posTo3f(
		DeviceArray<Vector3f> pos3f,
		DeviceDArray<Vector3d> pos3d
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (pId >= pos3d.size()) return;

		Vector3d curp = pos3d[pId];
		pos3f[pId] = Vector3f(curp[0], curp[1], curp[2]);
	}



	__global__ void PBD_updateRawParticleRho(
		DeviceDArray<double> parRho2D,
		DeviceDArray<Vector3d> posArr,
		DeviceDArray<double> massArr,
		DeviceDArray<ParticleType> particleType,
		NeighborList<int> neighbors,
		SpikyKernel2D<double> kern,
		double smoothingLength
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size() /*|| particleType[pId] != ParticleType::SAND*/) return;

		Vector3d pos_i = posArr[pId];

		// rho_2d on particle.
		double rho2d = 0.0;
		int nbSize = neighbors.getNeighborSize(pId);

		rho2d = _PBD_weight2D(pos_i, pos_i, kern, smoothingLength) *
			massArr[pId];
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors.getElement(pId, ne);

			if (particleType[j] != ParticleType::SAND && particleType[j] != ParticleType::BOUNDARY)
				continue;
			rho2d += _PBD_weight2D(pos_i, posArr[j], kern, smoothingLength) *
				massArr[j];
		}
		parRho2D[pId] = rho2d;
	}

	void PBDSandSolver::_updateRawParticleRho()
	{
		if (m_particleRho2D.size() != m_particlePos.size())
			m_particleRho2D.resize(m_particlePos.size());

		cuExecute(m_particlePos.size(), PBD_updateRawParticleRho,
			m_particleRho2D,
			m_particlePos,
			m_particleMass,
			m_particleType,
			m_neighbors.getValue(),
			m_kernel,
			m_smoothingLength.getValue()
		);
	}


	__global__ void PBD_updateParticlePos3D(
		DeviceDArray<Vector3d> pos3D,
		DeviceDArray<Vector3d> posArr,
		DeviceDArray<double> parRho2D,
		DeviceHeightField1d land,
		double rho0
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size() /*|| particleType[pId] != ParticleType::SAND*/) return;

		Vector3d pos_i = posArr[pId];
		pos_i[1] = parRho2D[pId] / rho0 + land.get(pos_i[0], pos_i[2]);
		pos3D[pId] = pos_i;

	}
	void PBDSandSolver::_updateParticlePos3D()
	{
		if (m_particlePos3D.size() != m_particlePos.size())
			m_particlePos3D.resize(m_particlePos.size());

		cuExecute(m_particlePos.size(), PBD_updateParticlePos3D,
			m_particlePos3D,
			m_particlePos,
			m_particleRho2D,
			m_land,
			m_rho0
		);


		//HostDArray<Vector3d> hostpos3d;
		//hostpos3d.resize(m_particlePos3D.size());
		//Function1Pt::copy(hostpos3d, m_particlePos3D);


		//hostpos3d.release();
	}

	//__global__ void PBD_updateGridWeightedHeight(
	//	DeviceHeightField1d height,
	//	DeviceHeightField1d weight,
	//	DeviceDArray<Vector3d> posArr,
	//	DeviceDArray<double> massArr,
	//	DeviceDArray<ParticleType> particleType,
	//	NeighborList<int> neighbors,
	//	SpikyKernel2D<double> kern,
	//	double smoothingLength
	//)
	//{
	//	int pId = threadIdx.x + (blockIdx.x * blockDim.x);
	//	if (pId >= posArr.size() || particleType[pId] != ParticleType::SAND) return;

	//	Vector3d pos_i = posArr[pId];

	//	// rho_2d on particle.
	//	double rho2d = 0.0;
	//	int nbSize = neighbors.getNeighborSize(pId);

	//	for (int ne = 0; ne < nbSize; ne++)
	//	{
	//		int j = neighbors.getElement(pId, ne);

	//		if (particleType[j] != ParticleType::SAND && particleType[j] != ParticleType::BOUNDARY)
	//			continue;
	//		rho2d += _PBD_weight2D(pos_i, posArr[j], kern, smoothingLength) *
	//			massArr[j];
	//	}
	//	parRho2D[pId] = rho2d;
	//}

	//void PBDSandSolver::_updateGridHeight()
	//{
	//}

	void PBDSandSolver::_doNeighborDetection()
	{
		if (m_particlePos.size() != m_position.getValue().size())
			m_position.getValue().resize(m_particlePos.size());
		//cuExecute(m_particlePos.size(), PBD_posTo3f,
		//	m_position.getValue(),
		//	m_particlePos);


		int n = m_particlePos.size();
		PBD_posTo3f<<<(n + 511)/512, 512>>>(
			m_position.getValue(),
			m_particlePos);
		cuSynchronize();

		// debug 
		CTimer timer;
		timer.start();

		m_neighborQuery->compute();

		timer.stop();
		printf("NeighborQuery time:  %lf\n", timer.getElapsedTime());
	}

	void PBDSandSolver::_updateGridHash()
	{
		if (m_particlePos.size() != m_position.getValue().size())
			m_position.getValue().resize(m_particlePos.size());
		//cuExecute(m_particlePos.size(), PBD_posTo3f,
		//	m_position.getValue(),
		//	m_particlePos);


		int n = m_particlePos.size();
		PBD_posTo3f << <(n + 511) / 512, 512 >> > (
			m_position.getValue(),
			m_particlePos);
		cuSynchronize();

		m_gridParticleHash.construct(m_position.getValue());
	}


	__global__ void _PBD_UpdateStaticHeightChange(
		DeviceHeightField1d dStaticH,

		DeviceHeightField1d staticHeight,
		DeviceHeightField1d height,
		DeviceHeightField1d land,

		//DeviceDArray<Vector3d> posArr,
		//DeviceDArray<ParticleType> particleType,
		//GridHash<DataType3f> gridhash,
		//SpikyKernel2D<double> kern,
		//double smoothingLength,
		//double rho0,
		double gama, double theta0, double h0,
		double dt
	)
	{
		int pIdx = threadIdx.x + (blockIdx.x * blockDim.x);
		int pIdy = threadIdx.y + (blockIdx.y * blockDim.y);
		if (pIdx >= staticHeight.Nx() || pIdy >= staticHeight.Ny()) return;

		Vector3d posg = staticHeight.gridCenterPosition(pIdx, pIdy);

		Vector3d grad;
		staticHeight.gradient(posg[0], posg[2], grad[0], grad[2]);
		Vector3d landgrad;
		land.gradient(posg[0], posg[2], landgrad[0], landgrad[2]);
		grad += landgrad;
		double theta = atan(grad.norm());
		double dynh = height(pIdx, pIdy) - staticHeight(pIdx, pIdy);
		double dh = -dt * gama * /*dynh**/  (theta - theta0);
		double hconstraint = staticHeight(pIdx, pIdy);

		// static_h + dh >=0;
		dh = (-hconstraint) < dh ? dh : (-hconstraint);	// hconstraint<0: dh=max(dh, -hconstraint);  hconstraint>0: dh = max(dh, -hconstraint)

		// stati_h + dh <= h - h0
		hconstraint = height(pIdx, pIdy) - h0 - hconstraint;
		dh = hconstraint > dh ? dh : hconstraint;

		dStaticH(pIdx, pIdy) += dh;

	}

	void PBDSandSolver::_updateStaticHeightChange(double dt)
	{
		uint3 totalsize = { m_SandInfo.nx, m_SandInfo.ny, 1 };
		cuExecute2D(totalsize, _PBD_UpdateStaticHeightChange,
			m_dStaticHeight,
			m_staticHeight,
			m_height,
			m_land,
			1.0, atan(m_mu), 
			m_h0, dt
		);
	}


	__global__ void _PBD_generateParticle(
		DeviceDArray<Vector3d> particlePos,
		DeviceDArray<Vector3d> particleVel,
		DeviceDArray<double> particleMass,
		DeviceDArray<ParticleType> particleType,
		int* count,

		DeviceHeightField1d dStaticH,
		DeviceHeightField1d staticHeight,
		DeviceHeightField1d height,
		GridHash<DataType3f> gridhash,
		double m0,
		double rho0,
		double dl
	)
	{
		int pIdx = threadIdx.x + (blockIdx.x * blockDim.x);
		int pIdy = threadIdx.y + (blockIdx.y * blockDim.y);
		if (pIdx >= staticHeight.Nx() || pIdy >= staticHeight.Ny()) return;

		double mindh = m0 / (rho0 * dl *dl);
		double dh = dStaticH(pIdx, pIdy);
		if (dh < -mindh)
		{
			// Generate particle.
			int idx = atomicAdd(count, 1) + particlePos.size();
			Vector3d pos = height.gridCenterPosition(pIdx, pIdy);
			//pos[1] = 0.5;
			particlePos[idx] = pos;
			particleVel[idx] = Vector3d(0, 0, 0);
			particleMass[idx] = m0;
			particleType[idx] = ParticleType::SAND;

			dStaticH(pIdx, pIdy) += mindh;

		}
	}

	__global__ void _PBD_eliminateParticle(
		DeviceDArray<Vector3d> particleVel,
		DeviceDArray<ParticleType> particleType,

		DeviceHeightField1d dStaticH,
		DeviceHeightField1d staticHeight,
		DeviceHeightField1d height,
		//GridHash<DataType3f> gridhash,
		HeightFieldGrid<int, double> gridParticle,
		HeightFieldGrid<int, double> gridParticleValid,
		double m0,
		double rho0,
		double dl
	)
	{
		int pIdx = threadIdx.x + (blockIdx.x * blockDim.x);
		int pIdy = threadIdx.y + (blockIdx.y * blockDim.y);
		if (pIdx >= staticHeight.Nx() || 2*pIdy >= staticHeight.Ny()) return;

		double mindh = m0 / (rho0 * dl *dl);
		double dh = dStaticH(pIdx, pIdy);
		if (dh > mindh && gridParticleValid(pIdx, pIdy) > 0)
		{
			int eliId = gridParticle(pIdx, pIdy);

			particleVel[eliId] = Vector3d();
			particleType[eliId] = ParticleType::SANDINACTIVE;

			dStaticH(pIdx, pIdy) -= mindh;



			//Vector3d posg = staticHeight.gridCenterPosition(pIdx, pIdy);
			//int hashid = gridhash.getIndex(Vector3f(posg[0], posg[1], posg[2]));

			//if (gridhash.getCounter(hashid) <= 0)
			//	printf(" %d %d, Hashid: %d, count: %d,    Pos: %lf %lf %lf\n ", pIdx, pIdy,
			//		hashid, gridhash.getCounter(hashid), posg[0], posg[1], posg[2]);
			//if (gridhash.getCounter(hashid) > 0)
			//{
			//	//printf("%lf %lf\n", dh, mindh);

			//	int eliId = gridhash.getParticleId(hashid,/* gridhash.getCounter(hashid) - 1*/0);
			//	int idx = atomicAdd(count, 1);
			//	idx = particlePos.size() - 1 - idx;

			//	particlePos[eliId] = particlePos[idx];
			//	particleVel[eliId] = particleVel[idx];
			//	particleMass[eliId] = particleMass[idx];
			//	particleType[eliId] = particleType[idx];

			//	dStaticH(pIdx, pIdy) -= mindh;

			//}
		}
	}


	__global__ void _PBD_findOneParticleOnGrid(
		HeightFieldGrid<int, double> gridParticle,
		HeightFieldGrid<int, double> gridParticleValid,
		DeviceDArray<ParticleType> particleType,
		DeviceDArray<Vector3d> particlePos
	) {
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= particlePos.size()) return;
		if (particleType[pId] != ParticleType::SAND) return;

		// Particle
		Vector3d posi = particlePos[pId];
		int2 gid = gridParticle.gridRawIndex(posi);
		if (!gridParticle.inRange(gid.x, gid.y)) return;

		int writeCount = atomicAdd(&(gridParticleValid(gid.x, gid.y)), 1);
		if (writeCount == 0)
		{
			gridParticle(gid.x, gid.y) = pId;
		}
	}

	void PBDSandSolver::_generateAndEliminateParticle(double dt)
	{
		//const float h_a[] = { 1.2f, 2.0f, 0.5f, 1.6f, 3.1f, 0.4f };
		//const int   h_b[] = { 3, 5, 4, 1, 2, 3 };

		//// move to device
		//thrust::device_vector<float> d_a(h_a, h_a + 6);
		//thrust::device_vector<int>   d_b(h_b, h_b + 6);

		//// define a tuple of the two vector's iterators
		//typedef thrust::tuple<thrust::device_vector<float>::iterator, thrust::device_vector<int>::iterator> IteratorTuple;

		//// define a zip iterator
		//typedef thrust::zip_iterator<IteratorTuple> ZipIterator;

		//ZipIterator zip_begin = thrust::make_zip_iterator(thrust::make_tuple(d_a.begin(), d_b.begin()));
		//ZipIterator zip_end = zip_begin + 6;

		// https://groups.google.com/g/thrust-users/c/6AE37kWCYSQ


		// Update sand height.
		this->computeSandHeight();


		// Update static layer height change.
		this->_updateStaticHeightChange(dt);



		// Enlarge array capability.
		if ((m_particlePos.size() + m_SandInfo.nx* m_SandInfo.ny) > m_particlePos.capability())
		{
			int capability = (int)((m_particlePos.size() + m_SandInfo.nx* m_SandInfo.ny) * 1.2);
			this->_particleNumReserve(capability);
		}
		cudaMemset(m_genEliCount.begin(), 0, sizeof(int));

		//// Generate particles.
		uint3 totalsize = { m_SandInfo.nx, m_SandInfo.ny, 1 };
		cuExecute2D(totalsize, _PBD_generateParticle,
			m_particlePos,
			m_particleVel,
			m_particleMass,
			m_particleType,
			m_genEliCount.begin(),

			m_dStaticHeight,
			m_staticHeight,
			m_height, 
			m_gridParticleHash,
			m_m0, m_rho0, m_SandInfo.griddl
		);


		



		//// Resize particle arrays.
		int numchange = 0;
		cudaMemcpy(&numchange, m_genEliCount.begin(), sizeof(int), cudaMemcpyDeviceToHost);
		this->_particleNumResize(m_particlePos.size() + numchange);

		//this->_updateGridHash();

		// Find grid particle.
		Vector3d origin = m_height.getOrigin();
		HeightFieldGrid<int, double> gridParticle;
		gridParticle.resize(m_SandInfo.nx, m_SandInfo.ny);
		gridParticle.setSpace(m_height.getDx(), m_height.getDz());
		gridParticle.setOrigin(origin[0], origin[1], origin[2]);
		HeightFieldGrid<int, double> gridParticleValid;
		gridParticleValid.resize(m_SandInfo.nx, m_SandInfo.ny);
		gridParticleValid.setSpace(m_height.getDx(), m_height.getDz());
		gridParticleValid.setOrigin(origin[0], origin[1], origin[2]);
		cuExecute(m_particlePos.size(), _PBD_findOneParticleOnGrid,
			gridParticle, gridParticleValid,
			m_particleType, m_particlePos
		);

		// Particle elimination.
		//cudaMemset(m_genEliCount.begin(), 0, sizeof(int));
		cuExecute2D(totalsize, _PBD_eliminateParticle,
			m_particleVel,
			m_particleType,

			m_dStaticHeight,
			m_staticHeight,
			m_height,
			//m_gridParticleHash,
			gridParticle, gridParticleValid,
			m_m0, m_rho0, m_SandInfo.griddl
		);

		// Resize particle arrays.
		numchange = 0;
		cudaMemcpy(&numchange, m_genEliCount.begin(), sizeof(int), cudaMemcpyDeviceToHost);
		this->_particleNumResize(m_particlePos.size() - numchange);


		// Compute particle static height.
		if (m_particlePos.size() > 0)
		{
			this->_doNeighborDetection();
		}
		this->computeSandStaticHeight();
	}

	void PBDSandSolver::_particleNumResize(int n)
	{
		m_particlePos.resize(n);
		m_prePosition.resize(n);
		m_dPosition.resize(n);
		m_particleVel.resize(n);
		m_particleMass.resize(n);
		m_particleRho2D.resize(n);
		m_particleType.resize(n);
		m_lambda.resize(n);
	}

	void PBDSandSolver::_particleNumReserve(int n)
	{
		m_particlePos.reserve(n);
		m_prePosition.reserve(n);
		m_dPosition.reserve(n);
		m_particleVel.reserve(n);
		m_particleMass.reserve(n);
		m_particleRho2D.reserve(n);
		m_particleType.reserve(n);
		m_lambda.reserve(n);
	}


	
	//void PBDSandSolver::_initBoundaryParticle()
	//{

	//}
}