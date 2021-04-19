#include "..\..\PositionBasedDynamics\TimeIntegration.h"
#include "TimeStepFluidModel.h"
#include "Demos/Simulation/TimeManager.h"
#include "PositionBasedDynamics/PositionBasedFluids.h"
#include "PositionBasedDynamics/TimeIntegration.h"
#include "PositionBasedDynamics/SPHKernels.h"
#include "Demos/Utils/Timing.h"


#include "Common/Common.h"
#include "Demos/Visualization/MiniGL.h"
#include "Demos/Visualization/Selection.h"
#include "GL/glut.h"
#include <Eigen/Dense>
#include "Demos/Simulation/SimulationModel.h"
#include "Demos/Simulation/TimeStepController.h"
#include <iostream>
#include "Demos/Utils/OBJLoader.h"
#include "Demos/Visualization/Visualization.h"
#include "Demos/Simulation/DistanceFieldCollisionDetection.h"
#include "Demos/Utils/SceneLoader.h"
#include "Demos/Utils/TetGenLoader.h"
#include "Demos/Simulation/CubicSDFCollisionDetection.h"
#include "Demos/Utils/Logger.h"
#include "Demos/Utils/FileSystem.h"



using namespace PBD;
using namespace std;

TimeStepFluidModel::TimeStepFluidModel()
{
}

TimeStepFluidModel::~TimeStepFluidModel(void)
{
}

void TimeStepFluidModel::step(string& rootPath, FluidModel &model, SDFGradientField &sdfGraField, Discregrid::CubicLagrangeDiscreteGrid &m_sdf, std::string &iniFileName, std::string &tarFileName) //zzl--�����˵ڶ�������sdfGraField��Ϊ�˼����Ƴ���
{
	START_TIMING("simulation step");
	TimeManager *tm = TimeManager::getCurrent (); 
	Real time = tm->getTime(); //��õ�ǰ�����˶೤ʱ��
	//std::cout << "===" << time << std::endl;
	const Real h = tm->getTimeStepSize();  //��õ�ǰ��ʱ����Ʋ�����Ĭ����0.005
	//std::cout << h << std::endl;
	//���������
	//std::cout << "particle num==" << model.getParticles().size() << std::endl;
	ParticleData &pd = model.getParticles();

	/*for (int i = 0; i < model.getParticles().size(); i++)
	{
		double sdf = model.getSignDistance(i);
		std::cout << "sdf: " << sdf << std::endl;
	}
*/

	clearAccelerations(model);

	//��������

	//computePotentialForce(model, sdfGraField);

	
	/** ----------start��zzl-------2018-9-10--------
	/*����GPUʵ��PBF���
	*/
	unsigned int num_particles = pd.size();
	double particleRadius = model.getParticleRadius();
	double cellGridSize = particleRadius * 2;
	double kernelRadius = model.getSupportRadius();
	Vector3r *particlePos = &pd.getPosition(0);
	Vector3r *particleVel = &pd.getVelocity(0);
	int gridNum_x, gridNum_y, gridNum_z;
	//���������Ĵ�Сȷ�� for test
	gridNum_x = 80;
	gridNum_y = 16;
	gridNum_z = 100;

	//��ʼ�����ž��볡
	model.resizeSignDistanceAndGradient(num_particles);
	
	model.computeSignDistanceAndGradient(m_sdf);

	pbf::GpuParticleSolver *gpuSolver = new pbf::GpuParticleSolver(num_particles, cellGridSize, particleRadius, kernelRadius, gridNum_x, gridNum_y, gridNum_z, particlePos, particleVel, model);
	
	//----------------------------------method for me-------------zzl------start-----------------------//
	////����ʵ�ֿ������ӷ������еĳ�ʼ������
	////gpuSolver->initializeForParticleControl(model);
	//
	////gpuSolver->updateMassForParticle(); //�������������̶�ʱ��ͨ���ı���������ʵ����״���

	////���������ܶȼ���
	//gpuSolver->neighborhoodSearch();
	//gpuSolver->computeDensity();
	////��������
	//gpuSolver->computeGradientOfColorField();
	//gpuSolver->computeFeedbackForce();
	////�����ٶȺ�λ��
	//gpuSolver->updateAcceleration();
	//gpuSolver->timeIntegratorForCloud();
	////����Ѱ������
	//gpuSolver->neighborhoodSearch();
	//gpuSolver->computeDensity();
	////ʵʩ����ѹ����
	//gpuSolver->enforecIncompressibility();
	////ʵʩ����ѹ��֮����ٶȸ��£� v=(newPos-oldPos)/timeStep
	//gpuSolver->updateVelocity();
	////����λ�ø�ֵ���洢λ�õı�����oldPos=newPos
	//gpuSolver->undatePostionAfterUpdateVelocity();

	//gpuSolver->copyDataToModel(model);

	////д�ļ�
	////gpuSolver->save(iniFileName, tarFileName, time, h);

	////
	////
	////gpuSolver->updatePhase();
	////
	////�������Ŀ�ʼʱ��
	////��������ʱ����������ע����
	////if (time > h * 10)
	////{
	////	gpuSolver->addOrDeleteParticle();
	////	gpuSolver->addParticle(model);
	////	gpuSolver->deleteParticle(model);
	////}

	////gpuSolver->computeMomentum();

	//gpuSolver->finalize();
	//----------------------------------method for me-------------zzl------end-----------------------//


	//�ͷ�Ϊʵ���ÿ������ӷ�������״�ݻ�������ı����ռ�
	//gpuSolver->finalizeForParticleControl();
	//updateTimeStepSizeCFL(model, 0.0001, 0.005);
	/** ----------end��zzl-------2018-9-10--------*/

	//--------------method for control-target-particle-method---------zzl---------//
	gpuSolver->simulateForConTarMethod(model);

	gpuSolver->saveForCompare(rootPath,iniFileName, tarFileName, time, h);
	//gpuSolver->saveForCompare(iniFileName, tarFileName, time, h);
	//�ͷ���Դ
	gpuSolver->finalize();
	gpuSolver->finalizeForParticleControl();
//===============================================================================//
//===============================================================================//

	///** ----------start��zzl-------2018-5-16-------- 
	///*����CPU�����SPH������������Ƶ��˶��ص㣬�����Ƶļ��ٶ�
	//*/
	//// step 1: ��������
	//START_TIMING("neighborhood search");
	//model.getNeighborhoodSearch()->neighborhoodSearch(&model.getParticles().getPosition(0), model.numBoundaryParticles(), &model.getBoundaryX(0));
	//STOP_TIMING_AVG;
	//// step 2: �����ܶ�
	//computeDensities(model);
	//// step 3: ����ѹ��
	////computePressure(model);
	////comoutePressureForce(model);
	//// step 4: ���㸡��
	//computeBuoyantByGradientOfTemperature(model); //�����¶��ݶȼ��㸡��
	////computeBuoyantByTemperature(model);         //���������¶�������������¶Ȳ���㸡��
	//// step 5: ��������
	//computePotentialForce(model, sdfGraField);
	//// step 6: ���¼��ٶ�
	//updateAccelerations(model);


	////-------------end:zzl-------2018-5-16--------

	//// Update time step size by CFL condition
	//updateTimeStepSizeCFL(model, 0.0001, 0.005);

	//// Time integration --ʱ����֣����ݼ��ٶȸ����ٶȺ�λ��
	//for (unsigned int i = 0; i < pd.size(); i++)
	//{ 
	//	model.getDeltaX(i).setZero();
	//	pd.getLastPosition(i) = pd.getOldPosition(i);
	//	pd.getOldPosition(i) = pd.getPosition(i);
	//	TimeIntegration::semiImplicitEuler(h, pd.getMass(i), pd.getPosition(i), pd.getVelocity(i), pd.getAcceleration(i));
	//}

	//
	////����SPH��������������λ�ú��������Ժ󣬲������㲻��ѹ����Լ��
	////����PBF�������������ܶȣ�ʵ�ֲ���ѹ����
	////������������------start:-zzl----------
	//model.getNeighborhoodSearch()->update();
	////-----end:zzl--------
	//// step 1: �����������
	//// Perform neighborhood search 
	//START_TIMING("neighborhood search");
	//model.getNeighborhoodSearch()->neighborhoodSearch(&model.getParticles().getPosition(0), model.numBoundaryParticles(), &model.getBoundaryX(0));
	//STOP_TIMING_AVG;
	//
	//// step 2: ����ܶ�Լ��
	//// Solve density constraint
	//START_TIMING("constraint projection");
	//constraintProjection(model);
	//STOP_TIMING_AVG;

	///*----------start: zzl---------2018-5-16---------
	///* ִ�����
	//*/
	////phaseChange(model);
	////------------end: zzl---------2018-5-16---------

	//// Update velocities	
	//for (unsigned int i = 0; i < pd.size(); i++)
	//{
	//	if (m_velocityUpdateMethod == 0)
	//		TimeIntegration::velocityUpdateFirstOrder(h, pd.getMass(i), pd.getPosition(i), pd.getOldPosition(i), pd.getVelocity(i));
	//	else
	//		TimeIntegration::velocityUpdateSecondOrder(h, pd.getMass(i), pd.getPosition(i), pd.getOldPosition(i), pd.getLastPosition(i), pd.getVelocity(i));
	//}

	//// Compute viscosity 
	//computeXSPHViscosity(model);

	//// Compute new time	
	//tm->setTime (tm->getTime () + h);
	//model.getNeighborhoodSearch()->update();

	tm->setTime(tm->getTime() + h);
	STOP_TIMING_AVG;
}


/** Clear accelerations and add gravitation.
 */
void TimeStepFluidModel::clearAccelerations(FluidModel &model)
{
	ParticleData &pd = model.getParticles();
	const unsigned int count = pd.size();
	const Vector3r grav(0.0, 9.81, 0.0);
	for (unsigned int i=0; i < count; i++)
	{
		// Clear accelerations of dynamic particles
		if (pd.getMass(i) != 0.0)
		{
			Vector3r &a = pd.getAcceleration(i);
			a = grav;
		}
	}
}

/** Determine densities of all fluid particles. 
*/
void TimeStepFluidModel::computeDensities(FluidModel &model)
{
	ParticleData &pd = model.getParticles();
	const unsigned int numParticles = pd.size();
	unsigned int **neighbors = model.getNeighborhoodSearch()->getNeighbors();
	unsigned int *numNeighbors = model.getNeighborhoodSearch()->getNumNeighbors();
	
	#pragma omp parallel default(shared)
	{
		#pragma omp for schedule(static)  
		for (int i = 0; i < (int) numParticles; i++)
		{
			Real &density = model.getDensity(i);
			Real density_err;
			PositionBasedFluids::computePBFDensity(i, numParticles, &pd.getPosition(0), &pd.getMass(0), &model.getBoundaryX(0), &model.getBoundaryPsi(0), numNeighbors[i], neighbors[i], model.getDensity0(), true, density_err, density);
		}
	}
}

/** Update time step size by CFL condition.
*/
void TimeStepFluidModel::updateTimeStepSizeCFL(FluidModel &model, const Real minTimeStepSize, const Real maxTimeStepSize)
{
	const Real radius = model.getParticleRadius();
	const Real cflFactor = 1.0;
	Real h = TimeManager::getCurrent()->getTimeStepSize();

	// Approximate max. position change due to current velocities
	Real maxVel = 0.1;
	ParticleData &pd = model.getParticles();
	const unsigned int numParticles = pd.size();
	const Real diameter = 2.0*radius;
	for (unsigned int i = 0; i < numParticles; i++)
	{
		/*const Vector3r &vel = pd.getVelocity(i);
		const Vector3r &accel = pd.getAcceleration(i);
		const Real velMag = (vel + accel*h).squaredNorm();
		if (velMag > maxVel)
			maxVel = velMag;*/

		//zzl---2019-1-4---
		const Vector3r &vel = pd.getVelocity(i);
		if (vel.squaredNorm() > maxVel)
			maxVel = vel.squaredNorm();
		//zzl---2019-1-4---end------
	}

	// Approximate max. time step size 		
	h = cflFactor * .4 * (diameter / (sqrt(maxVel)));

	h = min(h, maxTimeStepSize);
	h = max(h, minTimeStepSize);

	TimeManager::getCurrent()->setTimeStepSize(h);
}

/** Compute viscosity accelerations.
*/
void TimeStepFluidModel::computeXSPHViscosity(FluidModel &model)
{
	ParticleData &pd = model.getParticles();
	const unsigned int numParticles = pd.size();	

	unsigned int **neighbors = model.getNeighborhoodSearch()->getNeighbors();
	unsigned int *numNeighbors = model.getNeighborhoodSearch()->getNumNeighbors();

	const Real viscosity = model.getViscosity();
	const Real h = TimeManager::getCurrent()->getTimeStepSize();

	// Compute viscosity forces (XSPH)
	#pragma omp parallel default(shared)
	{
		#pragma omp for schedule(static)  
		for (int i = 0; i < (int)numParticles; i++)
		{
			const Vector3r &xi = pd.getPosition(i);
			Vector3r &vi = pd.getVelocity(i);
			const Real density_i = model.getDensity(i);
			for (unsigned int j = 0; j < numNeighbors[i]; j++)
			{
				const unsigned int neighborIndex = neighbors[i][j];
				if (neighborIndex < numParticles)		// Test if fluid particle
				{
					// Viscosity
					const Vector3r &xj = pd.getPosition(neighborIndex);
					const Vector3r &vj = pd.getVelocity(neighborIndex);
					const Real density_j = model.getDensity(neighborIndex);
					vi -= viscosity * (pd.getMass(neighborIndex) / density_j) * (vi - vj) * CubicKernel::W(xi - xj);

				}
// 				else 
// 				{
// 					const Vector3r &xj = model.getBoundaryX(neighborIndex - numParticles);
// 					vi -= viscosity * (model.getBoundaryPsi(neighborIndex - numParticles) / density_i) * (vi)* CubicKernel::W(xi - xj);
// 				}
			}
		}
	}
}




void TimeStepFluidModel::reset()
{

}


/** Solve density constraint.
*/
void TimeStepFluidModel::constraintProjection(FluidModel &model)
{
	const unsigned int maxIter = 5;
	unsigned int iter = 0;

	ParticleData &pd = model.getParticles();
	const unsigned int nParticles = pd.size();
	unsigned int **neighbors = model.getNeighborhoodSearch()->getNeighbors();
	unsigned int *numNeighbors = model.getNeighborhoodSearch()->getNumNeighbors();

	while (iter < maxIter)
	{
		Real avg_density_err = 0.0;

		#pragma omp parallel default(shared)
		{
			#pragma omp for schedule(static)  
			for (int i = 0; i < (int)nParticles; i++)
			{
				Real density_err;
				PositionBasedFluids::computePBFDensity(i, nParticles, &pd.getPosition(0), &pd.getMass(0), &model.getBoundaryX(0), &model.getBoundaryPsi(0), numNeighbors[i], neighbors[i], model.getDensity0(), true, density_err, model.getDensity(i));
				PositionBasedFluids::computePBFLagrangeMultiplier(i, nParticles, &pd.getPosition(0), &pd.getMass(0), &model.getBoundaryX(0), &model.getBoundaryPsi(0), model.getDensity(i), numNeighbors[i], neighbors[i], model.getDensity0(), true, model.getLambda(i));
			}
		}
		
		//�����ܶ�Լ�����õ����ӵ�����λ��
		#pragma omp parallel default(shared)
		{
			#pragma omp for schedule(static)  
			for (int i = 0; i < (int)nParticles; i++)
			{
				Vector3r corr;
				PositionBasedFluids::solveDensityConstraint(i, nParticles, &pd.getPosition(0), &pd.getMass(0), &model.getBoundaryX(0), &model.getBoundaryPsi(0), numNeighbors[i], neighbors[i], model.getDensity0(), true, &model.getLambda(0), corr);
				model.getDeltaX(i) = corr;
			}
		}

		//���ݵõ�������λ�ƣ��޸����ӵ�λ��
		#pragma omp parallel default(shared)
		{
			#pragma omp for schedule(static)  
			for (int i = 0; i < (int)nParticles; i++)
			{
				pd.getPosition(i) += model.getDeltaX(i);
			}
		}

		iter++;
	}
}

//--------------zzl--------------------------//
// add pressure force : computePressureForce
// add buoyancy force : computeBuoyancyForce

/** coumpute pressure
*/

void PBD::TimeStepFluidModel::computePressure(FluidModel &model)
{
	ParticleData &pd = model.getParticles();
	const unsigned int particleNum = pd.size();
	const Real densty0 = model.getDensity0();
	const Real k = 0.8; // k denotes as the gas constant 
	for (int i = 0; i < particleNum; i++)
	{
		const Real denstyi = model.getDensity(i);
		model.setPressure(i, k*(denstyi - densty0));
	}
}

/* �����Ƴ���
*/
void PBD::TimeStepFluidModel::computePotentialForce(FluidModel & model, SDFGradientField &potentialField)
{
	ParticleData &pd = model.getParticles();
	const unsigned int particleNum = pd.size();
	
	#pragma omp parallel default(shared)
	{
		#pragma omp for shcedule(static) 
		for (unsigned int i =0; i < particleNum; i++)
		{
			Eigen::Vector3d position = pd.getPosition(i); //��ȡ���ӵ�λ��
			const Real density = model.getDensity(i);
			Eigen::Vector3d &potential = model.getPotentialForce(i);
			//std::cout << position << std::endl;
			Eigen::Vector3d tempPotential = potentialField.getPotential(position); //����λ�û���Ƴ���С
			potential = tempPotential; //��������λ�õ��Ƴ�����С
			std::cout << "potential: x=" << tempPotential.x() << "  y=" << tempPotential.y() << "  z=" << tempPotential.z() << std::endl;
		}
	}

}

//void PBD::TimeStepFluidModel::computeSignDistanceAndGradient(FluidModel & model, SDFGradientField & potentialField)
//{
//	ParticleData &pd = model.getParticles();
//	const unsigned int pdNum = pd.size();
//
//	#pragma omp parallel default(shared)
//	{
//	#pragma omp for shcedule(static) 
//		for (unsigned int i = 0; i < pdNum; i++)
//		{
//			//potentialField
//			//model.setSignDistance(i,)
//		}
//	}
//}

/* ����������Χ���������Ӹ�����λ�ý������������ĸ��£���ӻ�ɾ����
*/
void PBD::TimeStepFluidModel::updateParticleNum(FluidModel & model)
{
	const Real densityTheasholdMax = model.getDensity0() * (1 + 0.2); //�����ܶȵ����ֵ�����ܶ�>��ֵ������Ҫɾ�����ӡ�
	const Real densityThreasholdMin = model.getDensity0() * (1 - 0.8); //�����ܶȵ���Сֵ�����ܶ�<��ֵ������Ҫ�������ӡ�
	
	ParticleData &pd = model.getParticles();
	const unsigned int pdNum = pd.size();
	#pragma omp parall default(shared)
	{
		#pragma omp for schedule(static)
		for (unsigned int i = 0; i < pdNum; i++)
		{
			const Real density_i = model.getDensity(i);
			
			//decrease particle's density
			if (density_i > densityTheasholdMax)
			{
				model.decreaseDensity(i);
			}
			//increase particle's density
			else if (density_i < densityThreasholdMin)
			{
				//ADD: ͨ�������������������ܶȴ���
				//model.increaseDensity(i,model.getDensity0()-density_i,0.01,);
			}
		}
	}
}

/* ���ļ��м���SDF
*/
Discregrid::CubicLagrangeDiscreteGrid PBD::TimeStepFluidModel::computeSDF(std::string fileName)
{
	
	//Discregrid::CubicLagrangeDiscreteGrid sdf = NULL;
	auto lastindex = fileName.find_last_of(".");
	auto extension = fileName.substr(lastindex + 1, fileName.length() - lastindex);
	std::cout << "LOAD SDF..." << endl;
	if (extension == "cdf")
	{
		//sdf = std::make_unique<Discregrid::CubicLagrangeDiscreteGrid>(fileName);
		Discregrid::CubicLagrangeDiscreteGrid sdf(fileName);
		return sdf;
	}
	else
	{
		std::cout << "LOAD SDF Fail" << endl;
		exit(1);
	}
	//return sdf;
}



/** determine the pressure force of all particles
*/
void PBD::TimeStepFluidModel::comoutePressureForce(FluidModel &model) 
{
	ParticleData &pd = model.getParticles();
	const unsigned int particleNum = pd.size();

	unsigned int ** neighbors = model.getNeighborhoodSearch()->getNeighbors();
	unsigned int * neighborsNum = model.getNeighborhoodSearch()->getNumNeighbors();
	#pragma omp parallel default(shared)
	{
		#pragma omp for schedule(static)	
		for (int i = 0; i < (int)particleNum; i++)
		{
			Vector3r & pressureForce_i = model.getPressureForce(i); //pressure force
			Real & pressure_i = model.getPressure(i);  //pressure
			Vector3r &xi = pd.getPosition(i);
			Vector3r tempPressureForce(0.0, 0.0, 0.0);
			for (int j = 0; j < neighborsNum[i]; j++)
			{
				Real & pressure_j = model.getPressure(j);
				Vector3r & xj = pd.getPosition(j);
				Real & density_j = model.getDensity(j);
				// �ο����ġ�Particle-Based Fluid Simulation for Interactive Applications��
				tempPressureForce -= (pd.getMass(j)*(pressure_i + pressure_j) / (2*density_j))*CubicKernel::gradW(xi - xj);
			}
			pressureForce_i = tempPressureForce;
		}
	}
}

/** determin the buoyant by temperature
*/
void PBD::TimeStepFluidModel::computeBuoyantByTemperature(FluidModel & model)
{
	ParticleData &pd = model.getParticles();
	const unsigned int numParticles = pd.size();

	const Vector3r gravity(0.0, 9.81, 0.0);
	const Real temv0 = 295;
	for (unsigned int i = 0; i < numParticles; i++)
	{
		/*  using the method in [Harris] to calculate buoyant 
		*/
		Vector3r &buoyant = model.getBuoyant(i);
		Real temv = pd.getTemperature(i) * (1 + 0.61 * pd.getVapor(i));
		buoyant = gravity * (temv / temv0 - pd.getCloud(i));
		
	}
}

void PBD::TimeStepFluidModel::computeBuoyantByGradientOfTemperature(FluidModel & model)
{
	ParticleData &pd = model.getParticles();
	const unsigned int numParticles = pd.size();

	// get neighbors of the particle i
	unsigned int **neighbors = model.getNeighborhoodSearch()->getNeighbors();
	unsigned int *numNeighbors = model.getNeighborhoodSearch()->getNumNeighbors();
	
	const Real h = TimeManager::getCurrent()->getTimeStepSize();
	
	// compute buoyant force
	#pragma omp parallel default(shared)
	{
		#pragma omp for schedule(static)		
		for (int i = 0; i < (int)numParticles; i++)
		{
			Vector3r &xi = pd.getPosition(i); // ��ǰ���ӵ�λ��
			Vector3r &buoyant = model.getBuoyant(i);
			Vector3r temp_buoyant(0.0,0.0,0.0);
			for (int j = 0; j < numNeighbors[i]; j++)
			{
				const int neighborIndex = neighbors[i][j];
				if (neighborIndex < numParticles) //�ж��ھ������Ƿ�����������
				{
					Vector3r &xj = pd.getPosition(neighborIndex);
					const Real density_j = model.getDensity(neighborIndex);
					// ���ݡ�Modeling and Characterization of Cloud Dynamics����������SPH����ԭ����㸡��
					temp_buoyant += (pd.getMass(neighborIndex) / density_j) * pd.getTemperature(neighborIndex) * CubicKernel::gradW(xi - xj); 
				}
			}
			buoyant = temp_buoyant;
			//cout << "x="<< buoyant.x() <<",  y="<< buoyant.y() <<",   z="<< buoyant.z() << endl;
		}
	}
}

/** �Ƶ���䣺������ˮ��֮��ı任���������Ļ����¶ȵı任
*/
void PBD::TimeStepFluidModel::phaseChange(FluidModel & model)
{
	ParticleData &pd = model.getParticles();
	const unsigned int particleNum = pd.size();
	//update vapor,cloud and temperature
	/** according to the method:"Modeling and Characterization of Cloud Dynamics"
	/*"Adaptive cloud simulation using position based fluids"
	/*  ��������ֵ�� satQv=A*exp(B/(T+C))
	*/
	//��ʽ��������ĳ�ʼ��
	const Real A = 100.0;
	const Real B = 3.0;
	const Real C = 2.3;
	const Real alpha = 0.2;
	const Real Q = 0.2; //latent heat coefficient
	#pragma omp parallel default(shared)
	{
		#pragma omp for schedule(static)
		for (int i = 0; i < particleNum; i++)
		{
			const Real vapor_old = pd.getVapor(i);
			const Real cloud_old = pd.getCloud(i);
			const Real temperature_old = pd.getTemperature(i);
			
			//���㱥�������ܶ�
			Real satQv; //���������ܶ�
			Real w1 = A*exp((-1 * B) / (temperature_old + C));
			Real w2 = vapor_old + cloud_old;

			if (w1 > w2)
			{
				satQv = w1;
			}
			else
			{
				satQv = w2;
			}

			//����vapor, cloud, temperature
			Real & vapor = pd.getVapor(i);
			Real & cloud = pd.getCloud(i);
			Real & temperature = pd.getTemperature(i);

			Real deltaC = alpha * (vapor_old - satQv);
			vapor = vapor_old + deltaC;
			cloud = cloud_old - deltaC;
			temperature = temperature_old - Q * 2 *deltaC; 
		}
	}
}

void PBD::TimeStepFluidModel::updateAccelerations(FluidModel &model)
{
	ParticleData &pd = model.getParticles();
	const unsigned int particleNum = pd.size();
	Vector3r gravityForce(0.0, -9.8, 0.0);
	
	#pragma omp parallel default(shared)
	{
		#pragma omp for schedule(static)
		for (int i = 0; i < particleNum; i++)
		{
			Vector3r &acceleration = pd.getAcceleration(i);
			Vector3r force(0.0, 0.0, 0.0);
			force += model.getPressureForce(i);
			force += model.getBuoyant(i);
			force += gravityForce;
			//acceleration = force / pd.getMass(i);
			acceleration = force / (pd.getCloud(i) + pd.getVapor(i));
		}

	}
	
}

// for SDFGradientField Class
PBD::SDFGradientField::SDFGradientField(Discregrid::CubicLagrangeDiscreteGrid sdf)
{
	m_domain = sdf.domain();
	m_cell_size = sdf.cellSize();
	m_resolution = sdf.resolution();
	m_inv_cell_size = sdf.invCellSize();
	auto n = Eigen::Matrix<unsigned int, 3, 1>::Map(sdf.resolution().data());
	//cout << "min: " << m_domain.min().x() << " " << m_domain.min().y() << " " << m_domain.min().z() << endl;
	//cout << "max: " << m_domain.max().x() << " " << m_domain.max().y() << " " << m_domain.max().z() << endl;
	//cout << n[0] << " " << n[1] << " " << n[2] << endl;
	// ���ݷֱ��ʹ�������
	auto nv = (n[0] + 1) * (n[1] + 1) * (n[2] + 1);
	auto ne_x = (n[0] + 0) * (n[1] + 1) * (n[2] + 1); //x ����ϸ��
	auto ne_y = (n[0 + 1]) * (n[1] + 0) * (n[2] + 1); //y ����ϸ��
	auto ne_z = (n[0] + 1) * (n[1] + 1) * (n[2] + 0); //z ����ϸ��
	auto ne = ne_x + ne_y + ne_z;
	auto n_nodes = nv + 2 * ne;

	//�ݶȳ�ʼ��
	m_gradient.resize(n_nodes);

	//����
	/*Eigen::Vector3d testx = { 0,0,0 };
	std::cout << sdf.interpolate(0, testx) << std::endl;
	std::cout << sdf.domain().max() << std::endl;*/

	//��ȡ��Χ�еķ�Χ
	double max_x = sdf.domain().max().x();
	double max_y = sdf.domain().max().y();
	double max_z = sdf.domain().max().z();

	double min_x = sdf.domain().min().x();
	double min_y = sdf.domain().min().y();
	double min_z = sdf.domain().min().z();

	int num = 0;
	#pragma omp parallel default(shared)
	{
	#pragma omp for shcedule(static) nowait
		for (int l = 0; l < static_cast<int>(n_nodes); l++)
		{
			Eigen::Vector3d & g = m_gradient[l];
			// ��õ�ǰ�����ά����
			Eigen::Vector3d x = indexToPosition(l);

			double positionsdf = sdf.interpolate(0, x); //��ȡ��ǰλ�õ�sdfֵ

			
			if (x.x() <= max_x && x.x() >= min_x && x.y() <= max_y && x.y() >= min_y && x.z() <= max_z && x.z() >= min_z)
			{
				Eigen::Vector3d grad_x;
				auto val2 = sdf.interpolate(0, x, &grad_x);
				g = grad_x;
			}	
			
			//// ����ǰ���ּ����ݶ�
			//double xsdf, ysdf, zsdf;
			//Eigen::Vector3d x_x1, x_y1, x_z1;
			//double maxX = m_domain.max().x() - static_cast<double>(m_cell_size(0)) / 3.0;
			//double maxY = m_domain.max().y() - static_cast<double>(m_cell_size(1)) / 3.0;
			//double maxZ = m_domain.max().z() - static_cast<double>(m_cell_size(2)) / 3.0;
			////x����
			//if (x(0) < maxX)
			//{
			//	x_x1 = { x(0) + static_cast<double>(m_cell_size(0)) / 3.0,x(1),x(2) };
			//}
			//else
			//{
			//	x_x1 = { x(0) - static_cast<double>(m_cell_size(0)) / 3.0,x(1),x(2) };
			//	//xsdf = static_cast<double>(sdf.interpolate(0, x_x1));
			//}
			//xsdf = static_cast<double>(sdf.interpolate(0, x_x1)); //x����SDF
	
			//// y direction
			//if (x(1) < maxY)
			//{
			//	x_y1 = { x(0),x(1) + static_cast<double>(m_cell_size(1)) / 3.0,x(2) };
			//}
			//else
			//{
			//	x_y1 = { x(0),x(1) - static_cast<double>(m_cell_size(1)) / 3.0,x(2) };
			//}
			//ysdf = static_cast<double>(sdf.interpolate(0, x_y1)); //y����SDF

			////z direction
			//if (x(2) < maxZ)
			//{
			//	x_z1 = { x(0),x(1),x(2) + static_cast<double>(m_cell_size(2)) / 3.0 };
			//}
			//else
			//{
			//	x_z1 = { x(0),x(1),x(2) - static_cast<double>(m_cell_size(2)) / 3.0 };
			//}
			//zsdf = static_cast<double>(sdf.interpolate(0, x_z1)); //z����SDF
			//
			//g[0] = (xsdf - positionsdf) / (static_cast<double>(m_cell_size(0)) / 3.0);
			//g[1] = (ysdf - positionsdf) / (static_cast<double>(m_cell_size(1)) / 3.0);
			//g[2] = (zsdf - positionsdf) / (static_cast<double>(m_cell_size(2)) / 3.0);
			
			//output gradient
			//cout << "[x]=" << g[0] << "[y]=" << g[1] << "[z]=" << g[2] << endl;
		}
	}

}

Eigen::Vector3d PBD::SDFGradientField::indexToPosition(unsigned int l)
{
	Eigen::Vector3d x;

	auto n = Eigen::Matrix<unsigned int, 3, 1>::Map(m_resolution.data());

	auto nv = (n[0] + 1) * (n[1] + 1) * (n[2] + 1);
	auto ne_x = (n[0] + 0) * (n[1] + 1) * (n[2] + 1);
	auto ne_y = (n[0] + 1) * (n[1] + 0) * (n[2] + 1);
	auto ne_z = (n[0] + 1) * (n[1] + 1) * (n[2] + 0);
	auto ne = ne_x + ne_y + ne_z;

	auto ijk = Eigen::Matrix<unsigned int, 3, 1>{};
	if (l < nv)
	{
		ijk(2) = l / ((n[0] + 1)*(n[1] + 1));
		auto temp = l % ((n[0] + 1)*(n[1] + 1));
		ijk(1) = temp / (n[0] + 1);
		ijk(0) = temp % (n[0] + 1);

		x = m_domain.min() + m_cell_size.cwiseProduct(ijk.cast<double>());
	}
	else if(l < nv + 2 * ne_x)
	{
		l -= nv;
		auto e_ind = l / 2;
		ijk(2) = e_ind / (n[0] * (n[1] + 1));
		auto temp = e_ind % (n[0] * (n[1] + 1));
		ijk(1) = temp / n[0];
		ijk(0) = temp % n[0];

		x = m_domain.min() + m_cell_size.cwiseProduct(ijk.cast<double>());
		x(0) += (1.0 + static_cast<double>(l%2)) / 3 * m_cell_size[0];
	}
	else if (l < nv + 2 * (ne_x + ne_y))
	{
		l -= (nv + 2 * ne_x);
		auto e_ind = l / 2;
		ijk(0) = e_ind / ((n[2] + 1) * n[1]);
		auto temp = e_ind % ((n[2] + 1) * n[1]);
		ijk(2) = temp / n[1];
		ijk(1) = temp % n[1];

		x = m_domain.min() + m_cell_size.cwiseProduct(ijk.cast<double>());
		x(1) += (1.0 + static_cast<double>(l % 2)) / 3.0 * m_cell_size[1];
	}
	else
	{
		l -= (nv + 2 * (ne_x + ne_y));
		auto e_ind = l / 2;
		ijk(1) = e_ind / ((n[0] + 1) * n[2]);
		auto temp = e_ind % ((n[0] + 1) * n[2]);
		ijk(0) = temp / n[2];
		ijk(2) = temp % n[2];

		x = m_domain.min() + m_cell_size.cwiseProduct(ijk.cast<double>());
		x(2) += (1.0 + static_cast<double>(l % 2)) / 3.0 * m_cell_size[2];
	}
	return x;
}

Eigen::Vector3d PBD::SDFGradientField::getGradient(Eigen::Vector3d &x)
{
	Eigen::Vector3d gradient;
	auto r_x = (x - m_domain.min()).cwiseProduct(m_inv_cell_size).cast<int>().eval(); //������ת���ɷֱ���
	if (r_x[0] >= m_resolution[0])
		r_x[0] = m_resolution[0] - 1;
	if (r_x[1] >= m_resolution[1])
		r_x[1] = m_resolution[1] - 1;
	if (r_x[2] >= m_resolution[2])
		r_x[2] = m_resolution[2] - 1;
	Eigen::Vector3d position(r_x(0),r_x(1),r_x(2));
	
	int  l = multiToSingleIndex(position);
	gradient = m_gradient[l];
	return gradient;
}

void PBD::SDFGradientField::normGradient()
{
	unsigned int size = m_gradient.size();
	for (int i = 0; i < size; i++)
	{
		Eigen::Vector3d &v = m_gradient[i];
		v = v / sqrt(v.squaredNorm());
	}
}

unsigned int PBD::SDFGradientField::multiToSingleIndex(Eigen::Vector3d &x)
{
	unsigned int l;
	l = m_resolution[2] * m_resolution[1] * x[2] + m_resolution[1] * x[1] + x[0];
	return l;
}


/* ����SDF���ݶȳ����ܶȳ����Ƴ�
* |SDF|>sdfThreshold && SDF<0 ----> windForce = |SDF|*(0,1,0)
* |SDF|<sdfThreshold && SDF<0 ----> windForce = |SDF|*normGradient
* SDF>0 ----> windForce = |SDF|*-1*normGradient
*/
void PBD::SDFGradientField::computPotentialField(Discregrid::CubicLagrangeDiscreteGrid & sdf)
{
	unsigned int field_id = 0;

	Eigen::Vector3d vertical{ 0,1,0 };
	double sdfThreshold = 0.8;
	normGradient(); //���ݶȳ����й�һ������
	const unsigned int gridNum = m_gradient.size(); //�������ĸ����
	m_potentialField.resize(gridNum); //��ʼ�����Ƴ���С

	#pragma omp parallel default(shared)
	{
		#pragma omp for shcedule(static)
		for (unsigned int i = 0; i < gridNum; i++)
		{
			Eigen::Vector3d &potentialForce = m_potentialField[i]; //��ȡ�����Ƴ������еĵ�i��ֵ

			Eigen::Vector3d position;
			position = indexToPosition(i); //����������ת������ά����
														 
			double sdfValue;
			sdfValue = sdf.interpolate(field_id, position); // ��ȡSDF��ֵ
			//��ȡGradient��ֵ
			Eigen::Vector3d gradient;
			gradient = m_gradient[i]; //��õ�ǰλ�õ��ݶ�����
			//std::cout << gradient.x() << " " << gradient.y() << " " << gradient.z() << std::endl;
			
			//�����Ƴ���С
			//
			double forceMagnitude = abs(sdfValue); //����sdfֵȷ�����Ĵ�С��Ҳ�����������������������Ĵ�С
			if (sdfValue <0 && abs(sdfValue)>sdfThreshold)
			{
				potentialForce = forceMagnitude * vertical; //���ǽ���ֵ��ֵΪ����������
			}
			else if (sdfValue < 0 && abs(sdfValue) <= sdfThreshold)
			{
				potentialForce = forceMagnitude * gradient;
			}
			else
			{
				//outside the target shpae : the direction is (-1*gradient)
				potentialForce = forceMagnitude * (-1) * gradient;
			}

		}
	}


}

Eigen::Vector3d PBD::SDFGradientField::getPotential(Eigen::Vector3d & x)
{
	Eigen::Vector3d potential;
	auto r_x = (x - m_domain.min()).cwiseProduct(m_inv_cell_size).cast<int>().eval(); //������ת���ɷֱ���
	if (r_x[0] >= m_resolution[0])
		r_x[0] = m_resolution[0] - 1;
	if (r_x[1] >= m_resolution[1])
		r_x[1] = m_resolution[1] - 1;
	if (r_x[2] >= m_resolution[2])
		r_x[2] = m_resolution[2] - 1;
	Eigen::Vector3d position(r_x(0), r_x(1), r_x(2));

	int  l = multiToSingleIndex(position);
	potential = m_potentialField[l];
	return potential;
}
