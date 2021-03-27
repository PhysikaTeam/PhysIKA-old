#include "TimeStepPCISPH.h"
#include "SPlisHSPlasH/TimeManager.h"
#include "SPlisHSPlasH/SPHKernels.h"
#include "SimulationDataPCISPH.h"
#include <iostream>
#include "Utilities/Timing.h"
#include "SPlisHSPlasH/Simulation.h"
#include "SPlisHSPlasH/BoundaryModel_Akinci2012.h"
#include "SPlisHSPlasH/BoundaryModel_Koschier2017.h"
#include "SPlisHSPlasH/BoundaryModel_Bender2019.h"


using namespace SPH;
using namespace std;

TimeStepPCISPH::TimeStepPCISPH() :
	TimeStep()
{
	m_simulationData.init();
	m_counter = 0;
	m_minIterations = 3;

	Simulation *sim = Simulation::getCurrent();
	const unsigned int nModels = sim->numberOfFluidModels();
	for (unsigned int fluidModelIndex = 0; fluidModelIndex < nModels; fluidModelIndex++)
	{
		FluidModel *model = sim->getFluidModel(fluidModelIndex);
		model->addField({ "pressure", FieldType::Scalar, [this, fluidModelIndex](const unsigned int i) -> Real* { return &m_simulationData.getPressure(fluidModelIndex, i); } });
		model->addField({ "advected density", FieldType::Scalar, [this, fluidModelIndex](const unsigned int i) -> Real* { return &m_simulationData.getDensityAdv(fluidModelIndex, i); } });
		model->addField({ "pressure acceleration", FieldType::Vector3, [this, fluidModelIndex](const unsigned int i) -> Real* { return &m_simulationData.getPressureAccel(fluidModelIndex, i)[0]; } });
	}
}

TimeStepPCISPH::~TimeStepPCISPH(void)
{
	Simulation *sim = Simulation::getCurrent();
	const unsigned int nModels = sim->numberOfFluidModels();
	for (unsigned int fluidModelIndex = 0; fluidModelIndex < nModels; fluidModelIndex++)
	{
		FluidModel *model = sim->getFluidModel(fluidModelIndex);
		model->removeFieldByName("pressure");
		model->removeFieldByName("advected density");
		model->removeFieldByName("pressure acceleration");
	}
}

void TimeStepPCISPH::step()
{
	Simulation *sim = Simulation::getCurrent();
	TimeManager *tm = TimeManager::getCurrent();
	const Real h = tm->getTimeStepSize();
	const unsigned int nModels = sim->numberOfFluidModels();

	performNeighborhoodSearch();

#ifdef USE_PERFORMANCE_OPTIMIZATION
	precomputeValues();
#endif


	if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Bender2019)
		computeVolumeAndBoundaryX();
	else if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Koschier2017)
		computeDensityAndGradient();

	// Compute accelerations: a(t)
	for (unsigned int fluidModelIndex = 0; fluidModelIndex < nModels; fluidModelIndex++)
	{
		clearAccelerations(fluidModelIndex);
		computeDensities(fluidModelIndex);
	}
	sim->computeNonPressureForces();

	sim->updateTimeStepSize();

	START_TIMING("pressureSolve");
	pressureSolve();
	STOP_TIMING_AVG;

	for (unsigned int fluidModelIndex = 0; fluidModelIndex < nModels; fluidModelIndex++)
	{
		FluidModel *model = sim->getFluidModel(fluidModelIndex);
		const int numParticles = (int)model->numActiveParticles();

		#pragma omp parallel default(shared)
		{
			#pragma omp for schedule(static)  
			for (int i = 0; i < numParticles; i++)
			{
				if (model->getParticleState(i) == ParticleState::Active)
				{
					const Vector3r accel = model->getAcceleration(i) + m_simulationData.getPressureAccel(fluidModelIndex, i);
					const Vector3r &lastX = m_simulationData.getLastPosition(fluidModelIndex, i);
					const Vector3r &lastV = m_simulationData.getLastVelocity(fluidModelIndex, i);
					Vector3r &x = model->getPosition(i);
					Vector3r &v = model->getVelocity(i);
					v = lastV + h * accel;
					x = lastX + h * v;
				}
			}
		}
	}

	sim->emitParticles();
	sim->animateParticles();

	// Compute new time		
	tm->setTime(tm->getTime() + h);
}

void TimeStepPCISPH::pressureSolve()
{
	Simulation *sim = Simulation::getCurrent();
	const unsigned int nFluids = sim->numberOfFluidModels();

	// Initialization
	for (unsigned int fluidModelIndex = 0; fluidModelIndex < nFluids; fluidModelIndex++)
	{
		FluidModel *model = sim->getFluidModel(fluidModelIndex);
		const int numParticles = (int)model->numActiveParticles();
		for (int i = 0; i < numParticles; i++)
		{
			Vector3r &lastX = m_simulationData.getLastPosition(fluidModelIndex, i);
			Vector3r &lastV = m_simulationData.getLastVelocity(fluidModelIndex, i);
			lastX = model->getPosition(i);
			lastV = model->getVelocity(i);
			m_simulationData.getPressure(fluidModelIndex, i) = 0.0;
			m_simulationData.getPressureAccel(fluidModelIndex, i).setZero();
		}
	}

	Real avg_density_err = 0;
	m_iterations = 0;
	bool chk = false;

	while ((!chk || (m_iterations < m_minIterations)) && (m_iterations < m_maxIterations))
	{
		chk = true;
		for (unsigned int i = 0; i < nFluids; i++)
		{
			FluidModel *model = sim->getFluidModel(i);
			const Real density0 = model->getDensity0();

			avg_density_err = 0.0;
			pressureSolveIteration(i, avg_density_err);

			// Maximal allowed density fluctuation
			const Real eta = m_maxError * static_cast<Real>(0.01) * density0;  // maxError is given in percent
			chk = chk && (avg_density_err <= eta);
		}

		m_iterations++;
	}
}

void TimeStepPCISPH::pressureSolveIteration(const unsigned int fluidModelIndex, Real &avg_density_err)
{
	Simulation *sim = Simulation::getCurrent();
	FluidModel *model = sim->getFluidModel(fluidModelIndex);
	const int numParticles = (int)model->numActiveParticles();
	if (numParticles == 0)
		return;

	const Real density0 = model->getDensity0();
	const Real h = TimeManager::getCurrent()->getTimeStepSize();
	const Real h2 = h*h;
	const Real invH2 = static_cast<Real>(1.0) / h2;
	const unsigned int nFluids = sim->numberOfFluidModels();
	const unsigned int nBoundaries = sim->numberOfBoundaryModels();

	#pragma omp parallel default(shared)
	{
		#pragma omp for schedule(static)  
		for (int i = 0; i < numParticles; i++)
		{
			if (model->getParticleState(i) == ParticleState::Active)
			{
				const Vector3r accel = model->getAcceleration(i) + m_simulationData.getPressureAccel(fluidModelIndex, i);
				const Vector3r &lastX = m_simulationData.getLastPosition(fluidModelIndex, i);
				const Vector3r &lastV = m_simulationData.getLastVelocity(fluidModelIndex, i);
				Vector3r &x = model->getPosition(i);
				Vector3r &v = model->getVelocity(i);
				v = lastV + h * accel;
				x = lastX +h * v;
 				if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Bender2019)
 					computeVolumeAndBoundaryX(fluidModelIndex, i, x);
 				else if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Koschier2017)
 					computeDensityAndGradient(fluidModelIndex, i, x);
			}
		}
	}


	// Predict density 
	#pragma omp parallel default(shared)
	{
		#pragma omp for schedule(static)  
		for (int i = 0; i < numParticles; i++)
		{
			const Vector3r &xi = model->getPosition(i);
			Real &densityAdv = m_simulationData.getDensityAdv(fluidModelIndex, i);
			densityAdv = model->getVolume(i) * sim->W_zero();
				
			//////////////////////////////////////////////////////////////////////////
			// Fluid
			//////////////////////////////////////////////////////////////////////////
			forall_fluid_neighbors(
				densityAdv += fm_neighbor->getVolume(neighborIndex) * sim->W(xi - xj);
			);

			//////////////////////////////////////////////////////////////////////////
			// Boundary
			//////////////////////////////////////////////////////////////////////////
			if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Akinci2012)
			{
				forall_boundary_neighbors(
					densityAdv += bm_neighbor->getVolume(neighborIndex) * sim->W(xi - xj);
				);
			}
			else if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Koschier2017)
			{
				forall_density_maps(
					densityAdv += rho;
				);
			}
			else if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Bender2019)
			{
				forall_volume_maps(
					densityAdv += static_cast<Real>(0.25)*Vj * sim->W(xi - xj);		// reduce volume to make it more stable
				);
			}

			densityAdv = max(densityAdv, static_cast<Real>(1.0));
			const Real density_err = density0 * (densityAdv - static_cast<Real>(1.0));
			Real &pressure = m_simulationData.getPressure(fluidModelIndex, i);
			pressure += invH2 * m_simulationData.getPCISPH_ScalingFactor(fluidModelIndex) * (densityAdv - static_cast<Real>(1.0));

			#pragma omp atomic
			avg_density_err += density_err;
		}
	}

	avg_density_err /= numParticles;

	// Compute pressure forces
	#pragma omp parallel default(shared)
	{
		#pragma omp for schedule(static)  
		for (int i = 0; i < (int)numParticles; i++)
		{
			const Vector3r &xi = m_simulationData.getLastPosition(fluidModelIndex, i);

			Vector3r &ai = m_simulationData.getPressureAccel(fluidModelIndex, i);
			ai.setZero();

			const Real dpi = m_simulationData.getPressure(fluidModelIndex, i);	

			//////////////////////////////////////////////////////////////////////////
			// Fluid
			//////////////////////////////////////////////////////////////////////////
			forall_fluid_neighbors(
				// Pressure 
				const Real dpj = m_simulationData.getPressure(pid, neighborIndex);
				ai -= fm_neighbor->getVolume(neighborIndex) * (dpi + (fm_neighbor->getDensity0() / density0) * dpj) * sim->gradW(xi - xj);
			);

			//////////////////////////////////////////////////////////////////////////
			// Boundary
			//////////////////////////////////////////////////////////////////////////
			if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Akinci2012)
			{
				forall_boundary_neighbors(
					// Pressure 
					const Vector3r a = bm_neighbor->getVolume(neighborIndex) * dpi * sim->gradW(xi - xj);
					ai -= a;
					bm_neighbor->addForce(xj, model->getMass(i) * a);
				);
			}
			else if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Koschier2017)
			{
				forall_density_maps(
					const Vector3r a = -dpi * gradRho;
					ai -= a;
					bm_neighbor->addForce(xj, model->getMass(i) * a);
				);
			}
 			else if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Bender2019)
 			{
 				forall_volume_maps(
 					const Vector3r a = 0.25*Vj *dpi * sim->gradW(xi - xj);		// reduce volume to make it more stable
 					ai -= a;
 					bm_neighbor->addForce(xj, model->getMass(i) * a);
 				);
 			}
		}
	}
}

void TimeStepPCISPH::reset()
{
	TimeStep::reset();
	m_simulationData.reset();
	m_counter = 0;
}

void TimeStepPCISPH::performNeighborhoodSearch()
{
	if (Simulation::getCurrent()->zSortEnabled())
	{
		if (m_counter % 500 == 0)
		{
			Simulation::getCurrent()->performNeighborhoodSearchSort();
			m_simulationData.performNeighborhoodSearchSort();
		}
		m_counter++;
	}

	Simulation::getCurrent()->performNeighborhoodSearch();
}

void TimeStepPCISPH::emittedParticles(FluidModel *model, const unsigned int startIndex)
{
	m_simulationData.emittedParticles(model, startIndex);
}

void TimeStepPCISPH::resize()
{
	m_simulationData.init();
}
