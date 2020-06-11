#include "ParticleRod.h"
#include "Framework/Topology/PointSet.h"
#include "Core/Utility.h"
#include "Framework/Topology/NeighborQuery.h"
#include "ParticleIntegrator.h"
#include "ElasticityModule.h"
#include "FixedPoints.h"
#include "SimpleDamping.h"

namespace PhysIKA
{
	IMPLEMENT_CLASS_1(ParticleRod, TDataType)

	template<typename TDataType>
	ParticleRod<TDataType>::ParticleRod(std::string name)
		: ParticleSystem<TDataType>(name)
	{
		m_horizon.setValue(0.008);
		this->attachField(&m_horizon, "horizon", "horizon");

		m_integrator = this->template setNumericalIntegrator<ParticleIntegrator<TDataType>>("integrator");
		this->getPosition()->connect(m_integrator->m_position);
		this->getVelocity()->connect(m_integrator->m_velocity);
		this->getForce()->connect(m_integrator->m_forceDensity);

		auto m_nbrQuery = this->template addComputeModule<NeighborQuery<TDataType>>("neighborhood");
		m_horizon.connect(m_nbrQuery->m_radius);
		this->m_position.connect(m_nbrQuery->m_position);

		m_elasticity = this->template addConstraintModule<ElasticityModule<TDataType>>("elasticity");
		this->getPosition()->connect(m_elasticity->m_position);
		this->getVelocity()->connect(m_elasticity->m_velocity);
		m_horizon.connect(m_elasticity->m_horizon);
		m_nbrQuery->m_neighborhood.connect(m_elasticity->m_neighborhood);
		m_elasticity->setIterationNumber(1);

		m_fixed = this->template addConstraintModule<FixedPoints<TDataType>>("fixed");
		this->getPosition()->connect(m_fixed->m_position);
		this->getVelocity()->connect(m_fixed->m_velocity);

		m_damping = this->template addConstraintModule<SimpleDamping<TDataType>>("damping");
		this->getVelocity()->connect(m_damping->m_velocity);
	}

	template<typename TDataType>
	ParticleRod<TDataType>::~ParticleRod()
	{
		
	}

	template<typename TDataType>
	void ParticleRod<TDataType>::setParticles(std::vector<Coord> particles)
	{
		this->m_pSet->setPoints(particles);
	}


	template<typename TDataType>
	void ParticleRod<TDataType>::setLength(Real length)
	{
		m_horizon.setValue(length);
	}

	template<typename TDataType>
	void ParticleRod<TDataType>::setMaterialStiffness(Real stiffness)
	{
		m_elasticity->setMu(0.01*stiffness);
		m_elasticity->setLambda(stiffness);
	}

	template<typename TDataType>
	bool ParticleRod<TDataType>::initialize()
	{
		ParticleSystem<TDataType>::resetStatus();

		auto& list = this->getModuleList();
		std::list<std::shared_ptr<Module>>::iterator iter = list.begin();
		for (; iter != list.end(); iter++)
		{
			(*iter)->initialize();
		}

		return true;
	}

	template<typename TDataType>
	void ParticleRod<TDataType>::setFixedParticle(int id, Coord pos)
	{
		m_fixed->setFixedPoint(id, pos);
	}



	template<typename TDataType>
	void ParticleRod<TDataType>::doCollision(Coord pos, Coord dir)
	{
		m_fixed->constrainPositionToPlane(pos, dir);
	}

	template<typename TDataType>
	void PhysIKA::ParticleRod<TDataType>::removeAllFixedPositions()
	{
		m_fixed->clear();
	}

	template<typename TDataType>
	void ParticleRod<TDataType>::getHostPosition(std::vector<Coord>& pos)
	{
		int pNum = this->m_position.getValue().size();
		if (pos.size() != pNum)
		{
			pos.resize(pNum);
		}

		cudaMemcpy(&pos[0], this->m_position.getValue().getDataPtr(), pNum*sizeof(Coord), cudaMemcpyDeviceToHost);
	}

	template<typename TDataType>
	void ParticleRod<TDataType>::setDamping(Real d)
	{
		m_damping->setDampingCofficient(d);
	}

	template<typename TDataType>
	bool ParticleRod<TDataType>::detectCollisionPoint(int& nearParticleId, Coord preV1, Coord preV2, Coord curV1, Coord curV2)
	{
		return false;
	}

	template<typename TDataType>
	void ParticleRod<TDataType>::advance(Real dt)
	{
		return;
		if (m_integrator != nullptr)
			m_integrator->begin();

		m_integrator->integrate();
 	
		for (int it = 0; it < 10; it++)
		{
			if (m_elasticity != nullptr)
				m_elasticity->constrain();

			if (m_fixed != nullptr)
				m_fixed->constrain();
		}

		if (m_damping != nullptr)
			m_damping->constrain();

		if (m_integrator != nullptr)
			m_integrator->end();
	}


	//Do nothing
// 	template<typename TDataType>
// 	void ParticleRod<TDataType>::updateTopology()
// 	{
// 		return;
// 		auto pts = this->m_pSet->getPoints();
// 
// 
// // 		HostArray<Coord> hostPts;
// // 		hostPts.resize(pts.size());
// 
// 
// 		Function1Pt::copy(pts, this->getPosition()->getValue());
// 
// // 		for (int i = 0; i < hostPts.size(); i++)
// // 		{
// // 			hostPts[i] *= 0.01;
// // 		}
// 
// //		Function1Pt::copy(pts, hostPts);
// 	}
}